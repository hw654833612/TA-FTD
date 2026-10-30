import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
# 1. 导入同目录下的 Perceiver
from deepseek_vl2.models.perceiver import PerceiverResampler 

# 2. 从原始库中导入基类和配置
# [关键修改] 将 DeepseekV2ForCausalLM 移到这里
from deepseek_vl2.models.modeling_deepseek_vl_v2 import (
    DeepseekVLV2PreTrainedModel, 
    DeepseekVLV2Config, 
    MlpProjector, 
    VisionTransformer,
    DeepseekVLV2ForCausalLM as OriginalDeepseek,
    DeepseekV2ForCausalLM  # <--- 从 transformers 移到这里！它在 DeepSeek 的代码库里
)

# 3. transformers 只保留标准类
from transformers import AutoConfig, AutoModelForCausalLM #, DeepseekV2ForCausalLM (删掉这个)
# ==========================================
# 1. 定义 UMBRAE 风格的 fMRI 编码器
# ==========================================
class UmbraeEncoder(nn.Module):
    def __init__(self, subject_voxel_counts, hidden_dim=1024, num_latents=64, use_token=True):
        super().__init__()
        self.num_latents = num_latents
        self.use_token = use_token
        
        self.lin1 = nn.ModuleDict()
        self.lin2 = nn.ModuleDict()
        # [新增] 专门给 Lin1 输出做归一化
        self.ln_mid = nn.ModuleDict() 
        
        self.tokens = nn.ParameterDict()
        
        for subj, v_count in subject_voxel_counts.items():
            # Layer 1
            self.lin1[subj] = nn.Linear(v_count, hidden_dim) 
            
            # [新增] LayerNorm，防止 GELU 后均值偏移导致的累积爆炸
            self.ln_mid[subj] = nn.LayerNorm(hidden_dim)
            
            # Layer 2
            self.lin2[subj] = nn.Linear(hidden_dim, hidden_dim)

            if use_token:
                self.tokens[subj] = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.perceiver = PerceiverResampler(
            dim=hidden_dim,
            num_latents=num_latents
        )
        
        self.act = nn.GELU()

    def print_stats(self, tensor, step_name):
        if tensor is None: return
        val = tensor.float()
        is_nan = torch.isnan(val).any().item()
        is_inf = torch.isinf(val).any().item()
        msg = f"[DEBUG] {step_name:<20} | Shape: {str(list(tensor.shape)):<16} | "
        if is_nan or is_inf:
            msg += f"!!! ABNORMAL: {'NaN' if is_nan else 'Inf'} !!!"
        else:
            msg += f"Mean: {val.mean():.4f} | Max: {val.max():.4f} | Min: {val.min():.4f}"
        print(msg)

    def forward_single_subject(self, sample_data, subj_name):
        '''
        # --- [修复 1] 输入归一化 (Input Normalization) ---
        # 原始 fMRI 波动太大 (-40 ~ +40)，我们需要 Instance Norm
        # 对每个样本单独计算均值和方差进行归一化
        mean = sample_data.mean(dim=-1, keepdim=True)
        std = sample_data.std(dim=-1, keepdim=True) + 1e-6 # 防除零
        x = (sample_data - mean) / std
        
        #self.print_stats(x, "0. Input (Normed)") # 期望: Mean~0, Max~3-5
        '''
        # Layer 1
        x = sample_data
        x = self.lin1[subj_name](x)
        x = self.act(x)
        #self.print_stats(x, "1. After Lin1+Act") # 期望: Mean>0 (GELU特性)

        # --- [修复 2] 中间层归一化 ---
        # 在进 Lin2 之前，把 GELU 产生的正均值拉回来
        x = self.ln_mid[subj_name](x) 
        #self.print_stats(x, "2. After LN_Mid") # 期望: Mean~0, Max不爆炸

        # Layer 2
        x = self.lin2[subj_name](x)
        #self.print_stats(x, "3. After Lin2") # 期望: Max 绝对不会再是 17000 了
        
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        if self.use_token:
            token = self.tokens[subj_name]
            x = torch.cat([token, x], dim=1) 

        x = self.perceiver(x)
        #self.print_stats(x, "4. After Perceiver")
        
        return x

    def forward(self, fmri_data, subj_ids, id2subj_map):
        batch_size = fmri_data.shape[0]
        outputs = []
        for i in range(batch_size):
            s_id = subj_ids[i].item()
            subj_name = id2subj_map[s_id]
            sample_data = fmri_data[i].unsqueeze(0) 
            processed_feat = self.forward_single_subject(sample_data, subj_name)
            outputs.append(processed_feat)
        return torch.cat(outputs, dim=0)
        
        
        
class FeatureDimensionReducer(nn.Module):
    """
    MLP特征维度转换器：1428 -> 1280
    """
    def __init__(self, input_dim=1428, output_dim=1280, dropout_rate=0.1):
        super(FeatureDimensionReducer, self).__init__()
        
        # 设计渐进式降维结构
        hidden_dim1 = 1600  # 先扩展
        hidden_dim2 = 1400  # 轻微降维
        hidden_dim3 = 1280  # 目标维度
        
        self.layers = nn.Sequential(
            # 第一层：扩展维度，增强表达能力
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # 第二层：轻微压缩
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # 输出层：投影到目标维度
            nn.Linear(hidden_dim2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)  # 保持非负特征
        )
        
        # 残差连接层（可选）
        self.use_residual = input_dim != output_dim
        if self.use_residual:
            self.residual_proj = nn.Linear(input_dim, output_dim)
            
    def forward(self, x):
        main_output = self.layers(x)
        
        # 如果维度不同，使用投影残差连接
        if self.use_residual:
            residual = self.residual_proj(x)
            return main_output + 0.1 * residual  # 弱残差连接
        else:
            return main_output
            
            
# ==========================================
# 2. 修改后的主模型类
# ==========================================
class DeepseekVLV2ForCausalLM(DeepseekVLV2PreTrainedModel):

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder (保留原有的 Vision Transformer) ------------
        vision_config = config.vision_config
        self.vision = VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)
        self.topology_encoder = FeatureDimensionReducer(1292, 1280)
        
        # =========================================================
        # [NEW] UMBRAE fMRI Encoder
        # =========================================================
        # [关键修改] 使用你日志里打印出来的真实数据！！！
        self.voxel_counts = {
            'subj01': 15724,  
            'subj02': 14278,  
            'subj05': 13039, 
            'subj07': 12682   
        }
        # 建立 int -> str 的映射，假设 Dataset 里 subj_id 0对应subj01...
        self.id2subj_map = {0: 'subj01', 1: 'subj02', 2: 'subj05', 3: 'subj07'}
        
        self.num_latents = 64 # UMBRAE 设定，必须与 Process.py 中的 num_fmri_tokens 一致
        
        self.fmri_encoder = UmbraeEncoder(
            subject_voxel_counts=self.voxel_counts,
            hidden_dim=projector_config.n_embed, # 确保维度对齐 (e.g. 2048)
            num_latents=self.num_latents,
            use_token=True # Step 5: 开启 Learnable Tokens
        )
        
        # ----------- Image Token Format Configs (保留) ------------
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos
        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        
        if self.tile_tag == "2D":
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        elif self.tile_tag == "1D":
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(f"len(candidate_resolutions) should be larger than 0")
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed)) * embed_std
            )

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            
            # 新增参数
            subj_ids: Optional[torch.LongTensor] = None,
            fmri_data: Optional[torch.FloatTensor] = None,
            topology_data: Optional[torch.FloatTensor] = None,
            
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ):
        # [Step 1] 初始化 Loss，防止报错
        self.consistency_loss = torch.tensor(0.0, device=self.device)
        # [Step 2] 拦截 embedding 生成
        if inputs_embeds is None and fmri_data is not None:
             # 调用自定义 embedding 函数
             inputs_embeds = self.MY_prepare_inputs_embeds_fixed_length(
                 input_ids, subj_ids, fmri_data, topology_data
             )
        
        # [Step 3] LLM 前向传播
        # Standard Forward Pass ...
        outputs = self.language.forward(
            input_ids=None, # inputs_embeds 已经有了
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        # [Step 4] 加上对比损失 (0.1 是权重，可调)
        if labels is not None:
             # 确保 consistency_loss 和 outputs.loss 在同一个 device
             if self.consistency_loss.device != outputs.loss.device:
                 self.consistency_loss = self.consistency_loss.to(outputs.loss.device)
             
             outputs.loss += 0.3 * self.consistency_loss
        return outputs

    def MY_prepare_inputs_embeds_fixed_length(self, input_ids, subj_ids, fmri_data, topology_data, task_ids=None):
            # 1. 获取文本 Embedding
            raw_embeds = self.language.get_input_embeddings()(input_ids)
            inputs_embeds = raw_embeds.clone() # [梯度安全]
    
            # 2. 处理 fMRI 数据
            if fmri_data.dim() == 3: # [B, 3, V] -> 训练/验证时的 3-Trial 模式
                b, n_trials, v = fmri_data.shape
                
                # --- 编码 ---
                flat_fmri = fmri_data.view(b * n_trials, v)
                flat_subj = subj_ids.repeat_interleave(n_trials)
                flat_feat = self.fmri_encoder(flat_fmri, flat_subj, self.id2subj_map)
                stacked_embeds = flat_feat.view(b, n_trials, self.num_latents, -1) # [B, 3, 64, H]
    
                # --- InfoNCE Loss 计算 ---
                # Flatten & Normalize
                flat_view = stacked_embeds.view(b, n_trials, -1) # [B, 3, 64*H]
                flat_view = F.normalize(flat_view, p=2, dim=-1)  # [Key Step]
                
                z1, z2, z3 = flat_view[:, 0], flat_view[:, 1], flat_view[:, 2]
    
                def contrastive_loss(v1, v2, temp=0.07):
                    logits = torch.matmul(v1, v2.T) / temp
                    labels = torch.arange(v1.size(0), device=v1.device)
                    return F.cross_entropy(logits, labels)
    
                loss_12 = contrastive_loss(z1, z2)
                loss_23 = contrastive_loss(z2, z3)
                loss_13 = contrastive_loss(z1, z3)
                self.consistency_loss = (loss_12 + loss_23 + loss_13) / 3.0
    
                # --- 选一个给 LLM ---
                if self.training:
                    ridx = torch.randint(0, 3, (1,)).item()
                    fmri_embeds = stacked_embeds[:, ridx]
                else:
                    fmri_embeds = stacked_embeds.mean(dim=1)
    
            else: # [B, V] -> 推理时的单 Trial 模式
                fmri_embeds = self.fmri_encoder(fmri_data, subj_ids, self.id2subj_map)
            
            topological_embeds = self.topology_encoder(topology_data)
            # 3. 替换占位符
            special_token_id = 128001 # [Confirm This ID!]
            
            for i in range(input_ids.shape[0]):
                # 这里的 reshape(-1) 很关键，防止维度不对
                positions = torch.nonzero(input_ids[i] == special_token_id, as_tuple=False).reshape(-1)
                
                if len(positions) > 0: # 安全检查
                    inputs_embeds[i, positions[0]-1] = topological_embeds[i]
                    length = min(len(positions), self.num_latents)
                    # 这里的 fmri_embeds[i] 对应第 i 个样本
                    inputs_embeds[i, positions[:length]] = fmri_embeds[i, :length]
    
            return inputs_embeds