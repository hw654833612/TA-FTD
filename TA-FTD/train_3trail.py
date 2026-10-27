import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import json
import h5py
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# --- 路径处理 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from deepseek_vl2.models.modeling_umbrae_deepseek import DeepseekVLV2ForCausalLM
from process import SimpleProcessor

# ==========================================
# 0. 日志配置 (保持不变)
# ==========================================
def setup_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(save_path, 'training.log'), mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# ==========================================
# 1. Dataset (解析真实的 Prompt + 新增拓扑特征加载)
# ==========================================
class UmbraeDynamicDataset(Dataset):
    def __init__(self, instruction_list, fmri_data_dict, topology_data,  # 【修改1】新增拓扑数据参数
                 processor, norm_stats=None, is_train=True, do_average_val=False):
        self.fmri_data = fmri_data_dict
        self.topology_data = topology_data  # 【修改2】保存拓扑特征数据 (subj01, 形状[N, D], D为拓扑特征维度)
        self.processor = processor
        self.is_train = is_train
        self.do_average_val = do_average_val
        self.id_map_str = {1: 'subj01', 2: 'subj02', 5: 'subj05', 7: 'subj07'} # 映射整数ID到字符串ID，通过关键字对应fmri数据
        self.id_map_idx = {1: 0, 2: 1, 5: 2, 7: 3}
        
        # --- 数据重组 ---
        print(f"Re-structuring dataset (Train={is_train})...")
        temp_group = {}
        for entry in tqdm(instruction_list, desc="Grouping"):
            try:
                # 格式: "25676#01#Can you...?:A red bird..."
                parts = entry.split("#", 2)
                if len(parts) != 3: continue
                
                trial_global_idx = int(parts[0]) 
                subj_id_raw = parts[1]
                content_part = parts[2] # "Can you...?:A red bird..."
                
                # --- [修改点] 解析 Prompt 和 Caption ---
                # 使用第一个冒号分割。split(':', 1) 保证只分割第一个冒号
                if ':' in content_part:
                    prompt_str, caption_str = content_part.split(':', 1)
                    # 补回冒号，因为 split 把冒号吃掉了。Prompt 通常包含冒号作为结束符
                    # 你的数据示例是 "...subject01?:A red..."
                    # split结果: prompt="...subject01?", caption="A red..."
                    # 我们把冒号加回 Prompt，这样模型知道这里是界限
                    prompt_str = prompt_str.strip() + ":" 
                    caption_str = caption_str.strip()
                else:
                    continue # 格式不对跳过
                
                img_idx = trial_global_idx // 3 
                
                if img_idx not in temp_group:
                    temp_group[img_idx] = {
                        'subj_raw': subj_id_raw,
                        'captions': [], # 改用 list 存储 (prompt, caption) 元组
                        'base_trial_idx': img_idx * 3
                    }
                
                # 存储 (Prompt, Caption) 对
                # 使用 list 而不是 set，因为 prompt 可能也会微变，或者直接存下来
                temp_group[img_idx]['captions'].append((prompt_str, caption_str))
                
            except Exception as e:
                continue
            
        self.grouped_data = list(temp_group.values())
        print(f"Dataset loaded. Unique Images: {len(self.grouped_data)}")

        # --- Z-Score 统计量 ---
        if norm_stats is None:
            print("Calculating Normalization Stats...")
            self.norm_params = {}
            for subj_id, data in self.fmri_data.items():
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std[std < 1e-6] = 1.0 # [保险] 防止除零
                self.norm_params[subj_id] = {'mean': mean, 'std': std}
        else:
            self.norm_params = norm_stats

        # 【修改4】拓扑特征数据校验 (确保与trial索引匹配)
        self.topology_shape = self.topology_data.shape
        print(f"Topology data loaded. Shape: {self.topology_shape} (N: {self.topology_shape[0]}, D: {self.topology_shape[1]})")

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, index):
        item = self.grouped_data[index]
        subj_str = self.id_map_str.get(int(float(item['subj_raw'])))
        base_idx = item['base_trial_idx']
        
        # --- 1. fMRI 处理 (保持不变) ---
        mean = self.norm_params[subj_str]['mean']
        std = self.norm_params[subj_str]['std']
        
        try:
            # 无论 Train 还是 Val，如果想算一致性损失，最好都取 3 个
            # 读取该图片对应的 3 次重复试验
            trials_data = []
            for k in range(3):
                # 假设数据是连续存储的: base, base+1, base+2
                raw = self.fmri_data[subj_str][base_idx + k]
                # 归一化
                norm = (raw - mean) / std
                trials_data.append(norm)
            
            # 堆叠起来: Shape 变为 [3, Voxel_Count]
            fmri_tensor = torch.FloatTensor(np.stack(trials_data, axis=0))

            # --- 2. 拓扑特征处理 (新增核心逻辑) ---
            # 【修改5】根据trial_global_idx匹配对应的拓扑特征 (注意索引对齐：拓扑特征索引与trailID对应，已重排序)
            if base_idx < self.topology_shape[0]:
                # 读取对应的拓扑特征
                topology_raw = self.topology_data[base_idx]  # 形状 [D]
                # 可选：拓扑特征归一化 (若需要，可复用fMRI的norm或单独计算)
                # topology_norm = (topology_raw - topology_mean) / topology_std
                topology_tensor = torch.FloatTensor(topology_raw)  # 转为tensor [D]
            else:
                # 索引越界时，生成零向量兜底
                topology_tensor = torch.zeros(self.topology_shape[1], dtype=torch.float32)
            
            # --- 3. 文本处理 (保持不变) ---
            # item['captions'] 里面现在存的是 [(prompt1, cap1), (prompt2, cap2)...]
            if self.is_train:
                selected_pair = random.choice(item['captions']) #5选1
            else:
                selected_pair = item['captions'][0]
            
            prompt_part = selected_pair[0] # e.g. "Can you... subject01?:"
            caption_part = selected_pair[1] # e.g. "A red bird..."
            
            # 构造完整的训练文本: "Prompt:Caption" (中间是否加空格取决于你的数据原貌，这里不加空格还原原始数据)
            full_text = prompt_part + caption_part 
            
            input_ids, label_ids = self.processor.process_batch_text_data(full_text)
            mapped_subj_idx = self.id_map_idx[int(float(item['subj_raw']))]

            # --- 4. 返回结果 (【修改6】新增拓扑特征tensor) ---
            return input_ids, mapped_subj_idx, label_ids, fmri_tensor, topology_tensor, prompt_part

        except Exception as e:
            # 异常时返回随机样本，避免训练中断
            return self.__getitem__(random.randint(0, len(self)-1))

# ==========================================
# 2. Collate_fn (修改：新增拓扑特征的padding处理)
# ==========================================
def collate_fn(batch):
    input_ids_list = [item[0] for item in batch]
    subj_ids_list = [torch.tensor(item[1], dtype=torch.long) for item in batch]
    target_ids_list = [item[2] for item in batch]
    fmri_list = [item[3] for item in batch] 
    topology_list = [item[4] for item in batch]  # 【修改7】提取拓扑特征列表
    prompt_list = [item[5] for item in batch]

    # 对原有数据进行padding (保持不变)
    padded_input_ids = pad_sequence([x.flip(0) for x in input_ids_list], batch_first=True, padding_value=2).flip(1)
    padded_target_ids = pad_sequence([x.flip(0) for x in target_ids_list], batch_first=True, padding_value=-100).flip(1)
    padded_fmri = pad_sequence(fmri_list, batch_first=True, padding_value=0.0)
    padded_topology = pad_sequence(
        topology_list,
        batch_first=True,
        padding_value=0.0  # 填充值可根据你的数据特性调整（如 0 或均值）
    )
    
    return {
        'input_ids': padded_input_ids,
        'subj_ids': torch.stack(subj_ids_list),
        'target_ids': padded_target_ids,
        'fmri_data': padded_fmri,
        'topology_data': padded_topology,  # 【修改9】返回填充后的拓扑特征
        'prompts': prompt_list
    }

# ==========================================
# 3. 生成与验证工具 (适配新增的拓扑特征，保持生成逻辑不变)
# ==========================================
def generate_eval_samples(model, tokenizer, batch, processor, max_new_tokens=60):
    model.eval()
    limit = min(4, batch['input_ids'].shape[0])
    subj_ids = batch['subj_ids'][:limit].cuda()
    fmri_data = batch['fmri_data'][:limit].cuda()
    topology_data = batch['topology_data'][:limit].cuda()  # 【修改10】加载拓扑特征 (验证时传入模型)
    target_ids = batch['target_ids'][:limit]
    
    # --- [修改点] 获取真实的 Prompt ---
    raw_prompt = batch['prompts'][0] 
    
    # Tokenize Prompt (只处理 input_ids, 不需要 labels)
    prompt_input_ids, _ = processor.process_val_text_data(raw_prompt)
    
    # 复制 prompt input_ids 给 batch 中的所有样本
    input_ids = prompt_input_ids.unsqueeze(0).repeat(limit, 1).cuda()
    
    results = []
    with torch.no_grad():
        base_model = model.base_model.model
        # 【修改11】传入拓扑特征到模型的输入嵌入准备函数 (需确保模型支持topology_data参数)
        inputs_embeds = base_model.MY_prepare_inputs_embeds_fixed_length(
            input_ids=input_ids, subj_ids=subj_ids, 
            fmri_data=fmri_data, topology_data=topology_data  # 新增拓扑特征传递
        )
        outputs = base_model.language.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=processor.pad_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            temperature=0.6,
            top_p=0.9
        )
        for i in range(limit):
            pred_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # GT
            gt_row = target_ids[i]
            gt_row = gt_row[gt_row != -100]
            gt_text = tokenizer.decode(gt_row, skip_special_tokens=True)
            
            results.append([str(subj_ids[i].item()), gt_text, pred_text])
            
    return results

# ==========================================
# 4. 主训练流程 (修改：加载拓扑特征并传入Dataset)
# ==========================================
def train(args):
    logger = setup_logger(args.save_path)
    
    tb_log_dir = os.path.join(args.save_path, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logging to: {tb_log_dir}")

    # --- 1. 加载 原始fMRI 数据 (Train & Val) ---
    logger.info("Loading fMRI H5 data...")
    train_fmri_dict = {}
    val_fmri_dict = {}
    
    # Load Train H5
    with h5py.File(args.train_h5, 'r') as hf:
        for s in ['subj01', 'subj02', 'subj05', 'subj07']:
            if s in hf:
                train_fmri_dict[s] = hf[s][:] # Load into RAM
    
    # Load Val H5
    with h5py.File(args.val_h5, 'r') as hf:
        for s in ['subj01', 'subj02', 'subj05', 'subj07']:
            if s in hf:
                val_fmri_dict[s] = hf[s][:]

    # --- 2. 加载 重排序拓扑特征 数据 (新增核心步骤) ---
    logger.info("Loading reordered topology H5 data...")
    topology_data_path = '/share/home/huafuchen01/LJP/Deepseek/datasets/fmri_feature/reordered_topological_features_by_index.h5'
    test_topology_data_path = '/share/home/huafuchen01/LJP/Deepseek/datasets/fmri_feature/test_reordered_topological_features_by_index.h5'
    subj01_topology = None
    val_subj01_topology = None
    with h5py.File(topology_data_path, 'r') as hf:
        if 'subj01' in hf and 'features' in hf['subj01']:
            subj01_topology = hf['subj01/features'][:]  # 加载重排序后的拓扑特征 [N, D]
            logger.info(f"Topology data loaded successfully. Shape: {subj01_topology.shape}")
        else:
            raise FileNotFoundError(f"Topology data not found in {topology_data_path} (missing 'subj01/features')")
            
    with h5py.File(test_topology_data_path, 'r') as hf:
        if 'subj01' in hf and 'features' in hf['subj01']:
            val_subj01_topology = hf['subj01/features'][:]  # 加载重排序后的拓扑特征 [N, D]
            logger.info(f"Topology data loaded successfully. Shape: {subj01_topology.shape}")
        else:
            raise FileNotFoundError(f"Topology data not found in {topology_data_path} (missing 'subj01/features')")

    # --- 3. 加载 JSON 指令 ---
    logger.info("Loading Text Instructions...")
    with open(args.train_json, 'r') as f:
        train_instructions = json.load(f)
    with open(args.val_json, 'r') as f:
        val_instructions = json.load(f)

    # --- 4. 初始化处理器与模型 ---
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = DeepseekVLV2ForCausalLM.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = SimpleProcessor(tokenizer, pad_id=2, ignore_id=2, num_fmri_tokens=64)

    # --- 5. 构建 Dataset (【修改12】传入拓扑特征数据) ---
    # 关键：先构建训练集，让它计算 Mean/Std
    logger.info("Building Training Dataset...")
    train_ds = UmbraeDynamicDataset(
        train_instructions, train_fmri_dict, subj01_topology,  # 传入拓扑特征
        processor, norm_stats=None, is_train=True
    )
    
    # 关键：构建验证集，传入训练集的 Mean/Std，并开启 Averaging
    logger.info("Building Validation Dataset...")
    val_ds = UmbraeDynamicDataset(
        val_instructions, val_fmri_dict, val_subj01_topology,  # 传入拓扑特征
        processor, norm_stats=train_ds.norm_params,
        is_train=False, 
        do_average_val=True
    )
    
    logger.info(f"Train Images: {len(train_ds)} | Val Images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # --- 6. LoRA & Optimizer ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["fmri_encoder", "topology_encoder"]  # 【修改13】新增拓扑编码器保存 (若模型有专门的拓扑encoder)
    )
    model = get_peft_model(model, peft_config)
    model.cuda()

    # 初始化 Encoder 权重 (新增拓扑encoder初始化，若模型有)
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
    
    # 初始化fMRI encoder
    encoder = model.base_model.model.fmri_encoder
    target_modules = encoder.modules_to_save["default"] if hasattr(encoder, "modules_to_save") else encoder
    target_modules.apply(init_weights)
    
    # 【修改14】初始化拓扑encoder (若模型有专门的topology_encoder)
    if hasattr(model.base_model.model, "topology_encoder"):
        topology_encoder = model.base_model.model.topology_encoder
        topology_target_modules = topology_encoder.modules_to_save["default"] if hasattr(topology_encoder, "modules_to_save") else topology_encoder
        topology_target_modules.apply(init_weights)
        logger.info("Topology encoder weights initialized successfully.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    total_training_steps = args.epochs * steps_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_training_steps, pct_start=0.1,
        div_factor=25, final_div_factor=10000
    )

    # --- 7. 训练循环 (【修改15】传入拓扑特征到模型前向传播) ---
    logger.info("Start Training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_epoch = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(pbar):
            outputs = model(
                input_ids=batch['input_ids'].cuda(), 
                labels=batch['target_ids'].cuda(),
                subj_ids=batch['subj_ids'].cuda(), 
                fmri_data=batch['fmri_data'].cuda(),
                topology_data=batch['topology_data'].cuda()  # 新增拓扑特征传入模型
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            current_loss = loss.item()
            train_loss_epoch += current_loss
            writer.add_scalar("Train/Step_Loss", current_loss, global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)
            global_step += 1
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        avg_train_loss = train_loss_epoch / len(train_loader)
        
        # --- 验证循环 (使用平均后的 fMRI + 拓扑特征) ---
        model.eval()
        val_loss = 0
        gen_table_md = "| Subj | GT | Pred |\n|---|---|---|\n"
        
        logger.info("Running Validation (Averaged fMRI + Topology)...")
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # 计算 Loss (传入拓扑特征)
                outputs = model(
                    input_ids=batch['input_ids'].cuda(), 
                    labels=batch['target_ids'].cuda(),
                    subj_ids=batch['subj_ids'].cuda(), 
                    fmri_data=batch['fmri_data'].cuda(),
                    topology_data=batch['topology_data'].cuda()  # 验证时也传入拓扑特征
                )
                val_loss += outputs.loss.item()
                
                # 生成演示 (仅第一个 batch)
                if i == 0:
                    try:
                        samples = generate_eval_samples(model, tokenizer, batch, processor)
                        for row in samples:
                            clean_gt = row[1].replace('|', '').replace('\n', ' ')[:50]
                            clean_pred = row[2].replace('|', '').replace('\n', ' ')[:50]
                            gen_table_md += f"| {row[0]} | {clean_gt}... | {clean_pred}... |\n"
                    except: pass
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)
        writer.add_scalar("Val/Epoch_Loss", avg_val_loss, epoch)
        writer.add_text("Val/Generations", gen_table_md, epoch)
        
        # 保存 ckpt
        save_dir = os.path.join(args.save_path, f"epoch_{epoch+1}")
        model.save_pretrained(save_dir)
        logger.info(f"Saved to {save_dir}")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 模型路径
    parser.add_argument('--model_path', type=str, default="/share/home/huafuchen01/LJP/Deepseek/deepseek-vl2-tiny/")
    
    # --- [修改] 数据路径参数 ---
    parser.add_argument('--train_json', type=str, 
                        default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/diffdata_25677/train/train_instructions.json")
    parser.add_argument('--train_h5', type=str, 
                        default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/diffdata_25677/train/fmri_data.h5")
    
    parser.add_argument('--val_json', type=str, 
                        default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/diffdata_25677/test/test_instructions.json")
    parser.add_argument('--val_h5', type=str, 
                        default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/diffdata_25677/test/fmri_data.h5")
    
    parser.add_argument('--save_path', type=str, default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/model/umbrae")
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20) # 建议增加 epoch
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    
    args = parser.parse_args()
    train(args)