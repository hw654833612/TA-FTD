import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import json
import h5py
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

# --- 1. 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from deepseek_vl2.models.modeling_umbrae_deepseek import DeepseekVLV2ForCausalLM
from process import SimpleProcessor

def setup_args():
    parser = argparse.ArgumentParser()
    # 基础模型
    parser.add_argument('--model_path', default="/share/home/huafuchen01/LJP/Deepseek/deepseek-vl2-tiny", type=str)
    # 训练好的 LoRA 权重路径
    parser.add_argument('--lora_path', default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/model/umbrae/epoch_18", type=str)
    
    # 输出文件
    parser.add_argument('--output_path', default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/result/umbrae/epoch_18.txt", type=str)
    
    # 测试集路径 (982张图)
    parser.add_argument('--test_json', default='/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/diffdata_25677/test/test_instructions.json', type=str)
    parser.add_argument('--test_h5', default='/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/diffdata_25677/test/fmri_data.h5', type=str)
    
    return parser.parse_args()

# ==========================================
# 2. 数据集类 (核心修改：取平均)
# ==========================================
class UmbraeTestDataset(Dataset):
    def __init__(self, json_path, h5_path, topology_data, processor, stats_path="./subj01_train_stats.npz"):
        self.processor = processor
        
        # 1. 加载测试集 fMRI 数据
        print(f"Loading Test fMRI data from {h5_path} ...")
        self.h5_file = h5py.File(h5_path, 'r')
        self.raw_data = self.h5_file['subj01'][:] 
        self.topology_data = topology_data
        self.topology_shape = self.topology_data.shape
        
        # 2. 【核心修改】加载训练集的统计量
        print(f"Loading Training Stats from {stats_path} ...")
        # 确保你把刚才生成的 .npz 文件放到了对应路径
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.mean = stats['mean']
            self.std = stats['std']
            print("Using TRAINING SET statistics for normalization.")
        else:
            print("WARNING: Stats file not found! Fallback to TEST SET statistics (Not Recommended).")
            self.mean = np.mean(self.raw_data, axis=0)
            self.std = np.std(self.raw_data, axis=0)
            self.std[self.std < 1e-6] = 1.0
        print(f"Norm Stats: Mean={self.mean.mean():.4f}, Std={self.std.mean():.4f}")

        # --- B. 解析 JSON 并按图片分组 ---
        print(f"Loading Instructions from {json_path} ...")
        with open(json_path, 'r') as f:
            raw_instructions = json.load(f)
            
        # Grouping Logic: Image_ID -> {trials: [], captions: [], prompt: str}
        self.grouped_samples = []
        temp_dict = {}
        
        for entry in tqdm(raw_instructions, desc="Grouping Data"):
            try:
                # 格式: "trial_idx#subj#Prompt:Caption"
                parts = entry.split("#", 2)
                if len(parts) != 3: continue
                
                trial_idx = int(parts[0])
                content = parts[2]
                
                # 解析 Prompt 和 Caption
                if ":" in content:
                    prompt_str, caption_str = content.split(":", 1)
                    # 补回冒号并加空格，这很重要
                    prompt_str = prompt_str.strip() + ": " 
                    caption_str = caption_str.strip()
                else:
                    continue

                # 核心逻辑：计算图片索引
                # 假设数据是按顺序排列的，每张图 3 个 trial
                img_idx = trial_idx // 3
                
                if img_idx not in temp_dict:
                    temp_dict[img_idx] = {
                        'img_idx': img_idx,
                        'base_trial_idx': img_idx * 3, # 这张图在H5里的起始位置
                        'prompt': prompt_str, # 同一张图 Prompt 是一样的，存一份即可
                        'captions': []
                    }
                temp_dict[img_idx]['captions'].append(caption_str)
                
            except Exception as e:
                continue
        
        # 字典转列表，并按 img_idx 排序确保顺序
        sorted_keys = sorted(temp_dict.keys())
        self.grouped_samples = [temp_dict[k] for k in sorted_keys]
        print(f"Found {len(self.grouped_samples)} unique images (expected ~982).")

    def __len__(self):
        return len(self.grouped_samples)

    def __getitem__(self, index):
        sample = self.grouped_samples[index]
        
        # 1. 获取 3 个 Trial 的 fMRI 数据
        base_idx = sample['base_trial_idx']
        # 读取 [base, base+1, base+2]
        trials = self.raw_data[base_idx : base_idx+3] 
        
        # 2. 【关键步骤】取平均 (Averaging)
        avg_fmri = np.mean(trials, axis=0)
        
        # 3. 归一化
        norm_fmri = (avg_fmri - self.mean) / self.std
        fmri_tensor = torch.FloatTensor(norm_fmri)
        
        # 4. 处理 Prompt
        prompt_text = sample['prompt']
        # process_val_text_data 返回 (input_ids, raw_text)
        input_ids, _ = self.processor.process_val_text_data(prompt_text)
        
        if base_idx < self.topology_shape[0]:
            # 读取对应的拓扑特征
            topology_raw = self.topology_data[base_idx]  # 形状 [D]
            # 可选：拓扑特征归一化 (若需要，可复用fMRI的norm或单独计算)
            # topology_norm = (topology_raw - topology_mean) / topology_std
            topology_tensor = torch.FloatTensor(topology_raw)  # 转为tensor [D]
        else:
            # 索引越界时，生成零向量兜底
            topology_tensor = torch.zeros(self.topology_shape[1], dtype=torch.float32)
        
        return {
            'img_id': sample['img_idx'],
            'input_ids': input_ids,
            'fmri_data': fmri_tensor,
            'topology_tensor': topology_tensor,
            'prompt_text': prompt_text,
            'gt_captions': sample['captions'] # 这是一个 list，包含 5 个参考答案
        }

def collate_fn(batch):
    # Batch Size = 1 推理，直接返回第一个元素
    return batch[0]

# ==========================================
# 3. 推理主函数
# ==========================================
def run_inference(args):
    # --- 模型加载 ---
    print(f"Loading Base Model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    
    print(f"Loading LoRA: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    # --- 处理器 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    processor = SimpleProcessor(tokenizer, pad_id=pad_id, ignore_id=-100, num_fmri_tokens=64)

    # --- 数据加载 ---
    val_subj01_topology = None
    test_topology_data_path = '/share/home/huafuchen01/LJP/Deepseek/datasets/fmri_feature/test_reordered_topological_features_by_index.h5'
    with h5py.File(test_topology_data_path, 'r') as hf:
        if 'subj01' in hf and 'features' in hf['subj01']:
            val_subj01_topology = hf['subj01/features'][:]  # 加载重排序后的拓扑特征 [N, D]
        else:
            raise FileNotFoundError(f"Topology data not found in {topology_data_path} (missing 'subj01/features')")
    dataset = UmbraeTestDataset(args.test_json, args.test_h5, val_subj01_topology, processor)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    # --- 推理循环 ---
    print(f"Starting Inference -> {args.output_path}")
    device = model.device
    dtype = model.dtype
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for step, sample in enumerate(tqdm(loader)):
            
            # 准备输入
            input_ids = sample['input_ids'].unsqueeze(0).to(device) # [1, Seq_Len]
            fmri_data = sample['fmri_data'].unsqueeze(0).to(device).to(dtype) # [1, Voxels]
            # 这里 subj_id 固定为 0 (对应 subj01)，因为我们已经在 Dataset 里手动映射了
            # 或者你可以像 train.py 那样传 id_map，这里简化处理直接给 0
            subj_id = torch.tensor([0]).to(device) 
            topology_data=sample['topology_tensor'].unsqueeze(0).to(device).to(dtype)
            
            with torch.no_grad():
                base_model = model.base_model.model
                
                # 1. 替换 Embedding
                inputs_embeds = base_model.MY_prepare_inputs_embeds_fixed_length(
                    input_ids=input_ids,
                    subj_ids=subj_id,
                    fmri_data=fmri_data, 
                    topology_data=topology_data
                )
                
                # 2. 生成 (Generation)
                # 使用和训练验证时相似的参数
                '''
                outputs = base_model.language.generate(
                    inputs_embeds=inputs_embeds,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=80,      # 生成长度
                    do_sample=True,         # 采样
                    temperature=0.6,        # 温度 (0.6 比较稳)
                    top_p=0.9,
                    use_cache=True
                )
                '''
                outputs = base_model.language.generate(
                    inputs_embeds=inputs_embeds,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=80,
                    
                    # --- 核心修改 ---
                    do_sample=False,        # 【关闭采样】：不再随机，而是追求确定性
                    num_beams=5,            # 【开启束搜索】：同时探索5条路径，选最好的
                    
                    # --- 质量控制 ---
                    repetition_penalty=1.2, # 【重复惩罚】：防止 fMRI 噪声导致模型复读 "a bird a bird..."
                    length_penalty=1.0,     # 【长度惩罚】：>0 鼓励长句，<0 鼓励短句，1.0 是中性
                    
                    use_cache=True
                )
                # 3. 解码
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # 4. 格式化输出 (方便阅读和后续 COCO Metric 计算)
                # 将 5 个 GT 拼接成字符串展示
                gt_str = " | ".join(sample['gt_captions'])
                
                # 写入文件
                f.write(f"IMG_ID: {sample['img_id']}\n")
                f.write(f"PROMPT: {sample['prompt_text']}\n")
                f.write(f"PRED:   {pred_text}\n")
                f.write(f"GT:     {gt_str}\n")
                f.write("-" * 40 + "\n")
                f.write(json.dumps({
                    "image_id": sample['img_id'],
                    "caption": pred_text
                }) + "\n") # 额外写一行 JSON 格式方便机器读取
                
                # 打印前几个看看效果
             
                print(f"\n[IMG {sample['img_id']}]")
                print(f"Prompt: {sample['prompt_text']}")
                print(f"Pred:   {pred_text}")
                print(f"GT:     {sample['gt_captions'][0]} ...")

    print("Done.")

if __name__ == "__main__":
    args = setup_args()
    run_inference(args)