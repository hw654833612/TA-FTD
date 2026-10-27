import torch
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import AutoModelForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import json
from process import SimpleProcessor
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import h5py
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
from accelerate import infer_auto_device_map, dispatch_model
from peft import LoraConfig, get_peft_model, TaskType

    
def freeze_layers(model, args, freeze_up_to_layer):
    """
    冻结指定层并只训练最后一层。
    
    参数：
    - model: 要训练的模型。
    - model_splits: 每个GPU上层的数量。
    - model_name: 模型的名称（用来从model_splits中获取层数信息）。
    - freeze_up_to_layer: 要冻结的层数（默认冻结前11层）。
    """
    layer_cnt = 0
    
    # 遍历每一层
    for i in range(12):
        # 冻结前 freeze_up_to_layer 层
        if layer_cnt < freeze_up_to_layer:
            for param in model.language.model.layers[layer_cnt].parameters():
                param.requires_grad = False
        else:
            # 从第 freeze_up_to_layer 层开始解冻
            for param in model.language.model.layers[layer_cnt].parameters():
                param.requires_grad = True
            
        layer_cnt += 1

    # 创建只包含可训练参数的优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr  # 根据需要调整学习率
    )
    
    return optimizer


def load_model(model_path):
    #device_map = split_model('deepseek-ai/deepseek-vl2-tiny')
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path)
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    return vl_gpt, tokenizer

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/share/home/huafuchen01/LJP/Deepseek/deepseek-vl2-tiny/", type=str, help='')
    parser.add_argument('--save_model_path', default="/share/home/huafuchen01/LJP/Deepseek/cotmodels/model/only_caption_wlora_fmri1280/", type=str, help='')
    parser.add_argument('--train_raw_path', default='/share/home/huafuchen01/LJP/decodeGPT/txt/trainData_S1.txt', type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=5000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1.5e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=1, type=int, required=False, help='print log steps')
    parser.add_argument('--save_epoch', default=1, type=int, required=False, help='save epochs')
    parser.add_argument('--dtype', default=torch.float16, type=None, required=False, help='dtype')
    parser.add_argument('--log_path', default="/share/home/huafuchen01/LJP/Deepseek/mymodels/logs", type=str, help='')
    # 早停参数
    parser.add_argument('--patience', default=2, type=int, required=False, help='early stopping patience')
    parser.add_argument('--min_delta', default=0.01, type=float, required=False, help='minimum change to qualify as improvement')

    return parser.parse_args()



class MyDataset(Dataset):
    def __init__(self, prompt_list, subj_list, label_list, subj01_features_list, main_fmri_list, other_fmri_list):
        self.prompt_list = prompt_list
        self.subj_list = subj_list
        self.label_list = label_list
        self.subj01_features_list = subj01_features_list
        self.main_fmri_list = main_fmri_list
        self.other_fmri_list = other_fmri_list

    def __getitem__(self, index):
        prompt_ids = self.prompt_list[index]
        subj_ids = self.subj_list[index]
        label_ids = self.label_list[index]
        subj01_features_list = self.subj01_features_list[index]
        main_fmri_list = self.main_fmri_list[index]
        other_fmri_list = self.other_fmri_list[index]

        return prompt_ids, subj_ids, label_ids, subj01_features_list, main_fmri_list, other_fmri_list

    def __len__(self):
        return len(self.prompt_list)

def collate_fn(batch):
    input_ids_list = [torch.LongTensor(sample[0]) for sample in batch]  # 转换为张量
    subj_ids_list = [torch.tensor(sample[1], dtype=torch.float) for sample in batch]
    subj_ids_tensor = torch.stack(subj_ids_list)
    target_ids_list = [torch.LongTensor(sample[2]) for sample in batch]  # 转换为张量
    subj01_features_list = [torch.FloatTensor(sample[3]) for sample in batch]  # 假定 fmri_stm 在第二位置
    subj01_features_tensor = torch.stack(subj01_features_list)
    main_fmri_list = [torch.FloatTensor(sample[4]) for sample in batch]  # 假定 fmri_stm 在第二位置
    main_fmri_tensor = torch.stack(main_fmri_list)
    other_fmri_list = [torch.FloatTensor(sample[5]) for sample in batch]  # 假定 fmri_stm 在第二位置
    other_fmri_tensor = torch.stack(other_fmri_list)

    # 使用 pad_sequence 对 input_ids 和 target_ids 进行填充，并实现左填充
    padded_input_ids = pad_sequence([ids.flip(0) for ids in input_ids_list], batch_first=True,
                                    padding_value=2)  # 先反转再填充
    padded_input_ids = padded_input_ids.flip(1)  # 再反转回来，实现左填充

    padded_target_ids = pad_sequence([ids.flip(0) for ids in target_ids_list], batch_first=True,
                                    padding_value=2)
    padded_target_ids = padded_target_ids.flip(1)

    return {
        'input_ids': padded_input_ids,
        'subj_ids': subj_ids_tensor,
        'target_ids': padded_target_ids,
        'subj01_features_ids': subj01_features_tensor,
        'main_fmri_ids': main_fmri_tensor,
        'other_fmri_ids': other_fmri_tensor
    }

def train_data_loader(args, vl_gpt, tokenizer, train_data, shuffle):
    # 读取文件中的对话内容
    prompt_list = []
    label_list = []
    subj_list=[]
    subj01_features_list = []
    main_fmri_list = []
    other_fmri_list = []
    processor = SimpleProcessor(tokenizer)
    subjs = ['subj01', 'subj02', 'subj05', 'subj07']
    
    with h5py.File('/share/home/huafuchen01/LJP/Deepseek/datasets/fmri_feature/all_subjects_topological_features_robust.h5', 'r') as hf:
        subj01_features = hf['subj01/features'][:]
        
    with h5py.File('/share/home/huafuchen01/LJP/decodeGPT/fmri_emb/fmri_emb_1280.h5', 'r') as hf:
        # 加载 main_data
        main_data = torch.FloatTensor(hf[f'main_data/subj01'][:])
        # 加载 other_data
        other_data = torch.FloatTensor(hf[f'other_data/subj01'][:])
        
    for _, dialogue in enumerate(train_data):
        text_split = dialogue.split("#", 2)
        stm, subjid, text = text_split
        subj = 'subj' + subjid
        prompt, label_ids = processor.process_batch_text_data(text)
        prompt_list.append(prompt)#prompt + task
        subj_list.append(int(subjid))
        label_list.append(label_ids)
        subj01_features_list.append(subj01_features[int(stm), :])
        main_fmri_list.append(main_data[int(stm), :])
        other_fmri_list.append(other_data[int(stm), :])
        

    dataset = MyDataset(prompt_list, subj_list, label_list, subj01_features_list, main_fmri_list, other_fmri_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader

def val_data_loader(args, vl_gpt, tokenizer, val_data, shuffle):
    # 读取文件中的对话内容
    prompt_list = []
    label_list = []
    subj_list=[]
    subj01_features_list = []
    main_fmri_list = []
    other_fmri_list = []
    processor = SimpleProcessor(tokenizer)
    subjs = ['subj01', 'subj02', 'subj05', 'subj07']
    
    with h5py.File('/share/home/huafuchen01/LJP/Deepseek/datasets/fmri_feature/all_subjects_topological_features_robust.h5', 'r') as hf:
        subj01_features = hf['subj01/features'][:]
        
    with h5py.File('/share/home/huafuchen01/LJP/decodeGPT/fmri_emb/fmri_emb_1280.h5', 'r') as hf:
        # 加载 main_data
        main_data = torch.FloatTensor(hf[f'main_data/subj01'][:])
        # 加载 other_data
        other_data = torch.FloatTensor(hf[f'other_data/subj01'][:])
        
    for _, dialogue in enumerate(val_data):
        text_split = dialogue.split("#", 2)
        stm, subjid, text = text_split
        subj = 'subj' + subjid
        prompt, label_ids = processor.process_batch_text_data(text)
        prompt_list.append(prompt)#prompt + task
        subj_list.append(int(subjid))
        label_list.append(label_ids)
        subj01_features_list.append(subj01_features[int(stm), :])
        main_fmri_list.append(main_data[int(stm), :])
        other_fmri_list.append(other_data[int(stm), :])
        

    dataset = MyDataset(prompt_list, subj_list, label_list, subj01_features_list, main_fmri_list, other_fmri_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader
    
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=3, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): 在多少个epoch内验证损失没有改善时停止训练
            min_delta (float): 最小改善量，小于这个值不算改善
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'Validation loss improved to {val_loss:.4f}')
        
        return self.early_stop
        
def validate(model, val_dataloader, device):
    """验证函数"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    base_model = model.base_model.model if hasattr(model, 'base_model') else model

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            # 从批次中获取数据
            input_ids = batch['input_ids'].to(device)
            subj_ids = batch['subj_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            main_fmri_ids = batch['main_fmri_ids'].to(device)
            other_fmri_ids = batch['other_fmri_ids'].to(device)
            fmri_topological_features = batch['subj01_features_ids'].to(device)
            
            # 准备输入embeddings
            inputs_embeds = base_model.MY_prepare_inputs_embeds(
                input_ids=input_ids, 
                subj_ids=subj_ids, 
                main_fmri=main_fmri_ids, 
                other_fmri=other_fmri_ids, 
                fmri_topological_features=fmri_topological_features
            )

            # 计算损失
            loss = model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=target_ids,
                use_cache=True
            ).loss
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss
    
def save_model_with_custom_modules(model, tokenizer, save_path):
    """保存LoRA权重和自定义模块权重"""
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 保存LoRA adapters
    model.save_pretrained(save_path, safe_serialization=False)
    
    # 2. 单独保存自定义模块的权重
    custom_modules = {}
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    
    for name, param in base_model.named_parameters():
        if any(keyword in name for keyword in 
               ["fmri_projector", "visual_encoder", "attention", "weight_mask_m01", "weight_mask_o01"]):
            custom_modules[name] = param.data.cpu()
    
    # 保存自定义模块权重
    torch.save(custom_modules, os.path.join(save_path, 'custom_modules.pt'))
    
    # 3. 保存tokenizer
    tokenizer.save_pretrained(save_path)
    
    print(f"Model saved: LoRA adapters + custom modules to {save_path}")

def train(args, model, tokenizer, train_dataloader, val_dataloader):
    num_training_steps = args.epochs * len(train_dataloader)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
    # === 模型配置 ===
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    
    # === 设置可训练参数 ===
    for name, param in model.named_parameters():
        if "lora_" in name or any(keyword in name for keyword in 
            ["fmri_projector", "visual_encoder", "attention", "weight_mask_m01", "weight_mask_o01"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    model.to(device)
    
    base_model = model.base_model.model
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True)
    
    model.train()
    cur_epochs = 0
    batch_steps = 0
    writer = SummaryWriter(log_dir=args.log_path)
    begin = time.perf_counter()
    
    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(args.epochs):
        cur_epochs += 1
        epoch_train_loss = 0
        
        print(f"\n{'='*50}")
        print(f"Epoch {cur_epochs}/{args.epochs}")
        print(f"{'='*50}")
        
        # 训练阶段
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {cur_epochs}"):
            batch_steps += 1
            
            # 从批次中获取数据
            input_ids = batch['input_ids'].to(device)
            subj_ids = batch['subj_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            main_fmri_ids = batch['main_fmri_ids'].to(device)
            other_fmri_ids = batch['other_fmri_ids'].to(device)
            fmri_topological_features = batch['subj01_features_ids'].to(device)
            
            # 准备输入embeddings
            inputs_embeds = base_model.MY_prepare_inputs_embeds(
                input_ids=input_ids, 
                subj_ids=subj_ids, 
                main_fmri=main_fmri_ids, 
                other_fmri=other_fmri_ids, 
                fmri_topological_features=fmri_topological_features
            )

            # 前向传播
            loss = model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=target_ids,
                use_cache=True
            ).loss
            
            epoch_train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 记录训练损失
            writer.add_scalar('Loss/training', loss.item(), epoch * len(train_dataloader) + batch_steps)
            
            if batch_steps % args.log_step == 0:
                end = time.perf_counter()
                cost_time = end - begin
                begin = time.perf_counter()
                print(f"\nBatch {batch_steps}/{num_training_steps}, Loss: {loss.item():.4f}, Time: {cost_time:.2f}s")
                    
            # 清除未使用的显存
            if batch_steps % 100 == 0: 
                torch.cuda.empty_cache()

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        print(f"\nEpoch {cur_epochs} - Average Training Loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        print(f"\nValidating...")
        val_loss = validate(model, val_dataloader, device)
        print(f"Epoch {cur_epochs} - Validation Loss: {val_loss:.4f}")
        
        # 记录验证损失
        writer.add_scalar('Loss/validation', val_loss, cur_epochs)
        writer.add_scalar('Loss/train_vs_val', avg_train_loss, cur_epochs)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.save_model_path, 'best_model')
            # 使用方法1（推荐）
            save_model_with_custom_modules(model, tokenizer, best_model_path)
    
        # 定期保存checkpoint
        if cur_epochs % args.save_epoch == 0:
            checkpoint_path = os.path.join(args.save_model_path, f'epoch{cur_epochs}')
            save_model_with_custom_modules(model, tokenizer, checkpoint_path)
        
        # 早停检查
        if early_stopping(val_loss, cur_epochs):
            print(f"\n{'='*50}")
            print(f"Early stopping triggered!")
            print(f"Best epoch: {early_stopping.best_epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            print(f"{'='*50}")
            break
            
    writer.close()
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_model_path}")
    print(f"{'='*50}")
    
args = setup_args()
model, tokenizers = load_model(args.model_path)

with open('/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/subject1_train_onlycaptiondata.json', 'r', encoding='utf-8') as json_file:
    train_data = json.load(json_file)
    
with open('/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/subject1_val_partialonlycaptiondata.json', 'r', encoding='utf-8') as json_file:
    val_data = json.load(json_file)

train_dataloader = train_data_loader(args, model, tokenizers, train_data, shuffle=True)
val_dataloader = val_data_loader(args, model, tokenizers, val_data, shuffle=False)
train(args, model, tokenizers, train_dataloader, val_dataloader)




