import numpy as np
import torch
import os
import json
import tempfile
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# 移除：SPICE 相关导入语句

# 只有在使用深度学习模型时才需要，COCO eval本身是CPU的
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class EfficientCaptionEvaluator:
    def __init__(self):
        # 初始化 scorers - 移除：SPICE 评估器配置
        self.scorers = {
            'BLEU': Bleu(4),
            'METEOR': Meteor(),
            'ROUGE': Rouge(),
            'CIDEr': Cider() # 注意：CIDEr在单对单Max模式下效果不好，建议看Standard结果
        }
        self.tokenizer = PTBTokenizer()

    def tokenize_all(self, data_dict):
        """批量分词，极大提升速度"""
        print(f"正在进行批量分词，共 {len(data_dict)} 条数据...")
        # PTBTokenizer 期望输入格式: {id: [{'caption': str}]}
        input_for_tok = {k: [{'caption': v}] for k, v in data_dict.items()}
        return self.tokenizer.tokenize(input_for_tok)

    def compute_max_strategy(self, gts_raw, res_raw):
        """
        实现核心需求：单样本多标签取最大值
        Max(Score(Pred, Label1), Score(Pred, Label2)...)
        """
        print("\n=== 开始计算 (Max Strategy) ===")
        
        # 1. 扁平化所有 Label 以便批量分词
        flat_gts = {} 
        flat_res = {}
        mapping = [] # 记录 (sample_id, label_count)

        for i, (labels, pred) in enumerate(zip(gts_raw, res_raw)):
            flat_res[str(i)] = pred
            sample_label_keys = []
            for j, label in enumerate(labels):
                key = f"{i}_{j}"
                flat_gts[key] = label
                sample_label_keys.append(key)
            mapping.append((str(i), sample_label_keys))

        # 2. 批量分词 (全部 Ref 和 全部 Pred)
        gts_tok = self.tokenize_all(flat_gts)
        res_tok = self.tokenize_all(flat_res)

        # 3. 逐样本计算最大值 - 移除：SPICE 指标，保留 BLEU1/2/3/4 及原有指标
        metrics_sum = {
            "BLEU1": 0.0, "BLEU2": 0.0, "BLEU3": 0.0, "BLEU4": 0.0,
            "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0
        }
        
        sample_metrics = {k: [] for k in metrics_sum.keys()}
        valid_count = 0

        for sample_id, label_keys in mapping:
            pred_sent = res_tok[sample_id][0] # 获取分词后的预测
            current_labels = [gts_tok[lk][0] for lk in label_keys]
            
            if not current_labels or not pred_sent:
                continue

            # 临时存储该样本所有标签的得分 - 移除：SPICE 相关项
            curr_scores = {k: [] for k in metrics_sum.keys()}

            # --- 针对每个 Label 计算分数 ---
            for label_sent in current_labels:
                # 构造符合 scorer 输入的临时字典
                c_res = {0: [pred_sent]}
                c_gts = {0: [label_sent]}

                # 计算BLEU指标 - 保留 BLEU1/2/3/4
                score, _ = self.scorers['BLEU'].compute_score(c_gts, c_res)
                curr_scores['BLEU1'].append(score[0]) # Bleu@1
                curr_scores['BLEU2'].append(score[1]) # Bleu@2
                curr_scores['BLEU3'].append(score[2]) # Bleu@3
                curr_scores['BLEU4'].append(score[3]) # Bleu@4

                # 计算METEOR指标
                score, _ = self.scorers['METEOR'].compute_score(c_gts, c_res)
                curr_scores['METEOR'].append(score)

                # 计算ROUGE指标
                score, _ = self.scorers['ROUGE'].compute_score(c_gts, c_res)
                curr_scores['ROUGE_L'].append(score)
                
                # 计算CIDEr指标
                try:
                    score, _ = self.scorers['CIDEr'].compute_score(c_gts, c_res)
                    curr_scores['CIDEr'].append(score)
                except:
                    curr_scores['CIDEr'].append(0.0)

                # 移除：SPICE 指标的所有计算逻辑

            # --- 取最大值 ---
            for k in metrics_sum.keys():
                if curr_scores[k]:
                    max_val = max(curr_scores[k])
                    metrics_sum[k] += max_val
                    sample_metrics[k].append(max_val)
                else:
                    sample_metrics[k].append(0.0)
            
            valid_count += 1
            if valid_count % 100 == 0:
                print(f"已处理 {valid_count} 个样本...")

        # 4. 计算平均值
        final_avgs = {k: v / valid_count for k, v in metrics_sum.items() if valid_count > 0}
        
        return final_avgs, sample_metrics

    def compute_standard_coco(self, gts_raw, res_raw):
        """
        标准 COCO 评估方式 (1 Pred vs N Refs)
        这通常是学术界对比的标准。
        """
        print("\n=== 开始计算 (Standard COCO) ===")
        
        # 构造标准字典 {id: [ref1, ref2...]}
        gts = {}
        res = {}
        
        for i, (labels, pred) in enumerate(zip(gts_raw, res_raw)):
            gts[str(i)] = [{'caption': l} for l in labels]
            res[str(i)] = [{'caption': pred}]
            
        # 批量分词
        tokenizer = PTBTokenizer()
        gts_tok = tokenizer.tokenize(gts)
        res_tok = tokenizer.tokenize(res)
        
        scores = {}
        for name, scorer in self.scorers.items():
            score, _ = scorer.compute_score(gts_tok, res_tok)
            if isinstance(score, list):
                if name == 'BLEU':
                    # 保留 BLEU1/2/3/4 完整结果
                    scores['BLEU1'] = score[0]
                    scores['BLEU2'] = score[1]
                    scores['BLEU3'] = score[2]
                    scores['BLEU4'] = score[3]
                else:
                    scores[name] = score[0]
            else:
                scores[name] = score
                
        return scores


def main():
    # 配置路径（保持你的原始路径不变）
    path = '/share/home/huafuchen01/LJP/Deepseek/cotmodels/result/umbrae'
    results_path = f'{path}/epoch_18.txt'
    save_path = f'{path}/epoch_18.npy'

    # 读取数据
    print(f"读取文件: {results_path}")
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception as e:
        print(f"文件读取失败: {e}")
        return

    # 解析数据（适配现有txt格式，保留前5句GT）
    gts_list = []  # List[List[str]] 每个元素是单个样本的前5个GT句子列表
    res_list = []  # List[str] 每个元素是单个样本的PRED句子
    pred_str = None
    gt_str = None

    for line in lines:
        # 1. 提取PRED后面的文本
        if line.startswith('PRED:'):
            pred_str = line.split('PRED:', 1)[1].strip()
        
        # 2. 提取GT后面的文本并处理
        elif line.startswith('GT:'):
            gt_full = line.split('GT:', 1)[1].strip()
            gt_sentences = [sent.strip() for sent in gt_full.split('|') if sent.strip()]
            gt_top5 = gt_sentences[:5]  # 只保留前5句，不足5句则保留全部
            gt_str = gt_top5
        
        # 3. 验证并保存有效数据
        if pred_str is not None and gt_str is not None and len(gt_str) > 0 and pred_str != "":
            res_list.append(pred_str)
            gts_list.append(gt_str)
            # 重置临时变量，准备处理下一个样本
            pred_str = None
            gt_str = None

    # 打印解析结果统计
    print(f"有效样本数: {len(res_list)}")
    if len(res_list) > 0:
        print(f"示例 - 第1个样本PRED: {res_list[0]}")
        print(f"示例 - 第1个样本GT(前5句): {gts_list[0]}")

    # 初始化评估器
    evaluator = EfficientCaptionEvaluator()

    # 1. 计算 "Max Strategy" 指标
    max_avgs, sample_scores = evaluator.compute_max_strategy(gts_list, res_list)
    
    print("\n========= 结果 (Max Strategy) =========")
    for k, v in max_avgs.items():
        print(f"{k}: {v:.4f}")

    # 2. 计算标准 COCO 指标作为参考
    standard_scores = evaluator.compute_standard_coco(gts_list, res_list)
    print("\n========= 结果 (Standard COCO) =========")
    for k, v in standard_scores.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # 保存结果
    save_data = {
        "overall_metrics": max_avgs,
        "sample_metrics": sample_scores,
        "standard_metrics": standard_scores
    }
    np.save(save_path, save_data)
    print(f"\n结果已保存至: {save_path}")

if __name__ == "__main__":
    main()