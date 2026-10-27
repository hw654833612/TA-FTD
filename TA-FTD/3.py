import pandas as pd
import numpy as np
import h5py

# -------------------------- 1. 配置文件路径 --------------------------
# CSV文件路径（包含index列（新顺序）和trailID列（1~27750，对应原始拓扑索引））
csv_file_path = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/test_metadata_with_trailID.csv"
# 原始拓扑特征H5文件路径
raw_h5_path = "/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step02_readData/subj01/beta-hdf5/floc-words/VWFA-2-27750x505.hdf5"
# 新排序后拓扑特征H5文件保存路径（避免覆盖原始文件）
new_h5_path = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/cot_prompt/VWFA2/sub01/test_VWFA2_505.h5"

# -------------------------- 2. 读取数据 --------------------------
print("正在读取原始拓扑特征和CSV文件...")

# 2.1 读取原始拓扑特征（H5文件，27750个样本，索引0~27749）
raw_features = None
with h5py.File(raw_h5_path, 'r') as hf:
    # 读取subj01的拓扑特征，确认样本数为27750
    raw_features = hf['beta'][:]
    print(f"原始拓扑特征形状：{raw_features.shape}")
    
    # 验证原始拓扑特征样本数是否为27750
    if raw_features.shape[0] != 27750:
        raise ValueError(f"原始拓扑特征样本数不为27750，当前为{raw_features.shape[0]}，请检查H5文件！")

# 2.2 读取CSV文件（包含index列和trailID列）
df = pd.read_csv(
    csv_file_path,
    header=0,
    encoding='utf-8-sig',  # 兼容Excel打开无乱码，若乱码可改为encoding='gbk'
    index_col=False
)

# -------------------------- 3. 数据校验与格式统一 --------------------------
print("正在进行数据校验与格式统一...")
required_columns = ["index", "trailID"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"CSV文件中缺少必要列「{col}」，请检查文件列名是否正确！")

# 3.1 统一数据格式，处理trailID（1~27750）
df["trailID"] = df["trailID"].astype(int)  # 转为整数，确保匹配准确

# 【修改1】：先验证index列是否为纯数字，再转为数值类型（支持数字升序排序），同时保留原始字符串格式
# 先去空格，再尝试转为整数（处理数字型index）
df["index_str"] = df["index"].astype(str).str.strip()  # 保留字符串格式用于后续保存
try:
    df["index_num"] = df["index_str"].astype(int)  # 转为整数类型，用于数字升序排序
except ValueError as e:
    print(f"警告：index列包含非数字字符，将使用自然排序，错误信息：{e}")
    df["index_num"] = df["index_str"]  # 若含非数字，回退到字符串格式

# 3.2 筛选CSV中的有效trailID（1~27750，对应原始拓扑索引0~27749）
df_valid = df[
    (df["trailID"] >= 1) & 
    (df["trailID"] <= 27750)
].copy().reset_index(drop=True)

if len(df_valid) == 0:
    raise ValueError("CSV文件中无有效trailID（1~27750），请检查数据！")
print(f"CSV文件有效数据行数（trailID 1~27750）：{len(df_valid)}")

# -------------------------- 4. 构建核心映射关系（关键：处理trailID与原始索引的偏移） --------------------------
print("正在构建映射关系，准备重排序...")

# 4.1 构建「trailID → 原始拓扑索引」的映射（核心修正：trailID-1 = 原始索引，因为trailID从1开始，索引从0开始）
# trailID 1 → 原始索引0，trailID 27750 → 原始索引27749
trailID_to_raw_idx = {}
for _, row in df_valid.iterrows():
    trail_id = row["trailID"]
    raw_idx = trail_id - 1  # 处理偏移，这是修正的核心
    trailID_to_raw_idx[trail_id] = raw_idx

# 【修改2】：按index列的「数字升序」排序（而非字典序），使用新增的index_num列
# 若index列含非数字，将使用字符串自然排序（兼容两种场景）
df_sorted_by_index = df_valid.sort_values(by="index_num", ascending=True).reset_index(drop=True)
new_order_trailID = df_sorted_by_index["trailID"].values

# 4.3 转换为原始拓扑索引序列（按index列顺序）
new_order_raw_idx = np.array([trailID_to_raw_idx[tid] for tid in new_order_trailID], dtype=int)
print(f"新顺序有效原始索引数：{len(new_order_raw_idx)}（对应重排序后特征尺寸）")

# -------------------------- 5. 核心步骤：按index列顺序重排序拓扑特征 --------------------------
print("正在按CSV的index列重排序拓扑特征...")

# 利用NumPy高级索引，快速提取原始特征并完成重排序（支持非27750输出尺寸）
reordered_features = raw_features[new_order_raw_idx]

# 验证重排序后特征形状
print(f"原始拓扑特征形状：{raw_features.shape}")
print(f"重排序后拓扑特征形状：{reordered_features.shape}（行数=CSV有效数据按index排序后的行数）")

# -------------------------- 6. 保存重排序后的新拓扑特征（H5文件） --------------------------
print("正在保存重排序后的新H5拓扑文件...")
with h5py.File(new_h5_path, 'w') as new_hf:
    # 创建subj01组，保持与原始H5文件结构一致
    subj_group = new_hf.create_group("subj01")
    
    # 保存重排序后的拓扑特征（支持非27750尺寸，保持原始数据类型）
    subj_group.create_dataset(
        "features",
        data=reordered_features,
        dtype=np.float32,  # 与原始拓扑特征数据类型一致，避免精度损失
        compression="gzip"  # 压缩存储，减少文件大小
    )
    
    # 【修改3】：保存原始字符串格式的index列，方便后续追溯
    new_hf.create_dataset("new_order_index", data=np.array(df_sorted_by_index["index_str"].values, dtype=h5py.special_dtype(vlen=str)))
    new_hf.create_dataset("new_order_trailID", data=new_order_trailID)
    new_hf.create_dataset("new_order_raw_idx", data=new_order_raw_idx)

print(f"新拓扑文件已保存至：{new_h5_path}")
print(f"新文件中subj01/features形状：{reordered_features.shape}")

# -------------------------- 7. 结果验证（可选） --------------------------
print("\n===== 重排序结果验证 =====")
# 【修改4】：输出验证使用字符串格式的index，保持显示一致
print(f"CSV index列前5个值（新顺序，数字升序）：{df_sorted_by_index['index_str'].head(5).values}")
print(f"对应trailID前5个值：{new_order_trailID[:5]}")
print(f"对应原始拓扑索引前5个值（偏移后）：{new_order_raw_idx[:5]}")
print(f"重排序后特征前5行前2个维度：\n{reordered_features[:5, :2]}")