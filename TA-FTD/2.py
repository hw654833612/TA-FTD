import pandas as pd

# -------------------------- 1. 配置文件路径和参数 --------------------------
# 输入文件路径（替换为你的实际文件路径）
csv_path_A = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/trailID-subj01.csv"  # 包含cocoID和trailID的A文件
csv_path_B = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/test_metadata.csv"  # 目标B文件
# 输出新文件路径（新增trailID列后的B文件）
new_csv_path_B = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/test_metadata_with_trailID.csv"
# 新增：trailID阈值（大于该值的不参与匹配）
TRAILID_THRESHOLD = 27750

# -------------------------- 2. 读取A、B两个CSV文件 --------------------------
print("正在读取CSV文件...")
# 读取A文件（保留cocoID和trailID列即可，提升效率）
df_A = pd.read_csv(
    csv_path_A,
    header=0,
    encoding='utf-8',  # 若文件乱码，替换为 encoding='gbk'
    index_col=False
)

# 读取B文件（保留所有列，后续仅新增一列）
df_B = pd.read_csv(
    csv_path_B,
    header=0,
    encoding='utf-8',
    index_col=False
)

# -------------------------- 3. 数据校验（避免关键列缺失导致错误） --------------------------
print("正在进行数据校验...")
# 检查必要列是否存在
required_cols_A = ["cocoID", "trailID"]
required_cols_B = ["cocoID"]

for col in required_cols_A:
    if col not in df_A.columns:
        raise ValueError(f"错误：A文件中缺少必要列「{col}」，请检查文件列名！")

for col in required_cols_B:
    if col not in df_B.columns:
        raise ValueError(f"错误：B文件中缺少必要列「{col}」，请检查文件列名！")

# 检查trailID列的数据类型，确保可以数值比较
if not pd.api.types.is_numeric_dtype(df_A["trailID"]):
    try:
        df_A["trailID"] = pd.to_numeric(df_A["trailID"], errors='raise')
    except:
        raise ValueError("错误：A文件中的trailID列无法转换为数值类型，无法进行阈值判断！")

# -------------------------- 4. 统一cocoID格式（关键：避免格式不匹配导致匹配失败） --------------------------
print("正在统一cocoID格式，确保匹配准确...")
# 转换为字符串并去除首尾空格（兼容数字、字符串格式的cocoID）
df_A["cocoID"] = df_A["cocoID"].astype(str).str.strip()
df_B["cocoID"] = df_B["cocoID"].astype(str).str.strip()

# -------------------------- 5. 核心步骤：为重复cocoID匹配符合阈值的不同trailID --------------------------
print("正在为重复cocoID匹配符合阈值的不同trailID...")

# 步骤1：过滤A文件，只保留trailID ≤ 27750的有效数据
df_A_filtered = df_A[df_A["trailID"] <= TRAILID_THRESHOLD].copy()
print(f"过滤后A文件有效数据行数：{len(df_A_filtered)}（原行数：{len(df_A)}）")

# 步骤2：为过滤后的A文件中每个cocoID的trailID添加序号（标记是该cocoID下的第几个有效trailID）
df_A_sorted = df_A_filtered[["cocoID", "trailID"]].drop_duplicates(subset=["cocoID", "trailID"])  # 去重相同的cocoID+trailID组合
df_A_sorted["seq_num"] = df_A_sorted.groupby("cocoID").cumcount() + 1  # 每个cocoID内从1开始计数

# 步骤3：为B文件中重复的cocoID添加序号（标记是该cocoID在B文件中第几次出现）
df_B["seq_num"] = df_B.groupby("cocoID").cumcount() + 1  # 每个cocoID内从1开始计数

# 步骤4：基于cocoID+seq_num进行匹配，实现重复cocoID对应不同的有效trailID
df_B_with_trailID = pd.merge(
    df_B,  # 左表：带序号的B文件
    df_A_sorted,  # 右表：带序号的A文件有效映射表
    on=["cocoID", "seq_num"],  # 匹配键：cocoID+序号
    how="left"  # 保留B文件所有行，未匹配到的显示NaN
)

# 步骤5：清理临时列，恢复原始列顺序（移除seq_num临时列）
df_B_with_trailID = df_B_with_trailID.drop(columns=["seq_num"])

# 步骤6：循环填充有效trailID（当B文件中cocoID重复次数多于A文件中有效trailID数量时）
def fill_remaining_trailIDs(group):
    """为trailID不足的cocoID循环填充已有有效trailID（≤27750）"""
    # 提取该cocoID下所有有效trailID（去重）
    valid_trail_ids = group[group["trailID"].notna()]["trailID"].unique()
    if len(valid_trail_ids) == 0:
        return group  # 无匹配的有效trailID，直接返回
    # 循环填充NaN值
    na_mask = group["trailID"].isna()
    fill_vals = [valid_trail_ids[i % len(valid_trail_ids)] for i in range(na_mask.sum())]
    group.loc[na_mask, "trailID"] = fill_vals
    return group

# 启用循环填充（确保重复的cocoID都能匹配到有效trailID）
df_B_with_trailID = df_B_with_trailID.groupby("cocoID", group_keys=False).apply(fill_remaining_trailIDs)

# -------------------------- 6. 保存新增列后的新B文件 --------------------------
print("正在保存新的CSV文件...")
df_B_with_trailID.to_csv(
    new_csv_path_B,
    index=False,  # 不保存pandas的默认索引列
    encoding='utf-8-sig'  # utf-8-sig：兼容中文，避免Excel打开乱码
)

# -------------------------- 7. 输出匹配结果概览（方便验证） --------------------------
matched_count = df_B_with_trailID["trailID"].notna().sum()
unmatched_count = df_B_with_trailID["trailID"].isna().sum()
# 统计匹配到的trailID是否都符合阈值
invalid_trailid_count = (df_B_with_trailID["trailID"] > TRAILID_THRESHOLD).sum()

# 统计重复cocoID的匹配情况
duplicate_cocoIDs = df_B["cocoID"].value_counts()[df_B["cocoID"].value_counts() > 1].index.tolist()

print("\n===== 匹配结果概览 =====")
print(f"原始B文件总行数：{len(df_B)}")
print(f"成功匹配到有效trailID（≤{TRAILID_THRESHOLD}）的行数：{matched_count}")
print(f"未匹配到任何有效trailID的行数：{unmatched_count}")
print(f"匹配结果中无效trailID（>{TRAILID_THRESHOLD}）的行数：{invalid_trailid_count}")
print(f"\n重复出现的cocoID数量：{len(duplicate_cocoIDs)}")
print(f"前5个重复cocoID的匹配示例：")
for cid in duplicate_cocoIDs[:5]:
    trail_ids = df_B_with_trailID[df_B_with_trailID["cocoID"] == cid]["trailID"].tolist()
    print(f"  {cid}: {trail_ids}")
print(f"\n新文件已保存至：{new_csv_path_B}")