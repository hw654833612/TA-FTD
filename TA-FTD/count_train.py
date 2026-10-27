import pandas as pd

# -------------------------- 1. 读取CSV文件 --------------------------
csv_file_path = "/share/home/huafuchen01/huangwei/WORKS/Task00_dataset/Natural-Scenes-Dataset/NSD-Code/step00_excel/trailID-subj01.csv"

# 读取CSV文件（header=0 表示第一行是列名/文字描述，skiprows=0 不跳过任何行）
df = pd.read_csv(
    csv_file_path,
    header=0,  # 第一行作为列标题（文字描述）
    encoding='utf-8'  # 编码格式，若文件是gbk可改为 encoding='gbk'
)

# -------------------------- 2. 先截取前 27750 个样本（核心新增） --------------------------
# 关键说明：
# 1. pandas 行索引从 0 开始，df.iloc[:27750] 表示取第 0 行到第 27749 行，共 27750 个样本
# 2. 若原始数据不足 27750 行，会自动取全部数据，避免索引越界报错
df_top_27750 = df.iloc[:27750]

# 可选：打印截取后的数据量，确认是否符合要求
print(f"截取前 27750 个样本，实际获取 {len(df_top_27750)} 个样本\n")

# -------------------------- 3. 再筛选：第11列的值为FALSE --------------------------
# 关键说明：
# 1. 第11列对应 pandas 索引 10（索引从0开始：第1列→0，第11列→10）
# 2. 兼容两种情况：列值是布尔值 False / 字符串 'FALSE'/'false'
col_11 = df_top_27750.iloc[:, 11]  # 从截取后的前 27750 个样本中定位第11列
# 筛选条件：第11列为 FALSE（布尔值）或 'FALSE'/'false'（字符串）
df_filtered = df_top_27750[
    (col_11 == False) | 
    (col_11.astype(str).str.upper() == 'FALSE')
]

# 若筛选后无数据，给出提示并退出
if df_filtered.empty:
    print("警告：筛选后无符合条件的数据（第11列不为FALSE）！")
    exit()

# -------------------------- 4. 定位第四列并统计样本序号重复次数 --------------------------
# 方法1：通过列索引定位（第四列 → 索引3，推荐，不受列名影响）
sample_serial_column = df_filtered.iloc[:, 3]  # 从筛选后的数据中提取第四列

# 统计每个序号的出现次数（即相同序号的样本数量）
serial_count = sample_serial_column.value_counts()

# -------------------------- 5. 结果输出与整理 --------------------------
# 打印统计结果（直观展示）
print("===== 样本序号统计结果（前27750个样本，第11列为FALSE，序号 → 样本数量） =====")
print(serial_count)

# 可选：将统计结果转换为DataFrame格式，更规范（便于保存和进一步分析）
serial_count_df = serial_count.reset_index()
serial_count_df.columns = ["样本序号", "对应样本数量"]  # 重命名列名，更清晰

print("\n===== 规范格式统计结果 =====")
print(serial_count_df)

# -------------------------- 6. 可选：将统计结果保存为新的CSV文件 --------------------------
serial_count_df.to_csv(
    "样本序号统计结果（前27750个样本，第11列FALSE）.csv",
    index=False,  # 不保存行索引
    encoding='utf-8-sig'  # utf-8-sig 支持中文导出无乱码
)
print("\n统计结果已保存为「样本序号统计结果（前27750个样本，第11列FALSE）.csv」")

# -------------------------- 7. 筛选出现次数≥3的序号 --------------------------
min_count = 3  # 筛选出现次数≥3的序号
high_freq_serial = serial_count_df[serial_count_df["对应样本数量"] >= min_count]
print(f"\n出现次数≥{min_count}的序号（前27750个样本，第11列为FALSE）：")
print(high_freq_serial)