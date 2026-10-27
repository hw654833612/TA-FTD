import pandas as pd

# -------------------------- 1. 配置文件路径 --------------------------
# 文件A：已筛选（前27750行、第12列为FALSE）的CSV
csv_path_A = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/trailID-subj01.csv"  # 替换为你的文件A路径
# 文件B：顺序不同的另一个CSV
csv_path_B = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/metadata.csv"  # 替换为你的文件B路径
# 结果保存路径（可选）
result_save_path = "/share/home/huafuchen01/LJP/Deepseek/cotmodels/ablation/umbrae/cocoID匹配结果.csv"

# -------------------------- 2. 读取并处理文件A（保留原始索引，还原筛选条件） --------------------------
print("正在读取并处理文件A...")
# 步骤2.1：读取完整文件A，保留原始行索引（index_col=False 保留默认整数索引）
df_A_full = pd.read_csv(
    csv_path_A,
    header=0,
    encoding='utf-8',  # 乱码则改为 encoding='gbk'
    index_col=False  # 不将任何列作为索引，保留默认0开始的原始行索引
)

# 步骤2.2：还原筛选条件1：取前27750行
df_A_top_27750 = df_A_full.iloc[:27750].copy()
# 保留文件A的原始行索引（新增一列存储，避免后续合并丢失）
df_A_top_27750['文件A_原始行索引'] = df_A_top_27750.index

# 步骤2.3：还原筛选条件2：第12列为FALSE（兼容布尔值/字符串）
col_12 = df_A_top_27750.iloc[:, 11]  # 第12列对应索引11
df_A_filtered = df_A_top_27750[
    (col_12 == False) | 
    (col_12.astype(str).str.upper() == 'FALSE')
].copy()

# 检查文件A筛选结果
if df_A_filtered.empty:
    print("警告：文件A筛选后无符合条件的数据！")
    exit()
print(f"文件A筛选完成：前27750行中第12列为FALSE的样本数：{len(df_A_filtered)}")

# -------------------------- 3. 读取文件B --------------------------
print("\n正在读取文件B...")
df_B = pd.read_csv(
    csv_path_B,
    header=0,
    encoding='utf-8',  # 乱码则改为 encoding='gbk'
    index_col=False
)

# 检查cocoID列是否存在
if "cocoID" not in df_A_filtered.columns or "cocoID" not in df_B.columns:
    raise ValueError("两个CSV文件中均需包含「cocoID」列，请检查列名是否正确！")

# -------------------------- 4. 基于cocoID进行关联匹配（核心步骤） --------------------------
print("\n正在基于cocoID进行匹配...")
# 采用左连接：以文件B为基准，匹配文件A的筛选结果（保留文件B所有样本，匹配不到则显示NaN）
# 若需仅保留双向匹配成功的样本，将 how='left' 改为 how='inner'
df_matched = pd.merge(
    df_B,  # 左表：文件B（顺序不同的文件）
    df_A_filtered[["cocoID", "文件A_原始行索引"]],  # 右表：文件A的cocoID+原始行索引（仅保留需要的列）
    on="cocoID",  # 关联键：cocoID
    how="left"  # 保留文件B所有样本，方便查看未匹配项
)

# -------------------------- 5. 结果输出与整理 --------------------------
# 步骤5.1：打印匹配结果概览
matched_count = df_matched["文件A_原始行索引"].notna().sum()
unmatched_count = df_matched["文件A_原始行索引"].isna().sum()
print(f"\n===== 匹配结果概览 =====")
print(f"文件B总样本数：{len(df_B)}")
print(f"与文件A筛选结果匹配成功的样本数：{matched_count}")
print(f"未匹配到的样本数：{unmatched_count}")

# 步骤5.2：打印前10条匹配结果（直观展示）
print(f"\n===== 前10条匹配结果（文件B → 文件A 行索引映射） =====")
display_columns = ["cocoID", "文件A_原始行索引"] + [col for col in df_B.columns if col not in ["cocoID"]][:2]
print(df_matched[display_columns].head(10))

# 步骤5.3：保存匹配结果到新CSV（方便后续分析）
df_matched.to_csv(
    result_save_path,
    index=False,
    encoding='utf-8-sig'  # 支持中文无乱码
)
print(f"\n匹配结果已保存为：{result_save_path}")

# -------------------------- 6. 可选：提取未匹配的样本（用于排查问题） --------------------------
df_unmatched = df_matched[df_matched["文件A_原始行索引"].isna()].copy()
if not df_unmatched.empty:
    df_unmatched.to_csv(
        "未匹配样本列表.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print("未匹配样本列表已保存为：未匹配样本列表.csv")