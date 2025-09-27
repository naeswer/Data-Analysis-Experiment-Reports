import pandas as pd
import numpy as np

# 读取数据（注意编码为latin1）
df = pd.read_csv('Pokemon.csv', encoding='latin1')

print("原始数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 彻底地清理无效行
print("\n最后10行原始数据:")
print(df.tail(10))

# 删除所有包含"undefined"或空值的行
df = df.replace('undefined', np.nan)
df = df.replace('', np.nan)

# 删除所有列都为空的行
df = df.dropna(how='all')

# 删除第一列为空的行
df = df.dropna(subset=['#'])

# 删除第一列不是数字的行
df = df[pd.to_numeric(df['#'], errors='coerce').notna()]

# 首先确保所有数值列都是正确的数据类型
numeric_columns = ['#', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 检查并处理Type 2列的异常值
print("\nType 2列的唯一值:")
print(df['Type 2'].unique())

# 将Type 2列中的非字符串和异常值转换为NaN
df['Type 2'] = df['Type 2'].apply(lambda x: x if isinstance(x, str) and x != '0' and x != 'BBB' and not x.replace('.', '').isdigit() else np.nan)

# 检查重复值
print(f"\n重复行数量: {df.duplicated().sum()}")
df = df.drop_duplicates()

# 处理Attack属性的异常值
print(f"\nAttack列统计信息:")
print(df['Attack'].describe())

# 查看Attack最高的几行数据
print("\nAttack最高的5行:")
print(df.nlargest(5, 'Attack')[['Name', 'Attack']])

# 处理异常高的Attack值（比如大于200的）
attack_threshold = 200
high_attack_mask = df['Attack'] > attack_threshold
print(f"\nAttack大于{attack_threshold}的行数: {high_attack_mask.sum()}")
if high_attack_mask.any():
    print("这些行是:")
    print(df[high_attack_mask][['Name', 'Attack']])

# 对于异常高的Attack值选择用中位数替换
attack_median = df['Attack'].median()
df.loc[high_attack_mask, 'Attack'] = attack_median

# 检查并修正generation与Legendary属性被置换的行
# 先检查数据类型
print(f"\n各列数据类型:")
print(df.dtypes)

# 检查Generation列中的非数值
print(f"\nGeneration列唯一值:")
print(df['Generation'].unique())

# 检查Legendary列中的非布尔值
print(f"\nLegendary列唯一值:")
print(df['Legendary'].unique())

# 修正属性置换的行
# 查找Generation列中包含布尔值的行
generation_bool_mask = df['Generation'].astype(str).str.upper().isin(['TRUE', 'FALSE'])
legendary_num_mask = pd.to_numeric(df['Legendary'], errors='coerce').notna()

# 交换这些行的Generation和Legendary列值
swap_mask = generation_bool_mask & legendary_num_mask
if swap_mask.any():
    print(f"\n需要修正的行数: {swap_mask.sum()}")
    temp_gen = df.loc[swap_mask, 'Generation'].copy()
    df.loc[swap_mask, 'Generation'] = df.loc[swap_mask, 'Legendary']
    df.loc[swap_mask, 'Legendary'] = temp_gen

df['Generation'] = pd.to_numeric(df['Generation'], errors='coerce')
df['Legendary'] = df['Legendary'].astype(str).str.upper().map({'TRUE': True, 'FALSE': False})

for col in ['HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']:
    if col in df.columns:
        # 处理异常高的值（大于1000的）
        high_mask = df[col] > 1000
        if high_mask.any():
            print(f"\n{col}列中有{high_mask.sum()}个值大于1000，用中位数替换")
            median_val = df[col].median()
            df.loc[high_mask, col] = median_val

print(f"\n各列缺失值数量:")
print(df.isnull().sum())

for col in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total', 'Generation']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in ['Type 1', 'Type 2', 'Legendary']:
    if col in df.columns:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)

print(f"\n清洗后数据形状: {df.shape}")
print(f"\n清洗后各列数据类型:")
print(df.dtypes)
print(f"\n清洗后缺失值数量:")
print(df.isnull().sum())

print("\n清洗后最后5行数据:")
print(df.tail())

df.to_csv('Pokemon_cleaned.csv', index=False, encoding='utf-8')
print("\n数据清洗完成! 清洗后的数据已保存为 'Pokemon_cleaned.csv'")

print(f"\n清洗后数据的基本统计信息:")
print(df.describe())

print(f"\n前5行清洗后的数据:")
print(df.head())

print("\n=== 数据质量报告 ===")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"数值列: {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"分类列: {df.select_dtypes(include=['object']).columns.tolist()}")
print(f"布尔列: {df.select_dtypes(include=['bool']).columns.tolist()}")