"""
任务1: 读取 policy_data.xlsx 前5行数据，显示全部列
"""
import os
import pandas as pd
from pathlib import Path

# 读取Excel文件
current_dir = Path(__file__).parent  # 脚本所在目录
file_path = os.path.join(current_dir, 'policy_data.xlsx')
data = pd.read_excel(file_path)

# 打印数据基本信息
print("=" * 80)
print("数据基本信息")
print("=" * 80)
print(f"数据形状: {data.shape}")
print(f"总行数: {len(data)}")
print(f"总列数: {len(data.columns)}")
print(f"\n列名列表:")
for i, col in enumerate(data.columns, 1):
    print(f"  {i}. {col}")

# 打印前5行数据，显示全部列
print("\n" + "=" * 80)
print("前5行数据（显示全部列）")
print("=" * 80)
for i in range(min(5, len(data))):
    print(f"\n行 {i+1}:")
    print("-" * 80)
    row = data.iloc[i]
    for col in data.columns:
        print(f"  {col}: {row[col]}")

# 也使用pandas的head方法显示
print("\n" + "=" * 80)
print("使用 pandas head() 方法显示前5行")
print("=" * 80)
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', 50)  # 限制每列最大宽度
print(data.head(5))

