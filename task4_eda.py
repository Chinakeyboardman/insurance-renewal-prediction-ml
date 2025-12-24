"""
任务4: 进行EDA（探索性数据分析）
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path
# 设置matplotlib缓存目录
os.environ['MPLCONFIGDIR'] = os.path.join(Path(__file__).parent, '.matplotlib')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("=" * 80)
print("EDA - 探索性数据分析")
print("=" * 80)
current_dir = Path(__file__).parent  # 脚本所在目录
file_path = os.path.join(current_dir, 'policy_data.xlsx')
data = pd.read_excel(file_path)

# ==================== 1. 数据基本信息 ====================
print("\n【1. 数据基本信息】")
print(f"数据形状: {data.shape}")
print(f"总行数: {len(data)}")
print(f"总列数: {len(data.columns)}")
print(f"\n列名及数据类型:")
print(data.dtypes)

# ==================== 2. 缺失值分析 ====================
print("\n【2. 缺失值分析】")
missing_data = data.isnull().sum()
missing_percent = (missing_data / len(data)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_data,
    '缺失比例(%)': missing_percent.round(2)
})
missing_df = missing_df[missing_df['缺失数量'] > 0]
if len(missing_df) > 0:
    print("存在缺失值的列:")
    print(missing_df)
else:
    print("✓ 数据完整，无缺失值")

# ==================== 3. 目标变量分析 ====================
print("\n【3. 目标变量(renewal)分析】")
renewal_counts = data['renewal'].value_counts()
renewal_percent = data['renewal'].value_counts(normalize=True) * 100
print("续保分布:")
for val, count in renewal_counts.items():
    print(f"  {val}: {count} ({renewal_percent[val]:.2f}%)")

# 续保率可视化
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(renewal_counts.index, renewal_counts.values, color=colors, alpha=0.8, edgecolor='black')
for bar, count, pct in zip(bars, renewal_counts.values, renewal_percent.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{count}\n({pct:.2f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('续保分布', fontsize=14, fontweight='bold')
ax.set_xlabel('续保状态', fontsize=12)
ax.set_ylabel('人数', fontsize=12)
plt.tight_layout()
plt.savefig('eda_renewal_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 续保分布图已保存: eda_renewal_distribution.png")

# ==================== 4. 数值型特征分析 ====================
print("\n【4. 数值型特征分析】")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if 'policy_id' in numeric_cols:
    numeric_cols.remove('policy_id')
print(f"数值型特征: {numeric_cols}")

if numeric_cols:
    print("\n数值型特征统计描述:")
    print(data[numeric_cols].describe())
    
    # 数值型特征与续保的关系
    print("\n数值型特征与续保的关系:")
    for col in numeric_cols:
        if col != 'policy_id':
            renewal_by_feature = data.groupby('renewal')[col].agg(['mean', 'median', 'std'])
            print(f"\n{col}:")
            print(renewal_by_feature)

# ==================== 5. 分类特征分析 ====================
print("\n【5. 分类特征分析】")
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
if 'renewal' in categorical_cols:
    categorical_cols.remove('renewal')
print(f"分类特征: {categorical_cols}")

# 各分类特征与续保的关系
print("\n各分类特征与续保的关系:")
for col in categorical_cols[:5]:  # 先分析前5个分类特征
    print(f"\n{col}:")
    crosstab = pd.crosstab(data[col], data['renewal'], margins=True)
    crosstab_pct = pd.crosstab(data[col], data['renewal'], normalize='index') * 100
    print("数量统计:")
    print(crosstab)
    print("\n续保比例:")
    print(crosstab_pct.round(2))

# ==================== 6. 年龄分析 ====================
print("\n【6. 年龄分析】")
print("年龄统计描述:")
print(data['age'].describe())

# 年龄分布直方图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 总体年龄分布
axes[0].hist(data['age'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_title('客户年龄分布', fontsize=12, fontweight='bold')
axes[0].set_xlabel('年龄', fontsize=10)
axes[0].set_ylabel('频数', fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# 按续保状态分组的年龄分布
for renewal_status in ['Yes', 'No']:
    subset = data[data['renewal'] == renewal_status]
    axes[1].hist(subset['age'], bins=20, alpha=0.6, label=f'续保={renewal_status}', edgecolor='black')
axes[1].set_title('按续保状态分组的年龄分布', fontsize=12, fontweight='bold')
axes[1].set_xlabel('年龄', fontsize=10)
axes[1].set_ylabel('频数', fontsize=10)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_age_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 年龄分析图已保存: eda_age_analysis.png")

# ==================== 7. 性别分析 ====================
print("\n【7. 性别分析】")
gender_renewal = pd.crosstab(data['gender'], data['renewal'], normalize='index') * 100
print("性别与续保关系:")
print(gender_renewal.round(2))

fig, ax = plt.subplots(figsize=(8, 6))
gender_counts = data['gender'].value_counts()
ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
       startangle=90, colors=['#3498db', '#e74c3c'])
ax.set_title('客户性别分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_gender_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 性别分布图已保存: eda_gender_distribution.png")

# ==================== 8. 保费分析 ====================
print("\n【8. 保费分析】")
if 'premium_amount' in data.columns:
    print("保费统计描述:")
    print(data['premium_amount'].describe())
    
    renewal_by_premium = data.groupby('renewal')['premium_amount'].agg(['mean', 'median', 'std'])
    print("\n按续保状态分组的保费统计:")
    print(renewal_by_premium)
    
    # 保费分布箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column='premium_amount', by='renewal', ax=ax)
    ax.set_title('按续保状态分组的保费分布', fontsize=12, fontweight='bold')
    ax.set_xlabel('续保状态', fontsize=10)
    ax.set_ylabel('保费金额', fontsize=10)
    plt.suptitle('')  # 移除默认标题
    plt.tight_layout()
    plt.savefig('eda_premium_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 保费分析图已保存: eda_premium_analysis.png")

# ==================== 9. 收入水平分析 ====================
print("\n【9. 收入水平分析】")
if 'income_level' in data.columns:
    income_renewal = pd.crosstab(data['income_level'], data['renewal'], normalize='index') * 100
    print("收入水平与续保关系:")
    print(income_renewal.round(2))

# ==================== 10. 理赔历史分析 ====================
print("\n【10. 理赔历史分析】")
if 'claim_history' in data.columns:
    claim_renewal = pd.crosstab(data['claim_history'], data['renewal'], normalize='index') * 100
    print("理赔历史与续保关系:")
    print(claim_renewal.round(2))

# ==================== 总结 ====================
print("\n" + "=" * 80)
print("EDA分析完成！")
print("=" * 80)
print("\n生成的可视化文件:")
print("  - eda_renewal_distribution.png: 续保分布图")
print("  - eda_age_analysis.png: 年龄分析图")
print("  - eda_gender_distribution.png: 性别分布图")
if 'premium_amount' in data.columns:
    print("  - eda_premium_analysis.png: 保费分析图")

