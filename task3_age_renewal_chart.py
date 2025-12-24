"""
任务3: 用柱状图呈现不同年龄层的续保比例
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib缓存目录到工作目录
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取Excel文件
data = pd.read_excel('policy_data.xlsx')

# 定义年龄层
def get_age_group(age):
    """将年龄划分为不同的年龄层"""
    if age < 30:
        return '20-29岁'
    elif age < 40:
        return '30-39岁'
    elif age < 50:
        return '40-49岁'
    elif age < 60:
        return '50-59岁'
    elif age < 70:
        return '60-69岁'
    else:
        return '70岁以上'

# 添加年龄层列
data['age_group'] = data['age'].apply(get_age_group)

# 统计不同年龄层的续保比例
age_renewal_stats = data.groupby('age_group').agg({
    'renewal': ['count', lambda x: (x == 'Yes').sum(), lambda x: (x == 'No').sum()]
}).reset_index()

age_renewal_stats.columns = ['age_group', '总人数', '续保人数', '不续保人数']
age_renewal_stats['续保比例'] = (age_renewal_stats['续保人数'] / age_renewal_stats['总人数'] * 100).round(2)

# 按年龄层排序
age_order = ['20-29岁', '30-39岁', '40-49岁', '50-59岁', '60-69岁', '70岁以上']
age_renewal_stats['age_group'] = pd.Categorical(age_renewal_stats['age_group'], categories=age_order, ordered=True)
age_renewal_stats = age_renewal_stats.sort_values('age_group')

# 创建柱状图
fig, ax = plt.subplots(figsize=(12, 6))

# 准备数据
age_groups = age_renewal_stats['age_group'].tolist()
renewal_rates = age_renewal_stats['续保比例'].tolist()
colors = ['#2ecc71' if rate >= 90 else '#f39c12' if rate >= 70 else '#e74c3c' for rate in renewal_rates]

# 绘制柱状图
bars = ax.bar(age_groups, renewal_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# 在柱状图上添加数值标签
for i, (bar, rate) in enumerate(zip(bars, renewal_rates)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 设置图表标题和标签
ax.set_title('不同年龄层的续保比例', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('年龄层', fontsize=12, fontweight='bold')
ax.set_ylabel('续保比例 (%)', fontsize=12, fontweight='bold')

# 设置y轴范围
ax.set_ylim(0, max(renewal_rates) * 1.15)

# 添加网格线
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加图例说明颜色含义
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='续保比例 ≥ 90%'),
    Patch(facecolor='#f39c12', label='续保比例 70-90%'),
    Patch(facecolor='#e74c3c', label='续保比例 < 70%')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('age_renewal_rate_chart.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存
print("=" * 80)
print("柱状图已生成并保存为 age_renewal_rate_chart.png")
print("=" * 80)

# 打印统计信息
print("\n不同年龄层续保比例统计:")
print(age_renewal_stats[['age_group', '总人数', '续保人数', '续保比例']].to_string(index=False))

