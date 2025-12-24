"""
任务2: 统计不同年龄层的续保比例
"""
import pandas as pd
import numpy as np

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
print("=" * 80)
print("不同年龄层的续保比例统计")
print("=" * 80)

# 按年龄层分组统计
age_renewal_stats = data.groupby('age_group').agg({
    'renewal': ['count', lambda x: (x == 'Yes').sum(), lambda x: (x == 'No').sum()]
}).reset_index()

# 重命名列
age_renewal_stats.columns = ['age_group', '总人数', '续保人数', '不续保人数']

# 计算续保比例
age_renewal_stats['续保比例'] = (age_renewal_stats['续保人数'] / age_renewal_stats['总人数'] * 100).round(2)
age_renewal_stats['不续保比例'] = (age_renewal_stats['不续保人数'] / age_renewal_stats['总人数'] * 100).round(2)

# 按年龄层排序
age_order = ['20-29岁', '30-39岁', '40-49岁', '50-59岁', '60-69岁', '70岁以上']
age_renewal_stats['age_group'] = pd.Categorical(age_renewal_stats['age_group'], categories=age_order, ordered=True)
age_renewal_stats = age_renewal_stats.sort_values('age_group')

# 打印结果
print("\n详细统计表:")
print(age_renewal_stats.to_string(index=False))

print("\n" + "=" * 80)
print("续保比例汇总（按年龄层）")
print("=" * 80)
for _, row in age_renewal_stats.iterrows():
    print(f"{row['age_group']:10s}: 总人数={row['总人数']:4d}, "
          f"续保={row['续保人数']:4d} ({row['续保比例']:5.2f}%), "
          f"不续保={row['不续保人数']:4d} ({row['不续保比例']:5.2f}%)")

# 保存结果到CSV
age_renewal_stats.to_csv('age_renewal_stats.csv', index=False, encoding='utf-8-sig')
print("\n统计结果已保存到 age_renewal_stats.csv")

