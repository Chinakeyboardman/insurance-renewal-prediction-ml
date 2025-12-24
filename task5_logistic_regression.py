"""
任务5: 使用逻辑回归对renewal进行建模，打印逻辑回归的系数（全部），并进行可视化（能看出系数的正负）
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import warnings
import os

warnings.filterwarnings('ignore')

# 设置matplotlib缓存目录
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("=" * 80)
print("任务5: 逻辑回归建模")
print("=" * 80)
print("\n读取数据...")
data = pd.read_excel('policy_data.xlsx')

# 数据预处理
print("\n数据预处理...")

# 处理分类特征
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
if 'renewal' in categorical_features:
    categorical_features.remove('renewal')  # 移除目标变量

print(f"分类特征: {categorical_features}")

# 使用LabelEncoder编码分类特征
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = data[feature].fillna('未知')
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# 处理日期特征
date_features = ['policy_start_date', 'policy_end_date']
for feature in date_features:
    if feature in data.columns:
        data[feature] = pd.to_datetime(data[feature])
        data[f'{feature}_year'] = data[feature].dt.year
        data[f'{feature}_month'] = data[feature].dt.month
        if feature == 'policy_start_date' and 'policy_end_date' in data.columns:
            data['policy_duration_days'] = (data['policy_end_date'] - data['policy_start_date']).dt.days

# 删除原始日期列
data = data.drop(columns=date_features, errors='ignore')

# 准备特征和目标变量
print("\n准备特征和目标变量...")
data['renewal'] = data['renewal'].map({'Yes': 1, 'No': 0})

# 选择特征（排除policy_id）
X = data.drop(['renewal', 'policy_id'], axis=1, errors='ignore')
y = data['renewal']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建逻辑回归模型
print("\n构建逻辑回归模型...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# 模型评估
print("\n模型评估...")
y_pred = lr.predict(X_test_scaled)
y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# ==================== 打印全部逻辑回归系数 ====================
print("\n" + "=" * 80)
print("逻辑回归系数（全部）")
print("=" * 80)

# 创建系数DataFrame
coef_df = pd.DataFrame({
    '特征名称': X.columns,
    '系数值': lr.coef_[0],
    '系数绝对值': np.abs(lr.coef_[0])
})

# 按系数绝对值排序
coef_df = coef_df.sort_values('系数绝对值', ascending=False)

# 设置pandas显示选项，确保显示所有行和列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("\n所有特征的逻辑回归系数:")
print(coef_df.to_string(index=False))

# 保存系数到CSV
coef_df.to_csv('lr_all_coefficients.csv', index=False, encoding='utf-8-sig')
print("\n✓ 所有系数已保存到 lr_all_coefficients.csv")

# 打印系数统计信息
print("\n系数统计信息:")
print(f"  正系数数量: {sum(coef_df['系数值'] > 0)}")
print(f"  负系数数量: {sum(coef_df['系数值'] < 0)}")
print(f"  零系数数量: {sum(coef_df['系数值'] == 0)}")
print(f"  最大正系数: {coef_df[coef_df['系数值'] > 0]['系数值'].max():.6f}")
print(f"  最小负系数: {coef_df[coef_df['系数值'] < 0]['系数值'].min():.6f}")

# ==================== 可视化系数（能看出正负） ====================
print("\n生成系数可视化图...")

# 图1: 所有系数的条形图（按系数值排序，显示正负）
fig, ax = plt.subplots(figsize=(14, max(8, len(X.columns) * 0.3)))
coef_df_sorted = coef_df.sort_values('系数值')

# 根据正负值设置颜色
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in coef_df_sorted['系数值']]

bars = ax.barh(range(len(coef_df_sorted)), coef_df_sorted['系数值'], color=colors, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(coef_df_sorted)))
ax.set_yticklabels(coef_df_sorted['特征名称'], fontsize=9)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('系数值', fontsize=12, fontweight='bold')
ax.set_title('逻辑回归系数可视化（全部特征，正负影响）', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, coef_df_sorted['系数值'])):
    ax.text(val, i, f' {val:.4f}', 
            va='center', ha='left' if val > 0 else 'right', 
            fontsize=8, fontweight='bold')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='正系数（促进续保）'),
    Patch(facecolor='#e74c3c', label='负系数（抑制续保）')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('lr_all_coefficients_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 系数可视化图已保存: lr_all_coefficients_visualization.png")

# 图2: 前20个最重要系数的可视化
top_n = min(20, len(coef_df))
fig, ax = plt.subplots(figsize=(12, 8))
top_coef = coef_df.head(top_n)
colors_top = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_coef['系数值']]

bars = ax.barh(range(len(top_coef)), top_coef['系数值'], color=colors_top, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_coef)))
ax.set_yticklabels(top_coef['特征名称'], fontsize=10)
ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('系数值', fontsize=12, fontweight='bold')
ax.set_title(f'逻辑回归系数可视化（前{top_n}个最重要特征）', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, top_coef['系数值'])):
    ax.text(val, i, f' {val:.4f}', 
            va='center', ha='left' if val > 0 else 'right', 
            fontsize=9, fontweight='bold')

ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('lr_top_coefficients_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 前{top_n}个系数可视化图已保存: lr_top_coefficients_visualization.png")

# 图3: 系数分布直方图
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(coef_df['系数值'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零线')
ax.set_xlabel('系数值', fontsize=12, fontweight='bold')
ax.set_ylabel('频数', fontsize=12, fontweight='bold')
ax.set_title('逻辑回归系数分布直方图', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('lr_coefficients_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 系数分布直方图已保存: lr_coefficients_distribution.png")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假正例率 (FPR)', fontsize=12)
ax.set_ylabel('真正例率 (TPR)', fontsize=12)
ax.set_title('逻辑回归模型ROC曲线', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lr_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC曲线已保存: lr_roc_curve.png")

print("\n" + "=" * 80)
print("逻辑回归建模完成！")
print("=" * 80)
print("\n生成的文件:")
print("  - lr_all_coefficients.csv: 所有系数数据")
print("  - lr_all_coefficients_visualization.png: 全部系数可视化图")
print("  - lr_top_coefficients_visualization.png: 前20个系数可视化图")
print("  - lr_coefficients_distribution.png: 系数分布直方图")
print("  - lr_roc_curve.png: ROC曲线")

