"""
任务8: 使用决策树（depth=4）对renewal进行建模，打印决策树，并进行可视化
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
print("任务8: 决策树建模（depth=4）")
print("=" * 80)
print("\n读取数据...")
data = pd.read_excel('policy_data.xlsx')

# 数据预处理
print("\n数据预处理...")

# 处理分类特征
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
if 'renewal' in categorical_features:
    categorical_features.remove('renewal')

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

# 特征标准化（决策树通常不需要标准化，但为了保持一致性）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建决策树模型（depth=4）
print("\n构建决策树模型（max_depth=4）...")
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)

# 模型评估
print("\n模型评估...")
y_pred = dt.predict(X_test_scaled)
y_pred_proba = dt.predict_proba(X_test_scaled)[:, 1]

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ==================== 打印决策树文本表示 ====================
print("\n" + "=" * 80)
print("决策树文本表示（max_depth=4）")
print("=" * 80)

tree_text = export_text(dt, feature_names=list(X.columns), max_depth=4)
print(tree_text)

# 保存决策树文本
with open('decision_tree_text.txt', 'w', encoding='utf-8') as f:
    f.write("决策树模型（max_depth=4）\n")
    f.write("=" * 80 + "\n\n")
    f.write(tree_text)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("模型准确率: {:.4f}\n".format(accuracy_score(y_test, y_pred)))
    f.write("特征数量: {}\n".format(len(X.columns)))
    f.write("树深度: {}\n".format(dt.get_depth()))
    f.write("叶子节点数: {}\n".format(dt.get_n_leaves()))

print("\n✓ 决策树文本已保存到 decision_tree_text.txt")

# ==================== 可视化决策树 ====================
print("\n生成决策树可视化图...")

# 图1: 完整决策树可视化（depth=4）
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt, max_depth=4, feature_names=X.columns, class_names=['不续保', '续保'], 
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title('决策树可视化（max_depth=4）', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 决策树可视化图已保存: decision_tree_visualization.png")

# 图2: 特征重要性
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特征重要性（前10名）:")
print(feature_importances.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importances.head(15)
sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
ax.set_title('决策树特征重要性（前15名）', fontsize=14, fontweight='bold')
ax.set_xlabel('重要性', fontsize=12)
ax.set_ylabel('特征名称', fontsize=12)
plt.tight_layout()
plt.savefig('decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 特征重要性图已保存: decision_tree_feature_importance.png")

# 图3: 混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('决策树混淆矩阵', fontsize=14, fontweight='bold')
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig('decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 混淆矩阵图已保存: decision_tree_confusion_matrix.png")

# 图4: ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假正例率 (FPR)', fontsize=12)
ax.set_ylabel('真正例率 (TPR)', fontsize=12)
ax.set_title('决策树模型ROC曲线', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('decision_tree_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC曲线已保存: decision_tree_roc_curve.png")

# 打印模型详细信息
print("\n" + "=" * 80)
print("决策树模型详细信息")
print("=" * 80)
print(f"树深度: {dt.get_depth()}")
print(f"叶子节点数: {dt.get_n_leaves()}")
print(f"训练样本数: {len(X_train)}")
print(f"测试样本数: {len(X_test)}")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n" + "=" * 80)
print("决策树建模完成！")
print("=" * 80)
print("\n生成的文件:")
print("  - decision_tree_text.txt: 决策树文本表示")
print("  - decision_tree_visualization.png: 决策树可视化图")
print("  - decision_tree_feature_importance.png: 特征重要性图")
print("  - decision_tree_confusion_matrix.png: 混淆矩阵图")
print("  - decision_tree_roc_curve.png: ROC曲线图")

