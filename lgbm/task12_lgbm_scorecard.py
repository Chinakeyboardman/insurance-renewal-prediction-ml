"""
任务12: 开发LightGBM打分模型
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
try:
    import lightgbm as lgb
except ImportError:
    print("=" * 80)
    print("错误: LightGBM未安装")
    print("=" * 80)
    print("请先安装LightGBM:")
    print("  pip install lightgbm")
    print("或者:")
    print("  conda install -c conda-forge lightgbm")
    exit(1)

import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# 设置matplotlib缓存目录
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("=" * 80)
print("任务12: LightGBM打分模型开发")
print("=" * 80)
print("\n读取数据...")
current_dir = Path(__file__).parent  # 脚本所在目录
file_path = os.path.join(current_dir, 'policy_data.xlsx')
data = pd.read_excel(file_path)

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

# 划分数据集：训练集70%，验证集15%，测试集15%
print("\n划分数据集（训练集70%，验证集15%，测试集15%）...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集大小: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"验证集大小: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 准备LightGBM数据格式
print("\n准备LightGBM数据格式...")
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)

# LightGBM参数设置
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42
}

print("\n训练LightGBM模型...")
print("参数设置:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 训练模型
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'eval'],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

# 模型评估
print("\n" + "=" * 80)
print("模型评估")
print("=" * 80)

# 预测
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 转换为二分类预测
y_train_pred_binary = (y_train_pred > 0.5).astype(int)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred_binary)
val_accuracy = accuracy_score(y_val, y_val_pred_binary)
test_accuracy = accuracy_score(y_test, y_test_pred_binary)

print(f"\n准确率:")
print(f"  训练集: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  验证集: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  测试集: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ROC-AUC
train_auc = auc(*roc_curve(y_train, y_train_pred)[:2])
val_auc = auc(*roc_curve(y_val, y_val_pred)[:2])
test_auc = auc(*roc_curve(y_test, y_test_pred)[:2])

print(f"\nROC-AUC:")
print(f"  训练集: {train_auc:.4f}")
print(f"  验证集: {val_auc:.4f}")
print(f"  测试集: {test_auc:.4f}")

# 测试集分类报告
print("\n测试集分类报告:")
print(classification_report(y_test, y_test_pred_binary, target_names=['不续保', '续保']))

# 混淆矩阵
print("\n测试集混淆矩阵:")
cm_test = confusion_matrix(y_test, y_test_pred_binary)
print(cm_test)

# 特征重要性
print("\n" + "=" * 80)
print("特征重要性（前15名）")
print("=" * 80)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# 保存模型
model_path = 'lgbm_scorecard_model.txt'
model.save_model(model_path)
print(f"\n✓ 模型已保存到: {model_path}")

# 保存编码器
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ 编码器已保存到: label_encoders.pkl")

# 保存特征列表
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("✓ 特征列表已保存到: feature_names.pkl")

# 可视化
print("\n生成可视化图表...")

# 1. 特征重要性图
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
ax.set_title('LightGBM特征重要性（前15名）', fontsize=14, fontweight='bold')
ax.set_xlabel('重要性', fontsize=12)
ax.set_ylabel('特征名称', fontsize=12)
plt.tight_layout()
plt.savefig('lgbm_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 特征重要性图已保存: lgbm_feature_importance.png")

# 2. ROC曲线
fig, ax = plt.subplots(figsize=(8, 6))
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

ax.plot(fpr_train, tpr_train, label=f'训练集 (AUC = {train_auc:.4f})', lw=2)
ax.plot(fpr_val, tpr_val, label=f'验证集 (AUC = {val_auc:.4f})', lw=2)
ax.plot(fpr_test, tpr_test, label=f'测试集 (AUC = {test_auc:.4f})', lw=2)
ax.plot([0, 1], [0, 1], 'k--', label='随机分类器', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假正例率 (FPR)', fontsize=12)
ax.set_ylabel('真正例率 (TPR)', fontsize=12)
ax.set_title('LightGBM模型ROC曲线', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lgbm_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC曲线已保存: lgbm_roc_curve.png")

# 3. 混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('LightGBM混淆矩阵（测试集）', fontsize=14, fontweight='bold')
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig('lgbm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 混淆矩阵图已保存: lgbm_confusion_matrix.png")

# 4. 预测概率分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 续保客户概率分布
axes[0].hist(y_test_pred[y_test == 1], bins=30, alpha=0.7, label='续保客户', color='green', edgecolor='black')
axes[0].hist(y_test_pred[y_test == 0], bins=30, alpha=0.7, label='不续保客户', color='red', edgecolor='black')
axes[0].set_xlabel('预测概率', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title('预测概率分布（测试集）', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 概率分布箱线图
prob_df = pd.DataFrame({
    '续保状态': ['续保' if x == 1 else '不续保' for x in y_test],
    '预测概率': y_test_pred
})
sns.boxplot(x='续保状态', y='预测概率', data=prob_df, ax=axes[1])
axes[1].set_title('预测概率箱线图（测试集）', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lgbm_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 概率分布图已保存: lgbm_probability_distribution.png")

print("\n" + "=" * 80)
print("LightGBM模型训练完成！")
print("=" * 80)
print(f"\n最佳迭代次数: {model.best_iteration}")
print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"测试集ROC-AUC: {test_auc:.4f}")

