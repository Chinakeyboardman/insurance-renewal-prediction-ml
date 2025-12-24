"""
任务12 v2: LightGBM参数优化和模型训练
基于 task12_lgbm_scorecard.py 的优化版本
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
try:
    import lightgbm as lgb
except ImportError:
    print("=" * 80)
    print("错误: LightGBM未安装")
    print("=" * 80)
    print("请先安装LightGBM: pip install lightgbm")
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
print("任务12 v2: LightGBM参数优化")
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

# ==================== 参数优化 ====================
print("\n" + "=" * 80)
print("参数优化（使用网格搜索）")
print("=" * 80)

# 定义参数搜索空间
param_grid = {
    'num_leaves': [15, 31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.8, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'min_data_in_leaf': [10, 20, 30],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# 基础参数
base_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    'feature_pre_filter': False  # 允许动态改变min_data_in_leaf
}

print("\n参数搜索空间:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# 网格搜索（简化版，只测试关键参数组合）
best_score = 0
best_params = None
best_model = None
iteration = 0
total_combinations = 1
for k, v in param_grid.items():
    total_combinations *= len(v)
print(f"\n总组合数: {total_combinations} (将进行简化搜索)")

# 关键参数组合（减少搜索空间）
key_combinations = [
    {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'min_data_in_leaf': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
    {'num_leaves': 15, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'min_data_in_leaf': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
    {'num_leaves': 31, 'learning_rate': 0.01, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'min_data_in_leaf': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
    {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.7, 'min_data_in_leaf': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
    {'num_leaves': 50, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'min_data_in_leaf': 10, 'reg_alpha': 0, 'reg_lambda': 0},
]

print(f"\n测试 {len(key_combinations)} 个关键参数组合...")

for i, params in enumerate(key_combinations, 1):
    print(f"\n[{i}/{len(key_combinations)}] 测试参数组合:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 合并参数
    current_params = {**base_params, **params}
    
    # 训练模型
    model = lgb.train(
        current_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['eval'],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)  # 不打印日志
        ]
    )
    
    # 评估
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = auc(*roc_curve(y_val, y_val_pred)[:2])
    
    print(f"  验证集AUC: {val_auc:.4f}")
    
    if val_auc > best_score:
        best_score = val_auc
        best_params = current_params.copy()
        best_model = model
        print(f"  ✓ 新的最佳参数！")

print("\n" + "=" * 80)
print("最佳参数:")
print("=" * 80)
for key, value in best_params.items():
    if key not in ['objective', 'metric', 'boosting_type', 'verbose', 'random_state']:
        print(f"  {key}: {value}")
print(f"\n最佳验证集AUC: {best_score:.4f}")

# ==================== 使用最佳参数重新训练 ====================
print("\n" + "=" * 80)
print("使用最佳参数训练最终模型")
print("=" * 80)

# 使用最佳参数训练最终模型
final_model = lgb.train(
    best_params,
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
print("最终模型评估")
print("=" * 80)

# 预测
y_train_pred = final_model.predict(X_train, num_iteration=final_model.best_iteration)
y_val_pred = final_model.predict(X_val, num_iteration=final_model.best_iteration)
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

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

# 过拟合检查
overfitting = train_accuracy - test_accuracy
print(f"\n过拟合分析:")
print(f"  训练集-测试集准确率差异: {overfitting:.4f} ({overfitting*100:.2f}%)")
if overfitting < 0.05:
    print("  ✓ 模型泛化能力良好，无明显过拟合")
elif overfitting < 0.10:
    print("  ⚠️  模型存在轻微过拟合")
else:
    print("  ❌ 模型存在明显过拟合，建议进一步调整参数")

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
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# 保存模型
model_path = 'lgbm_scorecard_model_optimized.txt'
final_model.save_model(model_path)
print(f"\n✓ 优化后的模型已保存到: {model_path}")

# 保存最佳参数
import json
with open('lgbm_best_params.json', 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)
print("✓ 最佳参数已保存到: lgbm_best_params.json")

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
ax.set_title('LightGBM特征重要性（优化后，前15名）', fontsize=14, fontweight='bold')
ax.set_xlabel('重要性', fontsize=12)
ax.set_ylabel('特征名称', fontsize=12)
plt.tight_layout()
plt.savefig('lgbm_feature_importance_optimized.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 特征重要性图已保存: lgbm_feature_importance_optimized.png")

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
ax.set_title('LightGBM模型ROC曲线（优化后）', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lgbm_roc_curve_optimized.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC曲线已保存: lgbm_roc_curve_optimized.png")

# 3. 混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('LightGBM混淆矩阵（优化后，测试集）', fontsize=14, fontweight='bold')
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig('lgbm_confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 混淆矩阵图已保存: lgbm_confusion_matrix_optimized.png")

# 4. 预测概率分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 续保客户概率分布
axes[0].hist(y_test_pred[y_test == 1], bins=30, alpha=0.7, label='续保客户', color='green', edgecolor='black')
axes[0].hist(y_test_pred[y_test == 0], bins=30, alpha=0.7, label='不续保客户', color='red', edgecolor='black')
axes[0].set_xlabel('预测概率', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title('预测概率分布（优化后，测试集）', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 概率分布箱线图
prob_df = pd.DataFrame({
    '续保状态': ['续保' if x == 1 else '不续保' for x in y_test],
    '预测概率': y_test_pred
})
sns.boxplot(x='续保状态', y='预测概率', data=prob_df, ax=axes[1])
axes[1].set_title('预测概率箱线图（优化后，测试集）', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lgbm_probability_distribution_optimized.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 概率分布图已保存: lgbm_probability_distribution_optimized.png")

print("\n" + "=" * 80)
print("LightGBM模型优化完成！")
print("=" * 80)
print(f"\n最佳迭代次数: {final_model.best_iteration}")
print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"测试集ROC-AUC: {test_auc:.4f}")
print(f"过拟合程度: {overfitting:.4f} ({overfitting*100:.2f}%)")

