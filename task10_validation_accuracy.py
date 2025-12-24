"""
任务10: 对决策树模型打印验证集的准确性
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import warnings
import os

warnings.filterwarnings('ignore')

# 读取数据
print("=" * 80)
print("任务10: 决策树模型验证集准确性评估")
print("=" * 80)
print("\n读取数据...")
data = pd.read_excel('policy_data.xlsx')

# 数据预处理
print("\n数据预处理...")

# 处理分类特征
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
if 'renewal' in categorical_features:
    categorical_features.remove('renewal')

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

# 划分数据集：训练集60%，验证集20%，测试集20%
print("\n划分数据集（训练集60%，验证集20%，测试集20%）...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集大小: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"验证集大小: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 构建决策树模型（depth=4）
print("\n构建决策树模型（max_depth=4）...")
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)

# ==================== 验证集评估 ====================
print("\n" + "=" * 80)
print("验证集评估结果")
print("=" * 80)

# 在验证集上进行预测
y_val_pred = dt.predict(X_val_scaled)
y_val_pred_proba = dt.predict_proba(X_val_scaled)[:, 1]

# 计算准确率
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\n验证集准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# 分类报告
print("\n验证集分类报告:")
print(classification_report(y_val, y_val_pred, target_names=['不续保', '续保']))

# 混淆矩阵
print("\n验证集混淆矩阵:")
cm_val = confusion_matrix(y_val, y_val_pred)
print(cm_val)
print("\n混淆矩阵详细说明:")
print(f"  真正例 (TP): {cm_val[1][1]} - 实际续保，预测续保")
print(f"  真负例 (TN): {cm_val[0][0]} - 实际不续保，预测不续保")
print(f"  假正例 (FP): {cm_val[0][1]} - 实际不续保，预测续保")
print(f"  假负例 (FN): {cm_val[1][0]} - 实际续保，预测不续保")

# 计算其他指标
precision = cm_val[1][1] / (cm_val[1][1] + cm_val[0][1]) if (cm_val[1][1] + cm_val[0][1]) > 0 else 0
recall = cm_val[1][1] / (cm_val[1][1] + cm_val[1][0]) if (cm_val[1][1] + cm_val[1][0]) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n验证集详细指标:")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  召回率 (Recall): {recall:.4f}")
print(f"  F1分数: {f1_score:.4f}")

# ROC曲线和AUC
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
roc_auc_val = auc(fpr_val, tpr_val)
print(f"  ROC-AUC: {roc_auc_val:.4f}")

# ==================== 对比训练集和测试集 ====================
print("\n" + "=" * 80)
print("模型在不同数据集上的表现对比")
print("=" * 80)

# 训练集评估
y_train_pred = dt.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# 测试集评估
y_test_pred = dt.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n验证集准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# 计算过拟合程度
overfitting_train = train_accuracy - val_accuracy
overfitting_test = train_accuracy - test_accuracy

print(f"\n过拟合分析:")
print(f"  训练集 vs 验证集差异: {overfitting_train:.4f} ({overfitting_train*100:.2f}%)")
print(f"  训练集 vs 测试集差异: {overfitting_test:.4f} ({overfitting_test*100:.2f}%)")

if abs(overfitting_train) < 0.05:
    print("  ✓ 模型泛化能力良好，无明显过拟合")
elif abs(overfitting_train) < 0.10:
    print("  ⚠️  模型存在轻微过拟合，建议调整参数")
else:
    print("  ❌ 模型存在明显过拟合，需要调整模型参数或增加正则化")

# ==================== 验证集详细统计 ====================
print("\n" + "=" * 80)
print("验证集详细统计")
print("=" * 80)

print(f"\n验证集样本分布:")
print(f"  总样本数: {len(y_val)}")
print(f"  续保样本 (Yes): {sum(y_val == 1)} ({sum(y_val == 1)/len(y_val)*100:.2f}%)")
print(f"  不续保样本 (No): {sum(y_val == 0)} ({sum(y_val == 0)/len(y_val)*100:.2f}%)")

print(f"\n验证集预测分布:")
print(f"  预测续保 (Yes): {sum(y_val_pred == 1)} ({sum(y_val_pred == 1)/len(y_val_pred)*100:.2f}%)")
print(f"  预测不续保 (No): {sum(y_val_pred == 0)} ({sum(y_val_pred == 0)/len(y_val_pred)*100:.2f}%)")

# 按类别统计准确率
if sum(y_val == 0) > 0:
    accuracy_class_0 = sum((y_val == 0) & (y_val_pred == 0)) / sum(y_val == 0)
    print(f"\n不续保类别准确率: {accuracy_class_0:.4f} ({accuracy_class_0*100:.2f}%)")

if sum(y_val == 1) > 0:
    accuracy_class_1 = sum((y_val == 1) & (y_val_pred == 1)) / sum(y_val == 1)
    print(f"续保类别准确率: {accuracy_class_1:.4f} ({accuracy_class_1*100:.2f}%)")

print("\n" + "=" * 80)
print("验证集评估完成！")
print("=" * 80)
print(f"\n最终验证集准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"ROC-AUC: {roc_auc_val:.4f}")

