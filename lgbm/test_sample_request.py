"""
测试示例请求参数
使用测试集中的第一条数据进行预测
"""
import json
import os
from task13_score_prediction_api import RenewalScorecard

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_file = os.path.join(script_dir, 'sample_request.json')

# 加载示例请求参数
if not os.path.exists(sample_file):
    print(f"错误: 找不到文件 {sample_file}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {script_dir}")
    exit(1)

with open(sample_file, 'r', encoding='utf-8') as f:
    sample_request = json.load(f)

print("=" * 80)
print("测试示例请求参数")
print("=" * 80)
print("\n请求参数:")
print(json.dumps(sample_request, ensure_ascii=False, indent=2))

# 初始化评分卡
print("\n" + "=" * 80)
print("初始化评分卡模型...")
print("=" * 80)
scorecard = RenewalScorecard(model_dir='.')

# 进行预测
print("\n" + "=" * 80)
print("预测结果")
print("=" * 80)
result = scorecard.predict_score(sample_request)

print("\n预测结果详情:")
print(result.to_string(index=False))

print("\n" + "=" * 80)
print("结果说明")
print("=" * 80)
print(f"续保概率: {result['renewal_probability'].iloc[0]:.4f} ({result['renewal_probability'].iloc[0]*100:.2f}%)")
print(f"续保评分: {result['renewal_score'].iloc[0]:.2f} 分")
print(f"预测结果: {'续保' if result['renewal_prediction'].iloc[0] == 1 else '不续保'}")
print(f"风险等级: {result['risk_level'].iloc[0]}")

