"""
任务13: 实现评分转换和预测接口
将LightGBM模型预测的概率转换为标准评分（300-850分）
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
try:
    import lightgbm as lgb
except ImportError:
    print("错误: LightGBM未安装，请先安装: pip install lightgbm")
    exit(1)

class RenewalScorecard:
    """续保评分卡类"""
    
    def __init__(self, model_dir='.'):
        """
        初始化评分卡
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.best_params = None
        
        # 评分卡参数（标准评分卡公式）
        self.base_score = 600  # 基础分
        self.pdo = 20  # Points to Double Odds（odds翻倍所需分数）
        self.factor = self.pdo / np.log(2)  # 评分因子
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和相关文件"""
        print(f"从 {self.model_dir} 目录加载模型...")
        
        # 加载模型
        model_path = os.path.join(self.model_dir, 'lgbm_scorecard_model_optimized.txt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = lgb.Booster(model_file=model_path)
        print("✓ 模型加载成功")
        
        # 加载编码器
        encoder_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("✓ 编码器加载成功")
        
        # 加载特征名称
        feature_path = os.path.join(self.model_dir, 'feature_names.pkl')
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print("✓ 特征名称加载成功")
        
        # 加载最佳参数
        params_path = os.path.join(self.model_dir, 'lgbm_best_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                self.best_params = json.load(f)
            print("✓ 参数加载成功")
    
    def _preprocess_data(self, data):
        """
        数据预处理
        
        Args:
            data: 原始数据DataFrame
            
        Returns:
            预处理后的数据DataFrame
        """
        data = data.copy()
        
        # 处理分类特征
        categorical_features = ['gender', 'birth_region', 'insurance_region', 
                              'income_level', 'education_level', 'occupation', 
                              'marital_status', 'policy_type', 'policy_term', 'claim_history']
        
        if self.label_encoders:
            for feature in categorical_features:
                if feature in data.columns and feature in self.label_encoders:
                    # 处理缺失值
                    data[feature] = data[feature].fillna('未知')
                    # 处理新值（不在训练集中的值）
                    le = self.label_encoders[feature]
                    known_classes = set(le.classes_)
                    data[feature] = data[feature].apply(
                        lambda x: x if x in known_classes else '未知'
                    )
                    data[feature] = le.transform(data[feature])
        
        # 处理日期特征
        date_features = ['policy_start_date', 'policy_end_date']
        for feature in date_features:
            if feature in data.columns:
                data[feature] = pd.to_datetime(data[feature], errors='coerce')
                data[f'{feature}_year'] = data[feature].dt.year
                data[f'{feature}_month'] = data[feature].dt.month
                if feature == 'policy_start_date' and 'policy_end_date' in data.columns:
                    # 确保两个日期列都是datetime类型
                    if data['policy_end_date'].dtype != 'datetime64[ns]':
                        data['policy_end_date'] = pd.to_datetime(data['policy_end_date'], errors='coerce')
                    if data['policy_start_date'].dtype != 'datetime64[ns]':
                        data['policy_start_date'] = pd.to_datetime(data['policy_start_date'], errors='coerce')
                    data['policy_duration_days'] = (data['policy_end_date'] - data['policy_start_date']).dt.days
        
        # 删除原始日期列
        data = data.drop(columns=date_features, errors='ignore')
        
        # 选择特征（排除policy_id和renewal）
        feature_cols = [col for col in data.columns 
                       if col not in ['policy_id', 'renewal']]
        
        # 确保特征顺序与训练时一致
        if self.feature_names:
            # 检查是否有缺失的特征
            missing_features = set(self.feature_names) - set(feature_cols)
            if missing_features:
                print(f"警告: 缺少特征 {missing_features}，将使用默认值填充")
                for feat in missing_features:
                    data[feat] = 0
            
            # 按训练时的特征顺序排列
            data = data[self.feature_names]
        
        return data
    
    def predict_proba(self, data):
        """
        预测续保概率
        
        Args:
            data: 输入数据（DataFrame或字典列表）
            
        Returns:
            续保概率数组
        """
        # 转换为DataFrame
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 预处理
        X = self._preprocess_data(data)
        
        # 预测
        probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        return probabilities
    
    def probability_to_score(self, probability):
        """
        将概率转换为评分
        
        标准评分卡公式: Score = Base + Factor × ln(odds)
        其中 odds = p / (1-p)
        
        Args:
            probability: 续保概率（0-1之间）
            
        Returns:
            评分（300-850分）
        """
        # 避免概率为0或1的情况
        probability = np.clip(probability, 0.0001, 0.9999)
        
        # 计算odds
        odds = probability / (1 - probability)
        
        # 计算评分
        score = self.base_score + self.factor * np.log(odds)
        
        # 限制评分范围在300-850之间
        score = np.clip(score, 300, 850)
        
        return score
    
    def predict_score(self, data):
        """
        预测续保评分
        
        Args:
            data: 输入数据（DataFrame或字典列表）
            
        Returns:
            包含预测结果的DataFrame
        """
        # 预测概率
        probabilities = self.predict_proba(data)
        
        # 转换为评分
        scores = self.probability_to_score(probabilities)
        
        # 构建结果
        if isinstance(data, list):
            result_data = pd.DataFrame(data)
        elif isinstance(data, dict):
            result_data = pd.DataFrame([data])
        else:
            result_data = data.copy()
        
        result = pd.DataFrame({
            'renewal_probability': probabilities,
            'renewal_score': scores,
            'renewal_prediction': (probabilities > 0.5).astype(int),
            'risk_level': pd.cut(scores, 
                               bins=[0, 500, 650, 800, 850],
                               labels=['高风险', '中风险', '低风险', '极低风险'])
        })
        
        # 合并原始数据
        if 'policy_id' in result_data.columns:
            result['policy_id'] = result_data['policy_id'].values
        
        return result
    
    def batch_predict(self, file_path, output_path=None):
        """
        批量预测
        
        Args:
            file_path: 输入文件路径（Excel或CSV）
            output_path: 输出文件路径（可选）
            
        Returns:
            预测结果DataFrame
        """
        print(f"\n读取数据文件: {file_path}")
        
        # 读取数据
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            raise ValueError("不支持的文件格式，请使用Excel或CSV文件")
        
        print(f"数据行数: {len(data)}")
        
        # 预测
        print("开始预测...")
        results = self.predict_score(data)
        
        # 保存结果
        if output_path:
            if output_path.endswith('.xlsx') or output_path.endswith('.xls'):
                results.to_excel(output_path, index=False)
            else:
                results.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✓ 预测结果已保存到: {output_path}")
        
        return results


def main():
    """主函数：演示评分卡使用"""
    print("=" * 80)
    print("任务13: 评分转换和预测接口")
    print("=" * 80)
    
    # 初始化评分卡（脚本在lgbm目录下，模型文件也在同一目录）
    scorecard = RenewalScorecard(model_dir='.')
    
    # 测试单个样本预测
    print("\n" + "=" * 80)
    print("测试单个样本预测")
    print("=" * 80)
    
    sample_data = {
        'policy_id': 10001,
        'age': 45,
        'gender': '男',
        'birth_region': '北京市',
        'insurance_region': '北京市',
        'income_level': '高',
        'education_level': '本科',
        'occupation': '工程师',
        'marital_status': '已婚',
        'family_members': 3,
        'policy_type': '守护百分百2021',
        'policy_term': '20年',
        'premium_amount': 15000.0,
        'policy_start_date': '2015-01-17',
        'policy_end_date': '2035-01-17',
        'claim_history': '否'
    }
    
    result = scorecard.predict_score(sample_data)
    print("\n预测结果:")
    print(result.to_string(index=False))
    
    # 测试批量预测（使用训练数据的前10条）
    print("\n" + "=" * 80)
    print("测试批量预测（训练数据前10条）")
    print("=" * 80)
    
    # 数据文件在上级目录
    data_file = '../policy_data.xlsx' if os.path.exists('../policy_data.xlsx') else 'policy_data.xlsx'
    train_data = pd.read_excel(data_file)
    test_samples = train_data.head(10).copy()
    
    batch_results = scorecard.predict_score(test_samples)
    print(f"\n批量预测结果（前5条）:")
    print(batch_results.head(5).to_string(index=False))
    
    # 保存预测结果示例
    output_file = 'prediction_sample_results.xlsx'
    batch_results.to_excel(output_file, index=False)
    print(f"\n✓ 批量预测结果已保存到: {output_file}")
    
    # 评分分布统计
    print("\n" + "=" * 80)
    print("评分分布统计")
    print("=" * 80)
    print(f"平均评分: {batch_results['renewal_score'].mean():.2f}")
    print(f"评分中位数: {batch_results['renewal_score'].median():.2f}")
    print(f"最低评分: {batch_results['renewal_score'].min():.2f}")
    print(f"最高评分: {batch_results['renewal_score'].max():.2f}")
    print(f"\n风险等级分布:")
    print(batch_results['risk_level'].value_counts())
    
    print("\n" + "=" * 80)
    print("评分转换和预测接口实现完成！")
    print("=" * 80)
    print("\n使用方法:")
    print("1. 初始化: scorecard = RenewalScorecard(model_dir='.')  # 在lgbm目录下")
    print("   或: scorecard = RenewalScorecard(model_dir='lgbm')  # 在test目录下")
    print("2. 单个预测: result = scorecard.predict_score(data_dict)")
    print("3. 批量预测: results = scorecard.batch_predict('input.xlsx', 'output.xlsx')")


if __name__ == '__main__':
    main()

