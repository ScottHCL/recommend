"""
OneTrans模型推理示例
演示如何加载训练好的模型进行在线推理和部署
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OneTransModel
from config import OneTransConfig
from data_loader import create_sample_batch


class OneTransInferenceEngine:
    """OneTrans推理引擎：支持在线推理和批量推理"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.load_model()
        
        # 推理统计
        self.inference_stats = {
            'total_requests': 0,
            'avg_latency_ms': 0.0,
            'successful_requests': 0,
            'failed_requests': 0
        }
    
    def load_model(self):
        """加载模型"""
        print(f"加载模型从: {self.model_path}")
        
        # 加载配置
        config_path = self.model_path / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            self.config = OneTransConfig.from_dict(config_dict)
        
        # 创建模型
        self.model = OneTransModel(self.config)
        
        # 加载权重
        weights_path = self.model_path / 'model_weights.h5'
        if not weights_path.exists():
            raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
        
        self.model.load_weights(str(weights_path))
        print("模型加载成功!")
    
    def preprocess_input(self, 
                        user_features: Dict,
                        item_features: Dict,
                        context_features: Dict,
                        sequence_features: Dict) -> Tuple[Dict, Dict]:
        """预处理输入特征"""
        
        # 合并非序列特征
        non_seq_features = {}
        non_seq_features.update(user_features)
        non_seq_features.update(item_features)
        non_seq_features.update(context_features)
        
        # 处理序列特征（简化实现）
        processed_seq_features = {}
        for seq_type, seq_data in sequence_features.items():
            if isinstance(seq_data, list):
                seq_data = np.array(seq_data)
            
            # 确保序列长度不超过最大长度
            if len(seq_data) > self.config.max_seq_len:
                seq_data = seq_data[-self.config.max_seq_len:]
            
            # 填充或截断
            if len(seq_data) < self.config.max_seq_len:
                pad_len = self.config.max_seq_len - len(seq_data)
                seq_data = np.pad(seq_data, ((pad_len, 0), (0, 0)), mode='constant')
            
            processed_seq_features[seq_type] = seq_data
        
        return non_seq_features, processed_seq_features
    
    def single_inference(self, 
                        user_features: Dict,
                        item_features: Dict,
                        context_features: Dict,
                        sequence_features: Dict) -> Dict[str, float]:
        """单样本推理"""
        
        start_time = time.time()
        
        try:
            # 预处理
            non_seq_features, seq_features = self.preprocess_input(
                user_features, item_features, context_features, sequence_features
            )
            
            # 添加批次维度
            for key in non_seq_features:
                non_seq_features[key] = np.expand_dims(non_seq_features[key], 0)
            
            for key in seq_features:
                seq_features[key] = np.expand_dims(seq_features[key], 0)
            
            # 推理
            predictions = self.model((non_seq_features, seq_features), training=False)
            
            # 提取结果
            results = {}
            for task, pred in predictions.items():
                results[task] = float(pred.numpy()[0, 0])
            
            # 更新统计
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            self._update_stats(success=True, latency=latency)
            
            return results
            
        except Exception as e:
            self._update_stats(success=False, latency=0)
            raise e
    
    def batch_inference(self, 
                       batch_data: List[Tuple[Dict, Dict, Dict, Dict]]) -> List[Dict[str, float]]:
        """批量推理"""
        
        start_time = time.time()
        
        try:
            # 预处理所有样本
            batch_non_seq = []
            batch_seq = []
            
            for user_feats, item_feats, context_feats, seq_feats in batch_data:
                non_seq, seq = self.preprocess_input(user_feats, item_feats, context_feats, seq_feats)
                batch_non_seq.append(non_seq)
                batch_seq.append(seq)
            
            # 合并批次
            batched_non_seq = {}
            batched_seq = {}
            
            # 合并非序列特征
            for key in batch_non_seq[0].keys():
                batched_non_seq[key] = np.stack([sample[key] for sample in batch_non_seq])
            
            # 合并序列特征
            for key in batch_seq[0].keys():
                batched_seq[key] = np.stack([sample[key] for sample in batch_seq])
            
            # 推理
            predictions = self.model((batched_non_seq, batched_seq), training=False)
            
            # 提取结果
            batch_results = []
            batch_size = len(batch_data)
            
            for i in range(batch_size):
                sample_results = {}
                for task, pred in predictions.items():
                    sample_results[task] = float(pred.numpy()[i, 0])
                batch_results.append(sample_results)
            
            # 更新统计
            total_latency = (time.time() - start_time) * 1000
            avg_latency = total_latency / len(batch_data)
            self._update_stats(success=True, latency=avg_latency, batch_size=len(batch_data))
            
            return batch_results
            
        except Exception as e:
            self._update_stats(success=False, latency=0)
            raise e
    
    def _update_stats(self, success: bool, latency: float, batch_size: int = 1):
        """更新推理统计"""
        self.inference_stats['total_requests'] += batch_size
        
        if success:
            self.inference_stats['successful_requests'] += batch_size
            
            # 更新平均延迟（指数移动平均）
            alpha = 0.1  # 平滑因子
            old_avg = self.inference_stats['avg_latency_ms']
            new_avg = alpha * latency + (1 - alpha) * old_avg
            self.inference_stats['avg_latency_ms'] = new_avg
        else:
            self.inference_stats['failed_requests'] += batch_size
    
    def get_stats(self) -> Dict:
        """获取推理统计"""
        stats = self.inference_stats.copy()
        
        if stats['successful_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests'] * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计"""
        self.inference_stats = {
            'total_requests': 0,
            'avg_latency_ms': 0.0,
            'successful_requests': 0,
            'failed_requests': 0
        }


def create_sample_inference_data(config: OneTransConfig) -> Tuple[Dict, Dict, Dict, Dict]:
    """创建示例推理数据"""
    
    # 用户特征
    user_features = {
        'user_id': np.array([123]),
        'age': np.array([25]),
        'gender': np.array([1])
    }
    
    # 物品特征
    item_features = {
        'item_id': np.array([456]),
        'category': np.array([3]),
        'price': np.array([99.9])
    }
    
    # 上下文特征
    context_features = {
        'time_of_day': np.array([14]),  # 下午2点
        'day_of_week': np.array([2]),   # 周二
        'device_type': np.array([1])    # 手机
    }
    
    # 序列特征
    sequence_features = {}
    for seq_type in config.feature_config['sequence_features']:
        # 创建随机序列
        seq_len = np.random.randint(1, config.max_seq_len + 1)
        seq_data = np.random.randn(seq_len, 64).astype(np.float32)
        sequence_features[seq_type] = seq_data
    
    return user_features, item_features, context_features, sequence_features


def single_inference_demo():
    """单样本推理演示"""
    print("=== 单样本推理演示 ===")
    
    # 创建推理引擎（使用示例模型）
    engine = OneTransInferenceEngine('./example_models/best_model')
    
    # 创建示例数据
    user_feats, item_feats, context_feats, seq_feats = create_sample_inference_data(engine.config)
    
    # 执行推理
    print("执行单样本推理...")
    results = engine.single_inference(user_feats, item_feats, context_feats, seq_feats)
    
    print("推理结果:")
    for task, score in results.items():
        print(f"  {task}: {score:.4f}")
    
    # 显示统计
    stats = engine.get_stats()
    print(f"\n推理统计:")
    print(f"  平均延迟: {stats['avg_latency_ms']:.2f} ms")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    
    return engine


def batch_inference_demo():
    """批量推理演示"""
    print("\n=== 批量推理演示 ===")
    
    # 创建推理引擎
    engine = OneTransInferenceEngine('./example_models/best_model')
    
    # 创建批量数据
    batch_size = 5
    batch_data = []
    
    for i in range(batch_size):
        user_feats, item_feats, context_feats, seq_feats = create_sample_inference_data(engine.config)
        batch_data.append((user_feats, item_feats, context_feats, seq_feats))
    
    # 执行批量推理
    print(f"执行批量推理，批次大小: {batch_size}")
    batch_results = engine.batch_inference(batch_data)
    
    print("批量推理结果:")
    for i, results in enumerate(batch_results):
        print(f"样本 {i+1}:")
        for task, score in results.items():
            print(f"    {task}: {score:.4f}")
    
    # 显示统计
    stats = engine.get_stats()
    print(f"\n批量推理统计:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  平均延迟: {stats['avg_latency_ms']:.2f} ms")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    
    return engine


def performance_benchmark_demo():
    """性能基准测试演示"""
    print("\n=== 性能基准测试演示 ===")
    
    # 创建推理引擎
    engine = OneTransInferenceEngine('./example_models/best_model')
    
    # 测试不同批次大小的性能
    batch_sizes = [1, 4, 8, 16, 32]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"测试批次大小: {batch_size}")
        
        # 创建测试数据
        test_data = []
        for _ in range(batch_size):
            user_feats, item_feats, context_feats, seq_feats = create_sample_inference_data(engine.config)
            test_data.append((user_feats, item_feats, context_feats, seq_feats))
        
        # 预热
        engine.batch_inference(test_data[:1])
        
        # 性能测试
        start_time = time.time()
        num_runs = 10 if batch_size <= 8 else 5  # 调整运行次数
        
        for _ in range(num_runs):
            engine.batch_inference(test_data)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / num_runs) * 1000 / batch_size  # 每个样本的平均延迟
        throughput = batch_size * num_runs / total_time  # 样本/秒
        
        results[batch_size] = {
            'avg_latency_ms': avg_latency,
            'throughput_samples_per_second': throughput
        }
        
        print(f"  平均延迟: {avg_latency:.2f} ms")
        print(f"  吞吐量: {throughput:.2f} 样本/秒")
    
    # 显示性能对比
    print("\n性能对比:")
    print("批次大小 | 平均延迟(ms) | 吞吐量(样本/秒)")
    print("-" * 50)
    for batch_size, perf in results.items():
        print(f"{batch_size:8d} | {perf['avg_latency_ms']:12.2f} | {perf['throughput_samples_per_second']:15.2f}")
    
    return results


def api_service_simulation():
    """API服务模拟演示"""
    print("\n=== API服务模拟演示 ===")
    
    # 创建推理引擎
    engine = OneTransInferenceEngine('./example_models/best_model')
    
    # 模拟API请求
    print("模拟API服务处理请求...")
    
    num_requests = 20
    request_interval = 0.1  # 100ms间隔
    
    for i in range(num_requests):
        # 创建请求数据
        user_feats, item_feats, context_feats, seq_feats = create_sample_inference_data(engine.config)
        
        try:
            # 处理请求
            results = engine.single_inference(user_feats, item_feats, context_feats, seq_feats)
            
            print(f"请求 {i+1}/{num_requests} - CTR: {results.get('ctr', 0):.4f}, CVR: {results.get('cvr', 0):.4f}")
            
        except Exception as e:
            print(f"请求 {i+1}/{num_requests} - 失败: {e}")
        
        # 模拟请求间隔
        time.sleep(request_interval)
    
    # 显示服务统计
    stats = engine.get_stats()
    print(f"\nAPI服务统计:")
    print(f"  处理请求数: {stats['total_requests']}")
    print(f"  成功请求数: {stats['successful_requests']}")
    print(f"  失败请求数: {stats['failed_requests']}")
    print(f"  成功率: {stats['success_rate']:.1f}%")
    print(f"  平均延迟: {stats['avg_latency_ms']:.2f} ms")
    
    # 计算QPS
    total_time = num_requests * request_interval
    qps = stats['successful_requests'] / total_time
    print(f"  估算QPS: {qps:.2f}")


def main():
    """主函数：运行所有推理示例"""
    print("OneTrans模型推理示例演示")
    print("=" * 50)
    
    try:
        # 1. 单样本推理演示
        engine1 = single_inference_demo()
        
        # 2. 批量推理演示
        engine2 = batch_inference_demo()
        
        # 3. 性能基准测试
        performance_results = performance_benchmark_demo()
        
        # 4. API服务模拟
        api_service_simulation()
        
        print("\n" + "=" * 50)
        print("所有推理示例执行成功!")
        print("\n部署建议:")
        print("1. 对于高QPS场景，建议使用批量推理")
        print("2. 根据实际硬件调整批次大小以获得最佳性能")
        print("3. 考虑使用TensorFlow Serving进行生产部署")
        print("4. 监控推理延迟和成功率指标")
        
    except Exception as e:
        print(f"推理示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()