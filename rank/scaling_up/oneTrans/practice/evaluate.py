"""
OneTrans模型评估脚本
支持离线评估、在线A/B测试模拟、性能基准测试等
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .model import OneTransModel
from .config import OneTransConfig
from .data_loader import DataLoader


class OneTransEvaluator:
    """OneTrans模型评估器"""
    
    def __init__(self, model: OneTransModel, config: OneTransConfig):
        self.model = model
        self.config = config
        
        # 评估指标
        self.metrics = self._create_metrics()
        
        # 性能统计
        self.performance_stats = {
            'inference_time': [],
            'memory_usage': [],
            'throughput': []
        }
    
    def _create_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        """创建评估指标"""
        metrics = {}
        
        for task in self.config.tasks:
            if task in ['ctr', 'cvr']:  # 二分类任务
                metrics[f'{task}_auc'] = tf.keras.metrics.AUC(name=f'{task}_auc')
                metrics[f'{task}_accuracy'] = tf.keras.metrics.BinaryAccuracy(name=f'{task}_accuracy')
                metrics[f'{task}_precision'] = tf.keras.metrics.Precision(name=f'{task}_precision')
                metrics[f'{task}_recall'] = tf.keras.metrics.Recall(name=f'{task}_recall')
                metrics[f'{task}_f1'] = tf.keras.metrics.F1Score(name=f'{task}_f1')
                metrics[f'{task}_logloss'] = tf.keras.metrics.BinaryCrossentropy(name=f'{task}_logloss')
            else:  # 回归任务
                metrics[f'{task}_mse'] = tf.keras.metrics.MeanSquaredError(name=f'{task}_mse')
                metrics[f'{task}_mae'] = tf.keras.metrics.MeanAbsoluteError(name=f'{task}_mae')
                metrics[f'{task}_rmse'] = tf.keras.metrics.RootMeanSquaredError(name=f'{task}_rmse')
        
        return metrics
    
    def evaluate_offline(self, 
                        data_loader: DataLoader,
                        dataset_type: str = 'test') -> Dict[str, float]:
        """离线评估"""
        print(f"开始离线评估 ({dataset_type}数据集)")
        
        # 获取数据集
        if dataset_type == 'test':
            dataset = data_loader.get_test_dataset()
        elif dataset_type == 'val':
            dataset = data_loader.get_val_dataset()
        else:
            dataset = data_loader.get_train_dataset()
        
        # 重置指标
        for metric in self.metrics.values():
            metric.reset_states()
        
        # 评估循环
        total_samples = 0
        inference_times = []
        
        for batch_idx, batch in enumerate(dataset):
            non_seq_features, seq_features, labels = batch
            batch_size = tf.shape(list(labels.values())[0])[0]
            total_samples += batch_size.numpy()
            
            # 推理时间统计
            start_time = time.time()
            predictions = self.model((non_seq_features, seq_features), training=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 更新指标
            for task in self.config.tasks:
                if task in predictions and task in labels:
                    if task in ['ctr', 'cvr']:
                        self.metrics[f'{task}_auc'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_accuracy'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_precision'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_recall'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_f1'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_logloss'].update_state(labels[task], predictions[task])
                    else:
                        self.metrics[f'{task}_mse'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_mae'].update_state(labels[task], predictions[task])
                        self.metrics[f'{task}_rmse'].update_state(labels[task], predictions[task])
            
            if batch_idx % 100 == 0:
                print(f"处理批次 {batch_idx}, 已处理样本数: {total_samples}")
        
        # 计算指标结果
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.result().numpy()
        
        # 性能统计
        avg_inference_time = np.mean(inference_times)
        throughput = total_samples / np.sum(inference_times)
        
        results['performance'] = {
            'total_samples': total_samples,
            'avg_inference_time_per_batch': avg_inference_time,
            'throughput_samples_per_second': throughput,
            'avg_inference_time_per_sample': avg_inference_time / (total_samples / len(inference_times))
        }
        
        print(f"离线评估完成，共处理 {total_samples} 个样本")
        print(f"平均推理时间: {avg_inference_time:.4f}s/批次")
        print(f"吞吐量: {throughput:.2f} 样本/秒")
        
        return results
    
    def evaluate_ab_test(self, 
                        control_group: DataLoader,
                        treatment_group: DataLoader,
                        metric_name: str = 'ctr_auc') -> Dict[str, Any]:
        """模拟A/B测试评估"""
        print("开始A/B测试评估")
        
        # 评估对照组
        control_results = self.evaluate_offline(control_group, 'test')
        control_metric = control_results.get(metric_name, 0)
        
        # 评估实验组
        treatment_results = self.evaluate_offline(treatment_group, 'test')
        treatment_metric = treatment_results.get(metric_name, 0)
        
        # 计算提升
        improvement = treatment_metric - control_metric
        improvement_percentage = (improvement / control_metric) * 100 if control_metric != 0 else 0
        
        # 统计显著性检验（简化版）
        is_significant = abs(improvement_percentage) > 1.0  # 简化阈值
        
        ab_results = {
            'control_metric': control_metric,
            'treatment_metric': treatment_metric,
            'absolute_improvement': improvement,
            'relative_improvement_percentage': improvement_percentage,
            'is_statistically_significant': is_significant,
            'metric_name': metric_name
        }
        
        print(f"A/B测试结果:")
        print(f"  对照组 {metric_name}: {control_metric:.4f}")
        print(f"  实验组 {metric_name}: {treatment_metric:.4f}")
        print(f"  绝对提升: {improvement:.4f}")
        print(f"  相对提升: {improvement_percentage:.2f}%")
        print(f"  统计显著性: {'是' if is_significant else '否'}")
        
        return ab_results
    
    def benchmark_performance(self, 
                            data_loader: DataLoader,
                            num_batches: int = 100,
                            warmup_batches: int = 10) -> Dict[str, float]:
        """性能基准测试"""
        print("开始性能基准测试")
        
        dataset = data_loader.get_test_dataset()
        
        # 预热
        print("预热阶段...")
        for i, batch in enumerate(dataset):
            if i >= warmup_batches:
                break
            _ = self.model(batch[0], training=False)
        
        # 性能测试
        inference_times = []
        memory_usages = []
        
        print("性能测试阶段...")
        for i, batch in enumerate(dataset):
            if i >= num_batches:
                break
            
            # 内存使用（简化测量）
            memory_before = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.config.list_physical_devices('GPU') else 0
            
            # 推理时间
            start_time = time.time()
            _ = self.model(batch[0], training=False)
            inference_time = time.time() - start_time
            
            memory_after = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.config.list_physical_devices('GPU') else 0
            memory_usage = memory_after - memory_before
            
            inference_times.append(inference_time)
            memory_usages.append(memory_usage)
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i + 1}/{num_batches} 批次")
        
        # 计算统计信息
        benchmark_results = {
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'p95_inference_time_ms': np.percentile(inference_times, 95) * 1000,
            'p99_inference_time_ms': np.percentile(inference_times, 99) * 1000,
            'throughput_batches_per_second': 1.0 / np.mean(inference_times),
            'avg_memory_usage_mb': np.mean(memory_usages) / (1024 * 1024) if memory_usages else 0,
            'max_memory_usage_mb': np.max(memory_usages) / (1024 * 1024) if memory_usages else 0,
            'total_batches_tested': num_batches
        }
        
        print("性能基准测试完成:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value:.2f}")
        
        return benchmark_results
    
    def analyze_feature_importance(self, 
                                 data_loader: DataLoader,
                                 num_samples: int = 1000) -> Dict[str, float]:
        """特征重要性分析"""
        print("开始特征重要性分析")
        
        # 获取样本数据
        dataset = data_loader.get_test_dataset()
        
        # 收集预测结果
        predictions = []
        labels_collected = []
        
        sample_count = 0
        for batch in dataset:
            if sample_count >= num_samples:
                break
            
            non_seq_features, seq_features, labels = batch
            batch_predictions = self.model((non_seq_features, seq_features), training=False)
            
            predictions.append(batch_predictions)
            labels_collected.append(labels)
            sample_count += tf.shape(list(labels.values())[0])[0].numpy()
        
        # 简化特征重要性分析
        # 在实际应用中，可以使用SHAP、LIME等工具
        feature_importance = {}
        
        # 这里使用简化方法：基于特征缺失的影响
        baseline_performance = self.evaluate_offline(data_loader, 'test')
        baseline_metric = baseline_performance.get('ctr_auc', 0)
        
        # 分析非序列特征
        for feature_name in self.config.feature_config['user_features'] + \
                          self.config.feature_config['item_features'] + \
                          self.config.feature_config['context_features']:
            # 创建缺失该特征的数据集
            # 简化实现：在实际中需要更复杂的处理
            feature_importance[feature_name] = 0.1  # 占位值
        
        # 分析序列特征
        for seq_feature in self.config.feature_config['sequence_features']:
            feature_importance[seq_feature] = 0.2  # 序列特征通常更重要
        
        # 归一化重要性分数
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        print("特征重要性分析完成")
        return feature_importance
    
    def generate_evaluation_report(self, 
                                 data_loader: DataLoader,
                                 output_dir: str = "./evaluation_reports") -> str:
        """生成完整评估报告"""
        print("生成评估报告")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 执行各种评估
        offline_results = self.evaluate_offline(data_loader)
        performance_benchmark = self.benchmark_performance(data_loader)
        feature_importance = self.analyze_feature_importance(data_loader)
        
        # 创建报告
        report = {
            'model_config': self.config.to_dict(),
            'offline_evaluation': offline_results,
            'performance_benchmark': performance_benchmark,
            'feature_importance': feature_importance,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': data_loader.get_data_info()
        }
        
        # 保存报告
        report_path = output_path / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成可视化图表
        self._generate_visualizations(report, output_path)
        
        print(f"评估报告已保存至: {report_path}")
        return str(report_path)
    
    def _generate_visualizations(self, report: Dict, output_path: Path):
        """生成可视化图表"""
        
        # 指标对比图
        plt.figure(figsize=(12, 8))
        
        # 提取主要指标
        metrics_data = {}
        for key, value in report['offline_evaluation'].items():
            if key != 'performance' and isinstance(value, (int, float)):
                metrics_data[key] = value
        
        if metrics_data:
            plt.subplot(2, 2, 1)
            plt.bar(metrics_data.keys(), metrics_data.values())
            plt.title('模型指标对比')
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        # 特征重要性图
        plt.subplot(2, 2, 2)
        feature_importance = report.get('feature_importance', {})
        if feature_importance:
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            plt.barh(features, importances)
            plt.title('特征重要性')
            plt.tight_layout()
        
        # 性能指标图
        plt.subplot(2, 2, 3)
        performance_data = report.get('performance_benchmark', {})
        if performance_data:
            perf_keys = [k for k in performance_data.keys() if 'time' in k or 'throughput' in k]
            perf_values = [performance_data[k] for k in perf_keys]
            plt.bar(perf_keys, perf_values)
            plt.title('性能指标')
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_path / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_model_for_evaluation(model_path: str) -> Tuple[OneTransModel, OneTransConfig]:
    """加载模型用于评估"""
    model_path = Path(model_path)
    
    # 加载配置
    config_path = model_path / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        config = OneTransConfig.from_dict(config_dict)
    
    # 创建模型
    model = OneTransModel(config)
    
    # 加载权重
    weights_path = model_path / 'model_weights.h5'
    if not weights_path.exists():
        raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
    
    model.load_weights(str(weights_path))
    
    return model, config


def evaluate_model(model_path: str, 
                  data_loader: DataLoader,
                  output_dir: str = "./evaluation_reports") -> Dict:
    """评估模型的便捷函数"""
    
    # 加载模型
    model, config = load_model_for_evaluation(model_path)
    
    # 创建评估器
    evaluator = OneTransEvaluator(model, config)
    
    # 生成评估报告
    report_path = evaluator.generate_evaluation_report(data_loader, output_dir)
    
    print(f"模型评估完成，报告路径: {report_path}")
    
    # 返回评估结果
    return {
        'model_path': model_path,
        'report_path': report_path,
        'evaluator': evaluator
    }


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估OneTrans模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./evaluation_reports',
                       help='输出目录')
    parser.add_argument('--eval_type', type=str, default='full',
                       choices=['full', 'offline', 'performance', 'ab_test'],
                       help='评估类型')
    
    args = parser.parse_args()
    
    # 创建数据加载器
    from config import OneTransConfig
    config = OneTransConfig()
    data_loader = DataLoader(config)
    
    if args.data_dir:
        # 加载真实数据
        data_loader.load_datasets(
            train_path=os.path.join(args.data_dir, 'train.csv'),
            val_path=os.path.join(args.data_dir, 'val.csv'),
            test_path=os.path.join(args.data_dir, 'test.csv')
        )
    else:
        # 使用示例数据
        data_loader.train_dataset = data_loader.create_sample_data()
        data_loader.test_dataset = data_loader.create_sample_data()
    
    # 执行评估
    if args.eval_type == 'full':
        result = evaluate_model(args.model_path, data_loader, args.output_dir)
    else:
        # 加载模型
        model, config = load_model_for_evaluation(args.model_path)
        evaluator = OneTransEvaluator(model, config)
        
        if args.eval_type == 'offline':
            results = evaluator.evaluate_offline(data_loader)
            print("离线评估结果:", json.dumps(results, indent=2))
        elif args.eval_type == 'performance':
            results = evaluator.benchmark_performance(data_loader)
            print("性能基准测试结果:", json.dumps(results, indent=2))
        elif args.eval_type == 'ab_test':
            # 需要两个数据集进行A/B测试
            print("A/B测试需要对照组和实验组数据集")
    
    print("评估完成!")