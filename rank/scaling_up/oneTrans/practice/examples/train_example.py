"""
OneTrans模型训练示例
演示如何使用OneTrans模型进行训练和评估
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OneTransModel
from config import OneTransConfig, get_model_config
from data_loader import DataLoader, create_sample_batch
from train import OneTransTrainer, train_one_trans_model
from evaluate import OneTransEvaluator, evaluate_model


def basic_training_example():
    """基础训练示例"""
    print("=== OneTrans基础训练示例 ===")
    
    # 创建配置
    config = OneTransConfig()
    config.hidden_dim = 128
    config.num_layers = 4
    config.num_heads = 4
    config.max_seq_len = 50
    config.batch_size = 32
    
    print(f"模型配置: {config}")
    
    # 创建模型
    model = OneTransModel(config)
    
    # 创建示例数据批次
    non_seq_features, seq_features, labels = create_sample_batch(batch_size=2, config=config)
    
    # 测试前向传播
    predictions = model((non_seq_features, seq_features), training=True)
    
    print("模型前向传播测试成功!")
    print(f"预测输出形状: { {k: v.shape for k, v in predictions.items()} }")
    
    return model, config


def full_training_pipeline():
    """完整训练流程示例"""
    print("\n=== OneTrans完整训练流程示例 ===")
    
    # 使用预定义配置
    config = get_model_config('small')
    
    # 创建数据加载器
    data_loader = DataLoader(config)
    
    # 创建训练器
    trainer = OneTransTrainer(config, model_dir='./example_models')
    
    # 训练模型（简化版，只训练少量epoch）
    print("开始简化训练...")
    history = trainer.train(
        train_loader=data_loader,
        val_loader=data_loader,
        epochs=2,
        save_freq=1,
        early_stopping_patience=3
    )
    
    print("训练完成!")
    print(f"训练历史: {history}")
    
    return trainer


def evaluation_example(trainer):
    """评估示例"""
    print("\n=== OneTrans评估示例 ===")
    
    # 创建数据加载器
    config = trainer.config
    data_loader = DataLoader(config)
    
    # 创建评估器
    evaluator = OneTransEvaluator(trainer.model, config)
    
    # 离线评估
    print("执行离线评估...")
    offline_results = evaluator.evaluate_offline(data_loader)
    
    print("离线评估结果:")
    for metric_name, value in offline_results.items():
        if metric_name != 'performance':
            print(f"  {metric_name}: {value:.4f}")
    
    # 性能基准测试
    print("\n执行性能基准测试...")
    performance_results = evaluator.benchmark_performance(data_loader, num_batches=10)
    
    print("性能基准测试结果:")
    for metric_name, value in performance_results.items():
        print(f"  {metric_name}: {value:.2f}")
    
    return evaluator


def model_inference_example(trainer):
    """模型推理示例"""
    print("\n=== OneTrans模型推理示例 ===")
    
    config = trainer.config
    
    # 创建示例输入
    non_seq_features, seq_features, _ = create_sample_batch(batch_size=1, config=config)
    
    # 推理
    predictions = trainer.model((non_seq_features, seq_features), training=False)
    
    print("推理结果:")
    for task, pred in predictions.items():
        print(f"  {task}: {pred.numpy()[0][0]:.4f}")
    
    # 批量推理
    print("\n批量推理示例:")
    non_seq_features_batch, seq_features_batch, _ = create_sample_batch(batch_size=5, config=config)
    batch_predictions = trainer.model((non_seq_features_batch, seq_features_batch), training=False)
    
    print("批量推理结果形状:")
    for task, pred in batch_predictions.items():
        print(f"  {task}: {pred.shape}")


def advanced_features_demo():
    """高级功能演示"""
    print("\n=== OneTrans高级功能演示 ===")
    
    # 演示混合参数化
    config = OneTransConfig()
    config.mixed_param_config = {
        'S_token_ratio': 0.3,
        'NS_token_ratio': 0.7,
        'param_sharing_strategy': 'hierarchical'
    }
    
    # 演示金字塔堆叠
    config.pyramid_config = {
        'enabled': True,
        'layer_retention_ratios': [0.5, 0.3, 0.1, 0.01],
        'query_reduction_strategy': 'exponential'
    }
    
    # 演示系统优化
    config.system_config = {
        'mixed_precision': True,
        'kv_cache_enabled': True,
        'flash_attention': True,
        'activation_recomputation': True
    }
    
    model = OneTransModel(config)
    
    print("高级功能配置:")
    print(f"混合参数化: {config.mixed_param_config}")
    print(f"金字塔堆叠: {config.pyramid_config}")
    print(f"系统优化: {config.system_config}")
    
    # 测试高级功能
    non_seq_features, seq_features, _ = create_sample_batch(batch_size=2, config=config)
    predictions = model((non_seq_features, seq_features), training=True)
    
    print("高级功能模型测试成功!")
    
    return model


def main():
    """主函数：运行所有示例"""
    print("OneTrans模型示例演示")
    print("=" * 50)
    
    try:
        # 1. 基础训练示例
        model, config = basic_training_example()
        
        # 2. 完整训练流程
        trainer = full_training_pipeline()
        
        # 3. 评估示例
        evaluator = evaluation_example(trainer)
        
        # 4. 推理示例
        model_inference_example(trainer)
        
        # 5. 高级功能演示
        advanced_model = advanced_features_demo()
        
        print("\n" + "=" * 50)
        print("所有示例执行成功!")
        print("\n下一步:")
        print("1. 查看生成的模型文件: ./example_models/")
        print("2. 修改配置参数进行自定义训练")
        print("3. 使用真实数据替换示例数据进行训练")
        print("4. 参考README.md了解更多使用细节")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()