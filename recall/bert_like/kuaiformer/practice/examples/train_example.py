"""
KuaiFormer训练示例
演示如何使用KuaiFormer进行模型训练
"""

import os
import sys
sys.path.append('..')

from config import KuaiFormerConfig
from data_loader import create_synthetic_data
from train import KuaiFormerTrainer, TrainingMonitor
import tensorflow as tf

def train_example():
    """训练示例"""
    
    # 1. 配置模型
    config = KuaiFormerConfig()
    config.batch_size = 128
    config.num_epochs = 10
    config.learning_rate = 1e-4
    
    print("=== KuaiFormer训练示例 ===")
    print(f"配置参数:")
    print(f"  嵌入维度: {config.embedding_dim}")
    print(f"  Transformer层数: {config.num_layers}")
    print(f"  注意力头数: {config.num_heads}")
    print(f"  查询token数: {config.num_query_tokens}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print()
    
    # 2. 创建合成数据
    print("创建合成数据...")
    data_loader, video_features = create_synthetic_data(
        config, num_users=1000, num_videos=10000
    )
    
    # 3. 创建训练数据集
    train_dataset = data_loader.create_training_pairs()
    
    # 分割训练/验证集
    dataset_size = sum(1 for _ in train_dataset)
    train_size = int(0.8 * dataset_size)
    
    train_data = train_dataset.take(train_size)
    val_data = train_dataset.skip(train_size)
    
    print(f"数据集信息:")
    print(f"  总样本数: {dataset_size}")
    print(f"  训练集: {train_size}")
    print(f"  验证集: {dataset_size - train_size}")
    print()
    
    # 4. 创建训练监控器
    monitor = TrainingMonitor('logs/train_example')
    
    # 5. 创建训练器并开始训练
    print("开始训练...")
    trainer = KuaiFormerTrainer(config)
    
    try:
        trainer.train(train_data, val_data, config.num_epochs, monitor)
        print("训练完成!")
    except KeyboardInterrupt:
        print("训练被中断")
    
    # 6. 保存模型
    model_save_path = 'trained_models/kuaiformer_example'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trainer.model.save(model_save_path)
    print(f"模型已保存到: {model_save_path}")

def quick_start():
    """快速开始示例"""
    
    print("=== KuaiFormer快速开始 ===")
    
    # 最小配置
    config = KuaiFormerConfig()
    config.batch_size = 64
    config.num_epochs = 5
    config.max_sequence_length = 128
    config.early_group_size = 32
    config.mid_group_size = 16
    
    # 快速数据生成
    from data_loader import create_synthetic_data
    data_loader, _ = create_synthetic_data(config, num_users=100, num_videos=1000)
    
    # 训练
    train_dataset = data_loader.create_training_pairs()
    monitor = TrainingMonitor('logs/quick_start')
    trainer = KuaiFormerTrainer(config)
    
    print("快速训练中...")
    trainer.train(train_dataset, train_dataset.take(5), 2, monitor)
    print("快速训练完成!")

if __name__ == "__main__":
    # 设置GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU配置完成")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    
    # 运行示例
    train_example()
    
    # 可选：运行快速开始
    # quick_start()