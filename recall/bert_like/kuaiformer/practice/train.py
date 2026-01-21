"""
KuaiFormer训练脚本
支持分布式训练、模型保存和评估
"""

import tensorflow as tf
import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

from config import KuaiFormerConfig
from model import KuaiFormer, KuaiFormerLoss
from data_loader import KuaiFormerDataLoader, create_synthetic_data

class TrainingMonitor:
    """训练监控器，记录指标和可视化"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.train_loss = []
        self.val_metrics = []
        
        # 创建TensorBoard写入器
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'train')
        )
        self.val_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'val')
        )
    
    def record_train_loss(self, loss: float, step: int):
        """记录训练损失"""
        self.train_loss.append((step, loss))
        
        with self.train_writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
    
    def record_val_metrics(self, metrics: Dict[str, float], step: int):
        """记录验证指标"""
        self.val_metrics.append((step, metrics))
        
        with self.val_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

class LearningRateScheduler:
    """学习率调度器，支持warmup和余弦衰减"""
    
    def __init__(self, config: KuaiFormerConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.warmup_steps = config.warmup_steps
        
    def __call__(self, step: int) -> float:
        """计算当前学习率"""
        if step < self.warmup_steps:
            # Warmup阶段：线性增加
            return self.config.learning_rate * (step / self.warmup_steps)
        else:
            # 余弦衰减
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.config.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))

class KuaiFormerTrainer:
    """KuaiFormer训练器"""
    
    def __init__(self, config: KuaiFormerConfig):
        self.config = config
        self.model = KuaiFormer(config)
        self.loss_fn = KuaiFormerLoss(config)
        self.optimizer = self._create_optimizer()
        
        # 训练状态
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        
        # 检查点管理
        self.checkpoint_dir = os.path.join('checkpoints', 'kuaiformer')
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=5
        )
    
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """创建优化器"""
        return tf.keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            beta_1=self.config.adam_beta1,
            beta_2=self.config.adam_beta2,
            epsilon=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
    
    @tf.function
    def train_step(self, batch_data: Dict[str, tf.Tensor]) -> tf.Tensor:
        """单步训练"""
        with tf.GradientTape() as tape:
            # 获取正负样本
            positive_video = batch_data['positive_video']
            negative_videos = batch_data['negative_videos']
            popularity = batch_data['popularity']
            
            # 获取候选视频嵌入
            candidate_videos = tf.concat([
                tf.expand_dims(positive_video, 1),  # [batch_size, 1]
                negative_videos  # [batch_size, num_negatives]
            ], axis=1)  # [batch_size, 1+num_negatives]
            
            candidate_embeddings = self.model.get_candidate_embeddings(candidate_videos)
            popularity = tf.cast(popularity, tf.float32)
            
            # 前向传播 - 统一调用接口
            interest_representations = self.model(batch_data['history_features'], training=True)
            
            # 处理因果掩码模式的多序列输出
            if self.model.use_causal_mask:
                # 重塑为 [batch_size * (seq_len-1), num_interests, feature_dim]
                batch_size = tf.shape(interest_representations)[0]
                seq_len_minus_1 = tf.shape(interest_representations)[1]
                interest_representations = tf.reshape(
                    interest_representations, 
                    [batch_size * seq_len_minus_1, self.model.config.num_interests, -1]
                )
                
                # 扩展候选嵌入以匹配所有位置
                candidate_embeddings = tf.tile(
                    tf.expand_dims(candidate_embeddings, 1),
                    [1, seq_len_minus_1, 1, 1]
                )
                candidate_embeddings = tf.reshape(
                    candidate_embeddings,
                    [batch_size * seq_len_minus_1, tf.shape(candidate_embeddings)[2], -1]
                )
                
                # 扩展流行度
                popularity = tf.tile(
                    tf.expand_dims(popularity, 1),
                    [1, seq_len_minus_1, 1]
                )
                popularity = tf.reshape(
                    popularity,
                    [batch_size * seq_len_minus_1, tf.shape(popularity)[2]]
                )
            
            # 计算分数
            scores = self.model.compute_scores(interest_representations, candidate_embeddings)
            
            # 计算损失
            loss = self.loss_fn(scores, popularity)
        
        # 反向传播
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, 
              epochs: int, monitor: TrainingMonitor):
        """训练循环"""
        
        # 恢复检查点
        self._restore_checkpoint()
        
        # 学习率调度器
        total_steps = epochs * len(train_dataset)
        lr_scheduler = LearningRateScheduler(self.config, total_steps)
        
        print(f"开始训练，总步数: {total_steps}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练阶段
            train_losses = []
            start_time = time.time()
            
            for step, batch in enumerate(train_dataset):
                # 更新学习率
                current_lr = lr_scheduler(int(self.global_step))
                self.optimizer.learning_rate.assign(current_lr)
                
                # 训练步骤
                loss = self.train_step(batch)
                train_losses.append(loss.numpy())
                
                # 更新全局步数
                self.global_step.assign_add(1)
                
                # 记录指标
                if step % 100 == 0:
                    monitor.record_train_loss(loss.numpy(), int(self.global_step))
                    print(f"Step {step}: loss = {loss.numpy():.4f}, lr = {current_lr:.6f}")
            
            # 验证阶段
            val_metrics = self.evaluate(val_dataset)
            monitor.record_val_metrics(val_metrics, int(self.global_step))
            
            # 保存检查点
            self._save_checkpoint()
            
            # 输出epoch统计
            epoch_time = time.time() - start_time
            avg_loss = np.mean(train_losses)
            
            print(f"Epoch {epoch + 1} 完成:")
            print(f"  平均训练损失: {avg_loss:.4f}")
            print(f"  验证指标: {val_metrics}")
            print(f"  耗时: {epoch_time:.2f}秒")
    
    def evaluate(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """评估模型"""
        hits_at_k = {10: 0, 50: 0, 100: 0}
        total_samples = 0
        
        for batch in val_dataset:
            # 前向传播
            interest_representations = self.model(batch['history_features'], training=False)
            
            # 获取所有候选物品
            all_candidates = tf.concat([
                tf.expand_dims(batch['positive_video'], 1),
                batch['negative_videos']
            ], axis=1)
            
            candidate_embeddings = self.model.embedding_module.video_id_embedding(all_candidates)
            
            # 计算分数
            scores = self.model.compute_scores(interest_representations, candidate_embeddings)
            
            # 计算Hit Rate@K
            batch_size = tf.shape(scores)[0]
            total_samples += batch_size
            
            for k in hits_at_k.keys():
                # 获取top-k索引
                _, top_k_indices = tf.math.top_k(scores, k=k)
                
                # 检查正例是否在top-k中（索引0是正例）
                hits = tf.reduce_sum(tf.cast(tf.equal(top_k_indices, 0), tf.float32))
                hits_at_k[k] += hits.numpy()
        
        # 计算指标
        metrics = {}
        for k, hits in hits_at_k.items():
            metrics[f'HR@{k}'] = hits / total_samples if total_samples > 0 else 0.0
        
        return metrics
    
    def _save_checkpoint(self):
        """保存检查点"""
        save_path = self.checkpoint_manager.save()
        print(f"检查点已保存: {save_path}")
    
    def _restore_checkpoint(self):
        """恢复检查点"""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"从检查点恢复: {self.checkpoint_manager.latest_checkpoint}")
            print(f"当前步数: {self.global_step.numpy()}")

def main(use_causal_mask: bool = False):
    """主训练函数
    
    Args:
        use_causal_mask: 是否使用因果注意力掩码
            - False: 有明确标签的模式（输入历史序列，预测下一个item）
            - True: 只有行为序列的模式（预测序列中每个位置后面的候选item）
    """
    
    # 配置
    config = KuaiFormerConfig(use_causal_mask=use_causal_mask)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('logs', f'kuaiformer_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # 创建监控器
    monitor = TrainingMonitor(log_dir)
    
    # 创建数据
    print("创建合成数据...")
    data_loader, video_features = create_synthetic_data(config)
    
    # 创建训练和验证数据集
    full_dataset = data_loader.create_training_pairs()
    dataset_size = sum(1 for _ in full_dataset)
    train_size = int(0.8 * dataset_size)
    
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    print(f"数据集大小: {dataset_size}")
    print(f"训练集: {train_size}, 验证集: {dataset_size - train_size}")
    
    # 创建训练器
    trainer = KuaiFormerTrainer(config)
    
    # 开始训练
    print("开始训练KuaiFormer模型...")
    trainer.train(train_dataset, val_dataset, config.num_epochs, monitor)
    
    # 保存最终模型
    model_save_path = os.path.join(log_dir, 'final_model')
    trainer.model.save(model_save_path)
    print(f"最终模型已保存: {model_save_path}")

if __name__ == "__main__":
    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # 默认使用双向注意力（有明确标签的模式）
    main(use_causal_mask=False)
    
    # 如果需要使用因果注意力（只有行为序列的模式），取消注释下面这行
    # main(use_causal_mask=True)