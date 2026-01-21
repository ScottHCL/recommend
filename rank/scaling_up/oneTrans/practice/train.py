"""
OneTrans模型训练脚本
支持多任务训练、混合精度训练、梯度累积、模型检查点等
"""

import os
import time
import json
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .model import OneTransModel
from .config import OneTransConfig, get_model_config
from .data_loader import DataLoader


class OneTransTrainer:
    """OneTrans模型训练器"""
    
    def __init__(self, config: OneTransConfig, model_dir: str = "./models"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模型
        self.model = OneTransModel(config)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 损失函数
        self.loss_functions = self._create_loss_functions()
        
        # 指标跟踪
        self.train_metrics = self._create_metrics()
        self.val_metrics = self._create_metrics()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }
        
        # 混合精度训练
        self.use_mixed_precision = config.system_config['mixed_precision']
        if self.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """创建优化器"""
        optimizer_config = self.config.optimizer_config
        
        if optimizer_config['dense_optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=optimizer_config['learning_rate'],
                beta_1=optimizer_config.get('beta1', 0.9),
                beta_2=optimizer_config.get('beta2', 0.999),
                epsilon=optimizer_config.get('epsilon', 1e-8)
            )
        elif optimizer_config['dense_optimizer'] == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=optimizer_config['learning_rate'],
                rho=optimizer_config.get('rho', 0.9),
                momentum=optimizer_config.get('momentum', 0.0),
                epsilon=optimizer_config.get('epsilon', 1e-7)
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=optimizer_config['learning_rate']
            )
        
        return optimizer
    
    def _create_loss_functions(self) -> Dict[str, tf.keras.losses.Loss]:
        """创建损失函数"""
        loss_functions = {}
        
        for task in self.config.tasks:
            if task in ['ctr', 'cvr']:  # 二分类任务
                loss_functions[task] = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                )
            else:  # 回归任务
                loss_functions[task] = tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                )
        
        return loss_functions
    
    def _create_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        """创建评估指标"""
        metrics = {}
        
        for task in self.config.tasks:
            if task in ['ctr', 'cvr']:
                metrics[f'{task}_auc'] = tf.keras.metrics.AUC(name=f'{task}_auc')
                metrics[f'{task}_accuracy'] = tf.keras.metrics.BinaryAccuracy(name=f'{task}_accuracy')
                metrics[f'{task}_precision'] = tf.keras.metrics.Precision(name=f'{task}_precision')
                metrics[f'{task}_recall'] = tf.keras.metrics.Recall(name=f'{task}_recall')
            else:
                metrics[f'{task}_mse'] = tf.keras.metrics.MeanSquaredError(name=f'{task}_mse')
                metrics[f'{task}_mae'] = tf.keras.metrics.MeanAbsoluteError(name=f'{task}_mae')
        
        return metrics
    
    @tf.function
    def train_step(self, batch_data: Tuple[Dict, Dict, Dict]) -> Dict[str, tf.Tensor]:
        """单步训练"""
        non_seq_features, seq_features, labels = batch_data
        
        with tf.GradientTape() as tape:
            # 前向传播
            predictions = self.model((non_seq_features, seq_features), training=True)
            
            # 计算损失
            total_loss = 0.0
            task_losses = {}
            
            for task in self.config.tasks:
                if task in predictions and task in labels:
                    task_loss = self.loss_functions[task](labels[task], predictions[task])
                    task_losses[task] = task_loss
                    total_loss += task_loss
        
        # 反向传播
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # 梯度裁剪
        if self.config.gradient_clip > 0:
            gradients = [tf.clip_by_norm(g, self.config.gradient_clip) for g in gradients]
        
        # 更新参数
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # 更新指标
        for task in self.config.tasks:
            if task in predictions and task in labels:
                if task in ['ctr', 'cvr']:
                    self.train_metrics[f'{task}_auc'].update_state(labels[task], predictions[task])
                    self.train_metrics[f'{task}_accuracy'].update_state(labels[task], predictions[task])
                    self.train_metrics[f'{task}_precision'].update_state(labels[task], predictions[task])
                    self.train_metrics[f'{task}_recall'].update_state(labels[task], predictions[task])
                else:
                    self.train_metrics[f'{task}_mse'].update_state(labels[task], predictions[task])
                    self.train_metrics[f'{task}_mae'].update_state(labels[task], predictions[task])
        
        return {
            'total_loss': total_loss,
            'task_losses': task_losses
        }
    
    @tf.function
    def val_step(self, batch_data: Tuple[Dict, Dict, Dict]) -> Dict[str, tf.Tensor]:
        """验证步骤"""
        non_seq_features, seq_features, labels = batch_data
        
        # 前向传播
        predictions = self.model((non_seq_features, seq_features), training=False)
        
        # 计算损失
        total_loss = 0.0
        task_losses = {}
        
        for task in self.config.tasks:
            if task in predictions and task in labels:
                task_loss = self.loss_functions[task](labels[task], predictions[task])
                task_losses[task] = task_loss
                total_loss += task_loss
        
        # 更新验证指标
        for task in self.config.tasks:
            if task in predictions and task in labels:
                if task in ['ctr', 'cvr']:
                    self.val_metrics[f'{task}_auc'].update_state(labels[task], predictions[task])
                    self.val_metrics[f'{task}_accuracy'].update_state(labels[task], predictions[task])
                    self.val_metrics[f'{task}_precision'].update_state(labels[task], predictions[task])
                    self.val_metrics[f'{task}_recall'].update_state(labels[task], predictions[task])
                else:
                    self.val_metrics[f'{task}_mse'].update_state(labels[task], predictions[task])
                    self.val_metrics[f'{task}_mae'].update_state(labels[task], predictions[task])
        
        return {
            'total_loss': total_loss,
            'task_losses': task_losses
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 10,
              save_freq: int = 1,
              early_stopping_patience: int = 5) -> Dict:
        """训练模型"""
        
        print(f"开始训练OneTrans模型，共{epochs}个epoch")
        print(f"模型配置: {self.config}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_dataset = train_loader.get_train_dataset()
        val_dataset = val_loader.get_val_dataset() if val_loader else None
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练阶段
            train_losses = []
            for step, batch in enumerate(train_dataset):
                loss_info = self.train_step(batch)
                train_losses.append(loss_info['total_loss'].numpy())
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss_info['total_loss'].numpy():.4f}")
            
            avg_train_loss = np.mean(train_losses)
            
            # 验证阶段
            if val_dataset:
                val_losses = []
                for batch in val_dataset:
                    loss_info = self.val_step(batch)
                    val_losses.append(loss_info['total_loss'].numpy())
                
                avg_val_loss = np.mean(val_losses)
            else:
                avg_val_loss = avg_train_loss
            
            # 记录历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            # 计算指标
            train_metrics = {name: metric.result().numpy() for name, metric in self.train_metrics.items()}
            val_metrics = {name: metric.result().numpy() for name, metric in self.val_metrics.items()}
            
            self.history['train_metrics'][epoch] = train_metrics
            self.history['val_metrics'][epoch] = val_metrics
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model('best_model')
                print(f"新的最佳模型已保存，验证损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 定期保存
            if (epoch + 1) % save_freq == 0:
                self.save_model(f'model_epoch_{epoch+1}')
            
            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                break
            
            # 重置指标
            for metric in self.train_metrics.values():
                metric.reset_states()
            for metric in self.val_metrics.values():
                metric.reset_states()
        
        # 保存最终模型
        self.save_model('final_model')
        
        print("训练完成!")
        return self.history
    
    def save_model(self, model_name: str):
        """保存模型"""
        model_path = self.model_dir / model_name
        
        # 保存模型权重
        self.model.save_weights(str(model_path / 'model_weights.h5'))
        
        # 保存配置
        config_path = model_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # 保存训练历史
        history_path = model_path / 'training_history.json'
        with open(history_path, 'w') as f:
            # 转换numpy类型为Python原生类型
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list):
                    history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
                elif isinstance(value, dict):
                    history_serializable[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            history_serializable[key][sub_key] = {}
                            for metric_name, metric_value in sub_value.items():
                                history_serializable[key][sub_key][metric_name] = float(metric_value) if isinstance(metric_value, (np.floating, float)) else metric_value
                        else:
                            history_serializable[key][sub_key] = float(sub_value) if isinstance(sub_value, (np.floating, float)) else sub_value
                else:
                    history_serializable[key] = float(value) if isinstance(value, (np.floating, float)) else value
            
            json.dump(history_serializable, f, indent=2)
    
    def load_model(self, model_path: str):
        """加载模型"""
        model_path = Path(model_path)
        
        # 加载配置
        config_path = model_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = OneTransConfig.from_dict(config_dict)
        
        # 重新创建模型
        self.model = OneTransModel(self.config)
        
        # 加载权重
        weights_path = model_path / 'model_weights.h5'
        if weights_path.exists():
            self.model.load_weights(str(weights_path))
        
        # 加载历史
        history_path = model_path / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)


def train_one_trans_model(config_name: str = 'small', 
                         data_paths: Dict[str, str] = None,
                         epochs: int = 10,
                         model_dir: str = "./models") -> OneTransTrainer:
    """训练OneTrans模型的便捷函数"""
    
    # 获取配置
    config = get_model_config(config_name)
    
    # 创建数据加载器
    data_loader = DataLoader(config)
    
    if data_paths:
        data_loader.load_datasets(
            train_path=data_paths.get('train'),
            val_path=data_paths.get('val'),
            test_path=data_paths.get('test')
        )
    else:
        # 使用示例数据
        data_loader.train_dataset = data_loader.create_sample_data()
        data_loader.val_dataset = data_loader.create_sample_data()
    
    # 创建训练器
    trainer = OneTransTrainer(config, model_dir)
    
    # 开始训练
    history = trainer.train(
        train_loader=data_loader,
        val_loader=data_loader,
        epochs=epochs
    )
    
    return trainer


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练OneTrans模型')
    parser.add_argument('--config', type=str, default='small', 
                       help='模型配置: small, base, large')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='批次大小')
    parser.add_argument('--model_dir', type=str, default='./models', 
                       help='模型保存目录')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='数据目录')
    
    args = parser.parse_args()
    
    # 数据路径
    data_paths = None
    if args.data_dir:
        data_paths = {
            'train': os.path.join(args.data_dir, 'train.csv'),
            'val': os.path.join(args.data_dir, 'val.csv'),
            'test': os.path.join(args.data_dir, 'test.csv')
        }
    
    # 训练模型
    trainer = train_one_trans_model(
        config_name=args.config,
        data_paths=data_paths,
        epochs=args.epochs,
        model_dir=args.model_dir
    )
    
    print("训练完成!")
    print(f"模型保存在: {args.model_dir}")
    
    # 打印训练历史摘要
    final_train_loss = trainer.history['train_loss'][-1] if trainer.history['train_loss'] else None
    final_val_loss = trainer.history['val_loss'][-1] if trainer.history['val_loss'] else None
    
    print(f"最终训练损失: {final_train_loss:.4f}")
    print(f"最终验证损失: {final_val_loss:.4f}")