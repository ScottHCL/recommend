"""
OneTrans数据加载和预处理模块
处理序列特征和非序列特征，支持工业级推荐系统数据格式
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .config import OneTransConfig


class FeatureProcessor:
    """特征处理器：处理数值和类别特征"""
    
    def __init__(self, config: OneTransConfig):
        self.config = config
        self.feature_stats = {}
        self.vocab_sizes = {}
        
    def fit(self, data: pd.DataFrame):
        """拟合特征处理器"""
        # 数值特征统计
        for feature in self._get_numerical_features():
            if feature in data.columns:
                self.feature_stats[feature] = {
                    'mean': data[feature].mean(),
                    'std': data[feature].std(),
                    'min': data[feature].min(),
                    'max': data[feature].max()
                }
        
        # 类别特征词汇表大小
        for feature in self._get_categorical_features():
            if feature in data.columns:
                self.vocab_sizes[feature] = int(data[feature].max() + 1)
    
    def _get_numerical_features(self) -> List[str]:
        """获取数值特征列表"""
        return ['price', 'age', 'ctr']  # 示例数值特征
    
    def _get_categorical_features(self) -> List[str]:
        """获取类别特征列表"""
        return ['user_id', 'item_id', 'category', 'brand', 'location', 'device']
    
    def process_numerical_feature(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        """处理数值特征：标准化"""
        if feature_name not in self.feature_stats:
            return values
        
        stats = self.feature_stats[feature_name]
        # 标准化
        normalized = (values - stats['mean']) / (stats['std'] + 1e-8)
        # 截断异常值
        normalized = np.clip(normalized, -3, 3)
        return normalized
    
    def process_categorical_feature(self, feature_name: str, values: np.ndarray) -> np.ndarray:
        """处理类别特征：one-hot编码"""
        if feature_name not in self.vocab_sizes:
            return values
        
        vocab_size = self.vocab_sizes[feature_name]
        # 简单的one-hot编码（实际中可能使用嵌入）
        return tf.one_hot(values.astype(int), depth=vocab_size).numpy()


class SequenceProcessor:
    """序列处理器：处理用户行为序列"""
    
    def __init__(self, config: OneTransConfig):
        self.config = config
        self.max_seq_len = config.max_seq_len
        
    def process_sequence(self, sequence_data: np.ndarray, 
                         sequence_type: str) -> np.ndarray:
        """处理单个序列"""
        if len(sequence_data) == 0:
            return np.zeros((self.max_seq_len, 64))  # 默认特征维度
        
        # 截断或填充序列
        if len(sequence_data) > self.max_seq_len:
            # 保留最近的max_seq_len个事件
            processed = sequence_data[-self.max_seq_len:]
        else:
            # 填充到max_seq_len
            pad_len = self.max_seq_len - len(sequence_data)
            processed = np.pad(sequence_data, 
                             ((pad_len, 0), (0, 0)), 
                             mode='constant')
        
        return processed
    
    def process_multi_sequences(self, sequences: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """处理多行为序列"""
        processed_sequences = {}
        
        for seq_type, seq_data in sequences.items():
            processed_sequences[seq_type] = self.process_sequence(seq_data, seq_type)
        
        return processed_sequences


class OneTransDataset:
    """OneTrans数据集类"""
    
    def __init__(self, config: OneTransConfig, data_path: Optional[str] = None):
        self.config = config
        self.feature_processor = FeatureProcessor(config)
        self.sequence_processor = SequenceProcessor(config)
        
        # 数据存储
        self.non_seq_data = {}
        self.seq_data = {}
        self.labels = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """加载数据（示例实现）"""
        # 这里应该是从文件加载数据的逻辑
        # 为了示例，我们创建一些模拟数据
        self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据"""
        num_samples = 1000
        
        # 非序列特征
        self.non_seq_data = {
            'user_id': np.random.randint(0, 1000, num_samples),
            'item_id': np.random.randint(0, 5000, num_samples),
            'price': np.random.uniform(0, 1000, num_samples),
            'category': np.random.randint(0, 50, num_samples),
            'time': np.random.randint(0, 24, num_samples)
        }
        
        # 序列特征
        self.seq_data = {}
        for seq_type in self.config.feature_config['sequence_features']:
            # 为每个样本创建不同长度的序列
            sequences = []
            for i in range(num_samples):
                seq_len = np.random.randint(1, self.config.max_seq_len + 1)
                seq = np.random.randn(seq_len, 64)  # 64维特征
                sequences.append(seq)
            self.seq_data[seq_type] = sequences
        
        # 标签
        self.labels = {
            'ctr': np.random.randint(0, 2, num_samples).astype(np.float32),
            'cvr': np.random.randint(0, 2, num_samples).astype(np.float32)
        }
    
    def _process_features(self, idx: int) -> Tuple[Dict, Dict]:
        """处理单个样本的特征"""
        # 处理非序列特征
        non_seq_features = {}
        for feature_name, feature_data in self.non_seq_data.items():
            if isinstance(feature_data, np.ndarray):
                value = feature_data[idx]
                if feature_name in ['user_id', 'item_id', 'category']:
                    # 类别特征
                    processed = self.feature_processor.process_categorical_feature(
                        feature_name, np.array([value])
                    )[0]
                else:
                    # 数值特征
                    processed = self.feature_processor.process_numerical_feature(
                        feature_name, np.array([value])
                    )[0]
                non_seq_features[feature_name] = processed
        
        # 处理序列特征
        seq_features = {}
        for seq_type, seq_list in self.seq_data.items():
            if idx < len(seq_list):
                seq_data = seq_list[idx]
                processed_seq = self.sequence_processor.process_sequence(seq_data, seq_type)
                seq_features[seq_type] = processed_seq
        
        return non_seq_features, seq_features
    
    def __len__(self) -> int:
        """数据集大小"""
        if not self.non_seq_data:
            return 0
        return len(next(iter(self.non_seq_data.values())))
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict]:
        """获取单个样本"""
        non_seq_features, seq_features = self._process_features(idx)
        
        # 获取标签
        sample_labels = {}
        for task, label_data in self.labels.items():
            if idx < len(label_data):
                sample_labels[task] = label_data[idx]
        
        return non_seq_features, seq_features, sample_labels
    
    def get_tf_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """创建TensorFlow数据集"""
        def generator():
            for i in range(len(self)):
                non_seq, seq, labels = self[i]
                yield (non_seq, seq), labels
        
        # 定义输出签名
        output_signature = (
            {
                'non_seq_features': tf.TensorSpec(shape=(None,), dtype=tf.float32),
                'seq_features': tf.TensorSpec(shape=(None, self.config.max_seq_len, 64), dtype=tf.float32)
            },
            {
                'ctr': tf.TensorSpec(shape=(), dtype=tf.float32),
                'cvr': tf.TensorSpec(shape=(), dtype=tf.float32)
            }
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class DataLoader:
    """数据加载器：管理训练、验证、测试数据集"""
    
    def __init__(self, config: OneTransConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_datasets(self, train_path: str, val_path: str, test_path: str):
        """加载训练、验证、测试数据集"""
        # 训练集
        self.train_dataset = OneTransDataset(self.config, train_path)
        
        # 验证集
        self.val_dataset = OneTransDataset(self.config, val_path)
        
        # 测试集
        self.test_dataset = OneTransDataset(self.config, test_path)
    
    def get_train_dataset(self, batch_size: int = None) -> tf.data.Dataset:
        """获取训练数据集"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if self.train_dataset is None:
            raise ValueError("训练数据集未加载")
        
        return self.train_dataset.get_tf_dataset(batch_size, shuffle=True)
    
    def get_val_dataset(self, batch_size: int = None) -> tf.data.Dataset:
        """获取验证数据集"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if self.val_dataset is None:
            raise ValueError("验证数据集未加载")
        
        return self.val_dataset.get_tf_dataset(batch_size, shuffle=False)
    
    def get_test_dataset(self, batch_size: int = None) -> tf.data.Dataset:
        """获取测试数据集"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if self.test_dataset is None:
            raise ValueError("测试数据集未加载")
        
        return self.test_dataset.get_tf_dataset(batch_size, shuffle=False)
    
    def get_data_info(self) -> Dict:
        """获取数据集信息"""
        info = {}
        
        if self.train_dataset:
            info['train_samples'] = len(self.train_dataset)
        if self.val_dataset:
            info['val_samples'] = len(self.val_dataset)
        if self.test_dataset:
            info['test_samples'] = len(self.test_dataset)
        
        return info


# 工具函数
def create_sample_batch(batch_size: int = 2, config: Optional[OneTransConfig] = None) -> Tuple[Dict, Dict, Dict]:
    """创建示例批次数据"""
    if config is None:
        from .config import OneTransConfig
        config = OneTransConfig()
    
    # 非序列特征
    non_seq_features = {}
    for feature in config.feature_config['user_features']:
        non_seq_features[feature] = tf.random.uniform([batch_size, 1], maxval=100, dtype=tf.int32)
    
    for feature in config.feature_config['item_features']:
        non_seq_features[feature] = tf.random.uniform([batch_size, 1], maxval=1000, dtype=tf.int32)
    
    for feature in config.feature_config['context_features']:
        non_seq_features[feature] = tf.random.uniform([batch_size, 1])
    
    # 序列特征
    seq_features = {}
    for seq_type in config.feature_config['sequence_features']:
        seq_len = np.random.randint(1, config.max_seq_len + 1)
        seq_features[seq_type] = tf.random.uniform([batch_size, seq_len, 64])
    
    # 标签
    labels = {}
    for task in config.tasks:
        labels[task] = tf.random.uniform([batch_size, 1])
    
    return non_seq_features, seq_features, labels


# 测试代码
if __name__ == "__main__":
    from config import OneTransConfig
    
    # 创建配置
    config = OneTransConfig()
    config.hidden_dim = 128
    config.max_seq_len = 100
    
    # 创建数据加载器
    data_loader = DataLoader(config)
    
    # 创建示例数据集
    data_loader.train_dataset = OneTransDataset(config)
    
    # 获取TensorFlow数据集
    dataset = data_loader.get_train_dataset(batch_size=4)
    
    # 测试数据加载
    for batch in dataset.take(1):
        features, labels = batch
        print("非序列特征形状:", {k: v.shape for k, v in features['non_seq_features'].items()})
        print("序列特征形状:", {k: v.shape for k, v in features['seq_features'].items()})
        print("标签形状:", {k: v.shape for k, v in labels.items()})
    
    print("数据加载器测试成功!")