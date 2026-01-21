"""
OneTrans模型配置文件
定义模型超参数、训练参数和系统配置
"""

import tensorflow as tf
from typing import Dict, List, Union

class OneTransConfig:
    """OneTrans模型配置类"""
    
    def __init__(self):
        # 模型架构参数
        self.hidden_dim = 384  # 隐藏层维度
        self.num_layers = 8    # OneTrans块层数
        self.num_heads = 4     # 注意力头数
        self.ffn_dim = 1536    # FFN隐藏层维度
        
        # 输入配置
        self.max_seq_len = 2048  # 最大序列长度
        self.num_ns_tokens = 12  # 非序列token数量
        self.sep_token_id = 0    # [SEP] token ID
        
        # 混合参数化配置
        self.shared_s_params = True  # 序列token共享参数
        self.dedicated_ns_params = True  # 非序列token专属参数
        
        # 金字塔堆叠配置
        self.pyramid_enabled = True
        self.pyramid_ratios = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]  # 各层保留比例
        
        # 训练参数
        self.batch_size = 2048
        self.learning_rate = 0.005
        self.num_epochs = 100
        self.warmup_steps = 10000
        
        # 优化器配置
        self.optimizer_config = {
            'dense_optimizer': 'rmsprop',
            'sparse_optimizer': 'adagrad',
            'dense_lr': 0.005,
            'sparse_lr': 0.1,
            'beta1': 0.1,
            'beta2': 1.0,
            'momentum': 0.99999
        }
        
        # 正则化配置
        self.dropout_rate = 0.1
        self.weight_decay = 0.0
        self.gradient_clip_norm = 90.0
        
        # 特征配置
        self.feature_config = {
            'user_features': ['user_id', 'age', 'gender', 'location'],
            'item_features': ['item_id', 'category', 'price', 'brand'],
            'context_features': ['time', 'device', 'platform'],
            'sequence_features': ['click_seq', 'cart_seq', 'purchase_seq']
        }
        
        # 任务配置
        self.tasks = ['ctr', 'cvr']  # 多任务学习
        
        # 系统优化配置
        self.use_mixed_precision = True
        self.use_kv_cache = True
        self.use_flash_attention = True
        self.use_activation_recompute = True
        
    def to_dict(self) -> Dict:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'OneTransConfig':
        """从字典创建配置对象"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class OneTransSmallConfig(OneTransConfig):
    """小型OneTrans配置"""
    
    def __init__(self):
        super().__init__()
        self.hidden_dim = 256
        self.num_layers = 6
        self.ffn_dim = 1024


class OneTransLargeConfig(OneTransConfig):
    """大型OneTrans配置"""
    
    def __init__(self):
        super().__init__()
        self.hidden_dim = 512
        self.num_layers = 12
        self.num_heads = 8
        self.ffn_dim = 2048


def get_model_config(model_type: str = 'default') -> OneTransConfig:
    """根据模型类型获取配置"""
    config_map = {
        'small': OneTransSmallConfig,
        'default': OneTransConfig,
        'large': OneTransLargeConfig
    }
    
    if model_type not in config_map:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return config_map[model_type]()


# 默认配置实例
DEFAULT_CONFIG = OneTransConfig()