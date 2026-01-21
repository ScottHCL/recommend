"""
KuaiFormer配置文件
定义模型超参数、训练配置和路径设置
"""

import tensorflow as tf
from typing import Dict, Any

class KuaiFormerConfig:
    """KuaiFormer模型配置类"""
    
    def __init__(self, use_causal_mask: bool = False):
        # 模型架构参数
        self.embedding_dim = 128  # 嵌入维度
        self.hidden_dim = 256     # Transformer隐藏层维度
        self.num_layers = 6       # Transformer层数
        self.num_heads = 8        # 注意力头数
        self.ffn_dim = 512        # 前馈网络维度
        
        # 序列处理参数
        self.max_sequence_length = 256  # 最大序列长度
        self.early_group_size = 64      # 早期物品组大小
        self.mid_group_size = 16        # 中间物品组大小
        self.late_group_size = 48       # 最新物品组大小
        self.num_query_tokens = 4       # 查询token数量
        
        # 训练参数
        self.batch_size = 512
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.warmup_steps = 10000
        self.label_smoothing = 0.1      # 标签平滑参数
        
        # 特征配置
        self.video_id_vocab_size = 10000000  # 视频ID词汇表大小
        self.category_vocab_size = 10000     # 类别词汇表大小
        self.tag_vocab_size = 50000          # 标签词汇表大小
        
        # 连续特征分桶配置
        self.duration_buckets = 1000     # 时长分桶数
        self.max_duration = 300          # 最大时长(秒)
        self.time_buckets = 1000         # 时间分桶数
        
        # 优化器配置
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.98
        self.adam_epsilon = 1e-9
        self.weight_decay = 0.01
        
        # 注意力模式
        self.use_causal_mask = use_causal_mask
        
        # 推理配置
        self.top_k = 1000                # 检索Top-K数量
        self.faiss_index_type = "IVF1024,Flat"  # FAISS索引类型
        
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# 创建默认配置实例
config = KuaiFormerConfig()