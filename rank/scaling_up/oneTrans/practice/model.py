"""
OneTrans核心模型实现
统一Transformer骨干网络，融合特征交互与序列建模
"""

import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from .config import OneTransConfig


class RMSNorm(tf.keras.layers.Layer):
    """RMSNorm归一化层"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = tf.Variable(tf.ones([dim]))
        self.eps = eps
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """前向传播"""
        variance = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(variance + self.eps)
        return x * self.scale


class MixedMHA(tf.keras.layers.Layer):
    """混合多头注意力层：序列token共享参数，非序列token专属参数"""
    
    def __init__(self, config: OneTransConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_ns_tokens = config.num_ns_tokens
        
        # 共享参数（序列token使用）
        self.Wq_shared = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.Wk_shared = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.Wv_shared = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        
        # 专属参数（非序列token使用）
        self.Wq_dedicated = [
            tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
            for _ in range(self.num_ns_tokens)
        ]
        self.Wk_dedicated = [
            tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
            for _ in range(self.num_ns_tokens)
        ]
        self.Wv_dedicated = [
            tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
            for _ in range(self.num_ns_tokens)
        ]
        
        # 输出投影
        self.Wo = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        
    def build(self, input_shape):
        self.causal_mask = self._create_causal_mask(input_shape[1])
        
    def _create_causal_mask(self, seq_len: int) -> tf.Tensor:
        """创建因果注意力掩码"""
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask
    
    def _get_projection_weights(self, idx: int) -> Tuple[tf.keras.layers.Dense, tf.keras.layers.Dense, tf.keras.layers.Dense]:
        """根据token索引获取对应的投影权重"""
        if idx < self.num_ns_tokens:
            # 非序列token使用专属参数
            return (self.Wq_dedicated[idx], self.Wk_dedicated[idx], self.Wv_dedicated[idx])
        else:
            # 序列token使用共享参数
            return (self.Wq_shared, self.Wk_shared, self.Wv_shared)
    
    def call(self, x: tf.Tensor, training: bool = False, 
             kv_cache: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """前向传播"""
        batch_size, seq_len, _ = tf.shape(x)
        
        # 计算Q/K/V
        q_list, k_list, v_list = [], [], []
        
        for i in range(seq_len):
            Wq, Wk, Wv = self._get_projection_weights(i)
            q_list.append(Wq(x[:, i:i+1, :]))
            k_list.append(Wk(x[:, i:i+1, :]))
            v_list.append(Wv(x[:, i:i+1, :]))
        
        q = tf.concat(q_list, axis=1)
        k = tf.concat(k_list, axis=1)
        v = tf.concat(v_list, axis=1)
        
        # 处理KV缓存
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = tf.concat([k_cache, k], axis=1)
            v = tf.concat([v_cache, v], axis=1)
        
        # 多头注意力
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, tf.shape(k)[1], self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, tf.shape(v)[1], self.num_heads, self.head_dim])
        
        # 注意力计算
        attn_scores = tf.einsum('bqhd,bkhd->bhqk', q, k) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # 应用因果掩码
        mask = self.causal_mask[:seq_len, :tf.shape(k)[1]]
        attn_scores = tf.where(mask == 1, attn_scores, -1e9)
        
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_output = tf.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.hidden_dim])
        
        # 输出投影
        output = self.Wo(attn_output)
        
        # 更新KV缓存
        new_kv_cache = (k, v)
        
        return output, new_kv_cache


class MixedFFN(tf.keras.layers.Layer):
    """混合前馈网络：序列token共享参数，非序列token专属参数"""
    
    def __init__(self, config: OneTransConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_ns_tokens = config.num_ns_tokens
        
        # 共享FFN（序列token使用）
        self.ffn_shared = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ffn_dim, activation='gelu'),
            tf.keras.layers.Dense(self.hidden_dim)
        ])
        
        # 专属FFN（非序列token使用）
        self.ffn_dedicated = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(self.ffn_dim, activation='gelu'),
                tf.keras.layers.Dense(self.hidden_dim)
            ]) for _ in range(self.num_ns_tokens)
        ]
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = tf.shape(x)
        
        outputs = []
        for i in range(seq_len):
            if i < self.num_ns_tokens:
                # 非序列token使用专属FFN
                output = self.ffn_dedicated[i](x[:, i:i+1, :])
            else:
                # 序列token使用共享FFN
                output = self.ffn_shared(x[:, i:i+1, :])
            outputs.append(output)
        
        return tf.concat(outputs, axis=1)


class OneTransBlock(tf.keras.layers.Layer):
    """单个OneTrans块：预归一化因果Transformer"""
    
    def __init__(self, config: OneTransConfig):
        super().__init__()
        self.config = config
        
        # 归一化层
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)
        
        # 注意力层
        self.attention = MixedMHA(config)
        
        # 前馈网络
        self.ffn = MixedFFN(config)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        
    def call(self, x: tf.Tensor, training: bool = False,
             kv_cache: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """前向传播"""
        
        # 预归一化 + 注意力 + 残差连接
        x_norm = self.norm1(x)
        attn_output, new_kv_cache = self.attention(x_norm, training=training, kv_cache=kv_cache)
        x = x + self.dropout(attn_output, training=training)
        
        # 预归一化 + FFN + 残差连接
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output, training=training)
        
        return x, new_kv_cache


class Tokenizer(tf.keras.layers.Layer):
    """统一分词器：处理序列特征和非序列特征"""
    
    def __init__(self, config: OneTransConfig):
        super().__init__()
        self.config = config
        
        # 非序列特征分词器（自动拆分）
        self.ns_tokenizer = tf.keras.Sequential([
            tf.keras.layers.Dense(config.hidden_dim * config.num_ns_tokens),
            tf.keras.layers.Reshape([config.num_ns_tokens, config.hidden_dim])
        ])
        
        # 序列特征投影（共享MLP）
        self.seq_projections = [
            tf.keras.layers.Dense(config.hidden_dim) for _ in range(len(config.feature_config['sequence_features']))
        ]
        
        # [SEP] token嵌入
        self.sep_embedding = tf.keras.layers.Embedding(1, config.hidden_dim)
        
    def call(self, non_seq_features: Dict[str, tf.Tensor], 
             seq_features: Dict[str, tf.Tensor]) -> tf.Tensor:
        """前向传播"""
        
        # 处理非序列特征
        ns_tokens = self._process_non_seq_features(non_seq_features)
        
        # 处理序列特征
        s_tokens = self._process_seq_features(seq_features)
        
        # 拼接序列token和非序列token
        tokens = tf.concat([s_tokens, ns_tokens], axis=1)
        
        return tokens
    
    def _process_non_seq_features(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        """处理非序列特征"""
        # 拼接所有非序列特征
        feature_list = []
        for feature_name in self.config.feature_config['user_features'] + \
                          self.config.feature_config['item_features'] + \
                          self.config.feature_config['context_features']:
            if feature_name in features:
                feature_list.append(features[feature_name])
        
        if not feature_list:
            return tf.zeros([tf.shape(list(features.values())[0])[0], 
                           self.config.num_ns_tokens, self.config.hidden_dim])
        
        concatenated = tf.concat(feature_list, axis=-1)
        return self.ns_tokenizer(concatenated)
    
    def _process_seq_features(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        """处理序列特征（时间戳感知融合）"""
        seq_tokens_list = []
        
        for i, seq_name in enumerate(self.config.feature_config['sequence_features']):
            if seq_name in features:
                seq_data = features[seq_name]  # [batch, seq_len, feature_dim]
                
                # 投影到统一维度
                projected_seq = self.seq_projections[i](seq_data)
                seq_tokens_list.append(projected_seq)
                
                # 添加[SEP] token（除了最后一个序列）
                if i < len(self.config.feature_config['sequence_features']) - 1:
                    batch_size = tf.shape(seq_data)[0]
                    sep_token = self.sep_embedding(tf.zeros([batch_size, 1], dtype=tf.int32))
                    seq_tokens_list.append(sep_token)
        
        if not seq_tokens_list:
            return tf.zeros([tf.shape(list(features.values())[0])[0], 0, self.config.hidden_dim])
        
        return tf.concat(seq_tokens_list, axis=1)


class PyramidScheduler:
    """金字塔调度器：逐步缩减序列token长度"""
    
    def __init__(self, config: OneTransConfig):
        self.config = config
        self.pyramid_ratios = config.pyramid_ratios
        
    def get_layer_config(self, layer_idx: int, total_seq_len: int) -> Dict:
        """获取指定层的金字塔配置"""
        if not self.config.pyramid_enabled or layer_idx >= len(self.pyramid_ratios):
            return {'keep_ratio': 1.0, 'query_indices': None}
        
        keep_ratio = self.pyramid_ratios[layer_idx]
        keep_len = max(1, int(total_seq_len * keep_ratio))
        
        # 保留尾部的token作为查询
        query_indices = list(range(total_seq_len - keep_len, total_seq_len))
        
        return {
            'keep_ratio': keep_ratio,
            'query_indices': query_indices,
            'keep_len': keep_len
        }


class OneTransModel(tf.keras.Model):
    """OneTrans统一Transformer模型"""
    
    def __init__(self, config: OneTransConfig):
        super().__init__()
        self.config = config
        
        # 分词器
        self.tokenizer = Tokenizer(config)
        
        # OneTrans块堆叠
        self.blocks = [OneTransBlock(config) for _ in range(config.num_layers)]
        
        # 金字塔调度器
        self.pyramid_scheduler = PyramidScheduler(config)
        
        # 输出归一化
        self.output_norm = RMSNorm(config.hidden_dim)
        
        # 任务头
        self.task_heads = {}
        for task in config.tasks:
            self.task_heads[task] = tf.keras.Sequential([
                tf.keras.layers.Dense(config.hidden_dim // 2, activation='gelu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        
        # KV缓存（推理时使用）
        self.kv_cache = None
        
    def call(self, non_seq_features: Dict[str, tf.Tensor], 
             seq_features: Dict[str, tf.Tensor],
             training: bool = False,
             use_kv_cache: bool = False) -> Dict[str, tf.Tensor]:
        """前向传播"""
        
        # 分词
        tokens = self.tokenizer(non_seq_features, seq_features)
        batch_size, total_seq_len, _ = tf.shape(tokens)
        
        # 金字塔调度
        current_tokens = tokens
        
        for layer_idx, block in enumerate(self.blocks):
            layer_config = self.pyramid_scheduler.get_layer_config(layer_idx, total_seq_len)
            
            if layer_config['query_indices'] is not None and self.config.pyramid_enabled:
                # 金字塔模式：仅处理查询token
                query_indices = layer_config['query_indices']
                
                # 提取查询token
                query_tokens = tf.gather(current_tokens, query_indices, axis=1)
                
                # 处理完整序列
                if use_kv_cache and self.kv_cache is not None:
                    # 使用KV缓存
                    block_output, self.kv_cache = block(
                        current_tokens, training=training, kv_cache=self.kv_cache
                    )
                else:
                    # 完整前向传播
                    block_output, self.kv_cache = block(
                        current_tokens, training=training, kv_cache=None
                    )
                
                # 仅保留查询token的输出
                current_tokens = tf.gather(block_output, query_indices, axis=1)
            else:
                # 完整模式：处理所有token
                if use_kv_cache and self.kv_cache is not None:
                    current_tokens, self.kv_cache = block(
                        current_tokens, training=training, kv_cache=self.kv_cache
                    )
                else:
                    current_tokens, self.kv_cache = block(
                        current_tokens, training=training, kv_cache=None
                    )
        
        # 输出归一化
        output_tokens = self.output_norm(current_tokens)
        
        # 任务预测（使用最后一个非序列token）
        predictions = {}
        for task, head in self.task_heads.items():
            # 取最后一个token作为任务输入
            task_input = output_tokens[:, -1, :]
            predictions[task] = head(task_input)
        
        return predictions
    
    def reset_kv_cache(self):
        """重置KV缓存"""
        self.kv_cache = None
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum([tf.keras.backend.count_params(w) for w in self.trainable_weights])
        
        return {
            'total_parameters': total_params,
            'num_layers': self.config.num_layers,
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads
        }


def create_onetrans_model(model_type: str = 'default') -> OneTransModel:
    """创建OneTrans模型"""
    from .config import get_model_config
    
    config = get_model_config(model_type)
    return OneTransModel(config)


# 测试代码
if __name__ == "__main__":
    # 创建小型模型进行测试
    config = OneTransConfig()
    config.hidden_dim = 128
    config.num_layers = 2
    config.num_ns_tokens = 4
    
    model = OneTransModel(config)
    
    # 模拟输入数据
    batch_size = 2
    seq_len = 10
    
    non_seq_features = {
        'user_id': tf.random.uniform([batch_size, 1], maxval=100, dtype=tf.int32),
        'item_id': tf.random.uniform([batch_size, 1], maxval=1000, dtype=tf.int32),
        'price': tf.random.uniform([batch_size, 1])
    }
    
    seq_features = {
        'click_seq': tf.random.uniform([batch_size, seq_len, 64]),
        'cart_seq': tf.random.uniform([batch_size, seq_len // 2, 64])
    }
    
    # 前向传播
    predictions = model(non_seq_features, seq_features, training=True)
    
    print("模型测试成功!")
    print(f"预测结果: {predictions}")
    print(f"模型信息: {model.get_model_info()}")