"""
KuaiFormer核心模型实现
基于Transformer的检索模型，支持多兴趣建模和长序列压缩
"""

import tensorflow as tf
from tensorflow.python.keras import layers, Model
import numpy as np
from typing import List, Tuple, Optional
from config import KuaiFormerConfig

class EmbeddingModule(layers.Layer):
    """嵌入模块：处理离散和连续特征"""
    
    def __init__(self, config: KuaiFormerConfig):
        super(EmbeddingModule, self).__init__()
        self.config = config
        
        # 离散特征嵌入层
        self.video_id_embedding = layers.Embedding(
            input_dim=config.video_id_vocab_size,
            output_dim=config.embedding_dim,
            name='video_id_embedding'
        )
        
        self.category_embedding = layers.Embedding(
            input_dim=config.category_vocab_size,
            output_dim=config.embedding_dim,
            name='category_embedding'
        )
        
        self.tag_embedding = layers.Embedding(
            input_dim=config.tag_vocab_size,
            output_dim=config.embedding_dim,
            name='tag_embedding'
        )
        
        # 连续特征分桶嵌入层
        self.duration_embedding = layers.Embedding(
            input_dim=config.duration_buckets,
            output_dim=config.embedding_dim,
            name='duration_embedding'
        )
        
        self.time_embedding = layers.Embedding(
            input_dim=config.time_buckets,
            output_dim=config.embedding_dim,
            name='time_embedding'
        )
        
        # MLP用于特征融合
        self.mlp = tf.keras.Sequential([
            layers.Dense(config.embedding_dim, activation='relu'),
            layers.Dense(config.embedding_dim),
            layers.LayerNormalization()
        ], name='feature_fusion_mlp')
    
    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        处理输入特征，生成token表征
        
        Args:
            inputs: 包含视频ID、类别、标签、时长、时间等特征的字典
            
        Returns:
            token表征序列 [batch_size, seq_len, embedding_dim]
        """
        # 嵌入离散特征
        video_emb = self.video_id_embedding(inputs['video_ids'])
        category_emb = self.category_embedding(inputs['categories'])
        tag_emb = self.tag_embedding(inputs['tags'])
        
        # 处理连续特征：分桶后嵌入
        duration_buckets = tf.cast(
            inputs['durations'] / self.config.max_duration * self.config.duration_buckets,
            tf.int32
        )
        duration_emb = self.duration_embedding(duration_buckets)
        
        time_buckets = tf.cast(
            inputs['timestamps'] % self.config.time_buckets,
            tf.int32
        )
        time_emb = self.time_embedding(time_buckets)
        
        # 拼接所有特征
        combined_emb = tf.concat([
            video_emb, category_emb, tag_emb, duration_emb, time_emb
        ], axis=-1)
        
        # 通过MLP融合特征
        token_representation = self.mlp(combined_emb)
        
        return token_representation

class AdaptiveCompressionModule(layers.Layer):
    """自适应物品压缩模块：处理长序列"""
    
    def __init__(self, config: KuaiFormerConfig):
        super(AdaptiveCompressionModule, self).__init__()
        self.config = config
        
        # 双向Transformer用于物品组压缩
        self.bi_transformer_early = self._build_bi_transformer()
        self.bi_transformer_mid = self._build_bi_transformer()
    
    def _build_bi_transformer(self) -> tf.keras.layers.Layer:
        """构建双向Transformer层"""
        return tf.keras.Sequential([
            layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.embedding_dim // self.config.num_heads
            ),
            layers.LayerNormalization(),
            layers.Dense(self.config.embedding_dim, activation='relu'),
            layers.LayerNormalization()
        ])
    
    def call(self, token_sequence: tf.Tensor) -> tf.Tensor:
        """
        压缩长序列，减少计算复杂度
        
        Args:
            token_sequence: 原始token序列 [batch_size, seq_len, embedding_dim]
            
        Returns:
            压缩后的序列 [batch_size, compressed_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = token_sequence.shape
        
        # 划分序列为三部分
        early_end = self.config.early_group_size * 2  # 前128个物品
        mid_end = early_end + self.config.mid_group_size * 5  # 中间80个物品
        
        early_tokens = token_sequence[:, :early_end]
        mid_tokens = token_sequence[:, early_end:mid_end]
        late_tokens = token_sequence[:, mid_end:]
        
        # 压缩早期物品组
        early_compressed = self._compress_group(
            early_tokens, self.config.early_group_size, self.bi_transformer_early
        )
        
        # 压缩中间物品组
        mid_compressed = self._compress_group(
            mid_tokens, self.config.mid_group_size, self.bi_transformer_mid
        )
        
        # 拼接压缩后的序列
        compressed_sequence = tf.concat([
            early_compressed, mid_compressed, late_tokens
        ], axis=1)
        
        return compressed_sequence
    
    def _compress_group(self, tokens: tf.Tensor, group_size: int, 
                        transformer: tf.keras.layers.Layer) -> tf.Tensor:
        """压缩物品组"""
        batch_size, seq_len, embedding_dim = tokens.shape
        num_groups = seq_len // group_size
        
        # 重塑为组形式
        groups = tf.reshape(
            tokens, 
            [batch_size, num_groups, group_size, embedding_dim]
        )
        
        # 对每个组应用双向Transformer并取平均
        compressed_groups = []
        for i in range(num_groups):
            group = groups[:, i, :, :]
            # 双向注意力（无掩码）
            attended = transformer(group, group)
            # 平均池化
            compressed = tf.reduce_mean(attended, axis=1)
            compressed_groups.append(compressed)
        
        return tf.stack(compressed_groups, axis=1)

class TransformerBlock(layers.Layer):
    """Transformer块，支持双向和因果注意力"""
    
    def __init__(self, config: KuaiFormerConfig, use_causal_mask: bool = False):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.use_causal_mask = use_causal_mask
        
        # RMS归一化
        self.rms_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.rms_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 多头注意力
        self.attention = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.embedding_dim // config.num_heads
        )
        
        # 前馈网络
        self.ffn = tf.keras.Sequential([
            layers.Dense(config.ffn_dim, activation='swish'),
            layers.Dense(config.embedding_dim)
        ])
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """前向传播"""
        # 残差连接1
        attn_output = self.attention(
            self.rms_norm1(x), 
            self.rms_norm1(x),
            use_causal_mask=self.use_causal_mask
        )
        x = x + attn_output
        
        # 残差连接2
        ffn_output = self.ffn(self.rms_norm2(x))
        x = x + ffn_output
        
        return x

class KuaiFormer(Model):
    """KuaiFormer主模型，支持两种模式"""
    
    def __init__(self, config: KuaiFormerConfig, use_causal_mask: bool = False):
        super(KuaiFormer, self).__init__()
        self.config = config
        self.use_causal_mask = use_causal_mask
        
        # 模块初始化
        self.embedding_module = EmbeddingModule(config)
        self.compression_module = AdaptiveCompressionModule(config)
        
        # 查询token（用于多兴趣提取）
        self.query_tokens = tf.Variable(
            tf.random.normal([config.num_query_tokens, config.embedding_dim]),
            trainable=True,
            name='query_tokens'
        )
        
        # Transformer层堆叠
        self.transformer_layers = [
            TransformerBlock(config, use_causal_mask=use_causal_mask) 
            for _ in range(config.num_layers)
        ]
        
        # 输出层
        self.output_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        统一的前向传播接口（完全矩阵计算优化）
        
        Args:
            inputs: 输入特征字典
            training: 是否训练模式
            
        Returns:
            如果use_causal_mask=False: [batch_size, num_query_tokens, embedding_dim]
            如果use_causal_mask=True: [batch_size, seq_len-1, num_query_tokens, embedding_dim]
        """
        # 1. 嵌入层
        token_sequence = self.embedding_module(inputs)
        
        # 2. 序列压缩
        compressed_sequence = self.compression_module(token_sequence)
        
        # 3. 根据注意力模式选择处理方式
        if not self.use_causal_mask:
            # 双向注意力模式：处理整个序列
            return self._call_bidirectional_single_sequence(compressed_sequence, training)
        else:
            # 因果注意力模式：序列到序列预测（使用融合的矩阵计算实现）
            return self._call_causal_sequence_to_sequence(compressed_sequence, training)
    
    def _call_bidirectional_single_sequence(self, compressed_sequence: tf.Tensor, training: bool) -> tf.Tensor:
        """双向模式：单个序列预测"""
        # 3. 添加查询token
        batch_size = tf.shape(compressed_sequence)[0]
        query_tokens_batch = tf.tile(
            tf.expand_dims(self.query_tokens, 0),
            [batch_size, 1, 1]
        )
        
        # 拼接序列和查询token
        transformer_input = tf.concat([
            compressed_sequence, query_tokens_batch
        ], axis=1)
        
        # 4. 通过Transformer层
        x = transformer_input
        for layer in self.transformer_layers:
            x = layer(x, training=training)
        
        # 5. 提取查询token对应的输出
        seq_len = tf.shape(compressed_sequence)[1]
        interest_representations = x[:, seq_len:, :]
        
        # 6. 归一化
        normalized_output = self.output_norm(interest_representations)
        
        return normalized_output
    
    def _call_causal_sequence_to_sequence(self, compressed_sequence: tf.Tensor, training: bool, 
                                         target_position: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        因果掩码模式：序列到序列预测（预计算映射表优化）
        
        支持两种模式：
        1. 序列到序列：target_position=None，返回所有位置的多兴趣表征
        2. 单个位置预测：target_position指定，返回特定位置的多兴趣表征
        
        Args:
            compressed_sequence: 压缩后的序列 [batch_size, seq_len, hidden_dim]
            training: 训练模式
            target_position: 目标位置 [batch_size]，如果为None则预测所有位置
            
        Returns:
            如果target_position=None: [batch_size, seq_len-1, num_query_tokens, hidden_dim]
            如果target_position指定: [batch_size, num_query_tokens, hidden_dim]
        """
        batch_size = tf.shape(compressed_sequence)[0]
        seq_len = tf.shape(compressed_sequence)[1]
        num_query_tokens = self.config.num_query_tokens
        hidden_dim = self.config.hidden_dim
        
        # 判断模式：序列到序列 vs 单个位置预测
        is_sequence_to_sequence = target_position is None
        
        if is_sequence_to_sequence:
            # 序列到序列模式：预测所有位置
            num_predictions = seq_len - 1
            prediction_positions = tf.range(num_predictions)  # [num_predictions]
        else:
            # 单个位置预测模式
            num_predictions = 1
            prediction_positions = tf.expand_dims(target_position, 0)  # [1]
        
        # 1. 预计算或获取映射表（确保训练和推理一致）
        position_mappings = self._get_or_create_position_mappings(seq_len, num_query_tokens, num_predictions)
        
        # 扩展映射到batch维度 [batch_size, num_predictions, seq_len + num_query_tokens]
        batch_mappings = tf.tile(tf.expand_dims(position_mappings, 0), [batch_size, 1, 1])
        
        # 2. 准备扩展序列（原始序列 + 查询token）
        query_tokens_batch = tf.tile(
            tf.expand_dims(self.query_tokens, 0),
            [batch_size, 1, 1]
        )  # [batch_size, num_query_tokens, hidden_dim]
        
        extended_sequence = tf.concat([compressed_sequence, query_tokens_batch], axis=1)
        # [batch_size, seq_len + num_query_tokens, hidden_dim]
        
        # 3. 构建Transformer输入 [batch_size, num_predictions, seq_len + num_query_tokens, hidden_dim]
        transformer_input = tf.gather(extended_sequence, batch_mappings, batch_dims=1)
        
        # 重塑为批量处理 [batch_size * num_predictions, seq_len + num_query_tokens, hidden_dim]
        transformer_input_flat = tf.reshape(
            transformer_input,
            [batch_size * num_predictions, seq_len + num_query_tokens, hidden_dim]
        )
        
        # 4. 通过Transformer层
        x = transformer_input_flat
        for layer in self.transformer_layers:
            x = layer(x, training=training)
        
        # 5. 预计算查询token提取索引
        query_indices = self._get_or_create_query_indices(seq_len, num_query_tokens, num_predictions)
        
        # 扩展索引到batch维度 [batch_size, num_predictions, num_query_tokens]
        batch_query_indices = tf.tile(tf.expand_dims(query_indices, 0), [batch_size, 1, 1])
        
        # 重塑Transformer输出 [batch_size, num_predictions, seq_len + num_query_tokens, hidden_dim]
        x_reshaped = tf.reshape(x, [batch_size, num_predictions, seq_len + num_query_tokens, hidden_dim])
        
        # 提取查询token输出 [batch_size, num_predictions, num_query_tokens, hidden_dim]
        query_output = tf.gather(x_reshaped, batch_query_indices, batch_dims=2)
        
        # 6. 归一化
        normalized_output = self.output_norm(query_output)
        
        # 7. 根据模式调整输出形状
        if is_sequence_to_sequence:
            # 序列到序列模式：返回 [batch_size, seq_len-1, num_query_tokens, hidden_dim]
            return normalized_output
        else:
            # 单个位置预测模式：返回 [batch_size, num_query_tokens, hidden_dim]
            return tf.squeeze(normalized_output, axis=1)
    
    def _get_or_create_position_mappings(self, seq_len: tf.Tensor, num_query_tokens: int, num_predictions: tf.Tensor) -> tf.Tensor:
        """
        预计算位置映射表，确保训练和推理一致性
        
        Args:
            seq_len: 序列长度
            num_query_tokens: 查询token数量
            num_predictions: 预测位置数量
            
        Returns:
            位置映射表 [num_predictions, seq_len + num_query_tokens]
        """
        # 将动态张量转换为静态值用于缓存键
        seq_len_val = seq_len.numpy() if hasattr(seq_len, 'numpy') else int(seq_len)
        num_predictions_val = num_predictions.numpy() if hasattr(num_predictions, 'numpy') else int(num_predictions)
        
        cache_key = f"position_mappings_{seq_len_val}_{num_query_tokens}_{num_predictions_val}"
        
        # 检查是否已缓存
        if hasattr(self, '_position_mappings_cache') and cache_key in self._position_mappings_cache:
            return self._position_mappings_cache[cache_key]
        
        # 首次调用时初始化缓存
        if not hasattr(self, '_position_mappings_cache'):
            self._position_mappings_cache = {}
        
        # 构建位置映射表
        prediction_positions = tf.range(num_predictions_val)  # [num_predictions]
        
        def build_position_mapping(pred_pos):
            # 构建一个映射表：扩展序列位置 -> 原始序列/查询token索引
            mapping = []
            
            # 前pred_pos+1个token
            for i in range(pred_pos + 1):
                mapping.append(i)
            
            # 查询token（使用特殊索引：seq_len + 0, seq_len + 1, ...）
            for i in range(num_query_tokens):
                mapping.append(seq_len_val + i)
            
            # 后seq_len-pred_pos-1个token
            for i in range(pred_pos + 1, seq_len_val):
                mapping.append(i)
            
            return tf.constant(mapping, dtype=tf.int32)
        
        # 为每个预测位置构建映射
        position_mappings = tf.map_fn(
            build_position_mapping,
            prediction_positions,
            fn_output_signature=tf.TensorSpec(shape=[seq_len_val + num_query_tokens], dtype=tf.int32)
        )  # [num_predictions, seq_len + num_query_tokens]
        
        # 缓存结果
        self._position_mappings_cache[cache_key] = position_mappings
        
        return position_mappings
    
    def _get_or_create_query_indices(self, seq_len: tf.Tensor, num_query_tokens: int, num_predictions: tf.Tensor) -> tf.Tensor:
        """
        预计算查询token提取索引，确保训练和推理一致性
        
        Args:
            seq_len: 序列长度
            num_query_tokens: 查询token数量
            num_predictions: 预测位置数量
            
        Returns:
            查询token索引 [num_predictions, num_query_tokens]
        """
        # 将动态张量转换为静态值用于缓存键
        seq_len_val = seq_len.numpy() if hasattr(seq_len, 'numpy') else int(seq_len)
        num_predictions_val = num_predictions.numpy() if hasattr(num_predictions, 'numpy') else int(num_predictions)
        
        cache_key = f"query_indices_{seq_len_val}_{num_query_tokens}_{num_predictions_val}"
        
        # 检查是否已缓存
        if hasattr(self, '_query_indices_cache') and cache_key in self._query_indices_cache:
            return self._query_indices_cache[cache_key]
        
        # 首次调用时初始化缓存
        if not hasattr(self, '_query_indices_cache'):
            self._query_indices_cache = {}
        
        # 构建查询token提取索引
        prediction_positions = tf.range(num_predictions_val)  # [num_predictions]
        
        def get_query_indices(pred_pos):
            start_idx = pred_pos + 1
            return tf.range(start_idx, start_idx + num_query_tokens)
        
        query_indices = tf.map_fn(
            get_query_indices,
            prediction_positions,
            fn_output_signature=tf.TensorSpec(shape=[num_query_tokens], dtype=tf.int32)
        )  # [num_predictions, num_query_tokens]
        
        # 缓存结果
        self._query_indices_cache[cache_key] = query_indices
        
        return query_indices
    
    def clear_cache(self):
        """
        清理预计算的映射表缓存
        在模型参数变化或序列长度变化时调用
        """
        if hasattr(self, '_position_mappings_cache'):
            self._position_mappings_cache.clear()
        if hasattr(self, '_query_indices_cache'):
            self._query_indices_cache.clear()
    
    def call_with_position(self, inputs: Dict[str, tf.Tensor], 
                          target_position: tf.Tensor, 
                          training: bool = False) -> tf.Tensor:
        """
        因果注意力模式的前向传播（单个位置预测，使用融合的矩阵计算实现）
        
        Args:
            inputs: 输入特征字典
            target_position: 目标预测位置 [batch_size]
            training: 是否训练模式
            
        Returns:
            多兴趣表征 [batch_size, num_query_tokens, embedding_dim]
        """
        if not self.use_causal_mask:
            raise ValueError("此方法仅适用于因果注意力模式")
        
        # 1. 嵌入层
        token_sequence = self.embedding_module(inputs)
        
        # 2. 序列压缩
        compressed_sequence = self.compression_module(token_sequence)
        
        # 3. 使用融合的矩阵计算实现（单个位置预测作为序列到序列的特例）
        return self._call_causal_sequence_to_sequence(compressed_sequence, training, target_position)
    

    
    def compute_scores(self, interest_representations: tf.Tensor, 
                      candidate_embeddings: tf.Tensor) -> tf.Tensor:
        """
        计算候选物品与兴趣表征的相似度分数
        
        Args:
            interest_representations: 兴趣表征 [batch_size, num_queries, embedding_dim]
            candidate_embeddings: 候选物品嵌入 [batch_size, num_candidates, embedding_dim]
            
        Returns:
            相似度分数 [batch_size, num_candidates]
        """
        # 计算所有兴趣表征与候选物品的内积
        scores = tf.matmul(
            candidate_embeddings,  # [batch_size, num_candidates, embedding_dim]
            interest_representations,  # [batch_size, embedding_dim, num_queries]
            transpose_b=True
        )  # [batch_size, num_candidates, num_queries]
        
        # 取最大分数（argmax策略）
        max_scores = tf.reduce_max(scores, axis=-1)  # [batch_size, num_candidates]
        
        return max_scores

class KuaiFormerLoss:
    """KuaiFormer损失函数，包含LogQ校正和标签平滑"""
    
    def __init__(self, config: KuaiFormerConfig):
        self.config = config
    
    def __call__(self, positive_scores: tf.Tensor, negative_scores: tf.Tensor,
                item_popularity: tf.Tensor) -> tf.Tensor:
        """
        计算带LogQ校正和标签平滑的损失
        
        Args:
            positive_scores: 正例分数 [batch_size]
            negative_scores: 负例分数 [batch_size, num_negatives]
            item_popularity: 物品流行度 [batch_size, num_negatives + 1]
            
        Returns:
            损失值
        """
        batch_size = tf.shape(positive_scores)[0]
        num_negatives = tf.shape(negative_scores)[1]
        
        # LogQ校正
        positive_logq = tf.math.log(item_popularity[:, 0] + 1e-8)
        negative_logq = tf.math.log(item_popularity[:, 1:] + 1e-8)
        
        positive_scores_corrected = positive_scores - positive_logq
        negative_scores_corrected = negative_scores - negative_logq
        
        # 标签平滑损失
        alpha = self.config.label_smoothing
        
        # 正例损失
        positive_loss = -(1 - alpha) * tf.math.log(
            tf.nn.softmax(tf.concat([
                tf.expand_dims(positive_scores_corrected, 1),
                negative_scores_corrected
            ], axis=1), axis=1)[:, 0] + 1e-8
        )
        
        # 负例损失
        negative_loss = -alpha / tf.cast(num_negatives, tf.float32) * tf.reduce_sum(
            tf.math.log(1 - tf.nn.softmax(negative_scores_corrected, axis=1) + 1e-8),
            axis=1
        )
        
        total_loss = tf.reduce_mean(positive_loss + negative_loss)
        
        return total_loss