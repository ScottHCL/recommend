"""
KuaiFormer评估脚本
支持离线评估和在线指标计算
"""

import tensorflow as tf
import numpy as np
import faiss
import os
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score

from config import KuaiFormerConfig
from model import KuaiFormer
from data_loader import KuaiFormerDataLoader

class KuaiFormerEvaluator:
    """KuaiFormer评估器"""
    
    def __init__(self, config: KuaiFormerConfig, model_path: str):
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        self.faiss_index = None
        
    def build_faiss_index(self, video_embeddings: np.ndarray) -> None:
        """构建FAISS索引用于快速检索"""
        embedding_dim = video_embeddings.shape[1]
        
        # 创建索引
        if self.config.faiss_index_type == "IVF1024,Flat":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, embedding_dim, 1024
            )
            
            # 训练索引
            self.faiss_index.train(video_embeddings.astype('float32'))
        else:
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        
        # 添加向量到索引
        self.faiss_index.add(video_embeddings.astype('float32'))
        
        print(f"FAISS索引构建完成，包含 {self.faiss_index.ntotal} 个向量")
    
    def evaluate_retrieval_metrics(self, test_dataset: tf.data.Dataset, 
                                 video_embeddings: np.ndarray) -> Dict[str, float]:
        """评估检索指标"""
        
        # 构建索引
        self.build_faiss_index(video_embeddings)
        
        metrics = {
            'recall@1': 0, 'recall@5': 0, 'recall@10': 0, 'recall@50': 0, 'recall@100': 0,
            'ndcg@10': 0, 'ndcg@50': 0, 'ndcg@100': 0,
            'mrr': 0, 'map': 0
        }
        total_samples = 0
        
        for batch_idx, batch in enumerate(test_dataset):
            # 生成用户兴趣表征
            interest_representations = self.model(batch['history_features'], training=False)
            
            # 获取正例视频ID
            positive_videos = batch['positive_video'].numpy()
            
            # 为每个兴趣表征进行检索
            batch_size = interest_representations.shape[0]
            total_samples += batch_size
            
            for i in range(batch_size):
                user_interests = interest_representations[i].numpy()  # [num_queries, dim]
                
                # 对每个兴趣查询进行检索
                all_scores = []
                all_indices = []
                
                for j in range(user_interests.shape[0]):
                    query = user_interests[j:j+1].astype('float32')
                    
                    # FAISS检索
                    scores, indices = self.faiss_index.search(query, self.config.top_k)
                    all_scores.append(scores[0])
                    all_indices.append(indices[0])
                
                # 合并多兴趣结果
                combined_scores = np.max(np.array(all_scores), axis=0)
                combined_indices = all_indices[np.argmax(np.array(all_scores), axis=0)]
                
                # 计算指标
                positive_video_id = positive_videos[i]
                
                # 找到正例在检索结果中的位置
                try:
                    rank = np.where(combined_indices == positive_video_id)[0][0] + 1
                except IndexError:
                    rank = self.config.top_k + 1  # 未找到
                
                # Recall@K
                for k in [1, 5, 10, 50, 100]:
                    if rank <= k:
                        metrics[f'recall@{k}'] += 1
                
                # NDCG@K
                for k in [10, 50, 100]:
                    if rank <= k:
                        metrics[f'ndcg@{k}'] += 1 / np.log2(rank + 1)
                
                # MRR
                if rank <= self.config.top_k:
                    metrics['mrr'] += 1 / rank
        
        # 计算平均值
        for key in metrics:
            if 'recall' in key or 'ndcg' in key:
                k = int(key.split('@')[1])
                metrics[key] /= total_samples
            elif key == 'mrr':
                metrics[key] /= total_samples
        
        # 计算MAP
        metrics['map'] = self._calculate_map(test_dataset, video_embeddings)
        
        return metrics
    
    def _calculate_map(self, test_dataset: tf.data.Dataset, 
                      video_embeddings: np.ndarray) -> float:
        """计算平均精度均值"""
        total_ap = 0
        total_samples = 0
        
        for batch in test_dataset:
            interest_representations = self.model(batch['history_features'], training=False)
            positive_videos = batch['positive_video'].numpy()
            
            batch_size = interest_representations.shape[0]
            total_samples += batch_size
            
            for i in range(batch_size):
                user_interests = interest_representations[i].numpy()
                positive_video_id = positive_videos[i]
                
                # 检索
                query = user_interests.mean(axis=0, keepdims=True).astype('float32')
                scores, indices = self.faiss_index.search(query, self.config.top_k)
                
                # 计算AP
                relevant_positions = np.where(indices[0] == positive_video_id)[0]
                if len(relevant_positions) > 0:
                    precision_at_k = []
                    for k in range(1, self.config.top_k + 1):
                        relevant_in_top_k = np.sum(indices[0][:k] == positive_video_id)
                        precision_at_k.append(relevant_in_top_k / k)
                    
                    ap = np.mean(precision_at_k) if precision_at_k else 0
                    total_ap += ap
        
        return total_ap / total_samples if total_samples > 0 else 0
    
    def evaluate_classification_metrics(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """评估分类指标"""
        all_labels = []
        all_predictions = []
        
        for batch in test_dataset:
            # 前向传播
            interest_representations = self.model(batch['history_features'], training=False)
            
            # 获取正例和负例
            positive_video = batch['positive_video']
            negative_videos = batch['negative_videos']
            
            # 计算分数
            positive_embedding = self.model.embedding_module.video_id_embedding(positive_video)
            negative_embeddings = self.model.embedding_module.video_id_embedding(negative_videos)
            
            positive_scores = self.model.compute_scores(
                interest_representations, 
                tf.expand_dims(positive_embedding, 1)
            )
            negative_scores = self.model.compute_scores(
                interest_representations,
                negative_embeddings
            )
            
            # 收集预测和标签
            batch_labels = np.concatenate([
                np.ones(positive_scores.shape[0]),
                np.zeros(negative_scores.shape[0] * negative_scores.shape[1])
            ])
            
            batch_predictions = np.concatenate([
                positive_scores.numpy().flatten(),
                negative_scores.numpy().flatten()
            ])
            
            all_labels.extend(batch_labels)
            all_predictions.extend(batch_predictions)
        
        # 计算指标
        auc = roc_auc_score(all_labels, all_predictions)
        ap = average_precision_score(all_labels, all_predictions)
        
        return {'auc': auc, 'average_precision': ap}
    
    def benchmark_latency(self, test_dataset: tf.data.Dataset, 
                          num_requests: int = 1000) -> Dict[str, float]:
        """基准测试：延迟和吞吐量"""
        import time
        
        latencies = []
        
        # 预热
        for batch in test_dataset.take(10):
            _ = self.model(batch['history_features'], training=False)
        
        # 测试
        start_time = time.time()
        
        request_count = 0
        for batch in test_dataset:
            batch_start = time.time()
            
            # 推理
            _ = self.model(batch['history_features'], training=False)
            
            batch_latency = (time.time() - batch_start) * 1000  # 毫秒
            latencies.append(batch_latency)
            
            request_count += batch['history_features']['video_ids'].shape[0]
            if request_count >= num_requests:
                break
        
        total_time = time.time() - start_time
        
        metrics = {
            'total_requests': request_count,
            'total_time_seconds': total_time,
            'throughput_rps': request_count / total_time,
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
        
        return metrics

def main():
    """主评估函数"""
    
    # 配置
    config = KuaiFormerConfig()
    
    # 模型路径
    model_path = "checkpoints/kuaiformer/final_model"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行train.py进行训练")
        return
    
    # 创建评估器
    evaluator = KuaiFormerEvaluator(config, model_path)
    
    # 创建测试数据
    from data_loader import create_synthetic_data
    data_loader, video_features = create_synthetic_data(config, num_users=100, num_videos=1000)
    test_dataset = data_loader.create_training_pairs().take(10)  # 取前10个batch
    
    # 获取视频嵌入
    video_ids = list(video_features.keys())
    video_id_tensor = tf.constant(video_ids, dtype=tf.int32)
    video_embeddings = evaluator.model.embedding_module.video_id_embedding(video_id_tensor)
    video_embeddings = video_embeddings.numpy()
    
    print("开始评估...")
    
    # 检索指标
    print("\n1. 检索指标评估:")
    retrieval_metrics = evaluator.evaluate_retrieval_metrics(test_dataset, video_embeddings)
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 分类指标
    print("\n2. 分类指标评估:")
    classification_metrics = evaluator.evaluate_classification_metrics(test_dataset)
    for metric, value in classification_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 性能基准测试
    print("\n3. 性能基准测试:")
    latency_metrics = evaluator.benchmark_latency(test_dataset, num_requests=100)
    for metric, value in latency_metrics.items():
        if 'latency' in metric:
            print(f"  {metric}: {value:.2f} ms")
        elif 'throughput' in metric:
            print(f"  {metric}: {value:.2f} requests/sec")
        else:
            print(f"  {metric}: {value}")
    
    # 保存评估结果
    results = {
        'retrieval_metrics': retrieval_metrics,
        'classification_metrics': classification_metrics,
        'latency_metrics': latency_metrics,
        'evaluation_time': datetime.now().isoformat()
    }
    
    results_file = 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估结果已保存到: {results_file}")

if __name__ == "__main__":
    main()