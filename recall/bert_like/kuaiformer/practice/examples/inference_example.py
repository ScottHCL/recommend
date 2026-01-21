"""
KuaiFormer推理示例
演示如何使用训练好的模型进行推荐推理
"""

import os
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import faiss
from typing import List, Dict, Any

from config import KuaiFormerConfig
from model import KuaiFormer
from data_loader import KuaiFormerDataLoader, UserBehaviorSequence

class KuaiFormerInference:
    """KuaiFormer推理器"""
    
    def __init__(self, model_path: str, config: KuaiFormerConfig):
        self.config = config
        self.model = tf.keras.models.load_model(model_path)
        self.faiss_index = None
        self.video_embeddings = None
        
    def build_index(self, video_features: Dict[int, Dict[str, Any]]) -> None:
        """构建FAISS索引"""
        # 获取所有视频ID
        video_ids = list(video_features.keys())
        
        # 获取视频嵌入
        video_id_tensor = tf.constant(video_ids, dtype=tf.int32)
        self.video_embeddings = self.model.embedding_module.video_id_embedding(
            video_id_tensor
        ).numpy()
        
        # 构建FAISS索引
        embedding_dim = self.video_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(self.video_embeddings.astype('float32'))
        
        print(f"索引构建完成，包含 {len(video_ids)} 个视频")
    
    def recommend_for_user(self, user_sequence: List[int], 
                          video_features: Dict[int, Dict[str, Any]],
                          top_k: int = 100) -> List[Dict[str, Any]]:
        """为用户生成推荐"""
        
        if self.faiss_index is None:
            self.build_index(video_features)
        
        # 创建用户行为序列
        user_data = {
            'video_ids': np.array(user_sequence),
            'categories': np.array([video_features[vid].get('category_id', 0) for vid in user_sequence]),
            'tags': np.array([video_features[vid].get('tag_id', 0) for vid in user_sequence]),
            'durations': np.array([video_features[vid].get('duration', 30) for vid in user_sequence]),
            'timestamps': np.array([video_features[vid].get('timestamp', 0) for vid in user_sequence])
        }
        
        # 填充序列
        for key in user_data:
            padded = np.pad(
                user_data[key],
                (0, self.config.max_sequence_length - len(user_data[key])),
                mode='constant'
            )
            user_data[key] = tf.expand_dims(padded[:self.config.max_sequence_length], 0)
        
        # 生成兴趣表征
        interest_representations = self.model(user_data, training=False)
        
        # 多兴趣融合
        user_interest = tf.reduce_mean(interest_representations, axis=1).numpy()
        
        # FAISS检索
        scores, indices = self.faiss_index.search(
            user_interest.astype('float32'), 
            top_k
        )
        
        # 生成推荐结果
        recommendations = []
        for i, (score, video_idx) in enumerate(zip(scores[0], indices[0])):
            video_id = list(video_features.keys())[video_idx]
            recommendations.append({
                'rank': i + 1,
                'video_id': video_id,
                'score': float(score),
                'features': video_features[video_id]
            })
        
        return recommendations
    
    def batch_recommend(self, user_sequences: List[List[int]],
                       video_features: Dict[int, Dict[str, Any]],
                       top_k: int = 100) -> List[List[Dict[str, Any]]]:
        """批量推荐"""
        
        all_recommendations = []
        
        for i, sequence in enumerate(user_sequences):
            if i % 100 == 0:
                print(f"处理第 {i}/{len(user_sequences)} 个用户")
            
            recommendations = self.recommend_for_user(sequence, video_features, top_k)
            all_recommendations.append(recommendations)
        
        return all_recommendations

def inference_example():
    """推理示例"""
    
    print("=== KuaiFormer推理示例 ===")
    
    # 配置
    config = KuaiFormerConfig()
    
    # 模型路径（需要先训练模型）
    model_path = "trained_models/kuaiformer_example"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行train_example.py进行训练")
        return
    
    # 创建推理器
    inference_engine = KuaiFormerInference(model_path, config)
    
    # 创建示例视频特征
    video_features = {}
    for i in range(1000):
        video_features[i] = {
            'category_id': i % 10,  # 10个类别
            'tag_id': i % 50,       # 50个标签
            'duration': np.random.uniform(10, 300),
            'timestamp': np.random.randint(0, 1000000),
            'title': f"视频_{i}"
        }
    
    # 示例用户行为序列
    user_sequence = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 用户观看过的视频
    
    print(f"用户历史行为序列: {user_sequence}")
    print("生成推荐...")
    
    # 生成推荐
    recommendations = inference_engine.recommend_for_user(
        user_sequence, video_features, top_k=10
    )
    
    # 输出推荐结果
    print("\nTop-10 推荐结果:")
    print("排名\t视频ID\t分数\t类别\t时长")
    print("-" * 50)
    
    for rec in recommendations:
        print(f"{rec['rank']}\t{rec['video_id']}\t{rec['score']:.4f}\t"
              f"{rec['features']['category_id']}\t{rec['features']['duration']:.1f}s")
    
    return recommendations

def real_time_inference_demo():
    """实时推理演示"""
    
    print("\n=== 实时推理演示 ===")
    
    config = KuaiFormerConfig()
    
    # 模拟实时推荐场景
    class RealTimeRecommender:
        def __init__(self, model_path: str, config: KuaiFormerConfig):
            self.inference_engine = KuaiFormerInference(model_path, config)
            self.user_sessions = {}  # 用户会话存储
            
        def add_user_interaction(self, user_id: int, video_id: int, 
                                video_features: Dict[str, Any]):
            """添加用户交互"""
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            
            # 限制序列长度
            if len(self.user_sessions[user_id]) >= config.max_sequence_length:
                self.user_sessions[user_id].pop(0)
            
            self.user_sessions[user_id].append(video_id)
            
        def get_recommendations(self, user_id: int, video_features: Dict[int, Dict[str, Any]],
                               top_k: int = 10) -> List[Dict[str, Any]]:
            """获取实时推荐"""
            if user_id not in self.user_sessions:
                return []
            
            sequence = self.user_sessions[user_id]
            return self.inference_engine.recommend_for_user(
                sequence, video_features, top_k
            )
    
    # 创建推荐器
    model_path = "trained_models/kuaiformer_example"
    if not os.path.exists(model_path):
        print("请先训练模型")
        return
    
    recommender = RealTimeRecommender(model_path, config)
    
    # 创建视频库
    video_features = {i: {
        'category_id': i % 10,
        'tag_id': i % 50,
        'duration': np.random.uniform(10, 300),
        'timestamp': np.random.randint(0, 1000000)
    } for i in range(1000)}
    
    # 模拟用户交互
    user_id = 123
    watched_videos = [1, 5, 10, 15, 20]
    
    print(f"用户 {user_id} 观看历史: {watched_videos}")
    
    for vid in watched_videos:
        recommender.add_user_interaction(user_id, vid, video_features)
    
    # 获取推荐
    recommendations = recommender.get_recommendations(user_id, video_features)
    
    print("\n实时推荐结果:")
    for rec in recommendations[:5]:  # 只显示前5个
        print(f"推荐视频 {rec['video_id']} (分数: {rec['score']:.4f})")
    
    # 模拟新交互后的推荐更新
    print("\n用户观看新视频后...")
    recommender.add_user_interaction(user_id, 25, video_features)
    
    updated_recommendations = recommender.get_recommendations(user_id, video_features)
    
    print("更新后的推荐:")
    for rec in updated_recommendations[:5]:
        print(f"推荐视频 {rec['video_id']} (分数: {rec['score']:.4f})")

if __name__ == "__main__":
    # 运行推理示例
    recommendations = inference_example()
    
    # 运行实时推理演示
    # real_time_inference_demo()