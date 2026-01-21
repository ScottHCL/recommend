"""
KuaiFormer数据加载器
处理用户行为序列数据，支持批处理和负采样
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import KuaiFormerConfig

class VideoDataset:
    """视频数据集类，管理视频特征和元数据"""
    
    def __init__(self, config: KuaiFormerConfig):
        self.config = config
        self.video_features = {}
        self.popularity_stats = {}
        
    def add_video(self, video_id: int, features: Dict[str, any]):
        """添加视频特征"""
        self.video_features[video_id] = features
        
    def update_popularity(self, video_id: int, count: int):
        """更新视频流行度统计"""
        self.popularity_stats[video_id] = count
        
    def get_video_popularity(self, video_id: int) -> float:
        """获取视频流行度"""
        return self.popularity_stats.get(video_id, 1.0)

class UserBehaviorSequence:
    """用户行为序列数据"""
    
    def __init__(self, user_id: int, video_sequence: List[int], 
                 timestamps: List[float], interactions: List[Dict[str, any]]):
        self.user_id = user_id
        self.video_sequence = video_sequence
        self.timestamps = timestamps
        self.interactions = interactions
        
    def get_sequence_features(self, video_dataset: VideoDataset) -> Dict[str, np.ndarray]:
        """获取序列特征"""
        features = {
            'video_ids': [],
            'categories': [],
            'tags': [],
            'durations': [],
            'timestamps': []
        }
        
        for video_id in self.video_sequence:
            if video_id in video_dataset.video_features:
                video_feat = video_dataset.video_features[video_id]
                features['video_ids'].append(video_id)
                features['categories'].append(video_feat.get('category_id', 0))
                features['tags'].append(video_feat.get('tag_id', 0))
                features['durations'].append(video_feat.get('duration', 30))
                # 使用相对时间戳
                features['timestamps'].append(video_feat.get('timestamp', 0))
        
        # 转换为numpy数组
        for key in features:
            features[key] = np.array(features[key])
            
        return features

class KuaiFormerDataLoader:
    """KuaiFormer数据加载器"""
    
    def __init__(self, config: KuaiFormerConfig):
        self.config = config
        self.video_dataset = VideoDataset(config)
        self.user_sequences = []
        self.negative_sampler = NegativeSampler(config)
        
    def load_user_data(self, user_data: List[Dict[str, any]]):
        """加载用户行为数据"""
        for user_record in user_data:
            sequence = UserBehaviorSequence(
                user_id=user_record['user_id'],
                video_sequence=user_record['video_sequence'],
                timestamps=user_record['timestamps'],
                interactions=user_record.get('interactions', [])
            )
            self.user_sequences.append(sequence)
    
    def load_video_features(self, video_features: Dict[int, Dict[str, any]]):
        """加载视频特征"""
        for video_id, features in video_features.items():
            self.video_dataset.add_video(video_id, features)
    
    def create_training_pairs(self) -> tf.data.Dataset:
        """创建训练数据对"""
        def generator():
            for sequence in self.user_sequences:
                if len(sequence.video_sequence) < 2:
                    continue
                
                if not self.config.use_causal_mask:
                    # 模式1：有明确标签（预测下一个video）
                    for i in range(1, len(sequence.video_sequence)):
                        # 历史序列
                        history_videos = sequence.video_sequence[:i]
                        
                        # 正例视频
                        positive_video = sequence.video_sequence[i]
                        
                        # 负采样
                        negative_videos = self.negative_sampler.sample_negatives(
                            positive_video, self.config.batch_size - 1
                        )
                        
                        # 获取特征
                        history_features = sequence.get_sequence_features(
                            self.video_dataset
                        )
                        
                        # 获取流行度
                        popularity = [self.video_dataset.get_video_popularity(positive_video)]
                        popularity.extend([
                            self.video_dataset.get_video_popularity(neg) 
                            for neg in negative_videos
                        ])
                        
                        yield {
                            'history_features': history_features,
                            'positive_video': positive_video,
                            'negative_videos': negative_videos,
                            'popularity': popularity
                        }
                else:
                    # 模式2：只有行为序列（预测序列中每个位置后面的候选video）
                    for i in range(len(sequence.video_sequence) - 1):  # 最后一个位置不需要预测
                        # 完整序列（使用因果掩码，只能看到当前位置及之前的信息）
                        full_videos = sequence.video_sequence
                        
                        # 正例（下一个视频）
                        positive_video = sequence.video_sequence[i + 1]
                        
                        # 自监督负采样策略：
                        # 1. 排除正例视频
                        # 2. 排除序列中已经出现的视频（避免信息泄露）
                        # 3. 从剩余视频中采样
                        seen_videos = set(sequence.video_sequence[:i + 2])  # 包括当前和正例
                        
                        # 负采样（排除已见视频）
                        negative_videos = self.negative_sampler.sample_negatives_with_exclusion(
                            positive_video, 
                            self.config.batch_size - 1,
                            exclude_videos=list(seen_videos)
                        )
                        
                        # 获取特征
                        full_features = sequence.get_sequence_features(
                            self.video_dataset
                        )
                        
                        # 获取流行度
                        popularity = [self.video_dataset.get_video_popularity(positive_video)]
                        popularity.extend([
                            self.video_dataset.get_video_popularity(neg) 
                            for neg in negative_videos
                        ])
                        
                        yield {
                            'history_features': full_features,
                            'positive_video': positive_video,
                            'negative_videos': negative_videos,
                            'popularity': popularity,
                            'target_position': i  # 目标预测位置
                        }
        
        # 创建TensorFlow数据集
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                'history_features': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'positive_video': tf.TensorSpec(shape=(), dtype=tf.int32),
                'negative_videos': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'popularity': tf.TensorSpec(shape=(None,), dtype=tf.float32)
            }
        )
        
        return dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
    
    def create_inference_data(self, user_id: int) -> Optional[Dict[str, tf.Tensor]]:
        """创建推理数据"""
        user_sequence = None
        for seq in self.user_sequences:
            if seq.user_id == user_id:
                user_sequence = seq
                break
        
        if not user_sequence:
            return None
        
        features = user_sequence.get_sequence_features(self.video_dataset)
        
        # 转换为模型输入格式
        model_inputs = {}
        for key, value in features.items():
            # 填充到最大序列长度
            padded = tf.pad(
                value,
                [[0, self.config.max_sequence_length - len(value)]],
                mode='CONSTANT'
            )
            model_inputs[key] = tf.expand_dims(padded[:self.config.max_sequence_length], 0)
        
        return model_inputs

class NegativeSampler:
    """负采样器，支持均匀采样和流行度感知采样"""
    
    def __init__(self, config: KuaiFormerConfig):
        self.config = config
        self.video_popularity = {}
        
    def update_popularity_distribution(self, popularity_stats: Dict[int, float]):
        """更新流行度分布"""
        self.video_popularity = popularity_stats
        
    def sample_negatives(self, positive_video: int, num_negatives: int) -> List[int]:
        """负采样"""
        # 获取所有候选视频ID
        all_videos = list(self.video_popularity.keys())
        
        if positive_video in all_videos:
            all_videos.remove(positive_video)
        
        # 基于流行度的采样概率
        if self.video_popularity:
            probabilities = []
            for video in all_videos:
                prob = self.video_popularity.get(video, 1.0)
                probabilities.append(prob)
            
            # 归一化概率
            probabilities = np.array(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            
            # 采样
            negatives = np.random.choice(
                all_videos, 
                size=min(num_negatives, len(all_videos)),
                p=probabilities,
                replace=False
            )
        else:
            # 均匀采样
            negatives = np.random.choice(
                all_videos,
                size=min(num_negatives, len(all_videos)),
                replace=False
            )
        
        return negatives.tolist()
    
    def sample_negatives_with_exclusion(self, positive_video: int, num_negatives: int, 
                                      exclude_videos: List[int]) -> List[int]:
        """带排除列表的负采样（用于自监督模式）"""
        # 获取所有候选视频ID
        all_videos = list(self.video_popularity.keys())
        
        # 排除正例和已见视频
        exclude_set = set(exclude_videos)
        if positive_video in exclude_set:
            exclude_set.remove(positive_video)  # 确保正例被排除
        
        candidate_videos = [v for v in all_videos if v not in exclude_set]
        
        if not candidate_videos:
            # 如果没有候选视频，返回空列表
            return []
        
        # 基于流行度的采样概率
        if self.video_popularity:
            probabilities = []
            for video in candidate_videos:
                prob = self.video_popularity.get(video, 1.0)
                probabilities.append(prob)
            
            # 归一化概率
            probabilities = np.array(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            
            # 采样
            negatives = np.random.choice(
                candidate_videos, 
                size=min(num_negatives, len(candidate_videos)),
                p=probabilities,
                replace=False
            )
        else:
            # 均匀采样
            negatives = np.random.choice(
                candidate_videos,
                size=min(num_negatives, len(candidate_videos)),
                replace=False
            )
        
        return negatives.tolist()

def create_synthetic_data(config: KuaiFormerConfig, num_users: int = 1000, 
                         num_videos: int = 10000) -> Tuple[KuaiFormerDataLoader, Dict]:
    """创建合成数据用于测试"""
    data_loader = KuaiFormerDataLoader(config)
    
    # 创建视频特征
    video_features = {}
    for i in range(num_videos):
        video_features[i] = {
            'category_id': np.random.randint(0, config.category_vocab_size),
            'tag_id': np.random.randint(0, config.tag_vocab_size),
            'duration': np.random.uniform(10, 300),
            'timestamp': np.random.randint(0, 1000000)
        }
    
    data_loader.load_video_features(video_features)
    
    # 创建用户行为序列
    user_data = []
    for user_id in range(num_users):
        # 每个用户有10-50个观看记录
        seq_length = np.random.randint(10, 50)
        video_sequence = np.random.choice(
            list(video_features.keys()), 
            size=seq_length, 
            replace=False
        ).tolist()
        
        timestamps = np.sort(np.random.randint(0, 1000000, size=seq_length)).tolist()
        
        user_data.append({
            'user_id': user_id,
            'video_sequence': video_sequence,
            'timestamps': timestamps,
            'interactions': [{'type': 'watch'} for _ in range(seq_length)]
        })
    
    data_loader.load_user_data(user_data)
    
    # 更新流行度统计
    popularity_stats = {}
    for video_id in video_features.keys():
        popularity_stats[video_id] = np.random.poisson(10) + 1
    
    data_loader.negative_sampler.update_popularity_distribution(popularity_stats)
    
    return data_loader, video_features