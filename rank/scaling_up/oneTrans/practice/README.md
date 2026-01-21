# OneTrans：工业级推荐系统中融合特征交互与序列建模的统一Transformer模型

## 论文核心思想总结

OneTrans是一种创新的推荐系统排序架构，通过统一的Transformer骨干网络同时执行用户行为序列建模和特征交互。与传统"先编码后交互"流水线不同，OneTrans采用统一分词器将序列特征和非序列特征转换为单一token序列，并通过堆叠的OneTrans块进行联合建模。

### 核心创新点

1. **统一架构**：将序列建模和特征交互整合到单一Transformer骨干网络中
2. **混合参数化**：序列token共享参数，非序列token分配专属参数
3. **金字塔堆叠**：通过逐步裁剪序列token实现计算效率优化
4. **跨请求KV缓存**：复用候选间的用户侧计算，显著降低推理延迟

### 模型架构

- **输入层**：统一分词器处理序列特征和非序列特征
- **OneTrans块**：预归一化因果Transformer，支持混合参数化
- **金字塔调度**：逐层缩减token序列长度
- **任务头**：CTR/CVR等多任务预测

## 模型架构图解说明

```
输入特征
    ↓
统一分词器
    ↓
序列token (S-token) + 非序列token (NS-token)
    ↓
堆叠的OneTrans块（金字塔堆叠）
    ↓
任务专属预测头
    ↓
输出预测（CTR/CVR等）
```

## 实现的关键技术点

### 1. 混合参数化策略
- **序列token**：共享一组Q/K/V和FFN权重
- **非序列token**：每个token分配专属参数

### 2. 因果注意力机制
- 标准因果掩码，NS-token可关注所有S-token历史
- 支持跨序列交互和序列-特征交互

### 3. 金字塔堆叠优化
- 逐步裁剪序列token，保留最具信息量的尾部事件
- 显著降低计算复杂度和内存占用

### 4. 跨请求KV缓存
- 复用候选间的用户侧计算
- 将时间复杂度从O(C)降至O(1)

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练示例
```bash
python train_example.py
```

### 推理示例
```bash
python inference_example.py
```

## 性能基准和验证结果

基于论文实验数据：
- **离线性能**：相较于DCNv2+DIN基准，CTR AUC提升+1.53%，CVR AUC提升+1.14%
- **在线效果**：在工业级A/B测试中，人均GMV提升5.68%
- **计算效率**：通过优化技术，推理延迟降低69%，内存占用降低30%

## 评估训练资源

以样本量10亿为基准，模型每日更新为目标：

### 训练资源需求
- **GPU内存**：约32GB（H100 GPU）
- **训练时间**：约8-12小时（16个H100 GPU并行）
- **存储需求**：约500GB（模型检查点和日志）

### 优化策略
- FlashAttention-2减少注意力I/O
- 混合精度训练（BF16/FP16）
- 激活重计算节省内存

## 部署资源评估

以QPS为10000，返回耗时在500ms内为基准：
- **CPU需求**：16-32核心，支持AVX指令集
- **内存需求**：32-64GB RAM
- **存储需求**：SSD存储，500GB以上
- **网络需求**：10Gbps网络带宽
- **成本估算**：
  - 服务器成本：$2000-4000/月（云服务器）
  - 带宽成本：$500-1000/月（10Gbps）
  - 总成本：$2500-5000/月

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```bash
# 训练示例
cd examples
python train_example.py

# 推理示例
python inference_example.py
```

### 使用真实数据训练
```bash
# 准备数据格式
# train.csv, val.csv, test.csv 包含推荐系统特征数据

# 训练模型
python train.py --config base --epochs 50 --batch_size 2048 --data_dir /path/to/data

# 评估模型
python evaluate.py --model_path ./models/best_model --data_dir /path/to/data
```

### API服务部署
```python
from examples.inference_example import OneTransInferenceEngine

# 加载模型
engine = OneTransInferenceEngine('./models/best_model')

# 在线推理
results = engine.single_inference(user_features, item_features, context_features, sequence_features)
```

## 文件结构说明

```
practice/
├── model.py              # 核心模型实现
├── data_loader.py        # 数据预处理和加载
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── config.py            # 配置文件
├── requirements.txt     # 依赖库列表
├── examples/           # 使用示例
│   ├── train_example.py
│   └── inference_example.py
└── README.md           # 说明文档
```

## 配置说明

模型支持三种预定义配置：
- `small`: 轻量级配置，适合快速实验
- `base`: 基础配置，平衡性能与效率
- `large`: 大型配置，追求最佳效果

自定义配置：
```python
from config import OneTransConfig

config = OneTransConfig()
config.hidden_dim = 512      # 隐藏层维度
config.num_layers = 12       # 层数
config.num_heads = 8         # 注意力头数
config.max_seq_len = 200     # 最大序列长度
```

## 高级功能

### 混合参数化
支持S-token（共享参数）和NS-token（专属参数）的混合参数化设计

### 金字塔堆叠
通过查询集随深度缩减策略提升模型效率

### 系统优化
- FlashAttention-2加速注意力计算
- 混合精度训练
- KV缓存机制
- 激活重计算

## 已知限制和改进方向

### 当前限制
1. **序列长度限制**：最大支持2048个token
2. **特征维度**：非序列特征数量有限制
3. **训练复杂度**：需要大量计算资源

### 改进方向
1. **动态序列长度**：支持可变长度序列处理
2. **稀疏注意力**：进一步优化长序列处理
3. **多模态融合**：支持图像、文本等多模态输入
4. **联邦学习**：支持隐私保护的分布式训练

## 文件结构

```
practice/
├── model.py              # 核心模型实现
├── data_loader.py        # 数据预处理和加载
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── config.py             # 配置文件（超参数等）
├── requirements.txt      # 依赖库列表
└── examples/            # 使用示例
    ├── train_example.py
    └── inference_example.py
```

## 引用

如果您使用本实现，请引用原始论文：
```
赵启等. "OneTrans：工业级推荐系统中融合特征交互与序列建模的统一Transformer模型." 2025.
```