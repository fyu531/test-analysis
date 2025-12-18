# LSTM文本分类项目

这是一个基于PyTorch实现的LSTM文本分类项目，支持中文文本的情感分析和分类任务。

## 项目特性

- 🚀 基于PyTorch框架实现
- 🔧 使用uv进行Python包管理
- 📝 支持中文文本处理（使用jieba分词）
- 🎯 提供标准LSTM和带注意力机制的LSTM模型
- 📊 完整的训练监控和可视化
- 🎮 交互式预测界面
- 📈 详细的模型评估指标

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- uv包管理器

## 快速开始

### 1. 安装依赖

```bash
# 使用uv安装依赖

```

### 2. 快速演示

运行演示脚本快速体验：

```bash
python demo.py
```

### 3. 训练模型

使用示例数据训练：

```bash
python train.py --epochs 10 --batch_size 16
```

使用自定义数据训练：

```bash
python train.py --data_path your_data.csv --epochs 20 --model_type attention_lstm
```

### 4. 预测文本

单个文本预测：

```bash
python predict.py --model_path output/train_20251217_225641/best_model.pth \
                  --processor_path output/train_20251217_225641/processor.pkl \
                  --config_path output/train_20251217_225641/config.json \
                  --text "这个产品质量很好"
```

交互式预测：

```bash
python predict.py --model_path output/train_xxx/best_model.pth \
                  --processor_path output/train_xxx/processor.pkl \
                  --config_path output/train_xxx/config.json \
                  --interactive
```

## 项目结构

```
lstm_text_classify/
├── src/                    # 核心代码
│   ├── __init__.py
│   ├── data_processor.py   # 数据处理
│   ├── model.py           # LSTM模型
│   ├── trainer.py         # 训练器
│   └── predictor.py       # 预测器
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── demo.py                # 演示脚本
├── pyproject.toml         # 项目配置
└── README.md              # 说明文档
```

## 数据格式

训练数据应为CSV格式，包含两列：

| text | label |
|------|-------|
| 这部电影真的很好看 | 正面 |
| 服务态度很差 | 负面 |

## 模型类型

### 1. 标准LSTM (`lstm`)
- 支持单向/双向LSTM
- 可配置层数和隐藏单元数
- 适用于一般的文本分类任务

### 2. 注意力LSTM (`attention_lstm`)
- 集成注意力机制
- 可视化注意力权重
- 更好的长文本处理能力

## 训练参数

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--model_type` | 模型类型 | lstm |
| `--vocab_size` | 词汇表大小 | 10000 |
| `--max_length` | 最大序列长度 | 200 |
| `--embedding_dim` | 词嵌入维度 | 128 |
| `--hidden_dim` | 隐藏层维度 | 128 |
| `--num_layers` | LSTM层数 | 2 |
| `--dropout` | Dropout率 | 0.3 |
| `--batch_size` | 批大小 | 32 |
| `--epochs` | 训练轮数 | 20 |
| `--learning_rate` | 学习率 | 0.001 |

## 输出文件

训练完成后会在`output/`目录下生成：

- `best_model.pth` - 最佳模型权重
- `processor.pkl` - 数据处理器
- `config.json` - 训练配置
- `results.json` - 训练结果
- `training_history.png` - 训练历史图
- `confusion_matrix.png` - 混淆矩阵图

## 模型评估指标

项目提供以下评估指标：

- **准确率 (Accuracy)**: 正确预测的样本比例
- **精确率 (Precision)**: 预测为正类的样本中实际为正类的比例
- **召回率 (Recall)**: 实际为正类的样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均值

## 使用示例

```python
from src.data_processor import TextDataProcessor
from src.model import create_model
from src.predictor import TextClassifier

# 加载模型和处理器
processor = pickle.load(open('processor.pkl', 'rb'))
model = create_model(vocab_size=10000, num_classes=2)
model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])

# 创建分类器
classifier = TextClassifier(model, processor, device)

# 预测文本
result = classifier.predict("这个产品质量很好")
print(f"预测结果: {result['predicted_label']}")
print(f"置信度: {result['confidence']}")
```

## 注意事项

1. 确保输入文本编码为UTF-8
2. 较长的文本会被截断到`max_length`
3. 训练数据量较小时建议减少模型复杂度
4. GPU可显著加速训练过程

## 📚 详细文档

- [混淆矩阵详解](docs/confusion_matrix_guide.md) - 深入了解混淆矩阵的概念、计算和应用
- [模型评估指标详解](docs/evaluation_metrics.md) - 全面了解准确率、精确率、召回率、F1分数等评估指标

## 许可证

MIT License