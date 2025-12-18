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