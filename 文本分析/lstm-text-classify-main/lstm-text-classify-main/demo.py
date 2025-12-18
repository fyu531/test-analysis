"""
演示脚本 - 快速体验LSTM文本分类
"""

import torch
import os
import json
import pickle
from src.data_processor import TextDataProcessor, TextDataset, load_and_split_data
from src.model import create_model
from src.trainer import Trainer
from src.predictor import TextClassifier
from torch.utils.data import DataLoader


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("LSTM文本分类演示")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 创建示例数据并训练模型
    print("\n1. 准备示例数据...")
    train_texts, test_texts, train_labels, test_labels = load_and_split_data()
    
    print(f"训练样本: {len(train_texts)}")
    print(f"测试样本: {len(test_texts)}")
    print(f"标签类别: {set(train_labels)}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    processor = TextDataProcessor(vocab_size=5000, max_length=100)
    train_sequences, train_label_indices = processor.process_data(train_texts, train_labels)
    test_sequences = [processor.text_to_sequence(text) for text in test_texts]
    test_label_indices = [processor.label_to_index(label) for label in test_labels]
    
    # 创建数据集
    train_dataset = TextDataset(train_sequences, train_label_indices)
    test_dataset = TextDataset(test_sequences, test_label_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 3. 创建和训练模型
    print("\n3. 创建LSTM模型...")
    model = create_model(
        vocab_size=len(processor.word_to_idx),
        num_classes=len(processor.label_to_idx),
        model_type="lstm",
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.2,
        bidirectional=True
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 4. 训练模型
    print("\n4. 开始训练模型...")
    trainer = Trainer(model, device, learning_rate=0.01)
    
    best_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=5,
        save_path="demo_model.pth"
    )
    
    # 5. 创建分类器进行预测
    print("\n5. 创建文本分类器...")
    classifier = TextClassifier(model, processor, device)
    
    # 6. 测试预测
    print("\n6. 测试文本分类...")
    test_examples = [
        "这个产品质量很好，非常满意",
        "服务态度太差了，完全不推荐",
        "价格合理，性价比不错",
        "包装破损，商品有问题",
        "功能强大，使用体验很棒"
    ]
    
    for text in test_examples:
        result = classifier.predict(text)
        print(f"文本: {text}")
        print(f"预测: {result['predicted_label']} (置信度: {result['confidence']:.3f})")
        print("-" * 50)
    
    # 7. 交互式预测
    print("\n7. 交互式预测 (输入 'quit' 退出):")
    while True:
        user_input = input("\n请输入要分类的文本: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        try:
            result = classifier.predict(user_input)
            print(f"预测结果: {result['predicted_label']}")
            print(f"置信度: {result['confidence']:.3f}")
            print("各类别概率:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.3f}")
        except Exception as e:
            print(f"预测出错: {e}")
    
    # 清理临时文件
    if os.path.exists("demo_model.pth"):
        os.remove("demo_model.pth")
    
    print("\n演示结束！")


if __name__ == '__main__':
    run_demo()