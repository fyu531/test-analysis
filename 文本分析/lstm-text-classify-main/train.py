"""
训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import os
from datetime import datetime

from src.data_processor import TextDataProcessor, TextDataset, load_and_split_data
from src.model import create_model
from src.trainer import Trainer


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='LSTM文本分类训练')
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径 (CSV格式)')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'attention_lstm'], help='模型类型')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表大小')
    parser.add_argument('--max_length', type=int, default=200, help='最大序列长度')
    parser.add_argument('--embedding_dim', type=int, default=128, help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--bidirectional', action='store_true', help='是否使用双向LSTM')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载和分割数据
    print("加载数据...")
    train_texts, test_texts, train_labels, test_labels = load_and_split_data(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    print(f"训练样本数: {len(train_texts)}")
    print(f"测试样本数: {len(test_texts)}")
    print(f"标签类别: {set(train_labels)}")
    
    # 数据预处理
    print("预处理数据...")
    processor = TextDataProcessor(
        vocab_size=args.vocab_size,
        max_length=args.max_length
    )
    
    train_sequences, train_label_indices = processor.process_data(train_texts, train_labels)
    test_sequences = [processor.text_to_sequence(text) for text in test_texts]
    test_label_indices = [processor.label_to_index(label) for label in test_labels]
    
    # 保存处理器
    import pickle
    with open(os.path.join(output_dir, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor, f)
    
    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_sequences, train_label_indices)
    test_dataset = TextDataset(test_sequences, test_label_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print(f"创建{args.model_type}模型...")
    num_classes = len(processor.label_to_idx)
    model = create_model(
        vocab_size=len(processor.word_to_idx),
        num_classes=num_classes,
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional if args.model_type == 'lstm' else False
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 训练模型
    print("开始训练...")
    best_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        save_path=os.path.join(output_dir, 'best_model.pth')
    )
    
    # 绘制训练历史
    trainer.plot_training_history(
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # 绘制混淆矩阵
    class_names = list(processor.idx_to_label.values())
    trainer.plot_confusion_matrix(
        metrics=best_metrics,
        class_names=class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # 保存最终结果
    results = {
        'best_accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'num_classes': num_classes,
        'vocab_size': len(processor.word_to_idx),
        'training_samples': len(train_texts),
        'test_samples': len(test_texts)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练完成！结果保存在: {output_dir}")
    print(f"最佳准确率: {results['best_accuracy']:.2f}%")
    print(f"精确率: {results['precision']:.2f}%")
    print(f"召回率: {results['recall']:.2f}%")
    print(f"F1分数: {results['f1']:.2f}%")


if __name__ == '__main__':
    main()