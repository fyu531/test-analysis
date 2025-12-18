"""
预测脚本
"""

import torch
import pickle
import argparse
import json
import os
from src.model import create_model
from src.predictor import TextClassifier, evaluate_model, visualize_attention
from src.trainer import load_model


def main():
    """主预测函数"""
    parser = argparse.ArgumentParser(description='LSTM文本分类预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--processor_path', type=str, required=True, help='处理器文件路径')
    parser.add_argument('--config_path', type=str, required=True, help='配置文件路径')
    parser.add_argument('--text', type=str, default=None, help='要预测的文本')
    parser.add_argument('--interactive', action='store_true', help='交互式预测')
    parser.add_argument('--attention', action='store_true', help='显示注意力权重')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载处理器
    with open(args.processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    # 创建模型
    num_classes = len(processor.label_to_idx)
    model = create_model(
        vocab_size=len(processor.word_to_idx),
        num_classes=num_classes,
        model_type=config['model_type'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config.get('bidirectional', False)
    )
    
    # 加载训练好的模型
    model = load_model(model, args.model_path, device)
    
    # 创建分类器
    classifier = TextClassifier(model, processor, device)
    
    if args.interactive:
        # 交互式预测
        print("=" * 50)
        print("LSTM文本分类预测系统")
        print("=" * 50)
        print("输入 'quit' 退出程序")
        print("-" * 50)
        
        while True:
            text = input("\n请输入要分类的文本: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            try:
                if args.attention and hasattr(model, 'attention'):
                    result = classifier.predict_with_attention(text)
                    print(f"\n预测结果: {result['predicted_label']}")
                    print(f"置信度: {result['confidence']:.3f}")
                    
                    # 可视化注意力权重
                    visualize_attention(result)
                else:
                    result = classifier.predict(text)
                    print(f"\n预测结果: {result['predicted_label']}")
                    print(f"置信度: {result['confidence']:.3f}")
                    print("\n各类别概率:")
                    for label, prob in result['probabilities'].items():
                        print(f"  {label}: {prob:.3f}")
                        
            except Exception as e:
                print(f"预测出错: {e}")
    
    elif args.text:
        # 单个文本预测
        result = classifier.predict(args.text)
        print(f"输入文本: {args.text}")
        print(f"预测结果: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.3f}")
        print("\n各类别概率:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.3f}")
    
    else:
        print("请使用 --text 参数指定文本或使用 --interactive 进行交互式预测")


if __name__ == '__main__':
    main()