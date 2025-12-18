"""
文本分类预测器
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from src.data_processor import TextDataProcessor


class TextClassifier:
    """文本分类预测器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        processor: TextDataProcessor,
        device: torch.device
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()
    
    def predict(self, text: str) -> Dict:
        """预测单个文本"""
        # 文本预处理
        sequence = self.processor.text_to_sequence(text)
        
        # 转换为张量
        input_tensor = torch.LongTensor([sequence]).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 获取预测标签
        predicted_label = self.processor.idx_to_label[predicted_class]
        
        # 获取所有类别的概率
        class_probabilities = {}
        for idx, prob in enumerate(probabilities[0]):
            label = self.processor.idx_to_label[idx]
            class_probabilities[label] = prob.item()
        
        return {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': class_probabilities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_with_attention(self, text: str) -> Dict:
        """预测并返回注意力权重（仅适用于注意力模型）"""
        if not hasattr(self.model, 'attention'):
            raise ValueError("模型不支持注意力可视化")
        
        # 文本预处理
        sequence = self.processor.text_to_sequence(text)
        tokens = self.processor.tokenize(text)
        
        # 转换为张量
        input_tensor = torch.LongTensor([sequence]).to(self.device)
        
        # 预测
        with torch.no_grad():
            # 获取词嵌入
            embedded = self.model.embedding(input_tensor)
            
            # LSTM输出
            lstm_out, _ = self.model.lstm(embedded)
            
            # 注意力权重
            attention_weights = torch.softmax(
                self.model.attention(lstm_out).squeeze(2), dim=1
            )
            
            # 最终预测
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 获取预测标签
        predicted_label = self.processor.idx_to_label[predicted_class]
        
        # 处理注意力权重
        attention_weights = attention_weights[0].cpu().numpy()
        
        # 匹配token和注意力权重
        token_attention = []
        for i, token in enumerate(tokens[:len(attention_weights)]):
            if i < len(attention_weights):
                token_attention.append((token, attention_weights[i]))
        
        return {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'attention_weights': token_attention
        }


def evaluate_model(
    classifier: TextClassifier,
    test_texts: List[str],
    test_labels: List[str]
) -> Dict:
    """评估模型性能"""
    from sklearn.metrics import accuracy_score, classification_report
    
    # 批量预测
    predictions = classifier.predict_batch(test_texts)
    predicted_labels = [pred['predicted_label'] for pred in predictions]
    
    # 计算准确率
    accuracy = accuracy_score(test_labels, predicted_labels)
    
    # 生成分类报告
    report = classification_report(
        test_labels,
        predicted_labels,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions
    }


def create_demo_interface():
    """创建简单的交互式演示界面"""
    print("=" * 50)
    print("LSTM文本分类演示")
    print("=" * 50)
    print("输入 'quit' 退出程序")
    print("输入 'batch' 进行批量测试")
    print("-" * 50)
    
    # 这里需要在main.py中实现具体的交互逻辑
    pass


def visualize_attention(result: Dict, save_path: str = None):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt
    
    tokens = [item[0] for item in result['attention_weights']]
    weights = [item[1] for item in result['attention_weights']]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(tokens)), weights, color='skyblue', alpha=0.7)
    
    # 高亮最重要的词
    max_weight_idx = np.argmax(weights)
    bars[max_weight_idx].set_color('red')
    bars[max_weight_idx].set_alpha(1.0)
    
    plt.title(f'注意力权重可视化\n预测: {result["predicted_label"]} (置信度: {result["confidence"]:.3f})')
    plt.xlabel('词语位置')
    plt.ylabel('注意力权重')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印权重排序
    print("\n词语注意力权重排序:")
    sorted_attention = sorted(
        result['attention_weights'],
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (token, weight) in enumerate(sorted_attention[:10]):
        print(f"{i+1}. {token}: {weight:.4f}")