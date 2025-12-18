"""
数据处理模块
"""

import pandas as pd
import numpy as np
import jieba
import re
from collections import Counter
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TextDataProcessor:
    """文本数据处理器"""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 200):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        text = self.clean_text(text)
        tokens = list(jieba.cut(text))
        return [token for token in tokens if token.strip()]
    
    def build_vocab(self, texts: List[str]) -> None:
        """构建词汇表"""
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        # 统计词频
        word_counts = Counter(all_tokens)
        
        # 保留高频词
        most_common = word_counts.most_common(self.vocab_size - 4)
        
        # 构建词汇表
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        for word, _ in most_common:
            self.word_to_idx[word] = len(self.word_to_idx)
        
        # 反向映射
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def build_label_vocab(self, labels: List[str]) -> None:
        """构建标签词汇表"""
        unique_labels = list(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def text_to_sequence(self, text: str) -> List[int]:
        """将文本转换为序列"""
        tokens = self.tokenize(text)
        sequence = []
        
        for token in tokens[:self.max_length]:
            idx = self.word_to_idx.get(token, self.word_to_idx['<UNK>'])
            sequence.append(idx)
        
        # 填充到固定长度
        while len(sequence) < self.max_length:
            sequence.append(self.word_to_idx['<PAD>'])
        
        return sequence
    
    def label_to_index(self, label: str) -> int:
        """将标签转换为索引"""
        return self.label_to_idx[label]
    
    def process_data(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """处理数据"""
        # 构建词汇表
        self.build_vocab(texts)
        self.build_label_vocab(labels)
        
        # 转换文本和标签
        sequences = [self.text_to_sequence(text) for text in texts]
        label_indices = [self.label_to_index(label) for label in labels]
        
        return np.array(sequences), np.array(label_indices)


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    data = {
        'text': [
            '这部电影真的很好看，剧情精彩',
            '服务态度很差，食物也不好吃',
            '商品质量不错，物流很快',
            '这家餐厅环境优雅，菜品美味',
            '价格太贵了，性价比不高',
            '客服回复及时，解决问题很专业',
            '产品功能强大，使用体验良好',
            '包装破损，商品有问题',
            '演员演技精湛，值得推荐',
            '等待时间太长，服务效率低',
            '质量很好，超出预期',
            '界面设计不够美观，操作复杂',
            '故事情节引人入胜',
            '售后服务态度恶劣',
            '功能实用，价格合理'
        ],
        'label': [
            '正面', '负面', '正面', '正面', '负面',
            '正面', '正面', '负面', '正面', '负面',
            '正面', '负面', '正面', '负面', '正面'
        ]
    }
    return pd.DataFrame(data)


def load_and_split_data(data_path: str = None, test_size: float = 0.2, random_state: int = 42):
    """加载和分割数据"""
    if data_path is None:
        # 使用示例数据
        df = create_sample_data()
    else:
        df = pd.read_csv(data_path)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state)