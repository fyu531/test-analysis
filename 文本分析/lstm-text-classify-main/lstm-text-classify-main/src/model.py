"""
LSTM文本分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """LSTM文本分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 其他权重使用正态分布初始化
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """前向传播"""
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 双向LSTM，连接前向和后向的最后隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # 全连接层
        out = F.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class AttentionLSTMClassifier(nn.Module):
    """带注意力机制的LSTM文本分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(AttentionLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力层
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        """前向传播"""
        # 词嵌入
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # 注意力权重
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(2), dim=1
        )  # (batch_size, seq_len)
        
        # 加权平均
        attended_output = torch.sum(
            lstm_out * attention_weights.unsqueeze(2), dim=1
        )  # (batch_size, hidden_dim*2)
        
        # Dropout和分类
        attended_output = self.dropout(attended_output)
        output = self.fc(attended_output)
        
        return output


def create_model(
    vocab_size: int,
    num_classes: int,
    model_type: str = "lstm",
    **kwargs
) -> nn.Module:
    """创建模型"""
    if model_type == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "attention_lstm":
        return AttentionLSTMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")