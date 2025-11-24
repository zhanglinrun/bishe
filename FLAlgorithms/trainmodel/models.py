"""
自动调制识别模型定义
包括 CNN1D, ResNet1D 和 MCLDNN 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_AMR(nn.Module):
    """
    1D CNN 用于自动调制识别
    输入: [B, 2, L] (L = 128 或 1024)
    输出: [B, num_classes]
    """
    
    def __init__(self, num_classes=11, signal_length=128):
        super(CNN1D_AMR, self).__init__()
        
        self.num_classes = num_classes
        self.signal_length = signal_length
        
        # 第一个卷积块
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # 计算展平后的维度
        # 经过三次池化，长度变为 L // 8
        flattened_size = 256 * (signal_length // 8)
        
        # 全连接层
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def extract_features(self, x):
        """提取分类前的特征表示"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return x  # [B, 256]

    def classify_from_features(self, features):
        """基于特征向量得到分类 logits"""
        x = self.dropout(features)
        return self.fc2(x)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classify_from_features(feats)


class ResidualBlock1D(nn.Module):
    """1D 残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D_AMR(nn.Module):
    """
    1D ResNet 用于自动调制识别
    输入: [B, 2, L]
    输出: [B, num_classes]
    """
    
    def __init__(self, num_classes=11, signal_length=128):
        super(ResNet1D_AMR, self).__init__()
        
        self.num_classes = num_classes
        self.signal_length = signal_length
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def extract_features(self, x):
        """返回分类前的全局特征"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x  # [B, 256]

    def classify_from_features(self, features):
        return self.fc(features)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classify_from_features(feats)


class MCLDNN_AMR(nn.Module):
    """
    MCLDNN (Multi-Channel Long Short-Term Memory Deep Neural Network) 用于自动调制识别
    架构: CNN提取特征 → LSTM处理时序 → Dense分类
    参考论文: A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition
    
    输入: [B, 2, L] (I/Q两通道)
    输出: [B, num_classes]
    """
    
    def __init__(self, num_classes=11, signal_length=128):
        super(MCLDNN_AMR, self).__init__()
        
        self.num_classes = num_classes
        self.signal_length = signal_length
        
        # CNN层：提取空间特征
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # LSTM层：处理时序依赖
        # 输入: (batch, seq_len, input_size)
        # 输出: (batch, seq_len, hidden_size * 2) 因为是双向LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # 全连接层：分类
        # LSTM输出是双向的，所以维度是 hidden_size * 2 = 256
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def extract_features(self, x):
        """返回 LSTM 之后的融合特征"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = x.permute(0, 2, 1)  # [B, L, 128]
        _, (h_n, _) = self.lstm(x)
        
        forward_hidden = h_n[-2, :, :]  # [B, 128]
        backward_hidden = h_n[-1, :, :]  # [B, 128]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, 256]
        hidden = self.fc1(hidden)
        hidden = F.relu(hidden)
        return hidden  # [B, 128]

    def classify_from_features(self, features):
        x = self.dropout(features)
        return self.fc2(x)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classify_from_features(feats)


def get_model(model_name, num_classes, signal_length):
    """
    获取模型实例
    
    Args:
        model_name: 模型名称 ('CNN1D', 'ResNet1D' 或 'MCLDNN')
        num_classes: 类别数
        signal_length: 信号长度
        
    Returns:
        模型实例
    """
    if model_name == 'CNN1D':
        return CNN1D_AMR(num_classes=num_classes, signal_length=signal_length)
    elif model_name == 'ResNet1D':
        return ResNet1D_AMR(num_classes=num_classes, signal_length=signal_length)
    elif model_name == 'MCLDNN':
        return MCLDNN_AMR(num_classes=num_classes, signal_length=signal_length)
    else:
        raise ValueError(f"未知模型: {model_name}. 支持的模型: CNN1D, ResNet1D, MCLDNN")

