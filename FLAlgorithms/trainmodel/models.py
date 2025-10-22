"""
自动调制识别模型定义
包括 CNN1D 和 ResNet1D 架构
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
    
    def forward(self, x):
        # 卷积块 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 卷积块 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 卷积块 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


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
    
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x


def get_model(model_name, num_classes, signal_length):
    """
    获取模型实例
    
    Args:
        model_name: 模型名称 ('CNN1D' 或 'ResNet1D')
        num_classes: 类别数
        signal_length: 信号长度
        
    Returns:
        模型实例
    """
    if model_name == 'CNN1D':
        return CNN1D_AMR(num_classes=num_classes, signal_length=signal_length)
    elif model_name == 'ResNet1D':
        return ResNet1D_AMR(num_classes=num_classes, signal_length=signal_length)
    else:
        raise ValueError(f"未知模型: {model_name}. 支持的模型: CNN1D, ResNet1D")

