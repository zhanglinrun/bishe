import torch
import torch.nn as nn
import torch.nn.functional as F

# 全局统一模型，作为学生模型
class GlobalModel(nn.Module):
    def __init__(self, num_classes=10):  # 支持自定义分类数
        super().__init__()
        self.num_classes = num_classes
        
        # 更简单的特征提取器，避免过拟合
        self.features = nn.Sequential(
            # 第一层卷积块 - 处理IQ信号
            nn.Conv2d(1, 32, kernel_size=(2, 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 第二层卷积块
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            
            # 第三层卷积块
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 更简单的分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def extract_logits(self, x):
        """提取特征用于知识蒸馏"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)  # 返回完整的logits而不是特征


def get_global_model(num_classes=10):  # 将分类数传入
    return GlobalModel(num_classes=num_classes)
