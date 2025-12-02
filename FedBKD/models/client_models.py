import torch
import torch.nn as nn
from typing import cast


class BaseClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._to_linear = None

    def _get_flattened_size(self, net: nn.Module, shape=(1, 1, 2, 128)):
        with torch.no_grad():
            net = net.to("cpu")
            x = torch.zeros(shape).to("cpu")
            # 类型检查器无法正确推断nn.Module的forward方法
            x = net(x)  # type: ignore[operator]
            self._to_linear = x.view(1, -1).shape[1]
            print(f"[DEBUG] {self.__class__.__name__} flatten size: {self._to_linear}")

    def _calculate_conv_output_size(self, x):
        with torch.no_grad():
            # 类型检查器无法正确推断nn.Module的forward方法
            x = self.conv(x)  # type: ignore[operator]
            return x.view(x.size(0), -1).shape[1]


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 处理stride参数，如果是tuple则取第一个值
        if isinstance(stride, tuple):
            stride = stride[0]
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class LightweightBlock(nn.Module):
    """轻量块 - 深度可分离卷积"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(LightweightBlock, self).__init__()
        # 处理stride参数，如果是tuple则取第一个值
        if isinstance(stride, tuple):
            stride = stride[0]
            
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.depthwise(x)))
        out = torch.relu(self.bn2(self.pointwise(out)))
        return out


class ClientModel1(BaseClientModel):
    """模型1: 通用块 + 残差块6 + 轻量块1 + 分类器"""
    def __init__(self, num_classes=10):
        super(ClientModel1, self).__init__()
        
        # 通用块
        self.common_block = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(15, 2), padding=(7, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差块6
        self.residual_block6 = nn.Sequential(
            ResidualBlock(64, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 轻量块1
        self.lightweight_block1 = nn.Sequential(
            LightweightBlock(256, 64, stride=2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._get_flattened_size(self.features, shape=(1, 1, 2, 128))

    @property
    def features(self):
        return nn.Sequential(self.common_block, self.residual_block6, self.lightweight_block1)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ClientModel2(BaseClientModel):
    """模型2: 通用块 + 残差块3 + 残差块4 + 残差块5 + 残差块6 + 轻量块2 + 分类器"""
    def __init__(self, num_classes=10):
        super(ClientModel2, self).__init__()
        
        # 通用块
        self.common_block = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(15, 2), padding=(7, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差块3
        self.residual_block3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        
        # 残差块4
        self.residual_block4 = nn.Sequential(
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )
        
        # 残差块5
        self.residual_block5 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 残差块6
        self.residual_block6 = nn.Sequential(
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 轻量块2
        self.lightweight_block2 = nn.Sequential(
            LightweightBlock(256, 128, stride=2),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._get_flattened_size(self.features, shape=(1, 1, 2, 128))

    @property
    def features(self):
        return nn.Sequential(
            self.common_block, self.residual_block3, self.residual_block4,
            self.residual_block5, self.residual_block6, self.lightweight_block2
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ClientModel3(BaseClientModel):
    """模型3: 通用块 + 残差块3 + 残差块4 + 残差块6 + 轻量块2 + 分类器"""
    def __init__(self, num_classes=10):
        super(ClientModel3, self).__init__()
        
        # 通用块
        self.common_block = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(15, 2), padding=(7, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差块3
        self.residual_block3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        
        # 残差块4
        self.residual_block4 = nn.Sequential(
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )
        
        # 残差块6
        self.residual_block6 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 轻量块2
        self.lightweight_block2 = nn.Sequential(
            LightweightBlock(256, 128, stride=2),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._get_flattened_size(self.features, shape=(1, 1, 2, 128))

    @property
    def features(self):
        return nn.Sequential(
            self.common_block, self.residual_block3, self.residual_block4,
            self.residual_block6, self.lightweight_block2
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ClientModel4(BaseClientModel):
    """模型4: 通用块 + 自主块 + 残差块1 + 残差块2 + 残差块5 + 残差块6 + 轻量块3 + 分类器"""
    def __init__(self, num_classes=10):
        super(ClientModel4, self).__init__()
        
        # 通用块
        self.common_block = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(15, 2), padding=(7, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 自主块
        self.autonomous_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 残差块1
        self.residual_block1 = nn.Sequential(
            ResidualBlock(256, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        
        # 残差块2
        self.residual_block2 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        
        # 残差块5
        self.residual_block5 = nn.Sequential(
            ResidualBlock(64, 256, stride=1),
            ResidualBlock(256, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 残差块6
        self.residual_block6 = nn.Sequential(
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 轻量块3
        self.lightweight_block3 = nn.Sequential(
            LightweightBlock(256, 256, stride=2),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._get_flattened_size(self.features, shape=(1, 1, 2, 128))

    @property
    def features(self):
        return nn.Sequential(
            self.common_block, self.autonomous_block, self.residual_block1, self.residual_block2,
            self.residual_block5, self.residual_block6, self.lightweight_block3
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ClientModel5(BaseClientModel):
    """模型5: 通用块 + 自主块 + 残差块3 + 残差块4 + 残差块6 + 轻量块1 + 轻量块2 + 分类器"""
    def __init__(self, num_classes=10):
        super(ClientModel5, self).__init__()
        
        # 通用块
        self.common_block = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(15, 2), padding=(7, 0)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 自主块
        self.autonomous_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 残差块3
        self.residual_block3 = nn.Sequential(
            ResidualBlock(256, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        
        # 残差块4
        self.residual_block4 = nn.Sequential(
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )
        
        # 残差块6
        self.residual_block6 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )
        
        # 轻量块1
        self.lightweight_block1 = nn.Sequential(
            LightweightBlock(256, 64, stride=2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 轻量块2
        self.lightweight_block2 = nn.Sequential(
            LightweightBlock(64, 128, stride=2),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._get_flattened_size(self.features, shape=(1, 1, 2, 128))

    @property
    def features(self):
        return nn.Sequential(
            self.common_block, self.autonomous_block, self.residual_block3, self.residual_block4,
            self.residual_block6, self.lightweight_block1, self.lightweight_block2
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# 保留原有的简单模型作为备用
class ClientModelA(BaseClientModel):
    def __init__(self, num_classes=10):
        super(ClientModelA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(2, 5), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1),
            nn.ReLU()
        )
        self._get_flattened_size(self.conv, shape=(1, 1, 2, 128))
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self._to_linear != x.shape[1]:
            self._to_linear = x.shape[1]
            self.fc[0] = nn.Linear(self._to_linear, 256).to(x.device)
        return self.fc(x)


class ClientModelB(BaseClientModel):
    def __init__(self, num_classes=10):
        super(ClientModelB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.ReLU()
        )
        self._get_flattened_size(self.conv, shape=(1, 1, 2, 128))
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self._to_linear != x.shape[1]:
            self._to_linear = x.shape[1]
            self.fc[0] = nn.Linear(self._to_linear, 128).to(x.device)
        return self.fc(x)


class ClientModelC(BaseClientModel):
    def __init__(self, num_classes=10):
        super(ClientModelC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 5), stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3)),
            nn.ReLU()
        )
        self._get_flattened_size(self.conv, shape=(1, 1, 2, 128))
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self._to_linear != x.shape[1]:
            self._to_linear = x.shape[1]
            self.fc[0] = nn.Linear(self._to_linear, 128).to(x.device)
        return self.fc(x)


class ClientModelD(BaseClientModel):
    def __init__(self, num_classes=10):
        super(ClientModelD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 7)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 5)),
            nn.ReLU()
        )
        self._get_flattened_size(self.conv, shape=(1, 1, 2, 128))
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self._to_linear != x.shape[1]:
            self._to_linear = x.shape[1]
            self.fc[0] = nn.Linear(self._to_linear, 256).to(x.device)
        return self.fc(x)


class ClientModelE(BaseClientModel):
    def __init__(self, num_classes=10):
        super(ClientModelE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(2, 6)),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=(1, 4)),
            nn.ReLU()
        )
        self._get_flattened_size(self.conv, shape=(1, 1, 2, 128))
        self.fc = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self._to_linear != x.shape[1]:
            self._to_linear = x.shape[1]
            self.fc[0] = nn.Linear(self._to_linear, 256).to(x.device)
        return self.fc(x)


def get_client_model(model_type="A", num_classes=10):
    """获取客户端模型，支持新的复杂模型和原有的简单模型"""
    if model_type == "1":
        return ClientModel1(num_classes)
    elif model_type == "2":
        return ClientModel2(num_classes)
    elif model_type == "3":
        return ClientModel3(num_classes)
    elif model_type == "4":
        return ClientModel4(num_classes)
    elif model_type == "5":
        return ClientModel5(num_classes)
    elif model_type == "A":
        return ClientModelA(num_classes)
    elif model_type == "B":
        return ClientModelB(num_classes)
    elif model_type == "C":
        return ClientModelC(num_classes)
    elif model_type == "D":
        return ClientModelD(num_classes)
    elif model_type == "E":
        return ClientModelE(num_classes)
    else:
        raise ValueError(f"Unknown model type {model_type}")
