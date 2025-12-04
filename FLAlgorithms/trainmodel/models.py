"""
基线调制识别模型定义：CNN1D、ResNet1D、MCLDNN，并提供 get_model 工厂。
模型需暴露 extract_features / classify_from_features 以兼容对齐/蒸馏流程。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_AMR(nn.Module):
    def __init__(self, num_classes: int, signal_length: int):
        super().__init__()
        self.num_classes = num_classes
        self.signal_length = signal_length

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, signal_length)
            flat_dim = self.feature_extractor(dummy).view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 256)
        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def classify_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classify_from_features(feats)


class BasicBlock1D(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1D_AMR(nn.Module):
    def __init__(self, num_classes: int, signal_length: int):
        super().__init__()
        self.num_classes = num_classes
        self.signal_length = signal_length

        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)

        self.fc1 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, planes: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock1D(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers.append(BasicBlock1D(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.mean(x, dim=2)  # 全局平均池化
        x = F.relu(self.fc1(x))
        return x

    def classify_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classify_from_features(feats)


class MCLDNN_AMR(nn.Module):
    def __init__(self, num_classes: int, signal_length: int, lstm_hidden: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.signal_length = signal_length

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(lstm_hidden * 2, 256)
        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # B, T, C
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, H)
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        feats = torch.cat([forward_h, backward_h], dim=1)
        feats = F.relu(self.fc1(feats))
        return feats

    def classify_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classify_from_features(feats)


def get_model(name: str, num_classes: int, signal_length: int) -> nn.Module:
    """
    根据名称返回模型实例。
    """
    name = name.lower()
    if name == "cnn1d":
        return CNN1D_AMR(num_classes, signal_length)
    if name == "resnet1d":
        return ResNet1D_AMR(num_classes, signal_length)
    if name == "mcldnn":
        return MCLDNN_AMR(num_classes, signal_length)
    raise ValueError(f"Unknown model name: {name}")
