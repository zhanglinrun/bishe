"""
潜在扩散对齐模块（简化版）
用于在特征空间上进行噪声预测与对齐，适配 FDAM 联邦扩散对齐训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionAligner(nn.Module):
    """
    轻量级特征空间扩散对齐模块
    - 以特征向量为输入，预测噪声并给出对齐后的特征
    - 设计为简单的 MLP，方便在边缘设备上训练
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, diffusion_steps: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.diffusion_steps = diffusion_steps

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def predict_noise(self, noisy_feat: torch.Tensor) -> torch.Tensor:
        """预测噪声分量"""
        return self.mlp(noisy_feat)

    def denoise(self, noisy_feat: torch.Tensor) -> torch.Tensor:
        """
        给出对齐后的特征（简单一步去噪）
        Args:
            noisy_feat: 加噪后的特征
        Returns:
            对齐/去噪后的特征
        """
        pred_noise = self.predict_noise(noisy_feat)
        aligned = noisy_feat - pred_noise
        return aligned, pred_noise

    def forward(self, noisy_feat: torch.Tensor):
        return self.denoise(noisy_feat)
