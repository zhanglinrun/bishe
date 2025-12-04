"""扩散生成器实现，用于基于扩散模型的伪数据生成。"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """生成时间步的正弦嵌入。"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # padding when dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    """简单的 FiLM 残差块，融合时间和标签条件。"""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU()
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = self.conv1(self.activation(self.norm1(x)))
        h = h * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        h = self.conv2(self.activation(self.norm2(h)))
        return x + h


class ConditionalDenoiser(nn.Module):
    """简化版 1D 条件去噪网络。"""

    def __init__(self, in_channels: int, base_channels: int, cond_dim: int):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = FiLMBlock(base_channels, cond_dim)
        self.block2 = FiLMBlock(base_channels, cond_dim)
        self.output_proj = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.block1(h, cond)
        h = self.block2(h, cond)
        return self.output_proj(F.silu(h))


class DiffusionGenerator(nn.Module):
    """用于生成 I/Q 序列的条件扩散模型。"""

    def __init__(
        self,
        num_classes: int,
        signal_length: int,
        in_channels: int = 2,
        base_channels: int = 64,  # 修改1：降低通道数 (原128) 以减少参数量和显存
        time_emb_dim: int = 64,
        timesteps: int = 20,      # 修改2：降低默认时间步 (原100) 以支持反向传播
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.signal_length = signal_length
        self.timesteps = timesteps

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        # 注意：这里传入 base_channels
        self.denoiser = ConditionalDenoiser(in_channels, base_channels, time_emb_dim)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _get_condition(self, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        time_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)
        label_emb = self.label_emb(labels)
        return time_emb + label_emb

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self._get_condition(t, labels)
        return self.denoiser(x, cond)

    def sample(self, batch_size: int, labels: Optional[torch.Tensor] = None, device: Optional[str] = None) -> torch.Tensor:
        if device is None:
            device = self.betas.device
        if labels is None:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)

        x = torch.randn(batch_size, self.in_channels, self.signal_length, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            noise_pred = self.predict_noise(x, t, labels)
            beta_t = self.betas[step]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[step]
            x = self.sqrt_recip_alphas[step] * (x - beta_t / sqrt_one_minus_alpha * noise_pred)
            if step > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = x + sigma * noise
        return x