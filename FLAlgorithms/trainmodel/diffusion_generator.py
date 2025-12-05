"""KCDM: 知识条件扩散生成器，依据 I4D（Improved Distribution Difference Driven Diffusion）思路实现。
核心包含：
- 线性 beta 调度（论文默认 T=1000, beta_min=1e-4, beta_max=2e-2）。
- 时间/调制标签双 MLP 条件嵌入 (Eq.14)。
- 1D 条件噪声预测网络 epsilon_theta (Eq.16)。
- 反向采样公式 (Eq.15)，可选分类器引导。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim=64):
    """时间步正弦嵌入。"""
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    """FiLM 残差块，融合条件向量。"""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels * 2))

    def forward(self, x, cond):
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class ConditionalDenoiser(nn.Module):
    """简化 1D 条件去噪网络 epsilon_theta。"""

    def __init__(self, in_channels, base_channels, cond_dim):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.block1 = FiLMBlock(base_channels, cond_dim)
        self.block2 = FiLMBlock(base_channels, cond_dim)
        self.block3 = FiLMBlock(base_channels, cond_dim)
        self.output_proj = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, cond):
        h = self.input_proj(x)
        h = self.block1(h, cond)
        h = self.block2(h, cond)
        h = self.block3(h, cond)
        return self.output_proj(F.silu(h))


class DiffusionGenerator(nn.Module):
    """KCDM 生成器，支持分类器引导采样。"""

    def __init__(
        self,
        num_classes,
        signal_length,
        in_channels=2,
        base_channels=128,
        time_emb_dim=128,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.signal_length = signal_length
        self.timesteps = timesteps

        # 条件嵌入 K = MLP_t(t) + MLP_m(M)
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.mod_embed = nn.Sequential(
            nn.Embedding(num_classes, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.denoiser = ConditionalDenoiser(in_channels, base_channels, time_emb_dim)

        # 线性 beta 调度
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas)  # 简化 Eq.(15) 中 beta_t

    # === Forward diffusion q(s_t | s_0) ===
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    # 条件向量 K
    def _get_cond(self, t, labels):
        t_emb = sinusoidal_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        m_emb = self.mod_embed(labels)
        return t_emb + m_emb

    # epsilon_theta(st, K)
    def predict_noise(self, x, t, labels):
        cond = self._get_cond(t, labels)
        return self.denoiser(x, cond)

    # 训练损失 L_DM (Eq.16)
    def training_loss(self, x_start, labels):
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.predict_noise(x_noisy, t, labels)
        return F.mse_loss(noise_pred, noise)

    def predict_x0(self, x_noisy, t, noise_pred):
        sqrt_recip_alpha_bar = 1.0 / self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / self.alphas_cumprod[t] - 1).view(-1, 1, 1)
        return sqrt_recip_alpha_bar * x_noisy - sqrt_recipm1_alpha_bar * noise_pred

    # 反向采样 (Eq.15)
    def p_sample(self, x, t, labels, classifier=None, classifier_scale=0.0):
        noise_pred = self.predict_noise(x, t, labels)
        if classifier is not None and classifier_scale > 0:
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in)
            logp = F.log_softmax(logits, dim=1)
            selected = logp[torch.arange(x.size(0), device=x.device), labels].sum()
            grad = torch.autograd.grad(selected, x_in)[0]
            noise_pred = noise_pred - classifier_scale * grad.detach()

        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1)

        x_prev_mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred
        )
        if (t == 0).all():
            return x_prev_mean
        noise = torch.randn_like(x)
        return x_prev_mean + torch.sqrt(beta_t) * noise

    def sample(self, batch_size, labels=None, device=None, classifier=None, classifier_scale=0.0):
        if device is None:
            device = self.betas.device
        if labels is None:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)

        x = torch.randn(batch_size, self.in_channels, self.signal_length, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, labels, classifier=classifier, classifier_scale=classifier_scale)
        return x
