"""
升级版扩散生成器 (Based on 1D U-Net & AdaGN & Classifier-Free Guidance)
改进点：
1. [FIX] 修复 UpSample 通道数不匹配导致的 RuntimeError。
2. [Upgrade] 将条件注入机制升级为 AdaGN (Adaptive Group Norm)，显著提升类别控制力和 CFG 效果。
3. 优化了 ResidualBlock 和 AttentionBlock 的结构。
4. [New] sample 函数增加 dynamic_thresholding，解决 CFG 导致的信号幅度爆炸问题。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    """
    创建正弦时间步嵌入
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

class AdaGN1D(nn.Module):
    """
    Adaptive Group Normalization (AdaGN)
    利用条件嵌入 (time + class) 自适应地预测 GroupNorm 的 scale 和 shift。
    这是提升条件生成效果的关键组件。
    """
    def __init__(self, channels, cond_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False) # 关闭自带的 affine
        self.proj = nn.Linear(cond_dim, channels * 2) # 预测 scale 和 shift
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, cond_emb):
        # x: [B, C, L]
        # cond_emb: [B, cond_dim]
        scale, shift = self.proj(cond_emb).chunk(2, dim=1)
        x = self.norm(x)
        # [B, C, 1] for broadcasting
        scale = scale.unsqueeze(-1) + 1.0 # 初始 scale 为 1
        shift = shift.unsqueeze(-1)
        return x * scale + shift

class ResidualBlock1D(nn.Module):
    """集成 AdaGN 的 1D 残差块"""
    def __init__(self, in_channels, out_channels, cond_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 第一层卷积
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 第二层卷积 (使用 AdaGN 注入条件信息)
        self.adagn = AdaGN1D(out_channels, cond_dim)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 快捷连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cond_emb):
        # Block 1
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        # Block 2 (Condition Injection via AdaGN)
        h = self.adagn(h, cond_emb)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class AttentionBlock1D(nn.Module):
    """1D 自注意力机制"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
    
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1) # [B, L, C]
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1) # [B, C, L]
        return x + h

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class DiffusionGenerator(nn.Module):
    """
    基于 U-Net 1D 的条件扩散模型
    """
    def __init__(
        self,
        num_classes,
        signal_length=128,
        in_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4), 
        dropout=0.1,
        timesteps=1000,
        beta_schedule="cosine"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.timesteps = timesteps
        self.base_channels = base_channels

        # Beta Schedule
        if beta_schedule == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Time & Class Embedding
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # U-Net Encoder
        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.down_res_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        # Level 0 (64)
        self.down_res_blocks.append(nn.ModuleList([
            ResidualBlock1D(base_channels, base_channels, time_emb_dim, dropout),
            ResidualBlock1D(base_channels, base_channels, time_emb_dim, dropout)
        ]))
        self.down_samples.append(Downsample1D(base_channels))
        
        # Level 1 (128)
        self.down_res_blocks.append(nn.ModuleList([
            ResidualBlock1D(base_channels, base_channels*2, time_emb_dim, dropout),
            ResidualBlock1D(base_channels*2, base_channels*2, time_emb_dim, dropout)
        ]))
        self.down_samples.append(Downsample1D(base_channels*2))

        # Level 2 (256)
        self.down_res_blocks.append(nn.ModuleList([
            ResidualBlock1D(base_channels*2, base_channels*4, time_emb_dim, dropout),
            ResidualBlock1D(base_channels*4, base_channels*4, time_emb_dim, dropout)
        ]))
        # No downsample at bottom

        curr_ch = base_channels * 4 # 256
        
        # Bottleneck
        self.mid_block1 = ResidualBlock1D(curr_ch, curr_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock1D(curr_ch)
        self.mid_block2 = ResidualBlock1D(curr_ch, curr_ch, time_emb_dim, dropout)

        # U-Net Decoder
        self.up_samples = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()

        # Up Level 1 (256 -> 128)
        self.up_samples.append(Upsample1D(base_channels * 4)) 
        self.up_res_blocks.append(nn.ModuleList([
            # Concat: Upsampled(256) + Skip_L1(128) = 384
            ResidualBlock1D(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim, dropout),
            ResidualBlock1D(base_channels * 2, base_channels * 2, time_emb_dim, dropout)
        ]))

        # Up Level 0 (128 -> 64)
        self.up_samples.append(Upsample1D(base_channels * 2)) 
        self.up_res_blocks.append(nn.ModuleList([
            # Concat: Upsampled(128) + Skip_L0(64) = 192
            ResidualBlock1D(base_channels * 2 + base_channels, base_channels, time_emb_dim, dropout),
            ResidualBlock1D(base_channels, base_channels, time_emb_dim, dropout)
        ]))

        # Final
        self.final_res_block = ResidualBlock1D(base_channels, base_channels, time_emb_dim, dropout)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def _get_cond_embedding(self, t, labels):
        t_emb = get_timestep_embedding(t, self.base_channels).to(t.device)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(labels)
        return t_emb + c_emb

    def forward(self, x, t, labels):
        cond = self._get_cond_embedding(t, labels)
        
        skips = []
        h = self.input_conv(x)
        
        # --- Down ---
        # Level 0
        for block in self.down_res_blocks[0]:
            h = block(h, cond)
        skips.append(h) # Save L0
        h = self.down_samples[0](h)
        
        # Level 1
        for block in self.down_res_blocks[1]:
            h = block(h, cond)
        skips.append(h) # Save L1
        h = self.down_samples[1](h)
        
        # Level 2
        for block in self.down_res_blocks[2]:
            h = block(h, cond)
        # L2 output feeds to Mid
        
        # --- Mid ---
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        
        # --- Up ---
        # Up 1
        h = self.up_samples[0](h) 
        skip = skips.pop() 
        if h.shape[2] != skip.shape[2]:
            h = F.interpolate(h, size=skip.shape[2], mode='linear', align_corners=False)
        h = torch.cat([h, skip], dim=1)
        for block in self.up_res_blocks[0]:
            h = block(h, cond)
        
        # Up 0
        h = self.up_samples[1](h)
        skip = skips.pop()
        if h.shape[2] != skip.shape[2]:
            h = F.interpolate(h, size=skip.shape[2], mode='linear', align_corners=False)
        h = torch.cat([h, skip], dim=1)
        for block in self.up_res_blocks[1]:
            h = block(h, cond)
        
        # Final
        h = self.final_conv(h)
        return h

    def predict_noise(self, x, t, labels):
        return self.forward(x, t, labels)

    def training_loss(self, x_start, labels, cfg_prob=0.1):
        B = x_start.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        
        x_noisy = self.sqrt_alphas_cumprod[t, None, None] * x_start + \
                  self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        
        # CFG Training: Randomly drop labels
        if cfg_prob > 0:
            mask = torch.rand(B, device=x_start.device) < cfg_prob
            labels_in = labels.clone()
            labels_in[mask] = self.num_classes # Null token
        else:
            labels_in = labels

        noise_pred = self.predict_noise(x_noisy, t, labels_in)
        return F.mse_loss(noise_pred, noise)

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self.sqrt_recip_alphas_cumprod[t, None, None] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t, None, None] * eps
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t, None, None] * x_start +
            self.posterior_mean_coef2[t, None, None] * x_t
        )
        posterior_variance = self.posterior_variance[t, None, None]
        return posterior_mean, posterior_variance

    @torch.no_grad()
    def sample(self, batch_size, labels=None, device=None, guidance_scale=3.0, dynamic_threshold=True):
        """
        采样函数，支持 Classifier-Free Guidance 和 Dynamic Thresholding
        :param dynamic_threshold: 是否启用动态阈值截断，防止 CFG 导致的幅度爆炸
        """
        if device is None:
            device = self.betas.device
        if labels is None:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
            
        x = torch.randn(batch_size, self.in_channels, self.signal_length, device=device)
        null_labels = torch.full_like(labels, self.num_classes)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 1. 预测噪声
            if guidance_scale > 1.0:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)
                l_in = torch.cat([labels, null_labels], dim=0)
                
                noise_pred_all = self.predict_noise(x_in, t_in, l_in)
                noise_pred_cond, noise_pred_uncond = noise_pred_all.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.predict_noise(x, t, labels)

            # 2. 预测 x0 (x_start)
            # x0 = (xt - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
            x_recon = self.predict_xstart_from_eps(x, t, noise_pred)

            # 3. Dynamic Thresholding (关键步骤)
            # 限制 x0 的幅度，防止 CFG 导致信号数值爆炸
            # 对于标准化后的 IQ 信号，大部分数值在 [-3, 3] 之间
            if dynamic_threshold and guidance_scale > 1.0:
                # 简单截断通常足够有效，或者使用分位数截断
                x_recon = torch.clamp(x_recon, -3.0, 3.0)

            # 4. 计算均值 (q_posterior_mean)
            # mean = (1/sqrt(alpha)) * (x - (1-alpha)/sqrt(1-alpha_bar) * eps)
            # 或者使用 x0 形式的公式，这里沿用之前的 eps 形式公式
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            # 使用修正后的 x0 重新计算有效噪声 (implicit)，或者直接用 clamping 后的 x0 算均值
            # 为了保持代码简洁，我们直接用 clamping 后的 x0 来反推均值
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1}) * eps
            # 但 DDPM 标准采样是用 eps 算 mean。
            # 为了让 dynamic threshold 生效，我们需要基于 clamp 后的 x_recon 算出新的 mean
            
            model_mean = (
                (torch.sqrt(alpha_cumprod) * beta / (1 - alpha_cumprod)) * x_recon +
                ((1 - self.alphas_cumprod[i-1]) * torch.sqrt(alpha) / (1 - alpha_cumprod)) * x 
                if i > 0 else x_recon # i=0 时直接就是 x_recon
            )
            # 上面的公式比较繁琐，我们用标准的 DDPM 均值公式
            # mu = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * eps)
            # 但如果我们想利用 thresholding，最好用 x_recon 来表示 mu
            # mu = (sqrt(alpha_bar_prev) * beta / (1-alpha_bar)) * x0 + (sqrt(alpha) * (1-alpha_bar_prev) / (1-alpha_bar)) * xt
            
            if i > 0:
                alpha_bar_prev = self.alphas_cumprod[i-1]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            posterior_mean_coef1 = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_cumprod)
            posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
            
            model_mean = posterior_mean_coef1 * x_recon + posterior_mean_coef2 * x

            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta) # 或者使用后验方差
            else:
                noise = 0.0
                sigma = 0.0
            
            x = model_mean + sigma * noise
            
        return x