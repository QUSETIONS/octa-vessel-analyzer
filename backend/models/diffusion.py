# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 3D Diffusion 模型

================================================================================
扩散模型数学原理
================================================================================

扩散模型（Diffusion Models）是一类基于随机过程的生成模型，包含两个过程：

【1. 前向扩散过程（Forward Diffusion Process）】

逐步向数据添加高斯噪声，直到数据变成纯噪声：

\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})
\]

其中 β_t 是噪声调度参数，t ∈ {1, 2, ..., T}。

可以直接从 x_0 采样任意时刻 x_t：

\[
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})
\]

其中：
- α_t = 1 - β_t
- ᾱ_t = ∏_{s=1}^t α_s

【2. 反向去噪过程（Reverse Denoising Process）】

学习从噪声恢复数据的过程：

\[
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\]

网络预测噪声 ε_θ(x_t, t)，均值计算为：

\[
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
\]

【3. 训练目标】

简化的损失函数（预测噪声）：

\[
\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
\]

【4. 条件扩散（Conditional Diffusion）】

本系统使用条件扩散模型，条件是带拖尾的 OCTA 数据：

\[
p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t, c))
\]

其中 c 是条件信息（带拖尾的 OCTA 体数据）。

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from functools import partial
from tqdm import tqdm


# ============================================================================
# 噪声调度器
# ============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    线性噪声调度
    
    β_t 从 beta_start 线性增加到 beta_end
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    余弦噪声调度（推荐）
    
    相比线性调度，余弦调度在训练初期更平滑，生成质量更好
    
    公式：
    ᾱ_t = cos²((t/T + s) / (1 + s) × π/2)
    β_t = 1 - ᾱ_t / ᾱ_{t-1}
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    二次噪声调度
    """
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


# ============================================================================
# 时间嵌入
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置嵌入
    
    将时间步 t 编码为高维向量，使网络能够区分不同的噪声级别
    
    使用 Transformer 中的位置编码方法：
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


# ============================================================================
# 3D Diffusion U-Net 组件
# ============================================================================

class ResBlock3D(nn.Module):
    """
    残差块，带时间嵌入
    
    结构：
    x → Conv → Norm → Act → Conv → Norm → + → Act → out
    ↓                                       ↑
    └─────────── residual ──────────────────┘
    
    时间嵌入通过 MLP 映射后与特征相加
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: 特征 (B, C, D, H, W)
            t: 时间嵌入 (B, time_emb_dim)
        """
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None, None]
        
        h = self.block2(h)
        
        return h + self.shortcut(x)


class SelfAttention3D(nn.Module):
    """
    3D 自注意力层
    
    在特征图上计算全局注意力，捕获长程依赖
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # QKV 投影
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为序列形式
        q = q.view(B, self.num_heads, C // self.num_heads, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)
        
        # 注意力计算
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhcd,bhce->bhde', q, k) * scale, dim=-1)
        out = torch.einsum('bhde,bhce->bhcd', attn, v)
        
        # 恢复形状
        out = out.view(B, C, D, H, W)
        out = self.proj(out)
        
        return x + out


class DownBlock3D(nn.Module):
    """下采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_attention: bool = False,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.res_block = ResBlock3D(in_channels, out_channels, time_emb_dim)
        
        if use_attention:
            self.attention = SelfAttention3D(out_channels, num_heads)
        else:
            self.attention = None
        
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res_block(x, t)
        
        if self.attention is not None:
            h = self.attention(h)
        
        skip = h
        h = self.downsample(h)
        
        return h, skip


class UpBlock3D(nn.Module):
    """上采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_attention: bool = False,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.res_block = ResBlock3D(in_channels + out_channels, out_channels, time_emb_dim)
        
        if use_attention:
            self.attention = SelfAttention3D(out_channels, num_heads)
        else:
            self.attention = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        
        # 处理尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x, t)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


# ============================================================================
# 条件 3D Diffusion U-Net
# ============================================================================

class ConditionalUNet3D(nn.Module):
    """
    条件 3D Diffusion U-Net
    
    用于从噪声预测目标数据，条件是带拖尾的 OCTA 体数据
    
    输入：
        x_t: 噪声图像 (B, 1, D, H, W)
        t: 时间步 (B,)
        condition: 条件（带拖尾的 OCTA）(B, 1, D, H, W)
    
    输出：
        预测的噪声 ε (B, 1, D, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        condition_channels: int = 1,
        base_features: int = 64,
        time_emb_dim: int = 256,
        num_layers: int = 4,
        attention_layers: List[int] = [2, 3],  # 在哪些层使用注意力
        num_heads: int = 4
    ):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 输入卷积（合并噪声图像和条件）
        self.input_conv = nn.Conv3d(
            in_channels + condition_channels, 
            base_features, 
            kernel_size=3, 
            padding=1
        )
        
        # 特征通道数
        features = [base_features * (2 ** i) for i in range(num_layers)]
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        in_ch = base_features
        for i, out_ch in enumerate(features):
            use_attn = i in attention_layers
            self.down_blocks.append(
                DownBlock3D(in_ch, out_ch, time_emb_dim, use_attn, num_heads)
            )
            in_ch = out_ch
        
        # 中间块
        self.mid_block1 = ResBlock3D(features[-1], features[-1], time_emb_dim)
        self.mid_attn = SelfAttention3D(features[-1], num_heads)
        self.mid_block2 = ResBlock3D(features[-1], features[-1], time_emb_dim)
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features) - 1):
            in_ch = reversed_features[i]
            out_ch = reversed_features[i + 1]
            use_attn = (len(features) - 1 - i) in attention_layers
            self.up_blocks.append(
                UpBlock3D(in_ch, out_ch, time_emb_dim, use_attn, num_heads)
            )
        
        # 最后一个上采样块
        self.up_blocks.append(
            UpBlock3D(reversed_features[-1], base_features, time_emb_dim, False)
        )
        
        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_features),
            nn.SiLU(),
            nn.Conv3d(base_features, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 噪声图像 (B, C, D, H, W)
            t: 时间步 (B,)
            condition: 条件 (B, C, D, H, W)
        
        返回：
            预测的噪声 (B, C, D, H, W)
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 合并输入和条件
        x = torch.cat([x, condition], dim=1)
        x = self.input_conv(x)
        
        # 下采样
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb)
            skips.append(skip)
        
        # 中间处理
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # 上采样
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, t_emb)
        
        # 输出
        return self.output_conv(x)


# ============================================================================
# 扩散过程管理器
# ============================================================================

class DiffusionProcess:
    """
    扩散过程管理器
    
    管理前向扩散和反向去噪过程
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        device: str = 'cuda'
    ):
        """
        参数：
            timesteps: 扩散步数 T
            beta_schedule: 噪声调度类型 ('linear', 'cosine', 'quadratic')
            device: 计算设备
        """
        self.timesteps = timesteps
        self.device = device
        
        # 计算噪声调度参数
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps)
        else:
            raise ValueError(f"未知的噪声调度: {beta_schedule}")
        
        self.betas = betas.to(device)
        
        # 预计算常用参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于前向采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于反向采样的参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _extract(
        self, 
        a: torch.Tensor, 
        t: torch.Tensor, 
        x_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """从参数序列中提取对应时间步的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散采样：q(x_t | x_0)
        
        \[
        x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
        \]
        
        参数：
            x_start: 原始数据 x_0
            t: 时间步
            noise: 可选的噪声，默认随机生成
        
        返回：
            加噪后的数据 x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        反向去噪一步：p(x_{t-1} | x_t)
        
        \[
        x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta \right) + \sigma_t z
        \]
        
        参数：
            model: 噪声预测模型
            x_t: 当前噪声数据
            t: 时间步
            condition: 条件信息
        
        返回：
            去噪后的数据 x_{t-1}
        """
        # 预测噪声
        predicted_noise = model(x_t, t, condition)
        
        # 提取参数
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # 添加噪声（除了 t=0）
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        progress: bool = True
    ) -> torch.Tensor:
        """
        完整的反向去噪过程
        
        从纯噪声开始，逐步去噪得到生成结果
        
        参数：
            model: 噪声预测模型
            shape: 输出形状 (B, C, D, H, W)
            condition: 条件信息
            progress: 是否显示进度条
        
        返回：
            生成的数据
        """
        device = self.device
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        timesteps = reversed(range(self.timesteps))
        if progress:
            timesteps = tqdm(timesteps, desc='采样中', total=self.timesteps)
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        ddim_steps: int = 50,
        eta: float = 0.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        DDIM 采样（Denoising Diffusion Implicit Models）
        
        加速采样，使用更少的步数生成高质量结果
        
        参数：
            model: 噪声预测模型
            shape: 输出形状
            condition: 条件信息
            ddim_steps: DDIM 采样步数（远小于训练时的 timesteps）
            eta: 随机性参数（0 = 确定性，1 = DDPM）
            progress: 是否显示进度条
        
        返回：
            生成的数据
        """
        device = self.device
        batch_size = shape[0]
        
        # 选择采样时间步
        c = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, c))
        timesteps = list(reversed(timesteps))
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        if progress:
            timesteps_iter = tqdm(range(len(timesteps) - 1), desc='DDIM采样')
        else:
            timesteps_iter = range(len(timesteps) - 1)
        
        for i in timesteps_iter:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = model(x, t_batch, condition)
            
            # DDIM 更新
            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next]
            
            # 预测 x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # 计算 sigma
            sigma = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)
            
            # 计算方向
            dir_xt = torch.sqrt(1 - alpha_t_next - sigma ** 2) * predicted_noise
            
            # 更新
            noise = torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_t_next) * x0_pred + dir_xt + sigma * noise
        
        return x


# ============================================================================
# 完整的条件 Diffusion 模型
# ============================================================================

class OCTADiffusionModel(nn.Module):
    """
    OCTA 条件扩散模型
    
    用于从带拖尾的 OCTA 数据生成干净的血管网络
    
    使用方法：
    ```python
    model = OCTADiffusionModel()
    
    # 训练
    loss = model.compute_loss(clean_octa, tailed_octa)
    
    # 推理
    clean_pred = model.sample(tailed_octa)
    ```
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        base_features: int = 64,
        time_emb_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        
        # 扩散过程
        self.diffusion = DiffusionProcess(
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            device=device
        )
        
        # 噪声预测网络
        self.noise_net = ConditionalUNet3D(
            in_channels=1,
            out_channels=1,
            condition_channels=1,
            base_features=base_features,
            time_emb_dim=time_emb_dim
        )
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算训练损失
        
        \[
        \mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2
        \]
        
        参数：
            x_start: 干净的 OCTA 数据（目标）
            condition: 带拖尾的 OCTA 数据（条件）
            noise: 可选的噪声
        
        返回：
            MSE 损失
        """
        batch_size = x_start.shape[0]
        
        # 随机采样时间步
        t = torch.randint(
            0, self.diffusion.timesteps, 
            (batch_size,), 
            device=x_start.device
        ).long()
        
        # 生成噪声
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_t = self.diffusion.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.noise_net(x_t, t, condition)
        
        # MSE 损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        progress: bool = True
    ) -> torch.Tensor:
        """
        从条件生成干净的 OCTA 数据
        
        参数：
            condition: 带拖尾的 OCTA 数据
            use_ddim: 是否使用 DDIM 加速采样
            ddim_steps: DDIM 步数
            progress: 是否显示进度条
        
        返回：
            生成的干净 OCTA 数据
        """
        self.noise_net.eval()
        
        shape = condition.shape
        
        if use_ddim:
            output = self.diffusion.ddim_sample(
                self.noise_net,
                shape,
                condition,
                ddim_steps=ddim_steps,
                progress=progress
            )
        else:
            output = self.diffusion.p_sample_loop(
                self.noise_net,
                shape,
                condition,
                progress=progress
            )
        
        # 限制输出范围
        output = torch.clamp(output, 0, 1)
        
        return output
    
    def forward(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """训练时的前向传播，返回损失"""
        return self.compute_loss(x_start, condition)


# 测试代码
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建模型
    model = OCTADiffusionModel(
        timesteps=1000,
        beta_schedule='cosine',
        base_features=32,  # 减小以便测试
        device=device
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试前向传播
    batch_size = 2
    x_start = torch.randn(batch_size, 1, 32, 32, 32).to(device)
    condition = torch.randn(batch_size, 1, 32, 32, 32).to(device)
    
    # 计算损失
    loss = model.compute_loss(x_start, condition)
    print(f"损失: {loss.item():.4f}")
    
    # 测试采样（使用较少步数）
    print("测试 DDIM 采样...")
    output = model.sample(condition, use_ddim=True, ddim_steps=5, progress=False)
    print(f"输出形状: {output.shape}")
    
    print("\n模型测试通过！")
