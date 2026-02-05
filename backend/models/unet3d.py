# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 3D U-Net 分割网络

================================================================================
网络架构说明
================================================================================

3D U-Net 是用于三维医学图像分割的经典架构，由以下部分组成：

1. 编码器（Encoder）：
   - 逐层下采样，提取多尺度特征
   - 每层：卷积 → BN → ReLU → 卷积 → BN → ReLU → MaxPool

2. 瓶颈层（Bottleneck）：
   - 最深层特征提取

3. 解码器（Decoder）：
   - 逐层上采样，恢复空间分辨率
   - 跳跃连接（Skip Connection）融合编码器特征
   - 每层：上采样 → Concat → 卷积 → BN → ReLU → 卷积 → BN → ReLU

4. 输出层：
   - 1×1×1 卷积生成分割概率图

本实现针对 OCTA 血管分割进行了优化：
- 支持 patch-based 训练（处理大体积数据）
- 使用残差连接增强梯度流动
- 引入注意力机制提升血管边界分割精度

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ConvBlock3D(nn.Module):
    """
    3D 卷积块：Conv3d → BatchNorm → ReLU → Conv3d → BatchNorm → ReLU
    
    可选：残差连接
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接的通道匹配
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(
                in_channels, out_channels, 
                kernel_size=1
            )
        else:
            self.residual_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            out = out + residual
        
        out = self.relu(out)
        return out


class AttentionGate3D(nn.Module):
    """
    3D 注意力门控
    
    用于在跳跃连接中增强重要特征，抑制无关特征
    特别适合血管分割，可以聚焦于血管边界
    
    公式：
    α = σ(W_ψ × ψ + W_g × g + b)
    out = α × g
    
    其中：
    - g: 编码器特征（高分辨率）
    - ψ: 解码器特征（语义丰富）
    - α: 注意力权重
    """
    
    def __init__(
        self, 
        F_g: int,  # 解码器特征通道数
        F_l: int,  # 编码器特征通道数
        F_int: int  # 中间特征通道数
    ):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            g: 解码器特征 (B, F_g, D, H, W)
            x: 编码器特征 (B, F_l, D, H, W)
        
        返回：
            注意力加权后的编码器特征
        """
        # 确保 g 和 x 的空间尺寸匹配
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNet3D(nn.Module):
    """
    3D U-Net 分割网络
    
    用于 OCTA 血管分割：
    - 输入：带拖尾的 OCTA 体数据 (B, 1, D, H, W)
    - 输出：血管概率图 (B, 1, D, H, W) 或二值掩模
    
    特点：
    - 4 层编码器-解码器结构
    - 注意力门控跳跃连接
    - 残差连接
    - 支持深度监督
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        use_attention: bool = True,
        use_residual: bool = True,
        deep_supervision: bool = False
    ):
        """
        参数：
            in_channels: 输入通道数（默认 1，灰度 OCTA）
            out_channels: 输出通道数（默认 1，血管概率）
            base_features: 基础特征通道数（逐层翻倍）
            use_attention: 是否使用注意力门控
            use_residual: 是否使用残差连接
            deep_supervision: 是否使用深度监督
        """
        super().__init__()
        
        self.use_attention = use_attention
        self.deep_supervision = deep_supervision
        
        # 特征通道数：32 → 64 → 128 → 256 → 512
        features = [base_features * (2 ** i) for i in range(5)]
        
        # ============================================================
        # 编码器
        # ============================================================
        self.encoder1 = ConvBlock3D(in_channels, features[0], use_residual=use_residual)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock3D(features[0], features[1], use_residual=use_residual)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock3D(features[1], features[2], use_residual=use_residual)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = ConvBlock3D(features[2], features[3], use_residual=use_residual)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # ============================================================
        # 瓶颈层
        # ============================================================
        self.bottleneck = ConvBlock3D(features[3], features[4], use_residual=use_residual)
        
        # ============================================================
        # 解码器
        # ============================================================
        self.upconv4 = nn.ConvTranspose3d(
            features[4], features[3], 
            kernel_size=2, stride=2
        )
        if use_attention:
            self.att4 = AttentionGate3D(features[3], features[3], features[3] // 2)
        self.decoder4 = ConvBlock3D(features[4], features[3], use_residual=use_residual)
        
        self.upconv3 = nn.ConvTranspose3d(
            features[3], features[2], 
            kernel_size=2, stride=2
        )
        if use_attention:
            self.att3 = AttentionGate3D(features[2], features[2], features[2] // 2)
        self.decoder3 = ConvBlock3D(features[3], features[2], use_residual=use_residual)
        
        self.upconv2 = nn.ConvTranspose3d(
            features[2], features[1], 
            kernel_size=2, stride=2
        )
        if use_attention:
            self.att2 = AttentionGate3D(features[1], features[1], features[1] // 2)
        self.decoder2 = ConvBlock3D(features[2], features[1], use_residual=use_residual)
        
        self.upconv1 = nn.ConvTranspose3d(
            features[1], features[0], 
            kernel_size=2, stride=2
        )
        if use_attention:
            self.att1 = AttentionGate3D(features[0], features[0], features[0] // 2)
        self.decoder1 = ConvBlock3D(features[1], features[0], use_residual=use_residual)
        
        # ============================================================
        # 输出层
        # ============================================================
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # 深度监督输出头
        if deep_supervision:
            self.ds_out4 = nn.Conv3d(features[3], out_channels, kernel_size=1)
            self.ds_out3 = nn.Conv3d(features[2], out_channels, kernel_size=1)
            self.ds_out2 = nn.Conv3d(features[1], out_channels, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播
        
        参数：
            x: 输入张量 (B, C, D, H, W)
        
        返回：
            如果 deep_supervision=False：
                输出张量 (B, out_channels, D, H, W)
            如果 deep_supervision=True：
                (主输出, [中间输出列表])
        """
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 解码器 4
        up4 = self.upconv4(bottleneck)
        if up4.shape[2:] != enc4.shape[2:]:
            up4 = F.interpolate(up4, size=enc4.shape[2:], mode='trilinear', align_corners=False)
        if self.use_attention:
            enc4 = self.att4(up4, enc4)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        
        # 解码器 3
        up3 = self.upconv3(dec4)
        if up3.shape[2:] != enc3.shape[2:]:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='trilinear', align_corners=False)
        if self.use_attention:
            enc3 = self.att3(up3, enc3)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        
        # 解码器 2
        up2 = self.upconv2(dec3)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        if self.use_attention:
            enc2 = self.att2(up2, enc2)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        
        # 解码器 1
        up1 = self.upconv1(dec2)
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        if self.use_attention:
            enc1 = self.att1(up1, enc1)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))
        
        # 输出
        output = self.final_conv(dec1)
        
        if self.deep_supervision and self.training:
            # 深度监督输出
            ds4 = self.ds_out4(dec4)
            ds4 = F.interpolate(ds4, size=output.shape[2:], mode='trilinear', align_corners=False)
            
            ds3 = self.ds_out3(dec3)
            ds3 = F.interpolate(ds3, size=output.shape[2:], mode='trilinear', align_corners=False)
            
            ds2 = self.ds_out2(dec2)
            ds2 = F.interpolate(ds2, size=output.shape[2:], mode='trilinear', align_corners=False)
            
            return output, [ds4, ds3, ds2]
        
        return output


class VesselSegmentationNet(nn.Module):
    """
    血管分割网络（带后处理）
    
    在 3D U-Net 基础上添加：
    - Sigmoid 激活输出概率
    - 可选的后处理层（形态学操作等）
    """
    
    def __init__(
        self,
        **unet_kwargs
    ):
        super().__init__()
        
        self.unet = UNet3D(**unet_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        返回：
            血管概率图 (B, 1, D, H, W)，值域 [0, 1]
        """
        output = self.unet(x)
        
        if isinstance(output, tuple):
            main_out = output[0]
            return torch.sigmoid(main_out), [torch.sigmoid(o) for o in output[1]]
        
        return torch.sigmoid(output)
    
    def predict(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        预测二值掩模
        
        参数：
            x: 输入张量
            threshold: 二值化阈值
        
        返回：
            二值掩模 (B, 1, D, H, W)
        """
        with torch.no_grad():
            prob = self.forward(x)
            if isinstance(prob, tuple):
                prob = prob[0]
            return (prob > threshold).float()


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_features=32,
        use_attention=True,
        use_residual=True,
        deep_supervision=True
    )
    
    # 打印模型结构
    print("=" * 60)
    print("3D U-Net 模型结构")
    print("=" * 60)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(1, 1, 32, 32, 32)
    model.train()
    output = model(x)
    
    if isinstance(output, tuple):
        print(f"\n主输出形状: {output[0].shape}")
        print(f"深度监督输出数: {len(output[1])}")
        for i, ds in enumerate(output[1]):
            print(f"  DS{4-i} 输出形状: {ds.shape}")
    else:
        print(f"\n输出形状: {output.shape}")
    
    print("\n模型测试通过！")
