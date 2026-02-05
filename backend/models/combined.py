# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 联合模型

将 3D U-Net 分割网络与 Diffusion 模型结合，实现：
1. 粗分割：U-Net 快速生成初步血管 mask
2. 精修：Diffusion 模型根据条件优化细节

工作流程：
    输入(带拖尾 OCTA) 
        → U-Net 分割 
        → 初步 mask 
        → Diffusion 精修 
        → 高质量输出
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from .unet3d import UNet3D, VesselSegmentationNet
from .diffusion import OCTADiffusionModel, ConditionalUNet3D


class CombinedOCTAModel(nn.Module):
    """
    联合 OCTA 处理模型
    
    结合分割网络和扩散模型，实现两阶段处理：
    1. 分割阶段：快速生成粗分割结果
    2. 扩散阶段：利用扩散模型精修细节
    
    参数：
        seg_base_features: 分割网络基础特征数
        diff_base_features: 扩散网络基础特征数
        diff_timesteps: 扩散步数
        diff_beta_schedule: 噪声调度类型
        use_attention: 是否使用注意力机制
        device: 计算设备
    """
    
    def __init__(
        self,
        seg_base_features: int = 32,
        diff_base_features: int = 64,
        diff_timesteps: int = 1000,
        diff_beta_schedule: str = 'cosine',
        use_attention: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        
        # ====================================================================
        # 阶段1: 3D U-Net 分割网络
        # 输入: 带拖尾的 OCTA (B, 1, D, H, W)
        # 输出: 粗分割血管概率图 (B, 1, D, H, W)
        # ====================================================================
        self.segmentation_net = VesselSegmentationNet(
            in_channels=1,
            base_features=seg_base_features,
            use_attention=use_attention
        )
        
        # ====================================================================
        # 阶段2: 条件 Diffusion 模型
        # 输入: 
        #   - x_t: 噪声扰动的目标 (B, 1, D, H, W)
        #   - condition: 原始 OCTA + 粗分割结果 (B, 2, D, H, W)
        #   - t: 时间步
        # 输出: 预测噪声或干净目标
        # ====================================================================
        self.diffusion_model = OCTADiffusionModel(
            timesteps=diff_timesteps,
            beta_schedule=diff_beta_schedule,
            base_features=diff_base_features,
            in_channels=1,
            condition_channels=2,  # 原始 + 粗分割
            device=device
        )
        
        # 训练模式标志
        self.train_segmentation = True
        self.train_diffusion = True
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数：
            x: 输入的带拖尾 OCTA (B, 1, D, H, W)
            target: 训练时的金标准 (B, 1, D, H, W)，推理时为 None
            return_intermediate: 是否返回中间结果
        
        返回：
            包含各阶段输出的字典
        """
        outputs = {}
        
        # ====================================================================
        # 阶段1: 分割
        # ====================================================================
        seg_logits = self.segmentation_net.unet(x)
        seg_prob = torch.sigmoid(seg_logits)
        
        outputs['seg_logits'] = seg_logits
        outputs['seg_prob'] = seg_prob
        
        # ====================================================================
        # 阶段2: 扩散精修
        # ====================================================================
        # 构建条件输入：拼接原始 OCTA 和粗分割结果
        # condition shape: (B, 2, D, H, W)
        condition = torch.cat([x, seg_prob.detach()], dim=1)
        
        if target is not None and self.training:
            # 训练模式：计算扩散损失
            diff_loss = self.diffusion_model.compute_loss(target, condition)
            outputs['diff_loss'] = diff_loss
        else:
            # 推理模式：通过扩散采样生成结果
            # 使用 DDIM 加速采样
            refined = self.diffusion_model.sample(
                condition,
                use_ddim=True,
                ddim_steps=50,
                progress=False
            )
            outputs['refined'] = refined
        
        if return_intermediate:
            outputs['condition'] = condition
        
        return outputs
    
    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        seg_weight: float = 1.0,
        diff_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合训练损失
        
        参数：
            x: 输入的带拖尾 OCTA
            target: 金标准
            seg_weight: 分割损失权重
            diff_weight: 扩散损失权重
        
        返回：
            包含各项损失的字典
        """
        from training.losses import DiceLoss
        
        # 前向传播
        outputs = self.forward(x, target)
        
        # 分割损失
        seg_logits = outputs['seg_logits']
        dice_loss = DiceLoss()(torch.sigmoid(seg_logits), target)
        bce_loss = nn.BCEWithLogitsLoss()(seg_logits, target)
        seg_loss = dice_loss + bce_loss
        
        # 扩散损失
        diff_loss = outputs.get('diff_loss', torch.tensor(0.0, device=self.device))
        
        # 总损失
        total_loss = seg_weight * seg_loss + diff_weight * diff_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'dice_loss': dice_loss,
            'bce_loss': bce_loss,
            'diff_loss': diff_loss
        }
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        use_diffusion: bool = True,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        seg_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        推理预测
        
        参数：
            x: 输入的带拖尾 OCTA
            use_diffusion: 是否使用扩散模型精修
            use_ddim: 是否使用 DDIM 加速
            ddim_steps: DDIM 采样步数
            seg_threshold: 分割阈值
        
        返回：
            包含预测结果的字典
        """
        self.eval()
        
        # 分割
        seg_logits = self.segmentation_net.unet(x)
        seg_prob = torch.sigmoid(seg_logits)
        seg_mask = (seg_prob > seg_threshold).float()
        
        results = {
            'seg_prob': seg_prob,
            'seg_mask': seg_mask
        }
        
        if use_diffusion:
            # 构建条件
            condition = torch.cat([x, seg_prob], dim=1)
            
            # 扩散采样
            refined = self.diffusion_model.sample(
                condition,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                progress=False
            )
            
            results['refined'] = refined
            results['refined_mask'] = (refined > seg_threshold).float()
        
        return results
    
    def get_segmentation_params(self):
        """获取分割网络参数"""
        return self.segmentation_net.parameters()
    
    def get_diffusion_params(self):
        """获取扩散模型参数"""
        return self.diffusion_model.parameters()
    
    def freeze_segmentation(self):
        """冻结分割网络"""
        for param in self.segmentation_net.parameters():
            param.requires_grad = False
        self.train_segmentation = False
    
    def unfreeze_segmentation(self):
        """解冻分割网络"""
        for param in self.segmentation_net.parameters():
            param.requires_grad = True
        self.train_segmentation = True
    
    def freeze_diffusion(self):
        """冻结扩散模型"""
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
        self.train_diffusion = False
    
    def unfreeze_diffusion(self):
        """解冻扩散模型"""
        for param in self.diffusion_model.parameters():
            param.requires_grad = True
        self.train_diffusion = True


class CascadeTrainer:
    """
    级联训练器
    
    实现两阶段训练策略：
    1. 先训练分割网络直到收敛
    2. 冻结分割网络，训练扩散模型
    """
    
    def __init__(
        self,
        model: CombinedOCTAModel,
        seg_epochs: int = 50,
        diff_epochs: int = 50,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        self.model = model
        self.seg_epochs = seg_epochs
        self.diff_epochs = diff_epochs
        self.learning_rate = learning_rate
        self.device = device
        
        # 优化器（将在训练时创建）
        self.seg_optimizer = None
        self.diff_optimizer = None
    
    def train_stage1(
        self,
        train_loader,
        val_loader=None,
        callbacks=None
    ):
        """
        阶段1: 训练分割网络
        """
        print("=" * 60)
        print("阶段 1: 训练分割网络")
        print("=" * 60)
        
        self.model.freeze_diffusion()
        self.model.unfreeze_segmentation()
        
        self.seg_optimizer = torch.optim.AdamW(
            self.model.get_segmentation_params(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.seg_optimizer,
            T_max=self.seg_epochs
        )
        
        best_loss = float('inf')
        
        for epoch in range(self.seg_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.seg_optimizer.zero_grad()
                
                losses = self.model.compute_loss(inputs, targets, seg_weight=1.0, diff_weight=0.0)
                loss = losses['seg_loss']
                
                loss.backward()
                self.seg_optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.seg_epochs} - 分割 Loss: {epoch_loss:.4f}")
            
            if callbacks:
                for cb in callbacks:
                    cb(epoch, epoch_loss, 'segmentation')
        
        print("阶段 1 完成")
    
    def train_stage2(
        self,
        train_loader,
        val_loader=None,
        callbacks=None
    ):
        """
        阶段2: 训练扩散模型
        """
        print("=" * 60)
        print("阶段 2: 训练扩散模型")
        print("=" * 60)
        
        self.model.freeze_segmentation()
        self.model.unfreeze_diffusion()
        
        self.diff_optimizer = torch.optim.AdamW(
            self.model.get_diffusion_params(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.diff_optimizer,
            T_max=self.diff_epochs
        )
        
        for epoch in range(self.diff_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.diff_optimizer.zero_grad()
                
                losses = self.model.compute_loss(inputs, targets, seg_weight=0.0, diff_weight=1.0)
                loss = losses['diff_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.get_diffusion_params(), 1.0)
                self.diff_optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.diff_epochs} - 扩散 Loss: {epoch_loss:.4f}")
            
            if callbacks:
                for cb in callbacks:
                    cb(epoch, epoch_loss, 'diffusion')
        
        print("阶段 2 完成")
    
    def train(
        self,
        train_loader,
        val_loader=None,
        callbacks=None
    ):
        """
        完整的级联训练
        """
        # 阶段1
        self.train_stage1(train_loader, val_loader, callbacks)
        
        # 阶段2
        self.train_stage2(train_loader, val_loader, callbacks)
        
        # 解冻所有参数
        self.model.unfreeze_segmentation()
        self.model.unfreeze_diffusion()
        
        print("=" * 60)
        print("级联训练完成!")
        print("=" * 60)
