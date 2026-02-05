# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 损失函数

提供各种用于血管分割和生成的损失函数：
1. Dice Loss：处理类别不平衡
2. Focal Loss：关注困难样本
3. Boundary Loss：关注边界区域
4. Combined Loss：多损失组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice 损失函数
    
    Dice 系数衡量两个集合的相似度：
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        DiceLoss = 1 - Dice
    
    对于类别不平衡问题（血管占比小）特别有效
    
    参数：
        smooth: 平滑因子，防止除零
        reduction: 归约方式 ('mean', 'sum', 'none')
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Dice Loss
        
        参数：
            pred: 预测概率 (B, C, D, H, W) 或 (B, C, H, W)
            target: 目标标签，与 pred 形状相同
        
        返回：
            Dice Loss 值
        """
        # 确保输入在 [0, 1] 范围
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 展平空间维度
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum(dim=1)
        
        # 计算 Dice
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        
        # 损失 = 1 - Dice
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    通过调制因子 (1 - p_t)^γ 降低易分类样本的权重，
    使模型更关注困难样本
    
    公式：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    参数：
        alpha: 类别权重（正样本权重）
        gamma: 调制因子指数
        reduction: 归约方式
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Focal Loss
        
        参数：
            pred: 预测概率或 logits
            target: 目标标签 (0 或 1)
        """
        # 如果输入是 logits，先转换为概率
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # 计算 p_t
        p_t = target * pred + (1 - target) * (1 - pred)
        
        # 计算 α_t
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # 计算 focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # 计算 BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # 加权 BCE
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    边界损失函数
    
    关注血管边界区域，提高分割精度
    使用距离变换计算边界权重
    
    参数：
        reduction: 归约方式
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        dist_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算边界损失
        
        参数：
            pred: 预测概率
            target: 目标标签
            dist_map: 距离图（可选），如果不提供则自动计算
        """
        if dist_map is None:
            # 简化的边界检测：使用梯度
            # 真实应用中应该使用距离变换
            dist_map = self._compute_boundary_weight(target)
        
        # 加权 BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = bce * dist_map
        
        if self.reduction == 'mean':
            return weighted_bce.mean()
        elif self.reduction == 'sum':
            return weighted_bce.sum()
        else:
            return weighted_bce
    
    def _compute_boundary_weight(self, target: torch.Tensor) -> torch.Tensor:
        """
        计算边界权重图
        
        使用 Sobel 算子检测边界
        """
        # 对于 3D 数据
        if target.dim() == 5:
            # 简化处理：沿最后一个维度计算梯度
            grad = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
            grad = F.pad(grad, (0, 1), mode='constant', value=0)
        else:
            # 2D 数据
            grad = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            grad = F.pad(grad, (0, 1), mode='constant', value=0)
        
        # 边界区域权重更高
        weight = 1.0 + grad * 4.0
        
        return weight


class TverskyLoss(nn.Module):
    """
    Tversky 损失函数
    
    Dice 的广义形式，可以调整假阳和假阴的权重
    
    公式：Tversky = TP / (TP + α*FP + β*FN)
    
    当 α = β = 0.5 时等价于 Dice
    α < β 时更关注假阴（漏检）
    α > β 时更关注假阳（误检）
    
    参数：
        alpha: 假阳权重
        beta: 假阴权重
        smooth: 平滑因子
    """
    
    def __init__(
        self, 
        alpha: float = 0.3, 
        beta: float = 0.7, 
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算 Tversky Loss"""
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # True Positives
        tp = (pred_flat * target_flat).sum(dim=1)
        
        # False Positives
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        
        # False Negatives
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)
        
        # Tversky 指数
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        return (1 - tversky).mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合多种损失函数的优点
    
    参数：
        dice_weight: Dice Loss 权重
        bce_weight: BCE Loss 权重
        focal_weight: Focal Loss 权重
        boundary_weight: Boundary Loss 权重
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        focal_weight: float = 0.0,
        boundary_weight: float = 0.0
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算组合损失
        """
        # 确保预测在 [0, 1] 范围
        if pred.min() < 0 or pred.max() > 1:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
        
        total_loss = 0.0
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(pred_prob, target)
        
        if self.bce_weight > 0:
            total_loss += self.bce_weight * self.bce_loss(pred_prob, target)
        
        if self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal_loss(pred_prob, target)
        
        if self.boundary_weight > 0:
            total_loss += self.boundary_weight * self.boundary_loss(pred_prob, target)
        
        return total_loss


class VesselLoss(nn.Module):
    """
    专门针对血管分割的损失函数
    
    特点：
    1. 结合 Dice + BCE 处理类别不平衡
    2. 添加连续性惩罚，鼓励平滑的血管结构
    3. 可选的拓扑损失
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        continuity_weight: float = 0.1
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.continuity_weight = continuity_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self, 
        pred_logits: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算血管分割损失
        
        参数：
            pred_logits: 预测 logits（未经 sigmoid）
            target: 目标标签
        """
        pred_prob = torch.sigmoid(pred_logits)
        
        # Dice Loss
        dice = self.dice_loss(pred_prob, target)
        
        # BCE Loss
        bce = self.bce_loss(pred_logits, target)
        
        # 连续性损失：惩罚预测中的不连续
        continuity = self._continuity_loss(pred_prob)
        
        total = (
            self.dice_weight * dice + 
            self.bce_weight * bce + 
            self.continuity_weight * continuity
        )
        
        return total
    
    def _continuity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        计算连续性损失
        
        使用总变分（Total Variation）鼓励平滑
        """
        if pred.dim() == 5:
            # 3D: (B, C, D, H, W)
            diff_d = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
            diff_h = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
            diff_w = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
            tv = diff_d.mean() + diff_h.mean() + diff_w.mean()
        else:
            # 2D: (B, C, H, W)
            diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            tv = diff_h.mean() + diff_w.mean()
        
        return tv
