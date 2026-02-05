# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 数据集定义

提供用于训练的数据集类：
1. OCTAPatchDataset: 从大型 3D 体数据提取 patch
2. OCTASliceDataset: 2D 切片数据集
3. 数据增强变换
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Callable
from scipy.ndimage import rotate, zoom, affine_transform
import random


class OCTAPatchDataset(Dataset):
    """
    OCTA 3D Patch 数据集
    
    从大型 3D 体数据中提取固定大小的 patch 用于训练
    支持数据增强和在线采样
    
    参数：
        octa_volume: 原始 OCTA 体数据 (X, Y, Z)
        label_volume: 标签体数据 (X, Y, Z)
        patch_size: patch 大小
        stride: 滑动步长（默认为 patch_size 的一半）
        augment: 是否启用数据增强
        min_vessel_ratio: 最小血管占比（过滤全背景 patch）
        transform: 额外的变换函数
    """
    
    def __init__(
        self,
        octa_volume: np.ndarray,
        label_volume: np.ndarray,
        patch_size: int = 32,
        stride: Optional[int] = None,
        augment: bool = True,
        min_vessel_ratio: float = 0.001,
        transform: Optional[Callable] = None
    ):
        self.octa_volume = octa_volume.astype(np.float32)
        self.label_volume = label_volume.astype(np.float32)
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2
        self.augment = augment
        self.min_vessel_ratio = min_vessel_ratio
        self.transform = transform
        
        # 预计算所有有效 patch 的位置
        self.patch_positions = self._compute_patch_positions()
        
        print(f"数据集初始化完成:")
        print(f"  - 体数据大小: {octa_volume.shape}")
        print(f"  - Patch 大小: {patch_size}³")
        print(f"  - 步长: {self.stride}")
        print(f"  - 有效 Patch 数量: {len(self.patch_positions)}")
    
    def _compute_patch_positions(self) -> List[Tuple[int, int, int]]:
        """计算所有有效 patch 的起始位置"""
        positions = []
        
        X, Y, Z = self.octa_volume.shape
        ps = self.patch_size
        
        for x in range(0, X - ps + 1, self.stride):
            for y in range(0, Y - ps + 1, self.stride):
                for z in range(0, Z - ps + 1, self.stride):
                    # 提取标签 patch
                    label_patch = self.label_volume[x:x+ps, y:y+ps, z:z+ps]
                    
                    # 检查血管占比
                    vessel_ratio = label_patch.sum() / label_patch.size
                    
                    if vessel_ratio >= self.min_vessel_ratio:
                        positions.append((x, y, z))
        
        return positions
    
    def __len__(self) -> int:
        return len(self.patch_positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, z = self.patch_positions[idx]
        ps = self.patch_size
        
        # 提取 patch
        octa_patch = self.octa_volume[x:x+ps, y:y+ps, z:z+ps].copy()
        label_patch = self.label_volume[x:x+ps, y:y+ps, z:z+ps].copy()
        
        # 数据增强
        if self.augment:
            octa_patch, label_patch = self._augment(octa_patch, label_patch)
        
        # 额外变换
        if self.transform:
            octa_patch, label_patch = self.transform(octa_patch, label_patch)
        
        # 转换为 tensor
        # 添加通道维度: (D, H, W) -> (1, D, H, W)
        octa_tensor = torch.from_numpy(octa_patch).unsqueeze(0)
        label_tensor = torch.from_numpy(label_patch).unsqueeze(0)
        
        return octa_tensor, label_tensor
    
    def _augment(
        self, 
        octa: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强
        
        包括：
        1. 随机翻转（3 个轴）
        2. 随机旋转（90° 倍数）
        3. 强度扰动
        """
        # 随机翻转
        for axis in range(3):
            if random.random() > 0.5:
                octa = np.flip(octa, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        # 随机旋转 90° 倍数
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            # 沿 XY 平面旋转
            octa = np.rot90(octa, k=k, axes=(0, 1)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()
        
        # 强度扰动（只对 OCTA 数据）
        if random.random() > 0.5:
            # 添加高斯噪声
            noise = np.random.normal(0, 0.02, octa.shape).astype(np.float32)
            octa = np.clip(octa + noise, 0, 1)
        
        if random.random() > 0.5:
            # 亮度调整
            factor = random.uniform(0.9, 1.1)
            octa = np.clip(octa * factor, 0, 1)
        
        return octa, label


class OCTASliceDataset(Dataset):
    """
    OCTA 2D 切片数据集
    
    用于 2D 模型训练或快速原型验证
    
    参数：
        octa_volume: 原始 OCTA 体数据
        label_volume: 标签体数据
        axis: 切片轴向 (0, 1, 2)
        augment: 是否启用数据增强
    """
    
    def __init__(
        self,
        octa_volume: np.ndarray,
        label_volume: np.ndarray,
        axis: int = 1,
        augment: bool = True
    ):
        self.octa_volume = octa_volume.astype(np.float32)
        self.label_volume = label_volume.astype(np.float32)
        self.axis = axis
        self.augment = augment
        
        self.num_slices = octa_volume.shape[axis]
    
    def __len__(self) -> int:
        return self.num_slices
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 提取切片
        if self.axis == 0:
            octa_slice = self.octa_volume[idx, :, :].copy()
            label_slice = self.label_volume[idx, :, :].copy()
        elif self.axis == 1:
            octa_slice = self.octa_volume[:, idx, :].copy()
            label_slice = self.label_volume[:, idx, :].copy()
        else:
            octa_slice = self.octa_volume[:, :, idx].copy()
            label_slice = self.label_volume[:, :, idx].copy()
        
        # 数据增强
        if self.augment:
            octa_slice, label_slice = self._augment_2d(octa_slice, label_slice)
        
        # 转换为 tensor
        octa_tensor = torch.from_numpy(octa_slice).unsqueeze(0)
        label_tensor = torch.from_numpy(label_slice).unsqueeze(0)
        
        return octa_tensor, label_tensor
    
    def _augment_2d(
        self, 
        octa: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """2D 数据增强"""
        # 随机翻转
        if random.random() > 0.5:
            octa = np.flip(octa, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        
        if random.random() > 0.5:
            octa = np.flip(octa, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        
        # 随机旋转
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            octa = np.rot90(octa, k=k).copy()
            label = np.rot90(label, k=k).copy()
        
        return octa, label


class OnlinePatchSampler:
    """
    在线 Patch 采样器
    
    在训练过程中动态采样 patch，支持：
    1. 难例挖掘
    2. 类别平衡采样
    """
    
    def __init__(
        self,
        octa_volume: np.ndarray,
        label_volume: np.ndarray,
        patch_size: int = 32,
        samples_per_epoch: int = 1000,
        hard_mining: bool = True
    ):
        self.octa_volume = octa_volume
        self.label_volume = label_volume
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.hard_mining = hard_mining
        
        self.X, self.Y, self.Z = octa_volume.shape
        
        # 预计算血管位置（用于平衡采样）
        self.vessel_coords = np.argwhere(label_volume > 0.5)
        self.background_coords = np.argwhere(label_volume < 0.5)
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样一个 batch"""
        ps = self.patch_size
        
        octa_batch = []
        label_batch = []
        
        for _ in range(batch_size):
            # 50% 概率从血管区域采样，50% 从背景采样
            if random.random() > 0.5 and len(self.vessel_coords) > 0:
                # 从血管区域采样
                idx = random.randint(0, len(self.vessel_coords) - 1)
                center = self.vessel_coords[idx]
            else:
                # 随机采样
                center = [
                    random.randint(ps//2, self.X - ps//2 - 1),
                    random.randint(ps//2, self.Y - ps//2 - 1),
                    random.randint(ps//2, self.Z - ps//2 - 1)
                ]
            
            # 计算 patch 边界
            x = max(0, min(center[0] - ps//2, self.X - ps))
            y = max(0, min(center[1] - ps//2, self.Y - ps))
            z = max(0, min(center[2] - ps//2, self.Z - ps))
            
            # 提取 patch
            octa_patch = self.octa_volume[x:x+ps, y:y+ps, z:z+ps]
            label_patch = self.label_volume[x:x+ps, y:y+ps, z:z+ps]
            
            octa_batch.append(octa_patch)
            label_batch.append(label_patch)
        
        return np.stack(octa_batch), np.stack(label_batch)


def create_dataloader(
    octa_volume: np.ndarray,
    label_volume: np.ndarray,
    batch_size: int = 4,
    patch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    创建数据加载器的便捷函数
    
    参数：
        octa_volume: OCTA 体数据
        label_volume: 标签体数据
        batch_size: 批次大小
        patch_size: patch 大小
        num_workers: 工作进程数
        **kwargs: 传递给 OCTAPatchDataset 的其他参数
    
    返回：
        DataLoader 实例
    """
    dataset = OCTAPatchDataset(
        octa_volume=octa_volume,
        label_volume=label_volume,
        patch_size=patch_size,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
