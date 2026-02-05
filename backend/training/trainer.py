# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 训练管理模块

功能：
1. 数据集定义和加载
2. 训练循环管理
3. 损失函数定义
4. 检查点保存和恢复
5. 训练进度跟踪
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from queue import Queue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# 导入模型
from models.unet3d import UNet3D, VesselSegmentationNet
from models.diffusion import OCTADiffusionModel


# ============================================================================
# 数据集定义
# ============================================================================

class OCTAPatchDataset(Dataset):
    """
    OCTA 3D Patch 数据集
    
    从大型 3D 体数据中提取小块进行训练
    
    参数：
        octa_volume: 带拖尾的 OCTA 体数据 (X, Y, Z)
        label_volume: 金标准标签 (X, Y, Z)
        patch_size: 3D patch 尺寸
        stride: 采样步长
        augment: 是否进行数据增强
    """
    
    def __init__(
        self,
        octa_volume: np.ndarray,
        label_volume: np.ndarray,
        patch_size: int = 32,
        stride: int = 16,
        augment: bool = True
    ):
        self.octa = octa_volume.astype(np.float32)
        self.label = label_volume.astype(np.float32)
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        
        # 计算有效的 patch 位置
        self.positions = self._compute_positions()
    
    def _compute_positions(self) -> List[Tuple[int, int, int]]:
        """计算所有有效的 patch 起始位置"""
        X, Y, Z = self.octa.shape
        ps = self.patch_size
        stride = self.stride
        
        positions = []
        for x in range(0, X - ps + 1, stride):
            for y in range(0, Y - ps + 1, stride):
                for z in range(0, Z - ps + 1, stride):
                    # 可选：跳过全背景区域
                    label_patch = self.label[x:x+ps, y:y+ps, z:z+ps]
                    if label_patch.sum() > 10:  # 至少有一些血管
                        positions.append((x, y, z))
        
        return positions
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, z = self.positions[idx]
        ps = self.patch_size
        
        # 提取 patch
        octa_patch = self.octa[x:x+ps, y:y+ps, z:z+ps].copy()
        label_patch = self.label[x:x+ps, y:y+ps, z:z+ps].copy()
        
        # 数据增强
        if self.augment:
            octa_patch, label_patch = self._augment(octa_patch, label_patch)
        
        # 转换为 tensor
        octa_tensor = torch.from_numpy(octa_patch).unsqueeze(0)  # (1, D, H, W)
        label_tensor = torch.from_numpy(label_patch).unsqueeze(0)
        
        return octa_tensor, label_tensor
    
    def _augment(
        self, 
        octa: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强"""
        # 随机翻转
        for axis in range(3):
            if np.random.rand() > 0.5:
                octa = np.flip(octa, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        # 随机旋转（90度的倍数）
        k = np.random.randint(0, 4)
        if k > 0:
            octa = np.rot90(octa, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
        
        # 强度扰动
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, octa.shape)
            octa = np.clip(octa + noise, 0, 1)
        
        return octa, label


# ============================================================================
# 损失函数
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss
    
    \[
    \text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|} = \frac{2 \sum p \cdot g}{\sum p + \sum g}
    \]
    
    Dice Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    组合损失：Dice + BCE
    
    L = α × Dice + (1-α) × BCE
    """
    
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss


# ============================================================================
# 训练状态跟踪
# ============================================================================

@dataclass
class TrainingState:
    """训练状态数据类"""
    status: str = "idle"  # idle, running, paused, completed, error
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_loss: float = float('inf')
    train_losses: List[float] = None
    val_losses: List[float] = None
    dice_scores: List[float] = None
    elapsed_time: float = 0.0
    eta: float = 0.0
    message: str = ""
    
    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.dice_scores is None:
            self.dice_scores = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 训练管理器
# ============================================================================

class TrainingManager:
    """
    训练管理器
    
    管理模型训练的完整生命周期：
    - 创建训练任务
    - 启动/暂停/停止训练
    - 保存/加载检查点
    - 跟踪训练进度
    """
    
    def __init__(
        self,
        data_dir: Path,
        models_dir: Path,
        device: str = 'cuda'
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 创建目录
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.state = TrainingState()
        self.training_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        
        # 当前模型和优化器
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.scaler: Optional[GradScaler] = None
        
        # 进度回调
        self.progress_callbacks: List[Callable] = []
    
    def create_segmentation_model(
        self,
        base_features: int = 32,
        use_attention: bool = True
    ) -> UNet3D:
        """创建分割模型"""
        model = UNet3D(
            in_channels=1,
            out_channels=1,
            base_features=base_features,
            use_attention=use_attention,
            use_residual=True,
            deep_supervision=True
        ).to(self.device)
        
        return model
    
    def create_diffusion_model(
        self,
        timesteps: int = 1000,
        base_features: int = 64
    ) -> OCTADiffusionModel:
        """创建扩散模型"""
        model = OCTADiffusionModel(
            timesteps=timesteps,
            beta_schedule='cosine',
            base_features=base_features,
            device=self.device
        ).to(self.device)
        
        return model
    
    def prepare_dataloaders(
        self,
        octa_train: np.ndarray,
        label_train: np.ndarray,
        octa_val: np.ndarray,
        label_val: np.ndarray,
        patch_size: int = 32,
        batch_size: int = 4,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """准备数据加载器"""
        train_dataset = OCTAPatchDataset(
            octa_train, label_train,
            patch_size=patch_size,
            stride=patch_size // 2,
            augment=True
        )
        
        val_dataset = OCTAPatchDataset(
            octa_val, label_val,
            patch_size=patch_size,
            stride=patch_size,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_segmentation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        save_every: int = 10
    ):
        """
        训练分割网络
        
        在后台线程中运行
        """
        def train_fn():
            try:
                self._train_segmentation_impl(
                    train_loader, val_loader,
                    epochs, learning_rate, save_every
                )
            except Exception as e:
                self.state.status = "error"
                self.state.message = str(e)
        
        self.training_thread = threading.Thread(target=train_fn)
        self.training_thread.start()
    
    def _train_segmentation_impl(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        save_every: int
    ):
        """分割网络训练的实际实现"""
        # 初始化模型
        if self.model is None:
            self.model = self.create_segmentation_model()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 损失函数
        criterion = CombinedLoss(dice_weight=0.5)
        
        # 更新状态
        self.state.status = "running"
        self.state.total_epochs = epochs
        self.state.total_steps = len(train_loader) * epochs
        
        start_time = time.time()
        
        for epoch in range(epochs):
            if self.stop_flag.is_set():
                break
            
            # 等待暂停结束
            while self.pause_flag.is_set():
                time.sleep(0.1)
            
            self.state.current_epoch = epoch + 1
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (octa, label) in enumerate(train_loader):
                if self.stop_flag.is_set():
                    break
                
                octa = octa.to(self.device)
                label = label.to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    output = self.model(octa)
                    
                    if isinstance(output, tuple):
                        # 深度监督
                        main_out, ds_outs = output
                        loss = criterion(main_out, label)
                        for ds_out in ds_outs:
                            loss += 0.3 * criterion(ds_out, label)
                    else:
                        loss = criterion(output, label)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                self.state.current_step = epoch * len(train_loader) + batch_idx + 1
            
            train_loss /= len(train_loader)
            self.state.train_loss = train_loss
            self.state.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss, dice_score = self._validate(val_loader, criterion)
            self.state.val_loss = val_loss
            self.state.val_losses.append(val_loss)
            self.state.dice_scores.append(dice_score)
            
            # 更新最佳模型
            if val_loss < self.state.best_loss:
                self.state.best_loss = val_loss
                self._save_checkpoint('best_segmentation.pt')
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'segmentation_epoch_{epoch+1}.pt')
            
            # 更新时间
            self.state.elapsed_time = time.time() - start_time
            self.state.eta = self.state.elapsed_time / (epoch + 1) * (epochs - epoch - 1)
            
            # 更新学习率
            self.scheduler.step()
            
            # 回调通知
            self._notify_progress()
        
        self.state.status = "completed"
        self._save_checkpoint('final_segmentation.pt')
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        dice_scores = []
        
        with torch.no_grad():
            for octa, label in val_loader:
                octa = octa.to(self.device)
                label = label.to(self.device)
                
                output = self.model(octa)
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = criterion(output, label)
                val_loss += loss.item()
                
                # 计算 Dice
                pred = (torch.sigmoid(output) > 0.5).float()
                intersection = (pred * label).sum()
                dice = (2 * intersection + 1) / (pred.sum() + label.sum() + 1)
                dice_scores.append(dice.item())
        
        val_loss /= len(val_loader)
        avg_dice = np.mean(dice_scores)
        
        return val_loss, avg_dice
    
    def train_diffusion(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        save_every: int = 10
    ):
        """训练扩散模型"""
        def train_fn():
            try:
                self._train_diffusion_impl(
                    train_loader, val_loader,
                    epochs, learning_rate, save_every
                )
            except Exception as e:
                self.state.status = "error"
                self.state.message = str(e)
        
        self.training_thread = threading.Thread(target=train_fn)
        self.training_thread.start()
    
    def _train_diffusion_impl(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        save_every: int
    ):
        """扩散模型训练实现"""
        # 初始化模型
        if self.model is None or not isinstance(self.model, OCTADiffusionModel):
            self.model = self.create_diffusion_model()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 更新状态
        self.state.status = "running"
        self.state.total_epochs = epochs
        
        start_time = time.time()
        
        for epoch in range(epochs):
            if self.stop_flag.is_set():
                break
            
            self.state.current_epoch = epoch + 1
            self.model.noise_net.train()
            train_loss = 0.0
            
            for batch_idx, (octa, label) in enumerate(train_loader):
                if self.stop_flag.is_set():
                    break
                
                # octa: 带拖尾 (条件)
                # label: 干净 (目标)
                octa = octa.to(self.device)
                label = label.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 计算扩散损失
                loss = self.model.compute_loss(label, octa)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.state.train_loss = train_loss
            self.state.train_losses.append(train_loss)
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'diffusion_epoch_{epoch+1}.pt')
            
            self.state.elapsed_time = time.time() - start_time
            self._notify_progress()
        
        self.state.status = "completed"
        self._save_checkpoint('final_diffusion.pt')
    
    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'state': self.state.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.models_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str) -> bool:
        """加载检查点"""
        path = self.models_dir / filename
        if not path.exists():
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复状态
        state_dict = checkpoint.get('state', {})
        for key, value in state_dict.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        return True
    
    def stop_training(self):
        """停止训练"""
        self.stop_flag.set()
        if self.training_thread:
            self.training_thread.join(timeout=5)
        self.state.status = "stopped"
    
    def pause_training(self):
        """暂停训练"""
        self.pause_flag.set()
        self.state.status = "paused"
    
    def resume_training(self):
        """恢复训练"""
        self.pause_flag.clear()
        self.state.status = "running"
    
    def get_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        return self.state.to_dict()
    
    def add_progress_callback(self, callback: Callable):
        """添加进度回调"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """通知进度更新"""
        state_dict = self.state.to_dict()
        for callback in self.progress_callbacks:
            try:
                callback(state_dict)
            except Exception:
                pass
