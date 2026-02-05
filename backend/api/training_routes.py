# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 训练相关路由

功能：
1. 创建和管理训练任务
2. 启动/停止/暂停训练
3. 查询训练进度
4. 管理模型检查点
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np

# 导入训练管理器
from training.trainer import TrainingManager, TrainingState

router = APIRouter()

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = DATA_DIR / "models"

# 全局训练管理器实例
training_manager: Optional[TrainingManager] = None

# ============================================================================
# 数据模型
# ============================================================================

class TrainingConfig(BaseModel):
    """训练配置"""
    # 数据配置
    volume_id: str
    annotation_project_id: str
    
    # 训练模式
    mode: str = "segmentation"  # "segmentation", "diffusion", "combined"
    
    # 超参数
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    patch_size: int = 32
    
    # Diffusion 特定参数
    diffusion_timesteps: int = 1000
    diffusion_beta_schedule: str = "cosine"  # "linear", "cosine", "quadratic"
    
    # 其他
    save_every: int = 10
    num_workers: int = 4
    use_amp: bool = True  # 混合精度训练

class TrainingTask(BaseModel):
    """训练任务"""
    id: str
    config: TrainingConfig
    status: str = "pending"  # pending, running, paused, completed, error
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

# ============================================================================
# 辅助函数
# ============================================================================

def get_training_manager() -> TrainingManager:
    """获取或创建训练管理器"""
    global training_manager
    if training_manager is None:
        training_manager = TrainingManager(
            data_dir=DATA_DIR,
            models_dir=MODELS_DIR
        )
    return training_manager

def load_training_data(volume_id: str, annotation_project_id: str):
    """加载训练数据"""
    # 加载 OCTA 体数据
    volume_path = PROCESSED_DIR / volume_id / "volume.npy"
    if not volume_path.exists():
        raise ValueError(f"体数据不存在: {volume_id}")
    
    octa_data = np.load(volume_path)
    
    # 加载金标准 mask
    mask_path = ANNOTATIONS_DIR / annotation_project_id / "masks" / "gold_standard.npy"
    if not mask_path.exists():
        raise ValueError(f"金标准 mask 不存在，请先生成")
    
    label_data = np.load(mask_path)
    
    # 验证形状匹配
    if octa_data.shape != label_data.shape:
        raise ValueError(f"数据形状不匹配: OCTA {octa_data.shape}, 标签 {label_data.shape}")
    
    return octa_data, label_data

# ============================================================================
# 训练任务管理接口
# ============================================================================

@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """
    启动训练任务
    
    根据配置选择训练模式：
    - segmentation: 仅训练 3D U-Net 分割网络
    - diffusion: 仅训练 Diffusion 网络
    - combined: 级联训练（先分割后扩散）
    """
    manager = get_training_manager()
    
    # 检查是否有正在进行的训练
    if manager.state.status == "running":
        raise HTTPException(status_code=400, detail="已有训练任务正在运行")
    
    try:
        # 加载数据
        octa_data, label_data = load_training_data(
            config.volume_id, 
            config.annotation_project_id
        )
        
        # 划分训练集和验证集（按 Y 维度划分）
        X, Y, Z = octa_data.shape
        split_idx = int(Y * 0.8)
        
        octa_train = octa_data[:, :split_idx, :]
        label_train = label_data[:, :split_idx, :]
        octa_val = octa_data[:, split_idx:, :]
        label_val = label_data[:, split_idx:, :]
        
        # 准备数据加载器
        train_loader, val_loader = manager.prepare_dataloaders(
            octa_train, label_train,
            octa_val, label_val,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # 根据模式启动训练
        if config.mode == "segmentation":
            manager.model = manager.create_segmentation_model()
            manager.train_segmentation(
                train_loader, val_loader,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                save_every=config.save_every
            )
        
        elif config.mode == "diffusion":
            manager.model = manager.create_diffusion_model(
                timesteps=config.diffusion_timesteps
            )
            manager.train_diffusion(
                train_loader, val_loader,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                save_every=config.save_every
            )
        
        elif config.mode == "combined":
            # 级联训练：先分割后扩散
            # 第一阶段：分割
            manager.model = manager.create_segmentation_model()
            manager.train_segmentation(
                train_loader, val_loader,
                epochs=config.epochs // 2,
                learning_rate=config.learning_rate,
                save_every=config.save_every
            )
            
            # TODO: 第二阶段需要等第一阶段完成后启动
            # 这里简化处理，实际应该使用更复杂的调度
        
        else:
            raise HTTPException(status_code=400, detail=f"未知的训练模式: {config.mode}")
        
        return {
            "message": "训练任务已启动",
            "mode": config.mode,
            "status": "running"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")

@router.post("/stop")
async def stop_training():
    """停止训练"""
    manager = get_training_manager()
    manager.stop_training()
    
    return {"message": "训练已停止", "status": manager.state.status}

@router.post("/pause")
async def pause_training():
    """暂停训练"""
    manager = get_training_manager()
    manager.pause_training()
    
    return {"message": "训练已暂停", "status": manager.state.status}

@router.post("/resume")
async def resume_training():
    """恢复训练"""
    manager = get_training_manager()
    manager.resume_training()
    
    return {"message": "训练已恢复", "status": manager.state.status}

@router.get("/status")
async def get_training_status():
    """获取训练状态"""
    manager = get_training_manager()
    return manager.get_state()

@router.get("/logs")
async def get_training_logs(limit: int = 100):
    """获取训练日志"""
    manager = get_training_manager()
    state = manager.get_state()
    
    return {
        "status": state["status"],
        "current_epoch": state["current_epoch"],
        "total_epochs": state["total_epochs"],
        "train_losses": state["train_losses"][-limit:],
        "val_losses": state["val_losses"][-limit:],
        "dice_scores": state["dice_scores"][-limit:],
        "elapsed_time": state["elapsed_time"],
        "eta": state["eta"],
        "message": state["message"]
    }

# ============================================================================
# 模型检查点管理接口
# ============================================================================

@router.get("/checkpoints")
async def list_checkpoints():
    """列出所有模型检查点"""
    checkpoints = []
    
    if MODELS_DIR.exists():
        for ckpt_file in MODELS_DIR.glob("*.pt"):
            stat = ckpt_file.stat()
            checkpoints.append({
                "name": ckpt_file.name,
                "path": str(ckpt_file),
                "size_mb": stat.st_size / (1024 * 1024),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    return sorted(checkpoints, key=lambda x: x["modified_at"], reverse=True)

@router.delete("/checkpoints/{filename}")
async def delete_checkpoint(filename: str):
    """删除检查点"""
    ckpt_path = MODELS_DIR / filename
    
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="检查点不存在")
    
    ckpt_path.unlink()
    
    return {"message": f"检查点 {filename} 已删除"}

@router.post("/checkpoints/load/{filename}")
async def load_checkpoint(filename: str):
    """加载检查点"""
    manager = get_training_manager()
    
    success = manager.load_checkpoint(filename)
    
    if not success:
        raise HTTPException(status_code=404, detail="检查点不存在或加载失败")
    
    return {"message": f"检查点 {filename} 已加载"}

# ============================================================================
# 训练数据验证接口
# ============================================================================

@router.post("/validate-data")
async def validate_training_data(volume_id: str, annotation_project_id: str):
    """验证训练数据的有效性"""
    try:
        octa_data, label_data = load_training_data(volume_id, annotation_project_id)
        
        # 统计信息
        vessel_voxels = int(label_data.sum())
        total_voxels = label_data.size
        vessel_fraction = vessel_voxels / total_voxels
        
        return {
            "valid": True,
            "octa_shape": list(octa_data.shape),
            "label_shape": list(label_data.shape),
            "octa_range": [float(octa_data.min()), float(octa_data.max())],
            "vessel_voxels": vessel_voxels,
            "vessel_fraction": f"{vessel_fraction * 100:.2f}%",
            "message": "数据验证通过"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": str(e)
        }

# ============================================================================
# 推荐配置接口
# ============================================================================

@router.get("/recommended-config")
async def get_recommended_config(volume_id: str):
    """根据数据大小推荐训练配置"""
    volume_path = PROCESSED_DIR / volume_id / "volume.npy"
    
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")
    
    # 加载数据获取形状
    data = np.load(volume_path)
    X, Y, Z = data.shape
    total_voxels = X * Y * Z
    
    # 根据数据大小推荐配置
    if total_voxels < 50_000_000:  # < 50M 体素
        batch_size = 8
        patch_size = 32
    elif total_voxels < 100_000_000:  # < 100M 体素
        batch_size = 4
        patch_size = 32
    else:  # >= 100M 体素
        batch_size = 2
        patch_size = 48
    
    return {
        "data_shape": [X, Y, Z],
        "total_voxels": total_voxels,
        "recommended": {
            "batch_size": batch_size,
            "patch_size": patch_size,
            "epochs": 100,
            "learning_rate": 1e-4,
            "diffusion_timesteps": 1000,
            "diffusion_beta_schedule": "cosine",
            "save_every": 10
        },
        "gpu_memory_estimate_gb": batch_size * patch_size ** 3 * 4 * 2 / (1024 ** 3)
    }
