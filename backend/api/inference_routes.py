# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 推理相关路由

功能：
1. 使用训练好的模型进行推理
2. 支持分割网络和 Diffusion 网络的单独或级联推理
3. 导出结果（NIfTI、STL 等格式）
"""

import os
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import torch

# 导入模型
from models.unet3d import UNet3D
from models.diffusion import OCTADiffusionModel
from core.mesh_generator import MeshGenerator

router = APIRouter()

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据模型
# ============================================================================

class InferenceConfig(BaseModel):
    """推理配置"""
    volume_id: str
    checkpoint_name: str
    mode: str = "segmentation"  # "segmentation", "diffusion", "combined"
    patch_size: int = 32
    overlap: int = 8
    threshold: float = 0.5
    use_ddim: bool = True
    ddim_steps: int = 50

class InferenceResult(BaseModel):
    """推理结果"""
    id: str
    volume_id: str
    checkpoint_name: str
    mode: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

# ============================================================================
# 推理状态管理
# ============================================================================

# 推理任务状态
inference_tasks: Dict[str, InferenceResult] = {}
current_inference_thread: Optional[threading.Thread] = None

# ============================================================================
# 推理核心函数
# ============================================================================

def sliding_window_inference(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: int = 32,
    overlap: int = 8,
    device: str = 'cuda',
    is_diffusion: bool = False,
    diffusion_config: Optional[Dict] = None
) -> np.ndarray:
    """
    滑动窗口推理
    
    对大型体数据进行分块推理，然后合并结果
    
    参数：
        model: 训练好的模型
        volume: 输入体数据 (X, Y, Z)
        patch_size: patch 大小
        overlap: 重叠区域大小
        device: 计算设备
        is_diffusion: 是否是 Diffusion 模型
        diffusion_config: Diffusion 配置
    
    返回：
        output: 推理结果 (X, Y, Z)
    """
    model.eval()
    
    X, Y, Z = volume.shape
    stride = patch_size - overlap
    
    # 输出和权重累加
    output = np.zeros((X, Y, Z), dtype=np.float32)
    weights = np.zeros((X, Y, Z), dtype=np.float32)
    
    # 创建高斯权重（用于平滑拼接）
    weight_patch = _create_gaussian_weight(patch_size)
    
    # 计算 patch 数量
    x_steps = max(1, (X - patch_size) // stride + 1)
    y_steps = max(1, (Y - patch_size) // stride + 1)
    z_steps = max(1, (Z - patch_size) // stride + 1)
    
    total_patches = x_steps * y_steps * z_steps
    processed = 0
    
    with torch.no_grad():
        for xi, x in enumerate(range(0, X - patch_size + 1, stride)):
            for yi, y in enumerate(range(0, Y - patch_size + 1, stride)):
                for zi, z in enumerate(range(0, Z - patch_size + 1, stride)):
                    # 提取 patch
                    patch = volume[x:x+patch_size, y:y+patch_size, z:z+patch_size]
                    
                    # 转换为 tensor
                    input_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    # 推理
                    if is_diffusion and diffusion_config:
                        # Diffusion 推理
                        pred = model.sample(
                            input_tensor,
                            use_ddim=diffusion_config.get('use_ddim', True),
                            ddim_steps=diffusion_config.get('ddim_steps', 50),
                            progress=False
                        )
                    else:
                        # 分割网络推理
                        pred = model(input_tensor)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        pred = torch.sigmoid(pred)
                    
                    # 转换回 numpy
                    pred_np = pred.squeeze().cpu().numpy()
                    
                    # 累加到输出
                    output[x:x+patch_size, y:y+patch_size, z:z+patch_size] += pred_np * weight_patch
                    weights[x:x+patch_size, y:y+patch_size, z:z+patch_size] += weight_patch
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"推理进度: {processed}/{total_patches} ({100*processed/total_patches:.1f}%)")
    
    # 处理边界（未被覆盖的区域）
    # 简化处理：直接用最近的预测填充
    weights = np.maximum(weights, 1e-8)
    output = output / weights
    
    return output

def _create_gaussian_weight(size: int) -> np.ndarray:
    """创建 3D 高斯权重"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    
    # 高斯权重
    sigma = 0.5
    weight = np.exp(-dist**2 / (2 * sigma**2))
    
    return weight.astype(np.float32)

def run_inference(task_id: str, config: InferenceConfig):
    """
    执行推理任务（在后台线程中运行）
    """
    global inference_tasks
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 更新状态
        inference_tasks[task_id].status = "running"
        
        # 加载体数据
        volume_path = PROCESSED_DIR / config.volume_id / "volume.npy"
        volume = np.load(volume_path)
        
        # 加载模型
        checkpoint_path = MODELS_DIR / config.checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 根据模式选择模型
        if config.mode == "segmentation":
            model = UNet3D(
                in_channels=1,
                out_channels=1,
                base_features=32,
                use_attention=True
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            output = sliding_window_inference(
                model, volume,
                patch_size=config.patch_size,
                overlap=config.overlap,
                device=device,
                is_diffusion=False
            )
            
        elif config.mode == "diffusion":
            model = OCTADiffusionModel(
                timesteps=1000,
                beta_schedule='cosine',
                base_features=64,
                device=device
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            output = sliding_window_inference(
                model, volume,
                patch_size=config.patch_size,
                overlap=config.overlap,
                device=device,
                is_diffusion=True,
                diffusion_config={
                    'use_ddim': config.use_ddim,
                    'ddim_steps': config.ddim_steps
                }
            )
            
        elif config.mode == "combined":
            # 级联推理：先分割后扩散
            # 第一阶段：分割
            seg_model = UNet3D(
                in_channels=1,
                out_channels=1,
                base_features=32,
                use_attention=True
            ).to(device)
            # 假设分割和扩散模型在同一个 checkpoint 中
            seg_model.load_state_dict(checkpoint['segmentation_state_dict'])
            
            seg_output = sliding_window_inference(
                seg_model, volume,
                patch_size=config.patch_size,
                overlap=config.overlap,
                device=device,
                is_diffusion=False
            )
            
            # 第二阶段：扩散精修
            diff_model = OCTADiffusionModel(
                timesteps=1000,
                beta_schedule='cosine',
                base_features=64,
                device=device
            ).to(device)
            diff_model.load_state_dict(checkpoint['diffusion_state_dict'])
            
            output = sliding_window_inference(
                diff_model, seg_output,
                patch_size=config.patch_size,
                overlap=config.overlap,
                device=device,
                is_diffusion=True,
                diffusion_config={
                    'use_ddim': config.use_ddim,
                    'ddim_steps': config.ddim_steps
                }
            )
        
        else:
            raise ValueError(f"未知的推理模式: {config.mode}")
        
        # 二值化
        binary_output = (output > config.threshold).astype(np.uint8)
        
        # 保存结果
        result_dir = RESULTS_DIR / task_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存概率图
        np.save(result_dir / "probability.npy", output)
        
        # 保存二值 mask
        np.save(result_dir / "binary_mask.npy", binary_output)
        
        # 计算指标
        vessel_voxels = int(binary_output.sum())
        total_voxels = binary_output.size
        
        metrics = {
            "vessel_voxels": vessel_voxels,
            "vessel_fraction": vessel_voxels / total_voxels,
            "output_range": [float(output.min()), float(output.max())],
            "threshold": config.threshold
        }
        
        # 保存元信息
        with open(result_dir / "metadata.json", 'w') as f:
            json.dump({
                "task_id": task_id,
                "config": config.model_dump(),
                "metrics": metrics,
                "shape": list(binary_output.shape)
            }, f, indent=2)
        
        # 更新任务状态
        inference_tasks[task_id].status = "completed"
        inference_tasks[task_id].completed_at = datetime.now().isoformat()
        inference_tasks[task_id].result_path = str(result_dir)
        inference_tasks[task_id].metrics = metrics
        
    except Exception as e:
        inference_tasks[task_id].status = "error"
        inference_tasks[task_id].metrics = {"error": str(e)}

# ============================================================================
# 推理 API 接口
# ============================================================================

@router.post("/start", response_model=InferenceResult)
async def start_inference(config: InferenceConfig, background_tasks: BackgroundTasks):
    """
    启动推理任务
    """
    global current_inference_thread
    
    # 验证输入
    volume_path = PROCESSED_DIR / config.volume_id / "volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")
    
    checkpoint_path = MODELS_DIR / config.checkpoint_name
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="模型检查点不存在")
    
    # 检查是否有正在进行的推理
    if current_inference_thread and current_inference_thread.is_alive():
        raise HTTPException(status_code=400, detail="已有推理任务正在运行")
    
    # 创建任务
    task_id = f"inf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    result = InferenceResult(
        id=task_id,
        volume_id=config.volume_id,
        checkpoint_name=config.checkpoint_name,
        mode=config.mode,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    inference_tasks[task_id] = result
    
    # 在后台启动推理
    current_inference_thread = threading.Thread(
        target=run_inference,
        args=(task_id, config)
    )
    current_inference_thread.start()
    
    return result

@router.get("/status/{task_id}")
async def get_inference_status(task_id: str):
    """获取推理任务状态"""
    if task_id not in inference_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return inference_tasks[task_id]

@router.get("/list")
async def list_inference_results():
    """列出所有推理结果"""
    results = []
    
    if RESULTS_DIR.exists():
        for result_dir in RESULTS_DIR.iterdir():
            if result_dir.is_dir():
                meta_path = result_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    results.append(meta)
    
    return results

@router.get("/result/{task_id}/slice/{axis}/{index}")
async def get_result_slice(task_id: str, axis: str, index: int):
    """获取推理结果的切片图像"""
    import cv2
    import io
    from fastapi.responses import StreamingResponse
    
    result_dir = RESULTS_DIR / task_id
    prob_path = result_dir / "probability.npy"
    
    if not prob_path.exists():
        raise HTTPException(status_code=404, detail="结果不存在")
    
    data = np.load(prob_path)
    
    # 提取切片
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower())
    
    if axis_idx is None:
        raise HTTPException(status_code=400, detail="无效的轴向")
    
    if index < 0 or index >= data.shape[axis_idx]:
        raise HTTPException(status_code=400, detail="索引超出范围")
    
    if axis_idx == 0:
        slice_data = data[index, :, :]
    elif axis_idx == 1:
        slice_data = data[:, index, :]
    else:
        slice_data = data[:, :, index]
    
    # 转换为图像
    slice_img = (slice_data * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', slice_img)
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png"
    )

@router.get("/result/{task_id}/download/{format}")
async def download_result(task_id: str, format: str = "npy"):
    """下载推理结果"""
    result_dir = RESULTS_DIR / task_id
    
    if format.lower() == "npy":
        file_path = result_dir / "binary_mask.npy"
        media_type = "application/octet-stream"
        filename = f"{task_id}_mask.npy"
        
    elif format.lower() == "nii":
        import nibabel as nib
        
        mask = np.load(result_dir / "binary_mask.npy")
        nii_img = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))
        
        nii_path = result_dir / f"{task_id}.nii.gz"
        nib.save(nii_img, str(nii_path))
        
        file_path = nii_path
        media_type = "application/gzip"
        filename = f"{task_id}.nii.gz"
        
    elif format.lower() in ["stl", "ply"]:
        # 生成网格
        mask = np.load(result_dir / "binary_mask.npy")
        
        generator = MeshGenerator()
        mesh = generator.generate(mask, threshold=0.5, smooth=True)
        
        mesh_path = result_dir / f"{task_id}.{format.lower()}"
        mesh.export(str(mesh_path))
        
        file_path = mesh_path
        media_type = "application/octet-stream"
        filename = f"{task_id}.{format.lower()}"
        
    else:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename
    )

@router.delete("/result/{task_id}")
async def delete_result(task_id: str):
    """删除推理结果"""
    import shutil
    
    result_dir = RESULTS_DIR / task_id
    
    if result_dir.exists():
        shutil.rmtree(result_dir)
    
    if task_id in inference_tasks:
        del inference_tasks[task_id]
    
    return {"message": f"结果 {task_id} 已删除"}
