# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 数据管理路由

功能：
1. DICOM/OCTA 数据上传
2. 数据解析和预览
3. 3D 体数据切片获取
4. 网格数据生成与获取
"""

import os
import uuid
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np
import trimesh  # 新增：用于读取/返回网格数据

# 导入核心处理模块
from core.dicom_reader import DicomReader
from core.volume_processor import VolumeProcessor
from core.mesh_generator import MeshGenerator
from core.distortion_correction import DistortionCorrector

router = APIRouter()

# 数据目录配置
DATA_DIR = Path(__file__).parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================================
# 数据模型定义
# ============================================================================

class VolumeInfo(BaseModel):
    """体数据信息模型"""
    id: str
    filename: str
    shape: List[int]
    spacing: List[float]
    origin: List[float]
    axis_labels: List[str]  # 如 ["X", "Y", "Z"]
    value_range: List[float]
    vessel_fraction: float  # 血管占比
    created_at: str

class SliceRequest(BaseModel):
    """切片请求模型"""
    volume_id: str
    axis: str  # "x", "y", "z"
    index: int
    apply_correction: bool = False  # 是否应用畸变矫正

class MeshRequest(BaseModel):
    """网格生成请求模型"""
    threshold: float = 0.5          # 等值面阈值
    smooth: bool = True             # 是否平滑
    decimate_ratio: float = 0.5     # 网格简化比例（0-1，1 表示不简化）

# ============================================================================
# 全局数据存储（生产环境应使用数据库）
# ============================================================================

loaded_volumes = {}

# ============================================================================
# 辅助函数
# ============================================================================

def generate_volume_id() -> str:
    return f"vol_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_volume_metadata(volume_id: str, info: dict):
    meta_path = PROCESSED_DIR / volume_id / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def load_volume_metadata(volume_id: str) -> Optional[dict]:
    meta_path = PROCESSED_DIR / volume_id / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# ============================================================================
# 数据上传和解析接口
# ============================================================================

@router.post("/upload", response_model=VolumeInfo)
async def upload_volume(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    volume_id = generate_volume_id()

    upload_path = UPLOADS_DIR / volume_id
    upload_path.mkdir(parents=True, exist_ok=True)

    file_path = upload_path / file.filename
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    try:
        reader = DicomReader()
        volume_data, metadata = reader.read_volume(str(file_path))

        processor = VolumeProcessor()
        processed_data = processor.preprocess(volume_data)

        processed_path = PROCESSED_DIR / volume_id
        processed_path.mkdir(parents=True, exist_ok=True)
        np.save(processed_path / "volume.npy", processed_data)

        _generate_preview_slices(volume_id, processed_data)

        info = VolumeInfo(
            id=volume_id,
            filename=file.filename,
            shape=list(processed_data.shape),
            spacing=metadata.get('spacing', [1.0, 1.0, 1.0]),
            origin=metadata.get('origin', [0.0, 0.0, 0.0]),
            axis_labels=metadata.get('axis_labels', ['X', 'Y', 'Z']),
            value_range=[float(processed_data.min()), float(processed_data.max())],
            vessel_fraction=float(np.sum(processed_data > 0.5) / processed_data.size),
            created_at=datetime.now().isoformat()
        )

        save_volume_metadata(volume_id, info.model_dump())
        loaded_volumes[volume_id] = info.model_dump()

        return info

    except Exception as e:
        import shutil
        if upload_path.exists():
            shutil.rmtree(upload_path)
        raise HTTPException(status_code=400, detail=f"数据解析失败: {str(e)}")

def _generate_preview_slices(volume_id: str, data: np.ndarray):
    import cv2

    preview_dir = PROCESSED_DIR / volume_id / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for axis, axis_name in enumerate(['x', 'y', 'z']):
        mid_idx = data.shape[axis] // 2

        if axis == 0:
            slice_data = data[mid_idx, :, :]
        elif axis == 1:
            slice_data = data[:, mid_idx, :]
        else:
            slice_data = data[:, :, mid_idx]

        slice_norm = ((slice_data - slice_data.min()) /
                      (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)

        cv2.imwrite(str(preview_dir / f"preview_{axis_name}.png"), slice_norm)

@router.get("/list")
async def list_volumes():
    volumes = []
    for vol_dir in PROCESSED_DIR.iterdir():
        if vol_dir.is_dir():
            meta = load_volume_metadata(vol_dir.name)
            if meta:
                volumes.append(meta)
    return volumes

@router.get("/{volume_id}/info")
async def get_volume_info(volume_id: str):
    meta = load_volume_metadata(volume_id)
    if not meta:
        raise HTTPException(status_code=404, detail="体数据不存在")
    return meta

@router.delete("/{volume_id}")
async def delete_volume(volume_id: str):
    import shutil

    upload_path = UPLOADS_DIR / volume_id
    if upload_path.exists():
        shutil.rmtree(upload_path)

    processed_path = PROCESSED_DIR / volume_id
    if processed_path.exists():
        shutil.rmtree(processed_path)

    if volume_id in loaded_volumes:
        del loaded_volumes[volume_id]

    return {"message": f"体数据 {volume_id} 已删除"}

# ============================================================================
# 切片获取接口
# ============================================================================

@router.get("/{volume_id}/slice/{axis}/{index}")
async def get_slice(
    volume_id: str,
    axis: str,
    index: int,
    apply_correction: bool = Query(False, description="是否应用畸变矫正")
):
    import cv2
    import io

    volume_path = PROCESSED_DIR / volume_id / "volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")

    data = np.load(volume_path)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis.lower() not in axis_map:
        raise HTTPException(status_code=400, detail="无效的轴向，必须是 x, y, z")

    axis_idx = axis_map[axis.lower()]
    if index < 0 or index >= data.shape[axis_idx]:
        raise HTTPException(
            status_code=400,
            detail=f"索引超出范围，{axis} 轴范围: [0, {data.shape[axis_idx] - 1}]"
        )

    if axis_idx == 0:
        slice_data = data[index, :, :]
    elif axis_idx == 1:
        slice_data = data[:, index, :]
    else:
        slice_data = data[:, :, index]

    if apply_correction:
        corrector = DistortionCorrector()
        slice_data = corrector.correct_slice(slice_data, axis=axis.lower())

    slice_norm = ((slice_data - slice_data.min()) /
                  (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)

    _, buffer = cv2.imencode('.png', slice_norm)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename=slice_{axis}_{index}.png"}
    )

@router.get("/{volume_id}/slices/{axis}")
async def get_all_slice_indices(volume_id: str, axis: str):
    meta = load_volume_metadata(volume_id)
    if not meta:
        raise HTTPException(status_code=404, detail="体数据不存在")

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis.lower() not in axis_map:
        raise HTTPException(status_code=400, detail="无效的轴向")

    axis_idx = axis_map[axis.lower()]
    max_index = meta['shape'][axis_idx]

    return {
        "axis": axis,
        "count": max_index,
        "indices": list(range(max_index))
    }

# ============================================================================
# 网格生成与获取接口
# ============================================================================

@router.post("/{volume_id}/mesh")
async def generate_mesh(
    volume_id: str,
    request: MeshRequest = Body(default_factory=MeshRequest)  # 允许空 body
):
    volume_path = PROCESSED_DIR / volume_id / "volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")

    data = np.load(volume_path)

    generator = MeshGenerator()
    mesh = generator.generate(
        data,
        threshold=request.threshold,
        smooth=request.smooth,
        decimate_ratio=request.decimate_ratio
    )

    mesh_dir = PROCESSED_DIR / volume_id / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_filename = f"mesh_t{request.threshold:.2f}.ply"
    mesh_path = mesh_dir / mesh_filename
    mesh.export(str(mesh_path))

    return {
        "message": "网格生成成功",
        "mesh_path": str(mesh_path),
        "download_url": f"/api/data/{volume_id}/mesh/download/{mesh_filename}",
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }

@router.get("/{volume_id}/mesh-data")
async def get_mesh_data(
    volume_id: str,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    smooth: bool = Query(True),
    decimate_ratio: float = Query(0.5, ge=0.0, le=1.0),
    regenerate: bool = Query(False, description="强制重新生成网格")
):
    """
    返回网格的顶点/面数据（前端可直接 Three.js 使用）。
    若已有对应网格文件则直接读取，否则生成后再返回。
    """
    volume_path = PROCESSED_DIR / volume_id / "volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")

    mesh_dir = PROCESSED_DIR / volume_id / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_filename = f"mesh_t{threshold:.2f}.ply"
    mesh_path = mesh_dir / mesh_filename

    mesh = None
    if regenerate or not mesh_path.exists():
        data = np.load(volume_path)
        generator = MeshGenerator()
        mesh = generator.generate(
            data,
            threshold=threshold,
            smooth=smooth,
            decimate_ratio=decimate_ratio
        )
        mesh.export(str(mesh_path))
    else:
        try:
            mesh = trimesh.load(str(mesh_path), process=False)
        except Exception:
            # 读取失败则重新生成
            data = np.load(volume_path)
            generator = MeshGenerator()
            mesh = generator.generate(
                data,
                threshold=threshold,
                smooth=smooth,
                decimate_ratio=decimate_ratio
            )
            mesh.export(str(mesh_path))

    return {
        "vertices": mesh.vertices.tolist(),
        "faces": mesh.faces.tolist(),
        "metadata": {
            "threshold": threshold,
            "smooth": smooth,
            "decimate_ratio": decimate_ratio,
            "vertex_count": int(mesh.vertices.shape[0]),
            "face_count": int(mesh.faces.shape[0]),
            "mesh_file": str(mesh_path)
        }
    }

@router.get("/{volume_id}/mesh/download/{filename}")
async def download_mesh(volume_id: str, filename: str):
    mesh_path = PROCESSED_DIR / volume_id / "meshes" / filename
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="网格文件不存在")

    return FileResponse(
        str(mesh_path),
        media_type="application/octet-stream",
        filename=filename
    )

# ============================================================================
# 体数据导出接口
# ============================================================================

@router.get("/{volume_id}/export/{format}")
async def export_volume(
    volume_id: str,
    format: str = "npy"
):
    volume_path = PROCESSED_DIR / volume_id / "volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")  # 修正 status_code

    if format.lower() == "npy":
        return FileResponse(
            str(volume_path),
            media_type="application/octet-stream",
            filename=f"{volume_id}.npy"
        )
    elif format.lower() == "nii":
        import nibabel as nib

        data = np.load(volume_path)
        _ = load_volume_metadata(volume_id)

        nii_img = nib.Nifti1Image(data, np.eye(4))
        nii_path = PROCESSED_DIR / volume_id / f"{volume_id}.nii.gz"
        nib.save(nii_img, str(nii_path))

        return FileResponse(
            str(nii_path),
            media_type="application/gzip",
            filename=f"{volume_id}.nii.gz"
        )
    else:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")


