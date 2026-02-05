# -*- coding: utf-8 -*-
"""
OCTA 血管标注 - V12版本

核心思路（用户提出）：
- 噪点：亮度高，但面积小
- 血管：亮度中等，但面积大（连续区域）

方法：
1. 从高亮度向低亮度扫描
2. 在每个阈值下找连通域
3. 当连通域面积在"血管大小范围"内时，标记为候选
4. 对候选区域做椭圆拟合
"""

CODE_VERSION = "V12_BRIGHT_AREA"

import os
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import math

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
from core.artifact_correction import ArtifactCorrection

router = APIRouter()

DATA_DIR = Path(__file__).parent.parent / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[ANNOTATION] 版本: {CODE_VERSION}")

# ============================================================================
# 数据模型
# ============================================================================

class AnnotationPoint(BaseModel):
    x: float
    y: float

class EllipseAnnotation(BaseModel):
    id: str
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    rotation: float = 0.0
    slice_index: int
    axis: str
    vessel_id: Optional[str] = None
    confidence: float = 1.0
    is_manual: bool = False
    created_at: str = ""

class BrushStroke(BaseModel):
    id: str
    points: List[AnnotationPoint]
    brush_size: float
    is_eraser: bool = False
    slice_index: int
    axis: str

class AnnotationLayer(BaseModel):
    slice_index: int
    axis: str
    ellipses: List[EllipseAnnotation] = []
    brush_strokes: List[BrushStroke] = []

class AnnotationProject(BaseModel):
    id: str
    volume_id: str
    name: str
    created_at: str
    modified_at: str
    layers: Dict[str, AnnotationLayer] = {}
    undo_stack: List[Dict] = []
    redo_stack: List[Dict] = []

class CreateProjectRequest(BaseModel):
    volume_id: str
    name: str

class SaveAnnotationRequest(BaseModel):
    project_id: str
    layer_key: str
    ellipses: List[EllipseAnnotation] = []
    brush_strokes: List[BrushStroke] = []

class AutoFitAtPointRequest(BaseModel):
    slice_index: int
    axis: str
    click_x: float
    click_y: float
    roi_size: int = 60
    method: str = "auto"

class AutoLabelRequest(BaseModel):
    axis: str = "y"
    min_radius: int = 3
    max_radius: int = 15
    continuity_threshold: float = 20.0
    min_slices: int = 1
    regen: bool = True
    sensitivity: str = "medium"
    use_clahe: bool = True
    detect_method: str = "bright_area"

class ArtifactCorrectionRequest(BaseModel):
    slice_index: int
    axis: str
    center_x: float
    center_y: float
    major_axis: float
    minor_axis: float
    roi_size: int = 80
    use_spatial_continuity: bool = True

# ============================================================================
# 项目存储
# ============================================================================

def get_project_path(project_id: str) -> Path:
    return ANNOTATIONS_DIR / project_id / "project.json"

def load_project(project_id: str) -> Optional[AnnotationProject]:
    path = get_project_path(project_id)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return AnnotationProject(**json.load(f))
    except:
        return None

def save_project(project: AnnotationProject):
    path = get_project_path(project.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(project.model_dump(), f, ensure_ascii=False, indent=2)


# ============================================================================
# 核心检测算法 V12：基于亮区域面积
# ============================================================================

def detect_vessels_by_bright_area(
    img: np.ndarray,
    min_r: int = 3,
    max_r: int = 15,
    sensitivity: str = "medium"
) -> List[Dict]:
    """
    基于亮区域面积的血管检测 - V12.1
    
    核心改进：
    - 高阈值下第一个出现的valid区域 = 最可能是血管
    - 因为血管比背景亮，但比噪点暗
    - 噪点面积太小会被过滤，血管面积合适会被保留
    
    置信度计算：
    - 阈值越高（越亮），置信度越高
    - 面积越接近理想范围，置信度越高
    """
    # 确保是 uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    h, w = img.shape
    
    # 灵敏度参数
    params = {
        "low": {"thresh_step": 8, "max_count": 10},
        "medium": {"thresh_step": 5, "max_count": 20},
        "high": {"thresh_step": 3, "max_count": 40},
    }.get(sensitivity, {"thresh_step": 5, "max_count": 20})
    
    # 计算面积范围 - 更严格一些
    min_area = math.pi * (min_r ** 2) * 0.5
    max_area = math.pi * (max_r ** 2) * 2.0
    
    # 预处理：去除竖条纹
    col_mean = img.mean(axis=0, keepdims=True).astype(np.float32)
    global_mean = img.mean()
    processed = np.clip(img.astype(np.float32) - col_mean + global_mean, 0, 255).astype(np.uint8)
    
    # 轻微平滑
    processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
    
    # 已找到的候选位置（用于去重）
    found_centers = []
    candidates = []
    
    # 计算阈值范围
    img_max = int(processed.max())
    img_mean = int(processed.mean())
    
    # 从高阈值向低阈值扫描
    for thresh in range(img_max - 5, img_mean, -params["thresh_step"]):
        _, binary = cv2.threshold(processed, thresh, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 面积过滤
            if area < min_area or area > max_area:
                continue
            
            cx, cy = centroids[i]
            
            # 边界检查
            if cx < min_r or cx > w - min_r or cy < min_r or cy > h - min_r:
                continue
            
            # 宽高比检查
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 3.5:
                continue
            
            # 去重：检查是否与已找到的候选重叠
            is_duplicate = False
            for (fcx, fcy) in found_centers:
                dist = math.sqrt((cx - fcx)**2 + (cy - fcy)**2)
                if dist < max_r:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # 提取轮廓并拟合椭圆
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours or len(contours[0]) < 5:
                continue
            
            try:
                ellipse = cv2.fitEllipse(contours[0])
                (ecx, ecy), (axis1, axis2), angle = ellipse
                rx, ry = axis1 / 2, axis2 / 2
                
                if rx < ry:
                    rx, ry = ry, rx
                    angle = (angle + 90) % 180
                
                # 尺寸检查
                if min(rx, ry) < min_r * 0.3 or max(rx, ry) > max_r * 2.5:
                    continue
                
                # 关键改进：置信度基于阈值高低
                # 阈值越高 = 区域越亮 = 越可能是血管（而不是背景噪声合并）
                thresh_score = (thresh - img_mean) / (img_max - img_mean)  # 0~1
                
                # 面积匹配度
                ideal_area = math.pi * ((min_r + max_r) / 2) ** 2
                area_score = max(0, 1 - abs(area - ideal_area) / ideal_area)
                
                # 最终置信度：阈值权重更高
                confidence = 0.3 + thresh_score * 0.5 + area_score * 0.2
                confidence = min(0.95, max(0.4, confidence))
                
                found_centers.append((cx, cy))
                candidates.append({
                    "cx": float(ecx),
                    "cy": float(ecy),
                    "rx": float(rx),
                    "ry": float(ry),
                    "angle": float(angle),
                    "confidence": confidence,
                    "threshold": thresh,
                    "area": float(area),
                    "method": "bright_area_v12.1"
                })
                
            except cv2.error:
                continue
    
    # 按置信度排序（高阈值的会排在前面）
    candidates = sorted(candidates, key=lambda x: -x["confidence"])
    
    return candidates[:params["max_count"]]


def auto_fit_at_click(img, click_x, click_y, roi_size=80, min_r=3, max_r=25):
    """
    点击位置自动拟合 - 基于flood fill
    
    核心思路：
    1. 从点击位置开始
    2. 找到点击位置的亮度值
    3. 用flood fill找到与点击位置亮度相近的连通区域
    4. 对这个区域做椭圆拟合
    """
    h, w = img.shape
    click_x, click_y = int(click_x), int(click_y)
    
    if not (0 <= click_x < w and 0 <= click_y < h):
        return None
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # 去除竖条纹
    col_mean = img.mean(axis=0, keepdims=True).astype(np.float32)
    processed = np.clip(img.astype(np.float32) - col_mean + img.mean(), 0, 255).astype(np.uint8)
    
    # 轻微平滑
    processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
    
    # 提取ROI
    half = roi_size // 2
    x1, x2 = max(0, click_x - half), min(w, click_x + half)
    y1, y2 = max(0, click_y - half), min(h, click_y + half)
    
    roi = processed[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None
    
    local_x, local_y = click_x - x1, click_y - y1
    click_val = int(roi[local_y, local_x])
    
    # 方法1：从点击位置的亮度向下尝试不同阈值
    # 找到包含点击位置的最小连通域
    best_result = None
    best_area = float('inf')
    
    # 计算局部统计
    local_mean = float(roi.mean())
    
    # 尝试不同的阈值，从点击位置亮度向下
    for thresh in range(max(click_val - 10, int(local_mean)), int(local_mean) - 30, -5):
        _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
        
        # 检查点击位置是否在白色区域
        if binary[local_y, local_x] == 0:
            continue
        
        # 用flood fill从点击位置开始填充，找到连通区域
        mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
        flood_img = binary.copy()
        
        # flood fill会把连通区域标记为新值(128)
        cv2.floodFill(flood_img, mask, (local_x, local_y), 128)
        
        # 提取被填充的区域
        filled_region = (flood_img == 128).astype(np.uint8) * 255
        
        # 找轮廓
        contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        contour = contours[0]
        if len(contour) < 5:
            continue
        
        area = cv2.contourArea(contour)
        
        # 面积检查
        min_area = math.pi * (min_r ** 2) * 0.3
        max_area = math.pi * (max_r ** 2) * 3
        
        if area < min_area or area > max_area:
            continue
        
        # 宽高比检查
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 4:
            continue
        
        # 选择面积最合适的（不是最小的，而是在合理范围内的）
        ideal_area = math.pi * ((min_r + max_r) / 2) ** 2
        area_diff = abs(area - ideal_area)
        
        try:
            ellipse = cv2.fitEllipse(contour)
            (ecx, ecy), (axis1, axis2), angle = ellipse
            rx, ry = axis1 / 2, axis2 / 2
            
            if rx < ry:
                rx, ry = ry, rx
                angle = (angle + 90) % 180
            
            # 检查椭圆中心是否接近点击位置
            dist_to_click = math.sqrt((ecx - local_x)**2 + (ecy - local_y)**2)
            
            # 评分：面积接近理想值 + 中心接近点击位置
            score = area_diff + dist_to_click * 10
            
            if best_result is None or score < best_area:
                best_area = score
                best_result = {
                    "cx": float(ecx + x1),
                    "cy": float(ecy + y1),
                    "rx": float(rx),
                    "ry": float(ry),
                    "angle": float(angle),
                    "confidence": 0.8,
                    "threshold": thresh,
                    "area": area
                }
        except cv2.error:
            continue
    
    return best_result


# ============================================================================
# API 路由
# ============================================================================

@router.post("/projects", response_model=AnnotationProject)
async def create_project(request: CreateProjectRequest):
    volume_path = PROCESSED_DIR / request.volume_id / "volume.npy"
    if not volume_path.exists():
        raise HTTPException(status_code=404, detail="体数据不存在")
    project_id = f"ann_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()
    project = AnnotationProject(id=project_id, volume_id=request.volume_id, name=request.name, created_at=now, modified_at=now)
    save_project(project)
    return project

@router.get("/projects")
async def list_projects():
    projects = []
    if ANNOTATIONS_DIR.exists():
        for proj_dir in ANNOTATIONS_DIR.iterdir():
            if proj_dir.is_dir():
                project = load_project(proj_dir.name)
                if project:
                    projects.append({"id": project.id, "name": project.name, "volume_id": project.volume_id})
    return projects

@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project.model_dump()

@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    import shutil
    proj_dir = ANNOTATIONS_DIR / project_id
    if proj_dir.exists():
        shutil.rmtree(proj_dir)
    return {"message": "已删除"}

@router.post("/projects/{project_id}/save")
async def save_annotation(project_id: str, request: SaveAnnotationRequest):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    parts = request.layer_key.split('_')
    layer = AnnotationLayer(slice_index=int(parts[1]), axis=parts[0], ellipses=request.ellipses, brush_strokes=request.brush_strokes)
    project.layers[request.layer_key] = layer
    project.modified_at = datetime.now().isoformat()
    save_project(project)
    return {"message": "已保存"}

@router.get("/projects/{project_id}/layer/{layer_key}")
async def get_layer(project_id: str, layer_key: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if layer_key not in project.layers:
        return {"ellipses": [], "brush_strokes": [], "slice_index": 0, "axis": "y"}
    layer = project.layers[layer_key]
    return {"ellipses": [e.model_dump() for e in layer.ellipses], "brush_strokes": [s.model_dump() for s in layer.brush_strokes]}

@router.post("/projects/{project_id}/undo")
async def undo(project_id: str):
    return {"message": "无操作"}

@router.post("/projects/{project_id}/redo")
async def redo(project_id: str):
    return {"message": "无操作"}

@router.post("/projects/{project_id}/generate-mask")
async def generate_gold_standard_mask(project_id: str):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    volume_path = PROCESSED_DIR / project.volume_id / "volume.npy"
    volume = np.load(volume_path)
    X, Y, Z = volume.shape
    mask = np.zeros((X, Y, Z), dtype=np.uint8)
    
    for layer_key, layer in project.layers.items():
        axis, slice_idx = layer.axis, layer.slice_index
        if axis == 'x': slice_shape = (Y, Z)
        elif axis == 'y': slice_shape = (X, Z)
        else: slice_shape = (X, Y)
        slice_mask = np.zeros(slice_shape, dtype=np.uint8)
        for e in layer.ellipses:
            cv2.ellipse(slice_mask, (int(e.center_x), int(e.center_y)), 
                       (max(1, int(e.radius_x)), max(1, int(e.radius_y))), e.rotation, 0, 360, 1, -1)
        if axis == 'x': mask[slice_idx, :, :] = np.maximum(mask[slice_idx, :, :], slice_mask)
        elif axis == 'y': mask[:, slice_idx, :] = np.maximum(mask[:, slice_idx, :], slice_mask)
        else: mask[:, :, slice_idx] = np.maximum(mask[:, :, slice_idx], slice_mask)
    
    mask_dir = ANNOTATIONS_DIR / project_id / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_dir / "gold_standard.npy", mask)
    return {"message": "已生成", "vessel_voxels": int(mask.sum())}

@router.get("/projects/{project_id}/mask")
async def get_mask(project_id: str):
    p = ANNOTATIONS_DIR / project_id / "masks" / "gold_standard.npy"
    if not p.exists():
        raise HTTPException(status_code=404, detail="不存在")
    return FileResponse(str(p))

@router.post("/projects/{project_id}/auto-fit-at-point")
async def auto_fit_at_point_api(project_id: str, request: AutoFitAtPointRequest):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    volume_path = PROCESSED_DIR / project.volume_id / "volume.npy"
    vol = np.load(volume_path).astype(np.float32)
    vol_norm = ((vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255).astype(np.uint8)
    
    ax = {'x': 0, 'y': 1, 'z': 2}.get(request.axis, 1)
    if ax == 0: sl = vol_norm[request.slice_index, :, :]
    elif ax == 1: sl = vol_norm[:, request.slice_index, :]
    else: sl = vol_norm[:, :, request.slice_index]
    
    result = auto_fit_at_click(sl, int(request.click_x), int(request.click_y), request.roi_size)
    if not result:
        raise HTTPException(status_code=400, detail="未检测到椭圆")
    return {"center_x": result["cx"], "center_y": result["cy"], "radius_x": result["rx"], 
            "radius_y": result["ry"], "rotation": result.get("angle", 0), "confidence": result.get("confidence", 0.8)}


@router.post("/projects/{project_id}/autolabel")
async def auto_label(project_id: str, req: AutoLabelRequest = Body(...)):
    import time
    start_time = time.time()
    
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    volume_path = PROCESSED_DIR / project.volume_id / "volume.npy"
    vol = np.load(volume_path).astype(np.float32)
    vol_norm = ((vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255).astype(np.uint8)
    
    ax = {'x': 0, 'y': 1, 'z': 2}.get(req.axis, 1)
    total = vol_norm.shape[ax]
    
    if req.regen:
        for k in list(project.layers.keys()):
            if k.startswith(f"{req.axis}_"):
                project.layers.pop(k)
    
    added = 0
    for idx in range(total):
        if ax == 0: sl = vol_norm[idx, :, :]
        elif ax == 1: sl = vol_norm[:, idx, :]
        else: sl = vol_norm[:, :, idx]
        
        cands = detect_vessels_by_bright_area(sl, req.min_radius, req.max_radius, req.sensitivity)
        
        if cands:
            lk = f"{req.axis}_{idx}"
            ellipses = [EllipseAnnotation(
                id=f"auto_{uuid.uuid4().hex[:8]}", center_x=c["cx"], center_y=c["cy"],
                radius_x=c["rx"], radius_y=c["ry"], rotation=c.get("angle", 0),
                slice_index=idx, axis=req.axis, confidence=c.get("confidence", 0.7),
                is_manual=False, created_at=datetime.now().isoformat()
            ) for c in cands]
            added += len(ellipses)
            project.layers[lk] = AnnotationLayer(slice_index=idx, axis=req.axis, ellipses=ellipses)
        
        if idx % 100 == 0:
            print(f"[V12] {idx}/{total}, 已检测 {added}")
    
    project.modified_at = datetime.now().isoformat()
    save_project(project)
    
    return {"message": "完成", "code_version": CODE_VERSION, "added_ellipses": added, 
            "total_slices": total, "time_seconds": round(time.time() - start_time, 1)}


@router.post("/fit-ellipse")
async def fit_ellipse_from_points(points: List[AnnotationPoint]):
    if len(points) < 5:
        raise HTTPException(status_code=400, detail="需要至少5个点")
    pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    ellipse = cv2.fitEllipse(pts)
    (cx, cy), (axis1, axis2), angle = ellipse
    rx, ry = axis1 / 2, axis2 / 2
    if rx < ry: rx, ry, angle = ry, rx, (angle + 90) % 180
    return {"center_x": float(cx), "center_y": float(cy), "radius_x": float(rx), "radius_y": float(ry), "rotation": float(angle)}


@router.get("/version")
async def get_version():
    return {"version": CODE_VERSION, "features": ["亮度×面积筛选", "多阈值扫描", "自动去重"]}


@router.post("/projects/{project_id}/debug-detection")
async def debug_detection(project_id: str, slice_index: int = Body(...), axis: str = Body("y")):
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    volume_path = PROCESSED_DIR / project.volume_id / "volume.npy"
    vol = np.load(volume_path).astype(np.float32)
    vol_norm = ((vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255).astype(np.uint8)
    
    ax = {'x': 0, 'y': 1, 'z': 2}.get(axis, 1)
    if ax == 0: img = vol_norm[slice_index, :, :]
    elif ax == 1: img = vol_norm[:, slice_index, :]
    else: img = vol_norm[:, :, slice_index]
    
    h, w = img.shape
    
    # 预处理
    col_mean = img.mean(axis=0, keepdims=True).astype(np.float32)
    processed = np.clip(img.astype(np.float32) - col_mean + img.mean(), 0, 255).astype(np.uint8)
    processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
    
    debug = {
        "version": CODE_VERSION,
        "image": {
            "shape": [h, w],
            "min": int(img.min()),
            "max": int(img.max()),
            "mean": round(float(img.mean()), 1)
        },
        "processed": {
            "min": int(processed.min()),
            "max": int(processed.max()),
            "mean": round(float(processed.mean()), 1)
        },
        "threshold_scan": [],
        "final_result": []
    }
    
    # 面积范围
    min_r, max_r = 3, 15
    min_area = math.pi * (min_r * 0.5) ** 2
    max_area = math.pi * (max_r * 1.5) ** 2
    
    # 扫描几个典型阈值
    img_max = int(processed.max())
    img_mean = int(processed.mean())
    
    for thresh in range(img_max - 5, img_mean, -15):
        _, binary = cv2.threshold(processed, thresh, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # 统计各种面积的连通域数量
        small_count = 0  # 面积 < min_area
        valid_count = 0  # min_area <= 面积 <= max_area
        large_count = 0  # 面积 > max_area
        
        valid_components = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                small_count += 1
            elif area > max_area:
                large_count += 1
            else:
                valid_count += 1
                cx, cy = centroids[i]
                valid_components.append({
                    "center": [round(float(cx), 1), round(float(cy), 1)],
                    "area": int(area),
                    "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])]
                })
        
        debug["threshold_scan"].append({
            "threshold": thresh,
            "total_components": num_labels - 1,
            "small": small_count,
            "valid": valid_count,
            "large": large_count,
            "valid_details": valid_components[:5]  # 只显示前5个
        })
    
    # 最终检测结果
    results = detect_vessels_by_bright_area(img, min_r, max_r, "medium")
    debug["final_result"] = [{
        "cx": round(r["cx"], 1),
        "cy": round(r["cy"], 1),
        "rx": round(r["rx"], 1),
        "ry": round(r["ry"], 1),
        "threshold": r.get("threshold", 0),
        "area": round(r.get("area", 0), 1),
        "confidence": round(r.get("confidence", 0), 2)
    } for r in results[:10]]
    
    debug["summary"] = {
        "min_area": round(min_area, 1),
        "max_area": round(max_area, 1),
        "detected_count": len(results)
    }
    
    return debug


# ============================================================================
# 伪影矫正 API
# ============================================================================

@router.post("/projects/{project_id}/artifact-correction")
async def artifact_correction(project_id: str, request: ArtifactCorrectionRequest):
    """
    OCTA拖尾伪影矫正API

    基于用户输入的长短轴，自动计算真实圆形血管截面
    """
    import traceback

    try:
        print(f"[ARTIFACT_CORRECTION] 开始处理请求: project_id={project_id}, request={request}")

        project = load_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="项目不存在")

        # 检查体积数据文件是否存在
        volume_path = PROCESSED_DIR / project.volume_id / "volume.npy"
        if not volume_path.exists():
            raise HTTPException(status_code=404, detail=f"体积数据文件不存在: {volume_path}")

        print(f"[ARTIFACT_CORRECTION] 加载体积数据: {volume_path}")
        vol = np.load(volume_path).astype(np.float32)
        vol_norm = ((vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255).astype(np.uint8)

        # 验证切片索引
        ax = {'x': 0, 'y': 1, 'z': 2}.get(request.axis, 1)
        if ax == 0:
            max_slice = vol_norm.shape[0] - 1
            if request.slice_index < 0 or request.slice_index > max_slice:
                raise HTTPException(status_code=400, detail=f"切片索引超出范围: {request.slice_index}/{max_slice}")
            slice_img = vol_norm[request.slice_index, :, :]
        elif ax == 1:
            max_slice = vol_norm.shape[1] - 1
            if request.slice_index < 0 or request.slice_index > max_slice:
                raise HTTPException(status_code=400, detail=f"切片索引超出范围: {request.slice_index}/{max_slice}")
            slice_img = vol_norm[:, request.slice_index, :]
        else:
            max_slice = vol_norm.shape[2] - 1
            if request.slice_index < 0 or request.slice_index > max_slice:
                raise HTTPException(status_code=400, detail=f"切片索引超出范围: {request.slice_index}/{max_slice}")
            slice_img = vol_norm[:, :, request.slice_index]

        print(f"[ARTIFACT_CORRECTION] 切片形状: {slice_img.shape}")

        # 获取相邻切片信息用于空间连贯性
        slice_context = []
        if request.use_spatial_continuity:
            for offset in [-2, -1, 1, 2]:
                neighbor_idx = request.slice_index + offset
                layer_key = f"{request.axis}_{neighbor_idx}"

                if layer_key in project.layers and project.layers[layer_key].ellipses:
                    # 只使用已矫正的圆形标注
                    circles = [e for e in project.layers[layer_key].ellipses
                              if hasattr(e, 'correction_applied') and e.correction_applied and
                              abs(e.radius_x - e.radius_y) < 2.0]
                    if circles:
                        circle = circles[0]  # 取第一个圆形
                        slice_context.append({
                            'center_x': circle.center_x,
                            'center_y': circle.center_y,
                            'radius': circle.radius_x
                        })

        print(f"[ARTIFACT_CORRECTION] 相邻切片上下文: {len(slice_context)} 个")

        # 执行伪影矫正
        print(f"[ARTIFACT_CORRECTION] 实例化ArtifactCorrection")
        corrector = ArtifactCorrection()
        result = corrector.generate_circle_annotation(
            slice_img,
            request.center_x,
            request.center_y,
            request.major_axis,
            request.minor_axis,
            request.slice_index,
            request.axis,
            project_id,
            slice_context if slice_context else None
        )

        print(f"[ARTIFACT_CORRECTION] 矫正完成: {result}")
        return result

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 捕获所有其他异常并记录详细错误
        error_detail = f"内部服务器错误: {str(e)}\n{traceback.format_exc()}"
        print(f"[ARTIFACT_CORRECTION] ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=f"伪影矫正失败: {str(e)}")


@router.post("/projects/{project_id}/save-corrected-annotation")
async def save_corrected_annotation(project_id: str, request: SaveAnnotationRequest):
    """
    保存矫正后的圆形标注

    与原有save_annotation接口兼容，但优先处理圆形标注
    """
    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    parts = request.layer_key.split('_')
    layer = AnnotationLayer(
        slice_index=int(parts[1]),
        axis=parts[0],
        ellipses=request.ellipses,
        brush_strokes=request.brush_strokes
    )

    # 智能合并：保留原有的手动标注，更新矫正后的圆形
    existing_layer = project.layers.get(request.layer_key)
    if existing_layer:
        # 保留手动标注的椭圆
        manual_ellipses = [e for e in existing_layer.ellipses if e.is_manual]
        # 保留画笔标注
        existing_brush_strokes = existing_layer.brush_strokes

        # 合并：手动标注 + 新的矫正圆形 + 原有画笔
        layer.ellipses = manual_ellipses + request.ellipses
        layer.brush_strokes = existing_brush_strokes + request.brush_strokes

    project.layers[request.layer_key] = layer
    project.modified_at = datetime.now().isoformat()
    save_project(project)

    return {"message": "矫正标注已保存"}
