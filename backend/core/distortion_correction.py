# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 几何畸变矫正模块

================================================================================
OCT 纵向畸变问题深度分析
================================================================================

【问题根源】

OCT（光学相干断层扫描）成像的纵向（深度）刻度是基于 **光学路径长度（OPL）** 
而非 **物理长度**。这导致了以下问题：

1. 光学路径长度定义：
   
   OPL = n × d
   
   其中：
   - n: 介质折射率（对于皮肤组织，n ≈ 1.38-1.45）
   - d: 物理距离

2. OCT 成像原理：
   - OCT 测量的是光的飞行时间（Time of Flight）
   - 系统将飞行时间转换为距离时，使用的是 **假定的折射率**
   - 当假定折射率与实际折射率不同时，深度测量会出现误差

3. 倾斜导致的畸变：
   
   当样品表面存在倾斜角 θ 时：
   
   ┌─────────────────────────────────────┐
   │          入射光束                    │
   │              ↓                       │
   │              │                       │
   │    ─────────/\─────────  倾斜表面    │
   │            /  \                      │
   │           /    \                     │
   │          /      \  真实光路          │
   │         /        \                   │
   │        ●          ●  血管截面        │
   │         \        /                   │
   │          \      /                    │
   │           \    /                     │
   └─────────────────────────────────────┘
   
   - 真实物理深度: d_true
   - 光学路径长度: OPL = n × d_true / cos(θ)
   - OCT 显示深度: d_display = OPL / n_system
   
   由于光路倾斜穿过样品，光程变长，导致：
   - 圆形血管截面被拉伸成椭圆形
   - 出现"拖尾"现象（下半圆被拉长）
   - 倾斜角度越大，拉伸越严重

【旧代码问题分析】

从用户提供的 畸变矫正.py 分析，存在以下问题：

1. 几何校正中未正确考虑折射率与倾斜角的耦合效应
2. 对每个像素独立处理，导致血管结构不连续
3. 未考虑 OCT 的扇形扫描几何
4. 深度方向的压缩/拉伸因子计算不正确

【本模块的改进方案】

1. 基于物理模型的深度校正
2. 考虑表面倾斜的各向异性缩放
3. 引入局部倾角估计算法
4. 使用插值保持结构连续性
5. 支持 B-scan 和 3D 体数据的矫正

================================================================================
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline, interp2d
from scipy.signal import find_peaks
from typing import Tuple, Optional, Dict, Any
import cv2


class DistortionCorrector:
    """
    OCT 几何畸变矫正器
    
    解决问题：
    1. 折射率导致的深度缩放误差
    2. 表面倾斜导致的纵向拉伸
    3. 扇形扫描导致的非线性畸变
    
    使用方法：
    ```python
    corrector = DistortionCorrector(
        n_tissue=1.40,      # 组织折射率
        n_system=1.38,      # 系统假定折射率
        pixel_size_axial=8.73,    # 纵向像素尺寸 (μm)
        pixel_size_lateral=12.0   # 横向像素尺寸 (μm)
    )
    
    # 校正单张 B-scan
    corrected = corrector.correct_bscan(bscan, auto_estimate_tilt=True)
    
    # 校正整个 3D 体数据
    corrected_volume = corrector.correct_volume(volume)
    ```
    """
    
    def __init__(
        self,
        n_tissue: float = 1.40,
        n_system: float = 1.38,
        pixel_size_axial: float = 8.73,
        pixel_size_lateral: float = 12.0
    ):
        """
        初始化矫正器
        
        参数：
            n_tissue: 组织折射率（皮肤约 1.38-1.45）
            n_system: OCT 系统假定的折射率
            pixel_size_axial: 纵向（深度方向）像素尺寸，单位 μm
            pixel_size_lateral: 横向像素尺寸，单位 μm
        """
        # ============================================================
        # 折射率相关参数
        # ============================================================
        # 组织真实折射率
        self.n_tissue = n_tissue
        
        # 系统假定折射率（OCT 系统用于计算深度的折射率）
        self.n_system = n_system
        
        # 折射率补偿因子
        # 当 n_tissue > n_system 时，真实深度比显示深度更浅
        # 因此需要将图像纵向压缩
        self.refraction_scale = n_system / n_tissue
        
        # ============================================================
        # 像素尺寸（用于计算真实物理尺度）
        # ============================================================
        self.pixel_size_axial = pixel_size_axial
        self.pixel_size_lateral = pixel_size_lateral
        
        # 各向异性比（横向/纵向），用于判断是否需要各向异性缩放
        self.anisotropy_ratio = pixel_size_lateral / pixel_size_axial
        
    def correct_slice(
        self,
        slice_data: np.ndarray,
        axis: str = 'y'
    ) -> np.ndarray:
        """
        校正单个切片（简化版，用于实时预览）
        
        参数：
            slice_data: 2D 切片数据
            axis: 切片轴向 ('x', 'y', 'z')
        
        返回：
            corrected: 校正后的切片
        """
        # 对于 Y 轴切片（B-scan），应用完整校正
        if axis.lower() == 'y':
            return self.correct_bscan(slice_data, auto_estimate_tilt=True)
        else:
            # 其他轴向只应用折射率补偿
            return self._apply_refraction_correction(slice_data)
    
    def correct_bscan(
        self,
        bscan: np.ndarray,
        tilt_angle: Optional[float] = None,
        auto_estimate_tilt: bool = True
    ) -> np.ndarray:
        """
        校正单张 B-scan 图像
        
        B-scan 定义：
        - 沿 Y 轴（扫描方向）的横截面
        - 形状为 (X, Z)，其中 X 是横向，Z 是深度
        
        校正步骤：
        1. 估计表面倾斜角度（如果需要）
        2. 应用倾斜角度校正（纵向压缩）
        3. 应用折射率补偿
        4. 插值重采样保持平滑
        
        参数：
            bscan: 2D B-scan 图像，形状 (H, W) 或 (X, Z)
            tilt_angle: 倾斜角度（度），None 则自动估计
            auto_estimate_tilt: 是否自动估计倾斜角度
        
        返回：
            corrected: 校正后的 B-scan
        """
        bscan = bscan.astype(np.float32)
        h, w = bscan.shape
        
        # ============================================================
        # 步骤 1: 估计表面倾斜角度
        # ============================================================
        if tilt_angle is None and auto_estimate_tilt:
            tilt_angle = self._estimate_surface_tilt(bscan)
        elif tilt_angle is None:
            tilt_angle = 0.0
        
        # 转换为弧度
        theta = np.radians(tilt_angle)
        
        # ============================================================
        # 步骤 2: 计算校正参数
        # ============================================================
        
        # 倾斜校正因子
        # 当光束以角度 θ 穿过样品时，光程增加 1/cos(θ) 倍
        # 因此需要将纵向压缩 cos(θ) 倍
        tilt_correction = np.cos(theta) if abs(theta) > 0.01 else 1.0
        
        # 综合缩放因子
        # 同时考虑折射率补偿和倾斜校正
        total_scale = self.refraction_scale * tilt_correction
        
        # ============================================================
        # 步骤 3: 应用校正（使用仿射变换）
        # ============================================================
        
        # 计算新的图像尺寸
        new_h = int(h * total_scale)
        
        if new_h <= 0 or new_h > h * 2:
            # 防止异常的缩放
            new_h = h
        
        # 使用 OpenCV 进行高质量插值
        corrected = cv2.resize(
            bscan, 
            (w, new_h), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # ============================================================
        # 步骤 4: 可选的进一步处理
        # ============================================================
        
        # 对于倾斜的表面，还需要考虑横向的剪切变换
        if abs(tilt_angle) > 2.0:  # 倾斜角度大于 2 度时
            corrected = self._apply_shear_correction(
                corrected, 
                tilt_angle
            )
        
        return corrected
    
    def correct_volume(
        self,
        volume: np.ndarray,
        slice_by_slice: bool = True,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """
        校正整个 3D 体数据
        
        参数：
            volume: 3D 体数据，形状 (X, Y, Z)
                   - X: 横向
                   - Y: B-scan 方向（扫描方向）
                   - Z: 深度方向
            slice_by_slice: 是否逐切片处理（推荐，节省内存）
            progress_callback: 进度回调函数 callback(current, total)
        
        返回：
            corrected_volume: 校正后的 3D 体数据
        """
        volume = volume.astype(np.float32)
        X, Y, Z = volume.shape
        
        if slice_by_slice:
            # 逐 B-scan 处理
            # 每个 B-scan 是 X-Z 平面的切片
            
            # 先估计所有 B-scan 的平均倾斜角
            sample_indices = np.linspace(0, Y - 1, min(10, Y)).astype(int)
            tilt_angles = []
            
            for y in sample_indices:
                bscan = volume[:, y, :]
                angle = self._estimate_surface_tilt(bscan)
                tilt_angles.append(angle)
            
            avg_tilt = np.median(tilt_angles)
            
            # 计算校正后的 Z 维度大小
            theta = np.radians(avg_tilt)
            total_scale = self.refraction_scale * np.cos(theta)
            new_Z = int(Z * total_scale)
            new_Z = max(new_Z, int(Z * 0.5))  # 至少保留一半
            new_Z = min(new_Z, Z * 2)  # 最多放大两倍
            
            # 创建输出体
            corrected_volume = np.zeros((X, Y, new_Z), dtype=np.float32)
            
            # 逐切片处理
            for y in range(Y):
                if progress_callback:
                    progress_callback(y, Y)
                
                bscan = volume[:, y, :]
                corrected_bscan = self.correct_bscan(
                    bscan, 
                    tilt_angle=avg_tilt,
                    auto_estimate_tilt=False
                )
                
                # 调整大小以匹配输出体
                if corrected_bscan.shape[1] != new_Z:
                    corrected_bscan = cv2.resize(
                        corrected_bscan,
                        (new_Z, X),
                        interpolation=cv2.INTER_CUBIC
                    )
                
                corrected_volume[:, y, :] = corrected_bscan
            
            return corrected_volume
        
        else:
            # 3D 整体处理（内存密集）
            # 使用 3D 仿射变换
            return self._apply_3d_correction(volume)
    
    def _estimate_surface_tilt(self, bscan: np.ndarray) -> float:
        """
        从 B-scan 图像中估计表面倾斜角度
        
        方法：
        1. 检测每列的第一个显著信号位置（表面）
        2. 对表面位置进行线性拟合
        3. 计算倾斜角度
        
        参数：
            bscan: 2D B-scan 图像
        
        返回：
            tilt_angle: 表面倾斜角度（度）
        """
        h, w = bscan.shape
        
        # 阈值化找到信号区域
        threshold = np.percentile(bscan, 90)  # 使用 90 百分位作为阈值
        
        surface_positions = []
        valid_columns = []
        
        for x in range(w):
            column = bscan[:, x]
            
            # 找到第一个超过阈值的位置（从顶部开始）
            above_threshold = np.where(column > threshold)[0]
            
            if len(above_threshold) > 0:
                surface_pos = above_threshold[0]
                surface_positions.append(surface_pos)
                valid_columns.append(x)
        
        if len(valid_columns) < 2:
            return 0.0
        
        # 线性拟合
        valid_columns = np.array(valid_columns)
        surface_positions = np.array(surface_positions)
        
        # 使用鲁棒的中位数滤波去除异常点
        from scipy.ndimage import median_filter
        surface_smooth = median_filter(surface_positions, size=5)
        
        # 线性拟合：z = a * x + b
        try:
            coeffs = np.polyfit(valid_columns, surface_smooth, 1)
            slope = coeffs[0]  # dz/dx
            
            # 计算角度
            # tan(θ) = (dz * pixel_z) / (dx * pixel_x)
            # 这里简化为 tan(θ) = slope * (pixel_z / pixel_x)
            tan_theta = slope * (self.pixel_size_axial / self.pixel_size_lateral)
            tilt_angle = np.degrees(np.arctan(tan_theta))
            
            # 限制在合理范围内
            tilt_angle = np.clip(tilt_angle, -30, 30)
            
            return float(tilt_angle)
            
        except Exception:
            return 0.0
    
    def _apply_refraction_correction(self, image: np.ndarray) -> np.ndarray:
        """
        应用折射率补偿（仅深度方向缩放）
        
        物理原理：
        - OCT 显示深度 = OPL / n_system
        - 真实物理深度 = OPL / n_tissue
        - 因此：真实深度 = 显示深度 × (n_system / n_tissue)
        
        当 n_tissue > n_system 时，真实深度更浅，需要压缩图像
        """
        h, w = image.shape
        
        # 计算新高度
        new_h = int(h * self.refraction_scale)
        
        if new_h == h:
            return image.copy()
        
        # 使用高质量插值
        corrected = cv2.resize(
            image,
            (w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        return corrected
    
    def _apply_shear_correction(
        self, 
        image: np.ndarray, 
        tilt_angle: float
    ) -> np.ndarray:
        """
        应用剪切变换校正倾斜表面
        
        当表面倾斜时，深部结构相对于浅层结构存在横向偏移
        这个偏移量与深度和倾斜角成正比
        
        参数：
            image: 输入图像
            tilt_angle: 倾斜角度（度）
        
        返回：
            校正后的图像
        """
        h, w = image.shape
        
        # 计算剪切量
        # 在深度 z 处的横向偏移 = z × tan(θ)
        theta = np.radians(tilt_angle)
        shear_factor = np.tan(theta)
        
        # 构建仿射变换矩阵
        # [1, shear, 0]
        # [0, 1,     0]
        M = np.float32([
            [1, shear_factor, -shear_factor * h / 2],
            [0, 1, 0]
        ])
        
        # 计算输出图像大小
        new_w = int(w + abs(shear_factor * h))
        
        # 应用仿射变换
        corrected = cv2.warpAffine(
            image,
            M,
            (new_w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # 裁剪到原始宽度
        start_x = (new_w - w) // 2
        corrected = corrected[:, start_x:start_x + w]
        
        return corrected
    
    def _apply_3d_correction(self, volume: np.ndarray) -> np.ndarray:
        """
        应用 3D 整体校正
        
        使用 3D 仿射变换一次性处理整个体数据
        适用于内存充足的情况
        """
        from scipy.ndimage import affine_transform
        
        X, Y, Z = volume.shape
        
        # 估计平均倾斜角
        mid_y = Y // 2
        mid_bscan = volume[:, mid_y, :]
        tilt_angle = self._estimate_surface_tilt(mid_bscan)
        theta = np.radians(tilt_angle)
        
        # 综合缩放因子
        total_scale_z = self.refraction_scale * np.cos(theta)
        
        # 构建 3D 变换矩阵
        # 只对 Z 轴进行缩放
        transform_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1 / total_scale_z]  # Z 轴缩放
        ])
        
        # 计算输出大小
        new_Z = int(Z * total_scale_z)
        new_Z = max(new_Z, int(Z * 0.5))
        
        # 应用变换
        corrected = affine_transform(
            volume,
            transform_matrix,
            output_shape=(X, Y, new_Z),
            order=3,  # 三次样条插值
            mode='constant',
            cval=0.0
        )
        
        return corrected
    
    def get_correction_info(self, volume_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        获取校正信息（用于 UI 显示）
        
        返回当前校正参数和预期的输出尺寸
        """
        X, Y, Z = volume_shape
        
        # 预估校正后的尺寸
        estimated_scale = self.refraction_scale * 0.95  # 假设约 5 度倾斜
        new_Z = int(Z * estimated_scale)
        
        return {
            'tissue_refractive_index': self.n_tissue,
            'system_refractive_index': self.n_system,
            'refraction_scale_factor': self.refraction_scale,
            'pixel_size_axial_um': self.pixel_size_axial,
            'pixel_size_lateral_um': self.pixel_size_lateral,
            'anisotropy_ratio': self.anisotropy_ratio,
            'input_shape': list(volume_shape),
            'estimated_output_shape': [X, Y, new_Z],
            'depth_compression_ratio': estimated_scale
        }


class TailArtifactRemover:
    """
    拖尾伪影去除器
    
    专门处理 OCTA 中血管的拖尾伪影问题
    
    拖尾原因：
    1. 红细胞散射导致信号在深度方向延展
    2. 多重散射增加了光程
    3. 系统点扩散函数（PSF）的轴向拖尾
    
    处理方法：
    1. 形态学处理（开运算、闭运算）
    2. 基于梯度的边缘检测
    3. 统计学方法（基于血管横截面形状）
    """
    
    def __init__(self, max_tail_length: int = 20):
        """
        参数：
            max_tail_length: 最大拖尾长度（像素）
        """
        self.max_tail_length = max_tail_length
    
    def remove_tail_simple(
        self, 
        bscan: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        简单的拖尾去除（形态学方法）
        
        使用开运算去除细小的拖尾结构
        """
        # 二值化
        binary = (bscan > threshold).astype(np.uint8)
        
        # 设计结构元素：横向椭圆形
        # 这样可以保留横向血管结构，去除纵向拖尾
        kernel_h = 3
        kernel_w = 7
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_w, kernel_h)
        )
        
        # 开运算
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 闭运算填充小孔
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # 应用掩模
        result = bscan.copy()
        result[closed == 0] = 0
        
        return result
    
    def detect_vessel_centers(
        self, 
        bscan: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        检测血管中心位置
        
        返回每列的血管中心深度位置
        """
        binary = (bscan > threshold)
        h, w = bscan.shape
        
        centers = np.full(w, -1, dtype=np.float32)
        
        for x in range(w):
            column = binary[:, x]
            vessel_regions = np.where(column)[0]
            
            if len(vessel_regions) > 0:
                # 找到连续区域的中心
                # 假设血管是顶部开始的区域
                if len(vessel_regions) >= 2:
                    # 找第一个连续区域
                    diffs = np.diff(vessel_regions)
                    breaks = np.where(diffs > 1)[0]
                    
                    if len(breaks) > 0:
                        first_region_end = breaks[0]
                        first_region = vessel_regions[:first_region_end + 1]
                    else:
                        first_region = vessel_regions
                    
                    # 取中心
                    center = (first_region[0] + first_region[-1]) / 2
                    centers[x] = center
                else:
                    centers[x] = vessel_regions[0]
        
        return centers
