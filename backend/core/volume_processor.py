# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 体数据处理模块

功能：
1. 体数据预处理（归一化、去噪等）
2. 最大强度投影（MIP）生成
3. 统计分析
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from typing import Tuple, Optional, Dict, Any


class VolumeProcessor:
    """
    体数据处理器
    
    提供各种体数据预处理和分析功能
    """
    
    def __init__(self):
        pass
    
    def preprocess(
        self,
        volume: np.ndarray,
        normalize: bool = True,
        denoise: bool = False,
        denoise_sigma: float = 1.0
    ) -> np.ndarray:
        """
        体数据预处理
        
        参数：
            volume: 输入体数据
            normalize: 是否归一化到 [0, 1]
            denoise: 是否去噪
            denoise_sigma: 高斯去噪的 sigma
        
        返回：
            处理后的体数据
        """
        volume = volume.astype(np.float32)
        
        # 去噪
        if denoise:
            volume = gaussian_filter(volume, sigma=denoise_sigma)
        
        # 归一化
        if normalize:
            volume = self._normalize(volume)
        
        return volume
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]"""
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        
        if data_range < 1e-10:
            return np.zeros_like(data)
        
        return (data - data_min) / data_range
    
    def compute_mip(
        self,
        volume: np.ndarray,
        axis: int = 2,
        mode: str = 'max'
    ) -> np.ndarray:
        """
        计算最大/最小/平均强度投影
        
        参数：
            volume: 3D 体数据
            axis: 投影轴 (0, 1, 2)
            mode: 投影模式 ('max', 'min', 'mean', 'sum')
        
        返回：
            2D 投影图像
        """
        if mode == 'max':
            return np.max(volume, axis=axis)
        elif mode == 'min':
            return np.min(volume, axis=axis)
        elif mode == 'mean':
            return np.mean(volume, axis=axis)
        elif mode == 'sum':
            return np.sum(volume, axis=axis)
        else:
            raise ValueError(f"未知的投影模式: {mode}")
    
    def compute_statistics(self, volume: np.ndarray) -> Dict[str, Any]:
        """
        计算体数据统计信息
        """
        return {
            'shape': list(volume.shape),
            'dtype': str(volume.dtype),
            'min': float(volume.min()),
            'max': float(volume.max()),
            'mean': float(volume.mean()),
            'std': float(volume.std()),
            'median': float(np.median(volume)),
            'non_zero_count': int(np.count_nonzero(volume)),
            'non_zero_fraction': float(np.count_nonzero(volume) / volume.size)
        }
    
    def extract_roi(
        self,
        volume: np.ndarray,
        center: Tuple[int, int, int],
        size: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        提取感兴趣区域（ROI）
        
        参数：
            volume: 输入体数据
            center: ROI 中心坐标 (x, y, z)
            size: ROI 尺寸 (sx, sy, sz)
        
        返回：
            ROI 体数据
        """
        cx, cy, cz = center
        sx, sy, sz = size
        
        # 计算边界
        x1 = max(0, cx - sx // 2)
        x2 = min(volume.shape[0], cx + sx // 2 + sx % 2)
        y1 = max(0, cy - sy // 2)
        y2 = min(volume.shape[1], cy + sy // 2 + sy % 2)
        z1 = max(0, cz - sz // 2)
        z2 = min(volume.shape[2], cz + sz // 2 + sz % 2)
        
        return volume[x1:x2, y1:y2, z1:z2].copy()
    
    def downsample(
        self,
        volume: np.ndarray,
        factor: int = 2
    ) -> np.ndarray:
        """
        下采样体数据
        
        用于大数据的预览和快速处理
        """
        from scipy.ndimage import zoom
        
        scale = 1.0 / factor
        return zoom(volume, (scale, scale, scale), order=1)
    
    def apply_threshold(
        self,
        volume: np.ndarray,
        threshold: float = 0.5,
        method: str = 'binary'
    ) -> np.ndarray:
        """
        阈值处理
        
        参数：
            volume: 输入体数据
            threshold: 阈值
            method: 方法 ('binary', 'otsu', 'adaptive')
        
        返回：
            二值化后的体数据
        """
        if method == 'binary':
            return (volume > threshold).astype(np.uint8)
        
        elif method == 'otsu':
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(volume)
            return (volume > thresh).astype(np.uint8)
        
        elif method == 'adaptive':
            # 简化的自适应阈值
            local_mean = ndimage.uniform_filter(volume, size=11)
            return (volume > local_mean * threshold).astype(np.uint8)
        
        else:
            raise ValueError(f"未知的阈值方法: {method}")
    
    def morphological_clean(
        self,
        binary_volume: np.ndarray,
        opening_size: int = 2,
        closing_size: int = 2
    ) -> np.ndarray:
        """
        形态学清理
        
        用于去除噪声和填充小孔
        """
        from scipy.ndimage import binary_opening, binary_closing
        
        # 创建结构元素
        struct_open = ndimage.generate_binary_structure(3, 1)
        struct_close = ndimage.generate_binary_structure(3, 1)
        
        # 开运算（去除小物体）
        if opening_size > 0:
            binary_volume = binary_opening(
                binary_volume, 
                structure=struct_open,
                iterations=opening_size
            )
        
        # 闭运算（填充小孔）
        if closing_size > 0:
            binary_volume = binary_closing(
                binary_volume,
                structure=struct_close,
                iterations=closing_size
            )
        
        return binary_volume.astype(np.uint8)
    
    def connected_components(
        self,
        binary_volume: np.ndarray,
        min_size: int = 100
    ) -> Tuple[np.ndarray, int]:
        """
        连通区域分析
        
        参数：
            binary_volume: 二值体数据
            min_size: 最小区域大小（体素数）
        
        返回：
            labeled: 标记的体数据
            num_components: 连通区域数量
        """
        from scipy.ndimage import label
        from scipy.ndimage import sum as ndi_sum
        
        # 标记连通区域
        labeled, num_features = label(binary_volume)
        
        # 过滤小区域
        if min_size > 0:
            sizes = ndi_sum(binary_volume, labeled, range(1, num_features + 1))
            mask = sizes >= min_size
            
            # 重新标记
            new_labeled = np.zeros_like(labeled)
            new_idx = 1
            for i, keep in enumerate(mask, start=1):
                if keep:
                    new_labeled[labeled == i] = new_idx
                    new_idx += 1
            
            labeled = new_labeled
            num_features = new_idx - 1
        
        return labeled, num_features
