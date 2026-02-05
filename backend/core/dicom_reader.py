# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - DICOM/OCTA 数据读取模块

功能：
1. 支持多种医学图像格式（DICOM、NIfTI、MATLAB、NumPy）
2. 解析并返回3D体数据及元信息
3. 处理DICOM序列（多文件）

兼容原有 scripts.zip 中的数据格式
"""

import os
import zipfile
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

class DicomReader:
    """
    医学图像数据读取器
    
    支持的格式：
    - DICOM (.dcm, .dicom) - 单文件或序列
    - NIfTI (.nii, .nii.gz)
    - MATLAB (.mat) - 兼容原有项目格式
    - NumPy (.npy)
    - ZIP 压缩包 - 包含多个 DICOM 切片
    """
    
    def __init__(self):
        # 默认的轴向标签
        self.default_axis_labels = ['X', 'Y', 'Z']
        
        # 默认的体素间距（微米）
        self.default_spacing = [12.0, 12.0, 8.73]  # 根据原项目参数
        
    def read_volume(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取体数据文件
        
        参数：
            file_path: 文件路径
            
        返回：
            volume: 3D numpy 数组，形状为 (X, Y, Z)
            metadata: 元信息字典
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        # 处理 .gz 后缀
        if suffix == '.gz':
            # 可能是 .nii.gz
            stem = file_path.stem
            if stem.endswith('.nii'):
                suffix = '.nii.gz'
        
        # 根据文件类型选择读取方法
        if suffix in ['.dcm', '.dicom']:
            return self._read_dicom(file_path)
        elif suffix in ['.nii', '.nii.gz']:
            return self._read_nifti(file_path)
        elif suffix == '.mat':
            return self._read_matlab(file_path)
        elif suffix == '.npy':
            return self._read_numpy(file_path)
        elif suffix == '.zip':
            return self._read_dicom_zip(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def _read_dicom(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取单个 DICOM 文件
        
        注意：单个 DICOM 文件通常是 2D 图像，
        如果需要 3D 体数据，应该使用 DICOM 序列或 ZIP 包
        """
        import pydicom
        
        ds = pydicom.dcmread(str(file_path))
        
        # 获取像素数据
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # 如果是 2D，扩展为 3D
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[:, :, np.newaxis]
        
        # 归一化到 [0, 1]
        pixel_array = self._normalize(pixel_array)
        
        # 提取元信息
        metadata = self._extract_dicom_metadata(ds)
        
        return pixel_array, metadata
    
    def _read_dicom_series(self, directory: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取 DICOM 序列（目录中的多个 DICOM 文件）
        """
        import pydicom
        from pydicom.errors import InvalidDicomError
        
        # 收集所有 DICOM 文件
        dicom_files = []
        for f in directory.iterdir():
            if f.is_file():
                try:
                    ds = pydicom.dcmread(str(f))
                    dicom_files.append((f, ds))
                except InvalidDicomError:
                    continue
        
        if not dicom_files:
            raise ValueError("目录中没有有效的 DICOM 文件")
        
        # 按实例编号或位置排序
        def sort_key(item):
            ds = item[1]
            if hasattr(ds, 'InstanceNumber'):
                return int(ds.InstanceNumber)
            elif hasattr(ds, 'SliceLocation'):
                return float(ds.SliceLocation)
            else:
                return 0
        
        dicom_files.sort(key=sort_key)
        
        # 堆叠切片
        slices = []
        for _, ds in dicom_files:
            slice_data = ds.pixel_array.astype(np.float32)
            slices.append(slice_data)
        
        volume = np.stack(slices, axis=-1)  # 形状: (H, W, D)
        volume = self._normalize(volume)
        
        # 提取元信息
        metadata = self._extract_dicom_metadata(dicom_files[0][1])
        metadata['num_slices'] = len(dicom_files)
        
        return volume, metadata
    
    def _read_dicom_zip(self, zip_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取包含 DICOM 序列的 ZIP 文件
        """
        # 解压到临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(str(zip_path), 'r') as zf:
                zf.extractall(temp_dir)
            
            # 读取 DICOM 序列
            return self._read_dicom_series(Path(temp_dir))
    
    def _read_nifti(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取 NIfTI 格式文件
        """
        import nibabel as nib
        
        img = nib.load(str(file_path))
        volume = img.get_fdata().astype(np.float32)
        volume = self._normalize(volume)
        
        # 提取元信息
        header = img.header
        affine = img.affine
        
        metadata = {
            'spacing': list(header.get_zooms()[:3]),
            'origin': list(affine[:3, 3]),
            'axis_labels': self.default_axis_labels,
            'format': 'NIfTI'
        }
        
        return volume, metadata
    
    def _read_matlab(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取 MATLAB .mat 格式文件
        
        兼容原项目中的数据格式：
        - 使用 h5py 读取 MATLAB v7.3 格式
        - 使用 scipy.io 读取旧版格式
        """
        import h5py
        from scipy import io as sio
        
        try:
            # 尝试使用 h5py 读取（MATLAB v7.3 格式）
            with h5py.File(str(file_path), 'r') as f:
                # 查找数据键（排除以 # 开头的元数据键）
                data_keys = [k for k in f.keys() if not k.startswith('#')]
                
                if not data_keys:
                    raise ValueError("MAT 文件中没有找到有效数据")
                
                # 使用第一个数据键
                data_key = data_keys[0]
                print(f"从 MAT 文件读取数据键: '{data_key}'")
                
                volume = np.array(f[data_key]).astype(np.float32)
                
                # MATLAB 存储格式可能需要转置
                # 根据原项目 inspect_mat.py 的分析，可能需要调整轴顺序
                if volume.ndim == 3:
                    # 默认转置为 (X, Y, Z)
                    volume = np.transpose(volume, (2, 1, 0))
                
                volume = self._normalize(volume)
                
                metadata = {
                    'spacing': self.default_spacing,
                    'origin': [0.0, 0.0, 0.0],
                    'axis_labels': self.default_axis_labels,
                    'format': 'MATLAB_v7.3',
                    'data_key': data_key
                }
                
                return volume, metadata
                
        except Exception as e:
            print(f"h5py 读取失败，尝试 scipy: {e}")
            
            # 尝试使用 scipy.io 读取旧版格式
            mat_data = sio.loadmat(str(file_path))
            
            # 查找数据变量（排除以 __ 开头的元数据）
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            
            if not data_keys:
                raise ValueError("MAT 文件中没有找到有效数据")
            
            data_key = data_keys[0]
            volume = mat_data[data_key].astype(np.float32)
            volume = self._normalize(volume)
            
            metadata = {
                'spacing': self.default_spacing,
                'origin': [0.0, 0.0, 0.0],
                'axis_labels': self.default_axis_labels,
                'format': 'MATLAB',
                'data_key': data_key
            }
            
            return volume, metadata
    
    def _read_numpy(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取 NumPy .npy 格式文件
        
        兼容原项目中的数据格式：
        - M_clean.npy: 血管掩模
        - I_clean.npy: 干净 OCTA
        - I_tailed.npy: 带拖尾的 OCTA
        """
        volume = np.load(str(file_path)).astype(np.float32)
        
        # 确保是 3D 数据
        if volume.ndim == 2:
            volume = volume[:, :, np.newaxis]
        elif volume.ndim != 3:
            raise ValueError(f"不支持的数组维度: {volume.ndim}")
        
        volume = self._normalize(volume)
        
        # 根据文件名推断数据类型
        filename = file_path.stem.lower()
        data_type = 'unknown'
        if 'clean' in filename and 'm_' in filename:
            data_type = 'vessel_mask'
        elif 'clean' in filename:
            data_type = 'clean_octa'
        elif 'tailed' in filename:
            data_type = 'tailed_octa'
        elif 'octa' in filename:
            data_type = 'octa'
        elif 'label' in filename:
            data_type = 'label'
        
        metadata = {
            'spacing': self.default_spacing,
            'origin': [0.0, 0.0, 0.0],
            'axis_labels': self.default_axis_labels,
            'format': 'NumPy',
            'data_type': data_type
        }
        
        return volume, metadata
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据到 [0, 1] 范围
        """
        data = data.astype(np.float32)
        
        # 处理极小值数据（如原项目中的微小数值范围）
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        
        if data_range < 1e-10:
            # 数据几乎是常数
            return np.zeros_like(data)
        elif data_range < 0.01:
            # 极小值范围，先缩放
            scale_factor = 1.0 / data_max if data_max > 0 else 1.0
            data = data * scale_factor
            data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        else:
            # 正常归一化
            data = (data - data_min) / data_range
        
        return data.astype(np.float32)
    
    def _extract_dicom_metadata(self, ds) -> Dict[str, Any]:
        """
        从 DICOM 数据集中提取元信息
        """
        metadata = {
            'axis_labels': self.default_axis_labels,
            'format': 'DICOM'
        }
        
        # 像素间距
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in ds.PixelSpacing]
        else:
            pixel_spacing = [self.default_spacing[0], self.default_spacing[1]]
        
        # 切片厚度
        if hasattr(ds, 'SliceThickness'):
            slice_thickness = float(ds.SliceThickness)
        else:
            slice_thickness = self.default_spacing[2]
        
        metadata['spacing'] = pixel_spacing + [slice_thickness]
        
        # 原点
        if hasattr(ds, 'ImagePositionPatient'):
            metadata['origin'] = [float(x) for x in ds.ImagePositionPatient]
        else:
            metadata['origin'] = [0.0, 0.0, 0.0]
        
        # 其他有用信息
        if hasattr(ds, 'PatientID'):
            metadata['patient_id'] = str(ds.PatientID)
        if hasattr(ds, 'StudyDescription'):
            metadata['study_description'] = str(ds.StudyDescription)
        if hasattr(ds, 'Modality'):
            metadata['modality'] = str(ds.Modality)
        
        return metadata

    def read_from_directory(self, directory: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        从目录读取数据（DICOM 序列或单个文件）
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise ValueError(f"目录不存在: {directory}")
        
        # 检查是否是 DICOM 序列目录
        dcm_files = list(directory.glob("*.dcm")) + list(directory.glob("*.dicom"))
        
        if dcm_files:
            return self._read_dicom_series(directory)
        
        # 查找其他支持的文件
        for suffix in ['.npy', '.mat', '.nii', '.nii.gz']:
            files = list(directory.glob(f"*{suffix}"))
            if files:
                return self.read_volume(str(files[0]))
        
        raise ValueError(f"目录中没有找到支持的数据文件: {directory}")
