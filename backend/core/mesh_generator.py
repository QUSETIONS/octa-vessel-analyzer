# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 网格生成模块

功能：
1. 使用 Marching Cubes 从体数据生成表面网格
2. 网格平滑和简化
3. 导出为多种格式（STL、PLY、OBJ）
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class Mesh:
    """
    网格数据结构
    """
    vertices: np.ndarray  # (N, 3) 顶点坐标
    faces: np.ndarray     # (M, 3) 面索引
    normals: Optional[np.ndarray] = None  # (N, 3) 顶点法向量
    colors: Optional[np.ndarray] = None   # (N, 3) 顶点颜色
    
    def export(self, filepath: str):
        """导出网格文件"""
        suffix = filepath.lower().split('.')[-1]
        
        if suffix == 'stl':
            self._export_stl(filepath)
        elif suffix == 'ply':
            self._export_ply(filepath)
        elif suffix == 'obj':
            self._export_obj(filepath)
        else:
            raise ValueError(f"不支持的格式: {suffix}")
    
    def _export_stl(self, filepath: str):
        """导出 STL 格式"""
        import struct
        
        with open(filepath, 'wb') as f:
            # 头部（80 字节）
            header = b'\x00' * 80
            f.write(header)
            
            # 三角形数量
            num_triangles = len(self.faces)
            f.write(struct.pack('<I', num_triangles))
            
            # 每个三角形
            for face in self.faces:
                v0, v1, v2 = self.vertices[face]
                
                # 计算法向量
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                
                # 写入法向量
                f.write(struct.pack('<3f', *normal))
                
                # 写入三个顶点
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                
                # 属性字节（2 字节，通常为 0）
                f.write(struct.pack('<H', 0))
    
    def _export_ply(self, filepath: str):
        """导出 PLY 格式"""
        with open(filepath, 'w') as f:
            # 头部
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(self.vertices)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            
            if self.normals is not None:
                f.write('property float nx\n')
                f.write('property float ny\n')
                f.write('property float nz\n')
            
            if self.colors is not None:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            
            f.write(f'element face {len(self.faces)}\n')
            f.write('property list uchar int vertex_indices\n')
            f.write('end_header\n')
            
            # 顶点
            for i, v in enumerate(self.vertices):
                line = f'{v[0]} {v[1]} {v[2]}'
                
                if self.normals is not None:
                    n = self.normals[i]
                    line += f' {n[0]} {n[1]} {n[2]}'
                
                if self.colors is not None:
                    c = self.colors[i]
                    line += f' {int(c[0])} {int(c[1])} {int(c[2])}'
                
                f.write(line + '\n')
            
            # 面
            for face in self.faces:
                f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
    
    def _export_obj(self, filepath: str):
        """导出 OBJ 格式"""
        with open(filepath, 'w') as f:
            # 顶点
            for v in self.vertices:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            
            # 法向量
            if self.normals is not None:
                for n in self.normals:
                    f.write(f'vn {n[0]} {n[1]} {n[2]}\n')
            
            # 面（OBJ 索引从 1 开始）
            for face in self.faces:
                if self.normals is not None:
                    f.write(f'f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n')
                else:
                    f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


class MeshGenerator:
    """
    网格生成器
    
    从体数据生成表面网格
    """
    
    def __init__(self):
        pass
    
    def generate(
        self,
        volume: np.ndarray,
        threshold: float = 0.5,
        smooth: bool = True,
        smooth_iterations: int = 3,
        decimate_ratio: float = 1.0,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Mesh:
        """
        从体数据生成网格
        
        使用 Marching Cubes 算法提取等值面
        
        参数：
            volume: 3D 体数据
            threshold: 等值面阈值
            smooth: 是否平滑网格
            smooth_iterations: 平滑迭代次数
            decimate_ratio: 网格简化比例（0-1，1 表示不简化）
            spacing: 体素间距 (dx, dy, dz)
        
        返回：
            Mesh 对象
        """
        from skimage import measure
        
        # 预处理：高斯平滑
        if smooth:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume.astype(float), sigma=1.0)
        
        # Marching Cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume,
                level=threshold,
                spacing=spacing
            )
        except Exception as e:
            # 如果失败，尝试使用较低的阈值
            print(f"Marching Cubes 失败 (阈值={threshold}): {e}")
            print("尝试降低阈值...")
            
            verts, faces, normals, values = measure.marching_cubes(
                volume,
                level=threshold * 0.5,
                spacing=spacing
            )
        
        # 网格简化
        if decimate_ratio < 1.0 and len(faces) > 1000:
            verts, faces = self._decimate_mesh(verts, faces, decimate_ratio)
            # 重新计算法向量
            normals = self._compute_normals(verts, faces)
        
        # 网格平滑
        if smooth and smooth_iterations > 0:
            verts = self._smooth_mesh(verts, faces, smooth_iterations)
            normals = self._compute_normals(verts, faces)
        
        # 基于深度的颜色
        colors = self._generate_depth_colors(verts)
        
        return Mesh(
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32) if normals is not None else None,
            colors=colors.astype(np.uint8)
        )
    
    def _decimate_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        网格简化（使用简单的顶点聚类方法）
        
        注意：这是一个简化的实现，生产环境应使用更高级的算法
        """
        # 计算目标面数
        target_faces = int(len(faces) * ratio)
        
        if target_faces >= len(faces):
            return vertices, faces
        
        # 简单的均匀采样
        step = max(1, len(faces) // target_faces)
        selected_face_indices = np.arange(0, len(faces), step)[:target_faces]
        
        # 提取选中的面和相关顶点
        selected_faces = faces[selected_face_indices]
        
        # 获取唯一顶点
        unique_vertices = np.unique(selected_faces.flatten())
        
        # 建立顶点映射
        vertex_map = {old: new for new, old in enumerate(unique_vertices)}
        
        # 更新顶点和面
        new_vertices = vertices[unique_vertices]
        new_faces = np.array([
            [vertex_map[v] for v in face]
            for face in selected_faces
        ])
        
        return new_vertices, new_faces
    
    def _smooth_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int
    ) -> np.ndarray:
        """
        拉普拉斯网格平滑
        """
        vertices = vertices.copy()
        
        # 构建邻接关系
        num_vertices = len(vertices)
        adjacency = [set() for _ in range(num_vertices)]
        
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # 迭代平滑
        lambda_factor = 0.5
        
        for _ in range(iterations):
            new_vertices = vertices.copy()
            
            for i in range(num_vertices):
                if len(adjacency[i]) > 0:
                    neighbor_verts = vertices[list(adjacency[i])]
                    centroid = neighbor_verts.mean(axis=0)
                    new_vertices[i] = vertices[i] + lambda_factor * (centroid - vertices[i])
            
            vertices = new_vertices
        
        return vertices
    
    def _compute_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        计算顶点法向量
        """
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # 累加到顶点
            for i in face:
                normals[i] += face_normal
        
        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms
        
        return normals
    
    def _generate_depth_colors(self, vertices: np.ndarray) -> np.ndarray:
        """
        基于深度生成颜色
        """
        # 使用 Z 坐标作为深度
        z = vertices[:, 2]
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min
        
        if z_range < 1e-8:
            z_norm = np.zeros_like(z)
        else:
            z_norm = (z - z_min) / z_range
        
        # 使用红-黄-绿色谱
        colors = np.zeros((len(vertices), 3), dtype=np.uint8)
        
        # 浅层：绿色
        # 中层：黄色
        # 深层：红色
        colors[:, 0] = (z_norm * 255).astype(np.uint8)  # R: 深度越大越红
        colors[:, 1] = ((1 - z_norm) * 255).astype(np.uint8)  # G: 深度越浅越绿
        colors[:, 2] = (np.abs(z_norm - 0.5) * 100).astype(np.uint8)  # B: 中间略蓝
        
        return colors
    
    def generate_from_binary(
        self,
        binary_volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Mesh:
        """
        从二值体数据生成网格
        """
        # 对二值数据使用 0.5 作为阈值
        return self.generate(
            binary_volume.astype(float),
            threshold=0.5,
            smooth=True,
            spacing=spacing
        )
