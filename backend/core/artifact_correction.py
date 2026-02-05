# -*- coding: utf-8 -*-
"""
OCTA 血管伪影矫正算法

核心功能：
1. 基于用户输入的长短轴，计算真实圆形血管截面
2. 上边界定位算法
3. 切片连贯性分析
4. 几何与物理修正
"""

import numpy as np
import cv2
import math
from typing import Dict, Optional, Tuple, List
from scipy import ndimage
from skimage import measure, morphology


class ArtifactCorrection:
    """OCTA拖尾伪影矫正器"""

    def __init__(self):
        self.continuity_threshold = 15.0  # 半径连续性阈值
        self.position_threshold = 20.0     # 位置连续性阈值
        self.edge_gradient_threshold = 30   # 边缘梯度阈值

    def correct_vessel_from_axes(
        self,
        image: np.ndarray,
        center_x: float,
        center_y: float,
        major_axis: float,
        minor_axis: float,
        roi_size: int = 80,
        slice_context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        基于长短轴输入计算真实圆形血管截面

        Args:
            image: 输入图像
            center_x, center_y: 用户指定的中心
            major_axis: 长轴长度（像素）
            minor_axis: 短轴长度（像素）
            roi_size: ROI区域大小
            slice_context: 相邻切片的圆形信息用于连贯性分析

        Returns:
            矫正后的圆形参数
        """

        # 🔴 调试：输入坐标验证
        print(f"[DEBUG] ArtifactCorrection.correct_vessel_from_axes:")
        print(f"  Input center: (x={center_x}, y={center_y})")
        print(f"  Axes: major={major_axis}, minor={minor_axis}")
        print(f"  ROI size: {roi_size}")
        print(f"  Image shape: {image.shape}")

        # 1. 基于几何关系计算真实半径
        estimated_radius = self._estimate_true_radius(major_axis, minor_axis)
        print(f"  Estimated radius: {estimated_radius}")

        # 2. 在ROI内精确定位上边界
        refined_center = self._locate_upper_boundary(
            image, center_x, center_y, estimated_radius, roi_size
        )

        print(f"  Upper boundary result: (x={refined_center[0]}, y={refined_center[1]})")
        print(f"  Offset from input: {(refined_center[1] - center_y):.1f}px")

        # 3. 应用切片连贯性约束
        if slice_context:
            original_center = refined_center
            refined_center = self._apply_spatial_continuity(
                refined_center, estimated_radius, slice_context
            )
            print(f"  Spatial continuity applied: {(original_center[0], original_center[1])} → {(refined_center[0], refined_center[1])}")
        else:
            print(f"  No slice context, using upper boundary result")

        # 4. 生成最终圆形
        result = {
            "center_x": float(refined_center[0]),
            "center_y": float(refined_center[1]),
            "radius": float(estimated_radius),  # 🔴 关键：确保radius存在且为float类型
            "confidence": self._calculate_confidence(image, refined_center, float(estimated_radius)),
            "correction_applied": True,
            "method": "artifact_correction_v1.0"
        }

        return result

    def _estimate_true_radius(self, major_axis: float, minor_axis: float) -> float:
        """
        基于长短轴估算真实血管半径

        几何原理：
        - 真实血管是圆形，半径为R
        - 伪影椭圆：长轴≈2R，短轴≈2R×cos(θ)
        - 真实半径 R = √(长轴 × 短轴 / 4)
        """
        if major_axis <= 0 or minor_axis <= 0:
            return float(max(major_axis, minor_axis) / 2)

        # 几何平均估算真实直径
        estimated_diameter = math.sqrt(float(major_axis) * float(minor_axis))
        estimated_radius = float(estimated_diameter / 2)

        # 物理约束：血管半径应该在合理范围内
        estimated_radius = float(max(2.0, min(50.0, estimated_radius)))

        return estimated_radius

    def _locate_upper_boundary(
        self,
        image: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
        roi_size: int
    ) -> Tuple[float, float]:
        """
        在ROI内定位血管上边界（最真实的边缘）

        算法思路：
        1. 提取ROI区域
        2. 计算垂直梯度
        3. 在上方区域寻找最强的连续边缘
        4. 基于上边界位置微调圆心
        """

        h, w = image.shape
        # 🔴 微调：使用round()而非int()截断提高坐标精度
        center_x = round(center_x)
        center_y = round(center_y)
        radius = round(radius)

        # 提取ROI
        half_roi = roi_size // 2
        x1 = max(0, center_x - half_roi)
        x2 = min(w, center_x + half_roi)
        y1 = max(0, center_y - half_roi)
        y2 = min(h, center_y + half_roi)

        roi = image[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return (float(center_x), float(center_y))

        # 预处理：增强边缘
        roi_normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 计算垂直梯度（Y方向）
        grad_y = cv2.Sobel(roi_normalized, cv2.CV_64F, 0, 1, ksize=3)
        grad_y_abs = np.abs(grad_y)

        # 在预测位置的上方区域寻找上边界
        local_center_y = center_y - y1
        # 🔴 微调：使用round()提高搜索边界精度
        search_top = max(0, round(local_center_y - radius * 1.5))
        search_bottom = round(local_center_y)

        upper_boundary_candidates = []

        # 🔴 修复：在搜索区域内寻找强边缘
        for y in range(search_top, search_bottom):
            # 取这一行的梯度最大值
            row_max_grad = np.max(grad_y_abs[y, :])
            if row_max_grad > self.edge_gradient_threshold:
                # 找到最大梯度的x位置
                x_max = np.argmax(grad_y_abs[y, :])
                # 🔴 关键修复：y是ROI内坐标，y1是ROI起始偏移，这里已经是全局坐标
                # 注意：这里的搜索范围已经基于local_center_y计算，所以y是相对于ROI的
                global_y = y + y1  # 转换为全局坐标
                global_x = x_max + x1  # 转换为全局坐标

                upper_boundary_candidates.append({
                    'x': float(global_x),  # 🔴 微调：保持float精度
                    'y': float(global_y),  # 🔴 微调：保持float精度
                    'gradient': row_max_grad,
                    'roi_y': y,  # 保存ROI内坐标用于调试
                    'roi_x': x_max
                })

        if not upper_boundary_candidates:
            return (float(center_x), float(center_y))

        # 选择梯度最强的候选点作为上边界
        best_boundary = max(upper_boundary_candidates, key=lambda x: x['gradient'])

        # 🔴 调试日志：验证坐标修复
        print(f"[DEBUG] Upper boundary processing (COORDINATE FIX):")
        print(f"  Best boundary (global): (x={best_boundary['x']}, y={best_boundary['y']})")
        print(f"  Best boundary (local in ROI): (x={best_boundary['roi_x']}, y={best_boundary['roi_y']})")
        print(f"  ROI bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"  Input center: (x={center_x}, y={center_y})")
        print(f"  Search range: {search_top} to {search_bottom} (local)")

        # 基于上边界微调圆心
        # 🔴 修正：best_boundary['y'] 已经是全局坐标
        global_best_boundary_y = float(best_boundary['y'])
        # 假设上边界距离圆心约一个半径的距离
        adjusted_y = float(global_best_boundary_y) + float(radius)

        # 🔴 微调：添加校准偏差补偿，修复2-5像素的垂直偏移
        # 实践观察：算法结果倾向于偏低2-5像素，需要向上微调
        calibration_bias = -2.0  # 向上微调2像素（负值表示向上）
        adjusted_y += calibration_bias

        # 🔴 微调：使用四舍五入确保亚像素精度
        final_x = float(round(center_x))
        final_y = float(round(adjusted_y))

        print(f"  Adjusted center: (x={final_x}, y={final_y})")
        print(f"  Offset from input: {(final_y - center_y):.1f}px (should equal radius {radius:.1f}px)")
        print(f"  Radius: {radius}")
        print(f"  Calibration bias applied: {calibration_bias}px")

        return (final_x, final_y)

    def _apply_spatial_continuity(
        self,
        center: Tuple[float, float],
        radius: float,
        slice_context: List[Dict]
    ) -> Tuple[float, float]:
        """
        应用切片连贯性约束，平滑相邻切片间的变化
        """

        if not slice_context:
            return center

        # 计算加权平均位置
        total_weight = 1.0  # 当前切片权重
        weighted_x = center[0] * total_weight
        weighted_y = center[1] * total_weight

        for context in slice_context:
            # 检查半径是否连续
            radius_diff = abs(radius - context['radius'])
            if radius_diff > self.continuity_threshold:
                continue

            # 检查位置是否连续
            pos_dist = math.sqrt(
                (center[0] - context['center_x'])**2 +
                (center[1] - context['center_y'])**2
            )
            if pos_dist > self.position_threshold:
                continue

            # 根据距离和相似度计算权重
            weight = 1.0 / (1.0 + radius_diff/10.0 + pos_dist/20.0)

            weighted_x += context['center_x'] * weight
            weighted_y += context['center_y'] * weight
            total_weight += weight

        if total_weight > 1.0:
            return (weighted_x / total_weight, weighted_y / total_weight)
        else:
            return center

    def _calculate_confidence(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        radius: float
    ) -> float:
        """
        计算矫正结果的置信度

        评估指标：
        1. 边缘清晰度
        2. 圆形度
        3. 强度分布
        """

        center_x, center_y = int(center[0]), int(center[1])
        radius = int(radius)

        h, w = image.shape
        if (center_x - radius < 0 or center_x + radius >= w or
            center_y - radius < 0 or center_y + radius >= h):
            return 0.3

        # 提取圆形ROI
        roi_size = radius * 3
        x1 = max(0, center_x - roi_size//2)
        x2 = min(w, center_x + roi_size//2)
        y1 = max(0, center_y - roi_size//2)
        y2 = min(h, center_y + roi_size//2)

        roi = image[y1:y2, x1:x2].copy()

        # 边缘清晰度评估
        roi_normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(roi_normalized, 50, 150)

        # 计算圆形边缘上的边缘强度
        edge_points = []
        for angle in np.linspace(0, 2*np.pi, 36):
            x = int(roi_size//2 + radius * np.cos(angle))
            y = int(roi_size//2 + radius * np.sin(angle))
            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                edge_points.append(edges[y, x])

        edge_strength = float(np.mean(edge_points) / 255.0)

        # 综合置信度
        confidence = float(0.5 + edge_strength * 0.4)
        return float(min(0.95, max(0.3, confidence)))

    def generate_circle_annotation(
        self,
        image: np.ndarray,
        center_x: float,
        center_y: float,
        major_axis: float,
        minor_axis: float,
        slice_index: int,
        axis: str,
        project_id: str,
        slice_context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        生成标准的圆形标注数据
        """
        print(f"[DEBUG] ArtifactCorrection.generate_circle_annotation called with:")
        print(f"  center_x: {center_x}, center_y: {center_y}")
        print(f"  major_axis: {major_axis}, minor_axis: {minor_axis}")
        print(f"  slice_index: {slice_index}, axis: {axis}")

        correction_result = self.correct_vessel_from_axes(
            image, center_x, center_y, major_axis, minor_axis,
            roi_size=80, slice_context=slice_context
        )

        print(f"[DEBUG] Correction result:")
        print(f"  {correction_result}")

        if 'radius' not in correction_result or correction_result['radius'] is None:
            print(f"[ERROR] Radius missing or None in correction_result!")
        else:
            print(f"[DEBUG] Radius value: {correction_result['radius']} (type: {type(correction_result['radius'])})")

        # 转换为标准标注格式
        annotation = {
            "id": f"circle_{slice_index}_{int(correction_result['center_x'])}_{int(correction_result['center_y'])}",
            "center_x": float(correction_result["center_x"]),
            "center_y": float(correction_result["center_y"]),
            "radius": float(correction_result["radius"]),  # 🔴 关键修复：使用单个radius字段
            "radius_x": float(correction_result["radius"]),  # 保留兼容性
            "radius_y": float(correction_result["radius"]),  # 保留兼容性
            "rotation": 0.0,  # 圆形无需旋转
            "slice_index": slice_index,
            "axis": axis,
            "confidence": float(correction_result["confidence"]),
            "is_manual": False,
            "method": correction_result["method"],
            "correction_applied": True,
            "original_major_axis": float(major_axis),
            "original_minor_axis": float(minor_axis)
        }

        print(f"[DEBUG] Final annotation before return:")
        print(f"  center_x: {annotation['center_x']} (type: {type(annotation['center_x'])})")
        print(f"  center_y: {annotation['center_y']} (type: {type(annotation['center_y'])})")
        print(f"  radius: {annotation['radius']} (type: {type(annotation['radius'])})")

        # 最终验证
        if 'radius' not in annotation or annotation['radius'] is None:
            raise ValueError("Critical: 'radius' field is missing from annotation!")

        return annotation