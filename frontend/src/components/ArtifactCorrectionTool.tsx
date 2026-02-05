/**
 * OCTA 血管伪影矫正工具
 *
 * 功能：
 * 1. 用户输入长轴和短轴
 * 2. 自动计算真实圆形血管截面
 * 3. 提供人工二次修正
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Target, MousePointer, Circle, CheckCircle, RotateCw,
  Zap, Settings, RefreshCw, Save
} from 'lucide-react';
import { api } from '../utils/api';

interface ArtifactCorrectionProps {
  projectId: string;
  sliceIndex: number;
  axis: string;
  imageUrl: string;
  onCorrectionComplete: (annotation: any) => void;
  onCancel: () => void;
}

interface AxesInput {
  center_x: number;
  center_y: number;
  major_axis: number;
  minor_axis: number;
}

interface CorrectionResult {
  /** 圆形中心点X坐标（像素） */
  center_x: number;
  /** 圆形中心点Y坐标（像素） */
  center_y: number;
  /** 圆形半径（像素） */
  radius: number;
  /** 矫正置信度 (0-1) */
  confidence: number;
  /** 处理方法描述 */
  method: string;
  /** 是否已应用矫正 */
  correction_applied: boolean;
  /** 原始长轴长度（可选，用于调试） */
  original_major_axis?: number;
  /** 原始短轴长度（可选，用于调试） */
  original_minor_axis?: number;
}

const ArtifactCorrectionTool: React.FC<ArtifactCorrectionProps> = ({
  projectId,
  sliceIndex,
  axis,
  imageUrl,
  onCorrectionComplete,
  onCancel
}) => {
  // 工作流状态机
  const [workflowStep, setWorkflowStep] = useState<'initializing' | 'input' | 'processing' | 'review'>('initializing');

  // 用户输入状态
  const [userInput, setUserInput] = useState<AxesInput>({
    center_x: 0,
    center_y: 0,
    major_axis: 20,
    minor_axis: 10
  });

  // 是否已设置血管中心
  const [hasCenterSet, setHasCenterSet] = useState(false);

  // 后端矫正结果
  const [correctionResult, setCorrectionResult] = useState<CorrectionResult | null>(null);

  // 加载和错误状态
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Canvas相关
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [scale, setScale] = useState(1);

  // 拖拽交互状态
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [tempCirclePos, setTempCirclePos] = useState({ x: 0, y: 0 });

  
  // 实时预览计算（本地算法，不调用后端）
  const calculatePreviewCircle = useCallback((input: AxesInput) => {
    const { major_axis, minor_axis, center_x, center_y } = input;

    // 几何关系计算真实半径（简化版前端计算）
    const estimated_radius = Math.sqrt(major_axis * minor_axis) / 2;

    return {
      center_x,
      center_y,
      radius: Math.max(2, Math.min(50, estimated_radius)),
      isPreview: true
    };
  }, []);

  // 获取当前预览圆形
  const currentPreview = calculatePreviewCircle(userInput);

  
  
  // 图像加载逻辑
  const [sliceImageUrl, setSliceImageUrl] = useState<string>('');
  const loadedImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const loadImage = async () => {
      setImageLoading(true);
      setWorkflowStep('initializing');

      try {
        let imageUrlToUse = imageUrl;

        // 如果传入的不是blob URL，通过API获取
        if (!imageUrl.startsWith('blob:')) {
          const response = await api.get(`/data/${projectId}/slice/${axis}/${sliceIndex}`, {
            responseType: 'blob'
          });
          imageUrlToUse = URL.createObjectURL(response.data);
        }

        // 加载图像
        const img = new Image();
        img.crossOrigin = 'anonymous';

        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = imageUrlToUse;
        });

        loadedImageRef.current = img;
        setSliceImageUrl(imageUrlToUse);
        setImageLoaded(true);
        setWorkflowStep('input');

      } catch (error) {
        console.error('Failed to load image:', error);
        setError('图像加载失败，请重试');
        setWorkflowStep('input'); // 允许重试
      } finally {
        setImageLoading(false);
      }
    };

    loadImage();
  }, [projectId, sliceIndex, axis, imageUrl]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !loadedImageRef.current || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = loadedImageRef.current;

    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 计算缩放比例
    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;
    const finalScale = Math.min(scaleX, scaleY);
    setScale(finalScale);

    // 🔴 Canvas坐标系统调试
    console.log('🔴 CANVAS COORDINATE SYSTEM DEBUG:');
    console.log('  Canvas尺寸:', { width: canvas.width, height: canvas.height });
    console.log('  Image尺寸:', { width: img.width, height: img.height });
    console.log('  Natural尺寸:', { width: img.naturalWidth, height: img.naturalHeight });

    // 🔴 微调：使用像素完美对齐绘制图像
    // 确保像素对齐：使用round()避免亚像素偏移
    const scaledWidth = Math.round(img.width * finalScale);
    const scaledHeight = Math.round(img.height * finalScale);

    // 居中绘制，确保像素对齐
    const offsetX = Math.round((canvas.width - scaledWidth) / 2);
    const offsetY = Math.round((canvas.height - scaledHeight) / 2);

    console.log('  像素完美对齐参数:', {
      finalScale: finalScale.toFixed(3),
      scaledWidth,
      scaledHeight,
      offsetX,
      offsetY
    });

    // 清空画布并设置背景
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 像素完美绘制图像
    ctx.imageSmoothingEnabled = false; // 禁用平滑以确保像素精度
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

    // 存储偏移量供坐标转换使用
    canvas.imageOffsetX = offsetX;
    canvas.imageOffsetY = offsetY;

    // 绘制标注
    drawAnnotations(ctx, finalScale, offsetX, offsetY);
  };

  // 当图像加载完成或工作流状态变化时重绘
  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [imageLoaded, workflowStep, userInput, correctionResult]);

  const drawAnnotations = (ctx: CanvasRenderingContext2D, scale: number, offsetX: number, offsetY: number) => {
    if (workflowStep === 'input' && userInput.center_x && userInput.center_y) {
      // 绘制用户输入的椭圆（伪影状态）
      ctx.strokeStyle = 'rgba(255, 200, 0, 0.8)';
      ctx.lineWidth = 2 / scale;
      ctx.setLineDash([5 / scale, 5 / scale]);

      // 🔴 微调：应用图像偏移量，使用亚像素精度
      const centerX = Math.round(userInput.center_x * scale) + offsetX;
      const centerY = Math.round(userInput.center_y * scale) + offsetY;

      ctx.beginPath();
      ctx.ellipse(
        centerX,
        centerY,
        (userInput.major_axis / 2) * scale,
        (userInput.minor_axis / 2) * scale,
        0, 0, 2 * Math.PI
      );
      ctx.stroke();
      ctx.setLineDash([]);

      // 绘制中心点
      ctx.fillStyle = 'rgba(255, 200, 0, 1)';
      ctx.beginPath();
      ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI);
      ctx.fill();

      // 绘制实时预览圆形（半透明）
      if (currentPreview && currentPreview.radius > 0) {
        ctx.strokeStyle = 'rgba(0, 255, 100, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);

        // 🔴 微调：应用图像偏移量
        const previewCenterX = Math.round(currentPreview.center_x * scale) + offsetX;
        const previewCenterY = Math.round(currentPreview.center_y * scale) + offsetY;
        const previewRadius = Math.round(currentPreview.radius * scale);

        ctx.beginPath();
        ctx.arc(
          previewCenterX,
          previewCenterY,
          previewRadius,
          0, 2 * Math.PI
        );
        ctx.stroke();
        ctx.setLineDash([]);

        // 显示预览标签
        ctx.fillStyle = 'rgba(0, 255, 100, 0.8)';
        ctx.font = `${10}px Arial`;
        ctx.fillText(
          '预览',
          previewCenterX + previewRadius + 5,
          previewCenterY - previewRadius
        );
      }

    } else if (workflowStep === 'review' && correctionResult) {
      // 绘制矫正后的圆形 - 红色，更加醒目
      // 🔴 微调：应用图像偏移量和缩放
      const baseCenterX = tempCirclePos.x || correctionResult.center_x;
      const baseCenterY = tempCirclePos.y || correctionResult.center_y;
      const baseRadius = correctionResult.radius;

      const centerX = Math.round(baseCenterX * scale) + offsetX;
      const centerY = Math.round(baseCenterY * scale) + offsetY;
      const radius = Math.round(baseRadius * scale);

      // 外层红色边框（最醒目）
      ctx.strokeStyle = 'rgba(255, 0, 0, 1.0)';
      ctx.lineWidth = 4; // 🔴 微调：使用绝对像素值
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.stroke();

      // 内层填充（半透明红色）
      ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.fill();

      // 中心十字标记（方便定位）
      ctx.strokeStyle = 'rgba(255, 0, 0, 1.0)';
      ctx.lineWidth = 2; // 🔴 微调：使用绝对像素值

      // 水平线
      ctx.beginPath();
      ctx.moveTo(centerX - radius * 0.3, centerY);
      ctx.lineTo(centerX + radius * 0.3, centerY);
      ctx.stroke();

      // 垂直线
      ctx.beginPath();
      ctx.moveTo(centerX, centerY - radius * 0.3);
      ctx.lineTo(centerX, centerY + radius * 0.3);
      ctx.stroke();

      // 🔴 微调：详细的坐标调试日志
      console.log('🔴 DRAWING RED CIRCLE (PIXEL-PERPRECISE FIX):');
      console.log('  Image offset:', { offsetX, offsetY });
      console.log('  Scale factor:', scale.toFixed(3));
      console.log('  Original coordinates (from API):', {
        x: correctionResult.center_x,
        y: correctionResult.center_y,
        radius: correctionResult.radius
      });
      console.log('  Scaled + offset coordinates:', {
        x: centerX,
        y: centerY,
        radius: radius
      });
      console.log('  Coordinate transform: (original * scale) + offset');
    }
  };

  // Canvas交互处理
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || workflowStep !== 'input') return;

    const rect = canvas.getBoundingClientRect();
    // 🔴 微调：考虑图像偏移量进行坐标转换
    const offsetX = canvas.imageOffsetX || 0;
    const offsetY = canvas.imageOffsetY || 0;

    const x = (e.clientX - rect.left - offsetX) / scale;
    const y = (e.clientY - rect.top - offsetY) / scale;

    const newInput = {
      ...userInput,
      center_x: x,
      center_y: y
    };

    setUserInput(newInput);
    setHasCenterSet(true);

    // 不自动触发API调用，只更新本地状态
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (workflowStep !== 'review' || !correctionResult) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // 🔴 微调：考虑图像偏移量进行坐标转换
    const offsetX = canvas.imageOffsetX || 0;
    const offsetY = canvas.imageOffsetY || 0;

    const x = (e.clientX - rect.left - offsetX) / scale;
    const y = (e.clientY - rect.top - offsetY) / scale;

    // 检查是否点击在圆形上
    const centerX = tempCirclePos.x || correctionResult.center_x;
    const centerY = tempCirclePos.y || correctionResult.center_y;
    const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));

    if (distance <= correctionResult.radius) {
      setIsDragging(true);
      setDragStart({ x: x - centerX, y: y - centerY });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !correctionResult) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // 🔴 微调：考虑图像偏移量进行坐标转换
    const offsetX = canvas.imageOffsetX || 0;
    const offsetY = canvas.imageOffsetY || 0;

    const x = (e.clientX - rect.left - offsetX) / scale;
    const y = (e.clientY - rect.top - offsetY) / scale;

    setTempCirclePos({
      x: x - dragStart.x,
      y: y - dragStart.y
    });

    drawCanvas();
  };

  const handleMouseUp = () => {
    if (isDragging && correctionResult && tempCirclePos.x && tempCirclePos.y) {
      // 🔴 关键修复：Canvas已经被缩放，直接使用拖拽坐标
      setCorrectionResult(prev => prev ? {
        ...prev,
        center_x: tempCirclePos.x,
        center_y: tempCirclePos.y
      } : null);
    }

    setIsDragging(false);
    setTempCirclePos({ x: 0, y: 0 });
  };

  // 手动触发矫正处理
  const handleManualCorrection = async () => {
    if (!hasCenterSet) {
      setError('请先在图像上点击设置血管中心位置');
      return;
    }

    if (userInput.major_axis <= 0 || userInput.minor_axis <= 0) {
      setError('请设置有效的长短轴长度');
      return;
    }

    setLoading(true);
    setError(null);
    setWorkflowStep('processing');

    try {
      const response = await api.post(`/annotation/projects/${projectId}/artifact-correction`, {
        slice_index: sliceIndex,
        axis: axis,
        center_x: userInput.center_x,
        center_y: userInput.center_y,
        major_axis: userInput.major_axis,
        minor_axis: userInput.minor_axis,
        roi_size: 80,
        use_spatial_continuity: true
      });

      // 详细检查API响应
      console.log('=== API RESPONSE DEBUG ===');
      console.log('Full response:', response);
      console.log('Response data:', response.data);
      console.log('Response type:', typeof response.data);

      // 验证关键字段
      const data = response.data;
      if (!data.center_x || !data.center_y || !data.radius) {
        console.error('❌ Missing required fields in response:', {
          center_x: data.center_x,
          center_y: data.center_y,
          radius: data.radius
        });
        setError('后端返回数据不完整，请检查坐标数据');
        setWorkflowStep('input');
        return;
      }

      // 验证数据类型和范围
      if (typeof data.center_x !== 'number' || typeof data.center_y !== 'number' || typeof data.radius !== 'number') {
        console.error('❌ Invalid data types:', {
          center_x_type: typeof data.center_x,
          center_y_type: typeof data.center_y,
          radius_type: typeof data.radius
        });
        setError('后端返回数据类型错误');
        setWorkflowStep('input');
        return;
      }

      console.log('✅ Data validation passed:', {
        center_x: data.center_x,
        center_y: data.center_y,
        radius: data.radius,
        confidence: data.confidence,
        method: data.method
      });

      // 🔴 坐标调试：检查API返回的坐标是否合理
      console.log('🔴 COORDINATE DEBUG BEFORE SETTING STATE:');
      console.log('  API返回的原始坐标:', {
        x: data.center_x,
        y: data.center_y,
        radius: data.radius
      });
      console.log('  当前Canvas缩放因子:', scale);
      console.log('  loadedImage尺寸:', loadedImageRef.current ? {
        width: loadedImageRef.current.width,
        height: loadedImageRef.current.height,
        naturalWidth: loadedImageRef.current.naturalWidth,
        naturalHeight: loadedImageRef.current.naturalHeight
      } : 'Image not loaded');

      setCorrectionResult(data);
      setWorkflowStep('review');
    } catch (err: any) {
      setError(err.response?.data?.detail || '矫正失败，请重试');
      setWorkflowStep('input'); // 失败时回到输入状态
    } finally {
      setLoading(false);
    }
  };

  // 保存矫正结果
  const saveCorrection = async () => {
    if (!correctionResult) return;

    setLoading(true);
    try {
      const finalAnnotation = {
        id: `circle_${sliceIndex}_${Math.round(correctionResult.center_x)}_${Math.round(correctionResult.center_y)}`,
        center_x: correctionResult.center_x,
        center_y: correctionResult.center_y,
        radius_x: correctionResult.radius,
        radius_y: correctionResult.radius,  // 圆形：长短轴相等
        rotation: 0.0,
        slice_index: sliceIndex,
        axis: axis,
        confidence: correctionResult.confidence,
        is_manual: false,
        method: correctionResult.method,
        correction_applied: true,
        created_at: new Date().toISOString()
      };

      await api.post(`/annotation/projects/${projectId}/save-corrected-annotation`, {
        project_id: projectId,
        layer_key: `${axis}_${sliceIndex}`,
        ellipses: [finalAnnotation],
        brush_strokes: []
      });

      onCorrectionComplete(finalAnnotation);
    } catch (err: any) {
      setError(err.response?.data?.detail || '保存失败');
      setWorkflowStep('review'); // 保存失败时停留在预览状态
    } finally {
      setLoading(false);
    }
  };

  // 重新调整 - 回到输入状态
  const handleRetry = () => {
    setWorkflowStep('input');
    setError(null);
  };

  // 取消操作
  const handleCancel = () => {
    onCancel();
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-4xl mx-auto">
      {/* 标题 */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Zap className="w-5 h-5 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-800">OCTA血管伪影矫正</h2>
        </div>
        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <span>切片: {sliceIndex}</span>
          <span>轴向: {axis.toUpperCase()}</span>
        </div>
      </div>

      {/* 步骤指示器 */}
      <div className="flex items-center justify-center mb-6">
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 ${workflowStep === 'input' ? 'text-blue-600' : 'text-gray-400'}`}>
            <MousePointer className="w-4 h-4" />
            <span>1. 设置血管参数</span>
          </div>
          <div className="text-gray-300">→</div>
          <div className={`flex items-center space-x-2 ${workflowStep === 'processing' ? 'text-blue-600' : 'text-gray-400'}`}>
            <RotateCw className={`w-4 h-4 ${workflowStep === 'processing' ? 'animate-spin' : ''}`} />
            <span>2. 算法矫正</span>
          </div>
          <div className="text-gray-300">→</div>
          <div className={`flex items-center space-x-2 ${workflowStep === 'review' ? 'text-blue-600' : 'text-gray-400'}`}>
            <CheckCircle className="w-4 h-4" />
            <span>3. 预览确认</span>
          </div>
        </div>
      </div>

      {/* 主要内容区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：图像显示区 */}
        <div className="lg:col-span-2">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="mb-2 text-sm text-gray-600">
              {workflowStep === 'initializing' && '正在加载图像...' }
              {workflowStep === 'input' && !hasCenterSet && '点击图像设置血管中心位置' }
              {workflowStep === 'input' && hasCenterSet && '调整长短轴参数，然后点击"开始矫正"' }
              {workflowStep === 'processing' && '算法正在处理，请稍候...' }
              {workflowStep === 'review' && '预览矫正结果，可拖拽微调位置，确认后保存' }
            </div>

            {/* 图像加载状态 */}
            {workflowStep === 'initializing' && (
              <div className="flex items-center justify-center h-96 bg-gray-100 rounded">
                <div className="text-center">
                  <RotateCw className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
                  <p className="text-gray-600">加载图像中...</p>
                </div>
              </div>
            )}

            {/* Canvas - 仅在图像加载完成后显示 */}
            {imageLoaded && (
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                className={`border border-gray-300 rounded bg-white block ${
                  workflowStep === 'input' ? 'cursor-crosshair' :
                  workflowStep === 'review' ? 'cursor-move' : 'cursor-default'
                }`}
                style={{
                  display: 'block',
                  verticalAlign: 'bottom',
                  imageRendering: 'crisp-edges',
                  imageRendering: 'pixelated'
                }}
                onClick={handleCanvasClick}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
            )}
          </div>
        </div>

        {/* 右侧：控制面板 */}
        <div className="space-y-4">
          {/* 阶段1：输入阶段 */}
          {workflowStep === 'input' && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-semibold text-gray-700 mb-3 flex items-center">
                <Settings className="w-4 h-4 mr-2" />
                参数设置
              </h3>

              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    中心位置 (点击图像自动设置)
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      value={Math.round(userInput.center_x ?? 0)}
                      onChange={(e) => {
                        const newInput = { ...userInput, center_x: Number(e.target.value) };
                        setUserInput(newInput);
                      }}
                      className="px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="X坐标"
                    />
                    <input
                      type="number"
                      value={Math.round(userInput.center_y ?? 0)}
                      onChange={(e) => {
                        const newInput = { ...userInput, center_y: Number(e.target.value) };
                        setUserInput(newInput);
                      }}
                      className="px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="Y坐标"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    长轴长度 (像素)
                  </label>
                  <input
                    type="range"
                    value={userInput.major_axis}
                    onChange={(e) => {
                      const newInput = { ...userInput, major_axis: Number(e.target.value) };
                      setUserInput(newInput);
                    }}
                    className="w-full"
                    min="5"
                    max="100"
                    step="1"
                  />
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>{userInput.major_axis}px</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    短轴长度 (像素)
                  </label>
                  <input
                    type="range"
                    value={userInput.minor_axis}
                    onChange={(e) => {
                      const newInput = { ...userInput, minor_axis: Number(e.target.value) };
                      setUserInput(newInput);
                    }}
                    className="w-full"
                    min="5"
                    max="100"
                    step="1"
                  />
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>{userInput.minor_axis}px</span>
                  </div>
                </div>

                <div className="text-xs text-gray-500 bg-blue-50 p-2 rounded">
                  <div className="font-medium mb-1">📐 几何关系：</div>
                  <div>真实半径 ≈ √(长轴×短轴)/2</div>
                  <div>倾斜角 = arccos(短轴/长轴)</div>
                </div>
              </div>

              <div className="mt-4 space-y-2">
                <button
                  onClick={handleManualCorrection}
                  disabled={loading || !hasCenterSet || userInput.major_axis <= 0 || userInput.minor_axis <= 0}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {loading ? (
                    <><RotateCw className="w-4 h-4 mr-2 animate-spin" />处理中...</>
                  ) : (
                    <><Zap className="w-4 h-4 mr-2" />开始矫正</>
                  )}
                </button>

                {!hasCenterSet && (
                  <div className="text-xs text-center text-yellow-600 bg-yellow-50 p-2 rounded">
                    💡 请先点击图像设置血管中心位置
                  </div>
                )}

                {hasCenterSet && (
                  <div className="text-xs text-center text-gray-600 bg-gray-100 p-2 rounded">
                    ✅ 血管中心已设置，调整参数后点击"开始矫正"
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 阶段3：预览确认阶段 */}
          {workflowStep === 'review' && correctionResult && (
            <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
              <h3 className="font-semibold text-red-700 mb-3 flex items-center">
                <CheckCircle className="w-4 h-4 mr-2" />
                矫正结果预览
              </h3>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">圆形中心 X:</span>
                  <span className="font-medium">{(correctionResult.center_x ?? 0).toFixed(0)} px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">圆形中心 Y:</span>
                  <span className="font-medium">{(correctionResult.center_y ?? 0).toFixed(0)} px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">矫正半径:</span>
                  <span className="font-medium text-red-600 font-bold">{(correctionResult.radius ?? 0).toFixed(1)} px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">原始长轴:</span>
                  <span className="font-medium">{userInput.major_axis}px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">原始短轴:</span>
                  <span className="font-medium">{userInput.minor_axis}px</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">置信度:</span>
                  <span className="font-medium">{((correctionResult.confidence ?? 0) * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">处理方法:</span>
                  <span className="font-medium text-xs">{correctionResult.method ?? '未知'}</span>
                </div>
              </div>

              <div className="mt-4 p-2 bg-red-100 rounded text-xs text-red-700">
                🔴 红色圆形已在图像上绘制，请检查是否准确
              </div>

              <div className="mt-4 space-y-2">
                <button
                  onClick={saveCorrection}
                  disabled={loading}
                  className="w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 disabled:bg-gray-400 flex items-center justify-center"
                >
                  {loading ? (
                    <><RotateCw className="w-4 h-4 mr-2 animate-spin" />保存中...</>
                  ) : (
                    <><Save className="w-4 h-4 mr-2" />确认保存</>
                  )}
                </button>

                <button
                  onClick={handleRetry}
                  disabled={loading}
                  className="w-full bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700 disabled:bg-gray-400 flex items-center justify-center"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  重新调整
                </button>
              </div>
            </div>
          )}

          {/* 错误提示 */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <div className="text-red-700 text-sm">{error}</div>
            </div>
          )}

          {/* 取消按钮 */}
          <button
            onClick={handleCancel}
            disabled={loading}
            className="w-full bg-gray-200 text-gray-700 py-2 px-4 rounded hover:bg-gray-300 disabled:bg-gray-100 disabled:text-gray-400"
          >
            取消
          </button>
        </div>
      </div>
    </div>
  );
};

export default ArtifactCorrectionTool;