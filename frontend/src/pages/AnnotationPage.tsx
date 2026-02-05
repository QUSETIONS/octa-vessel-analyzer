/**
 * OCTA 血管标注 - V7版本（优化版）
 * 
 * 改进：
 * 1. 支持椭圆旋转角度显示和编辑
 * 2. 多种自动拟合方法选择
 * 3. 增强的检测参数控制
 * 4. 更好的视觉反馈
 * 5. 快捷键支持
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Circle, Brush, Eraser, ZoomIn, ZoomOut, Save, Download,
  Crosshair, Play, Wand2, Loader2, Trash2, Eye, Target, CheckCircle,
  RotateCw, Move, Settings2, Sparkles, MousePointer, Scan, Layers, Zap
} from 'lucide-react';
import { api } from '../utils/api';
import ArtifactCorrectionTool from '../components/ArtifactCorrectionTool';

type ToolType = 'ellipse' | 'brush' | 'eraser' | 'select' | 'auto-fit' | 'artifact-correction';
type FitMethod = 'auto' | 'edge' | 'region' | 'blob';
type DetectMethod = 'fast' | 'combined' | 'blob' | 'edge';

interface Ellipse {
  id: string;
  centerX: number;
  centerY: number;
  radiusX: number;
  radiusY: number;
  rotation: number;
  confidence?: number;
  isManual?: boolean;
  method?: string;
}

interface BrushStroke {
  id: string;
  points: { x: number; y: number }[];
  brushSize: number;
  isEraser: boolean;
}

const AnnotationPage: React.FC = () => {
  // ========================================
  // 状态管理
  // ========================================
  const [volumes, setVolumes] = useState<any[]>([]);
  const [projects, setProjects] = useState<any[]>([]);
  const [selectedVolumeId, setSelectedVolumeId] = useState('');
  const [projectId, setProjectId] = useState('');

  const [currentAxis, setCurrentAxis] = useState<'x' | 'y' | 'z'>('y');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [maxSliceIndex, setMaxSliceIndex] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  const [tool, setTool] = useState<ToolType>('select');
  const [brushSize, setBrushSize] = useState(5);
  const [selectedEllipseId, setSelectedEllipseId] = useState<string | null>(null);

  const [ellipses, setEllipses] = useState<Ellipse[]>([]);
  const [strokes, setStrokes] = useState<BrushStroke[]>([]);
  const [currentStroke, setCurrentStroke] = useState<{ x: number; y: number }[]>([]);

  const [showAnnotations, setShowAnnotations] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [loadedImage, setLoadedImage] = useState<HTMLImageElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [currentMousePos, setCurrentMousePos] = useState<{ x: number; y: number } | null>(null);

  // 自动标注参数
  const [autoLabeling, setAutoLabeling] = useState(false);
  const [backendVersion, setBackendVersion] = useState('');
  const [autoAxis, setAutoAxis] = useState('y');
  const [autoMinRadius, setAutoMinRadius] = useState(3);
  const [autoMaxRadius, setAutoMaxRadius] = useState(15);
  const [autoSensitivity, setAutoSensitivity] = useState('medium');
  const [autoRegen, setAutoRegen] = useState(true);
  const [useCLAHE, setUseCLAHE] = useState(true);
  const [detectMethod, setDetectMethod] = useState<DetectMethod>('fast');

  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [showDebug, setShowDebug] = useState(false);

  // 伪影矫正工具状态
  const [showArtifactCorrection, setShowArtifactCorrection] = useState(false);

  // 调试检测
  const debugDetection = async () => {
    if (!projectId) return;
    try {
      const r = await api.post(`/annotation/projects/${projectId}/debug-detection`, {
        slice_index: sliceIndex,
        axis: currentAxis
      });
      setDebugInfo(r.data);
      setShowDebug(true);
      console.log('[DEBUG]', r.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || '调试失败');
    }
  };

  // 点击拟合参数
  const [fitMethod, setFitMethod] = useState<FitMethod>('auto');
  const [fitRoiSize, setFitRoiSize] = useState(50);
  const [isFitting, setIsFitting] = useState(false);

  // 椭圆编辑
  const [editingEllipse, setEditingEllipse] = useState<Ellipse | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ========================================
  // 数据加载
  // ========================================
  useEffect(() => {
    api.get('/data/list').then(r => setVolumes(r.data || [])).catch(() => {});
    api.get('/annotation/projects').then(r => setProjects(r.data || [])).catch(() => {});
    
    // 获取后端版本
    api.get('/annotation/version').then(r => setBackendVersion(r.data?.version || '')).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedVolumeId) return;
    const v = volumes.find(x => x.id === selectedVolumeId);
    if (v?.shape) {
      const max = v.shape[{x:0, y:1, z:2}[currentAxis]] - 1;
      setMaxSliceIndex(max);
      setSliceIndex(Math.floor(max / 2));
    }
  }, [selectedVolumeId, currentAxis, volumes]);

  useEffect(() => {
    if (!selectedVolumeId) return;
    api.get(`/data/${selectedVolumeId}/slice/${currentAxis}/${sliceIndex}`, { responseType: 'blob' })
      .then(r => {
        const img = new Image();
        img.onload = () => setLoadedImage(img);
        img.src = URL.createObjectURL(r.data);
      })
      .catch(() => setLoadedImage(null));
  }, [selectedVolumeId, currentAxis, sliceIndex]);

  const loadLayer = useCallback(async () => {
    if (!projectId) { setEllipses([]); setStrokes([]); return; }
    try {
      const r = await api.get(`/annotation/projects/${projectId}/layer/${currentAxis}_${sliceIndex}`);
      setEllipses((r.data.ellipses || []).map((e: any) => ({
        id: e.id, centerX: e.center_x, centerY: e.center_y,
        radiusX: e.radius_x, radiusY: e.radius_y, rotation: e.rotation || 0,
        confidence: e.confidence || 1, isManual: e.is_manual, method: e.method
      })));
      setStrokes((r.data.brush_strokes || []).map((s: any) => ({
        id: s.id, points: s.points, brushSize: s.brush_size, isEraser: s.is_eraser
      })));
    } catch { setEllipses([]); setStrokes([]); }
  }, [projectId, currentAxis, sliceIndex]);

  useEffect(() => { loadLayer(); }, [loadLayer]);

  // ========================================
  // 绘制
  // ========================================
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !loadedImage) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = loadedImage.width * zoom;
    canvas.height = loadedImage.height * zoom;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.drawImage(loadedImage, 0, 0);

    if (showAnnotations) {
      // 绘制椭圆
      ellipses.forEach((e, i) => {
        const sel = e.id === selectedEllipseId;
        const conf = e.confidence || 1;
        
        // 根据置信度选择颜色
        let color: string;
        if (e.isManual) {
          color = '#00ffff'; // 手动标注：青色
        } else if (conf >= 0.8) {
          color = '#00ff00'; // 高置信度：绿色
        } else if (conf >= 0.6) {
          color = '#ffff00'; // 中等置信度：黄色
        } else {
          color = '#ff8800'; // 低置信度：橙色
        }

        ctx.save();
        ctx.translate(e.centerX, e.centerY);
        ctx.rotate((e.rotation || 0) * Math.PI / 180);

        // 填充
        ctx.fillStyle = sel ? 'rgba(0,255,255,0.25)' : `${color}22`;
        ctx.beginPath();
        ctx.ellipse(0, 0, Math.max(1, e.radiusX), Math.max(1, e.radiusY), 0, 0, 2 * Math.PI);
        ctx.fill();

        // 边框
        ctx.strokeStyle = sel ? '#00ffff' : color;
        ctx.lineWidth = (sel ? 3 : 2) / zoom;
        ctx.stroke();

        // 中心点
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(0, 0, 3 / zoom, 0, 2 * Math.PI);
        ctx.fill();

        // 旋转指示线（如果有旋转）
        if (Math.abs(e.rotation || 0) > 1) {
          ctx.strokeStyle = '#ff00ff';
          ctx.lineWidth = 1 / zoom;
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(e.radiusX, 0);
          ctx.stroke();
        }

        ctx.restore();

        // 标签（在椭圆外部显示）
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${11 / zoom}px Arial`;
        ctx.textAlign = 'center';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2 / zoom;
        
        const labelY = e.centerY - Math.max(e.radiusX, e.radiusY) - 10 / zoom;
        let labelText = `${i + 1}`;
        
        if (showConfidence && !e.isManual) {
          labelText += ` (${(conf * 100).toFixed(0)}%)`;
        }
        
        ctx.strokeText(labelText, e.centerX, labelY);
        ctx.fillText(labelText, e.centerX, labelY);
      });

      // 绘制画笔笔迹
      strokes.forEach(s => {
        if (s.points.length < 2) return;
        ctx.strokeStyle = s.isEraser ? 'rgba(255,0,0,0.7)' : 'rgba(0,255,0,0.7)';
        ctx.lineWidth = s.brushSize / zoom;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(s.points[0].x, s.points[0].y);
        s.points.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.stroke();
      });

      // 当前笔迹
      if (currentStroke.length > 1) {
        ctx.strokeStyle = tool === 'eraser' ? 'rgba(255,0,0,0.7)' : 'rgba(0,255,0,0.7)';
        ctx.lineWidth = brushSize / zoom;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(currentStroke[0].x, currentStroke[0].y);
        currentStroke.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.stroke();
      }

      // 正在绘制的椭圆预览
      if (isDrawing && tool === 'ellipse' && drawStart && currentMousePos) {
        const cx = (drawStart.x + currentMousePos.x) / 2;
        const cy = (drawStart.y + currentMousePos.y) / 2;
        const rx = Math.abs(currentMousePos.x - drawStart.x) / 2;
        const ry = Math.abs(currentMousePos.y - drawStart.y) / 2;
        ctx.strokeStyle = '#00ffff';
        ctx.setLineDash([5 / zoom, 3 / zoom]);
        ctx.lineWidth = 2 / zoom;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 自动拟合工具的十字准星
      if (tool === 'auto-fit' && currentMousePos) {
        ctx.strokeStyle = '#ff00ff';
        ctx.lineWidth = 1 / zoom;
        ctx.setLineDash([3 / zoom, 3 / zoom]);
        
        const size = fitRoiSize / 2;
        ctx.strokeRect(
          currentMousePos.x - size,
          currentMousePos.y - size,
          fitRoiSize,
          fitRoiSize
        );
        
        // 十字线
        ctx.beginPath();
        ctx.moveTo(currentMousePos.x - 10 / zoom, currentMousePos.y);
        ctx.lineTo(currentMousePos.x + 10 / zoom, currentMousePos.y);
        ctx.moveTo(currentMousePos.x, currentMousePos.y - 10 / zoom);
        ctx.lineTo(currentMousePos.x, currentMousePos.y + 10 / zoom);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    
    ctx.restore();
  }, [loadedImage, zoom, showAnnotations, showConfidence, ellipses, strokes, currentStroke, brushSize, tool, isDrawing, drawStart, currentMousePos, selectedEllipseId, fitRoiSize]);

  // ========================================
  // 交互处理
  // ========================================
  const getCoords = (e: React.MouseEvent) => {
    const r = canvasRef.current?.getBoundingClientRect();
    return r ? { x: (e.clientX - r.left) / zoom, y: (e.clientY - r.top) / zoom } : { x: 0, y: 0 };
  };

  const handleAutoFitAtPoint = async (x: number, y: number) => {
    if (!projectId) return;
    
    setIsFitting(true);
    try {
      const r = await api.post(`/annotation/projects/${projectId}/auto-fit-at-point`, {
        slice_index: sliceIndex,
        axis: currentAxis,
        click_x: x,
        click_y: y,
        roi_size: fitRoiSize,
        method: fitMethod
      });
      
      const newEllipse: Ellipse = {
        id: `fit_${Date.now()}`,
        centerX: r.data.center_x,
        centerY: r.data.center_y,
        radiusX: r.data.radius_x,
        radiusY: r.data.radius_y,
        rotation: r.data.rotation || 0,
        confidence: r.data.confidence,
        isManual: false,
        method: r.data.method
      };
      
      setEllipses(prev => [...prev, newEllipse]);
      setSelectedEllipseId(newEllipse.id);
      
    } catch (err: any) {
      const msg = err.response?.data?.detail || '拟合失败，请尝试点击血管中心';
      console.warn('[AutoFit]', msg);
      // 可以显示一个小提示而不是 alert
    } finally {
      setIsFitting(false);
    }
  };

  const handleMouseDown = async (e: React.MouseEvent) => {
    const c = getCoords(e);

    // Alt + 点击：快速自动拟合（任何工具下都生效）
    if (e.altKey && projectId) {
      await handleAutoFitAtPoint(c.x, c.y);
      return;
    }

    // 自动拟合工具
    if (tool === 'auto-fit') {
      await handleAutoFitAtPoint(c.x, c.y);
      return;
    }

    // 选择工具
    if (tool === 'select') {
      const found = [...ellipses].reverse().find(el => {
        // 考虑旋转的点击检测
        const dx = c.x - el.centerX;
        const dy = c.y - el.centerY;
        const angle = -(el.rotation || 0) * Math.PI / 180;
        const rx = dx * Math.cos(angle) - dy * Math.sin(angle);
        const ry = dx * Math.sin(angle) + dy * Math.cos(angle);
        return (rx / el.radiusX) ** 2 + (ry / el.radiusY) ** 2 <= 1.5;
      });
      setSelectedEllipseId(found?.id || null);
      if (found) {
        setEditingEllipse({ ...found });
      } else {
        setEditingEllipse(null);
      }
      return;
    }

    setIsDrawing(true);
    if (tool === 'ellipse') {
      setDrawStart(c);
      setCurrentMousePos(c);
    } else if (tool === 'brush' || tool === 'eraser') {
      setCurrentStroke([c]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const c = getCoords(e);
    setCurrentMousePos(c);
    if (isDrawing && (tool === 'brush' || tool === 'eraser')) {
      setCurrentStroke(prev => [...prev, c]);
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    const c = getCoords(e);
    setIsDrawing(false);

    if (tool === 'ellipse' && drawStart) {
      const cx = (drawStart.x + c.x) / 2;
      const cy = (drawStart.y + c.y) / 2;
      const rx = Math.abs(c.x - drawStart.x) / 2;
      const ry = Math.abs(c.y - drawStart.y) / 2;
      if (rx > 2 && ry > 2) {
        const newEllipse: Ellipse = {
          id: `m_${Date.now()}`,
          centerX: cx,
          centerY: cy,
          radiusX: rx,
          radiusY: ry,
          rotation: 0,
          confidence: 1,
          isManual: true
        };
        setEllipses(prev => [...prev, newEllipse]);
        setSelectedEllipseId(newEllipse.id);
      }
      setDrawStart(null);
    } else if ((tool === 'brush' || tool === 'eraser') && currentStroke.length > 1) {
      setStrokes(prev => [...prev, {
        id: `s_${Date.now()}`,
        points: currentStroke,
        brushSize,
        isEraser: tool === 'eraser'
      }]);
      setCurrentStroke([]);
    }
  };

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 删除选中的椭圆
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedEllipseId) {
        setEllipses(prev => prev.filter(el => el.id !== selectedEllipseId));
        setSelectedEllipseId(null);
        setEditingEllipse(null);
      }
      
      // 工具快捷键
      if (!e.ctrlKey && !e.altKey && !e.metaKey) {
        switch (e.key.toLowerCase()) {
          case 'v': setTool('select'); break;
          case 'e': setTool('ellipse'); break;
          case 'b': setTool('brush'); break;
          case 'x': setTool('eraser'); break;
          case 'a': setTool('auto-fit'); break;
          case 'r': setShowArtifactCorrection(true); break;
        }
      }
      
      // 切片导航
      if (e.key === 'ArrowLeft' || e.key === ',') {
        setSliceIndex(prev => Math.max(0, prev - 1));
      }
      if (e.key === 'ArrowRight' || e.key === '.') {
        setSliceIndex(prev => Math.min(maxSliceIndex, prev + 1));
      }
      
      // Ctrl+S 保存
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        save();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedEllipseId, maxSliceIndex]);

  // ========================================
  // 操作函数
  // ========================================
  const save = async () => {
    if (!projectId) return alert('请选择项目');
    await api.post(`/annotation/projects/${projectId}/save`, {
      project_id: projectId,
      layer_key: `${currentAxis}_${sliceIndex}`,
      ellipses: ellipses.map(e => ({
        id: e.id,
        center_x: e.centerX,
        center_y: e.centerY,
        radius_x: e.radiusX,
        radius_y: e.radiusY,
        rotation: e.rotation || 0,
        slice_index: sliceIndex,
        axis: currentAxis,
        confidence: e.confidence,
        is_manual: e.isManual
      })),
      brush_strokes: strokes.map(s => ({
        id: s.id,
        points: s.points,
        brush_size: s.brushSize,
        is_eraser: s.isEraser,
        slice_index: sliceIndex,
        axis: currentAxis
      }))
    });
    alert('✅ 已保存');
  };

  const genMask = async () => {
    if (!projectId) return;
    const r = await api.post(`/annotation/projects/${projectId}/generate-mask`);
    alert(`✅ 生成完成\n体素数: ${r.data.vessel_voxels}\n占比: ${r.data.vessel_fraction}`);
  };

  const autoLabel = async () => {
    if (!projectId) return alert('请选择项目');
    setAutoLabeling(true);
    try {
      const r = await api.post(`/annotation/projects/${projectId}/autolabel`, {
        axis: autoAxis,
        min_radius: autoMinRadius,
        max_radius: autoMaxRadius,
        sensitivity: autoSensitivity,
        regen: autoRegen,
        min_slices: 1,
        continuity_threshold: 20,
        use_clahe: useCLAHE,
        detect_method: detectMethod
      });
      setBackendVersion(r.data.code_version || '');
      alert(`✅ 检测完成\n版本: ${r.data.code_version}\n椭圆数: ${r.data.added_ellipses}\n有旋转: ${r.data.ellipses_with_rotation}\n覆盖率: ${r.data.coverage_percent}`);
      setCurrentAxis(autoAxis as any);
      setTimeout(loadLayer, 300);
    } catch (err: any) {
      alert(err.response?.data?.detail || '检测失败');
    } finally {
      setAutoLabeling(false);
    }
  };

  const createProj = async () => {
    if (!selectedVolumeId) return alert('请选择数据');
    const name = prompt('项目名称');
    if (!name) return;
    const r = await api.post('/annotation/projects', { volume_id: selectedVolumeId, name });
    setProjects(prev => [...prev, r.data]);
    setProjectId(r.data.id);
  };

  const updateSelectedEllipse = (updates: Partial<Ellipse>) => {
    if (!selectedEllipseId) return;
    setEllipses(prev => prev.map(e => 
      e.id === selectedEllipseId ? { ...e, ...updates } : e
    ));
    if (editingEllipse) {
      setEditingEllipse({ ...editingEllipse, ...updates });
    }
  };

  // ========================================
  // 渲染
  // ========================================
  return (
    <div className="flex h-full gap-4">
      {/* 左侧面板 */}
      <div className="w-80 space-y-3 flex-shrink-0 overflow-y-auto">
        
        {/* 数据/项目选择 */}
        <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 space-y-2">
          <select 
            value={selectedVolumeId} 
            onChange={e => setSelectedVolumeId(e.target.value)} 
            className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm"
          >
            <option value="">选择数据...</option>
            {volumes.map(v => <option key={v.id} value={v.id}>{v.filename}</option>)}
          </select>
          <div className="flex gap-2">
            <select 
              value={projectId} 
              onChange={e => setProjectId(e.target.value)} 
              className="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm"
            >
              <option value="">选择项目...</option>
              {projects.filter(p => !selectedVolumeId || p.volume_id === selectedVolumeId).map(p => 
                <option key={p.id} value={p.id}>{p.name}</option>
              )}
            </select>
            <button onClick={createProj} className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm">新建</button>
          </div>
        </div>

        {/* 工具栏 */}
        <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 space-y-2">
          <div className="text-xs text-gray-400 mb-1">工具 (快捷键)</div>
          <div className="grid grid-cols-6 gap-1">
            {[
              { id: 'select', icon: MousePointer, label: '选择', key: 'V' },
              { id: 'ellipse', icon: Circle, label: '椭圆', key: 'E' },
              { id: 'brush', icon: Brush, label: '画笔', key: 'B' },
              { id: 'eraser', icon: Eraser, label: '橡皮', key: 'X' },
              { id: 'auto-fit', icon: Sparkles, label: '拟合', key: 'A' },
              { id: 'artifact-correction', icon: Zap, label: '矫正', key: 'R' },
            ].map(({ id, icon: Icon, label, key }) => (
              <button
                key={id}
                onClick={() => {
                  if (id === 'artifact-correction') {
                    setShowArtifactCorrection(true);
                  } else {
                    setTool(id as ToolType);
                  }
                }}
                className={`flex flex-col items-center p-1.5 rounded transition-colors ${
                  (id === 'artifact-correction' ? showArtifactCorrection : tool === id)
                    ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                }`}
                title={`${label} (${key})`}
              >
                <Icon size={14} />
                <span className="text-[10px] mt-0.5">{key}</span>
              </button>
            ))}
          </div>
          
          {/* 自动拟合参数 */}
          {tool === 'auto-fit' && (
            <div className="mt-2 pt-2 border-t border-gray-700 space-y-2">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">拟合方法</span>
                <select 
                  value={fitMethod} 
                  onChange={e => setFitMethod(e.target.value as FitMethod)}
                  className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5 text-sm"
                >
                  <option value="auto">自动</option>
                  <option value="edge">边缘</option>
                  <option value="region">区域生长</option>
                  <option value="blob">Blob</option>
                </select>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-400">ROI 大小</span>
                <input
                  type="number"
                  value={fitRoiSize}
                  min={20}
                  max={100}
                  onChange={e => setFitRoiSize(+e.target.value || 50)}
                  className="w-16 bg-gray-700 border border-gray-600 rounded px-2 py-0.5 text-center text-sm"
                />
              </div>
            </div>
          )}
          
          {/* 删除按钮 */}
          {selectedEllipseId && (
            <button
              onClick={() => {
                setEllipses(p => p.filter(e => e.id !== selectedEllipseId));
                setSelectedEllipseId(null);
                setEditingEllipse(null);
              }}
              className="w-full py-1.5 bg-red-600 hover:bg-red-700 rounded text-sm flex items-center justify-center gap-1"
            >
              <Trash2 size={12} />
              删除选中 (Del)
            </button>
          )}
        </div>

        {/* 椭圆编辑器 */}
        {editingEllipse && (
          <div className="bg-gray-800 rounded-lg p-3 border border-blue-600 space-y-2">
            <h3 className="text-sm font-medium flex items-center gap-1">
              <Settings2 size={14} />
              椭圆编辑
            </h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <label className="text-gray-400 text-xs">中心 X</label>
                <input
                  type="number"
                  value={editingEllipse.centerX.toFixed(1)}
                  onChange={e => updateSelectedEllipse({ centerX: +e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs">中心 Y</label>
                <input
                  type="number"
                  value={editingEllipse.centerY.toFixed(1)}
                  onChange={e => updateSelectedEllipse({ centerY: +e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs">半径 X</label>
                <input
                  type="number"
                  value={editingEllipse.radiusX.toFixed(1)}
                  onChange={e => updateSelectedEllipse({ radiusX: +e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                />
              </div>
              <div>
                <label className="text-gray-400 text-xs">半径 Y</label>
                <input
                  type="number"
                  value={editingEllipse.radiusY.toFixed(1)}
                  onChange={e => updateSelectedEllipse({ radiusY: +e.target.value })}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                />
              </div>
              <div className="col-span-2">
                <label className="text-gray-400 text-xs flex items-center gap-1">
                  <RotateCw size={10} />
                  旋转角度
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={-90}
                    max={90}
                    value={editingEllipse.rotation || 0}
                    onChange={e => updateSelectedEllipse({ rotation: +e.target.value })}
                    className="flex-1"
                  />
                  <span className="text-xs w-10 text-right">{(editingEllipse.rotation || 0).toFixed(0)}°</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 自动检测 */}
        <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 space-y-2">
          <h3 className="font-medium text-sm flex items-center gap-1">
            <Wand2 size={14} />
            自动检测
          </h3>

          {backendVersion && (
            <div className="text-xs p-1.5 rounded bg-green-500/20 text-green-300 flex items-center gap-1">
              <CheckCircle size={12} />{backendVersion}
            </div>
          )}

          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">轴向</span>
              <select value={autoAxis} onChange={e => setAutoAxis(e.target.value)} className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5">
                <option value="x">X</option>
                <option value="y">Y</option>
                <option value="z">Z</option>
              </select>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">半径范围</span>
              <div className="flex items-center gap-1">
                <input type="number" value={autoMinRadius} min={1} max={20} onChange={e => setAutoMinRadius(+e.target.value || 2)} className="w-10 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-center" />
                <span>~</span>
                <input type="number" value={autoMaxRadius} min={5} max={50} onChange={e => setAutoMaxRadius(+e.target.value || 15)} className="w-10 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-center" />
              </div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">灵敏度</span>
              <select value={autoSensitivity} onChange={e => setAutoSensitivity(e.target.value)} className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5">
                <option value="low">低</option>
                <option value="medium">中</option>
                <option value="high">高</option>
              </select>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400">检测方法</span>
              <select value={detectMethod} onChange={e => setDetectMethod(e.target.value as DetectMethod)} className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5">
                <option value="fast">⚡ 快速</option>
                <option value="combined">融合</option>
                <option value="edge">边缘</option>
                <option value="blob">Blob</option>
              </select>
            </div>
            
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <label className="flex items-center gap-1.5 text-xs">
                <input type="checkbox" checked={useCLAHE} onChange={e => setUseCLAHE(e.target.checked)} />
                CLAHE 增强
              </label>
              <label className="flex items-center gap-1.5 text-xs">
                <input type="checkbox" checked={autoRegen} onChange={e => setAutoRegen(e.target.checked)} />
                清空旧标注
              </label>
            </div>
          </div>

          <button
            onClick={autoLabel}
            disabled={autoLabeling || !projectId}
            className={`w-full py-2 rounded flex items-center justify-center gap-1 text-sm transition-colors ${
              autoLabeling ? 'bg-gray-600' : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {autoLabeling ? (
              <><Loader2 size={14} className="animate-spin" />检测中...</>
            ) : (
              <><Scan size={14} />开始全局检测</>
            )}
          </button>

          <button
            onClick={debugDetection}
            disabled={!projectId}
            className="w-full py-1.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded text-sm flex items-center justify-center gap-1"
          >
            🔍 调试当前切片
          </button>

          {debugInfo && showDebug && (
            <div className="mt-2 p-2 bg-gray-900 rounded text-xs max-h-60 overflow-y-auto">
              <div className="flex justify-between items-center mb-1">
                <span className="text-yellow-400 font-bold">调试信息</span>
                <button onClick={() => setShowDebug(false)} className="text-gray-400 hover:text-white">✕</button>
              </div>
              <div className="text-gray-300 space-y-1">
                <div>图像: {debugInfo.image_shape?.[0]}×{debugInfo.image_shape?.[1]}</div>
                <div>范围: {debugInfo.image_min?.toFixed(0)} ~ {debugInfo.image_max?.toFixed(0)}</div>
                <div>均值: {debugInfo.image_mean?.toFixed(1)}</div>
                {debugInfo.steps?.map((step: any, i: number) => (
                  <div key={i} className="border-t border-gray-700 pt-1 mt-1">
                    <div className="text-blue-400">{step.name}</div>
                    {step.stats && (
                      <div>均值:{step.stats.mean?.toFixed(1)} 标准差:{step.stats.std?.toFixed(1)}</div>
                    )}
                    {step.thresholds && step.thresholds.map((t: any, j: number) => (
                      <div key={j} className="text-gray-400">
                        P{t.percentile}: 阈值={t.threshold_value?.toFixed(0)}, 白={t.white_ratio}, 轮廓={t.contour_count}
                      </div>
                    ))}
                    {step.count !== undefined && (
                      <div className="text-green-400">检测到: {step.count} 个</div>
                    )}
                    {step.details && step.details.slice(0, 5).map((d: any, j: number) => (
                      <div key={j} className="text-gray-500 pl-2">
                        #{d.index}: 面积={d.area?.toFixed(0)}, 位置=[{d.center_approx?.[0]},{d.center_approx?.[1]}]
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="text-xs text-gray-400 border-t border-gray-700 pt-2 space-y-0.5">
            <div>💡 <strong>Alt+点击</strong> = 任意位置快速拟合</div>
            <div>💡 <strong>A 键</strong> = 切换到拟合工具</div>
          </div>
        </div>

        {/* 视图控制 */}
        <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 space-y-2">
          <div className="flex gap-1">
            {(['x', 'y', 'z'] as const).map(a => (
              <button
                key={a}
                onClick={() => setCurrentAxis(a)}
                className={`flex-1 py-1.5 rounded text-sm transition-colors ${
                  currentAxis === a ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                {a.toUpperCase()} 轴
              </button>
            ))}
          </div>

          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-0.5">
              <span>切片 (←→)</span>
              <span>{sliceIndex} / {maxSliceIndex}</span>
            </div>
            <input
              type="range"
              min={0}
              max={maxSliceIndex}
              value={sliceIndex}
              onChange={e => setSliceIndex(+e.target.value)}
              className="w-full"
            />
          </div>

          <div className="flex items-center gap-1">
            <button onClick={() => setZoom(z => Math.max(0.5, z - 0.25))} className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded">
              <ZoomOut size={12} />
            </button>
            <span className="flex-1 text-center text-xs">{(zoom * 100).toFixed(0)}%</span>
            <button onClick={() => setZoom(z => Math.min(4, z + 0.25))} className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded">
              <ZoomIn size={12} />
            </button>
          </div>

          <div className="flex gap-2">
            <label className="flex items-center gap-1.5 text-xs">
              <input type="checkbox" checked={showAnnotations} onChange={e => setShowAnnotations(e.target.checked)} />
              <Eye size={10} />标注
            </label>
            <label className="flex items-center gap-1.5 text-xs">
              <input type="checkbox" checked={showConfidence} onChange={e => setShowConfidence(e.target.checked)} />
              置信度
            </label>
          </div>

          <div className="text-xs border-t border-gray-700 pt-1.5 flex justify-between">
            <span className="text-gray-400">椭圆数:</span>
            <span className="text-yellow-400 font-medium">{ellipses.length}</span>
            <span className="text-gray-400">手动:</span>
            <span className="text-cyan-400 font-medium">{ellipses.filter(e => e.isManual).length}</span>
          </div>
        </div>

        {/* 操作按钮 */}
        <div className="bg-gray-800 rounded-lg p-3 border border-gray-700 space-y-1.5">
          <button
            onClick={save}
            disabled={!projectId}
            className="w-full py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-sm flex items-center justify-center gap-1"
          >
            <Save size={14} />
            保存 (Ctrl+S)
          </button>
          <button
            onClick={genMask}
            disabled={!projectId}
            className="w-full py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded text-sm flex items-center justify-center gap-1"
          >
            <Layers size={14} />
            生成金标准
          </button>
          <button
            onClick={() => { setEllipses([]); setStrokes([]); }}
            className="w-full py-1.5 bg-red-600/70 hover:bg-red-600 rounded text-sm flex items-center justify-center gap-1"
          >
            <Trash2 size={12} />
            清空当前切片
          </button>
        </div>
      </div>

      {/* 画布区域 */}
      <div ref={containerRef} className="flex-1 bg-gray-800 rounded-lg border border-gray-700 overflow-auto relative">
        <div className="p-4 min-h-full flex items-center justify-center">
          {selectedVolumeId && loadedImage ? (
            <div className="relative">
              <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => {
                  setIsDrawing(false);
                  setCurrentMousePos(null);
                }}
                className={`border border-gray-600 ${
                  tool === 'auto-fit' ? 'cursor-crosshair' :
                  tool === 'select' ? 'cursor-pointer' :
                  'cursor-crosshair'
                }`}
              />
              
              {/* 状态指示 */}
              <div className="absolute top-2 right-2 flex flex-col gap-1">
                {ellipses.length > 0 && (
                  <div className="bg-black/70 px-2 py-1 rounded text-xs text-yellow-400">
                    🎯 {ellipses.length} 椭圆
                  </div>
                )}
                {isFitting && (
                  <div className="bg-purple-600/80 px-2 py-1 rounded text-xs flex items-center gap-1">
                    <Loader2 size={10} className="animate-spin" />
                    拟合中...
                  </div>
                )}
              </div>

              {/* 当前工具提示 */}
              <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-1 rounded text-xs text-gray-300">
                {tool === 'auto-fit' ? '点击血管中心自动拟合' :
                 tool === 'ellipse' ? '拖拽绘制椭圆' :
                 tool === 'select' ? '点击选择椭圆' :
                 tool === 'brush' ? '绘制血管区域' :
                 '擦除区域'
                }
              </div>
            </div>
          ) : (
            <div className="text-gray-500 text-center">
              <Crosshair size={48} className="mx-auto mb-3 opacity-50" />
              <p>{selectedVolumeId ? '加载中...' : '请选择数据'}</p>
            </div>
          )}
        </div>
      </div>

      {/* 伪影矫正工具弹窗 */}
      {showArtifactCorrection && selectedVolumeId && loadedImage && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="w-full max-w-6xl h-[90vh] overflow-auto bg-white rounded-lg">
            <ArtifactCorrectionTool
              projectId={projectId}
              sliceIndex={sliceIndex}
              axis={currentAxis}
              imageUrl={loadedImage.src}
              onCorrectionComplete={(annotation) => {
                // 将矫正后的圆形添加到当前椭圆列表
                const newEllipse = {
                  id: annotation.id,
                  centerX: annotation.center_x,
                  centerY: annotation.center_y,
                  radiusX: annotation.radius_x,
                  radiusY: annotation.radius_y,
                  rotation: annotation.rotation,
                  confidence: annotation.confidence,
                  isManual: false,
                  method: annotation.method
                };
                setEllipses(prev => [...prev, newEllipse]);
                setShowArtifactCorrection(false);
                // 保存到服务器
                save();
              }}
              onCancel={() => {
                setShowArtifactCorrection(false);
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default AnnotationPage;
