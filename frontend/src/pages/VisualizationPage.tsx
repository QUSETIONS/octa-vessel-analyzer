/**
 * OCTA 血管网络分析系统 - 三维可视化页面
 * 
 * 功能：
 * 1. 3D 血管网络渲染
 * 2. 交互控制（旋转、缩放、平移）
 * 3. 显示模式切换
 * 4. 网格导出
 */

import React, { useState, useEffect, useRef, Suspense } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { 
  Box, 
  Layers, 
  Sun,
  Moon,
  RotateCcw,
  Download,
  Settings,
  Eye,
  Grid,
  Maximize2
} from 'lucide-react';
import { api } from '../utils/api';

// ============================================================================
// 3D 组件
// ============================================================================

interface VesselMeshProps {
  meshUrl: string | null;
  color: string;
  opacity: number;
  wireframe: boolean;
}

const VesselMesh: React.FC<VesselMeshProps> = ({ meshUrl, color, opacity, wireframe }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    if (!meshUrl) return;

    // 加载网格数据
    const loadMesh = async () => {
      try {
        const response = await fetch(meshUrl);
        const data = await response.json();
        
        // 创建几何体
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices.flat(), 3));
        geo.setIndex(data.faces.flat());
        geo.computeVertexNormals();
        
        // 居中
        geo.center();
        
        setGeometry(geo);
      } catch (error) {
        console.error('加载网格失败:', error);
      }
    };

    loadMesh();
  }, [meshUrl]);

  // 自动旋转
  useFrame((state) => {
    if (meshRef.current && !state.controls) {
      meshRef.current.rotation.y += 0.002;
    }
  });

  if (!geometry) return null;

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color={color}
        transparent={opacity < 1}
        opacity={opacity}
        wireframe={wireframe}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};

// 坐标轴辅助线
const AxesHelper: React.FC<{ size: number }> = ({ size }) => {
  return <primitive object={new THREE.AxesHelper(size)} />;
};

// 网格辅助线
const GridHelper: React.FC<{ size: number; divisions: number }> = ({ size, divisions }) => {
  return <primitive object={new THREE.GridHelper(size, divisions, '#444', '#333')} />;
};

// 场景控制器
const SceneController: React.FC<{ onReset: () => void }> = ({ onReset }) => {
  const { camera, controls } = useThree();
  
  useEffect(() => {
    if (controls) {
      (controls as any).addEventListener('change', () => {
        // 可以在这里添加控制变化的处理
      });
    }
  }, [controls]);

  return null;
};

// ============================================================================
// 主页面组件
// ============================================================================

const VisualizationPage: React.FC = () => {
  // 数据状态
  const [volumes, setVolumes] = useState<any[]>([]);
  const [inferenceResults, setInferenceResults] = useState<any[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [sourceType, setSourceType] = useState<'volume' | 'inference'>('volume');
  
  // 网格状态
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  
  // 显示设置
  const [settings, setSettings] = useState({
    color: '#ff6b6b',
    opacity: 1.0,
    wireframe: false,
    showAxes: true,
    showGrid: true,
    autoRotate: true,
    backgroundColor: '#1a1a2e',
    threshold: 0.5
  });

  // 加载数据列表
  useEffect(() => {
    const loadData = async () => {
      try {
        const [volRes, infRes] = await Promise.all([
          api.get('/data/list'),
          api.get('/inference/list')
        ]);
        setVolumes(volRes.data);
        setInferenceResults(infRes.data.filter((r: any) => r.status === 'completed'));
      } catch (error) {
        console.error('加载数据失败:', error);
      }
    };
    loadData();
  }, []);

  // 生成网格
  const generateMesh = async () => {
    if (!selectedSource) return;
    
    setIsGenerating(true);
    
    try {
      let response;
      
      if (sourceType === 'volume') {
        response = await api.post(`/data/${selectedSource}/mesh`, {
          threshold: settings.threshold,
          smooth: true
        });
      } else {
        // 从推理结果生成
        response = await api.post(`/inference/result/${selectedSource}/mesh`, {
          threshold: settings.threshold,
          smooth: true
        });
      }
      
      // 获取网格数据 URL
      setMeshUrl(`${api.defaults.baseURL}/data/${selectedSource}/mesh-data`);
    } catch (error) {
      console.error('生成网格失败:', error);
      alert('生成网格失败');
    } finally {
      setIsGenerating(false);
    }
  };

  // 导出网格
  const exportMesh = async (format: string) => {
    if (!selectedSource) return;
    
    try {
      const response = await api.get(
        sourceType === 'volume'
          ? `/data/${selectedSource}/export/${format}`
          : `/inference/result/${selectedSource}/download/${format}`,
        { responseType: 'blob' }
      );
      
      const url = URL.createObjectURL(response.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vessel_mesh.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('导出失败:', error);
    }
  };

  // 重置视图
  const resetView = () => {
    // 触发 Canvas 重新渲染
    setMeshUrl(meshUrl);
  };

  return (
    <div className="flex h-full gap-4">
      {/* 左侧：控制面板 */}
      <div className="w-72 space-y-4 flex-shrink-0 overflow-y-auto">
        {/* 数据源选择 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Box size={18} />
            数据源
          </h3>
          
          <div className="space-y-3">
            <div className="flex gap-2">
              <button
                onClick={() => setSourceType('volume')}
                className={`flex-1 py-2 rounded-lg text-sm transition-colors ${
                  sourceType === 'volume' 
                    ? 'bg-blue-600' 
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                原始数据
              </button>
              <button
                onClick={() => setSourceType('inference')}
                className={`flex-1 py-2 rounded-lg text-sm transition-colors ${
                  sourceType === 'inference' 
                    ? 'bg-blue-600' 
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                推理结果
              </button>
            </div>
            
            <select
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
            >
              <option value="">选择数据...</option>
              {sourceType === 'volume' 
                ? volumes.map(vol => (
                    <option key={vol.id} value={vol.id}>{vol.filename}</option>
                  ))
                : inferenceResults.map(res => (
                    <option key={res.task_id} value={res.task_id}>{res.task_id}</option>
                  ))
              }
            </select>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">阈值</label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={settings.threshold}
                  onChange={(e) => setSettings({ ...settings, threshold: parseFloat(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm w-12 text-right">{settings.threshold}</span>
              </div>
            </div>
            
            <button
              onClick={generateMesh}
              disabled={!selectedSource || isGenerating}
              className={`w-full py-2 rounded-lg transition-colors ${
                isGenerating 
                  ? 'bg-gray-600 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isGenerating ? '生成中...' : '生成 3D 网格'}
            </button>
          </div>
        </div>

        {/* 显示设置 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Settings size={18} />
            显示设置
          </h3>
          
          <div className="space-y-4">
            {/* 颜色 */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">血管颜色</label>
              <div className="flex gap-2">
                {['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9'].map(color => (
                  <button
                    key={color}
                    onClick={() => setSettings({ ...settings, color })}
                    className={`w-8 h-8 rounded-lg transition-transform ${
                      settings.color === color ? 'ring-2 ring-white scale-110' : ''
                    }`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>
            
            {/* 透明度 */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">透明度</label>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.1}
                value={settings.opacity}
                onChange={(e) => setSettings({ ...settings, opacity: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>
            
            {/* 切换选项 */}
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.wireframe}
                  onChange={(e) => setSettings({ ...settings, wireframe: e.target.checked })}
                />
                <span className="text-sm">线框模式</span>
              </label>
              
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.showAxes}
                  onChange={(e) => setSettings({ ...settings, showAxes: e.target.checked })}
                />
                <span className="text-sm">显示坐标轴</span>
              </label>
              
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.showGrid}
                  onChange={(e) => setSettings({ ...settings, showGrid: e.target.checked })}
                />
                <span className="text-sm">显示网格</span>
              </label>
              
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.autoRotate}
                  onChange={(e) => setSettings({ ...settings, autoRotate: e.target.checked })}
                />
                <span className="text-sm">自动旋转</span>
              </label>
            </div>
            
            {/* 背景颜色 */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">背景</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setSettings({ ...settings, backgroundColor: '#1a1a2e' })}
                  className={`flex-1 py-2 rounded-lg text-sm flex items-center justify-center gap-1 ${
                    settings.backgroundColor === '#1a1a2e' 
                      ? 'bg-gray-600 ring-1 ring-white' 
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <Moon size={14} />
                  深色
                </button>
                <button
                  onClick={() => setSettings({ ...settings, backgroundColor: '#f5f5f5' })}
                  className={`flex-1 py-2 rounded-lg text-sm flex items-center justify-center gap-1 ${
                    settings.backgroundColor === '#f5f5f5' 
                      ? 'bg-gray-600 ring-1 ring-white' 
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <Sun size={14} />
                  浅色
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 导出 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Download size={18} />
            导出网格
          </h3>
          
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => exportMesh('stl')}
              disabled={!meshUrl}
              className="py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm"
            >
              STL
            </button>
            <button
              onClick={() => exportMesh('ply')}
              disabled={!meshUrl}
              className="py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm"
            >
              PLY
            </button>
            <button
              onClick={() => exportMesh('obj')}
              disabled={!meshUrl}
              className="py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm"
            >
              OBJ
            </button>
            <button
              onClick={() => exportMesh('nii')}
              disabled={!selectedSource}
              className="py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm"
            >
              NIfTI
            </button>
          </div>
        </div>

        {/* 视图控制 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex gap-2">
            <button
              onClick={resetView}
              className="flex-1 flex items-center justify-center gap-2 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
            >
              <RotateCcw size={14} />
              重置视图
            </button>
            <button
              className="flex-1 flex items-center justify-center gap-2 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
            >
              <Maximize2 size={14} />
              全屏
            </button>
          </div>
        </div>
      </div>

      {/* 右侧：3D 视图 */}
      <div className="flex-1 bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <Canvas
          style={{ background: settings.backgroundColor }}
          camera={{ position: [5, 5, 5], fov: 50 }}
        >
          <Suspense fallback={null}>
            {/* 光照 */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <directionalLight position={[-10, -10, -5]} intensity={0.3} />
            
            {/* 相机控制 */}
            <OrbitControls 
              enableDamping 
              dampingFactor={0.05}
              autoRotate={settings.autoRotate}
              autoRotateSpeed={1}
            />
            
            {/* 辅助元素 */}
            {settings.showAxes && <AxesHelper size={3} />}
            {settings.showGrid && <GridHelper size={10} divisions={10} />}
            
            {/* 血管网格 */}
            {meshUrl && (
              <VesselMesh
                meshUrl={meshUrl}
                color={settings.color}
                opacity={settings.opacity}
                wireframe={settings.wireframe}
              />
            )}
            
            {/* 环境 */}
            <Environment preset="studio" />
          </Suspense>
        </Canvas>
        
        {/* 操作提示 */}
        <div className="absolute bottom-4 left-4 text-xs text-gray-400 bg-black/30 px-3 py-2 rounded-lg">
          <div>左键拖拽: 旋转</div>
          <div>右键拖拽: 平移</div>
          <div>滚轮: 缩放</div>
        </div>
        
        {/* 无数据提示 */}
        {!meshUrl && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center text-gray-500">
              <Layers size={64} className="mx-auto mb-4" />
              <p>选择数据并生成 3D 网格</p>
              <p className="text-sm mt-2">支持原始 OCTA 数据或推理结果</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VisualizationPage;
