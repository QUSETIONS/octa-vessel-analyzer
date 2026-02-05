/**
 * OCTA 血管网络分析系统 - 推理预测页面
 * 
 * 功能：
 * 1. 选择数据和模型
 * 2. 执行推理
 * 3. 对比显示原始/处理后结果
 * 4. 导出结果
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  Download, 
  RefreshCw,
  Layers,
  Eye,
  SplitSquareHorizontal,
  ChevronLeft,
  ChevronRight,
  Check,
  Loader2
} from 'lucide-react';
import { api } from '../utils/api';

interface InferenceTask {
  id: string;
  volume_id: string;
  checkpoint_name: string;
  mode: string;
  status: string;
  created_at: string;
  completed_at?: string;
  metrics?: {
    vessel_voxels: number;
    vessel_fraction: number;
  };
}

const InferencePage: React.FC = () => {
  // 数据状态
  const [volumes, setVolumes] = useState<any[]>([]);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  const [tasks, setTasks] = useState<InferenceTask[]>([]);
  
  // 配置
  const [config, setConfig] = useState({
    volume_id: '',
    checkpoint_name: '',
    mode: 'segmentation',
    patch_size: 32,
    overlap: 8,
    threshold: 0.5,
    use_ddim: true,
    ddim_steps: 50
  });
  
  // 当前任务
  const [currentTask, setCurrentTask] = useState<InferenceTask | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  
  // 对比视图
  const [compareMode, setCompareMode] = useState<'slider' | 'side-by-side'>('slider');
  const [sliderPosition, setSliderPosition] = useState(50);
  const [viewAxis, setViewAxis] = useState<'x' | 'y' | 'z'>('y');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [maxSliceIndex, setMaxSliceIndex] = useState(100);
  
  const [originalImage, setOriginalImage] = useState<string>('');
  const [resultImage, setResultImage] = useState<string>('');

  // 加载数据
  useEffect(() => {
    const loadData = async () => {
      try {
        const [volRes, ckptRes, taskRes] = await Promise.all([
          api.get('/data/list'),
          api.get('/training/checkpoints'),
          api.get('/inference/list')
        ]);
        setVolumes(volRes.data);
        setCheckpoints(ckptRes.data);
        setTasks(taskRes.data);
      } catch (error) {
        console.error('加载数据失败:', error);
      }
    };
    loadData();
  }, []);

  // 选择数据后更新切片范围
  useEffect(() => {
    if (!config.volume_id) return;
    
    const volume = volumes.find(v => v.id === config.volume_id);
    if (volume) {
      const axisMap = { x: 0, y: 1, z: 2 };
      const max = volume.shape[axisMap[viewAxis]] - 1;
      setMaxSliceIndex(max);
      setSliceIndex(Math.floor(max / 2));
    }
  }, [config.volume_id, viewAxis, volumes]);

  // 加载对比图像
  const loadCompareImages = useCallback(async () => {
    if (!config.volume_id || !currentTask?.id) return;
    
    try {
      // 加载原始图像
      const origRes = await api.get(
        `/data/${config.volume_id}/slice/${viewAxis}/${sliceIndex}`,
        { responseType: 'blob' }
      );
      setOriginalImage(URL.createObjectURL(origRes.data));
      
      // 加载结果图像
      const resultRes = await api.get(
        `/inference/result/${currentTask.id}/slice/${viewAxis}/${sliceIndex}`,
        { responseType: 'blob' }
      );
      setResultImage(URL.createObjectURL(resultRes.data));
    } catch (error) {
      console.error('加载图像失败:', error);
    }
  }, [config.volume_id, currentTask, viewAxis, sliceIndex]);

  useEffect(() => {
    loadCompareImages();
  }, [loadCompareImages]);

  // 开始推理
  const startInference = async () => {
    if (!config.volume_id || !config.checkpoint_name) {
      alert('请选择数据和模型');
      return;
    }
    
    setIsRunning(true);
    
    try {
      const response = await api.post('/inference/start', config);
      setCurrentTask(response.data);
      
      // 轮询状态
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await api.get(`/inference/status/${response.data.id}`);
          setCurrentTask(statusRes.data);
          
          if (statusRes.data.status === 'completed' || statusRes.data.status === 'error') {
            clearInterval(pollInterval);
            setIsRunning(false);
            
            // 刷新任务列表
            const taskRes = await api.get('/inference/list');
            setTasks(taskRes.data);
          }
        } catch (error) {
          console.error('获取状态失败:', error);
        }
      }, 2000);
      
    } catch (error: any) {
      alert(error.response?.data?.detail || '启动推理失败');
      setIsRunning(false);
    }
  };

  // 导出结果
  const exportResult = async (format: string) => {
    if (!currentTask?.id) return;
    
    try {
      const response = await api.get(
        `/inference/result/${currentTask.id}/download/${format}`,
        { responseType: 'blob' }
      );
      
      const url = URL.createObjectURL(response.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = `result_${currentTask.id}.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('导出失败:', error);
      alert('导出失败');
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full">
      {/* 左侧：配置面板 */}
      <div className="lg:col-span-1 space-y-4">
        {/* 数据选择 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4">推理配置</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">输入数据</label>
              <select
                value={config.volume_id}
                onChange={(e) => setConfig({ ...config, volume_id: e.target.value })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              >
                <option value="">选择数据...</option>
                {volumes.map(vol => (
                  <option key={vol.id} value={vol.id}>{vol.filename}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">模型检查点</label>
              <select
                value={config.checkpoint_name}
                onChange={(e) => setConfig({ ...config, checkpoint_name: e.target.value })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              >
                <option value="">选择模型...</option>
                {checkpoints.map(ckpt => (
                  <option key={ckpt.name} value={ckpt.name}>{ckpt.name}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">推理模式</label>
              <select
                value={config.mode}
                onChange={(e) => setConfig({ ...config, mode: e.target.value })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              >
                <option value="segmentation">仅分割网络</option>
                <option value="diffusion">仅扩散网络</option>
                <option value="combined">级联推理</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">阈值</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={config.threshold}
                onChange={(e) => setConfig({ ...config, threshold: parseFloat(e.target.value) })}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-400">{config.threshold}</div>
            </div>
            
            {config.mode !== 'segmentation' && (
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.use_ddim}
                    onChange={(e) => setConfig({ ...config, use_ddim: e.target.checked })}
                  />
                  <span className="text-sm">使用 DDIM 加速采样</span>
                </label>
                
                {config.use_ddim && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">DDIM 步数</label>
                    <input
                      type="number"
                      value={config.ddim_steps}
                      onChange={(e) => setConfig({ ...config, ddim_steps: parseInt(e.target.value) })}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* 开始推理按钮 */}
        <button
          onClick={startInference}
          disabled={isRunning || !config.volume_id || !config.checkpoint_name}
          className={`
            w-full flex items-center justify-center gap-2 py-3 rounded-lg transition-colors
            ${isRunning 
              ? 'bg-gray-600 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700'
            }
          `}
        >
          {isRunning ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              推理中...
            </>
          ) : (
            <>
              <Play size={18} />
              开始推理
            </>
          )}
        </button>

        {/* 任务状态 */}
        {currentTask && (
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h3 className="font-semibold mb-3">当前任务</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">状态</span>
                <span className={
                  currentTask.status === 'completed' ? 'text-green-500' :
                  currentTask.status === 'running' ? 'text-blue-500' :
                  currentTask.status === 'error' ? 'text-red-500' : 'text-gray-400'
                }>
                  {currentTask.status}
                </span>
              </div>
              {currentTask.metrics && (
                <>
                  <div className="flex justify-between">
                    <span className="text-gray-400">血管体素</span>
                    <span>{currentTask.metrics.vessel_voxels.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">占比</span>
                    <span>{(currentTask.metrics.vessel_fraction * 100).toFixed(2)}%</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* 导出按钮 */}
        {currentTask?.status === 'completed' && (
          <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
            <h3 className="font-semibold mb-3">导出结果</h3>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => exportResult('npy')}
                className="flex items-center justify-center gap-1 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
              >
                <Download size={14} />
                NumPy
              </button>
              <button
                onClick={() => exportResult('nii')}
                className="flex items-center justify-center gap-1 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
              >
                <Download size={14} />
                NIfTI
              </button>
              <button
                onClick={() => exportResult('stl')}
                className="flex items-center justify-center gap-1 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
              >
                <Layers size={14} />
                STL
              </button>
              <button
                onClick={() => exportResult('ply')}
                className="flex items-center justify-center gap-1 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
              >
                <Layers size={14} />
                PLY
              </button>
            </div>
          </div>
        )}

        {/* 历史任务 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">历史任务</h3>
            <button
              onClick={async () => {
                const res = await api.get('/inference/list');
                setTasks(res.data);
              }}
              className="p-1 hover:bg-gray-700 rounded"
            >
              <RefreshCw size={14} />
            </button>
          </div>
          
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {tasks.length > 0 ? (
              tasks.map(task => (
                <div
                  key={task.id}
                  onClick={() => {
                    setCurrentTask(task);
                    setConfig({ ...config, volume_id: task.volume_id });
                  }}
                  className={`
                    p-2 rounded-lg cursor-pointer transition-colors
                    ${currentTask?.id === task.id 
                      ? 'bg-blue-600/20 border border-blue-600' 
                      : 'bg-gray-700/50 hover:bg-gray-700'
                    }
                  `}
                >
                  <div className="text-sm font-medium truncate">{task.id}</div>
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <span className={
                      task.status === 'completed' ? 'text-green-500' : 'text-gray-500'
                    }>
                      {task.status}
                    </span>
                    <span>{task.mode}</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-gray-500 py-4 text-sm">
                暂无历史任务
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 右侧：对比视图 */}
      <div className="lg:col-span-3 space-y-4">
        {/* 视图控制 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center justify-between flex-wrap gap-4">
            {/* 对比模式 */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">对比模式:</span>
              <button
                onClick={() => setCompareMode('slider')}
                className={`p-2 rounded-lg transition-colors ${
                  compareMode === 'slider' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                }`}
                title="滑动对比"
              >
                <SplitSquareHorizontal size={18} />
              </button>
              <button
                onClick={() => setCompareMode('side-by-side')}
                className={`p-2 rounded-lg transition-colors ${
                  compareMode === 'side-by-side' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                }`}
                title="并排对比"
              >
                <Eye size={18} />
              </button>
            </div>
            
            {/* 轴向选择 */}
            <div className="flex items-center gap-2">
              {(['x', 'y', 'z'] as const).map(axis => (
                <button
                  key={axis}
                  onClick={() => setViewAxis(axis)}
                  className={`
                    px-3 py-1 rounded-lg text-sm font-medium transition-colors
                    ${viewAxis === axis
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }
                  `}
                >
                  {axis.toUpperCase()} 轴
                </button>
              ))}
            </div>
            
            {/* 切片导航 */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSliceIndex(Math.max(0, sliceIndex - 1))}
                className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600"
              >
                <ChevronLeft size={18} />
              </button>
              <span className="text-sm min-w-[80px] text-center">
                {sliceIndex} / {maxSliceIndex}
              </span>
              <button
                onClick={() => setSliceIndex(Math.min(maxSliceIndex, sliceIndex + 1))}
                className="p-2 bg-gray-700 rounded-lg hover:bg-gray-600"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </div>
          
          {/* 切片滑块 */}
          <div className="mt-4">
            <input
              type="range"
              min={0}
              max={maxSliceIndex}
              value={sliceIndex}
              onChange={(e) => setSliceIndex(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* 对比显示区域 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 flex-1">
          {currentTask?.status === 'completed' ? (
            compareMode === 'slider' ? (
              // 滑动对比模式
              <div className="relative aspect-square max-h-[600px] mx-auto overflow-hidden">
                {/* 原始图像（底层） */}
                <div className="absolute inset-0">
                  <img 
                    src={originalImage} 
                    alt="原始" 
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-xs">
                    原始
                  </div>
                </div>
                
                {/* 结果图像（通过 clip 显示） */}
                <div 
                  className="absolute inset-0"
                  style={{ 
                    clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` 
                  }}
                >
                  <img 
                    src={resultImage} 
                    alt="结果" 
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-2 left-2 bg-green-600/50 px-2 py-1 rounded text-xs">
                    处理后
                  </div>
                </div>
                
                {/* 滑块 */}
                <div
                  className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize"
                  style={{ left: `${sliderPosition}%` }}
                  onMouseDown={(e) => {
                    const container = e.currentTarget.parentElement;
                    if (!container) return;
                    
                    const handleMove = (moveEvent: MouseEvent) => {
                      const rect = container.getBoundingClientRect();
                      const x = moveEvent.clientX - rect.left;
                      const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
                      setSliderPosition(percent);
                    };
                    
                    const handleUp = () => {
                      document.removeEventListener('mousemove', handleMove);
                      document.removeEventListener('mouseup', handleUp);
                    };
                    
                    document.addEventListener('mousemove', handleMove);
                    document.addEventListener('mouseup', handleUp);
                  }}
                >
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 bg-white rounded-full shadow-lg flex items-center justify-center">
                    <SplitSquareHorizontal size={14} className="text-gray-800" />
                  </div>
                </div>
              </div>
            ) : (
              // 并排对比模式
              <div className="grid grid-cols-2 gap-4">
                <div className="relative aspect-square bg-black rounded-lg overflow-hidden">
                  <img 
                    src={originalImage} 
                    alt="原始" 
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-2 left-2 bg-black/50 px-2 py-1 rounded text-xs">
                    原始
                  </div>
                </div>
                <div className="relative aspect-square bg-black rounded-lg overflow-hidden">
                  <img 
                    src={resultImage} 
                    alt="结果" 
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute top-2 left-2 bg-green-600/50 px-2 py-1 rounded text-xs">
                    处理后
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="h-96 flex flex-col items-center justify-center text-gray-500">
              {isRunning ? (
                <>
                  <Loader2 size={48} className="animate-spin mb-4" />
                  <p>正在处理中，请稍候...</p>
                </>
              ) : (
                <>
                  <Eye size={48} className="mb-4" />
                  <p>选择数据和模型后点击开始推理</p>
                  <p className="text-sm mt-2">推理完成后可在此查看对比结果</p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default InferencePage;
