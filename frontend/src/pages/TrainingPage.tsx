/**
 * OCTA 血管网络分析系统 - 模型训练页面
 * 
 * 功能：
 * 1. 配置训练参数
 * 2. 启动/停止训练
 * 3. 实时显示训练进度
 * 4. 显示 Loss 曲线
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  RefreshCw,
  Settings,
  Activity,
  TrendingDown,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../utils/api';

interface TrainingState {
  status: string;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  train_loss: number;
  val_loss: number;
  best_loss: number;
  train_losses: number[];
  val_losses: number[];
  dice_scores: number[];
  elapsed_time: number;
  eta: number;
  message: string;
}

const TrainingPage: React.FC = () => {
  // 数据状态
  const [volumes, setVolumes] = useState<any[]>([]);
  const [projects, setProjects] = useState<any[]>([]);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);
  
  // 配置
  const [config, setConfig] = useState({
    volume_id: '',
    annotation_project_id: '',
    mode: 'segmentation',
    epochs: 100,
    batch_size: 4,
    learning_rate: 0.0001,
    patch_size: 32,
    diffusion_timesteps: 1000,
    diffusion_beta_schedule: 'cosine'
  });
  
  // 训练状态
  const [trainingState, setTrainingState] = useState<TrainingState | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  // 加载数据
  useEffect(() => {
    const loadData = async () => {
      try {
        const [volRes, projRes, ckptRes] = await Promise.all([
          api.get('/data/list'),
          api.get('/annotation/projects'),
          api.get('/training/checkpoints')
        ]);
        setVolumes(volRes.data);
        setProjects(projRes.data);
        setCheckpoints(ckptRes.data);
      } catch (error) {
        console.error('加载数据失败:', error);
      }
    };
    loadData();
  }, []);

  // 轮询训练状态
  const pollStatus = useCallback(async () => {
    try {
      const response = await api.get('/training/status');
      setTrainingState(response.data);
      
      // 如果训练正在进行，继续轮询
      if (response.data.status === 'running') {
        setIsPolling(true);
      } else {
        setIsPolling(false);
      }
    } catch (error) {
      console.error('获取状态失败:', error);
    }
  }, []);

  useEffect(() => {
    pollStatus();
  }, [pollStatus]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPolling) {
      interval = setInterval(pollStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [isPolling, pollStatus]);

  // 启动训练
  const startTraining = async () => {
    if (!config.volume_id || !config.annotation_project_id) {
      alert('请选择训练数据和标注项目');
      return;
    }
    
    try {
      await api.post('/training/start', config);
      setIsPolling(true);
      pollStatus();
    } catch (error: any) {
      alert(error.response?.data?.detail || '启动训练失败');
    }
  };

  // 暂停训练
  const pauseTraining = async () => {
    try {
      await api.post('/training/pause');
      pollStatus();
    } catch (error) {
      console.error('暂停失败:', error);
    }
  };

  // 恢复训练
  const resumeTraining = async () => {
    try {
      await api.post('/training/resume');
      setIsPolling(true);
    } catch (error) {
      console.error('恢复失败:', error);
    }
  };

  // 停止训练
  const stopTraining = async () => {
    if (!confirm('确定要停止训练吗？')) return;
    
    try {
      await api.post('/training/stop');
      setIsPolling(false);
      pollStatus();
    } catch (error) {
      console.error('停止失败:', error);
    }
  };

  // 格式化时间
  const formatTime = (seconds: number): string => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // 准备图表数据
  const chartData = trainingState?.train_losses.map((loss, idx) => ({
    epoch: idx + 1,
    train_loss: loss,
    val_loss: trainingState.val_losses[idx] || null,
    dice: trainingState.dice_scores[idx] || null
  })) || [];

  // 状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'completed': return 'text-blue-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Activity className="animate-pulse" />;
      case 'paused': return <Pause />;
      case 'completed': return <CheckCircle />;
      case 'error': return <AlertCircle />;
      default: return <Clock />;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* 左侧：配置面板 */}
      <div className="lg:col-span-1 space-y-4">
        {/* 数据选择 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Settings size={18} />
            训练配置
          </h3>
          
          <div className="space-y-4">
            {/* 数据选择 */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">训练数据</label>
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
            
            {/* 标注项目 */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">标注项目</label>
              <select
                value={config.annotation_project_id}
                onChange={(e) => setConfig({ ...config, annotation_project_id: e.target.value })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              >
                <option value="">选择项目...</option>
                {projects.map(proj => (
                  <option key={proj.id} value={proj.id}>{proj.name}</option>
                ))}
              </select>
            </div>
            
            {/* 训练模式 */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">训练模式</label>
              <select
                value={config.mode}
                onChange={(e) => setConfig({ ...config, mode: e.target.value })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              >
                <option value="segmentation">仅分割网络 (3D U-Net)</option>
                <option value="diffusion">仅扩散网络 (Diffusion)</option>
                <option value="combined">级联训练 (分割 + 扩散)</option>
              </select>
            </div>
          </div>
        </div>

        {/* 超参数 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4">超参数</h3>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">学习率</label>
              <input
                type="number"
                step="0.0001"
                value={config.learning_rate}
                onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-400 mb-1">Patch Size</label>
              <select
                value={config.patch_size}
                onChange={(e) => setConfig({ ...config, patch_size: parseInt(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
              >
                <option value={32}>32×32×32</option>
                <option value={48}>48×48×48</option>
                <option value={64}>64×64×64</option>
              </select>
            </div>
            
            {config.mode !== 'segmentation' && (
              <>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Diffusion 步数</label>
                  <input
                    type="number"
                    value={config.diffusion_timesteps}
                    onChange={(e) => setConfig({ ...config, diffusion_timesteps: parseInt(e.target.value) })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-400 mb-1">噪声调度</label>
                  <select
                    value={config.diffusion_beta_schedule}
                    onChange={(e) => setConfig({ ...config, diffusion_beta_schedule: e.target.value })}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2"
                  >
                    <option value="cosine">余弦 (推荐)</option>
                    <option value="linear">线性</option>
                    <option value="quadratic">二次</option>
                  </select>
                </div>
              </>
            )}
          </div>
        </div>

        {/* 控制按钮 */}
        <div className="flex gap-2">
          {trainingState?.status === 'running' ? (
            <>
              <button
                onClick={pauseTraining}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
              >
                <Pause size={18} />
                暂停
              </button>
              <button
                onClick={stopTraining}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
              >
                <Square size={18} />
                停止
              </button>
            </>
          ) : trainingState?.status === 'paused' ? (
            <>
              <button
                onClick={resumeTraining}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
              >
                <Play size={18} />
                继续
              </button>
              <button
                onClick={stopTraining}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
              >
                <Square size={18} />
                停止
              </button>
            </>
          ) : (
            <button
              onClick={startTraining}
              className="flex-1 flex items-center justify-center gap-2 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              <Play size={18} />
              开始训练
            </button>
          )}
        </div>
      </div>

      {/* 右侧：训练状态和图表 */}
      <div className="lg:col-span-2 space-y-4">
        {/* 状态概览 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">训练状态</h3>
            <div className={`flex items-center gap-2 ${getStatusColor(trainingState?.status || 'idle')}`}>
              {getStatusIcon(trainingState?.status || 'idle')}
              <span className="capitalize">{trainingState?.status || 'idle'}</span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="text-sm text-gray-400">Epoch</div>
              <div className="text-xl font-bold">
                {trainingState?.current_epoch || 0} / {trainingState?.total_epochs || config.epochs}
              </div>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="text-sm text-gray-400">训练 Loss</div>
              <div className="text-xl font-bold">
                {trainingState?.train_loss?.toFixed(4) || '-'}
              </div>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="text-sm text-gray-400">验证 Loss</div>
              <div className="text-xl font-bold">
                {trainingState?.val_loss?.toFixed(4) || '-'}
              </div>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-3">
              <div className="text-sm text-gray-400">最佳 Loss</div>
              <div className="text-xl font-bold text-green-500">
                {trainingState?.best_loss < Infinity ? trainingState.best_loss.toFixed(4) : '-'}
              </div>
            </div>
          </div>
          
          {/* 进度条 */}
          {trainingState?.status === 'running' && (
            <div className="mt-4">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>训练进度</span>
                <span>{((trainingState.current_epoch / trainingState.total_epochs) * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-600 transition-all"
                  style={{ width: `${(trainingState.current_epoch / trainingState.total_epochs) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-sm text-gray-400 mt-2">
                <span>已用时间: {formatTime(trainingState.elapsed_time)}</span>
                <span>预计剩余: {formatTime(trainingState.eta)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Loss 曲线 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <TrendingDown size={18} />
            Loss 曲线
          </h3>
          
          {chartData.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }} 
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="train_loss" 
                    stroke="#3B82F6" 
                    name="训练 Loss"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="val_loss" 
                    stroke="#10B981" 
                    name="验证 Loss"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-500">
              开始训练后将显示 Loss 曲线
            </div>
          )}
        </div>

        {/* 模型检查点 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">模型检查点</h3>
            <button 
              onClick={async () => {
                const res = await api.get('/training/checkpoints');
                setCheckpoints(res.data);
              }}
              className="p-2 hover:bg-gray-700 rounded-lg"
            >
              <RefreshCw size={16} />
            </button>
          </div>
          
          {checkpoints.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {checkpoints.map(ckpt => (
                <div 
                  key={ckpt.name}
                  className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
                >
                  <div>
                    <div className="font-medium">{ckpt.name}</div>
                    <div className="text-sm text-gray-400">
                      {ckpt.size_mb.toFixed(2)} MB
                    </div>
                  </div>
                  <div className="text-sm text-gray-400">
                    {new Date(ckpt.modified_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              暂无检查点
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingPage;
