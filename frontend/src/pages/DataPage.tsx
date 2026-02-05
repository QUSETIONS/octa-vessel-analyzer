/**
 * OCTA 血管网络分析系统 - 数据管理页面
 * 
 * 功能：
 * 1. 上传 DICOM/OCTA 数据
 * 2. 查看已上传的数据列表
 * 3. 预览体数据切片
 * 4. 3D 可视化预览
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Upload, 
  Trash2, 
  Eye, 
  Download, 
  RefreshCw,
  FileText,
  Box,
  Layers
} from 'lucide-react';
import { api } from '../utils/api';

interface VolumeInfo {
  id: string;
  filename: string;
  shape: number[];
  spacing: number[];
  vessel_fraction: number;
  created_at: string;
}

const DataPage: React.FC = () => {
  const [volumes, setVolumes] = useState<VolumeInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedVolume, setSelectedVolume] = useState<VolumeInfo | null>(null);
  const [previewAxis, setPreviewAxis] = useState<'x' | 'y' | 'z'>('y');
  const [previewIndex, setPreviewIndex] = useState(0);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  // 加载数据列表
  const loadVolumes = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.get('/data/list');
      setVolumes(response.data);
    } catch (error) {
      console.error('加载数据列表失败:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVolumes();
  }, [loadVolumes]);

  // 上传文件
  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/data/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // 刷新列表
      await loadVolumes();
      
      // 选中新上传的数据
      setSelectedVolume(response.data);
    } catch (error) {
      console.error('上传失败:', error);
      alert('上传失败，请检查文件格式');
    } finally {
      setUploading(false);
    }
  };

  // 删除数据
  const handleDelete = async (volumeId: string) => {
    if (!confirm('确定要删除这个数据吗？')) return;

    try {
      await api.delete(`/data/${volumeId}`);
      await loadVolumes();
      
      if (selectedVolume?.id === volumeId) {
        setSelectedVolume(null);
      }
    } catch (error) {
      console.error('删除失败:', error);
    }
  };

  // 加载切片预览
  const loadPreview = useCallback(async () => {
    if (!selectedVolume) return;

    try {
      const response = await api.get(
        `/data/${selectedVolume.id}/slice/${previewAxis}/${previewIndex}`,
        { responseType: 'blob' }
      );
      
      const url = URL.createObjectURL(response.data);
      setPreviewImage(url);
    } catch (error) {
      console.error('加载预览失败:', error);
    }
  }, [selectedVolume, previewAxis, previewIndex]);

  useEffect(() => {
    loadPreview();
  }, [loadPreview]);

  // 获取当前轴的最大索引
  const getMaxIndex = () => {
    if (!selectedVolume) return 0;
    const axisMap = { x: 0, y: 1, z: 2 };
    return selectedVolume.shape[axisMap[previewAxis]] - 1;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      {/* 左侧：数据列表 */}
      <div className="lg:col-span-1 space-y-4">
        {/* 上传按钮 */}
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <label className="block">
            <div className={`
              flex flex-col items-center justify-center p-6 border-2 border-dashed border-gray-600 
              rounded-lg cursor-pointer hover:border-blue-500 hover:bg-gray-700/50 transition-all
              ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
            `}>
              <Upload size={32} className="text-gray-400 mb-2" />
              <span className="text-gray-300 font-medium">
                {uploading ? '上传中...' : '点击或拖放文件上传'}
              </span>
              <span className="text-gray-500 text-sm mt-1">
                支持 DICOM, NIfTI, NumPy, MATLAB 格式
              </span>
            </div>
            <input
              type="file"
              className="hidden"
              onChange={handleUpload}
              disabled={uploading}
              accept=".dcm,.dicom,.nii,.nii.gz,.npy,.mat,.zip"
            />
          </label>
        </div>

        {/* 数据列表 */}
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
            <h3 className="font-semibold">已上传数据</h3>
            <button 
              onClick={loadVolumes}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
          
          <div className="max-h-96 overflow-y-auto">
            {loading ? (
              <div className="p-8 text-center text-gray-500">加载中...</div>
            ) : volumes.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                暂无数据，请上传
              </div>
            ) : (
              <div className="divide-y divide-gray-700">
                {volumes.map((vol) => (
                  <div 
                    key={vol.id}
                    className={`
                      p-4 cursor-pointer transition-colors
                      ${selectedVolume?.id === vol.id ? 'bg-blue-600/20' : 'hover:bg-gray-700/50'}
                    `}
                    onClick={() => {
                      setSelectedVolume(vol);
                      setPreviewIndex(Math.floor(vol.shape[1] / 2));
                    }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <Box className="text-blue-400" size={20} />
                        <div>
                          <div className="font-medium truncate max-w-[180px]">
                            {vol.filename}
                          </div>
                          <div className="text-sm text-gray-400">
                            {vol.shape.join(' × ')}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(vol.id);
                        }}
                        className="p-1 hover:bg-red-500/20 rounded text-gray-400 hover:text-red-400"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 右侧：预览和详情 */}
      <div className="lg:col-span-2 space-y-4">
        {selectedVolume ? (
          <>
            {/* 数据详情 */}
            <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
              <h3 className="font-semibold mb-4">数据详情</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-gray-400 text-sm">文件名</div>
                  <div className="font-medium truncate">{selectedVolume.filename}</div>
                </div>
                <div>
                  <div className="text-gray-400 text-sm">尺寸</div>
                  <div className="font-medium">{selectedVolume.shape.join(' × ')}</div>
                </div>
                <div>
                  <div className="text-gray-400 text-sm">体素间距</div>
                  <div className="font-medium">
                    {selectedVolume.spacing.map(s => s.toFixed(2)).join(' × ')} μm
                  </div>
                </div>
                <div>
                  <div className="text-gray-400 text-sm">血管占比</div>
                  <div className="font-medium">
                    {(selectedVolume.vessel_fraction * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>

            {/* 切片预览 */}
            <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">切片预览</h3>
                <div className="flex items-center gap-2">
                  {(['x', 'y', 'z'] as const).map((axis) => (
                    <button
                      key={axis}
                      onClick={() => {
                        setPreviewAxis(axis);
                        const axisMap = { x: 0, y: 1, z: 2 };
                        setPreviewIndex(Math.floor(selectedVolume.shape[axisMap[axis]] / 2));
                      }}
                      className={`
                        px-3 py-1 rounded-lg text-sm font-medium transition-colors
                        ${previewAxis === axis 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }
                      `}
                    >
                      {axis.toUpperCase()} 轴
                    </button>
                  ))}
                </div>
              </div>

              {/* 切片索引滑块 */}
              <div className="mb-4">
                <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
                  <span>切片索引</span>
                  <span>{previewIndex} / {getMaxIndex()}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={getMaxIndex()}
                  value={previewIndex}
                  onChange={(e) => setPreviewIndex(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* 预览图像 */}
              <div className="aspect-square bg-black rounded-lg overflow-hidden flex items-center justify-center">
                {previewImage ? (
                  <img 
                    src={previewImage} 
                    alt="切片预览" 
                    className="max-w-full max-h-full object-contain"
                  />
                ) : (
                  <div className="text-gray-500">加载中...</div>
                )}
              </div>
            </div>

            {/* 操作按钮 */}
            <div className="flex gap-3">
              <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
                <Eye size={18} />
                3D 预览
              </button>
              <button className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors">
                <Download size={18} />
                导出
              </button>
              <button className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors">
                <Layers size={18} />
                生成网格
              </button>
            </div>
          </>
        ) : (
          <div className="bg-gray-800 rounded-xl p-8 border border-gray-700 text-center">
            <FileText size={48} className="mx-auto text-gray-600 mb-4" />
            <h3 className="text-lg font-medium text-gray-400">选择一个数据查看详情</h3>
            <p className="text-gray-500 mt-2">
              从左侧列表选择已上传的数据，或上传新的数据文件
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataPage;
