/**
 * OCTA 血管网络分析系统 - API 工具
 * 
 * 封装与后端通信的 axios 实例和常用方法
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// API 基础配置
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// 创建 axios 实例
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 5 分钟超时（用于大文件上传和推理）
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证 token 等
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    if (error.response) {
      // 服务器返回错误
      console.error(`[API Error] ${error.response.status}:`, error.response.data);
      
      if (error.response.status === 500) {
        console.error('服务器内部错误');
      }
    } else if (error.request) {
      // 请求发送失败
      console.error('[API Error] 网络错误，无法连接到服务器');
    } else {
      console.error('[API Error]', error.message);
    }
    
    return Promise.reject(error);
  }
);

// ============================================================================
// API 方法封装
// ============================================================================

// 数据管理
export const dataApi = {
  // 上传文件
  upload: async (file: File, onProgress?: (percent: number) => void) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return api.post('/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event) => {
        if (event.total && onProgress) {
          onProgress(Math.round((event.loaded * 100) / event.total));
        }
      },
    });
  },
  
  // 获取数据列表
  list: () => api.get('/data/list'),
  
  // 获取数据详情
  get: (volumeId: string) => api.get(`/data/${volumeId}`),
  
  // 删除数据
  delete: (volumeId: string) => api.delete(`/data/${volumeId}`),
  
  // 获取切片
  getSlice: (volumeId: string, axis: string, index: number, applyCorrection = false) => 
    api.get(`/data/${volumeId}/slice/${axis}/${index}`, {
      responseType: 'blob',
      params: { apply_correction: applyCorrection },
    }),
  
  // 生成网格
  generateMesh: (volumeId: string, threshold = 0.5, smooth = true) =>
    api.post(`/data/${volumeId}/mesh`, { threshold, smooth }),
  
  // 导出数据
  export: (volumeId: string, format: string) =>
    api.get(`/data/${volumeId}/export/${format}`, { responseType: 'blob' }),
};

// 标注管理
export const annotationApi = {
  // 创建项目
  createProject: (volumeId: string, name: string) =>
    api.post('/annotation/projects', { volume_id: volumeId, name }),
  
  // 获取项目列表
  listProjects: () => api.get('/annotation/projects'),
  
  // 获取项目详情
  getProject: (projectId: string) => api.get(`/annotation/projects/${projectId}`),
  
  // 删除项目
  deleteProject: (projectId: string) => api.delete(`/annotation/projects/${projectId}`),
  
  // 保存标注
  saveAnnotation: (projectId: string, layerKey: string, ellipses: any[], brushStrokes: any[]) =>
    api.post(`/annotation/projects/${projectId}/save`, {
      project_id: projectId,
      layer_key: layerKey,
      ellipses,
      brush_strokes: brushStrokes,
    }),
  
  // 获取标注层
  getLayer: (projectId: string, layerKey: string) =>
    api.get(`/annotation/projects/${projectId}/layer/${layerKey}`),
  
  // 撤销
  undo: (projectId: string) => api.post(`/annotation/projects/${projectId}/undo`),
  
  // 重做
  redo: (projectId: string) => api.post(`/annotation/projects/${projectId}/redo`),
  
  // 生成金标准
  generateMask: (projectId: string) =>
    api.post(`/annotation/projects/${projectId}/generate-mask`),
  
  // 椭圆拟合
  fitEllipse: (points: { x: number; y: number }[]) =>
    api.post('/annotation/fit-ellipse', points),
};

// 训练管理
export const trainingApi = {
  // 开始训练
  start: (config: any) => api.post('/training/start', config),
  
  // 停止训练
  stop: () => api.post('/training/stop'),
  
  // 暂停训练
  pause: () => api.post('/training/pause'),
  
  // 恢复训练
  resume: () => api.post('/training/resume'),
  
  // 获取状态
  getStatus: () => api.get('/training/status'),
  
  // 获取日志
  getLogs: (limit = 100) => api.get('/training/logs', { params: { limit } }),
  
  // 获取检查点列表
  listCheckpoints: () => api.get('/training/checkpoints'),
  
  // 删除检查点
  deleteCheckpoint: (filename: string) => api.delete(`/training/checkpoints/${filename}`),
  
  // 加载检查点
  loadCheckpoint: (filename: string) => api.post(`/training/checkpoints/load/${filename}`),
  
  // 验证数据
  validateData: (volumeId: string, projectId: string) =>
    api.post('/training/validate-data', null, {
      params: { volume_id: volumeId, annotation_project_id: projectId },
    }),
  
  // 获取推荐配置
  getRecommendedConfig: (volumeId: string) =>
    api.get('/training/recommended-config', { params: { volume_id: volumeId } }),
};

// 推理管理
export const inferenceApi = {
  // 开始推理
  start: (config: any) => api.post('/inference/start', config),
  
  // 获取任务状态
  getStatus: (taskId: string) => api.get(`/inference/status/${taskId}`),
  
  // 获取任务列表
  list: () => api.get('/inference/list'),
  
  // 获取结果切片
  getResultSlice: (taskId: string, axis: string, index: number) =>
    api.get(`/inference/result/${taskId}/slice/${axis}/${index}`, { responseType: 'blob' }),
  
  // 下载结果
  downloadResult: (taskId: string, format: string) =>
    api.get(`/inference/result/${taskId}/download/${format}`, { responseType: 'blob' }),
  
  // 删除结果
  deleteResult: (taskId: string) => api.delete(`/inference/result/${taskId}`),
};

// 系统 API
export const systemApi = {
  // 健康检查
  health: () => api.get('/health'),
  
  // 获取配置
  getConfig: () => api.get('/config'),
};

export default api;
