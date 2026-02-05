/**
 * OCTA 血管网络分析系统 - 首页
 * 
 * 显示系统概览和快速入口
 */

import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  Database, 
  Edit3, 
  Cpu, 
  Play, 
  Eye,
  ArrowRight,
  FolderOpen,
  Activity
} from 'lucide-react';
import { api } from '../utils/api';

interface SystemStatus {
  status: string;
  data_dir_exists: boolean;
  uploads_count: number;
  models_count: number;
}

const HomePage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await api.get('/health');
        setSystemStatus(response.data);
      } catch (error) {
        console.error('获取系统状态失败:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, []);

  const features = [
    {
      title: '数据管理',
      description: '导入 DICOM/OCTA 数据，构建 3D 血管网络',
      icon: Database,
      path: '/data',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      title: '标注工具',
      description: '血管横截面手动标注，生成金标准数据',
      icon: Edit3,
      path: '/annotation',
      color: 'from-purple-500 to-pink-500'
    },
    {
      title: '模型训练',
      description: '3D U-Net + Diffusion 深度学习训练',
      icon: Cpu,
      path: '/training',
      color: 'from-orange-500 to-red-500'
    },
    {
      title: '推理预测',
      description: '使用训练好的模型消除拖尾伪影',
      icon: Play,
      path: '/inference',
      color: 'from-green-500 to-teal-500'
    },
    {
      title: '三维可视化',
      description: '交互式血管网络 3D 展示和导出',
      icon: Eye,
      path: '/visualization',
      color: 'from-indigo-500 to-purple-500'
    }
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* 欢迎区域 */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-4">
          欢迎使用 OCTA 血管网络分析系统
        </h1>
        <p className="text-lg text-blue-100 mb-6 max-w-2xl">
          本系统提供完整的 OCTA 数据处理流程，包括数据导入、标注、深度学习训练和三维可视化。
          通过先进的 3D U-Net 和 Diffusion 模型，自动消除血管拖尾伪影，生成高质量的三维血管网络。
        </p>
        <Link 
          to="/data"
          className="inline-flex items-center gap-2 bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors"
        >
          <FolderOpen size={20} />
          开始导入数据
          <ArrowRight size={20} />
        </Link>
      </div>

      {/* 系统状态 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center gap-3 mb-2">
            <div className={`w-3 h-3 rounded-full ${
              systemStatus?.status === 'healthy' ? 'bg-green-500' : 'bg-yellow-500'
            }`}></div>
            <span className="text-gray-400">系统状态</span>
          </div>
          <p className="text-2xl font-bold">
            {loading ? '检测中...' : systemStatus?.status === 'healthy' ? '正常运行' : '异常'}
          </p>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center gap-3 mb-2">
            <Activity size={20} className="text-blue-500" />
            <span className="text-gray-400">已上传数据</span>
          </div>
          <p className="text-2xl font-bold">
            {loading ? '-' : systemStatus?.uploads_count || 0} 个
          </p>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center gap-3 mb-2">
            <Cpu size={20} className="text-purple-500" />
            <span className="text-gray-400">训练模型</span>
          </div>
          <p className="text-2xl font-bold">
            {loading ? '-' : systemStatus?.models_count || 0} 个
          </p>
        </div>
      </div>

      {/* 功能入口 */}
      <div>
        <h2 className="text-xl font-semibold mb-4">功能模块</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Link
                key={feature.path}
                to={feature.path}
                className="group bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all hover:-translate-y-1"
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4`}>
                  <Icon size={24} className="text-white" />
                </div>
                <h3 className="text-lg font-semibold mb-2 group-hover:text-blue-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-400 text-sm">
                  {feature.description}
                </p>
              </Link>
            );
          })}
        </div>
      </div>

      {/* 工作流程说明 */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">推荐工作流程</h2>
        <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
          {[
            { step: 1, title: '导入数据', desc: '上传 DICOM/OCTA' },
            { step: 2, title: '手动标注', desc: '创建金标准' },
            { step: 3, title: '训练模型', desc: 'U-Net + Diffusion' },
            { step: 4, title: '推理预测', desc: '消除拖尾' },
            { step: 5, title: '可视化', desc: '3D 展示导出' },
          ].map((item, idx, arr) => (
            <React.Fragment key={item.step}>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-sm font-bold">
                  {item.step}
                </div>
                <div>
                  <div className="font-medium">{item.title}</div>
                  <div className="text-sm text-gray-400">{item.desc}</div>
                </div>
              </div>
              {idx < arr.length - 1 && (
                <ArrowRight className="hidden md:block text-gray-600" size={20} />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
