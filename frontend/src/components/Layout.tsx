/**
 * OCTA 血管网络分析系统 - 布局组件
 * 
 * 应用程序的整体布局结构：
 * - 顶部导航栏
 * - 左侧工具栏
 * - 主内容区域
 * - 右侧状态面板（可选）
 */

import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Home,
  Database,
  Edit3,
  Cpu,
  Play,
  Eye,
  Settings,
  HelpCircle,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

// 导航项配置
const navItems = [
  { path: '/', icon: Home, label: '首页', description: '系统概览' },
  { path: '/data', icon: Database, label: '数据管理', description: '导入和管理OCTA数据' },
  { path: '/annotation', icon: Edit3, label: '标注工具', description: '血管横截面标注' },
  { path: '/training', icon: Cpu, label: '模型训练', description: '深度学习模型训练' },
  { path: '/inference', icon: Play, label: '推理预测', description: '应用模型进行推理' },
  { path: '/visualization', icon: Eye, label: '三维可视化', description: '血管网络3D展示' },
];

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const location = useLocation();

  // 获取当前页面信息
  const currentPage = navItems.find(item => item.path === location.pathname) || navItems[0];

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* 左侧导航栏 */}
      <aside 
        className={`
          ${sidebarCollapsed ? 'w-16' : 'w-64'} 
          flex flex-col bg-gray-800 border-r border-gray-700 transition-all duration-300
        `}
      >
        {/* Logo */}
        <div className="h-16 flex items-center justify-center border-b border-gray-700">
          {sidebarCollapsed ? (
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-xl font-bold">O</span>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-xl font-bold">O</span>
              </div>
              <div>
                <h1 className="text-lg font-bold">OCTA 分析</h1>
                <p className="text-xs text-gray-400">血管网络系统</p>
              </div>
            </div>
          )}
        </div>

        {/* 导航菜单 */}
        <nav className="flex-1 py-4 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`
                  flex items-center gap-3 px-4 py-3 mx-2 rounded-lg transition-all
                  ${isActive 
                    ? 'bg-blue-600 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }
                `}
                title={sidebarCollapsed ? item.label : undefined}
              >
                <Icon size={20} className="flex-shrink-0" />
                {!sidebarCollapsed && (
                  <div className="min-w-0">
                    <div className="font-medium truncate">{item.label}</div>
                    <div className="text-xs text-gray-400 truncate">{item.description}</div>
                  </div>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* 底部工具 */}
        <div className="border-t border-gray-700 p-2">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-gray-400 hover:bg-gray-700 hover:text-white transition-all"
          >
            {sidebarCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            {!sidebarCollapsed && <span>收起侧栏</span>}
          </button>
          
          {!sidebarCollapsed && (
            <div className="flex gap-2 mt-2">
              <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:bg-gray-700 hover:text-white transition-all">
                <Settings size={18} />
                <span>设置</span>
              </button>
              <button className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:bg-gray-700 hover:text-white transition-all">
                <HelpCircle size={18} />
                <span>帮助</span>
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* 主内容区域 */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* 顶部标题栏 */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-gray-700 bg-gray-800/50">
          <div>
            <h2 className="text-lg font-semibold">{currentPage.label}</h2>
            <p className="text-sm text-gray-400">{currentPage.description}</p>
          </div>
          
          {/* 状态指示器 */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span className="text-gray-400">后端服务运行中</span>
            </div>
          </div>
        </header>

        {/* 页面内容 */}
        <div className="flex-1 overflow-auto p-6">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
