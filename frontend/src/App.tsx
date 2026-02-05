/**
 * OCTA 血管网络分析系统 - App 主组件
 * 
 * 应用程序的根组件，包含：
 * 1. 路由配置
 * 2. 全局状态管理
 * 3. 布局结构
 */

import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import DataPage from './pages/DataPage';
import AnnotationPage from './pages/AnnotationPage';
import TrainingPage from './pages/TrainingPage';
import InferencePage from './pages/InferencePage';
import VisualizationPage from './pages/VisualizationPage';

const App: React.FC = () => {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/data" element={<DataPage />} />
        <Route path="/annotation" element={<AnnotationPage />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/inference" element={<InferencePage />} />
        <Route path="/visualization" element={<VisualizationPage />} />
      </Routes>
    </Layout>
  );
};

export default App;
