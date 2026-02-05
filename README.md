# OCTA Vessel Analyzer

OCTA 血管网络分析系统，提供从 OCTA/DICOM 数据导入、标注、训练、推理到三维可视化与导出的完整闭环。前端为 Vite + React + Electron，后端为 FastAPI，核心算法包含几何畸变矫正、血管标注与 3D 深度学习分割/扩散模型。

## 功能概览

- 数据管理：DICOM/NIfTI/NumPy/MAT 上传、解析与切片预览
- 标注工具：椭圆标注、画笔/橡皮、撤销重做、自动检测辅助
- 模型训练：3D U-Net、Diffusion、级联训练
- 推理预测：滑窗推理、阈值与 DDIM 可配置
- 3D 可视化：网格渲染与导出（STL/PLY/OBJ）

## 技术栈

- 前端：React 18 + Vite + Electron + Tailwind + Three.js
- 后端：FastAPI + Uvicorn
- 计算：PyTorch + NumPy + SciPy + scikit-image
- 医学影像：pydicom / SimpleITK / nibabel

## 快速开始

### 方式一：一键启动

```
scripts/start.bat
```

### 方式二：手动启动

后端：

```
cd backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

前端：

```
cd frontend
npm install
npm run dev
```

打开：`http://localhost:5173`

## API 入口

- API 基础地址：`http://localhost:8000/api`
- 文档与接口说明：`docs/api_reference.md`
- 运行时 Swagger：`http://localhost:8000/docs`

## 目录结构

```
backend/            # FastAPI 后端与核心算法
  api/              # 数据/标注/训练/推理路由
  core/             # 影像处理与矫正
  models/           # 3D U-Net / Diffusion
  training/         # 训练管理器
  data/             # 上传、处理、模型、结果
frontend/           # React + Electron 前端
docs/               # 使用指南与 API 文档
scripts/            # 启动脚本
```

## 数据格式支持

- DICOM 单文件：`.dcm`, `.dicom`
- DICOM 序列：ZIP
- NIfTI：`.nii`, `.nii.gz`
- NumPy：`.npy`
- MATLAB：`.mat`

## 训练与推理

- 训练入口：`POST /api/training/start`
- 训练模式：`segmentation` / `diffusion` / `combined`
- 推理入口：`POST /api/inference/start`
- 导出格式：`npy` / `nii` / `stl` / `ply`

完整参数见 `docs/api_reference.md`。

## 构建桌面应用

```
cd frontend
npm run electron:build
```

Windows 打包：

```
npm run electron:build:win
```

## 常见问题

- 后端启动失败：确认 8000 端口与依赖安装
- 前端白屏：确认后端已启动并检查浏览器控制台
- 训练很慢：使用 GPU，降低 batch size/patch size

更多见 `docs/usage.md`。
