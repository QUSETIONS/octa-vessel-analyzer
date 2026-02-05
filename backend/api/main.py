# -*- coding: utf-8 -*-
"""
OCTA 血管网络分析系统 - 主 API 入口

功能：
1. 启动 FastAPI 服务
2. 注册所有路由
3. 配置 CORS 和静态文件服务
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# 导入各功能模块的路由
from api.data_routes import router as data_router
from api.annotation_routes import router as annotation_router
from api.training_routes import router as training_router
from api.inference_routes import router as inference_router

# ============================================================================
# 应用配置
# ============================================================================

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

# 创建必要的目录
for dir_path in [UPLOADS_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FastAPI 应用实例
# ============================================================================

app = FastAPI(
    title="OCTA 血管网络分析系统",
    description="""
    ## 功能概述
    
    本系统提供完整的 OCTA 数据处理和分析功能：
    
    - **数据管理**: DICOM/OCTA 数据上传、解析、预览
    - **3D 可视化**: 交互式血管网络渲染
    - **标注工具**: 血管横截面手动标注，生成金标准
    - **深度学习**: 3D U-Net + Diffusion 模型训练与推理
    - **三维重建**: 血管网络导出为 mesh 格式
    
    ## 技术特点
    
    - 几何畸变矫正：解决 OCT 纵向拉伸问题
    - 拖尾伪影消除：基于深度学习的血管清理
    - 高效处理：支持大体积数据分块加载
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# 中间件配置
# ============================================================================

# CORS 配置，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # 备用，确保兼容性
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============================================================================
# 静态文件服务
# ============================================================================

# 挂载处理后的数据目录，供前端访问切片图像等
if PROCESSED_DIR.exists():
    app.mount("/static/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")
if ANNOTATIONS_DIR.exists():
    app.mount("/static/annotations", StaticFiles(directory=str(ANNOTATIONS_DIR)), name="annotations")

# ============================================================================
# 注册路由
# ============================================================================

# 数据管理路由
app.include_router(data_router, prefix="/api/data", tags=["数据管理"])

# 标注相关路由
app.include_router(annotation_router, prefix="/api/annotation", tags=["标注工具"])

# 训练相关路由
app.include_router(training_router, prefix="/api/training", tags=["模型训练"])

# 推理相关路由
app.include_router(inference_router, prefix="/api/inference", tags=["模型推理"])

# ============================================================================
# 根路由和健康检查
# ============================================================================

@app.get("/", tags=["系统"])
async def root():
    """系统根路由，返回基本信息"""
    return {
        "name": "OCTA 血管网络分析系统",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs"
    }


def _get_health_data():
    """获取健康检查数据（内部函数）"""
    return {
        "status": "healthy",
        "data_dir_exists": DATA_DIR.exists(),
        "uploads_count": len(list(UPLOADS_DIR.glob("*"))) if UPLOADS_DIR.exists() else 0,
        "models_count": len(list(MODELS_DIR.glob("*.pt"))) if MODELS_DIR.exists() else 0
    }


@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查接口"""
    return _get_health_data()


@app.get("/api/health", tags=["系统"])
async def api_health_check():
    """健康检查接口（兼容前端 /api/health 路径）"""
    return _get_health_data()


@app.get("/api/config", tags=["系统"])
async def get_config():
    """获取系统配置信息"""
    return {
        "data_dir": str(DATA_DIR),
        "uploads_dir": str(UPLOADS_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "annotations_dir": str(ANNOTATIONS_DIR),
        "models_dir": str(MODELS_DIR),
        "results_dir": str(RESULTS_DIR),
        "max_upload_size_mb": 2048,
        "supported_formats": [".dcm", ".dicom", ".nii", ".nii.gz", ".npy", ".mat"]
    }

# ============================================================================
# 异常处理
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": f"内部服务器错误: {str(exc)}",
            "status_code": 500
        }
    )

# ============================================================================
# 启动入口
# ============================================================================

def main():
    """主函数：启动 API 服务"""
    print("=" * 60)
    print("OCTA 血管网络分析系统 - 后端服务")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"API 文档: http://localhost:8000/docs")
    print(f"健康检查: http://localhost:8000/health")
    print(f"健康检查: http://localhost:8000/api/health")
    print("=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )

if __name__ == "__main__":
    main()
