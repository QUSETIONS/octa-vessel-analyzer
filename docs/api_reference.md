# OCTA 血管网络分析系统 - API 参考文档

## 基础信息

- **基础URL**: `http://localhost:8000/api`
- **内容类型**: `application/json`

## 数据管理 API

### 上传数据
```http
POST /data/upload
Content-Type: multipart/form-data
```

**参数**:
- `file`: DICOM/OCTA 文件

**响应**:
```json
{
  "id": "vol_20231201_123456_abc123",
  "filename": "octa_data.npy",
  "shape": [500, 300, 500],
  "spacing": [12.0, 12.0, 8.73],
  "vessel_fraction": 0.0523,
  "created_at": "2023-12-01T12:34:56"
}
```

### 获取数据列表
```http
GET /data/list
```

### 获取切片图像
```http
GET /data/{volume_id}/slice/{axis}/{index}?apply_correction=false
```

**参数**:
- `volume_id`: 数据 ID
- `axis`: 轴向 (x, y, z)
- `index`: 切片索引
- `apply_correction`: 是否应用畸变矫正

### 生成 3D 网格
```http
POST /data/{volume_id}/mesh
Content-Type: application/json

{
  "threshold": 0.5,
  "smooth": true
}
```

### 删除数据
```http
DELETE /data/{volume_id}
```

## 标注 API

### 创建标注项目
```http
POST /annotation/projects
Content-Type: application/json

{
  "volume_id": "vol_xxx",
  "name": "项目名称"
}
```

### 保存标注
```http
POST /annotation/projects/{project_id}/save
Content-Type: application/json

{
  "project_id": "ann_xxx",
  "layer_key": "y_150",
  "ellipses": [
    {
      "id": "ellipse_1",
      "center_x": 100,
      "center_y": 200,
      "radius_x": 10,
      "radius_y": 8,
      "rotation": 0,
      "slice_index": 150,
      "axis": "y"
    }
  ],
  "brush_strokes": []
}
```

### 撤销/重做
```http
POST /annotation/projects/{project_id}/undo
POST /annotation/projects/{project_id}/redo
```

### 生成金标准 Mask
```http
POST /annotation/projects/{project_id}/generate-mask
```

**响应**:
```json
{
  "message": "金标准 mask 已生成",
  "mask_path": "/path/to/gold_standard.npy",
  "shape": [500, 300, 500],
  "vessel_voxels": 3921500,
  "vessel_fraction": "5.23%"
}
```

## 训练 API

### 启动训练
```http
POST /training/start
Content-Type: application/json

{
  "volume_id": "vol_xxx",
  "annotation_project_id": "ann_xxx",
  "mode": "segmentation",
  "epochs": 100,
  "batch_size": 4,
  "learning_rate": 0.0001,
  "patch_size": 32,
  "diffusion_timesteps": 1000,
  "diffusion_beta_schedule": "cosine"
}
```

**训练模式**:
- `segmentation`: 仅训练 3D U-Net
- `diffusion`: 仅训练 Diffusion 模型
- `combined`: 级联训练

### 获取训练状态
```http
GET /training/status
```

**响应**:
```json
{
  "status": "running",
  "current_epoch": 25,
  "total_epochs": 100,
  "train_loss": 0.1234,
  "val_loss": 0.1456,
  "best_loss": 0.1200,
  "train_losses": [...],
  "val_losses": [...],
  "dice_scores": [...],
  "elapsed_time": 3600,
  "eta": 10800,
  "message": "训练中..."
}
```

### 控制训练
```http
POST /training/pause
POST /training/resume
POST /training/stop
```

### 检查点管理
```http
GET /training/checkpoints
DELETE /training/checkpoints/{filename}
POST /training/checkpoints/load/{filename}
```

## 推理 API

### 启动推理
```http
POST /inference/start
Content-Type: application/json

{
  "volume_id": "vol_xxx",
  "checkpoint_name": "best_segmentation.pt",
  "mode": "segmentation",
  "patch_size": 32,
  "overlap": 8,
  "threshold": 0.5,
  "use_ddim": true,
  "ddim_steps": 50
}
```

### 获取推理状态
```http
GET /inference/status/{task_id}
```

### 获取结果切片
```http
GET /inference/result/{task_id}/slice/{axis}/{index}
```

### 下载结果
```http
GET /inference/result/{task_id}/download/{format}
```

**支持格式**: `npy`, `nii`, `stl`, `ply`

## 系统 API

### 健康检查
```http
GET /health
```

### 获取配置
```http
GET /config
```

## 错误响应

所有 API 错误使用标准 HTTP 状态码，响应格式：

```json
{
  "detail": "错误描述信息"
}
```

常见状态码：
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误
