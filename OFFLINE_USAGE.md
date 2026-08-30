# 离线使用指南

## 📦 模型文件

本项目支持完全离线运行,所有 YOLO26 模型已预下载到 `models/` 目录。

### 已下载的模型

- ✅ `models/yolo26n-pose.pt` (7.5MB) - 姿态估计模型

### 模型说明

| 模型 | 大小 | 用途 | 状态 |
|------|------|------|------|
| yolo26n-pose.pt | 7.5MB | 人体姿态估计 (17个关键点) | ✅ 已下载 |
| yolo26s-pose.pt | - | Small 版本 (更高精度) | ⬜ 可选 |
| yolo26m-pose.pt | - | Medium 版本 | ⬜ 可选 |
| yolo26l-pose.pt | - | Large 版本 | ⬜ 可选 |
| yolo26x-pose.pt | - | X-Large 版本 (最高精度) | ⬜ 可选 |

## 🔧 配置本地模型

在 `src/ai3d/config.py` 中,默认配置已指向本地模型:

```python
@dataclass
class ModelConfig:
    pose_model: str = "./models/yolo26n-pose.pt"  # 本地路径
    depth_model: str = "./models/yolo26n-depth.pt"
    device: str = "cpu"
```

## 💻 离线运行示例

```python
from ai3d import MotionCapturePipeline

# 初始化管线 (自动使用本地模型)
pipeline = MotionCapturePipeline()

# 处理视频 (无需网络连接)
result = pipeline.process_video('test_video.mp4')
```

## 📥 下载其他模型 (可选)

如果需要更高精度的模型,可以手动下载:

```bash
# 下载 Small 版本
python scripts/download_models.py --model yolo26s-pose.pt

# 下载 Medium 版本
python scripts/download_models.py --model yolo26m-pose.pt

# 列出所有可用模型
python scripts/download_models.py --list
```

下载后手动移动到 `models/` 目录,并修改配置文件中的 `pose_model` 路径。

## ⚙️ 切换模型

编辑 `src/ai3d/config.py`:

```python
# 使用 Small 版本 (更高精度,稍慢)
pose_model: str = "./models/yolo26s-pose.pt"

# 或使用 Nano 版本 (最快)
pose_model: str = "./models/yolo26n-pose.pt"
```

## 🌐 首次安装依赖

虽然模型可以离线使用,但首次安装 Python 依赖需要网络:

```bash
pip install -r requirements.txt
```

安装完成后,即可完全离线运行。

## 📝 注意事项

1. **模型文件不要删除**: `models/` 目录下的 `.pt` 文件是离线运行的关键
2. **路径配置**: 确保 `config.py` 中的模型路径正确
3. **设备选择**: 
   - CPU: 适合大多数场景
   - CUDA: 如果有 NVIDIA GPU,速度提升 10-50 倍
   - MPS: Apple Silicon Mac 用户可使用

## 🚀 性能对比

| 模型 | CPU 速度 | GPU 速度 | 精度 (mAP) |
|------|----------|----------|------------|
| yolo26n-pose | ~40ms/帧 | ~2ms/帧 | 57.2 |
| yolo26s-pose | ~85ms/帧 | ~3ms/帧 | 63.0 |
| yolo26m-pose | ~218ms/帧 | ~5ms/帧 | 68.8 |

推荐: 实时应用使用 `yolo26n-pose`,高精度需求使用 `yolo26s-pose`。
