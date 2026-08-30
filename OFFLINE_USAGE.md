# 离线使用指南

## 📦 模型文件

本项目支持完全离线运行,姿态/深度/分割模型均已预下载到 `models/` 目录。

### 已下载的模型

- ✅ `models/dwpose-det-yolox-m-640x640.onnx` (97MB) - DWPose 人体检测器 (YOLOX-m)
- ✅ `models/dwpose-pose-rtmw-x-l-192x256.onnx` (218MB) - DWPose 133 点姿态模型 (RTMW-x)
- ✅ `models/yolo26n-pose.pt` (7.5MB) - YOLO 姿态模型 (备用后端)
- ✅ `models/yolo26n-seg.pt` - 实例分割模型 (背景移除)
- ✅ `models/yolo26n-depth.pt` - 单目深度模型

### 模型说明

| 模型 | 大小 | 用途 | 状态 |
|------|------|------|------|
| dwpose-det-yolox-m-640x640.onnx | 97MB | DWPose 前置人体检测 | ✅ 已下载 |
| dwpose-pose-rtmw-x-l-192x256.onnx | 218MB | COCO-WholeBody 133 点 (身体+脚+面部+双手) | ✅ 已下载 |
| yolo26n-pose.pt | 7.5MB | 人体姿态估计 (COCO 17 个关键点) | ✅ 已下载 |
| yolo26s-pose.pt | - | Small 版本 (更高精度) | ⬜ 可选 |
| yolo26m-pose.pt | - | Medium 版本 | ⬜ 可选 |

### 关键点格式

| 后端 | 关键点数 | 覆盖部位 |
|------|----------|----------|
| `dwpose` (默认) | 133 | 身体 17 + 脚 6 + 面部 68 + 左右手各 21 |
| `yolo` | 17 | 仅身体 |

关键点名称、骨骼连接、3D 渲染子集的唯一真值源是 `src/ai3d/models/keypoints.py`,
后端会随 `animation_data.json` 一起把拓扑下发给前端,前端不再硬编码。
面部 68 点只出现在标注视频里,3D 视图只渲染身体+脚+双手共 65 点。

## 🔧 配置本地模型

在 `src/ai3d/config.py` 中,默认配置已指向本地模型:

```python
@dataclass
class ModelConfig:
    pose_backend: str = "dwpose"  # 'dwpose'(133 点) 或 'yolo'(17 点)
    pose_model: str = "./models/yolo26n-pose.pt"  # pose_backend='yolo' 时使用
    depth_model: str = "./models/yolo26n-depth.pt"
    device: str = "cpu"
    dwpose_det_model: str = "./models/dwpose-det-yolox-m-640x640.onnx"
    dwpose_pose_model: str = "./models/dwpose-pose-rtmw-x-l-192x256.onnx"
    dwpose_det_input_size: Tuple[int, int] = (640, 640)
    dwpose_pose_input_size: Tuple[int, int] = (192, 256)
```

> ⚠️ `dwpose_*_input_size` 必须与模型文件名里的尺寸一致,改模型时要同步改,否则关键点会整体错位。

## 💻 离线运行示例

```python
from ai3d import MotionCapturePipeline

# 初始化管线 (自动使用本地模型)
pipeline = MotionCapturePipeline()

# 处理视频 (无需网络连接)
result = pipeline.process_video('test_video.mp4')
```

## 📥 下载模型

DWPose 的两个 ONNX 首次需要联网拉取(之后完全离线):

```bash
python scripts/download_models.py --dwpose
```

如果需要其他 YOLO 姿态模型:

```bash
# 下载 Small 版本
python scripts/download_models.py --model yolo26s-pose.pt

# 列出所有可用模型
python scripts/download_models.py --list
```

下载后手动移动到 `models/` 目录,并修改配置文件中的 `pose_model` 路径。

## ⚙️ 切换姿态后端

编辑 `src/ai3d/config.py`:

```python
# 默认: DWPose 133 点,精度更高、含手部与面部
pose_backend: str = "dwpose"

# 切回 YOLO 17 点,速度更快
pose_backend: str = "yolo"
```

## 🌐 首次安装依赖

虽然模型可以离线使用,但首次安装 Python 依赖需要网络:

```bash
pip install -r requirements.txt
```

安装完成后,即可完全离线运行。

## 📝 注意事项

1. **模型文件不要删除**: `models/` 目录下的 `.pt` / `.onnx` 文件是离线运行的关键(通过 Git LFS 管理)
2. **路径配置**: 确保 `config.py` 中的模型路径正确
3. **不要只传文件名**: Ultralytics 与 rtmlib 在拿到裸文件名时会联网下载,项目内一律用 `resolve_model_path()` 解析成本地绝对路径
4. **设备选择**: 
   - CPU: 适合大多数场景
   - CUDA: 如果有 NVIDIA GPU,速度提升 10-50 倍
   - MPS: Apple Silicon Mac 用户可使用

## 🚀 性能对比

| 后端 | 关键点数 | CPU 速度 (1080p) | 说明 |
|------|----------|------------------|------|
| DWPose (RTMW-x) | 133 | ~110ms/帧 | 含检测+姿态两次推理,默认后端 |
| yolo26n-pose | 17 | ~40ms/帧 | 仅身体,速度优先 |
| yolo26s-pose | 17 | ~85ms/帧 | 仅身体,精度略高 |

推荐: 需要手部/面部细节时用 DWPose;只要躯干动作且追求实时时切到 `yolo`。
