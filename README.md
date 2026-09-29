# AI 3D Animation Generator
基于 DWPose(133 点全身姿态) + YOLO26 + 物理约束的单目视频 3D 动画生成工具

## 📦 安装

```bash
cd ai-3d
pip install -e .
```

## 🚀 快速开始

### 第一阶段:姿态检测 (已完成 ✅)

默认使用 DWPose(COCO-WholeBody 133 点:身体+脚+面部+双手),
可在 `config.model.pose_backend` 切回 YOLO 17 点。

```python
from ai3d import MotionCapturePipeline

# 初始化管线 (自动使用本地模型,支持离线运行)
pipeline = MotionCapturePipeline()

# 处理视频
result = pipeline.process_video('input.mp4')

# 访问骨骼数据
skeletons = result['skeletons']
print(f"关键点格式: {result['keypoint_format']}")  # coco_wholebody_133
first_frame = skeletons[0]
if first_frame:
    print(f"检测到 {len(first_frame.joints)} 个关键点")
```

### 第二阶段:3D 姿态重建 (已完成 ✅)

```python
from ai3d import MotionCapturePipeline

# 初始化管线 (启用深度估计)
pipeline = MotionCapturePipeline()

# 处理视频 (自动进行 2D->3D 转换)
result = pipeline.process_video('input.mp4')

# 访问 3D 骨骼数据
skeletons = result['skeletons']
if skeletons[0]:
    # 检查数据结构
    if isinstance(skeletons[0], dict):
        joints = skeletons[0]['joints']
    else:
        joints = skeletons[0].joints
    
    for joint in joints[:3]:
        pos = joint['position'] if isinstance(joint, dict) else joint.position
        print(f"{joint['name'] if isinstance(joint, dict) else joint.name}: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
```

### 第三阶段:Web预览器 (已完成 ✅)

```python
from ai3d import launch_viewer

# 启动Web预览服务器
launch_viewer('input.mp4', port=8765)

# 然后在浏览器访问: http://localhost:8765/api/skeleton-viewer/viewer
```

**功能特性:**
- 🎮 Three.js 3D渲染,支持旋转/缩放视角
- ▶️ 播放控制:播放/暂停/上一帧/下一帧
- 🔗 可集成到Electron或其他FastAPI项目
- 📡 RESTful API提供骨骼数据

详见 [VIEWER_INTEGRATION.md](docs/VIEWER_INTEGRATION.md)

> 💡 **离线使用**: DWPose 与 YOLO26 模型已预下载到 `models/` 目录,无需网络连接即可运行。详见 [OFFLINE_USAGE.md](OFFLINE_USAGE.md)。

> 🦴 **UniRig 骨架蒙皮新链路**: S2 绑定新增 `method=unirig` —— 用 UniRig 学习模型在 **CPU 进程内**直接推骨架+蒙皮,产物与旧链路同契约,旧链路零改动。首次需 `scripts/setup_unirig.py` + `scripts/download_models.py --unirig` 两步联网准备(权重落 `models/unirig/`,vendored 仓库落 `third_party/UniRig`),之后完全离线。详见 [OFFLINE_USAGE.md](OFFLINE_USAGE.md)。

### 完整功能 (开发中)

```python
from ai3d import MotionCapturePipeline

# 初始化管线
pipeline = MotionCapturePipeline(
    use_depth=True,
    use_physics=True,
    weapon_type='sword'
)

# 处理视频
result = pipeline.process_video('input.mp4', output_dir='./output')

# 导出为游戏引擎可用格式
result.export_to_fbx('character_animation.fbx')
```

## 📁 项目结构

```
ai-3d/
├── src/ai3d/              # 核心包
│   ├── __init__.py        # 包入口
│   ├── core/              # 核心算法
│   │   ├── dwpose_estimator.py  # DWPose 133 点姿态估计(默认)
│   │   ├── pose_estimator.py    # YOLO 17 点姿态估计(备用)
│   │   ├── depth_estimator.py   # 深度估计
│   │   ├── kinematic_constraints.py  # 骨长恒定约束
│   │   ├── root_stabilizer.py   # 尺度归一化 + 根节点稳定化 + 地面约束
│   │   ├── weapon_binder.py     # 武器绑定
│   │   └── physics_solver.py    # 物理解算
│   ├── models/            # 数据模型
│   │   ├── keypoints.py         # 关键点名称/骨骼拓扑(单一真值源)
│   │   ├── skeleton.py          # 骨骼数据结构
│   │   ├── animation.py         # 动画数据结构
│   │   └── weapon.py            # 武器数据结构
│   ├── pipeline/          # 处理管线
│   │   ├── base.py                # 基础管线
│   │   ├── mocap_pipeline.py      # 动捕管线
│   │   └── export_pipeline.py     # 导出管线
│   ├── utils/             # 工具函数
│   │   ├── video_io.py          # 视频读写
│   │   ├── smoothing.py         # OneEuro 时序平滑 / 轨迹中值滤波
│   │   └── coordinate.py        # 2D->3D 反投影(深度块采样 + 躯干锚定)
│   └── config/            # 配置管理
│       ├── default.yaml         # 默认配置
│       └── loader.py            # 配置加载器
├── tests/                 # 测试用例
├── examples/              # 示例代码
├── docs/                  # 文档
├── setup.py               # 安装脚本
├── pyproject.toml         # 项目配置
└── README.md              # 项目说明
```

## 🔧 作为第三方依赖集成

在你的游戏素材生成工程中:

```python
# requirements.txt
ai3d @ git+https://github.com/your-org/ai-3d.git@main

# 或者本地安装
pip install -e /path/to/ai-3d
```

```python
# 在你的工程中使用
from ai3d.pipeline import MotionCapturePipeline

def generate_game_asset(video_path):
    pipeline = MotionCapturePipeline()
    animation = pipeline.process_video(video_path)
    
    # 导出为 Unity/UE 可用格式
    animation.export_to_fbx('output.fbx')
    return 'output.fbx'
```

## 🎯 核心特性

- ✅ **DWPose 全身关键点**: COCO-WholeBody 133 点,含手部 42 点与面部 68 点
- ✅ **YOLO26 驱动**: 端到端目标检测、实例分割与备用姿态后端
- ✅ **深度感知**: 单目深度估计 + 关键点深度推算
- ✅ **3D 平滑稳定**: OneEuro 时序滤波 + 骨长约束 + 尺度归一化 + 根节点稳定化(见 `SmoothingConfig`)
- ✅ **武器绑定**: 刚体道具自动识别与空间绑定
- ✅ **物理修正**: 重力、碰撞、脚部接地自动优化
- ✅ **多格式导出**: 支持 FBX、BVH、GLTF 等游戏引擎格式

## 📝 License

MIT
