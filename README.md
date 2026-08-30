# AI 3D Animation Generator
基于 YOLO26 + MediaPipe + 物理约束的单目视频 3D 动画生成工具

## 📦 安装

```bash
cd ai-3d
pip install -e .
```

## 🚀 快速开始

### 第一阶段:姿态检测 (已完成 ✅)

```python
from ai3d import MotionCapturePipeline

# 初始化管线 (自动使用本地模型,支持离线运行)
pipeline = MotionCapturePipeline()

# 处理视频
result = pipeline.process_video('input.mp4')

# 访问骨骼数据
skeletons = result['skeletons']
first_frame = skeletons[0]
if first_frame:
    print(f"检测到 {len(first_frame.joints)} 个关键点")
```

> 💡 **离线使用**: 所有 YOLO26 模型已预下载到 `models/` 目录,无需网络连接即可运行。详见 [OFFLINE_USAGE.md](OFFLINE_USAGE.md)。

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
│   │   ├── pose_estimator.py    # 姿态估计
│   │   ├── depth_estimator.py   # 深度估计
│   │   ├── weapon_binder.py     # 武器绑定
│   │   └── physics_solver.py    # 物理解算
│   ├── models/            # 数据模型
│   │   ├── skeleton.py          # 骨骼数据结构
│   │   ├── animation.py         # 动画数据结构
│   │   └── weapon.py            # 武器数据结构
│   ├── pipeline/          # 处理管线
│   │   ├── base.py                # 基础管线
│   │   ├── mocap_pipeline.py      # 动捕管线
│   │   └── export_pipeline.py     # 导出管线
│   ├── utils/             # 工具函数
│   │   ├── video_io.py          # 视频读写
│   │   ├── smoothing.py         # 平滑滤波
│   │   └── coordinate.py        # 坐标转换
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

- ✅ **YOLO26 驱动**: 最新的端到端目标检测与姿态估计
- ✅ **深度感知**: 单目深度估计 + 关键点深度推算
- ✅ **武器绑定**: 刚体道具自动识别与空间绑定
- ✅ **物理修正**: 重力、碰撞、脚部接地自动优化
- ✅ **多格式导出**: 支持 FBX、BVH、GLTF 等游戏引擎格式

## 📝 License

MIT
