"""
Configuration management for AI 3D Animation Generator.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# 项目根目录（src/ai3d/config.py -> ai3d -> src -> 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_path(path: str) -> str:
    """把配置里的相对模型路径按项目根解析为绝对路径，避免受当前工作目录影响"""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def default_output_dir() -> str:
    """输出根目录：默认项目根下的 output/，可用环境变量 AI3D_OUTPUT_DIR 覆盖"""
    env_dir = os.environ.get("AI3D_OUTPUT_DIR")
    if env_dir:
        return str(Path(env_dir).expanduser().resolve())
    return str(PROJECT_ROOT / "output")


@dataclass
class ModelConfig:
    """姿态/深度模型配置"""
    pose_backend: str = "dwpose"  # 'dwpose'(133 点全身) 或 'yolo'(COCO 17 点)
    pose_model: str = "./models/yolo26n-pose.pt"  # pose_backend='yolo' 时使用
    depth_model: str = "./models/yolo26n-depth.pt"  # 使用本地深度模型
    device: str = "cpu"  # 'cpu', 'cuda', 'mps'
    confidence_threshold: float = 0.5
    img_size: int = 640

    # --- DWPose (rtmlib Wholebody, balanced 档位) ---
    # 两个 ONNX 由 scripts/download_models.py --dwpose 下载到本地，运行时不联网
    dwpose_det_model: str = "./models/dwpose-det-yolox-m-640x640.onnx"
    dwpose_pose_model: str = "./models/dwpose-pose-rtmw-x-l-192x256.onnx"
    # 输入尺寸为 (宽, 高)，必须与上面的模型文件严格对应，否则关键点会整体错位
    dwpose_det_input_size: Tuple[int, int] = (640, 640)
    dwpose_pose_input_size: Tuple[int, int] = (192, 256)
    dwpose_backend: str = "onnxruntime"  # 'onnxruntime', 'opencv', 'openvino'
    dwpose_keypoint_threshold: float = 0.3  # 低于该置信度的关键点不参与绘制


@dataclass
class PipelineConfig:
    """处理管线配置"""
    use_depth: bool = True
    use_physics: bool = True
    remove_background: bool = False  # 是否启用背景移除
    fps_override: Optional[int] = None  # 如果为 None,使用视频原始 FPS
    render_annotated_video: bool = True  # 是否用 YOLO 原生 plot 生成标注视频
    annotate_masks: bool = True  # 标注视频中绘制实例分割掩码（需 remove_background=True 才有分割结果）
    annotate_pose: bool = True  # 标注视频中绘制骨骼关键点
    annotate_mask_alpha: float = 0.45  # 掩码混合系数，越小越能看清原画面
    annotated_video_name: str = "annotated.mp4"  # 标注视频文件名


@dataclass
class SmoothingConfig:
    """3D 姿态平滑与稳定化配置

    处理顺序: 2D 时序平滑 -> 深度锚定反投影 -> 骨长约束 -> 根节点稳定化 -> 轨迹中值滤波。
    调参口径见 .qoder/.plans 下的 3D_Smoothing_Optimization_Plan.md。
    """
    # --- 2D 关键点时序平滑 (OneEuro) ---
    enable_2d_smoothing: bool = True
    min_cutoff: float = 1.0  # 基础平滑强度，越小越平滑
    beta: float = 0.007  # 速度系数，越大对快速运动响应越快
    d_cutoff: float = 1.0  # 速度估计的截止频率

    # --- 3D 反投影的深度融合 ---
    depth_patch_radius: int = 2  # 深度采样邻域半径(像素)，避免单点命中背景
    max_depth_deviation: float = 0.8  # 单点深度允许偏离躯干深度的上限(米)

    # --- 骨长约束 ---
    enable_bone_constraint: bool = True
    bone_constraint_iterations: int = 4

    # --- 尺度归一化 ---
    normalize_scale: bool = True  # 把反投影结果缩放回真实人体尺度，下面的 m/s 参数才成立
    target_torso_length: float = 0.5  # 成人肩中点到髋中点的参考长度(米)

    # --- 根节点稳定化 ---
    enable_root_stabilization: bool = True
    max_horizontal_velocity: float = 2.0  # 髋部最大水平速度(m/s)，收紧以掐掉漂移
    max_vertical_velocity: float = 8.0  # 髋部最大垂直速度(m/s)，放宽以容纳跳跃
    recenter_alpha: float = 0.9  # 相对坐标偏移的跟随系数，越大越保留真实位移
    ground_clamp: bool = True  # 脚不得穿透地面
    ground_tolerance: float = 0.03  # 允许的穿地容差(米)
    airborne_threshold: float = 0.1  # 腾空判定阈值(米)，腾空时不做地面吸附

    # --- 轨迹中值滤波 ---
    enable_trajectory_smoothing: bool = True
    trajectory_window: int = 5  # 居中窗口大小(奇数)，越大越平滑

    # 低于该置信度的关键点不参与统计，2D 平滑时保持上一次结果
    keypoint_confidence_threshold: float = 0.3


@dataclass
class OutputConfig:
    """输出目录配置：每个任务在 base_dir 下独占一个子目录，产物全部落在其中"""
    base_dir: str = field(default_factory=default_output_dir)
    animation_json_name: str = "animation_data.json"  # 骨骼动画数据文件名
    source_video_stem: str = "source"  # 原视频副本的文件名（不含扩展名）
    keep_source_video: bool = True  # 是否把输入视频留在任务目录

    def task_dir(self, task_id: str, create: bool = True) -> Path:
        """返回指定任务的输出目录，create=True 时自动创建"""
        path = Path(self.base_dir) / task_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class ExportConfig:
    """导出配置（输出目录见 OutputConfig）"""
    output_format: str = "fbx"  # 'fbx', 'bvh', 'gltf'


class Config:
    """全局配置管理器"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.pipeline = PipelineConfig()
        self.smoothing = SmoothingConfig()
        self.output = OutputConfig()
        self.export = ExportConfig()
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从 YAML 文件加载配置"""
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(config.model, key):
                    # 输入尺寸在 YAML 里是列表，回填时转成 tuple 交给 rtmlib
                    if key in ('dwpose_det_input_size', 'dwpose_pose_input_size'):
                        value = tuple(value)
                    setattr(config.model, key, value)
                    
        if 'pipeline' in data:
            for key, value in data['pipeline'].items():
                if hasattr(config.pipeline, key):
                    setattr(config.pipeline, key, value)

        if 'smoothing' in data:
            for key, value in data['smoothing'].items():
                if hasattr(config.smoothing, key):
                    setattr(config.smoothing, key, value)
                    
        if 'output' in data:
            for key, value in data['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)
                    
        if 'export' in data:
            for key, value in data['export'].items():
                if hasattr(config.export, key):
                    setattr(config.export, key, value)
                    
        return config
    
    def save_to_yaml(self, yaml_path: str):
        """保存配置到 YAML 文件"""
        import yaml
        
        data = {
            'model': {
                'pose_backend': self.model.pose_backend,
                'pose_model': self.model.pose_model,
                'depth_model': self.model.depth_model,
                'device': self.model.device,
                'confidence_threshold': self.model.confidence_threshold,
                'img_size': self.model.img_size,
                'dwpose_det_model': self.model.dwpose_det_model,
                'dwpose_pose_model': self.model.dwpose_pose_model,
                'dwpose_det_input_size': list(self.model.dwpose_det_input_size),
                'dwpose_pose_input_size': list(self.model.dwpose_pose_input_size),
                'dwpose_backend': self.model.dwpose_backend,
                'dwpose_keypoint_threshold': self.model.dwpose_keypoint_threshold,
            },
            'pipeline': {
                'use_depth': self.pipeline.use_depth,
                'use_physics': self.pipeline.use_physics,
                'remove_background': self.pipeline.remove_background,
                'fps_override': self.pipeline.fps_override,
                'render_annotated_video': self.pipeline.render_annotated_video,
                'annotate_masks': self.pipeline.annotate_masks,
                'annotate_pose': self.pipeline.annotate_pose,
                'annotate_mask_alpha': self.pipeline.annotate_mask_alpha,
                'annotated_video_name': self.pipeline.annotated_video_name,
            },
            'smoothing': {
                'enable_2d_smoothing': self.smoothing.enable_2d_smoothing,
                'min_cutoff': self.smoothing.min_cutoff,
                'beta': self.smoothing.beta,
                'd_cutoff': self.smoothing.d_cutoff,
                'depth_patch_radius': self.smoothing.depth_patch_radius,
                'max_depth_deviation': self.smoothing.max_depth_deviation,
                'enable_bone_constraint': self.smoothing.enable_bone_constraint,
                'bone_constraint_iterations': self.smoothing.bone_constraint_iterations,
                'normalize_scale': self.smoothing.normalize_scale,
                'target_torso_length': self.smoothing.target_torso_length,
                'enable_root_stabilization': self.smoothing.enable_root_stabilization,
                'max_horizontal_velocity': self.smoothing.max_horizontal_velocity,
                'max_vertical_velocity': self.smoothing.max_vertical_velocity,
                'recenter_alpha': self.smoothing.recenter_alpha,
                'ground_clamp': self.smoothing.ground_clamp,
                'ground_tolerance': self.smoothing.ground_tolerance,
                'airborne_threshold': self.smoothing.airborne_threshold,
                'enable_trajectory_smoothing': self.smoothing.enable_trajectory_smoothing,
                'trajectory_window': self.smoothing.trajectory_window,
                'keypoint_confidence_threshold': self.smoothing.keypoint_confidence_threshold,
            },
            'output': {
                'base_dir': self.output.base_dir,
                'animation_json_name': self.output.animation_json_name,
                'source_video_stem': self.output.source_video_stem,
                'keep_source_video': self.output.keep_source_video,
            },
            'export': {
                'output_format': self.export.output_format,
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
