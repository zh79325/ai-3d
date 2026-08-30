"""
Configuration management for AI 3D Animation Generator.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 项目根目录（src/ai3d/config.py -> ai3d -> src -> 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_output_dir() -> str:
    """输出根目录：默认项目根下的 output/，可用环境变量 AI3D_OUTPUT_DIR 覆盖"""
    env_dir = os.environ.get("AI3D_OUTPUT_DIR")
    if env_dir:
        return str(Path(env_dir).expanduser().resolve())
    return str(PROJECT_ROOT / "output")


@dataclass
class ModelConfig:
    """YOLO26 模型配置"""
    pose_model: str = "./models/yolo26n-pose.pt"  # 使用本地模型
    depth_model: str = "./models/yolo26n-depth.pt"  # 使用本地深度模型
    device: str = "cpu"  # 'cpu', 'cuda', 'mps'
    confidence_threshold: float = 0.5
    img_size: int = 640


@dataclass
class PipelineConfig:
    """处理管线配置"""
    use_depth: bool = True
    use_physics: bool = True
    remove_background: bool = False  # 是否启用背景移除
    smooth_frames: int = 5  # 平滑滤波窗口大小
    fps_override: Optional[int] = None  # 如果为 None,使用视频原始 FPS
    render_annotated_video: bool = True  # 是否用 YOLO 原生 plot 生成标注视频
    annotate_masks: bool = True  # 标注视频中绘制实例分割掩码（需 remove_background=True 才有分割结果）
    annotate_pose: bool = True  # 标注视频中绘制骨骼关键点
    annotate_mask_alpha: float = 0.45  # 掩码混合系数，越小越能看清原画面
    annotated_video_name: str = "annotated.mp4"  # 标注视频文件名


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
                    setattr(config.model, key, value)
                    
        if 'pipeline' in data:
            for key, value in data['pipeline'].items():
                if hasattr(config.pipeline, key):
                    setattr(config.pipeline, key, value)
                    
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
                'pose_model': self.model.pose_model,
                'depth_model': self.model.depth_model,
                'device': self.model.device,
                'confidence_threshold': self.model.confidence_threshold,
                'img_size': self.model.img_size,
            },
            'pipeline': {
                'use_depth': self.pipeline.use_depth,
                'use_physics': self.pipeline.use_physics,
                'remove_background': self.pipeline.remove_background,
                'smooth_frames': self.pipeline.smooth_frames,
                'fps_override': self.pipeline.fps_override,
                'render_annotated_video': self.pipeline.render_annotated_video,
                'annotate_masks': self.pipeline.annotate_masks,
                'annotate_pose': self.pipeline.annotate_pose,
                'annotate_mask_alpha': self.pipeline.annotate_mask_alpha,
                'annotated_video_name': self.pipeline.annotated_video_name,
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
