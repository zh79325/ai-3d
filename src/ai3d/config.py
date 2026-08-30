"""
Configuration management for AI 3D Animation Generator.
"""

import os
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class ExportConfig:
    """导出配置"""
    output_format: str = "fbx"  # 'fbx', 'bvh', 'gltf'
    output_dir: str = "./output"


class Config:
    """全局配置管理器"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.pipeline = PipelineConfig()
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
            },
            'export': {
                'output_format': self.export.output_format,
                'output_dir': self.export.output_dir,
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
