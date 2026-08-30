"""
YOLO26 Depth Estimator Module.
封装 Ultralytics YOLO26-depth 模型,用于单目深度估计。
"""

import os
import torch
import numpy as np
from ultralytics import YOLO
from typing import Optional
from ai3d.config import resolve_model_path

# 禁用 Ultralytics 自动下载模型
os.environ["YOLO_OFFLINE"] = "true"


class DepthEstimator:
    def __init__(self, model_name: str = "./models/yolo26n-depth.pt", device: str = "cpu"):
        """
        初始化深度估计器。
        
        Args:
            model_name: YOLO26-depth 模型路径，相对路径按项目根解析
            device: 运行设备 ('cpu', 'cuda', 'mps')
        """
        # 必须传本地绝对路径：只给文件名时 Ultralytics 会联网下载到当前工作目录
        model_path = resolve_model_path(model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO 深度模型缺失: {model_path}\n"
                f"请先运行: python scripts/download_models.py"
            )
        
        self.model = YOLO(model_path)
        self.device = device
    
    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        对单帧图像进行深度估计。
        
        Args:
            frame: BGR 格式的图像数组
            
        Returns:
            深度图数组 (H, W),单位为米。如果失败则返回 None
        """
        try:
            results = self.model(frame, verbose=False, device=self.device)
            
            if not results or not results[0].depth:
                return None
            
            # 获取深度图 (H, W),单位为米
            depth_map = results[0].depth.data.cpu().numpy()
            
            return depth_map
            
        except Exception as e:
            print(f"深度估计失败: {e}")
            return None
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: float, y: float) -> float:
        """
        从深度图中获取指定像素位置的深度值。
        
        Args:
            depth_map: 深度图数组 (H, W)
            x: 像素 x 坐标
            y: 像素 y 坐标
            
        Returns:
            深度值(米),如果超出范围则返回 0.0
        """
        h, w = depth_map.shape
        
        # 边界检查
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0
        
        # 取整到最近的像素
        x_int = int(round(x))
        y_int = int(round(y))
        
        # 再次边界检查
        if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
            return 0.0
        
        depth_value = depth_map[y_int, x_int]
        
        # 过滤无效值
        if depth_value <= 0 or np.isnan(depth_value) or np.isinf(depth_value):
            return 0.0
        
        return float(depth_value)
