"""
YOLO26 Pose Estimator Module.
封装 Ultralytics YOLO26-pose 模型，用于提取人体关键点。
"""

import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
from ai3d.models.skeleton import Skeleton


class PoseEstimator:
    def __init__(self, model_name: str = "yolo26n-pose.pt", device: str = "cpu"):
        """
        初始化姿态估计器。
        
        Args:
            model_name: YOLO26-pose 模型名称 (如 yolo26n-pose.pt)
            device: 运行设备 ('cpu', 'cuda', 'mps')
        """
        self.model = YOLO(model_name)
        self.device = device
        # COCO 17个关键点的名称映射
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def estimate_frame(self, frame: np.ndarray) -> Optional[Skeleton]:
        """
        对单帧图像进行姿态估计。
        
        Args:
            frame: BGR 格式的图像数组
            
        Returns:
            Skeleton 对象，如果未检测到人则返回 None
        """
        results = self.model(frame, verbose=False, device=self.device)
        
        if not results or not results[0].keypoints:
            return None
            
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        if len(keypoints_data) == 0:
            return None
            
        # 取置信度最高的那个人
        best_person = keypoints_data[0] 
        joints = []
        
        for i, (x, y, conf) in enumerate(best_person):
            if i < len(self.keypoint_names):
                # 这里暂时只返回 2D 坐标，后续结合深度图转为 3D
                joints.append({
                    "name": self.keypoint_names[i],
                    "position": [float(x), float(y), 0.0], 
                    "confidence": float(conf)
                })
                
        return Skeleton(joints=joints)
    
    def estimate_batch(self, frames: List[np.ndarray]) -> List[Optional[Skeleton]]:
        """
        批量处理多帧图像。
        
        Args:
            frames: BGR 格式的图像数组列表
            
        Returns:
            Skeleton 对象列表
        """
        skeletons = []
        for i, frame in enumerate(frames):
            skeleton = self.estimate_frame(frame)
            skeletons.append(skeleton)
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(frames)} 帧")
                
        return skeletons
