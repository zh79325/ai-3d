"""
YOLO26 Pose Estimator Module.
封装 Ultralytics YOLO26-pose 模型，用于提取人体关键点。
"""

import os
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Optional
from ai3d.config import resolve_model_path
from ai3d.models.keypoints import FORMAT_COCO_17, get_keypoint_names
from ai3d.models.skeleton import Skeleton

# 禁用 Ultralytics 自动下载模型
os.environ["YOLO_OFFLINE"] = "true"


class PoseEstimator:
    def __init__(self, model_name: str = "./models/yolo26n-pose.pt", device: str = "cpu"):
        """
        初始化姿态估计器。
        
        Args:
            model_name: YOLO26-pose 模型路径，相对路径按项目根解析
            device: 运行设备 ('cpu', 'cuda', 'mps')
        """
        # 必须传本地绝对路径：只给文件名时 Ultralytics 会联网下载到当前工作目录
        model_path = resolve_model_path(model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO 姿态模型缺失: {model_path}\n"
                f"请先运行: python scripts/download_models.py"
            )
        
        self.model = YOLO(model_path)
        self.device = device
        # COCO 17 个关键点的名称与拓扑统一由 ai3d.models.keypoints 提供
        self.keypoint_format = FORMAT_COCO_17
        self.keypoint_names = get_keypoint_names(FORMAT_COCO_17)

    def estimate_frame(self, frame: np.ndarray, return_result: bool = False):
        """
        对单帧图像进行姿态估计。
        
        Args:
            frame: BGR 格式的图像数组
            return_result: 为 True 时额外返回 Ultralytics 原始 Results 对象（供 plot 标注使用）
            
        Returns:
            Skeleton 对象（未检测到人时为 None）；
            return_result=True 时返回 (Skeleton, Results) 元组
        """
        results = self.model(frame, verbose=False, device=self.device)
        raw_result = results[0] if results else None
        
        if not results or not results[0].keypoints:
            return (None, raw_result) if return_result else None
            
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        if len(keypoints_data) == 0:
            return (None, raw_result) if return_result else None
            
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
                
        skeleton = Skeleton(joints=joints)
        return (skeleton, raw_result) if return_result else skeleton
    
    def estimate_batch(self, frames: List[np.ndarray], return_results: bool = False):
        """
        批量处理多帧图像。
        
        Args:
            frames: BGR 格式的图像数组列表
            return_results: 为 True 时额外返回每帧的 Ultralytics 原始 Results 列表
            
        Returns:
            Skeleton 对象列表；return_results=True 时返回 (skeletons, raw_results) 元组
        """
        skeletons = []
        raw_results = []
        for i, frame in enumerate(frames):
            skeleton, raw_result = self.estimate_frame(frame, return_result=True)
            skeletons.append(skeleton)
            raw_results.append(raw_result)
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(frames)} 帧")
                
        return (skeletons, raw_results) if return_results else skeletons
