"""
Skeleton Data Model.
定义 3D 骨骼数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Joint:
    name: str
    position: List[float]  # [x, y, z]
    confidence: float = 1.0


@dataclass
class Skeleton:
    """存储一帧中的完整骨骼信息

    joints 元素既可以是 Joint，也可以是同构的 dict（管线内为了直接序列化成 JSON 用的是 dict）。
    """
    joints: List[Any] = field(default_factory=list)
    
    def get_joint_pos(self, name: str) -> List[float]:
        for joint in self.joints:
            joint_name = joint["name"] if isinstance(joint, dict) else joint.name
            if joint_name == name:
                return joint["position"] if isinstance(joint, dict) else joint.position
        return [0.0, 0.0, 0.0]


@dataclass
class PoseKeypoints:
    """一帧的原始 2D 关键点

    用于把估计器的原始输出传给标注绘制（代替 Ultralytics 的 Results 对象）。

    Attributes:
        keypoints: (K, 2) 像素坐标
        scores: (K,) 每个关键点的置信度
        keypoint_format: 关键点格式标识，取值见 ai3d.models.keypoints
        num_persons: 该帧检测到的人数（当前只保留置信度最高的一人）
    """
    keypoints: np.ndarray
    scores: np.ndarray
    keypoint_format: str
    num_persons: int = 1

    @property
    def num_keypoints(self) -> int:
        return int(self.keypoints.shape[0]) if self.keypoints is not None else 0

    def slice(self, start: int, end: int) -> Optional[np.ndarray]:
        """按区间取一段关键点（左闭右开），越界时返回 None"""
        if self.keypoints is None or end > self.num_keypoints:
            return None
        return self.keypoints[start:end]
