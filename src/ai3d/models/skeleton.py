"""
Skeleton Data Model.
定义 3D 骨骼数据结构。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Joint:
    name: str
    position: List[float]  # [x, y, z]
    confidence: float = 1.0


@dataclass
class Skeleton:
    """存储一帧中的完整骨骼信息"""
    joints: List[Joint] = field(default_factory=list)
    
    def get_joint_pos(self, name: str) -> List[float]:
        for joint in self.joints:
            if joint.name == name:
                return joint.position
        return [0.0, 0.0, 0.0]
