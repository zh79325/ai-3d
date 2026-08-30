"""
Unit tests for pose estimation.
"""

import pytest
import numpy as np
from ai3d.core.pose_estimator import PoseEstimator
from ai3d.models.skeleton import Skeleton


class TestPoseEstimator:
    """姿态估计器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        estimator = PoseEstimator(device="cpu")
        assert estimator.model is not None
        assert estimator.device == "cpu"
        assert len(estimator.keypoint_names) == 17
    
    def test_keypoint_names(self):
        """测试关键点名称"""
        estimator = PoseEstimator(device="cpu")
        expected_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        assert estimator.keypoint_names == expected_names


class TestSkeleton:
    """骨骼数据模型测试"""
    
    def test_skeleton_creation(self):
        """测试骨骼创建"""
        joints = [
            {"name": "nose", "position": [0.5, 0.3, 0.0], "confidence": 0.9}
        ]
        skeleton = Skeleton(joints=joints)
        assert len(skeleton.joints) == 1
        assert skeleton.joints[0]["name"] == "nose"
    
    def test_get_joint_pos(self):
        """测试获取关键点位置"""
        joints = [
            {"name": "nose", "position": [0.5, 0.3, 0.0], "confidence": 0.9},
            {"name": "left_eye", "position": [0.4, 0.25, 0.0], "confidence": 0.85}
        ]
        skeleton = Skeleton(joints=joints)
        
        nose_pos = skeleton.get_joint_pos("nose")
        assert nose_pos == [0.5, 0.3, 0.0]
        
        # 测试不存在的关键点
        missing_pos = skeleton.get_joint_pos("nonexistent")
        assert missing_pos == [0.0, 0.0, 0.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
