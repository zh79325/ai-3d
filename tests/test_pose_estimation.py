"""
Unit tests for pose estimation.

关键点名称/拓扑的唯一真值源是 ai3d.models.keypoints，测试里不再重复硬编码一份。
"""

import numpy as np
import pytest

from ai3d.config import Config
from ai3d.core.pose_estimator import PoseEstimator
from ai3d.models.keypoints import (
    COCO_17_NAMES,
    FORMAT_COCO_17,
    FORMAT_WHOLEBODY_133,
    WHOLEBODY_GROUPS,
    describe_format,
    format_from_keypoint_count,
    get_bone_connections,
    get_keypoint_names,
    get_render_3d_indices,
)
from ai3d.models.skeleton import PoseKeypoints, Skeleton


class TestKeypointDefinitions:
    """关键点定义（纯数据，无需加载模型）"""

    def test_wholebody_counts(self):
        """133 点名称唯一，分组区间连续覆盖全部索引"""
        names = get_keypoint_names(FORMAT_WHOLEBODY_133)
        assert len(names) == 133
        assert len(set(names)) == 133

        bounds = sorted(WHOLEBODY_GROUPS.values())
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 133
        for (_, prev_end), (start, _) in zip(bounds, bounds[1:]):
            assert prev_end == start

    def test_bones_within_render_subset(self):
        """3D 只渲染身体+脚+双手，任何骨骼的两端都必须落在该子集内"""
        render_indices = set(get_render_3d_indices(FORMAT_WHOLEBODY_133))
        assert len(render_indices) == 65

        for start, end in get_bone_connections(FORMAT_WHOLEBODY_133):
            assert start in render_indices
            assert end in render_indices

    def test_format_from_keypoint_count(self):
        """按关键点数量反查格式，未知数量返回 None"""
        assert format_from_keypoint_count(17) == FORMAT_COCO_17
        assert format_from_keypoint_count(133) == FORMAT_WHOLEBODY_133
        assert format_from_keypoint_count(5) is None

    def test_describe_format_payload(self):
        """下发给前端的描述必须字段齐全，且长度与名称数一致"""
        payload = describe_format(FORMAT_WHOLEBODY_133)
        assert payload["format"] == FORMAT_WHOLEBODY_133
        assert len(payload["keypoint_names"]) == 133
        assert payload["bone_connections"]
        assert payload["render_3d_indices"]
        assert "left_hand" in payload["groups"]


class TestPoseEstimator:
    """YOLO 后端（COCO 17 点）"""

    def test_initialization(self):
        """测试初始化"""
        estimator = PoseEstimator(device="cpu")
        assert estimator.model is not None
        assert estimator.device == "cpu"
        assert len(estimator.keypoint_names) == 17

    def test_keypoint_names(self):
        """测试关键点名称"""
        estimator = PoseEstimator(device="cpu")
        assert estimator.keypoint_format == FORMAT_COCO_17
        assert estimator.keypoint_names == COCO_17_NAMES


class TestDWPoseEstimator:
    """DWPose 后端（COCO-WholeBody 133 点），模型缺失时跳过"""

    @pytest.fixture
    def estimator(self):
        from ai3d.core.dwpose_estimator import DWPoseEstimator

        config = Config().model
        try:
            return DWPoseEstimator(
                det_model=config.dwpose_det_model,
                pose_model=config.dwpose_pose_model,
                det_input_size=config.dwpose_det_input_size,
                pose_input_size=config.dwpose_pose_input_size,
                device="cpu",
                backend=config.dwpose_backend,
            )
        except (FileNotFoundError, ImportError) as e:
            pytest.skip(f"DWPose 不可用: {e}")

    def test_keypoint_format(self, estimator):
        """关键点格式与名称数量"""
        assert estimator.keypoint_format == FORMAT_WHOLEBODY_133
        assert len(estimator.keypoint_names) == 133

    def test_empty_frame_returns_none(self, estimator):
        """纯黑帧的姿态置信度极低，应被当作未检测到人而返回 None"""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert estimator.estimate_frame(blank) is None


class TestPoseKeypoints:
    """原始关键点容器"""

    def test_slice(self):
        """按分组区间切片，越界返回 None"""
        pose = PoseKeypoints(
            keypoints=np.zeros((133, 2), dtype=np.float32),
            scores=np.ones(133, dtype=np.float32),
            keypoint_format=FORMAT_WHOLEBODY_133,
        )
        assert pose.num_keypoints == 133

        face = pose.slice(*WHOLEBODY_GROUPS["face"])
        assert face is not None
        assert face.shape == (68, 2)
        assert pose.slice(0, 200) is None


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
