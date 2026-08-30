"""
3D 姿态平滑与稳定化的单元测试。

全部为纯数据用例，不加载任何模型。关键点拓扑取自 ai3d.models.keypoints。
"""

import numpy as np
import pytest

from ai3d.config import Config
from ai3d.core.kinematic_constraints import BoneLengthConstraint
from ai3d.core.root_stabilizer import RootStabilizer, estimate_metric_scale
from ai3d.models.keypoints import (
    FORMAT_COCO_17,
    FORMAT_WHOLEBODY_133,
    get_keypoint_names,
)
from ai3d.utils.coordinate import convert_skeleton_to_3d, sample_depth
from ai3d.utils.smoothing import (
    KeypointSmoother,
    OneEuroFilter,
    TrajectorySmoother,
    is_airborne,
)

KEYPOINT_COUNT = len(get_keypoint_names(FORMAT_COCO_17))


def make_pose(num_keypoints: int = KEYPOINT_COUNT, depth: float = 3.0) -> np.ndarray:
    """构造一副规整的 3D 骨架：所有点同深度，纵向铺开"""
    points = np.zeros((num_keypoints, 3), dtype=float)
    for i in range(num_keypoints):
        points[i] = [0.1 * (i % 3), 0.1 * i, depth]
    return points


class TestOneEuroFilter:
    """自适应低通滤波器"""

    def test_first_sample_passthrough(self):
        """首帧无历史，必须原样输出"""
        f = OneEuroFilter()
        out = f.filter(np.array([1.0, 2.0]))
        assert np.allclose(out, [1.0, 2.0])

    def test_noise_is_reduced(self):
        """常值信号加噪后，输出方差应显著小于输入"""
        rng = np.random.default_rng(0)
        noisy = 5.0 + rng.normal(0, 0.5, 200)
        f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
        out = np.array([float(f.filter(np.array([v]))[0]) for v in noisy])
        assert out[50:].std() < noisy[50:].std() / 2

    def test_gap_uses_real_dt(self):
        """时间戳跳变时按实际间隔计算，不把旧值当成刚刚发生的观测"""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f.filter(np.array([0.0]), t=0.0)
        near = float(f.filter(np.array([1.0]), t=1.0 / 30.0)[0])

        f.reset()
        f.filter(np.array([0.0]), t=0.0)
        far = float(f.filter(np.array([1.0]), t=1.0)[0])
        # 间隔越大越接近新观测
        assert far > near

    def test_hold_and_reset(self):
        f = OneEuroFilter()
        assert f.hold() is None
        f.filter(np.array([3.0]))
        assert np.allclose(f.hold(), [3.0])
        f.reset()
        assert f.hold() is None


class TestKeypointSmoother:
    """逐关键点平滑"""

    def test_shape_mismatch_raises(self):
        smoother = KeypointSmoother(num_keypoints=KEYPOINT_COUNT)
        with pytest.raises(ValueError):
            smoother.smooth(np.zeros((KEYPOINT_COUNT - 1, 2)))

    def test_low_confidence_holds_previous(self):
        """低置信度的点保持上一次结果，不被噪声甩出去"""
        smoother = KeypointSmoother(
            num_keypoints=2, min_cutoff=0.1, confidence_threshold=0.3
        )
        first = smoother.smooth(
            np.array([[10.0, 10.0], [20.0, 20.0]]), np.array([0.9, 0.9])
        )
        second = smoother.smooth(
            np.array([[10.0, 10.0], [900.0, 900.0]]), np.array([0.9, 0.05])
        )
        assert np.allclose(second[1], first[1])

    def test_jitter_is_reduced(self):
        """抖动的关键点序列，逐帧位移中位数应明显下降"""
        rng = np.random.default_rng(1)
        truth = np.stack([np.array([100.0 + i, 200.0]) for i in range(60)])
        noisy = truth + rng.normal(0, 3.0, truth.shape)

        smoother = KeypointSmoother(num_keypoints=1, fps=30.0, min_cutoff=0.5)
        smoothed = np.stack([
            smoother.smooth(p.reshape(1, 2), np.array([1.0]), i / 30.0)[0]
            for i, p in enumerate(noisy)
        ])

        raw_jitter = np.median(np.linalg.norm(np.diff(noisy, axis=0), axis=1))
        new_jitter = np.median(np.linalg.norm(np.diff(smoothed, axis=0), axis=1))
        assert new_jitter < raw_jitter / 2


class TestTrajectorySmoother:
    """3D 轨迹中值滤波"""

    def test_none_frames_preserved(self):
        sequence = [make_pose(3), None, make_pose(3), make_pose(3), make_pose(3)]
        out = TrajectorySmoother(window_size=3).smooth(sequence)
        assert len(out) == len(sequence)
        assert out[1] is None

    def test_even_window_becomes_odd(self):
        assert TrajectorySmoother(window_size=4).window_size == 5

    def test_outlier_is_removed(self):
        """孤立跳变帧被邻域中值压回去"""
        sequence = [make_pose(3) for _ in range(7)]
        sequence[3] = sequence[3] + 50.0
        out = TrajectorySmoother(window_size=5).smooth(sequence)
        assert np.allclose(out[3], make_pose(3))


class TestBoneLengthConstraint:
    """骨长恒定约束"""

    def test_unfitted_is_noop(self):
        constraint = BoneLengthConstraint(FORMAT_COCO_17)
        points = make_pose()
        assert np.allclose(constraint.apply(points), points)

    def test_fit_and_reduce_error(self):
        """把被拉伸的帧拉回全序列中位骨长，平均误差必须下降"""
        base = make_pose()
        sequence = [base.copy() for _ in range(10)]
        # 单帧整体放大 1.5 倍，模拟深度误差导致的肢体爆炸拉伸
        sequence[5] = base * 1.5

        constraint = BoneLengthConstraint(FORMAT_COCO_17, iterations=6)
        assert constraint.fit(sequence) is True

        before = constraint.mean_length_error(sequence)
        fixed = [constraint.apply(p) for p in sequence]
        after = constraint.mean_length_error(fixed)
        assert after < before

    def test_fit_fails_with_too_few_frames(self):
        constraint = BoneLengthConstraint(FORMAT_COCO_17, min_samples=3)
        assert constraint.fit([make_pose(), None]) is False

    def test_low_confidence_end_moves_more(self):
        """置信度低的那端多让，可信点尽量不动"""
        base = make_pose()
        constraint = BoneLengthConstraint(FORMAT_COCO_17, iterations=1)
        constraint.fit([base.copy() for _ in range(5)])

        stretched = base.copy()
        conf = np.full(len(base), 0.9)
        bone = next(iter(constraint.target_lengths))
        a, b = bone
        stretched[b] = stretched[b] + np.array([0.0, 0.5, 0.0])
        conf[b] = 0.1

        fixed = constraint.apply(stretched, conf)
        assert np.linalg.norm(fixed[b] - stretched[b]) > np.linalg.norm(fixed[a] - stretched[a])


class TestRootStabilizer:
    """根节点稳定化（相机坐标系，Y 向下）"""

    def _pose_at(self, x: float, y: float, z: float) -> np.ndarray:
        """构造髋部在 (x, y, z)、脚在其下方 0.9m 的骨架"""
        names = get_keypoint_names(FORMAT_COCO_17)
        points = np.zeros((len(names), 3), dtype=float)
        for i, name in enumerate(names):
            if name in ("left_hip", "right_hip"):
                points[i] = [x, y, z]
            elif name in ("left_ankle", "right_ankle"):
                points[i] = [x, y + 0.9, z]
            else:
                points[i] = [x, y - 0.3, z]
        return points

    def test_missing_hip_raises(self):
        with pytest.raises(ValueError):
            RootStabilizer("no_such_format")

    def test_recenter_puts_root_near_origin(self):
        """首帧直接把髋部水平居中，人物不会一开始就在视图外"""
        stabilizer = RootStabilizer(FORMAT_COCO_17)
        out = stabilizer.stabilize(self._pose_at(12.0, 0.0, 30.0), 0)
        root = stabilizer.root_position(out)
        assert abs(root[0]) < 1e-6
        assert abs(root[2]) < 1e-6

    def test_horizontal_drift_is_clamped(self):
        """整段慢漂移后，根节点水平位置仍应留在原点附近"""
        stabilizer = RootStabilizer(FORMAT_COCO_17, fps=30.0, max_horizontal_velocity=2.0)
        last = None
        for i in range(60):
            last = stabilizer.stabilize(self._pose_at(0.5 * i, 0.0, 30.0 + 0.5 * i), i)
        root = stabilizer.root_position(last)
        assert abs(root[0]) < 1.0
        assert abs(root[2]) < 1.0

    def test_ground_fit_and_clamp(self):
        """脚部落在估计出的地面上，站立帧不穿地"""
        sequence = [self._pose_at(0.0, 0.0, 30.0) for _ in range(10)]
        stabilizer = RootStabilizer(FORMAT_COCO_17, fps=30.0)
        ground_y = stabilizer.fit_ground(sequence)
        assert ground_y == pytest.approx(0.9, abs=1e-6)

        out = stabilizer.stabilize(sequence[0], 0)
        foot_y = max(float(out[i][1]) for i in stabilizer.foot_indices)
        assert foot_y <= stabilizer.ground_tolerance + 1e-6

    def test_airborne_not_snapped(self):
        """腾空帧不做地面吸附，跳跃高度得保留"""
        sequence = [self._pose_at(0.0, 0.0, 30.0) for _ in range(10)]
        stabilizer = RootStabilizer(FORMAT_COCO_17, fps=30.0, max_vertical_velocity=100.0)
        stabilizer.fit_ground(sequence)

        stabilizer.stabilize(sequence[0], 0)
        airborne = stabilizer.stabilize(self._pose_at(0.0, -0.6, 30.0), 1)
        foot_y = max(float(airborne[i][1]) for i in stabilizer.foot_indices)
        assert foot_y < -0.1

    def test_reset_clears_state(self):
        stabilizer = RootStabilizer(FORMAT_COCO_17)
        stabilizer.stabilize(self._pose_at(1.0, 0.0, 30.0), 0)
        stabilizer.reset()
        assert stabilizer.offset_xz is None
        assert stabilizer.prev_root is None
        assert stabilizer.ground_y == 0.0


class TestIsAirborne:
    def test_no_foot_indices(self):
        assert is_airborne(make_pose(3), []) is False

    def test_standing_vs_airborne(self):
        points = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
        assert is_airborne(points, [0, 1], ground_y=0.0, threshold=0.1) is False
        assert is_airborne(points - np.array([0, 1.0, 0]), [0, 1], 0.0, 0.1) is True


class TestDepthProjection:
    """反投影：绝不允许像素坐标混进米制场景"""

    def test_sample_depth_uses_patch_median(self):
        depth_map = np.full((20, 20), 3.0, dtype=float)
        depth_map[10, 10] = 0.0  # 空洞
        assert sample_depth(depth_map, 10, 10, patch_radius=2) == pytest.approx(3.0)

    def test_sample_depth_clamps_out_of_frame(self):
        """关键点被预测到画面外时夹到边界，而不是返回 0"""
        depth_map = np.full((20, 20), 3.0, dtype=float)
        assert sample_depth(depth_map, 999, -50, patch_radius=1) == pytest.approx(3.0)

    def test_sample_depth_all_invalid(self):
        assert sample_depth(np.zeros((10, 10)), 5, 5) == 0.0

    def _skeleton_2d(self):
        return {
            'joints': [
                {'name': 'left_shoulder', 'position': [100.0, 200.0, 0.0], 'confidence': 0.9},
                {'name': 'right_shoulder', 'position': [140.0, 200.0, 0.0], 'confidence': 0.9},
                {'name': 'left_hip', 'position': [105.0, 300.0, 0.0], 'confidence': 0.9},
                {'name': 'right_hip', 'position': [135.0, 300.0, 0.0], 'confidence': 0.9},
                {'name': 'left_ankle', 'position': [110.0, 460.0, 0.0], 'confidence': 0.8},
            ],
            'frame_index': 0,
            'timestamp': 0.0,
        }

    def test_background_depth_is_replaced(self):
        """采样到背景的点改用躯干参考深度，坐标量级保持在米"""
        depth_map = np.full((480, 640), 3.0, dtype=float)
        depth_map[440:480, :] = 40.0  # 脚下是远处背景

        result = convert_skeleton_to_3d(self._skeleton_2d(), depth_map, 640, 480)
        assert result is not None
        assert result['root_depth'] == pytest.approx(3.0)
        for joint in result['joints']:
            assert joint['position'][2] == pytest.approx(3.0)
            assert abs(joint['position'][0]) < 10.0
            assert abs(joint['position'][1]) < 10.0

    def test_no_depth_returns_none_not_pixels(self):
        """整帧深度失效且无兜底时返回 None，绝不回退成像素坐标"""
        assert convert_skeleton_to_3d(
            self._skeleton_2d(), np.zeros((480, 640), dtype=float), 640, 480
        ) is None

    def test_fallback_depth_is_used(self):
        result = convert_skeleton_to_3d(
            self._skeleton_2d(), np.zeros((480, 640), dtype=float), 640, 480,
            fallback_depth=2.5
        )
        assert result is not None
        assert result['root_depth'] == pytest.approx(2.5)

    def test_empty_joints(self):
        assert convert_skeleton_to_3d({'joints': []}, np.full((10, 10), 3.0)) is None


class TestMetricScale:
    """尺度归一化：把放大数倍的反投影结果拉回真人尺度"""

    def _pose(self, torso_length: float) -> np.ndarray:
        names = get_keypoint_names(FORMAT_COCO_17)
        points = np.zeros((len(names), 3), dtype=float)
        for i, name in enumerate(names):
            if name in ("left_shoulder", "right_shoulder"):
                points[i] = [0.0, 0.0, 10.0]
            elif name in ("left_hip", "right_hip"):
                points[i] = [0.0, torso_length, 10.0]
        return points

    def test_scale_matches_target(self):
        sequence = [self._pose(2.5) for _ in range(5)]
        scale = estimate_metric_scale(sequence, FORMAT_COCO_17, target_torso_length=0.5)
        assert scale == pytest.approx(0.2)

    def test_scaled_torso_hits_target(self):
        sequence = [self._pose(2.5) for _ in range(5)]
        scale = estimate_metric_scale(sequence, FORMAT_COCO_17, target_torso_length=0.5)
        scaled = sequence[0] * scale
        names = get_keypoint_names(FORMAT_COCO_17)
        shoulder = scaled[names.index("left_shoulder")]
        hip = scaled[names.index("left_hip")]
        assert float(np.linalg.norm(hip - shoulder)) == pytest.approx(0.5)

    def test_no_valid_frame_returns_one(self):
        assert estimate_metric_scale([None, None], FORMAT_COCO_17) == 1.0

    def test_low_confidence_frames_ignored(self):
        """躯干点置信度不够时不参与定标，宁可不缩放"""
        sequence = [self._pose(2.5)]
        conf = [np.zeros(len(get_keypoint_names(FORMAT_COCO_17)))]
        assert estimate_metric_scale(
            sequence, FORMAT_COCO_17, confidences=conf, confidence_threshold=0.3
        ) == 1.0

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            estimate_metric_scale([make_pose()], "no_such_format")


class TestSmoothingConfig:
    """配置项必须能被 yaml 往返"""

    def test_defaults_present(self):
        cfg = Config().smoothing
        assert cfg.enable_2d_smoothing is True
        assert cfg.trajectory_window % 2 == 1
        assert 0.0 <= cfg.recenter_alpha <= 1.0
        assert cfg.target_torso_length > 0

    def test_yaml_roundtrip(self, tmp_path):
        config = Config()
        config.smoothing.min_cutoff = 0.42
        config.smoothing.enable_bone_constraint = False
        config.smoothing.trajectory_window = 7
        config.smoothing.target_torso_length = 0.46

        path = tmp_path / "config.yaml"
        config.save_to_yaml(str(path))
        loaded = Config.from_yaml(str(path))

        assert loaded.smoothing.min_cutoff == pytest.approx(0.42)
        assert loaded.smoothing.enable_bone_constraint is False
        assert loaded.smoothing.trajectory_window == 7
        assert loaded.smoothing.target_torso_length == pytest.approx(0.46)


class TestWholebodyTopology:
    """133 点格式同样要能构造出约束与稳定化器"""

    def test_constraint_and_stabilizer_build(self):
        constraint = BoneLengthConstraint(FORMAT_WHOLEBODY_133)
        assert len(constraint.bones) > 0

        stabilizer = RootStabilizer(FORMAT_WHOLEBODY_133)
        assert len(stabilizer.hip_indices) == 2
        # 133 点含脚部 6 点，脚踝也在内
        assert len(stabilizer.foot_indices) >= 6
