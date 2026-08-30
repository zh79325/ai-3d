"""
根节点稳定化。

解决"人物飞出视图"：单目重建的髋部位置会随深度误差整体漂移，长视频里累积成大幅偏移。

坐标约定与 ai3d.utils.coordinate 的反投影一致：相机坐标系，X 向右、**Y 向下**、Z 向前，单位米。
常见方案按 Y 向上描述的判断在这里全部反向 —— 脚在地面之下意味着 y 更大。

输出坐标空间：
- 水平（X/Z）：以髋部中心为原点，偏移量缓慢跟随，去掉慢漂移但保留跳跃/快速位移
- 垂直（Y）：以估计出的地面为 y=0。前端 3D 视图取 -y 作为高度，人物正好站在网格上
"""

import logging
from typing import List, Optional, Sequence

import numpy as np

from ai3d.models.keypoints import get_keypoint_names
from ai3d.utils.smoothing import is_airborne

logger = logging.getLogger(__name__)

# 髋部（根节点）与脚部关键点名称，索引按格式实际解析，不写死数字
_HIP_NAMES = ("left_hip", "right_hip")
_SHOULDER_NAMES = ("left_shoulder", "right_shoulder")
_FOOT_NAMES = (
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_big_toe", "right_big_toe",
)


def estimate_metric_scale(
    sequence: Sequence[Optional[np.ndarray]],
    keypoint_format: str,
    target_torso_length: float = 0.5,
    confidences: Optional[Sequence[Optional[np.ndarray]]] = None,
    confidence_threshold: float = 0.3
) -> float:
    """
    估计把重建结果摆回真实人体尺度所需的缩放系数。

    单目深度是相对值、相机内参也是假设值，反投影出的"米"通常整体放大数倍，
    于是 m/s 限速、腾空阈值这些按真人标定的参数全部失真，前端固定视角也装不下人。
    这里用**全序列躯干长度中位数**（肩中点到髋中点）对齐到成人参考值定标：
    缩放是相似变换，不改变姿态本身。长度只取 X/Y 平面分量：躯干本来近于竖直，
    把噪声最大的深度差算进去只会把躯干算长、整个人缩小。

    Args:
        sequence: 每帧 (K, 3) 的 3D 关键点，None 表示该帧无姿态
        keypoint_format: 关键点格式，取值见 ai3d.models.keypoints
        target_torso_length: 成人肩中点到髋中点的参考长度(米)，按 X/Y 投影长度计
        confidences: 每帧 (K,) 置信度，可为 None
        confidence_threshold: 只采纳四个躯干点都达到该置信度的帧

    Returns:
        缩放系数；无法估计时返回 1.0（不缩放）
    """
    names = get_keypoint_names(keypoint_format)
    hips = [i for i, n in enumerate(names) if n in _HIP_NAMES]
    shoulders = [i for i, n in enumerate(names) if n in _SHOULDER_NAMES]
    if not hips or not shoulders or target_torso_length <= 0:
        return 1.0

    torso = hips + shoulders
    lengths: List[float] = []
    for idx, points in enumerate(sequence):
        if points is None or max(torso) >= len(points):
            continue
        if confidences is not None and idx < len(confidences):
            conf = confidences[idx]
            if conf is not None and any(conf[i] < confidence_threshold for i in torso):
                continue
        length = float(np.linalg.norm(
            np.mean(points[shoulders], axis=0)[:2] - np.mean(points[hips], axis=0)[:2]
        ))
        if length > 1e-6:
            lengths.append(length)

    if not lengths:
        logger.warning("没有可信的躯干长度，跳过尺度归一化")
        return 1.0

    scale = float(target_torso_length / np.median(lengths))
    logger.info("尺度归一化: 躯干中位长 %.3f -> %.3f，缩放 %.3f",
                float(np.median(lengths)), target_torso_length, scale)
    return scale


class RootStabilizer:
    """
    根节点（髋部中心）稳定化器。

    三层处理，逐帧按序生效：
    1. 限速：水平与垂直分开限幅。垂直放宽以容纳跳跃，水平收紧以掐掉漂移
    2. 相对坐标：偏移量以 recenter_alpha 缓慢跟随髋部中心，把慢漂移吃掉
    3. 地面约束：脚不得穿透地面；腾空时不做吸附
    """

    def __init__(
        self,
        keypoint_format: str,
        fps: float = 30.0,
        max_horizontal_velocity: float = 2.0,
        max_vertical_velocity: float = 8.0,
        recenter_alpha: float = 0.9,
        ground_clamp: bool = True,
        ground_tolerance: float = 0.03,
        ground_percentile: float = 90.0,
        airborne_threshold: float = 0.1,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            keypoint_format: 关键点格式，取值见 ai3d.models.keypoints
            fps: 视频帧率，用于把限速换算成每帧位移
            max_horizontal_velocity: 髋部最大水平速度(m/s)，2.0 已够日常行走/小跑
            max_vertical_velocity: 髋部最大垂直速度(m/s)，需足够大以容纳跳跃
            recenter_alpha: 偏移量跟随系数，越大越慢越保留真实位移
            ground_clamp: 是否启用地面约束
            ground_tolerance: 允许的穿地容差(米)，只修正超出容差的部分
            ground_percentile: 地面高度取脚部 y 的该百分位（Y 向下，取高位即"最低处"）
            airborne_threshold: 腾空判定阈值(米)
            confidence_threshold: 估计地面时只采纳达到该置信度的脚部点
        """
        names = get_keypoint_names(keypoint_format)
        self.hip_indices = [i for i, n in enumerate(names) if n in _HIP_NAMES]
        self.foot_indices = [i for i, n in enumerate(names) if n in _FOOT_NAMES]

        if not self.hip_indices:
            raise ValueError("关键点格式 %s 缺少髋部关键点，无法稳定化" % keypoint_format)

        self.fps = fps if fps and fps > 0 else 30.0
        self.max_h_velocity = float(max_horizontal_velocity)
        self.max_v_velocity = float(max_vertical_velocity)
        self.recenter_alpha = float(np.clip(recenter_alpha, 0.0, 1.0))
        self.ground_clamp = bool(ground_clamp)
        self.ground_tolerance = float(ground_tolerance)
        self.ground_percentile = float(ground_percentile)
        self.airborne_threshold = float(airborne_threshold)
        self.confidence_threshold = float(confidence_threshold)

        self.ground_y = 0.0
        self.offset_xz: Optional[np.ndarray] = None
        self.prev_root: Optional[np.ndarray] = None
        self.prev_frame_idx: Optional[int] = None

    def root_position(self, points: np.ndarray) -> np.ndarray:
        """髋部中心，作为整个人的根节点"""
        return np.mean(np.asarray(points, dtype=float)[self.hip_indices], axis=0)

    def fit_ground(
        self,
        sequence: Sequence[Optional[np.ndarray]],
        confidences: Optional[Sequence[Optional[np.ndarray]]] = None
    ) -> float:
        """
        从整段序列估计地面高度。

        取脚部 y 的高百分位：Y 向下，脚踩地时 y 最大，取高位即"站立时的地面"，
        同时把个别爆掉的帧排除在外。

        Returns:
            估计出的地面 y 值；无可用脚部点时返回 0.0（不做垂直平移）
        """
        if not self.foot_indices:
            logger.warning("关键点格式缺少脚部关键点，地面高度按 0 处理")
            self.ground_y = 0.0
            return self.ground_y

        samples: List[float] = []
        for idx, points in enumerate(sequence):
            if points is None:
                continue
            conf = None
            if confidences is not None and idx < len(confidences):
                conf = confidences[idx]
            for i in self.foot_indices:
                if i >= len(points):
                    continue
                if conf is not None and conf[i] < self.confidence_threshold:
                    continue
                samples.append(float(points[i][1]))

        if not samples:
            logger.warning("没有可信的脚部关键点，地面高度按 0 处理")
            self.ground_y = 0.0
            return self.ground_y

        self.ground_y = float(np.percentile(samples, self.ground_percentile))
        logger.info("地面高度估计: y=%.3f（%d 个脚部样本）", self.ground_y, len(samples))
        return self.ground_y

    def _clamp_velocity(self, root: np.ndarray, dt: float) -> np.ndarray:
        """返回需要施加到整个骨架上的平移量，把根节点位移限制在速度上限内"""
        correction = np.zeros(3, dtype=float)
        if self.prev_root is None or dt <= 0:
            return correction

        displacement = root - self.prev_root

        horizontal = displacement[[0, 2]]
        h_norm = float(np.linalg.norm(horizontal))
        h_limit = self.max_h_velocity * dt
        if h_norm > h_limit > 0:
            clamped = horizontal * (h_limit / h_norm)
            correction[0] = clamped[0] - horizontal[0]
            correction[2] = clamped[1] - horizontal[1]

        vertical = float(displacement[1])
        v_limit = self.max_v_velocity * dt
        if abs(vertical) > v_limit:
            correction[1] = np.sign(vertical) * v_limit - vertical

        return correction

    def stabilize(self, points: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """
        稳定化一帧 3D 骨架。

        Args:
            points: (K, 3) 3D 关键点
            frame_idx: 帧索引，用于按实际间隔换算限速（中间丢帧也能正确处理）

        Returns:
            稳定化后的 (K, 3) 数组
        """
        result = np.asarray(points, dtype=float).copy()
        root = self.root_position(result)

        if self.prev_root is not None and self.prev_frame_idx is not None:
            frame_gap = max(1, frame_idx - self.prev_frame_idx)
            correction = self._clamp_velocity(root, frame_gap / self.fps)
            if np.any(correction):
                result += correction
                root = root + correction

        if self.offset_xz is None:
            self.offset_xz = -root[[0, 2]].copy()
        else:
            # 偏移量缓慢跟随根节点：吃掉慢漂移，跳跃这类快速位移仍保留
            target = -root[[0, 2]]
            self.offset_xz = (
                self.recenter_alpha * self.offset_xz + (1.0 - self.recenter_alpha) * target
            )

        result[:, 0] += self.offset_xz[0]
        result[:, 2] += self.offset_xz[1]
        result[:, 1] -= self.ground_y

        if self.ground_clamp and self.foot_indices:
            if not is_airborne(result, self.foot_indices, 0.0, self.airborne_threshold):
                penetration = max(
                    float(result[i][1]) for i in self.foot_indices if i < len(result)
                )
                if penetration > self.ground_tolerance:
                    result[:, 1] -= penetration - self.ground_tolerance

        self.prev_root = root
        self.prev_frame_idx = frame_idx
        return result

    def reset(self) -> None:
        """重置稳定化状态（换视频时调用），地面估计一并清掉"""
        self.ground_y = 0.0
        self.offset_xz = None
        self.prev_root = None
        self.prev_frame_idx = None


__all__ = ["RootStabilizer", "estimate_metric_scale"]
