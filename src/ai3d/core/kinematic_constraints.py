"""
运动学约束优化。

单目 3D 反投影里每个关键点的深度是独立采样的，同一根骨头两端很容易一深一浅，
表现为肢体在帧间被反复拉伸/压缩。这里以**整段视频的骨长中位数**为目标把每帧拉回来：
真实骨长恒定，而中位数对个别爆掉的帧不敏感。

骨骼拓扑一律取自 ai3d.models.keypoints，不在此另建索引表。
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai3d.models.keypoints import get_bone_connections

logger = logging.getLogger(__name__)


class BoneLengthConstraint:
    """
    骨长恒定约束。

    用法分两步（管线已持有全部帧，故先统计再回改）::

        constraint = BoneLengthConstraint(FORMAT_WHOLEBODY_133)
        constraint.fit(sequence, confidences)
        fixed = constraint.apply(points, conf)
    """

    def __init__(
        self,
        keypoint_format: str,
        iterations: int = 4,
        confidence_threshold: float = 0.3,
        min_samples: int = 3
    ):
        """
        Args:
            keypoint_format: 关键点格式，取值见 ai3d.models.keypoints
            iterations: 每帧的迭代次数，骨链需要多轮才能收敛
            confidence_threshold: 统计骨长时只采纳两端都达到该置信度的帧
            min_samples: 一根骨头至少需要多少个样本才认为骨长可信
        """
        self.keypoint_format = keypoint_format
        self.bones: List[Tuple[int, int]] = [
            (int(a), int(b)) for a, b in get_bone_connections(keypoint_format)
        ]
        self.iterations = max(1, int(iterations))
        self.confidence_threshold = float(confidence_threshold)
        self.min_samples = max(1, int(min_samples))
        self.target_lengths: Dict[Tuple[int, int], float] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.target_lengths)

    def fit(
        self,
        sequence: Sequence[Optional[np.ndarray]],
        confidences: Optional[Sequence[Optional[np.ndarray]]] = None
    ) -> bool:
        """
        统计每根骨头的目标长度。

        Args:
            sequence: 每帧 (K, 3) 的 3D 关键点，None 表示该帧无姿态
            confidences: 每帧 (K,) 的置信度，可为 None

        Returns:
            统计到至少一根可信骨长返回 True
        """
        samples: Dict[Tuple[int, int], List[float]] = {bone: [] for bone in self.bones}

        for idx, points in enumerate(sequence):
            if points is None:
                continue
            conf = None
            if confidences is not None and idx < len(confidences):
                conf = confidences[idx]

            for bone in self.bones:
                a, b = bone
                if a >= len(points) or b >= len(points):
                    continue
                if conf is not None and (
                    conf[a] < self.confidence_threshold or conf[b] < self.confidence_threshold
                ):
                    continue
                samples[bone].append(float(np.linalg.norm(points[b] - points[a])))

        self.target_lengths = {
            bone: float(np.median(lengths))
            for bone, lengths in samples.items()
            if len(lengths) >= self.min_samples
        }

        if not self.target_lengths:
            logger.warning("未能统计出可信骨长（有效帧过少），骨长约束将不生效")
            return False

        logger.info(
            "骨长约束已就绪: %d/%d 根骨头有目标长度",
            len(self.target_lengths), len(self.bones)
        )
        return True

    def _weights(
        self,
        confidences: Optional[np.ndarray],
        a: int,
        b: int
    ) -> Tuple[float, float]:
        """两端分摊修正量的权重：置信度低的那端多让，避免把可信点拽偏"""
        if confidences is None:
            return 0.5, 0.5
        slack_a = 1.0 - float(confidences[a])
        slack_b = 1.0 - float(confidences[b])
        total = slack_a + slack_b
        if total <= 1e-6:
            return 0.5, 0.5
        return slack_a / total, slack_b / total

    def apply(
        self,
        points: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        把一帧的骨长拉回目标值。

        Args:
            points: (K, 3) 3D 关键点
            confidences: (K,) 置信度，用于决定两端各让多少

        Returns:
            修正后的 (K, 3) 数组；未 fit 时原样返回副本
        """
        result = np.asarray(points, dtype=float).copy()
        if not self.fitted:
            return result

        for _ in range(self.iterations):
            for bone, target in self.target_lengths.items():
                a, b = bone
                if a >= len(result) or b >= len(result):
                    continue

                vec = result[b] - result[a]
                length = float(np.linalg.norm(vec))
                if length < 1e-6:
                    continue

                error = length - target
                if abs(error) < 1e-6:
                    continue

                direction = vec / length
                weight_a, weight_b = self._weights(confidences, a, b)
                result[a] += error * weight_a * direction
                result[b] -= error * weight_b * direction

        return result

    def mean_length_error(
        self,
        sequence: Sequence[Optional[np.ndarray]]
    ) -> float:
        """整段序列骨长偏离目标值的平均绝对误差(米)，用于评估效果"""
        if not self.fitted:
            return 0.0

        errors: List[float] = []
        for points in sequence:
            if points is None:
                continue
            for bone, target in self.target_lengths.items():
                a, b = bone
                if a >= len(points) or b >= len(points):
                    continue
                errors.append(abs(float(np.linalg.norm(points[b] - points[a])) - target))

        return float(np.mean(errors)) if errors else 0.0


__all__ = ["BoneLengthConstraint"]
