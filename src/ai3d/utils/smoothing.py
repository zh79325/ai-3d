"""
时序平滑滤波。

3D 姿态稳定链路的第一环：先压掉 2D 关键点的帧间抖动，再由
ai3d.core.kinematic_constraints（骨长约束）与 ai3d.core.root_stabilizer（根节点稳定）
处理 3D 侧的约束。

滤波器都不依赖关键点数量与格式，17 点与 133 点通用。
"""

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


class OneEuroFilter:
    """
    OneEuro 自适应低通滤波器。

    截止频率随运动速度自动抬升：静止时强平滑压噪，快速运动时放开以保留细节，
    因此不需要为"慢动作/快动作"分别调参。

    与常见实现的区别：这里用**实际时间差**计算平滑系数，
    因此中间丢帧（未检测到人）不会让滤波器把旧值当成刚刚发生的观测。
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        default_dt: float = 1.0 / 30.0
    ):
        """
        Args:
            min_cutoff: 最小截止频率(Hz)，越小越平滑
            beta: 速度系数，越大对快速运动响应越快
            d_cutoff: 速度估计的截止频率
            default_dt: 未提供时间戳时使用的帧间隔(秒)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.default_dt = float(default_dt) if default_dt > 0 else 1.0 / 30.0

        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        """一阶低通的平滑系数，cutoff 可以是标量或逐分量的数组"""
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, _EPS))
        return 1.0 / (1.0 + tau / max(dt, _EPS))

    def filter(self, x, t: Optional[float] = None) -> np.ndarray:
        """
        过滤一次观测。

        Args:
            x: 当前观测，标量或任意形状的数组
            t: 时间戳(秒)，None 时按 default_dt 递推

        Returns:
            平滑后的值（形状与 x 一致）
        """
        x = np.asarray(x, dtype=float)

        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        if t is None or self.t_prev is None:
            dt = self.default_dt
        else:
            dt = t - self.t_prev
            if dt <= 0:
                dt = self.default_dt

        # 先平滑速度，再用速度决定位置的截止频率
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        # 首帧没给时间戳时 t_prev 一直是 None，此后按 default_dt 递推，不去和 None 相加
        if t is not None:
            self.t_prev = t
        elif self.t_prev is not None:
            self.t_prev = self.t_prev + dt

        return x_hat.copy()

    def hold(self) -> Optional[np.ndarray]:
        """返回上一次的输出，尚无历史时返回 None（不推进内部状态）"""
        return None if self.x_prev is None else self.x_prev.copy()

    def reset(self) -> None:
        """重置滤波器状态"""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class KeypointSmoother:
    """
    逐关键点独立平滑。

    每个关键点一套 OneEuroFilter；置信度低于阈值的点不喂进滤波器，
    而是沿用上一次的结果 —— 被遮挡的点原地停住远好于被噪声甩出去。
    """

    def __init__(
        self,
        num_keypoints: int,
        fps: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        confidence_threshold: float = 0.0
    ):
        """
        Args:
            num_keypoints: 关键点数量
            fps: 视频帧率，用于未提供时间戳时的默认帧间隔
            min_cutoff: 见 OneEuroFilter
            beta: 见 OneEuroFilter
            d_cutoff: 见 OneEuroFilter
            confidence_threshold: 低于该置信度的关键点保持上一次结果
        """
        self.num_keypoints = int(num_keypoints)
        self.confidence_threshold = float(confidence_threshold)
        default_dt = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0
        self.filters = [
            OneEuroFilter(
                min_cutoff=min_cutoff,
                beta=beta,
                d_cutoff=d_cutoff,
                default_dt=default_dt,
            )
            for _ in range(self.num_keypoints)
        ]

    def smooth(
        self,
        points: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        平滑一帧关键点。

        Args:
            points: (K, D) 关键点坐标，D 为 2 或 3
            confidences: (K,) 每个关键点的置信度，None 表示全部采纳
            timestamp: 该帧的时间戳(秒)

        Returns:
            平滑后的 (K, D) 数组
        """
        points = np.asarray(points, dtype=float)
        if points.shape[0] != self.num_keypoints:
            raise ValueError(
                "关键点数量不匹配: 期望 %d，实际 %d" % (self.num_keypoints, points.shape[0])
            )

        smoothed = points.copy()
        for i in range(self.num_keypoints):
            if confidences is not None and confidences[i] < self.confidence_threshold:
                held = self.filters[i].hold()
                if held is not None:
                    smoothed[i] = held
                    continue
            smoothed[i] = self.filters[i].filter(points[i], t=timestamp)

        return smoothed

    def reset(self) -> None:
        """重置所有关键点的滤波器"""
        for f in self.filters:
            f.reset()


class TrajectorySmoother:
    """
    3D 轨迹中值滤波，用于压掉残留的孤立跳变。

    管线是离线批处理，这里用**居中**窗口，不引入因果滤波那 (window-1)/2 帧的延迟。
    无姿态的帧（None）不参与取值，也不会被填补。
    """

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 窗口大小，偶数会自动加 1 变成奇数
        """
        size = max(1, int(window_size))
        if size % 2 == 0:
            size += 1
        self.window_size = size

    def smooth(
        self,
        sequence: Sequence[Optional[np.ndarray]]
    ) -> List[Optional[np.ndarray]]:
        """
        Args:
            sequence: 每帧 (K, 3) 的数组，None 表示该帧无姿态

        Returns:
            与输入等长的列表，None 位置保持 None
        """
        frames = list(sequence)
        if self.window_size <= 1:
            return frames

        half = self.window_size // 2
        valid_indices = [i for i, p in enumerate(frames) if p is not None]
        if len(valid_indices) < 3:
            return frames

        result: List[Optional[np.ndarray]] = [None] * len(frames)
        for i in valid_indices:
            window = [
                frames[j] for j in range(max(0, i - half), min(len(frames), i + half + 1))
                if frames[j] is not None and frames[j].shape == frames[i].shape
            ]
            if len(window) < 3:
                result[i] = frames[i].copy()
                continue
            result[i] = np.median(np.stack(window), axis=0)

        return result


def is_airborne(
    points: np.ndarray,
    foot_indices: Sequence[int],
    ground_y: float = 0.0,
    threshold: float = 0.1
) -> bool:
    """
    判断人物是否腾空。

    相机坐标系 Y 轴向下，脚点的 y 越小离地越高，因此取**最低那只脚**（y 最大）判断：
    连最低的脚都明显高于地面才算腾空。

    Args:
        points: (K, 3) 3D 关键点
        foot_indices: 脚部关键点索引
        ground_y: 地面所在的 y 值
        threshold: 判定阈值(米)

    Returns:
        腾空返回 True；无可用脚部点时返回 False
    """
    ys = [float(points[i][1]) for i in foot_indices if 0 <= i < len(points)]
    if not ys:
        return False
    return max(ys) < ground_y - threshold


__all__ = [
    "OneEuroFilter",
    "KeypointSmoother",
    "TrajectorySmoother",
    "is_airborne",
]
