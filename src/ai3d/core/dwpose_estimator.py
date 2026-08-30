"""
DWPose Estimator Module.
基于 rtmlib 的 COCO-WholeBody 133 点全身姿态估计器（YOLOX 检测 + RTMW 姿态）。

模型一律从本地 ONNX 文件加载（路径见 ai3d.config.ModelConfig），运行时不联网。
对外接口与 PoseEstimator 保持一致，可由 pose_backend 配置项互换。
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from ai3d.config import resolve_model_path
from ai3d.models.keypoints import FORMAT_WHOLEBODY_133, WHOLEBODY_GROUPS, get_keypoint_names
from ai3d.models.skeleton import PoseKeypoints, Skeleton

logger = logging.getLogger(__name__)

# 各后端支持的设备，传入不支持的设备时回退到 cpu
_SUPPORTED_DEVICES = {
    "onnxruntime": ("cpu", "cuda", "rocm", "mps"),
    "opencv": ("cpu", "cuda"),
    "openvino": ("cpu", "gpu", "npu"),
}


class DWPoseEstimator:
    """
    DWPose 133 点全身姿态估计器

    关键点索引（mmpose 风格，to_openpose=False）:
    - [0:17]    身体 17 点，与 COCO/YOLO-pose 完全一致
    - [17:23]   脚部 6 点
    - [23:91]   面部 68 点
    - [91:112]  左手 21 点
    - [112:133] 右手 21 点
    """

    def __init__(
        self,
        det_model: str,
        pose_model: str,
        det_input_size: Tuple[int, int] = (640, 640),
        pose_input_size: Tuple[int, int] = (192, 256),
        device: str = "cpu",
        backend: str = "onnxruntime",
        keypoint_threshold: float = 0.3,
    ):
        """
        初始化 DWPose 估计器

        Args:
            det_model: 人体检测 ONNX 路径（相对路径按项目根解析）
            pose_model: 133 点姿态 ONNX 路径
            det_input_size: 检测模型输入尺寸 (宽, 高)，必须与模型文件匹配
            pose_input_size: 姿态模型输入尺寸 (宽, 高)，必须与模型文件匹配
            device: 'cpu', 'cuda', 'mps'
            backend: 'onnxruntime', 'opencv', 'openvino'
            keypoint_threshold: 关键点置信度阈值，低于该值的点视为不可靠

        Raises:
            FileNotFoundError: 本地模型文件缺失（不会尝试联网下载）
            ImportError: 未安装 rtmlib
        """
        det_path = resolve_model_path(det_model)
        pose_path = resolve_model_path(pose_model)

        for label, path in (("检测", det_path), ("姿态", pose_path)):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "DWPose %s模型缺失: %s\n请先运行: python scripts/download_models.py --dwpose"
                    % (label, path)
                )

        try:
            from rtmlib import Wholebody
        except ImportError as e:
            raise ImportError(
                "未安装 rtmlib，无法使用 DWPose 后端。请运行: pip install rtmlib onnxruntime"
            ) from e

        allowed = _SUPPORTED_DEVICES.get(backend, ("cpu",))
        if device not in allowed and not device.startswith("cuda"):
            logger.warning(
                "后端 %s 不支持设备 %s，已回退到 cpu（支持: %s）",
                backend, device, ", ".join(allowed)
            )
            device = "cpu"

        self.device = device
        self.backend = backend
        self.keypoint_threshold = keypoint_threshold
        self.keypoint_format = FORMAT_WHOLEBODY_133
        self.keypoint_names = get_keypoint_names(FORMAT_WHOLEBODY_133)

        # to_openpose=False 保持 mmpose 风格索引，与 ai3d.models.keypoints 的拓扑一致
        self.model = Wholebody(
            det=det_path,
            det_input_size=tuple(det_input_size),
            pose=pose_path,
            pose_input_size=tuple(pose_input_size),
            to_openpose=False,
            backend=backend,
            device=device,
        )

        logger.info(
            "DWPose 估计器已加载: %d 点, backend=%s, device=%s",
            len(self.keypoint_names), backend, device
        )

    def _pick_best_person(
        self, keypoints: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """多人时取身体关键点平均置信度最高的那个人"""
        body_start, body_end = WHOLEBODY_GROUPS["body"]
        body_scores = scores[:, body_start:body_end]
        best = int(np.argmax(body_scores.mean(axis=1)))
        return keypoints[best], scores[best]

    def estimate_frame(self, frame: np.ndarray, return_result: bool = False):
        """
        对单帧图像进行姿态估计

        Args:
            frame: BGR 格式的图像数组
            return_result: 为 True 时额外返回 PoseKeypoints（供标注绘制使用）

        Returns:
            Skeleton 对象（未检测到人时为 None）；
            return_result=True 时返回 (Skeleton, PoseKeypoints) 元组，未检测到人时后者为 None
        """
        if frame is None or frame.size == 0:
            return (None, None) if return_result else None

        keypoints, scores = self.model(frame)

        if keypoints is None or len(keypoints) == 0:
            return (None, None) if return_result else None

        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        num_persons = int(keypoints.shape[0])

        person_kpts, person_scores = self._pick_best_person(keypoints, scores)

        # YOLOX 偶尔在无人画面上给出虚检框，此时姿态置信度整体极低，按未检测到人处理
        body_start, body_end = WHOLEBODY_GROUPS["body"]
        if float(person_scores[body_start:body_end].mean()) < self.keypoint_threshold:
            return (None, None) if return_result else None

        joints = []
        for i, name in enumerate(self.keypoint_names):
            if i >= person_kpts.shape[0]:
                break
            x, y = person_kpts[i]
            # 这里只有 2D 坐标，z 由后续深度转换填充
            joints.append({
                "name": name,
                "position": [float(x), float(y), 0.0],
                "confidence": float(person_scores[i]),
            })

        skeleton = Skeleton(joints=joints)

        if not return_result:
            return skeleton

        raw = PoseKeypoints(
            keypoints=person_kpts,
            scores=person_scores,
            keypoint_format=self.keypoint_format,
            num_persons=num_persons,
        )
        return skeleton, raw

    def estimate_batch(self, frames: List[np.ndarray], return_results: bool = False):
        """
        批量处理多帧图像（rtmlib 逐帧推理，此处仅做循环封装）

        Args:
            frames: BGR 格式的图像数组列表
            return_results: 为 True 时额外返回每帧的 PoseKeypoints 列表

        Returns:
            Skeleton 对象列表；return_results=True 时返回 (skeletons, raw_results) 元组
        """
        skeletons: List[Optional[Skeleton]] = []
        raw_results: List[Optional[PoseKeypoints]] = []

        for frame in frames:
            skeleton, raw = self.estimate_frame(frame, return_result=True)
            skeletons.append(skeleton)
            raw_results.append(raw)

        return (skeletons, raw_results) if return_results else skeletons
