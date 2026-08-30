"""
标注视频合成模块。
把姿态关键点、实例分割掩码烧进原视频帧，输出浏览器可直接播放的 H.264 MP4，
避免前端 Canvas 叠加时的坐标缩放误差。

姿态绘制支持两种输入:
- Ultralytics 的 Results 对象（YOLO 后端），直接用它的 plot()
- PoseKeypoints（DWPose 后端），按 ai3d.models.keypoints 的拓扑自绘
分割掩码仍由 YOLO-seg 的 Results.plot() 绘制。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ai3d.models.keypoints import get_bone_connections, get_groups
from ai3d.models.skeleton import PoseKeypoints

logger = logging.getLogger(__name__)

# 优先 H.264(avc1)，浏览器可直接播放；不可用时回退 mp4v
_FOURCC_CANDIDATES = ("avc1", "mp4v")

# 各部位颜色 (BGR)，未列出的分组用 _DEFAULT_COLOR
_GROUP_COLORS: Dict[str, tuple] = {
    "body": (0, 255, 0),
    "feet": (0, 255, 255),
    "face": (220, 180, 255),
    "left_hand": (255, 160, 0),
    "right_hand": (0, 160, 255),
}
_DEFAULT_COLOR = (0, 255, 0)

# 每种格式的 索引 -> 分组名 查询表，首次使用时构建并缓存
_group_lookup_cache: Dict[str, List[str]] = {}


def _group_lookup(keypoint_format: str) -> List[str]:
    """返回长度为关键点数的分组名列表，供按部位着色"""
    cached = _group_lookup_cache.get(keypoint_format)
    if cached is not None:
        return cached

    groups = get_groups(keypoint_format)
    total = max(end for _, end in groups.values())
    lookup = [""] * total
    for name, (start, end) in groups.items():
        for i in range(start, end):
            lookup[i] = name

    _group_lookup_cache[keypoint_format] = lookup
    return lookup


class VideoAnnotator:
    """逐帧接收推理结果，合成标注帧并写入 MP4"""

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        draw_masks: bool = True,
        draw_pose: bool = True,
        draw_boxes: bool = False,
        draw_labels: bool = False,
        mask_alpha: float = 0.45,
        keypoint_threshold: float = 0.3,
    ):
        """
        Args:
            output_path: 输出 MP4 路径
            fps: 输出帧率，<=0 时按 25 处理
            width: 帧宽（必须与写入帧一致）
            height: 帧高（必须与写入帧一致）
            draw_masks: 是否绘制实例分割掩码
            draw_pose: 是否绘制姿态关键点与骨架连线
            draw_boxes: 是否绘制检测框
            draw_labels: 是否绘制类别标签
            mask_alpha: 掩码图层与原帧的混合系数，1.0 为 plot 原始效果，越小越透
            keypoint_threshold: 自绘关键点时的置信度阈值，低于此值的点不绘制
        """
        self.output_path = str(output_path)
        self.fps = fps if fps and fps > 0 else 25.0
        self.width = int(width)
        self.height = int(height)
        self.draw_masks = draw_masks
        self.draw_pose = draw_pose
        self.draw_boxes = draw_boxes
        self.draw_labels = draw_labels
        self.mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
        self.keypoint_threshold = float(keypoint_threshold)
        self.frames_written = 0

        # 线宽/点径随分辨率缩放，避免高分辨率下骨架细得看不见
        short_side = max(1, min(self.width, self.height))
        self.line_thickness = max(1, round(short_side / 320))
        self.point_radius = max(2, round(short_side / 260))
        self.face_point_radius = max(1, self.point_radius - 2)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer, self.fourcc = self._open_writer()

    def _open_writer(self):
        """依次尝试候选编码器，返回可用的 VideoWriter"""
        for tag in _FOURCC_CANDIDATES:
            writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*tag),
                self.fps,
                (self.width, self.height),
            )
            if writer.isOpened():
                logger.info("标注视频编码器: %s -> %s", tag, self.output_path)
                return writer, tag
            writer.release()

        raise RuntimeError(
            f"无法创建视频写入器: {self.output_path} "
            f"(已尝试 {', '.join(_FOURCC_CANDIDATES)})"
        )

    def _draw_pose_keypoints(self, canvas: np.ndarray, pose: PoseKeypoints) -> np.ndarray:
        """按 ai3d.models.keypoints 的拓扑自绘关键点与骨骼连线（DWPose 无 Results 对象）"""
        kpts = pose.keypoints
        scores = pose.scores
        if kpts is None or scores is None or len(kpts) == 0:
            return canvas

        lookup = _group_lookup(pose.keypoint_format)
        thr = self.keypoint_threshold
        total = min(len(kpts), len(scores), len(lookup))

        # 先画连线，再画关键点，使点不被线遮盖
        for start, end in get_bone_connections(pose.keypoint_format):
            if start >= total or end >= total:
                continue
            if scores[start] < thr or scores[end] < thr:
                continue
            color = _GROUP_COLORS.get(lookup[end], _DEFAULT_COLOR)
            cv2.line(
                canvas,
                (int(kpts[start][0]), int(kpts[start][1])),
                (int(kpts[end][0]), int(kpts[end][1])),
                color,
                self.line_thickness,
                lineType=cv2.LINE_AA,
            )

        for i in range(total):
            if scores[i] < thr:
                continue
            group = lookup[i]
            radius = self.face_point_radius if group == "face" else self.point_radius
            cv2.circle(
                canvas,
                (int(kpts[i][0]), int(kpts[i][1])),
                radius,
                _GROUP_COLORS.get(group, _DEFAULT_COLOR),
                -1,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def compose_frame(
        self,
        frame_bgr: np.ndarray,
        seg_result: Optional[Any] = None,
        pose_result: Optional[Any] = None,
    ) -> np.ndarray:
        """
        在原始帧上叠加分割掩码与骨架，返回 BGR 标注帧。

        Args:
            frame_bgr: BGR 原始帧
            seg_result: yolo26n-seg 的单帧 Results 对象
            pose_result: 姿态结果，YOLO 后端为单帧 Results，DWPose 后端为 PoseKeypoints
        """
        canvas = frame_bgr

        if seg_result is not None and self.draw_masks:
            mask_layer = seg_result.plot(
                img=canvas,
                masks=True,
                boxes=self.draw_boxes,
                labels=self.draw_labels,
                conf=False,
            )
            # plot() 的掩码不透明度固定偏高，再与原帧混合一次保留人物细节
            if self.mask_alpha < 1.0:
                mask_layer = cv2.addWeighted(
                    mask_layer, self.mask_alpha, frame_bgr, 1.0 - self.mask_alpha, 0
                )
            canvas = mask_layer

        if pose_result is not None and self.draw_pose:
            if isinstance(pose_result, PoseKeypoints):
                # 自绘是原地修改，必须先拷贝，不能污染调用方的原帧
                if canvas is frame_bgr:
                    canvas = frame_bgr.copy()
                canvas = self._draw_pose_keypoints(canvas, pose_result)
            else:
                canvas = pose_result.plot(
                    img=canvas,
                    masks=False,
                    boxes=self.draw_boxes,
                    labels=self.draw_labels,
                    conf=False,
                    kpt_line=True,
                )

        # plot() 在无任何检测结果时可能原样返回输入，统一 copy 避免写入时被后续修改
        return canvas if canvas is not frame_bgr else frame_bgr.copy()

    def write(
        self,
        frame_bgr: np.ndarray,
        seg_result: Optional[Any] = None,
        pose_result: Optional[Any] = None,
    ) -> np.ndarray:
        """合成并写入一帧，返回合成后的帧"""
        annotated = self.compose_frame(frame_bgr, seg_result, pose_result)

        if annotated.shape[1] != self.width or annotated.shape[0] != self.height:
            annotated = cv2.resize(annotated, (self.width, self.height))

        self.writer.write(annotated)
        self.frames_written += 1
        return annotated

    def close(self) -> Optional[str]:
        """关闭写入器，返回输出路径（未写入任何帧时返回 None）"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.frames_written == 0:
            logger.warning("标注视频未写入任何帧: %s", self.output_path)
            return None

        logger.info("标注视频已生成: %s (%d 帧)", self.output_path, self.frames_written)
        return self.output_path

    def __enter__(self) -> "VideoAnnotator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
