"""
标注视频合成模块。
直接用 Ultralytics 原生 Results.plot() 把姿态关键点、实例分割掩码烧进原视频帧，
输出浏览器可直接播放的 H.264 MP4，避免前端 Canvas 叠加时的坐标缩放误差。
"""

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 优先 H.264(avc1)，浏览器可直接播放；不可用时回退 mp4v
_FOURCC_CANDIDATES = ("avc1", "mp4v")


class VideoAnnotator:
    """逐帧接收 YOLO 推理结果，合成标注帧并写入 MP4"""

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
        self.frames_written = 0

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
            pose_result: yolo26n-pose 的单帧 Results 对象
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
