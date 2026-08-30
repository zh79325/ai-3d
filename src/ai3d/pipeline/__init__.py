"""Processing pipelines for motion capture and export."""

from ai3d.pipeline.background_remover import BackgroundRemover
from ai3d.pipeline.mocap_pipeline import MotionCapturePipeline
from ai3d.pipeline.video_annotator import VideoAnnotator

__all__ = ["BackgroundRemover", "MotionCapturePipeline", "VideoAnnotator"]
