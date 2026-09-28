"""封装 ai3d.core 的 DWPose/YOLO，对前端回传的多视角图逐张推理出 2D 关键点 + 置信度。

- 模型本地离线加载（路径取自 ``ai3d.config.ModelConfig``，经 settings.model 暴露）。
- 估计器懒加载并缓存（DWPose ONNX 约 315MB，仅在 enable_pose_ai 时载入一次）。
- 输出统一为 ``{view_id: {keypoint_name: {"uv": [u, v], "conf": c}}}``，
  关键点命名遵循 ``ai3d.models.keypoints``（DWPose 前 23 点与 COCO17+feet 同名）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .settings import RetargetSettings, get_settings

logger = logging.getLogger(__name__)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

_estimator = None
_estimator_key: Optional[tuple] = None


def load_estimator(settings: Optional[RetargetSettings] = None, reload: bool = False):
    """按配置加载并缓存姿态估计器（'dwpose' 133 点 / 'yolo' COCO17 点）。"""
    global _estimator, _estimator_key
    settings = settings or get_settings()
    m = settings.model
    backend = (getattr(m, "pose_backend", "dwpose") or "dwpose").lower()
    key = (backend, getattr(m, "device", "cpu"))
    if _estimator is not None and _estimator_key == key and not reload:
        return _estimator
    if backend == "yolo":
        from ai3d.core import PoseEstimator
        logger.info("加载 YOLO 姿态估计器：%s", m.pose_model)
        _estimator = PoseEstimator(model_name=m.pose_model, device=m.device)
    else:
        from ai3d.core import DWPoseEstimator
        logger.info("加载 DWPose 估计器：det=%s pose=%s",
                    m.dwpose_det_model, m.dwpose_pose_model)
        _estimator = DWPoseEstimator(
            det_model=m.dwpose_det_model,
            pose_model=m.dwpose_pose_model,
            det_input_size=tuple(m.dwpose_det_input_size),
            pose_input_size=tuple(m.dwpose_pose_input_size),
            device=m.device,
            backend=m.dwpose_backend,
            keypoint_threshold=m.dwpose_keypoint_threshold,
        )
    _estimator_key = key
    return _estimator


def _read_image(path: Path):
    """读图为 BGR ndarray（用 cv2，失败回退 imageio/numpy）。"""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def infer_image(image_path: str | Path, estimator) -> Dict[str, Dict]:
    """单张图 → {keypoint_name: {"uv": [u, v], "conf": c}}；未检测到人返回 {}。"""
    img = _read_image(Path(image_path))
    if img is None:
        logger.warning("无法读取视角图：%s", image_path)
        return {}
    skeleton = estimator.estimate_frame(img)
    out: Dict[str, Dict] = {}
    if skeleton is None:
        return out
    for j in getattr(skeleton, "joints", []) or []:
        name = j["name"] if isinstance(j, dict) else j.name
        pos = j["position"] if isinstance(j, dict) else j.position
        conf = j["confidence"] if isinstance(j, dict) else j.confidence
        out[name] = {"uv": [float(pos[0]), float(pos[1])], "conf": float(conf)}
    return out


def list_view_images(views_dir: str | Path) -> List[Path]:
    """列出视角图（按文件名排序），文件名主干即 view_id。"""
    d = Path(views_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def infer_views(views_dir: str | Path, estimator=None,
                settings: Optional[RetargetSettings] = None) -> Dict[str, Dict[str, Dict]]:
    """对目录下所有视角图推理 → {view_id: {kp_name: {...}}}。"""
    estimator = estimator or load_estimator(settings)
    paths = list_view_images(views_dir)
    detections: Dict[str, Dict[str, Dict]] = {}
    for p in paths:
        vid = p.stem
        det = infer_image(p, estimator)
        detections[vid] = det
        logger.info("视角 %s 推理到 %d 个关键点", vid, len(det))
    return detections
