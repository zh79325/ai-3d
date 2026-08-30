"""
2D -> 3D 坐标反投影。

坐标约定：输出为相机坐标系，X 向右、**Y 向下**、Z 向前（深度），单位米。
下游的骨长约束与根节点稳定化沿用同一约定，见 ai3d.core.root_stabilizer。

深度融合的两条硬规则（违反其一就会出现人物飞出视图）：
1. 逐关键点的深度必须锚定到躯干深度附近，单点采样很容易落到背景上；
2. 深度取不到时只能用参考深度兜底，**绝不能退回像素坐标** —— 像素值混进米制场景
   会让关键点跑到几百米外。
"""

from typing import List, Optional, Tuple

import numpy as np

# 躯干关键点名称，用于估计整个人的参考深度（两个格式的前 17 点都含这些名字）
TORSO_JOINT_NAMES = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def pixel_to_camera_coords(
    x: float, 
    y: float, 
    depth: float,
    fx: float = 600.0,
    fy: float = 600.0,
    cx: float = 320.0,
    cy: float = 240.0
) -> Tuple[float, float, float]:
    """
    将像素坐标 + 深度值转换为相机坐标系下的 3D 坐标。
    
    Args:
        x: 像素 x 坐标
        y: 像素 y 坐标
        depth: 深度值(米)
        fx: 相机焦距 x (默认 600)
        fy: 相机焦距 y (默认 600)
        cx: 主点 x (图像中心,默认 320)
        cy: 主点 y (图像中心,默认 240)
        
    Returns:
        (X, Y, Z) 相机坐标系下的 3D 坐标(米)
    """
    if depth <= 0:
        return (0.0, 0.0, 0.0)
    
    # 针孔相机模型反投影
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    
    return (float(X), float(Y), float(Z))


def sample_depth(
    depth_map: np.ndarray,
    x: float,
    y: float,
    patch_radius: int = 2
) -> float:
    """
    取 (x, y) 邻域内有效深度的中值。

    单像素采样会命中身体边缘或深度图空洞，取小块中值稳得多。
    坐标越界时夹到图像边界 —— 姿态模型会把关键点预测到画面外，
    此时返回边界深度而不是 0，避免调用方误判为"无深度"。

    Args:
        depth_map: 深度图 (H, W)，单位米
        x: 像素 x 坐标
        y: 像素 y 坐标
        patch_radius: 邻域半径(像素)，0 表示只取单点

    Returns:
        深度值(米)，邻域内没有有效值时返回 0.0
    """
    h, w = depth_map.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    x_int = int(round(min(max(x, 0), w - 1)))
    y_int = int(round(min(max(y, 0), h - 1)))
    radius = max(0, int(patch_radius))

    patch = depth_map[
        max(0, y_int - radius):min(h, y_int + radius + 1),
        max(0, x_int - radius):min(w, x_int + radius + 1),
    ]
    if patch.size == 0:
        return 0.0

    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 0.0

    return float(np.median(valid))


def _reference_depth(
    depths: np.ndarray,
    confidences: np.ndarray,
    torso_mask: np.ndarray,
    confidence_threshold: float
) -> float:
    """
    估计整个人的参考深度：优先取高置信度躯干点的中值。

    躯干点面积大、遮挡少，深度最可信；躯干不可用时依次放宽到
    全部高置信度关键点、全部有效关键点。
    """
    valid = depths > 0

    for mask in (
        valid & torso_mask & (confidences >= confidence_threshold),
        valid & torso_mask,
        valid & (confidences >= confidence_threshold),
        valid,
    ):
        if np.any(mask):
            return float(np.median(depths[mask]))

    return 0.0


def convert_skeleton_to_3d(
    skeleton_2d: dict,
    depth_map: np.ndarray,
    img_width: int = 640,
    img_height: int = 480,
    confidence_threshold: float = 0.3,
    max_depth_deviation: float = 0.8,
    patch_radius: int = 2,
    fallback_depth: Optional[float] = None
) -> Optional[dict]:
    """
    将 2D 骨骼关键点转换为 3D 坐标。
    
    Args:
        skeleton_2d: 包含 joints 列表的字典,每个 joint 有 name, position [x, y, 0], confidence
        depth_map: 深度图数组 (H, W)，尺寸需与原始帧一致
        img_width: 图像宽度
        img_height: 图像高度
        confidence_threshold: 参考深度只采纳置信度不低于该值的关键点
        max_depth_deviation: 单个关键点允许偏离参考深度的最大值(米)，
            超出视为采样到了背景，改用参考深度
        patch_radius: 深度采样邻域半径(像素)
        fallback_depth: 本帧躯干深度全部失效时使用的深度(通常传上一帧的值)
        
    Returns:
        更新后的骨骼字典,joints 中的 position 变为 [X, Y, Z]，
        并附带本帧的参考深度 root_depth；连参考深度都取不到时返回 None
    """
    # 相机内参 (假设标准设置)
    fx = 600.0
    fy = 600.0
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    joints = skeleton_2d.get('joints') or []
    if not joints:
        return None

    depths = np.array([
        sample_depth(depth_map, j['position'][0], j['position'][1], patch_radius)
        for j in joints
    ], dtype=float)
    confidences = np.array([float(j.get('confidence', 1.0)) for j in joints], dtype=float)
    torso_mask = np.array([j.get('name') in TORSO_JOINT_NAMES for j in joints], dtype=bool)

    reference = _reference_depth(depths, confidences, torso_mask, confidence_threshold)
    if reference <= 0 and fallback_depth is not None and fallback_depth > 0:
        reference = float(fallback_depth)
    if reference <= 0:
        return None

    joints_3d: List[dict] = []
    for i, joint in enumerate(joints):
        x_2d, y_2d = joint['position'][0], joint['position'][1]
        depth = depths[i]

        # 落到背景/空洞上的点用参考深度顶上，保持整个人在同一深度层附近
        if depth <= 0 or abs(depth - reference) > max_depth_deviation:
            depth = reference

        X, Y, Z = pixel_to_camera_coords(x_2d, y_2d, depth, fx, fy, cx, cy)
        joints_3d.append({
            'name': joint['name'],
            'position': [X, Y, Z],
            'confidence': float(joint.get('confidence', 1.0))
        })
    
    return {
        'joints': joints_3d,
        'frame_index': skeleton_2d.get('frame_index', 0),
        'timestamp': skeleton_2d.get('timestamp', 0.0),
        'root_depth': reference
    }
