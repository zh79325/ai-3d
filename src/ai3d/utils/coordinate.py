"""
Coordinate transformation utilities for 2D to 3D conversion.
"""

import numpy as np
from typing import Tuple


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


def convert_skeleton_to_3d(
    skeleton_2d: dict,
    depth_map: np.ndarray,
    img_width: int = 640,
    img_height: int = 480
) -> dict:
    """
    将 2D 骨骼关键点转换为 3D 坐标。
    
    Args:
        skeleton_2d: 包含 joints 列表的字典,每个 joint 有 name, position [x, y, 0], confidence
        depth_map: 深度图数组 (H, W)
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        更新后的骨骼字典,joints 中的 position 变为 [X, Y, Z]
    """
    # 相机内参 (假设标准设置)
    fx = 600.0
    fy = 600.0
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    joints_3d = []
    
    for joint in skeleton_2d['joints']:
        x_2d, y_2d = joint['position'][0], joint['position'][1]
        confidence = joint.get('confidence', 1.0)
        
        # 从深度图获取深度值
        depth = get_depth_at_point(depth_map, x_2d, y_2d, img_width, img_height)
        
        # 转换为 3D 坐标
        if depth > 0:
            X, Y, Z = pixel_to_camera_coords(x_2d, y_2d, depth, fx, fy, cx, cy)
            position_3d = [X, Y, Z]
        else:
            # 如果深度无效,保持原样但标记为低置信度
            position_3d = joint['position']
            confidence = max(confidence * 0.5, 0.1)
        
        joints_3d.append({
            'name': joint['name'],
            'position': position_3d,
            'confidence': confidence
        })
    
    return {
        'joints': joints_3d,
        'frame_index': skeleton_2d.get('frame_index', 0),
        'timestamp': skeleton_2d.get('timestamp', 0.0)
    }


def get_depth_at_point(
    depth_map: np.ndarray,
    x: float,
    y: float,
    img_width: int,
    img_height: int
) -> float:
    """
    从深度图中安全地获取指定位置的深度值。
    
    Args:
        depth_map: 深度图数组 (H, W)
        x: 像素 x 坐标
        y: 像素 y 坐标
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        深度值(米),无效则返回 0.0
    """
    h, w = depth_map.shape
    
    # 边界检查
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    
    # 取整
    x_int = int(round(x))
    y_int = int(round(y))
    
    # 再次边界检查
    if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
        return 0.0
    
    depth_value = depth_map[y_int, x_int]
    
    # 过滤无效值
    if depth_value <= 0 or np.isnan(depth_value) or np.isinf(depth_value):
        return 0.0
    
    return float(depth_value)
