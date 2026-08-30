"""
Video I/O utilities for loading and extracting frames.
"""

import cv2
import numpy as np
from typing import List, Tuple


class VideoReader:
    """视频读取器"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_frame_at(self, frame_index: int) -> np.ndarray:
        """获取指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"无法读取第 {frame_index} 帧")
        
        return frame
    
    def extract_frames(self, start: int = 0, end: int = None, step: int = 1) -> List[np.ndarray]:
        """
        提取指定范围的帧
        
        Args:
            start: 起始帧索引
            end: 结束帧索引 (不包含),默认为最后一帧
            step: 跳帧步长
            
        Returns:
            BGR 格式的图像数组列表
        """
        if end is None:
            end = self.frame_count
            
        frames = []
        current_frame = start
        
        while current_frame < end:
            frame = self.get_frame_at(current_frame)
            frames.append(frame)
            current_frame += step
            
        return frames
    
    def extract_all_frames(self) -> List[np.ndarray]:
        """提取所有帧"""
        return self.extract_frames(0, self.frame_count)
    
    def close(self):
        """释放资源"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
