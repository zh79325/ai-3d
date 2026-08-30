"""
视频背景移除模块 - 基于 YOLO26n-seg 实例分割
逐帧处理视频，使用实例分割掩码提取前景（人物+道具），输出透明 PNG 序列供后续姿态估计使用
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

# 禁用 Ultralytics 自动下载
os.environ.setdefault("YOLO_OFFLINE", "true")

from ultralytics import YOLO


class BackgroundRemover:
    """基于 YOLO26 实例分割的视频背景移除器"""
    
    def __init__(self, model_name: str = "yolo26n-seg.pt"):
        # 优先使用 models 目录下的本地模型（项目根目录）
        model_path = Path(__file__).parent.parent.parent.parent / "models" / model_name
        if not model_path.exists():
            model_path = model_name  # fallback to default path
        
        self.model = YOLO(str(model_path))
        print(f"✅ YOLO26 实例分割模型已加载: {model_path}")
    
    def remove_frame(self, frame: np.ndarray, return_result: bool = False):
        """
        移除单帧背景
        Args:
            frame: BGR 格式的 OpenCV 图像 (H, W, 3)
            return_result: 为 True 时额外返回 Ultralytics 原始 Results 对象（供 plot 标注使用）
        Returns:
            RGBA 格式的图像 (H, W, 4)，背景为透明；
            return_result=True 时返回 (rgba, Results) 元组
        """
        # Ultralytics 对 numpy 输入按 BGR 处理，直接传原帧，不要先转 RGB
        results = self.model(frame, verbose=False)
        result = results[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 如果没有检测到任何对象，返回原图（带全不透明 alpha）
        if result.masks is None or len(result.masks.data) == 0:
            rgba = np.concatenate([rgb_frame, np.ones((frame.shape[0], frame.shape[1], 1), dtype=np.uint8) * 255], axis=2)
            return (rgba, result) if return_result else rgba
        
        # 合并所有检测到的实例掩码（人物 + 道具）
        # masks.data shape: (N, H, W), dtype uint8, values 0 or 1
        combined_mask = np.any(result.masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255
        
        # 调整掩码到原始帧尺寸（YOLO 可能 resize 过）
        h, w = frame.shape[:2]
        if combined_mask.shape[0] != h or combined_mask.shape[1] != w:
            combined_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 构建 RGBA：保留前景像素，背景设为透明
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb_frame
        rgba[:, :, 3] = combined_mask
        
        return (rgba, result) if return_result else rgba
    
    def process_video(
        self,
        input_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> list[str]:
        """
        处理整个视频，逐帧移除背景
        Args:
            input_path: 输入视频路径
            output_dir: 输出目录（保存透明 PNG）
            progress_callback: 进度回调 (current, total, message)
        Returns:
            输出文件路径列表
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[BackgroundRemover] 视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if progress_callback:
                progress_callback(
                    frame_idx + 1, 
                    total_frames, 
                    f"背景移除中 {frame_idx + 1}/{total_frames} 帧"
                )
            
            # 移除背景
            rgba_frame = self.remove_frame(frame)
            
            # 保存为 PNG（保留透明度，cv2 需要 BGRA 通道序）
            bgra_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
            output_file = output_path / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(output_file), bgra_frame)
            frame_paths.append(str(output_file))
            
            frame_idx += 1
        
        cap.release()
        print(f"[BackgroundRemover] 完成，共处理 {len(frame_paths)} 帧")
        
        return frame_paths
