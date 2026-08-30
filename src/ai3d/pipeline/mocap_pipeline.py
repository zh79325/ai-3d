"""
Motion Capture Pipeline.
整合视频读取、姿态估计、深度推算的完整处理流程。
"""

from typing import List, Optional
import numpy as np
from ai3d.core.pose_estimator import PoseEstimator
from ai3d.core.depth_estimator import DepthEstimator
from ai3d.utils.video_io import VideoReader
from ai3d.utils.coordinate import convert_skeleton_to_3d
from ai3d.models.skeleton import Skeleton
from ai3d.config import Config
from ai3d.pipeline.background_remover import BackgroundRemover


class MotionCapturePipeline:
    """动作捕捉处理管线"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化管线
        
        Args:
            config: 配置对象,使用默认配置如果为 None
        """
        self.config = config or Config()
        self.pose_estimator = PoseEstimator(
            model_name=self.config.model.pose_model,
            device=self.config.model.device
        )
        
        # 可选的深度估计器
        self.depth_estimator = None
        if self.config.pipeline.use_depth:
            try:
                self.depth_estimator = DepthEstimator(
                    model_name=self.config.model.depth_model,
                    device=self.config.model.device
                )
                print("✅ 深度估计器已加载")
            except Exception as e:
                print(f"⚠️  深度估计器加载失败: {e}")
                print("   将仅使用 2D 姿态估计")
    
    def process_video(self, video_path: str, output_dir: str = "./output", progress_callback=None) -> dict:
        """
        处理视频文件,提取姿态数据
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            progress_callback: 进度回调函数，签名为 callback(step_name, current, total, message)
            
        Returns:
            包含处理结果的字典
        """
        print(f"开始处理视频: {video_path}")
        
        # 定义辅助函数用于上报进度
        def report_progress(step_name, current, total, message):
            if progress_callback:
                progress_callback(step_name, current, total, message)
            print(f"[{step_name}] {message}")
        
        # 步骤1: 读取视频
        report_progress("video_read", 0, 100, "正在加载视频...")
        with VideoReader(video_path) as reader:
            print(f"视频信息: {reader.frame_count}帧, {reader.fps}FPS, {reader.width}x{reader.height}")
            report_progress("video_read", 100, 100, f"视频加载完成 ({reader.frame_count}帧)")
            
            # 步骤2: 提取所有帧
            report_progress("frame_extract", 0, len(frames := reader.extract_all_frames()), "正在提取视频帧...")
            print(f"已提取 {len(frames)} 帧")
            report_progress("frame_extract", len(frames), len(frames), f"帧提取完成 ({len(frames)}帧)")
            
            # 步骤2.5: 背景移除（如果启用）
            if self.config.pipeline.remove_background:
                total_frames = len(frames)
                report_progress("bg_remove", 0, total_frames, "正在移除背景...")
                
                bg_remover = BackgroundRemover()
                frames_with_alpha = []
                
                for i, frame in enumerate(frames):
                    rgba_frame = bg_remover.remove_frame(frame)
                    # 将 RGBA 转回 BGR（保留 alpha 用于后续处理）
                    bgr_frame = cv2.cvtColor(rgba_frame[:, :, :3], cv2.COLOR_RGB2BGR)
                    frames_with_alpha.append(bgr_frame)
                    
                    if (i + 1) % 10 == 0 or i == total_frames - 1:
                        report_progress("bg_remove", i + 1, total_frames, 
                                      f"背景移除中 {i+1}/{total_frames} 帧...")
                
                frames = frames_with_alpha
                report_progress("bg_remove", total_frames, total_frames, 
                              f"背景移除完成 ({total_frames}帧)")
                print("✅ 背景移除完成")
            
            # 步骤3: 批量姿态估计（最耗时）
            total_frames = len(frames)
            report_progress("pose_estimate", 0, total_frames, "正在进行姿态估计...")
            
            # 分批处理以支持进度上报
            skeletons_2d = []
            batch_size = 10
            for i in range(0, total_frames, batch_size):
                batch = frames[i:i+batch_size]
                batch_skeletons = self.pose_estimator.estimate_batch(batch)
                skeletons_2d.extend(batch_skeletons)
                
                processed = min(i + batch_size, total_frames)
                report_progress("pose_estimate", processed, total_frames, 
                              f"姿态估计中 {processed}/{total_frames} 帧...")
            
            report_progress("pose_estimate", total_frames, total_frames, 
                          f"姿态估计完成 ({total_frames}帧)")
            
            # 步骤4: 如果有深度估计器,转换为 3D
            skeletons_3d = []
            if self.depth_estimator:
                report_progress("depth_convert", 0, total_frames, "正在进行 3D 坐标转换...")
                print("正在进行 3D 坐标转换...")
                for i, (frame, skeleton) in enumerate(zip(frames, skeletons_2d)):
                    if skeleton is not None:
                        # 估计深度图
                        depth_map = self.depth_estimator.estimate_depth(frame)
                        if depth_map is not None:
                            # 转换为 3D
                            skeleton_dict = {
                                'joints': [
                                    {
                                        'name': j.name if hasattr(j, 'name') else j['name'],
                                        'position': j.position.tolist() if hasattr(j, 'position') else j['position'],
                                        'confidence': j.confidence if hasattr(j, 'confidence') else (j.get('confidence', 1.0) if isinstance(j, dict) else 1.0)
                                    }
                                    for j in skeleton.joints
                                ],
                                'frame_index': skeleton.frame_index if hasattr(skeleton, 'frame_index') else (skeleton.get('frame_index', i) if isinstance(skeleton, dict) else i),
                                'timestamp': skeleton.timestamp if hasattr(skeleton, 'timestamp') else (skeleton.get('timestamp', 0.0) if isinstance(skeleton, dict) else 0.0)
                            }
                            
                            skeleton_3d = convert_skeleton_to_3d(
                                skeleton_dict,
                                depth_map,
                                reader.width,
                                reader.height
                            )
                            skeletons_3d.append(skeleton_3d)
                        else:
                            skeletons_3d.append(None)
                    else:
                        skeletons_3d.append(None)
                    
                    if (i + 1) % 10 == 0:
                        report_progress("depth_convert", i + 1, total_frames, 
                                      f"3D转换中 {i+1}/{total_frames} 帧...")
                
                report_progress("depth_convert", total_frames, total_frames, 
                              f"3D转换完成 ({total_frames}帧)")
                final_skeletons = skeletons_3d
                print("✅ 3D 坐标转换完成")
            else:
                final_skeletons = skeletons_2d
            
            # 步骤5: 统计有效帧
            valid_frames = [s for s in final_skeletons if s is not None]
            report_progress("validation", 100, 100, 
                          f"检测到 {len(valid_frames)}/{len(final_skeletons)} 帧有效姿态")
            print(f"检测到 {len(valid_frames)}/{len(final_skeletons)} 帧有效姿态")
        
        result = {
            "video_path": video_path,
            "total_frames": len(frames),
            "valid_frames": len(valid_frames),
            "fps": reader.fps,
            "width": reader.width,
            "height": reader.height,
            "skeletons": final_skeletons,
            "use_depth": self.depth_estimator is not None
        }
        
        report_progress("complete", 100, 100, f"处理完成! 有效姿态帧: {len(valid_frames)}")
        print(f"✅ 处理完成! 有效姿态帧: {len(valid_frames)}")
        return result
    
    def get_skeleton_at_frame(self, skeletons: List[Optional[Skeleton]], frame_index: int) -> Optional[Skeleton]:
        """获取指定帧的骨骼数据"""
        if 0 <= frame_index < len(skeletons):
            return skeletons[frame_index]
        return None
