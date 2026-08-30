"""
Motion Capture Pipeline.
整合视频读取、姿态估计、深度推算的完整处理流程。
"""

from typing import List, Optional
from ai3d.core.pose_estimator import PoseEstimator
from ai3d.utils.video_io import VideoReader
from ai3d.models.skeleton import Skeleton
from ai3d.config import Config


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
    
    def process_video(self, video_path: str, output_dir: str = "./output") -> dict:
        """
        处理视频文件,提取姿态数据
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            
        Returns:
            包含处理结果的字典
        """
        print(f"开始处理视频: {video_path}")
        
        # 1. 读取视频
        with VideoReader(video_path) as reader:
            print(f"视频信息: {reader.frame_count}帧, {reader.fps}FPS, {reader.width}x{reader.height}")
            
            # 2. 提取所有帧
            frames = reader.extract_all_frames()
            print(f"已提取 {len(frames)} 帧")
            
            # 3. 批量姿态估计
            skeletons = self.pose_estimator.estimate_batch(frames)
            
            # 4. 统计有效帧
            valid_frames = [s for s in skeletons if s is not None]
            print(f"检测到 {len(valid_frames)}/{len(skeletons)} 帧有效姿态")
        
        result = {
            "video_path": video_path,
            "total_frames": len(frames),
            "valid_frames": len(valid_frames),
            "fps": reader.fps,
            "skeletons": skeletons
        }
        
        print(f"✅ 处理完成! 有效姿态帧: {len(valid_frames)}")
        return result
    
    def get_skeleton_at_frame(self, skeletons: List[Optional[Skeleton]], frame_index: int) -> Optional[Skeleton]:
        """获取指定帧的骨骼数据"""
        if 0 <= frame_index < len(skeletons):
            return skeletons[frame_index]
        return None
