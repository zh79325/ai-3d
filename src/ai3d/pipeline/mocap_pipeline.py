"""
Motion Capture Pipeline.
整合视频读取、姿态估计、深度推算的完整处理流程。
"""

from typing import List, Optional
from pathlib import Path
import cv2
import numpy as np
from ai3d.core.pose_estimator import PoseEstimator
from ai3d.core.depth_estimator import DepthEstimator
from ai3d.utils.video_io import VideoReader
from ai3d.utils.coordinate import convert_skeleton_to_3d
from ai3d.models.skeleton import Skeleton
from ai3d.config import Config
from ai3d.pipeline.background_remover import BackgroundRemover
from ai3d.pipeline.video_annotator import VideoAnnotator
from ai3d.pipeline.progress import ProgressTracker


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
    
    def process_video(self, video_path: str, output_dir: Optional[str] = None,
                      progress_callback=None, tracker=None) -> dict:
        """
        处理视频文件,提取姿态数据
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录,为 None 时使用 config.output.base_dir
            progress_callback: 进度回调,签名为 callback(snapshot: dict)，
                snapshot 含 overall_percent / current_step / steps 全量步骤状态
            tracker: 可选的外部 ProgressTracker。传入时复用同一实例，
                调用方可在 pipeline 结束后继续上报后续步骤（如 export_json）
            
        Returns:
            包含处理结果的字典
        """
        print(f"开始处理视频: {video_path}")
        
        if output_dir is None:
            output_dir = self.config.output.base_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if tracker is None:
            tracker = ProgressTracker(progress_callback)
        
        # 开局就把不执行的步骤标为 skipped,使前端步骤列表长度/顺序自始至终恒定
        render_annotated = self.config.pipeline.render_annotated_video
        if not self.config.pipeline.remove_background:
            tracker.skip("bg_remove", "未启用背景移除")
        if not render_annotated:
            tracker.skip("annotate_video", "未启用标注视频")
        if self.depth_estimator is None:
            tracker.skip("depth_convert", "深度估计器不可用")
        
        # 步骤1: 读取视频
        tracker.start("video_read", 1, "正在加载视频...")
        with VideoReader(video_path) as reader:
            print(f"视频信息: {reader.frame_count}帧, {reader.fps}FPS, {reader.width}x{reader.height}")
            tracker.finish("video_read", f"视频加载完成 ({reader.frame_count}帧)")
            
            # 步骤2: 提取所有帧
            tracker.start("frame_extract", reader.frame_count or 1, "正在提取视频帧...")
            frames = reader.extract_all_frames()
            print(f"已提取 {len(frames)} 帧")
            tracker.finish("frame_extract", f"帧提取完成 ({len(frames)}帧)")
            
            # 标注视频写入器（把 YOLO 原生 plot 结果烧进原视频帧）
            annotator = None
            annotated_video_path = None
            original_frames = frames  # 标注底图始终用原始帧,不用背景移除后的帧
            annotated_bases = None
            if render_annotated and frames:
                annotated_video_path = str(
                    Path(output_dir) / self.config.pipeline.annotated_video_name
                )
                annotator = VideoAnnotator(
                    annotated_video_path,
                    fps=reader.fps,
                    width=reader.width,
                    height=reader.height,
                    draw_masks=self.config.pipeline.annotate_masks,
                    draw_pose=self.config.pipeline.annotate_pose,
                    mask_alpha=self.config.pipeline.annotate_mask_alpha,
                )
                tracker.start("annotate_video", len(frames), "等待姿态估计结果...")
            elif render_annotated:
                # 开关开着但没有帧可写
                tracker.skip("annotate_video", "无可用帧")
            
            # 步骤2.5: 背景移除（如果启用）
            if self.config.pipeline.remove_background:
                total_frames = len(frames)
                tracker.start("bg_remove", total_frames, "正在移除背景...")
                
                bg_remover = BackgroundRemover()
                frames_with_alpha = []
                # 分割掩码在这一步就合成进标注底图,避免为标注再跑一遍分割模型
                if annotator is not None and self.config.pipeline.annotate_masks:
                    annotated_bases = []
                
                for i, frame in enumerate(frames):
                    rgba_frame, seg_result = bg_remover.remove_frame(frame, return_result=True)
                    # 用 alpha 通道把背景抹黑,再转回 BGR 供姿态估计使用
                    alpha = (rgba_frame[:, :, 3:4] > 0).astype(np.uint8)
                    rgb_frame = rgba_frame[:, :, :3] * alpha
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    frames_with_alpha.append(bgr_frame)
                    
                    if annotated_bases is not None:
                        annotated_bases.append(
                            annotator.compose_frame(frame, seg_result=seg_result)
                        )
                    
                    if (i + 1) % 10 == 0 or i == total_frames - 1:
                        tracker.update("bg_remove", i + 1,
                                       f"背景移除中 {i+1}/{total_frames} 帧...")
                
                frames = frames_with_alpha
                tracker.finish("bg_remove", f"背景移除完成 ({total_frames}帧)")
                print("✅ 背景移除完成")
            
            # 步骤3: 批量姿态估计（最耗时）
            total_frames = len(frames)
            tracker.start("pose_estimate", total_frames, "正在进行姿态估计...")
            
            # 分批处理以支持进度上报
            skeletons_2d = []
            batch_size = 10
            for i in range(0, total_frames, batch_size):
                batch = frames[i:i+batch_size]
                batch_skeletons, batch_results = self.pose_estimator.estimate_batch(
                    batch, return_results=True
                )
                skeletons_2d.extend(batch_skeletons)
                
                # 关键点直接叠加到底图上并写入标注视频,坐标与原视频像素严格一致
                if annotator is not None:
                    for j, pose_result in enumerate(batch_results):
                        idx = i + j
                        base = annotated_bases[idx] if annotated_bases else original_frames[idx]
                        annotator.write(base, pose_result=pose_result)
                
                processed = min(i + batch_size, total_frames)
                tracker.update("pose_estimate", processed,
                               f"姿态估计中 {processed}/{total_frames} 帧...")
                # 标注视频与姿态估计同步推进,各自上报自己的进度,不共用一条消息
                if annotator is not None:
                    tracker.update("annotate_video", processed,
                                   f"已烧录 {processed}/{total_frames} 帧标注...")
            
            tracker.finish("pose_estimate", f"姿态估计完成 ({total_frames}帧)")
            
            if annotator is not None:
                tracker.update("annotate_video", total_frames, "正在写出视频文件...")
                annotated_video_path = annotator.close()
                annotated_bases = None  # 及时释放底图内存
                if annotated_video_path:
                    print(f"✅ 标注视频已生成: {annotated_video_path}")
                    tracker.finish("annotate_video", "标注视频已生成")
                else:
                    tracker.fail("annotate_video", "标注视频写入失败")
            
            # 步骤4: 如果有深度估计器,转换为 3D
            skeletons_3d = []
            if self.depth_estimator:
                tracker.start("depth_convert", total_frames, "正在进行 3D 坐标转换...")
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
                        tracker.update("depth_convert", i + 1,
                                       f"3D转换中 {i+1}/{total_frames} 帧...")
                
                tracker.finish("depth_convert", f"3D转换完成 ({total_frames}帧)")
                final_skeletons = skeletons_3d
                print("✅ 3D 坐标转换完成")
            else:
                final_skeletons = skeletons_2d
            
            # 步骤5: 统计有效帧
            tracker.start("validation", 1, "正在校验姿态数据...")
            valid_frames = [s for s in final_skeletons if s is not None]
            tracker.finish("validation",
                           f"检测到 {len(valid_frames)}/{len(final_skeletons)} 帧有效姿态")
            print(f"检测到 {len(valid_frames)}/{len(final_skeletons)} 帧有效姿态")
        
        result = {
            "video_path": video_path,
            "output_dir": str(output_dir),
            "annotated_video_path": annotated_video_path,
            "total_frames": len(frames),
            "valid_frames": len(valid_frames),
            "fps": reader.fps,
            "width": reader.width,
            "height": reader.height,
            "skeletons": final_skeletons,
            "use_depth": self.depth_estimator is not None
        }
        
        print(f"✅ 处理完成! 有效姿态帧: {len(valid_frames)}")
        return result
    
    def get_skeleton_at_frame(self, skeletons: List[Optional[Skeleton]], frame_index: int) -> Optional[Skeleton]:
        """获取指定帧的骨骼数据"""
        if 0 <= frame_index < len(skeletons):
            return skeletons[frame_index]
        return None
