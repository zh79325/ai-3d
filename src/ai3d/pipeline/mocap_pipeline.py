"""
Motion Capture Pipeline.
整合视频读取、姿态估计、深度推算的完整处理流程。
"""

from typing import List, Optional
from pathlib import Path
import cv2
import numpy as np
from ai3d.core.pose_estimator import PoseEstimator
from ai3d.core.dwpose_estimator import DWPoseEstimator
from ai3d.core.depth_estimator import DepthEstimator
from ai3d.core.kinematic_constraints import BoneLengthConstraint
from ai3d.core.root_stabilizer import RootStabilizer, estimate_metric_scale
from ai3d.utils.video_io import VideoReader
from ai3d.utils.coordinate import convert_skeleton_to_3d
from ai3d.utils.smoothing import KeypointSmoother, TrajectorySmoother
from ai3d.models.skeleton import PoseKeypoints, Skeleton
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
        self.pose_estimator = self._create_pose_estimator()
        # 关键点格式随后端而定，需随结果一起下发给前端，前端据此取拓扑
        self.keypoint_format = self.pose_estimator.keypoint_format
        
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
    
    def _create_pose_estimator(self):
        """根据 config.model.pose_backend 创建姿态估计器

        两种后端对外接口一致（estimate_batch / estimate_frame），仅关键点数量不同：
        - 'dwpose': COCO-WholeBody 133 点（身体+脚+面部+双手）
        - 'yolo':   COCO 17 点（仅身体）
        """
        backend = (self.config.model.pose_backend or "dwpose").lower()
        
        if backend == "yolo":
            print("姿态后端: YOLO-pose (17 点)")
            return PoseEstimator(
                model_name=self.config.model.pose_model,
                device=self.config.model.device
            )
        
        if backend != "dwpose":
            print(f"⚠️  未知的 pose_backend: {backend}，按 dwpose 处理")
        
        print("姿态后端: DWPose (133 点)")
        return DWPoseEstimator(
            det_model=self.config.model.dwpose_det_model,
            pose_model=self.config.model.dwpose_pose_model,
            det_input_size=self.config.model.dwpose_det_input_size,
            pose_input_size=self.config.model.dwpose_pose_input_size,
            device=self.config.model.device,
            backend=self.config.model.dwpose_backend,
            keypoint_threshold=self.config.model.dwpose_keypoint_threshold,
        )
    
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
            tracker.skip("smooth_3d", "无 3D 数据")
        
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
                    keypoint_threshold=self.config.model.dwpose_keypoint_threshold,
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
            
            # 2D 时序平滑放在这里而不是姿态估计之后：标注视频在同一个循环里写出，
            # 让它和 3D 重建吃到同一份平滑结果，两个窗口不会一个抖一个不抖
            smoother = None
            if self.config.smoothing.enable_2d_smoothing:
                smoother = KeypointSmoother(
                    num_keypoints=len(self.pose_estimator.keypoint_names),
                    fps=reader.fps,
                    min_cutoff=self.config.smoothing.min_cutoff,
                    beta=self.config.smoothing.beta,
                    d_cutoff=self.config.smoothing.d_cutoff,
                    confidence_threshold=self.config.smoothing.keypoint_confidence_threshold,
                )
            
            # 分批处理以支持进度上报
            skeletons_2d = []
            batch_size = 10
            for i in range(0, total_frames, batch_size):
                batch = frames[i:i+batch_size]
                batch_skeletons, batch_results = self.pose_estimator.estimate_batch(
                    batch, return_results=True
                )
                
                if smoother is not None:
                    for j, skeleton in enumerate(batch_skeletons):
                        self._smooth_skeleton_2d(
                            smoother, skeleton, batch_results[j], i + j, reader.fps
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
                # 上一帧的躯干深度：整帧深度都失效时用它兜底，避免退回像素坐标
                last_root_depth = None
                for i, (frame, skeleton) in enumerate(zip(frames, skeletons_2d)):
                    skeleton_3d = None
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
                                reader.height,
                                confidence_threshold=self.config.smoothing.keypoint_confidence_threshold,
                                max_depth_deviation=self.config.smoothing.max_depth_deviation,
                                patch_radius=self.config.smoothing.depth_patch_radius,
                                fallback_depth=last_root_depth,
                            )
                            if skeleton_3d is not None:
                                last_root_depth = skeleton_3d.get('root_depth', last_root_depth)
                    
                    skeletons_3d.append(skeleton_3d)
                    
                    if (i + 1) % 10 == 0:
                        tracker.update("depth_convert", i + 1,
                                       f"3D转换中 {i+1}/{total_frames} 帧...")
                
                tracker.finish("depth_convert", f"3D转换完成 ({total_frames}帧)")
                print("✅ 3D 坐标转换完成")
                
                # 步骤4.5: 3D 平滑与稳定化
                final_skeletons = self._stabilize_3d(skeletons_3d, reader.fps, tracker)
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
            "keypoint_format": self.keypoint_format,
            "skeletons": final_skeletons,
            "use_depth": self.depth_estimator is not None
        }
        
        print(f"✅ 处理完成! 有效姿态帧: {len(valid_frames)}")
        return result
    
    def _smooth_skeleton_2d(self, smoother: KeypointSmoother, skeleton, pose_result,
                            frame_idx: int, fps: float) -> None:
        """就地平滑一帧的 2D 关键点

        平滑结果同时写回 pose_result，让标注视频与 3D 重建吃同一份坐标。
        用帧号换算时间戳而不是靠调用次数递推，中间的丢帧（未检测到人）不会被当成连续帧。
        """
        if skeleton is None or not skeleton.joints:
            return

        points = np.array([j['position'][:2] for j in skeleton.joints], dtype=float)
        if points.shape[0] != smoother.num_keypoints:
            return
        confidences = np.array(
            [float(j.get('confidence', 1.0)) for j in skeleton.joints], dtype=float
        )

        timestamp = frame_idx / fps if fps and fps > 0 else None
        smoothed = smoother.smooth(points, confidences, timestamp)

        for joint, (x, y) in zip(skeleton.joints, smoothed):
            joint['position'][0] = float(x)
            joint['position'][1] = float(y)

        # YOLO 后端的 raw_result 是 Ultralytics 的 Results，不可改写，只同步 PoseKeypoints
        if isinstance(pose_result, PoseKeypoints) and pose_result.keypoints is not None:
            pose_result.keypoints = smoothed.astype(np.float32)

    def _stabilize_3d(self, skeletons_3d: List[Optional[dict]], fps: float, tracker) -> List[Optional[dict]]:
        """3D 侧稳定化: 骨长约束 -> 根节点稳定化 -> 轨迹中值滤波

        这三步都要看整段序列（骨长目标取全视频中位数、地面高度取脚部百分位），
        所以放在深度转换全部结束之后统一处理，而不是逐帧做。
        """
        cfg = self.config.smoothing
        if not (cfg.enable_bone_constraint or cfg.enable_root_stabilization
                or cfg.enable_trajectory_smoothing or cfg.normalize_scale):
            tracker.skip("smooth_3d", "3D 平滑未启用")
            return skeletons_3d

        if not any(s is not None for s in skeletons_3d):
            tracker.skip("smooth_3d", "无有效 3D 姿态")
            return skeletons_3d

        tracker.start("smooth_3d", 4, "正在归一化尺度...")

        sequence: List[Optional[np.ndarray]] = []
        confidences: List[Optional[np.ndarray]] = []
        for skeleton in skeletons_3d:
            if skeleton is None or not skeleton.get('joints'):
                sequence.append(None)
                confidences.append(None)
                continue
            joints = skeleton['joints']
            sequence.append(np.array([j['position'] for j in joints], dtype=float))
            confidences.append(
                np.array([float(j.get('confidence', 1.0)) for j in joints], dtype=float)
            )

        # 尺度归一化必须排在最前：后面的骨长误差、m/s 限速、腾空阈值都按真人尺度标定
        if cfg.normalize_scale:
            scale = estimate_metric_scale(
                sequence,
                self.keypoint_format,
                target_torso_length=cfg.target_torso_length,
                confidences=confidences,
                confidence_threshold=cfg.keypoint_confidence_threshold,
            )
            if abs(scale - 1.0) > 1e-6:
                sequence = [None if p is None else p * scale for p in sequence]
                print(f"✅ 尺度归一化: 缩放 {scale:.3f}（躯干对齐到 {cfg.target_torso_length}m）")

        tracker.update("smooth_3d", 1, "正在优化骨长...")

        if cfg.enable_bone_constraint:
            constraint = BoneLengthConstraint(
                self.keypoint_format,
                iterations=cfg.bone_constraint_iterations,
                confidence_threshold=cfg.keypoint_confidence_threshold,
            )
            if constraint.fit(sequence, confidences):
                before = constraint.mean_length_error(sequence)
                sequence = [
                    None if points is None else constraint.apply(points, confidences[i])
                    for i, points in enumerate(sequence)
                ]
                after = constraint.mean_length_error(sequence)
                print(f"✅ 骨长约束: 平均骨长误差 {before:.3f}m -> {after:.3f}m")

        tracker.update("smooth_3d", 2, "正在稳定根节点...")

        if cfg.enable_root_stabilization:
            try:
                stabilizer = RootStabilizer(
                    self.keypoint_format,
                    fps=fps,
                    max_horizontal_velocity=cfg.max_horizontal_velocity,
                    max_vertical_velocity=cfg.max_vertical_velocity,
                    recenter_alpha=cfg.recenter_alpha,
                    ground_clamp=cfg.ground_clamp,
                    ground_tolerance=cfg.ground_tolerance,
                    airborne_threshold=cfg.airborne_threshold,
                    confidence_threshold=cfg.keypoint_confidence_threshold,
                )
                ground_y = stabilizer.fit_ground(sequence, confidences)
                sequence = [
                    None if points is None else stabilizer.stabilize(points, i)
                    for i, points in enumerate(sequence)
                ]
                print(f"✅ 根节点稳定化: 地面 y={ground_y:.3f}")
            except ValueError as e:
                print(f"⚠️  根节点稳定化跳过: {e}")

        tracker.update("smooth_3d", 3, "正在做轨迹滤波...")

        if cfg.enable_trajectory_smoothing:
            sequence = TrajectorySmoother(cfg.trajectory_window).smooth(sequence)

        for skeleton, points in zip(skeletons_3d, sequence):
            if skeleton is None or points is None:
                continue
            for joint, pos in zip(skeleton['joints'], points):
                joint['position'] = [float(pos[0]), float(pos[1]), float(pos[2])]

        tracker.finish("smooth_3d", "3D 平滑完成")
        print("✅ 3D 平滑与稳定化完成")
        return skeletons_3d

    def get_skeleton_at_frame(self, skeletons: List[Optional[Skeleton]], frame_index: int) -> Optional[Skeleton]:
        """获取指定帧的骨骼数据"""
        if 0 <= frame_index < len(skeletons):
            return skeletons[frame_index]
        return None
