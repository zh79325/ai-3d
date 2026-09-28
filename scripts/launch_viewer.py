"""
Preview launcher - loads 3D skeleton data and starts the web viewer.
加载3D骨骼数据并启动Web预览器。

支持两种模式:
1. 独立模式: 启动完整的FastAPI服务器
2. 集成模式: 将router注册到现有FastAPI应用
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ai3d import MotionCapturePipeline


def resolve_output_dir(config, output_dir: str = None, task_id: str = None) -> Path:
    """确定并创建本次处理的输出目录

    优先级: 显式传入的 output_dir > config.output.task_dir(task_id) > 时间戳目录
    """
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if task_id:
        return config.output.task_dir(task_id)

    # 无任务 ID 时（如 CLI 直接跑）用时间戳目录，避免多次运行互相覆盖
    from datetime import datetime
    return config.output.task_dir(datetime.now().strftime("%Y%m%d-%H%M%S"))


def process_video_to_json(video_path: str, output_dir: str = None, task_id: str = None) -> str:
    """处理视频并保存为JSON格式
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录，为 None 时按 config.output 解析（项目根 output/<task_id>/）
        task_id: 可选的任务ID，用于更新任务状态并作为输出子目录名
    """
    
    print(f" 处理视频: {video_path}")
    
    # 初始化处理状态（兼容旧接口）
    from ai3d.viewer import set_processing_status, update_task
    from ai3d.pipeline.progress import ProgressTracker
    
    set_processing_status({
        "is_processing": True,
        "current_frame": 0,
        "total_frames": 0,
        "progress_percent": 0.0,
        "message": "正在加载视频..."
    })
    
    # 如果提供了task_id，更新任务状态
    if task_id:
        update_task(task_id, {
            "status": "processing",
            "message": "正在加载视频...",
            "current_frame": 0,
            "total_frames": 0,
            "progress_percent": 0.0
        })
    
    # 进度回调：直接透传 tracker 快照。
    # 步骤清单、权重、label 全部由 ai3d.pipeline.progress 单一定义，
    # 此处不再重复维护任何步骤表，避免与前端/pipeline 不同步
    def on_progress(snapshot):
        payload = {
            "current_frame": snapshot["current_frame"],
            "total_frames": snapshot["total_frames"],
            "progress_percent": snapshot["overall_percent"],
            "message": snapshot["message"],
            "current_step": snapshot["current_step"],
            "current_step_label": snapshot["current_step_label"],
            "steps": snapshot["steps"],
        }
        set_processing_status(payload)
        if task_id:
            update_task(task_id, payload)
    
    tracker = ProgressTracker(on_progress)
    
    # 处理视频获取3D骨骼数据，传入进度回调
    from ai3d.config import Config
    config = Config()
    config.pipeline.remove_background = True  # 启用背景移除

    # 输出目录（标注视频与 JSON 都放这里）
    output_dir = resolve_output_dir(config, output_dir=output_dir, task_id=task_id)
    print(f"📁 输出目录: {output_dir}")

    pipeline = MotionCapturePipeline(config)
    result = pipeline.process_video(
        video_path,
        output_dir=str(output_dir),
        tracker=tracker
    )
    
    if not result['skeletons']:
        print("❌ 未检测到有效姿态")
        tracker.fail("export_json", "未检测到有效姿态")
        set_processing_status({
            "is_processing": False,
            "message": "处理失败：未检测到有效姿态"
        })
        if task_id:
            update_task(task_id, {
                "status": "failed",
                "message": "处理失败：未检测到有效姿态"
            })
        return None
    
    # 准备JSON数据
    skeletons_data = []
    total_frames = len(result['skeletons'])
    # 这一段属于 export_json 步骤，进度必须报在该步骤名下，
    # 不能只改 message —— 否则会挂在上一个步骤（数据校验）的行上
    tracker.start("export_json", total_frames, "正在序列化骨骼数据...")
    
    for i, skeleton in enumerate(result['skeletons']):
        # 每处理10帧更新一次进度
        if i % 10 == 0 or i == total_frames - 1:
            tracker.update("export_json", i + 1,
                           f"正在序列化第 {i+1}/{total_frames} 帧...")
        
        if skeleton is None:
            # 空帧也占位，保证数组下标与视频帧号一一对应
            skeletons_data.append({
                'frame_index': i,
                'joints': []
            })
            continue
        
        # 转换为字典格式
        if isinstance(skeleton, dict):
            joints = skeleton.get('joints', [])
        else:
            # Skeleton对象
            joints = [
                {
                    'name': j.name if hasattr(j, 'name') else j.get('name', f'joint_{idx}'),
                    'position': j.position.tolist() if hasattr(j, 'position') else j.get('position', [0, 0, 0]),
                    'confidence': j.confidence if hasattr(j, 'confidence') else j.get('confidence', 1.0)
                }
                for idx, j in enumerate(skeleton.joints if hasattr(skeleton, 'joints') else [])
            ]
        
        skeletons_data.append({
            'frame_index': i,
            'joints': joints
        })
    
    # 关键点拓扑随数据一起下发：前端不再硬编码骨骼连接，改为读后端给的这份描述
    from ai3d.models.keypoints import describe_format, format_from_keypoint_count

    keypoint_format = result.get('keypoint_format')
    if not keypoint_format:
        # 兜底：按第一帧的关键点数量推断
        first_joints = next((s['joints'] for s in skeletons_data if s['joints']), [])
        keypoint_format = format_from_keypoint_count(len(first_joints))

    animation_data = {
        'total_frames': len(skeletons_data),
        'fps': result.get('fps', 30.0),
        'width': result.get('width', 1920),
        'height': result.get('height', 1080),
        'valid_frames': result.get('valid_frames', 0),
        'has_annotated_video': bool(result.get('annotated_video_path')),
        'skeletons': skeletons_data
    }

    if keypoint_format:
        animation_data.update(describe_format(keypoint_format))
    
    json_path = output_dir / config.output.animation_json_name
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(animation_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据已保存到: {json_path}")
    print(f"   总帧数: {len(skeletons_data)}")
    print(f"   FPS: {result.get('fps', 30.0)}")
    print(f"   关键点格式: {keypoint_format or '未知'}")
    
    tracker.finish("export_json", f"已写入 {config.output.animation_json_name}")
    final_steps = tracker.snapshot()["steps"]
    
    # 标记处理完成
    set_processing_status({
        "is_processing": False,
        "current_frame": total_frames,
        "total_frames": total_frames,
        "progress_percent": 100.0,
        "message": "处理完成！",
        "steps": final_steps
    })
    
    if task_id:
        update_task(task_id, {
            "status": "completed",
            "current_frame": total_frames,
            "total_frames": total_frames,
            "progress_percent": 100.0,
            "message": "处理完成！",
            # 保留最终步骤快照，前端完成态直接渲染真实结果（含 skipped），不再一律打绿勾
            "steps": final_steps,
            "animation_data": animation_data,
            "json_path": str(json_path),
            "output_dir": str(output_dir),
            "annotated_video_path": result.get('annotated_video_path')
        })
    
    return str(json_path)


def launch_standalone(video_path: str = None, port: int = 8765):
    """独立模式: 启动服务器

    video_path 为 None 时只起服务，视频处理由页面上传触发（默认行为）。
    仅当显式传入 video_path 时，才在后台线程里预处理该视频。
    """
    
    import uvicorn
    import threading
    from ai3d.viewer import create_app
    
    print(f"\n🚀 启动服务...")
    print(f"   功能菜单: http://localhost:{port}/")
    print(f"   🎬 骨骼查看器:   http://localhost:{port}/api/skeleton-viewer/viewer")
    print(f"   🦴 动画迁移工具: http://localhost:{port}/retarget")
    print(f"   按 Ctrl+C 停止服务器\n")
    
    # 1. 先创建并启动FastAPI应用
    app = create_app()
    
    # 2. 只有显式指定视频时才预处理；否则等页面上传
    if video_path:
        from ai3d.viewer import create_task, update_task
        
        # 建一个真实任务，进度才能出现在页面任务列表里
        task_id = create_task(os.path.basename(video_path))
        update_task(task_id, {
            "status": "processing",
            "message": "正在加载视频...",
            "video_path": video_path
        })
        
        def process_video_background():
            try:
                json_path = process_video_to_json(video_path, task_id=task_id)
                if json_path:
                    from ai3d.viewer import load_animation_from_json, set_animation_data
                    data = load_animation_from_json(json_path)
                    set_animation_data(data)
                    update_task(task_id, {"animation_data": data, "json_path": json_path})
                    print(f"\n✅ 动画数据已加载: {len(data['skeletons'])} 帧")
                    print(f"   刷新浏览器即可看到3D骨骼动画\n")
            except Exception as e:
                update_task(task_id, {"status": "failed", "message": f"处理失败: {e}"})
                print(f"\n❌ 视频处理失败: {e}")
                import traceback
                traceback.print_exc()
        
        bg_thread = threading.Thread(target=process_video_background, daemon=True)
        bg_thread.start()
    else:
        print("   等待页面上传视频...\n")
    
    # 3. 主线程运行uvicorn(阻塞)
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()


def integrate_to_existing_app(existing_app, video_path: str = None, prefix: str = "/api/skeleton-viewer"):
    """
    集成模式: 将viewer router注册到现有FastAPI应用
    
    Args:
        existing_app: 现有的FastAPI应用实例
        video_path: 可选的视频路径,如果提供会自动处理并加载数据
        prefix: API路由前缀
    
    Example:
        from fastapi import FastAPI
        from ai3d.viewer import integrate_to_existing_app
        
        app = FastAPI()
        integrate_to_existing_app(app, video_path="test.mp4")
    """
    from ai3d.viewer import router, load_animation_from_json, set_animation_data
    
    # 注册router
    existing_app.include_router(router, prefix=prefix)
    print(f"✅ Skeleton viewer router registered at {prefix}")
    
    # 如果提供了视频路径,自动处理并加载
    if video_path:
        json_path = process_video_to_json(video_path)
        if json_path:
            data = load_animation_from_json(json_path)
            set_animation_data(data)
            print(f"✅ Animation data loaded: {len(data['skeletons'])} frames")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 3D Skeleton Viewer Launcher")
    parser.add_argument("video_path", nargs="?", default=None,
                       help="可选的输入视频路径。不传则只起服务，由页面上传触发处理")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口 (默认: 8765)")
    parser.add_argument("--mode", choices=["standalone", "integrate"], 
                       default="standalone", help="运行模式")
    
    args = parser.parse_args()
    
    if args.video_path and not os.path.exists(args.video_path):
        print(f"❌ 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    if args.mode == "standalone":
        launch_standalone(args.video_path, args.port)
    else:
        print("集成模式需要通过Python代码调用,请参考文档")
        print("示例:")
        print("  from ai3d.viewer import integrate_to_existing_app")
        print("  integrate_to_existing_app(your_fastapi_app, 'video.mp4')")
