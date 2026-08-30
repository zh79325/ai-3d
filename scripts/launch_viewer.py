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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai3d import MotionCapturePipeline


def process_video_to_json(video_path: str, output_dir: str = None, task_id: str = None) -> str:
    """处理视频并保存为JSON格式
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        task_id: 可选的任务ID，用于更新任务状态
    """
    
    print(f" 处理视频: {video_path}")
    
    # 初始化处理状态（兼容旧接口）
    from ai3d.viewer import set_processing_status, update_task
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
    
    # 定义进度回调函数，将pipeline的细粒度进度映射到前端可理解的步骤
    def progress_callback(step_name, current, total, message):
        # 计算总体进度百分比（基于步骤权重）
        step_weights = {
            "video_read": 5,      # 5%
            "frame_extract": 10,  # 10%
            "bg_remove": 20,      # 20% (背景移除)
            "pose_estimate": 40,  # 40% (最耗时)
            "depth_convert": 20,  # 20%
            "validation": 3,      # 3%
            "complete": 2         # 2%
        }
        
        weight = step_weights.get(step_name, 0)
        
        # 计算该步骤内的进度
        if total > 0:
            step_progress_val = (current / total) * weight
        else:
            step_progress_val = weight
        
        # 计算累计进度
        cumulative_weights = {
            "video_read": 0,
            "frame_extract": 5,
            "bg_remove": 15,
            "pose_estimate": 35,
            "depth_convert": 75,
            "validation": 95,
            "complete": 98
        }
        
        base_progress = cumulative_weights.get(step_name, 0)
        overall_progress = min(base_progress + step_progress_val, 100.0)
        
        # 构建多步骤进度数据结构
        step_progress_data = {}
        for sname in ["video_read", "frame_extract", "bg_remove", "pose_estimate", "depth_convert", "validation"]:
            sbase = cumulative_weights.get(sname, 0)
            sweight = step_weights.get(sname, 0)
            
            if sname == step_name:
                # 当前正在执行的步骤
                percent = int((current / total * 100)) if total > 0 else 0
                step_progress_data[sname] = {
                    "status": "processing",
                    "current": current,
                    "total": total,
                    "percent": percent,
                    "message": message
                }
            elif sbase < base_progress or (sbase == base_progress and step_name != "complete"):
                # 已完成的步骤
                step_progress_data[sname] = {
                    "status": "completed",
                    "current": 100 if sweight > 0 else 0,
                    "total": 100,
                    "percent": 100,
                    "message": ""
                }
            # 尚未开始的步骤不加入字典
        
        # 更新全局状态（兼容旧接口）
        set_processing_status({
            "current_frame": current,
            "total_frames": total,
            "progress_percent": round(overall_progress, 1),
            "message": message,
            "step_progress": step_progress_data
        })
        
        # 如果提供了task_id，更新任务状态
        if task_id:
            update_task(task_id, {
                "current_frame": current,
                "total_frames": total,
                "progress_percent": round(overall_progress, 1),
                "message": message,
                "step_progress": step_progress_data
            })
    
    # 处理视频获取3D骨骼数据，传入进度回调
    from ai3d.config import Config
    config = Config()
    config.pipeline.remove_background = True  # 启用背景移除

    # 输出目录（标注视频与 JSON 都放这里）
    if output_dir is None:
        output_dir = project_root / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = MotionCapturePipeline(config)
    result = pipeline.process_video(
        video_path,
        output_dir=str(output_dir),
        progress_callback=progress_callback
    )
    
    if not result['skeletons']:
        print("❌ 未检测到有效姿态")
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
    
    for i, skeleton in enumerate(result['skeletons']):
        # 每处理10帧更新一次进度
        if i % 10 == 0 or i == total_frames - 1:
            progress = (i + 1) / total_frames * 100
            
            # 更新全局状态（兼容旧接口）
            set_processing_status({
                "current_frame": i + 1,
                "total_frames": total_frames,
                "progress_percent": round(progress, 1),
                "message": f"正在处理第 {i+1}/{total_frames} 帧..."
            })
            
            # 如果提供了task_id，更新任务状态
            if task_id:
                update_task(task_id, {
                    "current_frame": i + 1,
                    "total_frames": total_frames,
                    "progress_percent": round(progress, 1),
                    "message": f"正在处理第 {i+1}/{total_frames} 帧..."
                })
        
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
    
    animation_data = {
        'total_frames': len(skeletons_data),
        'fps': result.get('fps', 30.0),
        'width': result.get('width', 1920),
        'height': result.get('height', 1080),
        'valid_frames': result.get('valid_frames', 0),
        'has_annotated_video': bool(result.get('annotated_video_path')),
        'skeletons': skeletons_data
    }
    
    json_path = output_dir / "animation_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(animation_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据已保存到: {json_path}")
    print(f"   总帧数: {len(skeletons_data)}")
    print(f"   FPS: {result.get('fps', 30.0)}")
    
    # 标记处理完成
    set_processing_status({
        "is_processing": False,
        "current_frame": total_frames,
        "total_frames": total_frames,
        "progress_percent": 100.0,
        "message": "处理完成！"
    })
    
    if task_id:
        update_task(task_id, {
            "status": "completed",
            "current_frame": total_frames,
            "total_frames": total_frames,
            "progress_percent": 100.0,
            "message": "处理完成！",
            "animation_data": animation_data,
            "json_path": str(json_path),
            "annotated_video_path": result.get('annotated_video_path')
        })
    
    return str(json_path)


def launch_standalone(video_path: str, port: int = 8765):
    """独立模式: 先启动服务器,再后台处理视频"""
    
    import uvicorn
    import threading
    from ai3d.viewer import create_app
    
    print(f"\n🚀 启动预览服务器...")
    print(f"   访问地址: http://localhost:{port}/api/skeleton-viewer/viewer")
    print(f"   按 Ctrl+C 停止服务器\n")
    
    # 1. 先创建并启动FastAPI应用
    app = create_app()
    
    # 2. 在后台线程中处理视频
    def process_video_background():
        try:
            json_path = process_video_to_json(video_path)
            if json_path:
                from ai3d.viewer import load_animation_from_json, set_animation_data
                data = load_animation_from_json(json_path)
                set_animation_data(data)
                print(f"\n✅ 动画数据已加载: {len(data['skeletons'])} 帧")
                print(f"   刷新浏览器即可看到3D骨骼动画\n")
        except Exception as e:
            print(f"\n❌ 视频处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 启动后台处理线程
    bg_thread = threading.Thread(target=process_video_background, daemon=True)
    bg_thread.start()
    
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
    parser.add_argument("video_path", help="输入视频路径")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口 (默认: 8765)")
    parser.add_argument("--mode", choices=["standalone", "integrate"], 
                       default="standalone", help="运行模式")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    if args.mode == "standalone":
        launch_standalone(args.video_path, args.port)
    else:
        print("集成模式需要通过Python代码调用,请参考文档")
        print("示例:")
        print("  from ai3d.viewer import integrate_to_existing_app")
        print("  integrate_to_existing_app(your_fastapi_app, 'video.mp4')")
