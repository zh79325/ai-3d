"""
AI 3D Skeleton Viewer - FastAPI Router Module.
可独立运行或集成到现有FastAPI项目。

Usage:
    # 独立运行
    python -m ai3d.viewer.server
    
    # 集成到其他FastAPI项目
    from ai3d.viewer import router
    app.include_router(router, prefix="/api/skeleton-viewer")
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import shutil
from pathlib import Path

# 创建路由器
router = APIRouter(prefix="/api/skeleton-viewer", tags=["skeleton-viewer"])

# 全局数据存储
_animation_data: Optional[Dict[str, Any]] = None

# 任务管理系统
_tasks: Dict[str, Dict[str, Any]] = {}

def create_task(video_filename: str) -> str:
    """创建新任务并返回任务ID"""
    import uuid
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "task_id": task_id,
        "filename": video_filename,
        "status": "pending",  # pending, processing, completed, failed
        "progress_percent": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "message": "等待处理...",
        "created_at": __import__('time').time(),
        "animation_data": None
    }
    return task_id


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态"""
    return _tasks.get(task_id)


def update_task(task_id: str, updates: Dict[str, Any]):
    """更新任务状态"""
    if task_id in _tasks:
        _tasks[task_id].update(updates)


def set_processing_status(status: Dict[str, Any]):
    """兼容旧接口，更新当前活跃任务状态

    有 task_id 的调用方已直接走 update_task，此处只兜底处理无任务上下文的场景，
    找不到处理中的任务属正常情况，不作日志输出。
    """
    for task_id, task in _tasks.items():
        if task["status"] == "processing":
            update_task(task_id, status)
            break


class SkeletonFrame(BaseModel):
    """单帧骨骼数据"""
    frame_index: int
    joints: List[dict]


class AnimationData(BaseModel):
    """完整动画数据"""
    total_frames: int
    fps: float
    width: int
    height: int
    skeletons: List[SkeletonFrame]


def load_animation_from_json(json_path: str) -> Dict[str, Any]:
    """从JSON文件加载动画数据"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Animation file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def set_animation_data(data: Dict[str, Any]):
    """设置动画数据(编程方式)"""
    global _animation_data
    _animation_data = data


def get_animation_data() -> Optional[Dict[str, Any]]:
    """获取当前动画数据"""
    return _animation_data


@router.get("/viewer")
async def get_viewer_page():
    """返回预览页面HTML"""
    viewer_dir = Path(__file__).parent
    html_path = viewer_dir / "index.html"
    
    if not html_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Viewer HTML not found"}
        )
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@router.post("/load")
async def load_animation(data: Dict[str, Any]):
    """加载动画数据"""
    global _animation_data
    
    try:
        # 验证数据结构
        required_fields = ['total_frames', 'fps', 'skeletons']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        _animation_data = {
            "total_frames": data["total_frames"],
            "fps": data["fps"],
            "width": data.get("width", 1920),
            "height": data.get("height", 1080),
            "skeletons": data["skeletons"]
        }
        
        return {
            "status": "success", 
            "frames": len(_animation_data["skeletons"]),
            "message": "Animation data loaded successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/load-from-file")
async def load_from_file(request: Dict[str, str]):
    """从JSON文件加载动画数据"""
    global _animation_data
    
    json_path = request.get('path')
    if not json_path:
        raise HTTPException(status_code=400, detail="Missing 'path' parameter")
    
    try:
        _animation_data = load_animation_from_json(json_path)
        return {
            "status": "success",
            "frames": len(_animation_data["skeletons"]),
            "message": f"Loaded from {json_path}"
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data")
async def get_animation_data_endpoint():
    """获取当前加载的动画数据"""
    if _animation_data is None:
        raise HTTPException(status_code=404, detail="No animation loaded")
    
    return _animation_data


@router.get("/frame/{frame_index}")
async def get_frame(frame_index: int):
    """获取指定帧的骨骼数据"""
    if _animation_data is None:
        raise HTTPException(status_code=404, detail="No animation loaded")
    
    skeletons = _animation_data.get("skeletons", [])
    
    if frame_index < 0 or frame_index >= len(skeletons):
        raise HTTPException(
            status_code=404, 
            detail=f"Frame index out of range (0-{len(skeletons)-1})"
        )
    
    return skeletons[frame_index]


@router.get("/info")
async def get_info():
    """获取动画信息"""
    if _animation_data is None:
        return {
            "loaded": False,
            "message": "No animation data loaded"
        }
    
    return {
        "loaded": True,
        "total_frames": _animation_data.get("total_frames", 0),
        "fps": _animation_data.get("fps", 30.0),
        "width": _animation_data.get("width", 1920),
        "height": _animation_data.get("height", 1080)
    }


@router.delete("/clear")
async def clear_data():
    """清除当前加载的数据"""
    global _animation_data
    _animation_data = None
    return {"status": "success", "message": "Animation data cleared"}


@router.get("/processing-status")
async def get_processing_status_endpoint():
    """获取视频处理进度（兼容旧接口）"""
    # 返回最近一个处理中的任务状态
    for task in _tasks.values():
        if task["status"] == "processing":
            return task
    return {"is_processing": False, "progress_percent": 0, "message": "无正在处理的任务"}


@router.post("/process-video")
async def process_video_endpoint(file: UploadFile = File(...)):
    """上传并处理视频文件，返回任务ID

    本次任务的全部产物（原视频副本、标注视频、animation_data.json）
    都写入 output/<task_id>/
    """
    import threading
    from ai3d.config import Config
    
    # 创建任务
    task_id = create_task(file.filename)
    
    # 任务输出目录（项目根 output/<task_id>/）
    config = Config()
    task_dir = config.output.task_dir(task_id)
    # 只取扩展名，避免上传文件名带路径分隔符
    suffix = Path(file.filename or "").suffix or ".mp4"
    saved_video_path = str(task_dir / f"{config.output.source_video_stem}{suffix}")
    
    try:
        with open(saved_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 更新任务状态为处理中
        update_task(task_id, {
            "status": "processing",
            "message": f"正在处理 {file.filename}...",
            "output_dir": str(task_dir),
            "video_path": saved_video_path
        })
        
        # 在后台线程中处理视频
        def process_in_background():
            try:
                from scripts.launch_viewer import process_video_to_json
                
                json_path = process_video_to_json(
                    saved_video_path, output_dir=str(task_dir), task_id=task_id
                )
                
                if json_path:
                    data = load_animation_from_json(json_path)
                    task = get_task(task_id) or {}
                    
                    update_task(task_id, {
                        "status": "completed",
                        "current_frame": data["total_frames"],
                        "total_frames": data["total_frames"],
                        "progress_percent": 100.0,
                        "message": "处理完成！",
                        "animation_data": data,
                        "json_path": json_path,
                        "output_dir": str(task_dir),
                        "video_path": saved_video_path,  # 保留原视频供前端流式传输
                        # 标注视频由 pipeline 写入 task_dir，launch_viewer 已回写该字段
                        "annotated_video_path": task.get("annotated_video_path")
                    })
                else:
                    update_task(task_id, {
                        "status": "failed",
                        "message": "处理失败：未检测到有效姿态"
                    })
                    
            except Exception as e:
                update_task(task_id, {
                    "status": "failed",
                    "message": f"处理失败: {str(e)}"
                })
                import traceback
                traceback.print_exc()
        
        bg_thread = threading.Thread(target=process_in_background, daemon=True)
        bg_thread.start()
        
        return {
            "task_id": task_id,
            "status": "processing",
            "filename": file.filename,
            "output_dir": str(task_dir),
            "message": f"Started processing {file.filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 返回时不包含完整的animation_data（可能很大）
    response = {k: v for k, v in task.items() if k != "animation_data"}
    response["has_data"] = task.get("animation_data") is not None
    return response


@router.get("/task/{task_id}/data")
async def get_task_data(task_id: str):
    """获取任务的动画数据"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is {task['status']}, not completed")
    
    animation_data = task.get("animation_data")
    if not animation_data:
        raise HTTPException(status_code=404, detail="Animation data not available")
    
    return animation_data


@router.get("/tasks")
async def list_tasks():
    """列出所有任务"""
    tasks_list = []
    for task_id, task in _tasks.items():
        task_summary = {k: v for k, v in task.items() if k != "animation_data"}
        task_summary["has_data"] = task.get("animation_data") is not None
        tasks_list.append(task_summary)
    
    # 按创建时间倒序排列
    tasks_list.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"tasks": tasks_list, "total": len(tasks_list)}


@router.get("/video/{task_id}")
async def stream_video(task_id: str):
    """流式传输任务关联的原始视频文件"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    video_path = task.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )


@router.get("/video/{task_id}/annotated")
async def stream_annotated_video(task_id: str):
    """流式传输已烧入 YOLO 检测结果（骨骼 + 实例分割）的标注视频"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    annotated_path = task.get("annotated_video_path")
    if not annotated_path or not os.path.exists(annotated_path):
        raise HTTPException(status_code=404, detail="Annotated video not found")
    
    return FileResponse(
        annotated_path,
        media_type="video/mp4",
        filename=os.path.basename(annotated_path)
    )


# 独立运行时使用的完整应用
def create_app() -> "FastAPI":
    """创建独立的FastAPI应用(包含静态文件服务)"""
    from fastapi import FastAPI
    
    app = FastAPI(
        title="AI 3D Skeleton Viewer",
        description="Web-based 3D skeleton animation viewer",
        version="0.1.0"
    )
    
    # 挂载路由器
    app.include_router(router)
    
    # 挂载静态文件
    viewer_dir = Path(__file__).parent
    static_dir = viewer_dir / "static"
    
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # 根路径重定向到viewer
    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/api/skeleton-viewer/viewer")
    
    return app


# 支持直接运行
if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    
    print("=" * 60)
    print("🚀 AI 3D Skeleton Viewer Server")
    print("=" * 60)
    print()
    print("访问地址: http://localhost:8765/api/skeleton-viewer/viewer")
    print()
    print("API端点:")
    print("  POST /api/skeleton-viewer/load          - 加载动画数据")
    print("  POST /api/skeleton-viewer/load-from-file - 从文件加载")
    print("  GET  /api/skeleton-viewer/data           - 获取数据")
    print("  GET  /api/skeleton-viewer/frame/{idx}    - 获取指定帧")
    print("  GET  /api/skeleton-viewer/info           - 获取信息")
    print("  DELETE /api/skeleton-viewer/clear        - 清除数据")
    print()
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8765)
