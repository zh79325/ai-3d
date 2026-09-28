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


# --------------------------------------------------------------------------- #
# 功能切换菜单（骨骼查看器 / 动画迁移工具）
# --------------------------------------------------------------------------- #
_NAV_HEIGHT = 48

_NAV_STYLE = """<style id="__appnav_style">
#__appnav{position:fixed;top:0;left:0;right:0;height:48px;z-index:2147483000;
  display:flex;align-items:stretch;gap:2px;padding:0 14px;
  background:rgba(15,17,21,.97);border-bottom:1px solid #262b34;backdrop-filter:blur(10px);
  box-shadow:0 1px 0 rgba(255,255,255,.03),0 4px 18px rgba(0,0,0,.35);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}
#__appnav .brand{display:flex;align-items:center;gap:8px;margin-right:20px;
  font-size:14px;font-weight:700;color:#e6e8ec;text-decoration:none;white-space:nowrap;letter-spacing:.3px;}
#__appnav .brand:hover{color:#fff;}
#__appnav a.tab{display:flex;align-items:center;gap:6px;padding:0 16px;font-size:13px;
  color:#9aa3b2;text-decoration:none;border-bottom:2px solid transparent;transition:color .15s,background .15s;}
#__appnav a.tab:hover{color:#e6e8ec;background:rgba(255,255,255,.04);}
#__appnav a.tab.__active{color:#4da3ff;border-bottom-color:#4da3ff;font-weight:600;}
#__appnav .spacer{flex:1;}
</style>"""

# 各功能页的根布局选择器：注入导航后把主布局下移，避免被固定导航遮挡
_PAGE_ROOT = {"skeleton": ".main-layout", "retarget": "#app"}


def _nav_html(active: str) -> str:
    def cls(key: str) -> str:
        return " __active" if key == active else ""
    return (
        '<div id="__appnav">'
        '<a class="brand" href="/" title="返回功能菜单">\U0001f9ca AI 3D 工具集</a>'
        f'<a class="tab{cls("skeleton")}" href="/api/skeleton-viewer/viewer">\U0001f3ac 骨骼查看器</a>'
        f'<a class="tab{cls("retarget")}" href="/retarget">\U0001f9b4 动画迁移</a>'
        '<span class="spacer"></span>'
        '</div>'
    )


def _inject_nav(html: str, active: str) -> str:
    """注入固定在顶部的导航栏，并把页面主布局下移到导航下方。"""
    root = _PAGE_ROOT.get(active)
    fix = ""
    if root:
        fix = (f"<style>{root}{{position:fixed!important;top:{_NAV_HEIGHT}px;"
               f"left:0;right:0;bottom:0;height:auto!important;margin:0!important;}}</style>")
    snippet = _NAV_STYLE + fix + _nav_html(active)
    if "</body>" in html:
        return html.replace("</body>", snippet + "</body>", 1)
    return html + snippet


_MENU_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI 3D \u5de5\u5177\u96c6</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  body{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
    background:radial-gradient(1200px 600px at 50% -10%,#1b2740,#0e1116 60%);color:#e6e8ec;padding:72px 24px 24px;}
  .title{font-size:26px;font-weight:700;letter-spacing:1px;margin-bottom:6px;}
  .sub{color:#9aa3b2;font-size:14px;margin-bottom:34px;}
  .grid{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;}
  .card{width:280px;background:#1e2229;border:1px solid #313845;border-radius:14px;padding:24px;
    text-decoration:none;color:inherit;transition:transform .15s,border-color .15s,box-shadow .15s;display:block;}
  .card:hover{transform:translateY(-4px);border-color:#4da3ff;box-shadow:0 10px 30px rgba(0,0,0,.4);}
  .card .icon{font-size:38px;margin-bottom:12px;}
  .card h3{font-size:17px;margin-bottom:8px;}
  .card p{font-size:13px;color:#9aa3b2;line-height:1.6;}
  .card.disabled{opacity:.5;pointer-events:none;}
  .soon{margin-top:10px;font-size:12px;color:#ffb020;}
  .foot{margin-top:36px;color:#5b6472;font-size:12px;}
</style>
</head>
<body>
  <div class="title">\U0001f9ca AI 3D \u5de5\u5177\u96c6</div>
  <div class="sub">\u9009\u62e9\u4e00\u4e2a\u529f\u80fd\u5f00\u59cb</div>
  <div class="grid">
    <a class="card" href="/api/skeleton-viewer/viewer">
      <div class="icon">\U0001f3ac</div>
      <h3>\u9aa8\u9abc\u67e5\u770b\u5668</h3>
      <p>\u4e0a\u4f20\u5355\u76ee\u89c6\u9891\uff0c\u79bb\u7ebf\u63a8\u7406 3D \u9aa8\u9abc\u52a8\u4f5c\uff0c\u5b9e\u65f6\u9884\u89c8\u4e0e\u9010\u5e27\u64ad\u653e\u3002</p>
    </a>
    <a class="card __RT_CLS__" href="__RT_HREF__">
      <div class="icon">\U0001f9b4</div>
      <h3>\u52a8\u753b\u8fc1\u79fb\u5de5\u5177</h3>
      <p>\u628a\u6e90\u6a21\u578b\uff08FBX/GLB\uff09\u7684\u9aa8\u9abc\u52a8\u753b\u8fc1\u79fb\u5230\u76ee\u6807\u6a21\u578b\uff0c\u8f93\u51fa\u6210\u54c1 GLB\u3002</p>
      __RT_NOTE__
    </a>
  </div>
  <div class="foot">\u79bb\u7ebf\u8fd0\u884c \u00b7 \u6a21\u578b\u672c\u5730\u52a0\u8f7d \u00b7 \u4ea7\u7269\u5199\u5165 output/</div>
</body>
</html>"""


def _render_menu(retarget_ok: bool = True) -> str:
    href = "/retarget" if retarget_ok else "#"
    cls = "" if retarget_ok else "disabled"
    note = "" if retarget_ok else "<p class='soon'>\u26a0\ufe0f \u672a\u6302\u8f7d\uff08retarget \u4f9d\u8d56\u4e0d\u53ef\u7528\uff09</p>"
    html = (_MENU_HTML.replace("__RT_HREF__", href)
            .replace("__RT_CLS__", cls).replace("__RT_NOTE__", note))
    return _inject_nav(html, "")


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
    
    return HTMLResponse(content=_inject_nav(html_content, "skeleton"))


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
def _mount_retarget(app) -> bool:
    """把动画迁移工具（retarget）的 /v1 路由与前端页面挂到同一应用。

    失败时降级（仅骨骼查看器可用），不影响主服务启动。返回是否挂载成功。
    """
    try:
        from ai3d.retarget.server import VIEWER_DIR, router as retarget_router
        from ai3d.retarget.settings import get_settings as retarget_settings
        from ai3d.retarget.task_store import get_store as retarget_store

        retarget_settings().ensure_dirs()
        retarget_store()  # 触发建表
        app.include_router(retarget_router)

        @app.get("/retarget", include_in_schema=False)
        async def retarget_page():
            html_path = VIEWER_DIR / "index.html"
            if not html_path.exists():
                return JSONResponse(status_code=404,
                                    content={"error": "retarget viewer/index.html 未找到"})
            return HTMLResponse(_inject_nav(html_path.read_text(encoding="utf-8"), "retarget"))

        return True
    except Exception as exc:  # noqa: BLE001 - 降级：retarget 不可用时仍提供骨骼查看器
        import traceback
        print(f"\u26a0\ufe0f  动画迁移工具挂载失败（仅骨骼查看器可用）：{exc}")
        traceback.print_exc()
        return False


def create_app() -> "FastAPI":
    """创建统一的 FastAPI 应用：骨骼查看器 + 动画迁移工具，根路径为功能菜单。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="AI 3D 工具集",
        description="3D 骨骼查看器 + FBX/GLB 骨骼动画迁移工具",
        version="0.1.0"
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    # 挂载骨骼查看器路由
    app.include_router(router)

    # 挂载动画迁移工具（失败则降级）
    retarget_ok = _mount_retarget(app)

    # 挂载静态文件
    viewer_dir = Path(__file__).parent
    static_dir = viewer_dir / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # 根路径：功能切换菜单
    @app.get("/", include_in_schema=False)
    async def root():
        return HTMLResponse(_render_menu(retarget_ok))

    return app


# 支持直接运行
if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    
    print("=" * 60)
    print("🚀 AI 3D 工具集 Server")
    print("=" * 60)
    print()
    print("功能菜单: http://localhost:8765/")
    print("  🎬 骨骼查看器:   http://localhost:8765/api/skeleton-viewer/viewer")
    print("  🦴 动画迁移工具: http://localhost:8765/retarget")
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
