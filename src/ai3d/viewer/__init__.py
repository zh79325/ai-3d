"""
AI 3D Skeleton Viewer Module.

Usage:
    # 方式1: 独立运行
    python -m ai3d.viewer.server
    
    # 方式2: 集成到现有FastAPI项目
    from fastapi import FastAPI
    from ai3d.viewer import router
    
    app = FastAPI()
    app.include_router(router)
    
    # 方式3: 使用完整应用
    from ai3d.viewer import create_app
    app = create_app()
"""

from ai3d.viewer.server import (
    router,
    create_app,
    load_animation_from_json,
    set_animation_data,
    get_animation_data,
    set_processing_status,
    create_task,
    get_task,
    update_task
)

__all__ = [
    "router",
    "create_app", 
    "load_animation_from_json",
    "set_animation_data",
    "get_animation_data",
    "set_processing_status",
    "create_task",
    "get_task",
    "update_task"
]
