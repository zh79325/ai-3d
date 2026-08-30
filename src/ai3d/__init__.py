"""
AI 3D Animation Generator Package.
基于 YOLO26 的单目视频 3D 动作捕捉与武器绑定工具。
"""

from pathlib import Path
from typing import Optional

from ai3d.pipeline.mocap_pipeline import MotionCapturePipeline

__version__ = "0.1.0"
__all__ = ["MotionCapturePipeline", "launch_viewer"]


def launch_viewer(video_path: Optional[str] = None, port: int = 8765):
    """
    启动Web预览器
    
    Args:
        video_path: 可选。仅在需要启动时就预处理某个视频时传入；
            为 None 时只起服务，处理由页面上传触发
        port: 服务器端口 (默认 8765)
    
    Example:
        >>> from ai3d import launch_viewer
        >>> launch_viewer()
        # 然后在浏览器访问 http://localhost:8765/api/skeleton-viewer/viewer 上传视频
    """
    # 直接导入并运行,避免 subprocess + argparse 的参数传递问题
    from scripts.launch_viewer import launch_standalone
    
    print(f"🚀 启动预览服务器...")
    if video_path:
        print(f"   预处理视频: {video_path}")
    print(f"   端口: {port}")
    print(f"   浏览器访问: http://localhost:{port}/api/skeleton-viewer/viewer\n")
    
    launch_standalone(video_path, port)
