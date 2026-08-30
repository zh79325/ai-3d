"""
Test server launcher - starts FastAPI viewer without requiring a video file.
用于测试前端UI的轻量级服务器启动脚本。
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from ai3d.viewer.server import router

# 创建独立的FastAPI应用
app = FastAPI(title="AI 3D Skeleton Viewer (Test)")

# 注册router
app.include_router(router)

# 提供index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = Path(__file__).parent.parent / "src" / "ai3d" / "viewer" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Index not found</h1>", status_code=404)

if __name__ == "__main__":
    print("🚀 Starting AI 3D Skeleton Viewer (test mode)...")
    print("📡 Server URL: http://localhost:8766")
    print("💡 Upload a video via the web interface to test background removal")
    
    uvicorn.run(app, host="0.0.0.0", port=8766)
