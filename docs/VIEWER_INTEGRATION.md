# AI 3D Skeleton Viewer - 集成指南

## 概述

Skeleton Viewer 提供两种使用方式:
1. **独立模式**: 作为完整的FastAPI应用独立运行
2. **集成模式**: 将router注册到现有FastAPI项目(如Electron后端)

---

## 方式1: 独立运行

### 命令行启动

```bash
python scripts/launch_viewer.py test.mp4 --port 8765
```

### Python代码启动

```python
from ai3d.viewer import launch_standalone

launch_standalone("test.mp4", port=8765)
```

访问地址: `http://localhost:8765/api/skeleton-viewer/viewer`

---

## 方式2: 集成到现有FastAPI项目

### 基础集成

```python
from fastapi import FastAPI
from ai3d.viewer import router, set_animation_data, load_animation_from_json

app = FastAPI()

# 注册router (可选自定义前缀)
app.include_router(router, prefix="/api/skeleton-viewer")

# 预加载数据
data = load_animation_from_json("output/animation_data.json")
set_animation_data(data)
```

### Electron项目集成示例

```python
# electron_backend/main.py
from fastapi import FastAPI
from ai3d.viewer import integrate_to_existing_app

app = FastAPI(title="Electron Backend")

# 其他路由...
@app.get("/health")
def health():
    return {"status": "ok"}

# 集成skeleton viewer
integrate_to_existing_app(app, video_path="assets/demo.mp4")

# 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### 动态加载数据

```python
from fastapi import FastAPI, UploadFile
from ai3d.viewer import router, set_animation_data
import json

app = FastAPI()
app.include_router(router)

@app.post("/upload-animation")
async def upload_animation(file: UploadFile):
    """上传动画JSON数据"""
    content = await file.read()
    data = json.loads(content)
    set_animation_data(data)
    return {"status": "success", "frames": len(data["skeletons"])}
```

---

## API端点

所有端点默认前缀为 `/api/skeleton-viewer`:

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/viewer` | 返回预览页面HTML |
| POST | `/load` | 加载动画数据(JSON) |
| POST | `/load-from-file` | 从文件加载 |
| GET | `/data` | 获取当前动画数据 |
| GET | `/frame/{idx}` | 获取指定帧 |
| GET | `/info` | 获取动画信息 |
| DELETE | `/clear` | 清除数据 |

### 自定义前缀

```python
# 使用自定义前缀
app.include_router(router, prefix="/custom-prefix")
# 访问: http://localhost:8000/custom-prefix/viewer
```

---

## 前端集成

### iframe嵌入

```html
<iframe 
    src="http://localhost:8765/api/skeleton-viewer/viewer" 
    width="100%" 
    height="600px"
    frameborder="0">
</iframe>
```

### Electron WebView

```javascript
// Electron主进程
const { BrowserWindow } = require('electron');

const viewerWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
        nodeIntegration: false
    }
});

viewerWindow.loadURL('http://localhost:8765/api/skeleton-viewer/viewer');
```

---

## 数据格式

### 输入JSON结构

```json
{
    "total_frames": 421,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "skeletons": [
        {
            "frame_index": 0,
            "joints": [
                {
                    "name": "nose",
                    "position": [0.5, -1.2, 2.3],
                    "confidence": 0.95
                },
                ...
            ]
        },
        ...
    ]
}
```

---

## 故障排查

### 端口被占用

```python
# 尝试其他端口
launch_standalone("test.mp4", port=8766)
```

### 静态文件404

确保 `src/ai3d/viewer/index.html` 存在:

```bash
ls -la src/ai3d/viewer/
```

### 数据未加载

检查是否调用了 `set_animation_data()`:

```python
from ai3d.viewer import get_animation_data

data = get_animation_data()
if data is None:
    print("No data loaded!")
```

---

## 高级用法

### 多视频切换

```python
from ai3d.viewer import set_animation_data, load_animation_from_json

# 加载第一个视频
data1 = load_animation_from_json("video1.json")
set_animation_data(data1)

# 切换到第二个视频
data2 = load_animation_from_json("video2.json")
set_animation_data(data2)
```

### 实时数据流

```python
import asyncio
from ai3d.viewer import set_animation_data

async def stream_frames(pipeline, video_path):
    """逐帧处理并更新显示"""
    for frame_data in pipeline.process_stream(video_path):
        set_animation_data({"skeletons": [frame_data]})
        await asyncio.sleep(0.033)  # 30 FPS
```

---

## 性能优化

1. **减少帧数**: 只保留关键帧
2. **压缩JSON**: 使用gzip压缩传输
3. **懒加载**: 按需加载特定帧范围
4. **WebGL加速**: 确保浏览器启用硬件加速

---

## 许可证

MIT License
