# IDE 运行指南

## PyCharm / IntelliJ IDEA

### 1. 配置 Python 解释器

1. 打开 **Settings** → **Project** → **Python Interpreter**
2. 点击齿轮图标 → **Add...**
3. 选择 **Existing environment**
4. 浏览到: `/Users/eleme/Desktop/dudu/server/.venv/bin/python`
5. 点击 **OK**

### 2. 运行预览

在 IDE 中右键点击以下任一文件，选择 **Run**:

- `run_viewer_ide.py` (推荐)
- `examples/view_3d_skeleton.py`

### 3. 修改视频路径

编辑 `run_viewer_ide.py` 中的 `video_path` 变量指向你的测试视频。

---

## VS Code

### 1. 选择解释器

1. 按 `Cmd+Shift+P` (Mac) 或 `Ctrl+Shift+P` (Windows/Linux)
2. 输入 **Python: Select Interpreter**
3. 选择: `~/Desktop/dudu/server/.venv/bin/python`

### 2. 运行

打开 `run_viewer_ide.py`，点击右上角的 **▶ Run** 按钮。

---

## 验证安装

在终端中运行:

```bash
/Users/eleme/Desktop/dudu/server/.venv/bin/python -c "import ai3d; print(ai3d.__version__)"
```

应输出: `0.1.0`

---

## 常见问题

### ModuleNotFoundError: No module named 'ai3d'

确保 IDE 使用的是正确的虚拟环境解释器（见上方配置步骤）。

### Port already in use

修改 `run_viewer_ide.py` 中的 `port = 8765` 为其他端口（如 8766、8767）。

### Video file not found

检查 `run_viewer_ide.py` 中的 `video_path` 是否指向存在的视频文件。

---

## API 端点

服务器启动后，可用以下端点测试:

| 端点 | 描述 |
|------|------|
| `GET /api/skeleton-viewer/viewer` | 预览页面 HTML |
| `GET /api/skeleton-viewer/info` | 动画信息 |
| `GET /api/skeleton-viewer/data` | 完整骨骼数据 |
| `GET /api/skeleton-viewer/frame/{idx}` | 指定帧数据 |

测试命令:

```bash
curl http://localhost:8765/api/skeleton-viewer/info
```
