#!/usr/bin/env python3
"""
IDE-friendly launcher for 3D Skeleton Viewer.
在 IDE 中直接运行此文件即可启动预览服务器。

前置条件:
- 确保 IDE 的 Python 解释器设置为: ~/Desktop/dudu/server/.venv/bin/python
- 或在终端中先激活环境: source ~/Desktop/dudu/server/.venv/bin/activate
"""

import sys
import os

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ai3d import launch_viewer


def main():
    port = 8766  # 改用 8766,因为 8765 被 ultraseek-http 占用
    
    print("=" * 60)
    print("🎬 AI 3D 工具集")
    print("=" * 60)
    print()
    print("使用说明:")
    print("1. 浏览器打开功能菜单: http://localhost:%d/" % port)
    print("   ├─ 🎬 骨骼查看器:   http://localhost:%d/api/skeleton-viewer/viewer" % port)
    print("   └─ 🦴 动画迁移工具: http://localhost:%d/retarget" % port)
    print("2. 骨骼查看器: 左侧上传视频→开始处理→播放/暂停/跳帧，鼠标拖拽旋转、滚轮缩放")
    print("3. 页面顶部有切换菜单，可在两个功能之间随时切换")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    print()
    
    # 只启动服务器,不预处理任何视频 —— 处理一律由页面上传后触发
    launch_viewer(port=port)


if __name__ == "__main__":
    main()
