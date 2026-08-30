"""
Example: Launch web viewer for 3D skeleton preview.
启动Web预览器查看3D骨骼动画。
"""

from ai3d import launch_viewer


def main():
    # 视频路径
    video_path = "/Users/eleme/Desktop/ai-game/ai-3d/examples/videoplayback.mp4"
    port = 8766
    print("=" * 60)
    print("🎬 AI 3D Skeleton Viewer")
    print("=" * 60)
    print()
    print("使用说明:")
    print("1. 等待数据处理完成 (可能需要几分钟)")
    print("2. 在浏览器中访问: http://localhost:%d"%port)
    print("3. 使用控制按钮播放/暂停/跳转帧")
    print("4. 鼠标拖拽旋转视角,滚轮缩放")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    print()
    
    # 启动预览
    launch_viewer(video_path, port)


if __name__ == "__main__":
    main()
