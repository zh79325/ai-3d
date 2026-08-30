"""
Example: Basic pose estimation from video.
演示如何从视频中提取姿态数据。
"""

from ai3d import MotionCapturePipeline


def main():
    # 初始化管线
    pipeline = MotionCapturePipeline()
    
    # 处理视频 (请替换为你的实际视频路径)
    video_path = "test_video.mp4"
    
    try:
        result = pipeline.process_video(video_path)
        
        print(f"\n处理结果:")
        print(f"  总帧数: {result['total_frames']}")
        print(f"  有效姿态帧: {result['valid_frames']}")
        print(f"  检测率: {result['valid_frames'] / result['total_frames'] * 100:.2f}%")
        
        # 访问第一帧的骨骼数据
        if result['skeletons'][0]:
            first_skeleton = result['skeletons'][0]
            print(f"\n第一帧关键点示例:")
            for joint in first_skeleton.joints[:5]:  # 只显示前5个关键点
                print(f"  {joint['name']}: ({joint['position'][0]:.2f}, {joint['position'][1]:.2f})")
                
    except FileNotFoundError:
        print(f"❌ 视频文件不存在: {video_path}")
        print("请先放置一个测试视频文件,或修改 video_path 变量")
    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    main()
