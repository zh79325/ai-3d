"""
Example: 3D pose estimation with depth.
演示如何使用深度估计进行 3D 姿态重建。
"""

from ai3d import MotionCapturePipeline


def main():
    # 初始化管线 (启用深度估计)
    pipeline = MotionCapturePipeline()
    
    # 处理视频 (请替换为你的实际视频路径)
    video_path = "/Users/eleme/Desktop/ai-game/ai-3d/examples/videoplayback.mp4"
    
    try:
        result = pipeline.process_video(video_path)
        
        print(f"\n处理结果:")
        print(f"  总帧数: {result['total_frames']}")
        print(f"  有效姿态帧: {result['valid_frames']}")
        print(f"  使用深度: {result['use_depth']}")
        print(f"  检测率: {result['valid_frames'] / result['total_frames'] * 100:.2f}%")
        
        # 访问第一帧的 3D 骨骼数据
        if result['skeletons'][0]:
            first_skeleton = result['skeletons'][0]
            print(f"\n第一帧 3D 关键点示例:")
            
            # 检查是字典还是对象
            if isinstance(first_skeleton, dict):
                joints = first_skeleton.get('joints', [])
            else:
                joints = first_skeleton.joints
            
            for joint in joints[:5]:  # 只显示前5个关键点
                if isinstance(joint, dict):
                    name = joint['name']
                    pos = joint['position']
                else:
                    name = joint.name
                    pos = joint.position
                
                print(f"  {name}: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
                
    except FileNotFoundError:
        print(f"❌ 视频文件不存在: {video_path}")
        print("请先放置一个测试视频文件,或修改 video_path 变量")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
