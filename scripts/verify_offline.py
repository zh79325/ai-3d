"""
Quick verification script for offline setup.
快速验证离线环境配置是否正确。
"""

import os
import sys


def check_dependencies():
    """检查依赖是否安装"""
    print("📦 检查 Python 依赖...")
    
    required_packages = [
        ("ultralytics", "ultralytics"),
        ("rtmlib", "rtmlib"),
        ("onnxruntime", "onnxruntime"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("pyyaml", "yaml")
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} (未安装)")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def check_models():
    """检查模型文件是否存在"""
    print("\n🤖 检查模型文件...")
    
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print(f"  ❌ 模型目录不存在: {models_dir}")
        return False
    
    model_files = os.listdir(models_dir)
    pose_models = [f for f in model_files if f.endswith("-pose.pt")]
    
    if not pose_models:
        print(f"  ❌ 未找到姿态估计模型")
        return False
    
    for model in pose_models:
        model_path = os.path.join(models_dir, model)
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"  ✅ {model} ({size_mb:.2f} MB)")
    
    return True


def check_dwpose_models():
    """检查 DWPose 的 ONNX 模型是否已下载到本地"""
    print("\n🧍 检查 DWPose 模型文件...")
    
    try:
        from ai3d.config import Config
        config = Config()
        onnx_paths = [
            ("检测器", config.model.dwpose_det_model),
            ("133 点姿态", config.model.dwpose_pose_model),
        ]
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False
    
    all_ok = True
    for label, path in onnx_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  ✅ {label}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {label} 缺失: {path}")
            all_ok = False
    
    if not all_ok:
        print("  请运行: python scripts/download_models.py --dwpose")
    
    return all_ok


def check_config():
    """检查配置是否正确"""
    print("\n⚙️  检查配置...")
    
    try:
        from ai3d.config import Config
        config = Config()
        
        pose_model_path = config.model.pose_model
        if os.path.exists(pose_model_path):
            print(f"  ✅ 姿态模型路径正确: {pose_model_path}")
        else:
            print(f"  ❌ 姿态模型路径无效: {pose_model_path}")
            return False
        
        print(f"  ✅ 设备: {config.model.device}")
        print(f"  ✅ 置信度阈值: {config.model.confidence_threshold}")
        print(f"  ✅ 姿态后端: {config.model.pose_backend}")
        
        return True
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("AI 3D Animation Generator - 离线环境验证")
    print("=" * 60)
    
    results = []
    
    # 检查依赖
    results.append(("Python 依赖", check_dependencies()))
    
    # 检查模型
    results.append(("YOLO 模型文件", check_models()))
    results.append(("DWPose 模型文件", check_dwpose_models()))
    
    # 检查配置
    results.append(("配置文件", check_config()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果总结:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 所有检查通过!项目已准备好离线运行。")
        print("\n使用示例:")
        print("  from ai3d import MotionCapturePipeline")
        print("  pipeline = MotionCapturePipeline()")
        print("  result = pipeline.process_video('test.mp4')")
        return 0
    else:
        print("\n⚠️  部分检查未通过,请根据上述提示修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
