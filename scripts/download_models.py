"""
Download YOLO26 models for offline use.
下载 YOLO26 模型到本地,支持离线运行。
"""

import os
from ultralytics import YOLO


def download_model(model_name: str = "yolo26n-pose.pt", save_dir: str = "./models"):
    """
    下载 YOLO26 模型到指定目录
    
    Args:
        model_name: 模型名称 (如 yolo26n-pose.pt)
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在下载模型: {model_name}")
    print(f"保存路径: {save_dir}")
    
    # 加载模型会自动下载到 ~/.config/Ultralytics/Datasets/weights/
    # 但我们可以通过手动下载来控制位置
    try:
        model = YOLO(model_name)
        
        # 获取模型的缓存路径
        cache_path = model.model_path if hasattr(model, 'model_path') else None
        
        if cache_path and os.path.exists(cache_path):
            # 复制到目标目录
            import shutil
            target_path = os.path.join(save_dir, model_name)
            
            if not os.path.exists(target_path):
                shutil.copy2(cache_path, target_path)
                print(f"✅ 模型已保存到: {target_path}")
            else:
                print(f"⚠️  模型已存在: {target_path}")
        else:
            print(f"✅ 模型已缓存到系统默认路径")
            print(f"   你可以在代码中直接使用: model = YOLO('{model_name}')")
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        raise


def list_available_models():
    """列出可用的 YOLO26-pose 模型"""
    models = [
        "yolo26n-pose.pt",  # Nano - 最快,精度最低
        "yolo26s-pose.pt",  # Small
        "yolo26m-pose.pt",  # Medium
        "yolo26l-pose.pt",  # Large
        "yolo26x-pose.pt",  # X-Large - 最慢,精度最高
    ]
    
    print("\n可用的 YOLO26-pose 模型:")
    print("-" * 60)
    for model in models:
        print(f"  • {model}")
    print("-" * 60)
    print("\n推荐: yolo26n-pose.pt (速度快,适合实时处理)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 YOLO26 模型")
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolo26n-pose.pt",
        help="模型名称 (默认: yolo26n-pose.pt)"
    )
    parser.add_argument(
        "--dir", 
        type=str, 
        default="./models",
        help="保存目录 (默认: ./models)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="列出可用模型"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    else:
        download_model(args.model, args.dir)
