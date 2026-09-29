"""
Download models for offline use.
下载 YOLO26 / DWPose / UniRig 模型到本地,支持离线运行。
"""

import os
import shutil
import tempfile
import urllib.request
import zipfile
from ultralytics import YOLO

# DWPose (rtmlib Wholebody, mode='balanced') 所需的两个 ONNX 模型
# 压缩包内均为 end2end.onnx,下载后重命名为带输入尺寸的文件名,便于与 config 中的 input_size 对齐
DWPOSE_MODELS = [
    {
        "target": "dwpose-det-yolox-m-640x640.onnx",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
               "onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "desc": "人体检测器 YOLOX-m (输入 640x640, 约 97MB)",
    },
    {
        "target": "dwpose-pose-rtmw-x-l-192x256.onnx",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
               "onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        "desc": "133 点全身姿态 RTMW-x (输入 192x256, 约 218MB)",
    },
]


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


def download_dwpose_models(save_dir: str = "./models"):
    """
    下载 DWPose (rtmlib balanced 模式) 的检测器与姿态 ONNX 模型到本地目录

    下载完成后即可完全离线运行,代码通过 ai3d.config.ModelConfig 中的
    dwpose_det_model / dwpose_pose_model 显式加载这两个文件。

    Args:
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    print("正在下载 DWPose 模型 (共 2 个, 合计约 315MB)")
    print(f"保存路径: {save_dir}\n")

    for item in DWPOSE_MODELS:
        target_path = os.path.join(save_dir, item["target"])
        if os.path.exists(target_path):
            size_mb = os.path.getsize(target_path) / 1024 / 1024
            print(f"⚠️  已存在,跳过: {item['target']} ({size_mb:.1f} MB)")
            continue

        print(f"⬇️  {item['desc']}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "model.zip")
            try:
                urllib.request.urlretrieve(item["url"], zip_path)
            except OSError as e:
                print(f"❌ 下载失败: {item['url']}\n   {e}")
                raise

            extract_dir = os.path.join(tmp_dir, "extracted")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            onnx_path = None
            for root, _, files in os.walk(extract_dir):
                for name in files:
                    if name.endswith(".onnx"):
                        onnx_path = os.path.join(root, name)
                        break
                if onnx_path:
                    break

            if not onnx_path:
                raise RuntimeError(f"压缩包内未找到 .onnx 文件: {item['url']}")

            shutil.copy2(onnx_path, target_path)

        size_mb = os.path.getsize(target_path) / 1024 / 1024
        print(f"✅ 已保存: {target_path} ({size_mb:.1f} MB)\n")

    print("🎉 DWPose 模型准备完成,后续运行无需联网。")


# UniRig 骨架/蒙皮新链路权重（HuggingFace）：两个 ckpt + OPT-350m 文本编码器。
# 运行期由 predict.py 注入 HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1，仅读本地。
UNIRIG_HF_FILES = [
    {
        "repo": "VAST-AI/UniRig",
        "filename": "skeleton/articulation-xl_quantization_256/model.ckpt",
        "target": "skeleton_articulationxl_256.ckpt",
        "desc": "UniRig 骨架预测 ckpt (articulation-xl, 量化 256)",
    },
    {
        "repo": "VAST-AI/UniRig",
        "filename": "skin/articulation-xl/model.ckpt",
        "target": "skin_articulationxl.ckpt",
        "desc": "UniRig 蒙皮预测 ckpt (articulation-xl)",
    },
]


def download_unirig_models(save_dir: str = "./models/unirig"):
    """
    下载 UniRig 骨架/蒙皮 ckpt 与 OPT-350m 到本地目录，下载后完全离线运行。

    Args:
        save_dir: 保存目录 (默认 ./models/unirig)
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(save_dir, exist_ok=True)
    print(f"正在下载 UniRig 权重 (骨架/蒙皮 ckpt + opt-350m, 合计数 GB)")
    print(f"保存路径: {save_dir}\n")

    for item in UNIRIG_HF_FILES:
        target_path = os.path.join(save_dir, item["target"])
        if os.path.exists(target_path):
            size_mb = os.path.getsize(target_path) / 1024 / 1024
            print(f"⚠️  已存在,跳过: {item['target']} ({size_mb:.1f} MB)")
            continue
        print(f"⬇️  {item['desc']}: {item['repo']}/{item['filename']}")
        cached = hf_hub_download(repo_id=item["repo"], filename=item["filename"])
        shutil.copy2(cached, target_path)
        size_mb = os.path.getsize(target_path) / 1024 / 1024
        print(f"✅ 已保存: {target_path} ({size_mb:.1f} MB)\n")

    opt_dir = os.path.join(save_dir, "opt-350m")
    if os.path.exists(os.path.join(opt_dir, "config.json")):
        print(f"⚠️  已存在,跳过: opt-350m/")
    else:
        print("⬇️  OPT-350m 文本编码器: facebook/opt-350m")
        snapshot_download(repo_id="facebook/opt-350m", local_dir=opt_dir)
        print(f"✅ 已保存: {opt_dir}\n")

    print("🎉 UniRig 权重准备完成,后续运行无需联网。")


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
    
    parser = argparse.ArgumentParser(description="下载 YOLO26 / DWPose 模型")
    parser.add_argument(
        "--dwpose",
        action="store_true",
        help="下载 DWPose 133 点姿态所需的 det + pose ONNX 模型"
    )
    parser.add_argument(
        "--unirig",
        action="store_true",
        help="下载 UniRig 骨架/蒙皮 ckpt 与 OPT-350m 到 ./models/unirig"
    )
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
    elif args.unirig:
        download_unirig_models(
            args.dir if args.dir != "./models" else "./models/unirig"
        )
    elif args.dwpose:
        download_dwpose_models(args.dir)
    else:
        download_model(args.model, args.dir)
