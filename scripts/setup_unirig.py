#!/usr/bin/env python3
"""幂等 vendor UniRig 仓库并应用 CPU/离线 patch。

用法（项目根执行）:
    ./.venv/bin/python scripts/setup_unirig.py

行为:
  1. third_party/UniRig 不存在 → git clone 并 checkout 固定 commit；
     已存在 → 校验 HEAD 是否等于固定 commit，不符则 fetch + checkout。
  2. 应用 third_party/patches/unirig_cpu_patches.py 中的字符串替换 patch（幂等：
     已应用则跳过；old/new 均不匹配则报错提示上游变更）。
  3. 校验 third_party/patches/shims/{spconv,flash_attn} 存在（运行时由
     src/ai3d/retarget/unirig/predict.py 注入 sys.path 顶替 CUDA-only 依赖）。

third_party/UniRig 已在 .gitignore 中，不入库。
"""

import argparse
import subprocess
import sys
from pathlib import Path

UNIRIG_REPO = "https://github.com/zh79325/UniRig"
# 固定 commit（fork zh79325/UniRig 的 main HEAD，等同上游 VAST-AI-Research "Update README.md"）；
# 升级需重验 patch 与 ckpt 兼容性
UNIRIG_COMMIT = "6793c6640ff01c8fb389f3993434124bb43d2933"

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "UniRig"
PATCH_FILE = ROOT / "third_party" / "patches" / "unirig_cpu_patches.py"
SHIMS_DIR = ROOT / "third_party" / "patches" / "shims"


def _git(*args: str, cwd: Path = DEST) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {proc.stderr.strip()}")
    return proc.stdout.strip()


def ensure_repo() -> None:
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup_unirig] clone {UNIRIG_REPO} → {DEST}")
        _git("clone", UNIRIG_REPO, str(DEST), cwd=ROOT)
    head = _git("rev-parse", "HEAD")
    if head != UNIRIG_COMMIT:
        print(f"[setup_unirig] HEAD {head[:9]} ≠ 固定 commit，checkout {UNIRIG_COMMIT[:9]}")
        _git("fetch", "origin")
        _git("checkout", UNIRIG_COMMIT)
        head = _git("rev-parse", "HEAD")
        if head != UNIRIG_COMMIT:
            raise RuntimeError(f"checkout 后 HEAD 仍为 {head}，期望 {UNIRIG_COMMIT}")
    print(f"[setup_unirig] 仓库就绪 @ {head[:9]}")


def load_patches() -> list:
    namespace: dict = {}
    exec(PATCH_FILE.read_text(encoding="utf-8"), namespace)  # noqa: S102 数据文件
    patches = namespace.get("PATCHES")
    if not isinstance(patches, list):
        raise RuntimeError(f"{PATCH_FILE} 未定义 PATCHES 列表")
    return patches


def apply_patches() -> None:
    applied = skipped = 0
    for item in load_patches():
        target = DEST / item["path"]
        if not target.exists():
            raise RuntimeError(f"patch 目标不存在: {target}")
        text = target.read_text(encoding="utf-8")
        if item["new"] in text:
            skipped += 1
            continue
        if item["old"] not in text:
            raise RuntimeError(
                f"patch 失配（上游已变更？）: {item['path']} 中找不到待替换文本"
            )
        target.write_text(text.replace(item["old"], item["new"], 1), encoding="utf-8")
        applied += 1
        print(f"[setup_unirig] patched: {item['path']}")
    print(f"[setup_unirig] patch 完成（应用 {applied}，已跳过 {skipped}）")


def check_shims() -> None:
    for name in ("spconv", "flash_attn"):
        init = SHIMS_DIR / name / "__init__.py"
        if not init.exists():
            raise RuntimeError(f"缺少 shim: {init}")
    print(f"[setup_unirig] shims 就绪: {SHIMS_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="vendor UniRig + CPU patch（幂等）")
    parser.parse_args()
    ensure_repo()
    apply_patches()
    check_shims()
    print("[setup_unirig] ✅ 完成。权重下载: ./.venv/bin/python scripts/download_models.py --unirig")
    return 0


if __name__ == "__main__":
    sys.exit(main())
