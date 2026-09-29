"""UniRig CPU 推理封装：extract → 骨架预测 → 蒙皮预测 → ``unirig_raw.npz``。

进程内调用 vendored UniRig（复刻其 ``run.py`` 的 predict 主流程），三个前置约定：

1. ``sys.path`` 注入 ``third_party/patches/shims``（纯 torch 实现的 spconv / flash_attn
   替身，macOS 无官方 wheel）与 ``third_party/UniRig``（import ``src.*`` / ``run``）；
2. UniRig 内部大量 ``configs/...`` 相对路径 → 推理期间 ``os.chdir`` 到仓库根，
   结束后恢复（服务进程共享 cwd，必须 try/finally 还原）；
3. 权重全部本地离线：``models/unirig/*.ckpt`` + ``models/unirig/opt-350m``，
   运行期注入 ``HF_HUB_OFFLINE`` 等环境变量，OPT 的 ``pretrained_model_name_or_path``
   指向本地目录。

调用方（``job_worker.run_binding_unirig``）传入的 ``glb_path`` / ``work_dir`` 会先被
resolve 成绝对路径再 chdir，故不受工作目录切换影响。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ai3d.config import PROJECT_ROOT

from ..settings import UniRigConfig, get_settings
from . import log_capture

logger = logging.getLogger(__name__)

UNIRIG_ROOT = PROJECT_ROOT / "third_party" / "UniRig"
SHIMS_DIR = PROJECT_ROOT / "third_party" / "patches" / "shims"
TASK_DIR = PROJECT_ROOT / "third_party" / "unirig_configs"
WEIGHTS_DIR = PROJECT_ROOT / "models" / "unirig"

SKELETON_TASK = TASK_DIR / "unirig_skeleton_cpu.yaml"
SKIN_TASK = TASK_DIR / "unirig_skin_cpu.yaml"
SKELETON_CKPT = WEIGHTS_DIR / "skeleton_articulationxl_256.ckpt"
SKIN_CKPT = WEIGHTS_DIR / "skin_articulationxl.ckpt"
OPT_DIR = WEIGHTS_DIR / "opt-350m"

RAW_NPZ = "raw_data.npz"
SKELETON_NPZ = "predict_skeleton.npz"
SKIN_NPZ = "predict_skin.npz"
MERGED_NPZ = "unirig_raw.npz"          # 合并产物（调试 + adapt 的唯一输入）
PROGRESS_JSON = "unirig_progress.json"

TARGET_FACES = 50000                   # extract 减面目标（上游 quick_inference 同值）

ProgressFn = Callable[[str, str], None]


class UniRigNotReadyError(RuntimeError):
    """环境/权重未就绪（未跑 setup_unirig.py 或 download_models.py --unirig）。"""


def check_ready() -> None:
    """准入自检：缺什么直接说清怎么补，别等推理跑到一半才炸。"""
    missing = []
    if not (UNIRIG_ROOT / "run.py").exists():
        missing.append(f"UniRig 仓库缺失（跑 ./.venv/bin/python scripts/setup_unirig.py）：{UNIRIG_ROOT}")
    for p in (SKELETON_TASK, SKIN_TASK):
        if not p.exists():
            missing.append(f"任务配置缺失：{p}")
    for p in (SKELETON_CKPT, SKIN_CKPT):
        if not p.exists():
            missing.append(f"权重缺失（跑 ./.venv/bin/python scripts/download_models.py --unirig）：{p.name}")
    if not (OPT_DIR / "config.json").exists():
        missing.append(f"OPT-350m 本地目录缺失（同上 --unirig）：{OPT_DIR}")
    if missing:
        raise UniRigNotReadyError("；".join(missing))


def _set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _write_progress(work_dir: Path, stage: str, message: str,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"stage": stage, "message": message,
                               "elapsed_s": round(time.time() - _T0, 1)}
    payload.update(extra or {})
    try:
        (work_dir / PROGRESS_JSON).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:                       # 进度文件只是调试 aids，写不动不碍事
        pass


_T0 = time.time()


def _extract(glb_path: Path, work_dir: Path, progress: Optional[ProgressFn]) -> Dict[str, Any]:
    """trimesh 读 GLB（合并节点变换，顶点即规范系米制）→ raw_data.npz。

    替代上游 bpy extract：``save_raw_data`` 本身就是 trimesh + fast_simplification
    实现（减面到 ``TARGET_FACES``），bpy 只在其 ``load()`` 里用，绕开即可。
    """
    import numpy as np
    import trimesh
    from src.data.extract import save_raw_data

    if progress:
        progress("UNIRIG_EXTRACT", "读取网格并减面（trimesh → raw_data.npz）…")
    _write_progress(work_dir, "UNIRIG_EXTRACT", "读取网格并减面")
    log_capture.set_phase("UNIRIG_EXTRACT 网格抽取/减面")
    logger.info("[unirig] UNIRIG_EXTRACT 开始：读网格并减面到 %d 面", TARGET_FACES)
    t0 = time.time()
    scene = trimesh.load(str(glb_path), force="scene")
    mesh = scene.dump(concatenate=True)
    npz = work_dir / RAW_NPZ
    save_raw_data(
        path=str(npz),
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        skin=None, joints=None, tails=None, parents=None,
        names=None, matrix_local=None,
        target_count=TARGET_FACES,
    )
    data = np.load(npz, allow_pickle=True)
    info = {"extract_s": round(time.time() - t0, 1),
            "origin_vertices": int(data["vertices"].shape[0]),
            "origin_faces": int(data["faces"].shape[0])}
    _write_progress(work_dir, "UNIRIG_EXTRACT", "网格抽取完成", info)
    logger.info("UniRig extract：%s V=%d F=%d（%.1fs）", glb_path.name,
                info["origin_vertices"], info["origin_faces"], info["extract_s"])
    return info


def _run_predict(task_yaml: Path, work_dir: Path, ckpt: Path, cfg: UniRigConfig) -> None:
    """复刻 UniRig ``run.py`` 的 predict 主流程（datapath files 模式）。

    与 ``scripts/spike_unirig_cpu.py`` 验证过的逻辑一致；差异只在路径全部运行时
    resolve（task yaml 里的 ``resume_from_checkpoint`` 是项目相对路径占位）。
    """
    import lightning as L
    from run import load

    from src.data.datapath import Datapath
    from src.data.dataset import UniRigDatasetModule
    from src.data.transform import TransformConfig
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    L.seed_everything(cfg.seed, workers=True)

    task = load("task", str(task_yaml))
    assert task.mode == "predict"

    transform_config = load(
        "transform", os.path.join("configs/transform", task.components.transform))
    predict_transform_config = TransformConfig.parse(
        config=transform_config.predict_transform_config)

    tokenizer_config = task.components.get("tokenizer", None)
    tokenizer = None
    if tokenizer_config is not None:
        tokenizer_config = load(
            "tokenizer", os.path.join("configs/tokenizer", tokenizer_config))
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
        tokenizer = get_tokenizer(config=tokenizer_config)

    model_config = load("model", os.path.join("configs/model", task.components.model))
    # OPT-350m 指向本地离线目录（HF_HUB_OFFLINE 下 transformers 只读本地）
    if "llm" in model_config:
        model_config["llm"]["pretrained_model_name_or_path"] = str(OPT_DIR)
    model = get_model(tokenizer=tokenizer, **model_config)

    datapath = Datapath(files=[str(work_dir)], cls="inference")
    data = UniRigDatasetModule(
        process_fn=model._process_fn,
        predict_transform_config=predict_transform_config,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=task.components.get("data_name", RAW_NPZ),
        datapath=datapath,
        cls="inference",
    )

    writer_config = dict(task.writer)
    writer_config.pop("__target__", None)
    writer = get_writer(
        __target__=task.writer["__target__"],
        **writer_config,
        order_config=predict_transform_config.order_config,
    )

    system_config = load("system", os.path.join("configs/system", task.components.system))
    system = get_system(
        **system_config, model=model,
        optimizer_config=None, loss_config=None, scheduler_config=None,
        steps_per_epoch=1,
    )

    trainer_config = dict(task.get("trainer", {}))
    trainer = L.Trainer(callbacks=[writer], logger=False,
                        enable_checkpointing=False, **trainer_config)
    trainer.predict(system, datamodule=data, ckpt_path=str(ckpt),
                    return_predictions=False,
                    # 蒙皮 ckpt 内含 python-box 配置对象，torch 2.6 默认 weights_only=True
                    # 会拒载；上游 UniRig 行为即 weights_only=False，权重来自官方 HF 仓库可信
                    weights_only=False)


def _run_with_timeout(fn: Callable[[], None], timeout_s: int, what: str) -> None:
    """在守护线程里跑推理并限时等待：CPU 分钟级任务，超时防挂死绑定线程。

    超时后线程无法强杀（daemon，进程退出即回收），但绑定会立刻落 FAILED，
    不会把 per-asset 锁以外的资源拖住——锁在 finally 里由 job_worker 释放。

    推理线程一起手就把自己 attach 到发起线程的日志捕获上（:mod:`log_capture`）：
    Lightning 的进度条与 UniRig 的 print 全是从**这个**线程写 stderr 的，不 attach
    等于 unirig.log 里什么推理输出都没有。
    """
    box: Dict[str, BaseException] = {}
    owner = threading.current_thread()

    def _wrap() -> None:
        log_capture.attach_thread(owner)
        try:
            fn()
        except BaseException as exc:      # noqa: BLE001 原样带回主线程重抛
            box["exc"] = exc
        finally:
            # 即时摘登记：超时后本线程可能还在跑，而它已结束的输出归属已无意义
            log_capture.detach_thread()

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise RuntimeError(f"UniRig {what} 超时（>{timeout_s}s），已放弃等待")
    if "exc" in box:
        raise box["exc"]


def _norm_to_canon(canon: "np.ndarray", norm: "np.ndarray") -> Tuple[float, "np.ndarray"]:
    """求「归一空间 → 规范系米制」的等比变换 ``(s, t)``：canon ≈ norm*s + t。

    UniRig 的 inference transform 把网格归一到 [-1,1] 立方体（等比、按 bbox 中心），
    故两个空间的 bbox 只差一个统一缩放与平移：用各自 bbox 的最长轴求 s，
    再用最小角点求 t。不用逐轴缩放——归一是等比的，逐轴会把姿势压变形。
    """
    import numpy as np

    c_min, c_max = canon.min(axis=0), canon.max(axis=0)
    n_min, n_max = norm.min(axis=0), norm.max(axis=0)
    span_c = float((c_max - c_min).max())
    span_n = float((n_max - n_min).max())
    s = span_c / span_n if span_n > 1e-9 else 1.0
    return s, c_min - n_min * s


def _merge_raw(work_dir: Path) -> Path:
    """合并 raw/skeleton/skin 三个 npz → ``unirig_raw.npz``（adapt 的唯一输入）。

    键：joints(J,3) / parents(J,) / names(J,) / skin(Ns,J)（采样顶点上的权重）/
    sampled_vertices(Ns,3) / vertices(N,3) / faces(F,3)（原始全分辨率网格）。

    **空间统一是这一步的核心职责**：raw_data 的顶点是规范系米制，而 skeleton/skin
    的 joints 与采样顶点在 UniRig 的 [-1,1] 归一空间（实测 bbox 正好 ±1）。直接拼会
    让 adapt 把归一坐标当米制（手臂链检测落空 → 整条臂被合成成水平 T-pose），
    并让 kNN 权重回传对着错位的点云查最近邻。故 joints/sampled_vertices 一律
    先换回米制再落盘。
    """
    import numpy as np

    raw = np.load(work_dir / RAW_NPZ, allow_pickle=True)
    skel = np.load(work_dir / SKELETON_NPZ, allow_pickle=True)
    skin = np.load(work_dir / SKIN_NPZ, allow_pickle=True)
    out = work_dir / MERGED_NPZ

    def _pick(d, key, default=None):
        return d[key] if key in d.files else default

    names = _pick(skel, "names")
    if names is None:
        names = np.array([f"bone_{i}" for i in range(skel["joints"].shape[0])])
    canon = np.asarray(raw["vertices"], dtype=np.float64)
    s, t = _norm_to_canon(canon, np.asarray(skel["vertices"], dtype=np.float64))
    joints = np.asarray(skel["joints"], dtype=np.float64) * s + t
    sampled = np.asarray(skin["vertices"], dtype=np.float64) * s + t
    # 自检：同序同数时归一顶点换回米制应与 raw 顶点重合（浮点级）；不重合说明
    # 上游归一方式变了，宁可 loudly 报错也不能静默产出错空间的骨架
    norm_v = np.asarray(skel["vertices"], dtype=np.float64)
    if norm_v.shape == canon.shape:
        resid = float(np.abs(norm_v * s + t - canon).max())
        if resid > 1e-6:
            raise RuntimeError(
                f"UniRig 归一空间还原自检失败（残差 {resid:.6g}m）："
                f"上游 transform 不再是等比 bbox 归一，需更新 _norm_to_canon")
        logger.info("UniRig 空间统一：归一→米制 s=%.6f t=%s（bbox 残差 %.2e m）",
                    s, np.round(t, 6).tolist(), resid)
    np.savez(
        out,
        joints=joints,
        parents=_pick(skel, "parents", np.full(skel["joints"].shape[0], -1)),
        names=names,
        skin=skin["skin"],
        sampled_vertices=sampled,
        vertices=canon,
        faces=raw["faces"],
    )
    return out


def predict_unirig(glb_path: Path, work_dir: Path,
                   cfg: Optional[UniRigConfig] = None,
                   progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    """跑完整 UniRig CPU 推理，返回摘要 dict 并把 ``unirig_raw.npz`` 落在 ``work_dir``。

    ``progress(stage, message)`` 供 job_worker 落库轮询进度；stage 用自由文本
    ``UNIRIG_EXTRACT / UNIRIG_SKELETON / UNIRIG_SKIN``（不进 ``schemas.Stage`` 枚举）。
    """
    cfg = cfg or get_settings().unirig
    check_ready()
    _set_offline_env()
    glb_path = Path(glb_path).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    import torch
    threads = int(cfg.threads) if int(cfg.threads) > 0 else (os.cpu_count() or 8)
    torch.set_num_threads(threads)

    global _T0
    _T0 = time.time()
    info: Dict[str, Any] = {"threads": threads}
    saved_cwd = Path.cwd()
    import sys
    for p in (str(SHIMS_DIR), str(UNIRIG_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    logger.info("[unirig] 推理开始 %s：torch 线程=%d seed=%d 单阶段超时=%ds",
                glb_path.name, threads, cfg.seed, cfg.timeout_s)
    try:
        os.chdir(UNIRIG_ROOT)       # UniRig 内部 'configs/...' 相对路径依赖 cwd
        info.update(_extract(glb_path, work_dir, progress))

        if progress:
            progress("UNIRIG_SKELETON", "UniRig 骨架预测（CPU 自回归，可能数分钟）…")
        _write_progress(work_dir, "UNIRIG_SKELETON", "骨架预测中")
        # 阶段标签进心跳行：这一段是单 batch 自回归（实测 ~200s），Lightning 进度条
        # 只在头尾各打一次，中间完全静默，靠心跳才看得出“没死、在算哪一步”
        log_capture.set_phase("UNIRIG_SKELETON 骨架自回归生成")
        logger.info("[unirig] UNIRIG_SKELETON 开始：OPT-350m 自回归生成骨骼序列（单 batch，中途无输出属正常）")
        t0 = time.time()
        _run_with_timeout(lambda: _run_predict(SKELETON_TASK, work_dir, SKELETON_CKPT, cfg),
                          cfg.timeout_s, "骨架预测")
        info["skeleton_s"] = round(time.time() - t0, 1)
        if not (work_dir / SKELETON_NPZ).exists():
            raise RuntimeError(f"UniRig 骨架预测未产出 {SKELETON_NPZ}")
        _write_progress(work_dir, "UNIRIG_SKELETON", "骨架预测完成",
                        {"skeleton_s": info["skeleton_s"]})
        logger.info("[unirig] UNIRIG_SKELETON 完成，耗时 %.1fs", info["skeleton_s"])

        if progress:
            progress("UNIRIG_SKIN", "UniRig 蒙皮预测（CPU PTv3，可能十分钟级）…")
        _write_progress(work_dir, "UNIRIG_SKIN", "蒙皮预测中")
        log_capture.set_phase("UNIRIG_SKIN 蒙皮稀疏卷积")
        logger.info("[unirig] UNIRIG_SKIN 开始：PTv3 稀疏卷积推蒙皮权重（voxelization 走 open3d）")
        t0 = time.time()
        _run_with_timeout(lambda: _run_predict(SKIN_TASK, work_dir, SKIN_CKPT, cfg),
                          cfg.timeout_s, "蒙皮预测")
        info["skin_s"] = round(time.time() - t0, 1)
        if not (work_dir / SKIN_NPZ).exists():
            raise RuntimeError(f"UniRig 蒙皮预测未产出 {SKIN_NPZ}")
        _write_progress(work_dir, "UNIRIG_SKIN", "蒙皮预测完成", {"skin_s": info["skin_s"]})
        logger.info("[unirig] UNIRIG_SKIN 完成，耗时 %.1fs", info["skin_s"])
    finally:
        os.chdir(saved_cwd)

    merged = _merge_raw(work_dir)
    log_capture.set_phase(None)               # 推理已结束，后续步骤无需心跳
    info["total_s"] = round(time.time() - _T0, 1)
    info["raw_npz"] = str(merged)
    _write_progress(work_dir, "UNIRIG_PREDICT_DONE", "推理完成", info)
    logger.info("UniRig 推理完成 %s：extract=%.1fs skeleton=%.1fs skin=%.1fs",
                glb_path.name, info["extract_s"], info["skeleton_s"], info["skin_s"])
    return info


def copy_debug_artifacts(work_dir: Path, out_dir: Path) -> None:
    """把 ``unirig_raw.npz`` / ``unirig_progress.json`` 拷到素材产物目录（调试留痕）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (MERGED_NPZ, PROGRESS_JSON):
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
