"""S1 导入矫正 worker（``/v2`` 素材库）。

S1 是模型与动画**共用**的公共模块，三步：

1. 格式归一：``glb_io.normalize_to_glb`` 把 FBX/glTF/GLB 统一成单文件 GLB。这里显式
   关掉 ``axis.enabled``——旧的「吸附到世界坐标轴」归一会被 :func:`axis_norm.align_asset`
   复位后由 OBB 结果接管，跑一遍纯属浪费，还会多写一个无用的 ``*_axis_frame.json``。
2. 结构概要：``read_summary`` 落 ``meta.json``，并校验「模型必须有网格」。
3. OBB 对齐 + 单位推断：产出 ``align.json`` 与 canon 节点（旋转 + 均匀缩放）。

人工修正走 :meth:`AssetWorker.run_align` 的第二条路径：只合并 ``manual`` 段后重算，
OBB 与自动指派每次都在原始模型坐标下重测（幂等，见 ``axis_norm._reset_canon``）。

S2~S4 的 ``JobWorker`` 在 P3~P5 接入；本模块只负责 S1，旧 :class:`worker.Worker`
与 ``STAGE_ORDER`` 原样保留服务 ``/v1``。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from . import axis_norm, glb_io
from .asset_store import AssetStore, NotFoundError, get_asset_store
from .schemas_v2 import AlignPatch, AssetKind, AssetState, BindingState
from .settings import get_settings

logger = logging.getLogger(__name__)


class AlignError(Exception):
    """S1 无法完成（归一失败、文件损坏、写盘错误等服务端问题）。"""


class AlignInputError(AlignError):
    """S1 因**输入本身**无法完成：缺原件、格式不支持、无网格、面映射非法。

    路由层据此区分 4xx（用户改输入就能修）与 5xx（服务端故障）。
    """


def merge_manual(current: Optional[Dict[str, Any]],
                 patch: Optional[AlignPatch]) -> Dict[str, Any]:
    """把 ``AlignPatch`` 合并进现有 align 的 ``manual`` 段，返回新的 manual。

    ``reset=True`` 直接清空全部人工修正（回到自动探测）；其余字段只在非 ``None`` 时
    覆盖，``face_map`` 按约定整体替换而非逐面合并。

    两项覆盖的**清除**约定（JSON 里 ``None`` 已被用作「不改」，故用哨兵值）：
    ``height_override <= 0`` 与 ``unit_override == ""`` 都表示取消该项人工覆盖。
    """
    manual: Dict[str, Any] = dict(((current or {}).get("manual") or {}))
    if patch is None:
        return manual
    if patch.reset:
        return {}
    if patch.face_map is not None:
        manual["face_map"] = {str(k): str(v) for k, v in patch.face_map.items()}
    if patch.trim_euler is not None:
        manual["trim_euler"] = [float(v) for v in patch.trim_euler]
    if patch.unit_override is not None:
        manual["unit_override"] = patch.unit_override.strip() or None
    if patch.height_override is not None:
        h = float(patch.height_override)
        manual["height_override"] = h if h > 0.0 else None
    return manual


class AssetWorker:
    """S1 导入矫正执行器。"""

    def __init__(self, store: Optional[AssetStore] = None):
        self.store = store or get_asset_store()

    # ------------------------------------------------------------------ #
    # S1
    # ------------------------------------------------------------------ #
    def run_align(self, asset_id: str, patch: Optional[AlignPatch] = None) -> Dict[str, Any]:
        """跑（或重跑）S1，返回完整 align.json 结构。

        ``patch`` 为空即首次自动对齐；带 patch 即人工修正重算（只改 manual）。
        失败时抛 :class:`AlignError`（路由转 4xx/5xx）；若上一次已有可用结果则保留
        它而不落 ``FAILED``（见 :meth:`_fail`）。
        """
        row = self.store.get_row(asset_id)
        if row is None:
            raise NotFoundError(asset_id)
        raw = self._raw_path(row)
        if raw is None:
            raise AlignInputError("素材尚未上传文件，无法做导入矫正")
        glb = self.store.glb_path(asset_id)
        prev_align = self.store.load_align(asset_id)
        prev_state = row["state"]
        manual = merge_manual(prev_align, patch)
        self.store.set_state(asset_id, AssetState.NORMALIZING, error=None)
        try:
            if patch is None or not glb.exists():
                # 首次（或产物缺失）才重做格式归一；人工修正只需在既有 asset.glb 上重算
                glb_io.normalize_to_glb(raw, glb, settings=self._no_axis_settings())
            summary = glb_io.read_summary(glb)
            if summary["meshes"] < 1 and summary["skins"] < 1:
                raise AlignInputError("文件既不含网格也不含骨架，无法做导入矫正")
            if row["kind"] == AssetKind.MODEL.value and summary["meshes"] < 1:
                raise AlignInputError("模型素材不含任何网格（mesh），无法绑定与迁移")
            align = axis_norm.align_asset(glb, manual or None,
                                          json_path=self.store.align_path(asset_id))
        except AlignInputError as exc:       # 输入问题：原样抛出（路由转 400）
            self._fail(asset_id, str(exc), prev_align, prev_state, glb.exists())
            raise
        except ValueError as exc:            # 面映射/语义非法：属用户输入错误
            self._fail(asset_id, str(exc), prev_align, prev_state, glb.exists())
            raise AlignInputError(str(exc)) from exc
        except AlignError as exc:
            self._fail(asset_id, str(exc), prev_align, prev_state, glb.exists())
            raise
        except (OSError, RuntimeError) as exc:
            # 写盘失败、assimp 转换失败（ConversionError 是 RuntimeError 子类）
            self._fail(asset_id, str(exc), prev_align, prev_state, glb.exists())
            raise AlignError(str(exc)) from exc
        self.store.save_align(asset_id, align)
        self.store.save_meta(asset_id, summary)
        self.store.set_state(asset_id, AssetState.ALIGN_READY, error=None)
        unit = align["unit"]
        logger.info("S1 完成 %s（%s）：unit=%s height=%.4fm changed=%s",
                    asset_id, row["kind"], unit["detected"], unit["height_m"],
                    align["final"]["changed"])
        return align

    def _fail(self, asset_id: str, message: str, prev_align: Optional[Dict[str, Any]],
              prev_state: str, glb_exists: bool) -> None:
        """失败落库：已有可用 S1 结果时**保留**上一次的 align 与状态。

        人工修正填错（面映射非法、单位写错）不该把已经算好的 S1 结果一并废掉：
        ``align_asset`` 只在算出 ``final`` 后才写盘，异常时 ``asset.glb`` 未被改动，
        因此把状态还原到修正前即可，前端刷新仍能拿到上一次的正确提案。
        """
        keepable = prev_state in (AssetState.ALIGN_READY.value, AssetState.READY.value)
        if prev_align is not None and glb_exists and keepable:
            self.store.set_state(asset_id, AssetState(prev_state), error=None)
            logger.warning("S1 重算失败（已保留上一次结果）%s：%s", asset_id, message)
            return
        self.store.set_state(asset_id, AssetState.FAILED, error=message)
        logger.warning("S1 失败 %s：%s", asset_id, message)

    def confirm(self, asset_id: str) -> Dict[str, Any]:
        """确认 S1 结果。

        动画素材到此即 ``READY``（可被任意模型复用）；模型素材也置 ``READY`` 并建
        ``bindings`` 行（``PENDING``）。**S2 不在这里起线程**：触发放在路由层
        （``POST /v2/assets/{id}/confirm``），两个 worker 互相依赖会让单测起真线程，
        而且直接调本方法的调用方（测试、脚本）不该被塞一个后台任务。
        """
        row = self.store.require(asset_id)
        state = AssetState(row["state"])
        if state not in (AssetState.ALIGN_READY, AssetState.READY):
            raise AlignInputError(f"素材处于 {state.value}，须先完成 S1 导入矫正再确认")
        glb = self.store.glb_path(asset_id)
        if not glb.exists():
            raise AlignInputError(f"素材缺少 {glb.name}，请先上传文件")
        self.store.set_state(asset_id, AssetState.READY, error=None)
        if row["kind"] == AssetKind.MODEL.value:
            self.store.ensure_binding(asset_id, BindingState.PENDING)
            logger.info("素材 %s 已确认 S1，等待路由层触发 S2 绑定", asset_id)
        else:
            logger.info("动画素材 %s 已入库，可被任意模型复用", asset_id)
        return self.store.get_asset_info(asset_id).model_dump(mode="json")

    def start_async(self, asset_id: str,
                    patch: Optional[AlignPatch] = None) -> threading.Thread:
        """后台跑 S1（FBX 走 assimp 可能较慢时用；默认路由是同步跑）。"""
        t = threading.Thread(target=self._safe_run, args=(asset_id, patch), daemon=True)
        t.start()
        return t

    def _safe_run(self, asset_id: str, patch: Optional[AlignPatch]) -> None:
        try:
            self.run_align(asset_id, patch)
        except (AlignError, NotFoundError) as exc:
            logger.warning("后台 S1 失败 %s：%s", asset_id, exc)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _raw_path(self, row: Dict[str, Any]) -> Optional[Path]:
        """登记过的上传原件路径；未上传或文件已丢失时返回 ``None``。"""
        filename = row.get("filename")
        if not filename:
            return None
        path = Path(row.get("asset_dir") or self.store.asset_dir(row["asset_id"])) / filename
        return path if path.exists() else None

    @staticmethod
    def _no_axis_settings():
        """关掉旧轴系吸附的 settings 副本（不污染全局单例）。"""
        s = get_settings()
        return replace(s, axis=replace(s.axis, enabled=False))


# ---- 单例 ----
_worker: Optional[AssetWorker] = None


def get_asset_worker(reload: bool = False) -> AssetWorker:
    global _worker
    if _worker is None or reload:
        _worker = AssetWorker()
    return _worker
