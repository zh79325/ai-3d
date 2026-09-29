"""S2~S4 的执行器（``/v2`` 四阶段流程）。

**S2（绑定）是素材级的**：产物写 ``output/assets/<asset_id>/``，一次绑定被所有引用该
模型的作业复用 —— 这正是 assets / jobs 拆分的意义（换个动画不必重算蒙皮）。
S3（重定向）与 S4（导出）是作业级的，在 P4 / P5 接入本类。

S2 有三条路径：

- ``pose_ai=True``（默认，与 ``/v1`` 一致）：等前端回传多视角图 → DWPose 出 2D 关键点
  → 加权三角化 + 对称/比例/骨长约束 → 三维关节 → 蒙皮。视角图没到就停在
  ``WAIT_VIEWS``，``POST /v2/assets/{id}/views`` 到达后自动续跑。
- ``pose_ai=False``：直接按网格包围盒的人体比例生成骨架 → 蒙皮。适合非人形/道具，
  或 DWPose 检测不到人时的兜底。
- ``method='unirig'``（:meth:`run_binding_unirig`）：UniRig 学习模型 CPU 推理骨架+蒙皮
  → 语义适配到 22 关节。不需要视角图（不产生 WAIT_VIEWS），产物契约与上两条一致
  （rig.json / skin.npz / skin_report.json），分钟级耗时同样后台线程 + 轮询进度。

蒙皮（libigl BBW）是 S2 最慢的一步（万级顶点数秒到十几秒），故一律后台线程跑，
子步骤进度落 ``bindings.stage`` / ``message``，前端轮询素材详情即可画进度条。

同一素材不允许并发跑 S2：两个线程同时写 ``rig.json`` / ``skin.npz`` 会互相覆盖，
故用 per-asset 锁把重复触发挡在门外（路由翻成 409）。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from . import stages
from .asset_store import AssetStore, NotFoundError, StateError, get_asset_store
from .schemas import RigPatch, Stage
from .schemas_v2 import AssetKind, BindingInfo, BindingState
from .settings import get_settings
from .skeleton import Rig, apply_patch
from .unirig import log_capture

logger = logging.getLogger(__name__)

BIND_METHODS = ("ai", "bbox", "unirig")   # /bind 与 /rebind 的 method 查询参数取值


class BindingError(Exception):
    """S2 无法完成（推理失败、三角化失败、蒙皮失败、写盘错误等服务端问题）。"""


class BindingInputError(BindingError):
    """S2 因**输入本身**无法完成：素材类型不对、缺 rig、关节补丁非法。

    路由层据此区分 4xx（用户改输入就能修）与 5xx（服务端故障）。
    """


class JobWorker:
    """S2~S4 执行器（当前实现 S2）。"""

    def __init__(self, store: Optional[AssetStore] = None):
        self.store = store or get_asset_store()
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ #
    # 并发保护
    # ------------------------------------------------------------------ #
    def _lock_for(self, asset_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(asset_id)
            if lock is None:
                lock = self._locks[asset_id] = threading.Lock()
            return lock

    def is_busy(self, asset_id: str) -> bool:
        """该素材是否有 S2 正在跑（路由用来决定 409 还是排队）。"""
        lock = self._lock_for(asset_id)
        if lock.acquire(blocking=False):
            lock.release()
            return False
        return True

    # ------------------------------------------------------------------ #
    # S2 绑定
    # ------------------------------------------------------------------ #
    def run_binding(self, asset_id: str, pose_ai: bool = True,
                    snapshot: Optional[Tuple[bool, Optional[str]]] = None) -> BindingInfo:
        """跑 S2：（可选）多视角推理 → 三角化 → 骨架 + 蒙皮。

        ``pose_ai=True`` 且视角图尚未回传时**不算失败**：置 ``WAIT_VIEWS`` 后正常返回，
        等前端 ``POST /views`` 再以 ``pose_ai=True`` 续跑。

        ``snapshot`` 由 :meth:`start_binding` 在置 RUNNING 前抓好传入；直接同步调用时
        缺省即可（此时库里还是上一轮的状态）。
        """
        row = self._require_model(asset_id)
        glb = self.store.glb_path(asset_id)
        out_dir = self.store.asset_dir(asset_id)
        had_artifacts, prev_state = snapshot or self._snapshot(asset_id)
        stage = Stage.RENDER_VIEWS.value
        lock = self._lock_for(asset_id)
        if not lock.acquire(blocking=False):
            raise StateError(f"素材 {asset_id} 的 S2 正在执行，请等它跑完")
        try:
            self.store.update_binding(asset_id, state=BindingState.RUNNING, error=None)
            rig: Optional[Rig] = None
            if pose_ai:
                views = self.store.list_views(asset_id)
                if not views:
                    self.store.set_binding_progress(
                        asset_id, Stage.RENDER_VIEWS.value,
                        "等待前端回传多视角图（S2 的三角化输入）", BindingState.WAIT_VIEWS)
                    logger.info("素材 %s 的 S2 挂起在 RENDER_VIEWS", asset_id)
                    return self.store.binding_info(asset_id)
                stage = Stage.POSE_INFER.value
                self.store.set_binding_progress(
                    asset_id, stage, f"{len(views)} 张视角图 2D 关键点推理中…")
                stages.infer_views(self.store.views_dir(asset_id),
                                   self.store.detections_path(asset_id))
                stage = Stage.SOLVE_RIG.value
                self.store.set_binding_progress(asset_id, stage, "多视角三角化求解三维关节…")
                rig = stages.solve_rig(glb, self.store.detections_path(asset_id),
                                       self.store.rig_path(asset_id),
                                       self.store.view_spec_path(asset_id))
            stage = Stage.BUILD_RIG.value
            self.store.set_binding_progress(
                asset_id, stage,
                "自动蒙皮（libigl BBW）计算中，万级顶点通常数秒…")
            rig, n_verts, report = stages.build_rig(glb, out_dir, rig)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("S2 失败 asset=%s stage=%s", asset_id, stage)
            self._fail_binding(asset_id, str(exc), stage, had_artifacts, prev_state)
            raise BindingError(str(exc)) from exc
        finally:
            lock.release()
        method = "多视角三角化" if pose_ai and rig.source.get("pelvis") == "solved" else "人体比例"
        message = (f"骨架 {len(rig.heads)} 关节（{method}，height={rig.height:.3f}m）"
                   f" + 蒙皮 {n_verts} 顶点（{stages.skin_summary(report)}）")
        self.store.mark_binding_done(asset_id, confidence=rig.overall_confidence(),
                                     revision=rig.revision, message=message)
        logger.info("S2 完成 %s（kind=%s）：%s", asset_id, row["kind"], message)
        return self.store.binding_info(asset_id)

    def run_binding_unirig(self, asset_id: str,
                           snapshot: Optional[Tuple[bool, Optional[str]]] = None
                           ) -> BindingInfo:
        """跑 S2 的 UniRig 新链路：CPU 推理骨架+蒙皮 → 语义适配到 22 关节。

        与 :meth:`run_binding` 同样 per-asset 锁、失败回滚（保留上一次可用产物）；
        不需要视角图故不产生 ``WAIT_VIEWS``。进度 stage 用自由文本
        ``UNIRIG_EXTRACT / UNIRIG_SKELETON / UNIRIG_SKIN / UNIRIG_ADAPT``
        （``bindings.stage`` 本就是 TEXT 列，不动 ``schemas.Stage`` 枚举）。

        全程输出（Lightning 进度条 / print / logging / 失败 traceback）由
        :mod:`unirig.log_capture` 按线程分流落 ``<asset_dir>/unirig.log``，前端用
        ``GET /v2/assets/{id}/unirig/log?offset=N`` 增量轮询看实时进度 ——
        stage 只有四个粗粒度值，一次推理好几分钟，不看日志根本不知道卡在哪。
        """
        from .unirig import adapt as unirig_adapt
        from .unirig import predict as unirig_predict

        row = self._require_model(asset_id)
        glb = self.store.glb_path(asset_id)
        out_dir = self.store.asset_dir(asset_id)
        work_dir = self.store.unirig_work_dir(asset_id)   # 中间产物（raw/骨架/蒙皮 npz）
        log_path = self.store.unirig_log_path(asset_id)
        had_artifacts, prev_state = snapshot or self._snapshot(asset_id)
        stage = "UNIRIG_EXTRACT"
        lock = self._lock_for(asset_id)
        if not lock.acquire(blocking=False):
            raise StateError(f"素材 {asset_id} 的 S2 正在执行，请等它跑完")
        header = (f"==== UniRig S2 asset={asset_id} kind={row['kind']} "
                  f"start={time.strftime('%Y-%m-%d %H:%M:%S')} ====\n"
                  f"输入 GLB：{glb}\n工作目录：{work_dir}")
        try:
            with log_capture.capture(log_path, header=header):
                try:
                    self.store.update_binding(asset_id, state=BindingState.RUNNING, error=None)
                    align = self.store.load_align(asset_id) or {}
                    height_m = ((align.get("unit") or {}).get("height_m")
                                or (align.get("final") or {}).get("height_m"))

                    def _progress(st: str, msg: str) -> None:
                        self.store.set_binding_progress(asset_id, st, msg)

                    info = unirig_predict.predict_unirig(
                        glb, work_dir, get_settings().unirig, progress=_progress)
                    stage = "UNIRIG_ADAPT"
                    log_capture.set_phase("UNIRIG_ADAPT 语义适配 + 权重聚合")
                    self.store.set_binding_progress(
                        asset_id, stage, "语义适配：UniRig 骨架 → 22 关节 + 权重聚合…")
                    rig, n_verts, report = unirig_adapt.build_binding(
                        glb, work_dir, out_dir,
                        height_m=float(height_m) if height_m else None)
                    unirig_predict.copy_debug_artifacts(work_dir, out_dir)
                    message = (
                        f"骨架 {len(rig.heads)} 关节（UniRig CPU，height={rig.height:.3f}m，"
                        f"推理 {info.get('total_s')}s） + 蒙皮 {n_verts} 顶点"
                        f"（{stages.skin_summary(report)}, coverage={report.get('coverage')}）")
                    self.store.mark_binding_done(asset_id,
                                                 confidence=rig.overall_confidence(),
                                                 revision=rig.revision, message=message)
                    # 成功摘要也要进日志：它写在 capture 里，看日志就能知道结果
                    logger.info("S2(unirig) 完成 %s（kind=%s）：%s",
                                asset_id, row["kind"], message)
                    return self.store.binding_info(asset_id)
                except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
                    # 在 capture 内 exception：traceback 跟着写进 unirig.log，
                    # 前端轮询到的最后几行就是失败原因，不用去翻服务控制台
                    logger.exception("S2(unirig) 失败 asset=%s stage=%s", asset_id, stage)
                    self._fail_binding(asset_id, str(exc), stage, had_artifacts, prev_state)
                    raise BindingError(str(exc)) from exc
        finally:
            lock.release()

    def run_reskin(self, asset_id: str,
                   snapshot: Optional[Tuple[bool, Optional[str]]] = None) -> BindingInfo:
        """人工微调关节后**只重算蒙皮**：``rig.json`` 保持人工版本与 revision 不变。

        关节一动，BBW 的 handle 选骨与权重全部失效，必须重算；但骨架是人工确认过的，
        重跑 :meth:`run_binding` 会把它覆盖回自动结果。
        """
        self._require_model(asset_id)
        glb = self.store.glb_path(asset_id)
        out_dir = self.store.asset_dir(asset_id)
        # need_rig=False：重算蒙皮只依赖旧的 skin.npz 是否可用（rig.json 必然存在）
        had_artifacts, prev_state = snapshot or self._snapshot(asset_id, need_rig=False)
        lock = self._lock_for(asset_id)
        if not lock.acquire(blocking=False):
            raise StateError(f"素材 {asset_id} 的 S2 正在执行，请等它跑完")
        try:
            self.store.set_binding_progress(
                asset_id, Stage.BUILD_RIG.value, "按人工关节重算蒙皮（libigl BBW）…")
            rig, n_verts, report = stages.reskin(glb, out_dir)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("S2 重算蒙皮失败 asset=%s", asset_id)
            self._fail_binding(asset_id, str(exc), Stage.BUILD_RIG.value,
                               had_artifacts, prev_state)
            raise BindingError(str(exc)) from exc
        finally:
            lock.release()
        message = f"已按人工关节重算蒙皮：{n_verts} 顶点（{stages.skin_summary(report)}）"
        # 不传 revision：rig.json 没被重写，人工版本号要留着
        self.store.mark_binding_done(asset_id, confidence=rig.overall_confidence(),
                                     message=message)
        logger.info("S2 蒙皮重算完成 %s：%s", asset_id, message)
        return self.store.binding_info(asset_id)

    def patch_rig(self, asset_id: str, patch: RigPatch) -> int:
        """人工微调关节位置（revision 乐观锁），返回新 revision。

        走 :func:`skeleton.apply_patch` 而不是直接改字典：它会把动过的关节标记为
        ``source="manual"``（门控与前端都据此区分自动/人工），并忽略未知关节名。
        """
        self._require_model(asset_id)
        rig_dict = self.store.load_binding_json(asset_id, "rig")
        if rig_dict is None:
            raise BindingInputError("rig.json 尚未生成，请先跑完 S2 再校正关节")
        rig = Rig.from_dict(rig_dict)
        rig.revision = patch.revision        # 以乐观锁版本为准，apply_patch 再 +1
        rig = apply_patch(rig, {jid: jp.model_dump(exclude_none=True)
                                for jid, jp in patch.joints.items()})
        try:
            return self.store.save_rig(asset_id, patch.revision, rig.to_dict())
        except NotFoundError:
            raise BindingInputError(f"素材不存在：{asset_id}") from None

    # ------------------------------------------------------------------ #
    # 后台执行
    # ------------------------------------------------------------------ #
    def start_binding(self, asset_id: str, pose_ai: bool = True,
                      method: str = "ai") -> threading.Thread:
        """后台跑 S2（蒙皮慢，路由不阻塞）。

        准入校验（素材存在 / 是模型 / S1 已落库 / 未在跑）在**当前线程**做完再起线程：
        放到线程里的话，400/409 只会躺在后台日志里，前端拿到 200 却不知道根本没启动。

        ``method='unirig'`` 走 :meth:`run_binding_unirig`（``pose_ai`` 忽略）；环境/权重
        未就绪也在这里同步抛 ``BindingInputError`` → 路由翻 400，而不是让素材落 FAILED。
        """
        self._require_model(asset_id)
        if method == "unirig":
            from .unirig import predict as unirig_predict

            cfg = get_settings().unirig
            if not cfg.enabled:
                raise BindingInputError("UniRig 链路未启用（retarget.yaml 的 unirig.enabled=false）")
            try:
                unirig_predict.check_ready()
            except RuntimeError as exc:
                raise BindingInputError(str(exc)) from exc
        self._require_idle(asset_id)
        snapshot = self._snapshot(asset_id)
        # 先同步置 RUNNING 再起线程：否则路由返回后前端立刻轮询，会读到上一轮的
        # READY（重算蒙皮尤其明显），把「还没开始」当成「已经跑完」。
        self.store.update_binding(asset_id, state=BindingState.RUNNING,
                                  stage=None, message=None, error=None)
        if method == "unirig":
            return self._start(self.run_binding_unirig, asset_id, snapshot)
        return self._start(self.run_binding, asset_id, pose_ai, snapshot)

    def start_reskin(self, asset_id: str) -> threading.Thread:
        """后台重算蒙皮（关节微调后触发），准入校验同 :meth:`start_binding`。"""
        self._require_model(asset_id)
        # 缺 rig.json 是**输入**问题（S2 还没跑完），必须在当前线程抛 → 路由翻 400；
        # 丢到后台线程里只会让素材莫名落 FAILED，前端拿到的是 200
        if not self.store.rig_path(asset_id).exists():
            raise BindingInputError("rig.json 尚未生成，请先跑完 S2 再重算蒙皮")
        self._require_idle(asset_id)
        snapshot = self._snapshot(asset_id, need_rig=False)
        self.store.set_binding_progress(asset_id, Stage.BUILD_RIG.value,
                                        "已排队，等待重算蒙皮…")
        return self._start(self.run_reskin, asset_id, snapshot)

    def _require_idle(self, asset_id: str) -> None:
        if self.is_busy(asset_id):
            raise StateError(f"素材 {asset_id} 的 S2 正在执行，请等它跑完")

    def _snapshot(self, asset_id: str, need_rig: bool = True) -> Tuple[bool, Optional[str]]:
        """起线程**之前**抓一份「上一次的绑定是否可用」快照 ``(产物齐备, 旧状态)``。

        必须抢在置 ``RUNNING`` 之前抓：:meth:`_fail_binding` 靠它决定失败时是保留上
        一次的 rig/skin 还是整条落 ``FAILED``，而置 RUNNING 会把 ``state`` 冲掉。
        """
        ok = self.store.skin_path(asset_id).exists()
        if need_rig:
            ok = ok and self.store.rig_path(asset_id).exists()
        return ok, (self.store.get_binding(asset_id) or {}).get("state")

    def _start(self, fn, asset_id: str, *args: Any) -> threading.Thread:
        t = threading.Thread(target=self._safe, args=(fn, asset_id, *args), daemon=True)
        t.start()
        return t

    def _safe(self, fn, asset_id: str, *args: Any) -> None:
        try:
            fn(asset_id, *args)
        except (BindingError, NotFoundError, StateError) as exc:
            logger.warning("后台 S2 失败 %s：%s", asset_id, exc)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _require_model(self, asset_id: str) -> Dict[str, Any]:
        """S2 准入：素材存在、是模型、且 S1 已落库（单位矫正过）。"""
        row = self.store.require_align_ready(asset_id)     # 抛 NotFoundError / StateError
        if row["kind"] != AssetKind.MODEL.value:
            raise BindingInputError("只有模型素材需要绑定骨骼（动画素材入库即可复用）")
        return row

    def _fail_binding(self, asset_id: str, reason: str, stage: str,
                      had_artifacts: bool, prev_state: Optional[str]) -> None:
        """失败落库：已有可用绑定产物时**保留**上一次的 rig/skin，只记原因。

        重跑失败（例如换了视角图后三角化崩了）不该把已经能用的蒙皮一并废掉 ——
        素材会因此退回到「必须重新绑定才能进 S3」的状态，而旧产物其实还是好的。
        """
        if had_artifacts and prev_state == BindingState.READY.value:
            self.store.update_binding(asset_id, state=BindingState.READY,
                                      stage=None, error=reason)
            logger.warning("S2 重跑失败（已保留上一次绑定）%s：%s", asset_id, reason)
            return
        self.store.mark_binding_failed(asset_id, reason, stage)


# ---- 单例 ----
_worker: Optional[JobWorker] = None


def get_job_worker(reload: bool = False) -> JobWorker:
    global _worker
    if _worker is None or reload:
        _worker = JobWorker()
    return _worker
