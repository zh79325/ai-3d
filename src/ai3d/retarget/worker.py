"""阶段状态机 worker。

P0：NORMALIZE（输入→GLB）、PRECHECK（结构校验）、RENDER_VIEWS（等待前端回传视角图）。
P1：确定性闭环 BUILD_RIG（比例骨架+自动蒙皮）→ MAP_SOURCE（别名映射）→
    RETARGET（增量重定向烘焙）→ EXPORT_VERIFY（组装 result.glb + 回读验证）。
    当 config.enable_pose_ai=False 时跳过 RENDER_VIEWS/POSE_INFER/SOLVE_RIG。
P2：POSE_INFER（DWPose/YOLO 多视角 2D 关键点）→ SOLVE_RIG（加权三角化+约束→三维关节），
    BUILD_RIG 复用求解出的 rig.json 做蒙皮。

运行方式：server 在后台线程调用 ``Worker.run_pipeline(task_id, from_stage)``。

各阶段的算法编排已抽到 :mod:`stages`（纯函数，只认产物路径），``/v2`` 的
:class:`job_worker.JobWorker` 共用同一份；本类只负责 ``/v1`` 的状态机与产物登记。
"""

from __future__ import annotations

import json
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import foot_ik, glb_io, stages
from .glb_build import build_result_glb
from .mapping import build_mapping
from .retarget import (
    _global_rest_positions,
    retarget_animation,
    stack_rotations,
    unstack_rotations,
    with_fps,
)
from .schemas import (
    STAGE_ORDER,
    ArtifactKind,
    JobConfig,
    JobState,
    Stage,
    StageStatus,
)
from .settings import get_settings
from .skeleton import JOINTS, Rig
from .task_store import TaskStore, _job_config_from_json, get_store

logger = logging.getLogger(__name__)


class StepResult(str, Enum):
    CONTINUE = "CONTINUE"   # 阶段完成，继续下一阶段
    WAIT = "WAIT"           # 需要外部输入（前端回传），挂起为 WAITING
    STOP = "STOP"           # 到此为止（下游未实现），挂起为 WAITING


class Worker:
    def __init__(self, store: Optional[TaskStore] = None):
        self.store = store or get_store()
        self._handlers: Dict[Stage, Callable[[str], StepResult]] = {
            Stage.NORMALIZE: self._normalize,
            Stage.PRECHECK: self._precheck,
            Stage.RENDER_VIEWS: self._render_views,
            Stage.POSE_INFER: self._pose_infer,
            Stage.SOLVE_RIG: self._solve_rig,
            Stage.BUILD_RIG: self._build_rig,
            Stage.MAP_SOURCE: self._map_source,
            Stage.RETARGET: self._retarget,
            Stage.EXPORT_VERIFY: self._export_verify,
        }

    # ------------------------------------------------------------------ #
    # 调度
    # ------------------------------------------------------------------ #
    def run_pipeline(self, task_id: str, from_stage: Optional[Stage] = None,
                     force: bool = False, to_stage: Optional[Stage] = None) -> None:
        if not self.store.exists(task_id):
            logger.warning("run_pipeline：任务不存在 %s", task_id)
            return
        start = from_stage or self._resume_stage(task_id)
        start_idx = STAGE_ORDER.index(Stage(start))
        end_idx = STAGE_ORDER.index(Stage(to_stage)) if to_stage else len(STAGE_ORDER) - 1
        self.store.update_task(task_id, state=JobState.RUNNING, error=None)
        for stage in STAGE_ORDER[start_idx:end_idx + 1]:
            self.store.update_task(task_id, current_stage=stage)
            if self._should_skip(task_id, stage):
                self.store.set_stage(task_id, stage, StageStatus.SKIPPED,
                                     "确定性路径跳过（未启用姿态 AI）")
                continue
            gate_reasons = None
            if stage == Stage.EXPORT_VERIFY:
                gate = self._gate(task_id, force=force)
                if gate is not None:
                    reasons, blocking = gate
                    if blocking:
                        self.store.update_task(task_id, state=JobState.NEEDS_REVIEW,
                                               review_reasons=reasons)
                        self.store.set_stage(task_id, stage, StageStatus.WAITING,
                                             "门控阻断导出：" + "；".join(reasons))
                        return
                    gate_reasons = reasons
            self.store.set_stage(task_id, stage, StageStatus.RUNNING)
            try:
                result = self._handlers[stage](task_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("阶段 %s 失败（task=%s）", stage.value, task_id)
                self.store.set_stage(task_id, stage, StageStatus.FAILED, str(exc))
                self.store.update_task(task_id, state=JobState.FAILED, error=str(exc))
                return
            if result == StepResult.CONTINUE:
                self.store.set_stage(task_id, stage, StageStatus.DONE)
                if gate_reasons is not None:
                    # 非阻断审核：EXPORT_VERIFY 已产出 result.glb（审核期「🎬 迁移动画」
                    # 预览用），挂起等待人工处置，不自动置 DONE
                    self.store.update_task(task_id, state=JobState.NEEDS_REVIEW,
                                           review_reasons=gate_reasons)
                    return
                continue
            if result == StepResult.WAIT:
                self.store.set_stage(task_id, stage, StageStatus.WAITING,
                                     "等待前端回传多视角图")
                self.store.update_task(task_id, state=JobState.WAITING)
                return
            # STOP（防御分支：当前所有阶段均已实现，正常不会到达）
            self.store.set_stage(task_id, stage, StageStatus.WAITING,
                                 f"阶段 {stage.value} 主动挂起，等待外部输入")
            self.store.update_task(task_id, state=JobState.WAITING)
            return
        # 部分运行（如仅目标侧骨骼检测）到此为止，挂起等待后续操作
        if to_stage is not None:
            self.store.update_task(task_id, state=JobState.WAITING, current_stage=Stage(to_stage))
            return
        # 全部阶段完成
        self.store.update_task(task_id, state=JobState.DONE, current_stage=Stage.EXPORT_VERIFY)

    def start_async(self, task_id: str, from_stage: Optional[Stage] = None,
                    force: bool = False, to_stage: Optional[Stage] = None) -> threading.Thread:
        t = threading.Thread(target=self.run_pipeline,
                             args=(task_id, from_stage, force, to_stage), daemon=True)
        t.start()
        return t

    def _resume_stage(self, task_id: str) -> Stage:
        row = self.store.get_row(task_id)
        cur = row.get("current_stage") if row else None
        if cur:
            try:
                return Stage(cur)
            except ValueError:
                pass
        return Stage.NORMALIZE

    # ------------------------------------------------------------------ #
    # 阶段实现
    # ------------------------------------------------------------------ #
    def _normalize(self, task_id: str) -> StepResult:
        """把上传的原始文件归一化为 source.glb / target.glb（允许仅有其一）。"""
        s = self.store
        raw_source = s.get_artifact_path(task_id, ArtifactKind.SOURCE_RAW)
        raw_target = s.get_artifact_path(task_id, ArtifactKind.TARGET_RAW)
        if not raw_source and not raw_target:
            raise RuntimeError("缺少原始上传文件（source_raw/target_raw）")
        if raw_source:
            src_glb = s.path_for(task_id, ArtifactKind.SOURCE)
            glb_io.normalize_to_glb(raw_source, src_glb)
            s.register_artifact(task_id, ArtifactKind.SOURCE, src_glb)
        if raw_target:
            tgt_glb = s.path_for(task_id, ArtifactKind.TARGET)
            glb_io.normalize_to_glb(raw_target, tgt_glb)
            s.register_artifact(task_id, ArtifactKind.TARGET, tgt_glb)
        return StepResult.CONTINUE

    def _precheck(self, task_id: str) -> StepResult:
        """结构校验：目标须含网格；源可缺（仅目标侧检测时告警不阻断）。"""
        s = self.store
        src_path = s.get_artifact_path(task_id, ArtifactKind.SOURCE)
        tgt_glb = Path(s.get_artifact_path(task_id, ArtifactKind.TARGET))
        if not tgt_glb or not Path(tgt_glb).exists():
            raise RuntimeError("目标模型未归一化（target.glb 缺失），无法检测")
        warnings = []
        if src_path and Path(src_path).exists():
            src_sum = glb_io.read_summary(Path(src_path))
            if not src_sum["has_skeleton"]:
                warnings.append("源文件未检测到骨架(skin)")
            if not src_sum["has_animation"]:
                warnings.append("源文件未检测到动画(animation)")
        else:
            src_sum = {"meshes": 0, "skins": 0, "animations": 0, "nodes": 0,
                       "has_skeleton": False, "has_animation": False}
            warnings.append("源文件未上传（仅做目标侧骨骼检测）")
        tgt_sum = glb_io.read_summary(tgt_glb)
        if tgt_sum["meshes"] < 1:
            raise RuntimeError("目标模型不含任何网格（mesh），无法迁移")
        precheck = {"source": src_sum, "target": tgt_sum, "warnings": warnings}
        (s.task_dir(task_id) / "precheck.json").write_text(
            json.dumps(precheck, ensure_ascii=False, indent=2), encoding="utf-8")
        msg = (f"源: mesh={src_sum['meshes']} skin={src_sum['skins']} "
               f"anim={src_sum['animations']}; 目标: mesh={tgt_sum['meshes']} "
               f"node={tgt_sum['nodes']}")
        if warnings:
            msg += " | 告警: " + "; ".join(warnings)
        s.set_stage(task_id, Stage.PRECHECK, StageStatus.RUNNING, msg)
        return StepResult.CONTINUE

    def _render_views(self, task_id: str) -> StepResult:
        """等待前端渲染并回传多视角图。"""
        if self.store.get_artifact_path(task_id, ArtifactKind.VIEWS):
            return StepResult.CONTINUE
        return StepResult.WAIT

    # ------------------------------------------------------------------ #
    # P2 自动求解（多视角 2D 关键点 → 三维关节）
    # ------------------------------------------------------------------ #
    def _pose_infer(self, task_id: str) -> StepResult:
        """对前端回传的多视角图逐张推理，输出 2D 关键点 + 置信度（detections.json）。"""
        s = self.store
        views_path = s.get_artifact_path(task_id, ArtifactKind.VIEWS)
        if not views_path:
            raise RuntimeError("缺少视角图，无法推理（RENDER_VIEWS 未完成）")
        detections = stages.infer_views(
            Path(views_path), s.task_dir(task_id) / stages.DETECTIONS_FILE)
        n_kp = sum(len(d) for d in detections.values())
        s.set_stage(task_id, Stage.POSE_INFER, StageStatus.RUNNING,
                    f"{len(detections)} 个视角推理完成，共 {n_kp} 个 2D 关键点")
        return StepResult.CONTINUE

    def _solve_rig(self, task_id: str) -> StepResult:
        """多视角加权三角化 + 对称/比例/骨长约束 → 三维关节，落盘 rig.json。"""
        s = self.store
        d = s.task_dir(task_id)
        rig = stages.solve_rig(
            Path(s.get_artifact_path(task_id, ArtifactKind.TARGET)),
            d / stages.DETECTIONS_FILE, d / stages.RIG_FILE, d / stages.VIEW_SPEC_FILE)
        s.register_artifact(task_id, ArtifactKind.RIG, d / stages.RIG_FILE)
        s.update_task(task_id, overall_confidence=round(rig.overall_confidence(), 4))
        solved = sum(1 for j in JOINTS if rig.source.get(j) == "solved")
        s.set_stage(task_id, Stage.SOLVE_RIG, StageStatus.RUNNING,
                    f"三角化求解 {solved}/{len(JOINTS)} 关节"
                    f"（overall={rig.overall_confidence():.2f}）")
        return StepResult.CONTINUE

    # ------------------------------------------------------------------ #
    # P1 确定性闭环
    # ------------------------------------------------------------------ #
    def _config(self, task_id: str) -> JobConfig:
        row = self.store.get_row(task_id)
        return _job_config_from_json(row["config"]) if row else JobConfig()

    def _should_skip(self, task_id: str, stage: Stage) -> bool:
        """未启用姿态 AI 时，跳过依赖多视角图/推理/求解的阶段。"""
        cfg = self._config(task_id)
        return (not cfg.enable_pose_ai) and stage in (
            Stage.RENDER_VIEWS, Stage.POSE_INFER, Stage.SOLVE_RIG)

    def _gate(self, task_id: str, force: bool = False):
        """置信度门控：返回 None（放行）或 (reasons, blocking)。

        规则（gating 配置）：overall>=auto_pass 且核心关节达标 → 放行；
        review_min<=overall<auto_pass 或核心关节<core_joint_min → 审核（非阻断）；
        overall<review_min → 阻断导出。force 或（非阻断且 auto_continue_on_review）→ 放行。
        """
        if force:
            return None
        cfg = self._config(task_id)
        g = get_settings().gating
        rig_dict = self.store.load_json_artifact(task_id, "rig.json")
        if rig_dict is None:
            return None
        rig = Rig.from_dict(rig_dict)
        conf = rig.overall_confidence()
        core_fail = rig.core_failure(g.core_joint_min)
        reasons: List[str] = []
        blocking = False
        if conf < g.review_min:
            reasons.append(f"整体置信度 {conf:.2f} < {g.review_min}，已阻断导出")
            blocking = True
        elif conf < g.auto_pass:
            reasons.append(f"整体置信度 {conf:.2f} 处于审核区间 [{g.review_min}, {g.auto_pass})")
        if core_fail:
            reasons.append(f"核心关节置信度 < {g.core_joint_min}：{', '.join(core_fail)}")
        if not reasons:
            return None
        if not blocking and cfg.auto_continue_on_review:
            return None
        return reasons, blocking

    def _build_rig(self, task_id: str) -> StepResult:
        """生成语义骨架（比例/已求解）+ 自动蒙皮，落盘 rig.json 与 skin.npz。"""
        s = self.store
        d = s.task_dir(task_id)
        rig_dict = s.load_json_artifact(task_id, stages.RIG_FILE)
        rig, n_verts, skin_report = stages.build_rig(
            Path(s.get_artifact_path(task_id, ArtifactKind.TARGET)), d,
            Rig.from_dict(rig_dict) if rig_dict else None)
        s.register_artifact(task_id, ArtifactKind.RIG, d / stages.RIG_FILE)
        s.update_task(task_id, overall_confidence=round(rig.overall_confidence(), 4))
        s.set_stage(task_id, Stage.BUILD_RIG, StageStatus.RUNNING,
                    f"骨架 {len(JOINTS)} 关节（height={rig.height:.3f}m）"
                    f"+ 蒙皮 {n_verts} 顶点（{stages.skin_summary(skin_report)}）")
        return StepResult.CONTINUE

    def _map_source(self, task_id: str) -> StepResult:
        """提取源骨架关节，按别名库映射到 22 语义关节，落盘 mapping.json。"""
        s = self.store
        settings = get_settings()
        src_glb = Path(s.get_artifact_path(task_id, ArtifactKind.SOURCE))
        g = glb_io._load(src_glb)
        skin = glb_io.extract_skin(g)
        if skin is None:
            raise RuntimeError("源文件不含骨架(skin)，无法映射")
        nodes = glb_io.extract_nodes(g)
        parent_of: Dict[int, Optional[int]] = {n["index"]: None for n in nodes}
        for n in nodes:
            for c in n.get("children") or []:
                parent_of[c] = n["index"]
        positions = _global_rest_positions(skin, nodes)
        source_joints = []
        for ni, nm in zip(skin["joints"], skin["names"]):
            pj = positions.get(ni)
            source_joints.append({
                "node": ni, "name": nm,
                "parent": parent_of.get(ni),
                "position": pj.tolist() if pj is not None else None,
            })
        mapping = build_mapping(source_joints, cfg=settings.mapping)
        map_path = s.task_dir(task_id) / "mapping.json"
        map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        s.register_artifact(task_id, ArtifactKind.MAPPING, map_path)
        s.set_stage(task_id, Stage.MAP_SOURCE, StageStatus.RUNNING,
                    f"源骨映射 {mapping['matched']}/{mapping['total']}"
                    f"（coverage={mapping['coverage']}）")
        if mapping["matched"] == 0:
            raise RuntimeError("源骨架无任何关节匹配到语义骨架（检查命名或提供 mapping 覆盖）")
        return StepResult.CONTINUE

    def _retarget(self, task_id: str) -> StepResult:
        """重定向源动画到目标骨架，逐帧烘焙，落盘 anim.npz。"""
        s = self.store
        settings = get_settings()
        src_glb = Path(s.get_artifact_path(task_id, ArtifactKind.SOURCE))
        mapping = s.load_json_artifact(task_id, "mapping.json")
        rig_dict = s.load_json_artifact(task_id, "rig.json")
        if mapping is None or rig_dict is None:
            raise RuntimeError("缺少 mapping.json / rig.json，无法重定向")
        rig = Rig.from_dict(rig_dict)
        cfg = self._config(task_id)
        rt_cfg = with_fps(settings.retarget, cfg.fps)
        result = retarget_animation(src_glb, mapping, rig, rt_cfg)
        # 脚部 IK 锁定（支撑相防滑步）；仅当任务与全局配置都启用时生效
        ik_msg = ""
        if cfg.foot_lock and settings.retarget.foot_lock:
            result["rotations"], ik_info = foot_ik.apply_foot_lock(
                rig, result["rotations"], result["root_translations"],
                result["times"], settings.retarget)
            result["meta"]["foot_ik"] = ik_info
            if ik_info.get("applied"):
                ik_msg = (f", 脚部锁定 L={ik_info['windows']['l']}"
                          f"/R={ik_info['windows']['r']} 窗口")
        (s.task_dir(task_id) / "retarget_meta.json").write_text(
            json.dumps(result["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        rot_stack = stack_rotations(result["rotations"], result["meta"]["frames"])
        np.savez(s.task_dir(task_id) / "anim.npz",
                 times=result["times"], rotations=rot_stack,
                 root_translations=result["root_translations"])
        m = result["meta"]
        s.set_stage(task_id, Stage.RETARGET, StageStatus.RUNNING,
                    f"烘焙 {m['frames']} 帧（duration={m['duration']}s, "
                    f"height_scale={m['height_scale']}{ik_msg}）")
        return StepResult.CONTINUE

    def _export_verify(self, task_id: str) -> StepResult:
        """组装 result.glb（mesh+skin+joints+animation）并回读验证，写 report.json。"""
        s = self.store
        settings = get_settings()
        tgt_glb = Path(s.get_artifact_path(task_id, ArtifactKind.TARGET))
        g = glb_io._load(tgt_glb)
        mesh = glb_io.merge_mesh(g)
        rig = Rig.from_dict(s.load_json_artifact(task_id, "rig.json"))
        mapping = s.load_json_artifact(task_id, "mapping.json") or {}
        skin_np = np.load(s.task_dir(task_id) / "skin.npz")
        anim_np = np.load(s.task_dir(task_id) / "anim.npz")
        rotations = unstack_rotations(anim_np["rotations"])
        result_glb = s.path_for(task_id, ArtifactKind.RESULT)
        build_result_glb(
            mesh, rig.heads, skin_np["joints"], skin_np["weights"],
            anim_np["times"], rotations, anim_np["root_translations"], result_glb,
            src_gltf=g)
        summary = glb_io.read_summary(result_glb)
        problems = []
        if summary["meshes"] < 1:
            problems.append("结果缺少网格")
        if not summary["has_skeleton"]:
            problems.append("结果缺少骨架(skin)")
        if not summary["has_animation"]:
            problems.append("结果缺少动画(animation)")
        joint_counts = summary["skin_joint_counts"] or [0]
        if joint_counts[0] != len(JOINTS):
            problems.append(f"骨架关节数={joint_counts[0]}≠{len(JOINTS)}")
        rt_meta = s.load_json_artifact(task_id, "retarget_meta.json") or {}
        gate_cfg = settings.gating
        conf = float(rig.overall_confidence())
        core_fail = rig.core_failure(gate_cfg.core_joint_min)
        method_counts: Dict[str, int] = {}
        for it in mapping.get("items", []):
            if it.get("source_node") is not None:
                method_counts[it["method"]] = method_counts.get(it["method"], 0) + 1
        if conf >= gate_cfg.auto_pass and not core_fail:
            decision = "PASS"
        elif conf >= gate_cfg.review_min:
            decision = "REVIEW"
        else:
            decision = "BLOCK"
        report = {
            "task_id": task_id,
            "result_glb": str(result_glb),
            "verify": {"ok": not problems, "problems": problems, "summary": summary},
            "rig": {
                "height": rig.height,
                "confidence": round(conf, 4),
                "core_failures": core_fail,
                "per_joint_confidence": {j: round(float(rig.confidence.get(j, 0.0)), 4)
                                         for j in JOINTS},
                "sources": dict(rig.source),
            },
            "mapping": {"matched": mapping.get("matched"),
                        "coverage": mapping.get("coverage"),
                        "methods": method_counts},
            "retarget": rt_meta,
            "gating": {"decision": decision, "auto_pass": gate_cfg.auto_pass,
                       "review_min": gate_cfg.review_min,
                       "core_joint_min": gate_cfg.core_joint_min},
        }
        report_path = s.task_dir(task_id) / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        s.register_artifact(task_id, ArtifactKind.RESULT, result_glb)
        s.register_artifact(task_id, ArtifactKind.REPORT, report_path)
        if problems:
            raise RuntimeError("结果回读验证失败：" + "; ".join(problems))
        s.set_stage(task_id, Stage.EXPORT_VERIFY, StageStatus.RUNNING,
                    f"result.glb 验证通过（joints={joint_counts[0]}, "
                    f"anim_channels={sum(summary['anim_channel_counts'])}）")
        return StepResult.CONTINUE


# ---- 单例 ----
_worker: Optional[Worker] = None


def get_worker(reload: bool = False) -> Worker:
    global _worker
    if _worker is None or reload:
        _worker = Worker()
    return _worker
