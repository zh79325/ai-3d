"""任务存储：DB 领域封装 + 磁盘产物 + revision 乐观锁。

一个 task = 对「目标模型 + 源动画」的一次迁移处理。上传即 create_task（落库 + 建目录），
之后凭 task_id 找回历史任务与其产物（服务重启后依然有效）。
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .db import Database, now_iso
from .schemas import (
    STAGE_ORDER,
    ArtifactInfo,
    ArtifactKind,
    JobConfig,
    JobInfo,
    JobState,
    JobSummary,
    Stage,
    StageInfo,
    StageStatus,
)
from .settings import RetargetSettings, get_settings

logger = logging.getLogger(__name__)

_STAGE_SEQ = {s: i for i, s in enumerate(STAGE_ORDER)}

# 各阶段产出的产物种类（用于 rerun 时清理下游）
_STAGE_ARTIFACTS: Dict[Stage, List[str]] = {
    Stage.NORMALIZE: [ArtifactKind.SOURCE_RAW, ArtifactKind.TARGET_RAW,
                      ArtifactKind.SOURCE, ArtifactKind.TARGET],
    Stage.PRECHECK: [],
    Stage.RENDER_VIEWS: [ArtifactKind.VIEWS],
    Stage.POSE_INFER: [],
    Stage.SOLVE_RIG: [],
    Stage.BUILD_RIG: [ArtifactKind.RIG],
    Stage.MAP_SOURCE: [ArtifactKind.MAPPING],
    Stage.RETARGET: [],
    Stage.EXPORT_VERIFY: [ArtifactKind.RESULT, ArtifactKind.REPORT],
}

_JSON_FIELDS = ("config", "review_reasons")
_ENUM_TASK_FIELDS = ("state", "current_stage")


class ConflictError(Exception):
    """revision 乐观锁冲突（PATCH 基于过期版本）。"""


class NotFoundError(Exception):
    """任务不存在。"""


class TaskStore:
    def __init__(self, settings: Optional[RetargetSettings] = None):
        self.settings = settings or get_settings()
        self.db = Database(self.settings.sqlite_file)

    # ------------------------------------------------------------------ #
    # 路径
    # ------------------------------------------------------------------ #
    def upload_dir(self, task_id: str) -> Path:
        """上传的原始文件直接与任务产物放同一目录（tasks/<task_id>/）。"""
        return self.task_dir(task_id)

    def task_dir(self, task_id: str) -> Path:
        return self.settings.task_dir(task_id)

    def views_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "views"

    def path_for(self, task_id: str, kind: ArtifactKind | str) -> Path:
        kind = ArtifactKind(kind)
        d = self.upload_dir(task_id) if kind in (
            ArtifactKind.SOURCE_RAW, ArtifactKind.TARGET_RAW) else self.task_dir(task_id)
        name = {
            ArtifactKind.SOURCE: "source.glb",
            ArtifactKind.TARGET: "target.glb",
            ArtifactKind.VIEWS: "views",
            ArtifactKind.RIG: "rig.json",
            ArtifactKind.MAPPING: "mapping.json",
            ArtifactKind.RESULT: "result.glb",
            ArtifactKind.REPORT: "report.json",
        }.get(kind)
        if name is None:  # raw 文件由调用方给出真实文件名
            return d
        return d / name

    # ------------------------------------------------------------------ #
    # 创建 / 查询
    # ------------------------------------------------------------------ #
    def create_task(self, source_filename: str, target_filename: str,
                    config: Optional[JobConfig] = None, *,
                    name: Optional[str] = None,
                    state: JobState = JobState.QUEUED) -> str:
        task_id = uuid.uuid4().hex[:12]
        cfg = config or JobConfig()
        self.upload_dir(task_id).mkdir(parents=True, exist_ok=True)
        self.task_dir(task_id).mkdir(parents=True, exist_ok=True)
        ts = now_iso()
        self.db.insert("tasks", {
            "task_id": task_id,
            "state": state.value,
            "current_stage": None,
            "name": name,
            "source_filename": source_filename,
            "target_filename": target_filename,
            "config": cfg.model_dump_json(),
            "rig_revision": 0,
            "mapping_revision": 0,
            "overall_confidence": None,
            "review_reasons": "[]",
            "error": None,
            "artifact_dir": str(self.task_dir(task_id)),
            "created_at": ts,
            "updated_at": ts,
        })
        # 预置全部阶段为 PENDING，便于前端展示进度
        for stage in STAGE_ORDER:
            self.db.insert("stages", {
                "task_id": task_id, "stage": stage.value,
                "status": StageStatus.PENDING.value, "message": None,
                "started_at": None, "finished_at": None, "seq": _STAGE_SEQ[stage],
            })
        logger.info("已创建任务 %s（name=%s 源=%s 目标=%s）",
                    task_id, name, source_filename, target_filename)
        return task_id

    def get_row(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM tasks WHERE task_id=?", (task_id,))
        return dict(row) if row else None

    def exists(self, task_id: str) -> bool:
        return self.get_row(task_id) is not None

    def get_job_info(self, task_id: str) -> Optional[JobInfo]:
        row = self.get_row(task_id)
        if row is None:
            return None
        stages = self.get_stages(task_id)
        artifacts = self.get_artifacts(task_id)
        return JobInfo(
            task_id=row["task_id"],
            state=JobState(row["state"]),
            current_stage=Stage(row["current_stage"]) if row["current_stage"] else None,
            source_filename=row["source_filename"],
            target_filename=row["target_filename"],
            name=row.get("name"),
            config=_job_config_from_json(row["config"]),
            rig_revision=row["rig_revision"],
            mapping_revision=row["mapping_revision"],
            overall_confidence=row["overall_confidence"],
            review_reasons=_parse_json(row["review_reasons"], []),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            stages=stages,
            artifacts=artifacts,
        )

    def list_jobs(self, limit: int = 50, offset: int = 0) -> Tuple[List[JobSummary], int]:
        total = int(self.db.scalar("SELECT COUNT(*) FROM tasks") or 0)
        rows = self.db.query(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        summaries = [
            JobSummary(
                task_id=r["task_id"], state=JobState(r["state"]),
                current_stage=Stage(r["current_stage"]) if r["current_stage"] else None,
                source_filename=r["source_filename"], target_filename=r["target_filename"],
                name=r["name"],
                overall_confidence=r["overall_confidence"],
                created_at=r["created_at"], updated_at=r["updated_at"],
            ) for r in rows
        ]
        return summaries, total

    def count_by_state(self) -> Dict[str, int]:
        rows = self.db.query("SELECT state, COUNT(*) c FROM tasks GROUP BY state")
        return {r["state"]: r["c"] for r in rows}

    # ------------------------------------------------------------------ #
    # 更新
    # ------------------------------------------------------------------ #
    def update_task(self, task_id: str, **fields: Any) -> None:
        values: Dict[str, Any] = {}
        for k, v in fields.items():
            if v is None and k not in ("error", "overall_confidence", "current_stage"):
                continue
            if k in _ENUM_TASK_FIELDS and hasattr(v, "value"):
                v = v.value
            if k in _JSON_FIELDS and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            values[k] = v
        values["updated_at"] = now_iso()
        self.db.update("tasks", values, "task_id=?", (task_id,))

    def set_state(self, task_id: str, state: JobState,
                  current_stage: Optional[Stage] = None, error: Optional[str] = None) -> None:
        self.update_task(task_id, state=state, current_stage=current_stage, error=error)

    def set_stage(self, task_id: str, stage: Stage, status: StageStatus,
                  message: Optional[str] = None) -> None:
        stage = Stage(stage)
        existing = self.db.query_one(
            "SELECT started_at, finished_at, message FROM stages WHERE task_id=? AND stage=?",
            (task_id, stage.value))
        started = existing["started_at"] if existing else None
        finished = existing["finished_at"] if existing else None
        # message=None 表示保留已有消息（不覆盖）
        msg = message if message is not None else (existing["message"] if existing else None)
        ts = now_iso()
        if status == StageStatus.RUNNING and not started:
            started = ts
        if status in (StageStatus.DONE, StageStatus.FAILED, StageStatus.SKIPPED):
            finished = ts
            started = started or ts
        self.db.upsert("stages", {
            "task_id": task_id, "stage": stage.value, "status": status.value,
            "message": msg, "started_at": started, "finished_at": finished,
            "seq": _STAGE_SEQ[stage],
        }, conflict_cols=["task_id", "stage"],
            update_cols=["status", "message", "started_at", "finished_at", "seq"])

    def get_stages(self, task_id: str) -> List[StageInfo]:
        rows = self.db.query(
            "SELECT * FROM stages WHERE task_id=? ORDER BY seq ASC", (task_id,))
        return [
            StageInfo(stage=Stage(r["stage"]), status=StageStatus(r["status"]),
                      message=r["message"], started_at=r["started_at"],
                      finished_at=r["finished_at"])
            for r in rows
        ]

    def register_artifact(self, task_id: str, kind: ArtifactKind | str, path: str | Path) -> None:
        kind = ArtifactKind(kind)
        self.db.upsert("artifacts", {
            "task_id": task_id, "kind": kind.value, "path": str(path),
            "created_at": now_iso(),
        }, conflict_cols=["task_id", "kind"], update_cols=["path", "created_at"])

    def get_artifacts(self, task_id: str) -> List[ArtifactInfo]:
        rows = self.db.query(
            "SELECT * FROM artifacts WHERE task_id=? ORDER BY id ASC", (task_id,))
        return [ArtifactInfo(kind=r["kind"], path=r["path"], created_at=r["created_at"])
                for r in rows]

    def get_artifact_path(self, task_id: str, kind: ArtifactKind | str) -> Optional[str]:
        kind = ArtifactKind(kind)
        row = self.db.query_one(
            "SELECT path FROM artifacts WHERE task_id=? AND kind=?", (task_id, kind.value))
        return row["path"] if row else None

    # ------------------------------------------------------------------ #
    # revision 乐观锁（rig / mapping）
    # ------------------------------------------------------------------ #
    def save_rig(self, task_id: str, expected_revision: int, rig: Dict[str, Any]) -> int:
        return self._save_revisioned(task_id, "rig_revision", expected_revision,
                                     ArtifactKind.RIG, "rig.json", rig)

    def save_mapping(self, task_id: str, expected_revision: int,
                     mapping: Dict[str, Any]) -> int:
        return self._save_revisioned(task_id, "mapping_revision", expected_revision,
                                     ArtifactKind.MAPPING, "mapping.json", mapping)

    def _save_revisioned(self, task_id: str, rev_col: str, expected: int,
                         kind: ArtifactKind, filename: str, payload: Dict[str, Any]) -> int:
        row = self.get_row(task_id)
        if row is None:
            raise NotFoundError(task_id)
        path = self.task_dir(task_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        new_rev = expected + 1
        affected = self.db.execute(
            f"UPDATE tasks SET {rev_col}=?, updated_at=? WHERE task_id=? AND {rev_col}=?",
            (new_rev, now_iso(), task_id, expected))
        if affected == 0:
            raise ConflictError(
                f"{rev_col} 冲突：期望 {expected}，实际 {row[rev_col]}（请刷新后重试）")
        self.register_artifact(task_id, kind, path)
        return new_rev

    def load_json_artifact(self, task_id: str, filename: str) -> Optional[Dict[str, Any]]:
        path = self.task_dir(task_id) / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # rerun：清理下游阶段与产物
    # ------------------------------------------------------------------ #
    def reset_from_stage(self, task_id: str, from_stage: Stage) -> None:
        from_stage = Stage(from_stage)
        start_seq = _STAGE_SEQ[from_stage]
        downstream = [s for s in STAGE_ORDER if _STAGE_SEQ[s] >= start_seq]
        # 清理产物（磁盘 + DB）
        for stage in downstream:
            for kind in _STAGE_ARTIFACTS.get(stage, []):
                kind = ArtifactKind(kind)
                p = self.get_artifact_path(task_id, kind)
                if p:
                    fp = Path(p)
                    if fp.is_dir():
                        shutil.rmtree(fp, ignore_errors=True)
                    elif fp.exists():
                        fp.unlink(missing_ok=True)
                self.db.execute(
                    "DELETE FROM artifacts WHERE task_id=? AND kind=?", (task_id, kind.value))
        # 重置阶段状态
        for stage in downstream:
            self.db.update("stages", {
                "status": StageStatus.PENDING.value, "message": None,
                "started_at": None, "finished_at": None,
            }, "task_id=? AND stage=?", (task_id, stage.value))
        self.update_task(task_id, state=JobState.QUEUED, current_stage=from_stage,
                         error=None, overall_confidence=None, review_reasons=[])

    def delete_task(self, task_id: str, remove_files: bool = True) -> bool:
        row = self.get_row(task_id)
        if row is None:
            return False
        self.db.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
        if remove_files:
            shutil.rmtree(self.task_dir(task_id), ignore_errors=True)
        return True


def _parse_json(raw: Optional[str], default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _job_config_from_json(raw: Optional[str]) -> JobConfig:
    data = _parse_json(raw, None)
    if not isinstance(data, dict):
        return JobConfig()
    try:
        return JobConfig(**data)
    except Exception:  # noqa: BLE001 - 容错：非法配置回退默认
        return JobConfig()


# ---- 单例 ----
_store: Optional[TaskStore] = None


def get_store(reload: bool = False) -> TaskStore:
    global _store
    if _store is None or reload:
        _store = TaskStore()
    return _store
