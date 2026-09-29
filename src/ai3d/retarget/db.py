"""SQLite 持久化层（标准库 sqlite3，无第三方 ORM）。

职责：连接管理、建表、通用 CRUD 与事务封装。

两套表并存（``/v1`` 与 ``/v2`` 各读写自己的一套，旧库靠 ``IF NOT EXISTS`` 自动补新表）：
- 旧（``/v1`` 一对一任务）：tasks / stages / artifacts，领域逻辑在 task_store.py。
- 新（``/v2`` 四阶段）：assets / bindings / jobs / job_stages / job_artifacts，
  领域逻辑在 asset_store.py。素材（模型/动画）与作业（模型 × 动画）拆开，
  动画入库一次可被任意模型复用；bindings 缓存 S2 结果供模型资产复用。

并发策略：WAL 日志 + 全局可重入锁串行化写；每次操作使用独立连接，避免跨线程复用。
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# 表结构
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id            TEXT PRIMARY KEY,
    state              TEXT NOT NULL,
    current_stage      TEXT,
    name               TEXT,                 -- 用户自定义任务名
    source_filename    TEXT,
    target_filename    TEXT,
    config             TEXT,                 -- JSON
    rig_revision       INTEGER NOT NULL DEFAULT 0,
    mapping_revision   INTEGER NOT NULL DEFAULT 0,
    overall_confidence REAL,
    review_reasons     TEXT,                 -- JSON list
    error              TEXT,
    artifact_dir       TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    message     TEXT,
    started_at  TEXT,
    finished_at TEXT,
    seq         INTEGER NOT NULL DEFAULT 0,
    UNIQUE(task_id, stage)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_stages_task ON stages(task_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);

-- ---------------------------------------------------------------------------
-- /v2 四阶段：素材库（assets + S2 结果缓存 bindings）与作业（jobs）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS assets (
    asset_id   TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,               -- 'model' | 'animation'
    name       TEXT,
    filename   TEXT,                        -- 上传原件名
    asset_dir  TEXT,
    state      TEXT NOT NULL,               -- CREATED/NORMALIZING/ALIGN_READY/READY/FAILED
    align_json TEXT,                        -- align.json 全文（S1 唯一产物）
    meta_json  TEXT,                        -- read_summary 等结构概要
    error      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bindings (
    asset_id    TEXT PRIMARY KEY REFERENCES assets(asset_id) ON DELETE CASCADE,
    state       TEXT NOT NULL,              -- PENDING/RUNNING/WAIT_VIEWS/READY/FAILED
    stage       TEXT,                       -- 当前子步骤（RENDER_VIEWS/POSE_INFER/SOLVE_RIG/BUILD_RIG）
    message     TEXT,                       -- 一行进度描述，前端直接展示
    confidence  REAL,
    revision    INTEGER NOT NULL DEFAULT 0, -- 人工微调关节的乐观锁
    rig_path    TEXT,
    skin_path   TEXT,
    report_path TEXT,
    error       TEXT,                       -- 失败原因（与 assets.error 分开：S1 与 S2 互不覆盖）
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    name             TEXT,
    model_asset_id   TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    anim_asset_id    TEXT REFERENCES assets(asset_id) ON DELETE SET NULL,
    state            TEXT NOT NULL,
    stage_group      TEXT,                  -- 'S1'|'S2'|'S3'|'S4'
    current_stage    TEXT,                  -- 九阶段子步骤名
    mapping_revision INTEGER NOT NULL DEFAULT 0,
    review_reasons   TEXT,                  -- JSON list
    error            TEXT,
    job_dir          TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    message     TEXT,
    started_at  TEXT,
    finished_at TEXT,
    seq         INTEGER NOT NULL DEFAULT 0,
    UNIQUE(job_id, stage)
);

CREATE TABLE IF NOT EXISTS job_artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_assets_kind_created ON assets(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_model ON jobs(model_asset_id);
CREATE INDEX IF NOT EXISTS idx_job_stages_job ON job_stages(job_id);
CREATE INDEX IF NOT EXISTS idx_job_artifacts_job ON job_artifacts(job_id);
"""


def now_iso() -> str:
    """UTC ISO8601 时间戳（秒精度）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    """轻量 SQLite 封装。"""

    # 后加列：(表, 列, 声明)。SQLite 的 ADD COLUMN 不支持 IF NOT EXISTS，故先查 table_info；
    # 建表语句里已含这些列，新库走 executescript，旧库走这里补齐。
    _ADDED_COLUMNS = (
        ("tasks", "name", "TEXT"),
        ("bindings", "stage", "TEXT"),
        ("bindings", "message", "TEXT"),
        ("bindings", "error", "TEXT"),
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init()

    # ---- 连接 ----
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    def init(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """旧库轻量迁移：按 :data:`_ADDED_COLUMNS` 补建后加列（幂等）。"""
        for table, column, decl in self._ADDED_COLUMNS:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ---- 通用执行 ----
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """执行写语句，返回受影响行数。"""
        with self._conn() as conn:
            cur = conn.execute(sql, tuple(params))
            return cur.rowcount

    def execute_in_tx(self, statements: List[tuple[str, Sequence[Any]]]) -> List[int]:
        """在单个事务内执行多条写语句；任一失败整体回滚。返回各语句 rowcount。"""
        with self._lock:
            conn = self._connect()
            counts: List[int] = []
            try:
                conn.execute("BEGIN")
                for sql, params in statements:
                    counts.append(conn.execute(sql, tuple(params)).rowcount)
                conn.execute("COMMIT")
                return counts
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return list(conn.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row is not None else None

    # ---- 便捷插入/更新 ----
    def insert(self, table: str, values: Dict[str, Any]) -> None:
        cols = list(values.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        self.execute(sql, [values[c] for c in cols])

    def update(self, table: str, values: Dict[str, Any], where: str,
               where_params: Sequence[Any] = ()) -> int:
        if not values:
            return 0
        sets = ", ".join(f"{k}=?" for k in values)
        sql = f"UPDATE {table} SET {sets} WHERE {where}"
        return self.execute(sql, [*values.values(), *where_params])

    def upsert(self, table: str, values: Dict[str, Any],
               conflict_cols: Iterable[str], update_cols: Iterable[str]) -> None:
        """INSERT ... ON CONFLICT(...) DO UPDATE。"""
        cols = list(values.keys())
        placeholders = ", ".join("?" for _ in cols)
        conflict = ", ".join(conflict_cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        )
        self.execute(sql, [values[c] for c in cols])


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None
