"""SQLite 持久化层（标准库 sqlite3，无第三方 ORM）。

职责：连接管理、建表、通用 CRUD 与事务封装。表：tasks / stages / artifacts。
上层领域逻辑（任务生命周期、磁盘产物、revision 乐观锁）在 task_store.py。

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
"""


def now_iso() -> str:
    """UTC ISO8601 时间戳（秒精度）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Database:
    """轻量 SQLite 封装。"""

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
