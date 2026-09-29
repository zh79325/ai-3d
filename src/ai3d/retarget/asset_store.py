"""``/v2`` 素材库存储：assets / bindings 的 DB 领域封装 + 磁盘产物布局。

与 :mod:`task_store`（``/v1`` 一对一任务）完全分离：本模块只读写 ``assets`` /
``bindings`` 表与 ``output/assets/<asset_id>/``，绝不触碰 ``tasks`` 表与
``output/tasks/``（后者原地封存，只读不删）。

目录布局（``output/assets/<asset_id>/``）::

    <name>_raw.<ext>     上传原件（保留原始扩展名，便于复查与重跑）
    asset.glb            格式归一 + S1 对齐后的产物（前端预览与后续阶段的唯一输入）
    align.json           S1 唯一产物：OBB / 六面 / 单位 / auto / manual / final
    meta.json            glb_io.read_summary 的结构概要
    rig.json             S2 语义骨架（人工微调走 revision 乐观锁）
    skin.npz             S2 蒙皮权重
    skin_report.json     S2 蒙皮质量报告
    view_spec.json       S2 下发给前端的多视角渲染规格（三角化读同一份）
    detections.json      S2 多视角 2D 关键点推理结果
    views/               S2 前端回传的多视角图
    unirig.log           S2 UniRig 链路的全部执行输出（前端按字节 offset 增量轮询）
    unirig/              UniRig 中间产物（raw_data / predict_skeleton / predict_skin npz）

状态机：``CREATED → NORMALIZING → ALIGN_READY → READY``，任一步失败落 ``FAILED``。
``ALIGN_READY`` 是 S2 的硬性准入门槛（见 :meth:`AssetStore.require_align_ready`）。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .db import Database, now_iso
from .schemas_v2 import AssetInfo, AssetKind, AssetState, AssetSummary, BindingInfo, BindingState
from .settings import RetargetSettings, get_settings

logger = logging.getLogger(__name__)

# 目录内固定文件名（原件名带用户上传的扩展名，单独处理）
ALIGN_FILE = "align.json"
GLB_FILE = "asset.glb"
META_FILE = "meta.json"
RIG_FILE = "rig.json"
SKIN_FILE = "skin.npz"
SKIN_REPORT_FILE = "skin_report.json"
DETECTIONS_FILE = "detections.json"
VIEW_SPEC_FILE = "view_spec.json"
VIEWS_DIR = "views"
UNIRIG_LOG_FILE = "unirig.log"
UNIRIG_WORK_DIR = "unirig"

# 单次日志增量拉取的字节上限（前端可用 max_bytes 覆盖）：推理几分钟能刷出几 MB，
# 一次全塞给浏览器会卡住页面，故按 offset 分片、前端只留尾部
LOG_CHUNK_BYTES = 256 * 1024

# 视角图只收这几种扩展名（与 inference.list_view_images 的可读集一致）
VIEW_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})

# 只允许字母/数字/下划线/点/连字符，且先取 Path().name 去掉任何目录成分（防穿越）
_UNSAFE_RE = re.compile(r"[^\w.\-]+", re.UNICODE)
_JSON_ASSET_FIELDS = ("align_json", "meta_json")
_ENUM_ASSET_FIELDS = ("state", "kind")
# 允许显式写 None 的列（其余字段传 None 视为「不改」）
_NULLABLE_ASSET_FIELDS = ("error", "filename", "name")


class NotFoundError(Exception):
    """素材不存在。"""


class ConflictError(Exception):
    """revision 乐观锁冲突（PATCH 基于过期版本）。"""


class StateError(Exception):
    """状态门槛不满足（如单位未矫正就想进 S2 绑定）。"""


def safe_stem(name: Optional[str], fallback: str = "asset") -> str:
    """把用户给的文件名/素材名压成安全的文件基名（不带扩展名）。

    去目录成分（防穿越）、去扩展名、替换非法字符、限长 64：
    ``hero.glb`` → ``hero``，``../../x`` → ``x``，``my hero.fbx`` → ``my_hero``。
    """
    stem = _UNSAFE_RE.sub("_", Path(str(name or "")).stem).strip("._")
    return stem[:64] or fallback


def parse_json(raw: Optional[str], default: Any) -> Any:
    """DB 里的 JSON 列容错解析（脏数据回退默认值，不让列表接口整体 500）。"""
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


class AssetStore:
    """素材库领域层。"""

    def __init__(self, settings: Optional[RetargetSettings] = None):
        self.settings = settings or get_settings()
        self.db = Database(self.settings.sqlite_file)

    # ------------------------------------------------------------------ #
    # 路径
    # ------------------------------------------------------------------ #
    def asset_dir(self, asset_id: str) -> Path:
        return self.settings.asset_dir(asset_id)

    def glb_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / GLB_FILE

    def align_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / ALIGN_FILE

    def meta_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / META_FILE

    def views_dir(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / VIEWS_DIR

    def view_spec_path(self, asset_id: str) -> Path:
        """下发给前端的多视角渲染规格（SOLVE_RIG 要读同一份相机参数）。"""
        return self.asset_dir(asset_id) / VIEW_SPEC_FILE

    def detections_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / DETECTIONS_FILE

    def rig_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / RIG_FILE

    def skin_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / SKIN_FILE

    def skin_report_path(self, asset_id: str) -> Path:
        return self.asset_dir(asset_id) / SKIN_REPORT_FILE

    def unirig_log_path(self, asset_id: str) -> Path:
        """UniRig 链路的全部执行输出（``unirig.log_capture`` 写入，覆盖式）。"""
        return self.asset_dir(asset_id) / UNIRIG_LOG_FILE

    def unirig_work_dir(self, asset_id: str) -> Path:
        """UniRig 中间产物目录（raw / 骨架 / 蒙皮 npz）。"""
        return self.asset_dir(asset_id) / UNIRIG_WORK_DIR

    def binding_path(self, asset_id: str, name: str) -> Path:
        """S2 产物路径，``name`` 取 ``rig`` / ``skin`` / ``skin_report``。"""
        filename = {"rig": RIG_FILE, "skin": SKIN_FILE,
                    "skin_report": SKIN_REPORT_FILE}.get(name)
        if filename is None:
            raise ValueError(f"未知绑定产物名：{name!r}")
        return self.asset_dir(asset_id) / filename

    def raw_path(self, asset_id: str, filename: str) -> Path:
        """上传原件路径：``<name>_raw.<ext>``（与旧 tasks/ 的 ``*_raw`` 约定一致）。"""
        ext = Path(str(filename or "")).suffix.lower() or ".bin"
        return self.asset_dir(asset_id) / f"{safe_stem(filename, 'asset')}_raw{ext}"

    # ------------------------------------------------------------------ #
    # 创建 / 查询
    # ------------------------------------------------------------------ #
    def create_asset(self, kind: AssetKind | str, name: Optional[str] = None) -> str:
        """建素材（落库 + 建目录），返回 asset_id。上传走 :meth:`save_upload`。"""
        kind = AssetKind(kind)
        asset_id = uuid.uuid4().hex[:12]
        self.asset_dir(asset_id).mkdir(parents=True, exist_ok=True)
        ts = now_iso()
        self.db.insert("assets", {
            "asset_id": asset_id,
            "kind": kind.value,
            "name": name or None,
            "filename": None,
            "asset_dir": str(self.asset_dir(asset_id)),
            "state": AssetState.CREATED.value,
            "align_json": None,
            "meta_json": None,
            "error": None,
            "created_at": ts,
            "updated_at": ts,
        })
        logger.info("已创建素材 %s（kind=%s name=%s）", asset_id, kind.value, name)
        return asset_id

    def get_row(self, asset_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM assets WHERE asset_id=?", (asset_id,))
        return dict(row) if row else None

    def exists(self, asset_id: str) -> bool:
        return self.get_row(asset_id) is not None

    def require(self, asset_id: str) -> Dict[str, Any]:
        row = self.get_row(asset_id)
        if row is None:
            raise NotFoundError(asset_id)
        return row

    def get_asset_info(self, asset_id: str) -> Optional[AssetInfo]:
        row = self.get_row(asset_id)
        if row is None:
            return None
        return AssetInfo(
            asset_id=row["asset_id"],
            kind=AssetKind(row["kind"]),
            name=row["name"],
            filename=row["filename"],
            state=AssetState(row["state"]),
            align=parse_json(row["align_json"], None),
            meta=parse_json(row["meta_json"], None),
            binding=self.binding_info(asset_id),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_assets(self, kind: Optional[AssetKind | str] = None, limit: int = 50,
                    offset: int = 0) -> Tuple[List[AssetSummary], int]:
        where = "WHERE kind=?" if kind else ""
        params: List[Any] = [AssetKind(kind).value] if kind else []
        total = int(self.db.scalar(f"SELECT COUNT(*) FROM assets {where}", params) or 0)
        rows = self.db.query(
            f"SELECT * FROM assets {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset])
        return [self._summary(r) for r in rows], total

    def _summary(self, row: Dict[str, Any]) -> AssetSummary:
        align = parse_json(row["align_json"], None) or {}
        unit = align.get("unit") or {}
        binding = self.db.query_one(
            "SELECT state FROM bindings WHERE asset_id=?", (row["asset_id"],))
        return AssetSummary(
            asset_id=row["asset_id"], kind=AssetKind(row["kind"]), name=row["name"],
            filename=row["filename"], state=AssetState(row["state"]),
            binding_state=BindingState(binding["state"]) if binding else BindingState.NONE,
            height_m=unit.get("height_m"), unit=unit.get("detected"),
            error=row["error"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def count_by_state(self) -> Dict[str, int]:
        rows = self.db.query("SELECT state, COUNT(*) c FROM assets GROUP BY state")
        return {r["state"]: r["c"] for r in rows}

    def list_ready_animations(self, limit: int = 200) -> List[AssetSummary]:
        """可供 S3 选用的动画素材（只列 READY 的，未确认的动画不能被复用）。"""
        rows = self.db.query(
            "SELECT * FROM assets WHERE kind=? AND state=? "
            "ORDER BY created_at DESC LIMIT ?",
            (AssetKind.ANIMATION.value, AssetState.READY.value, limit))
        return [self._summary(dict(r)) for r in rows]

    # ------------------------------------------------------------------ #
    # 更新
    # ------------------------------------------------------------------ #
    def update_asset(self, asset_id: str, **fields: Any) -> None:
        values: Dict[str, Any] = {}
        for k, v in fields.items():
            if v is None and k not in _NULLABLE_ASSET_FIELDS:
                continue
            if k in _ENUM_ASSET_FIELDS and hasattr(v, "value"):
                v = v.value
            if k in _JSON_ASSET_FIELDS and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            values[k] = v
        values["updated_at"] = now_iso()
        self.db.update("assets", values, "asset_id=?", (asset_id,))

    def set_state(self, asset_id: str, state: AssetState | str,
                  error: Optional[str] = None) -> None:
        self.update_asset(asset_id, state=AssetState(state), error=error)

    def save_upload(self, asset_id: str, filename: str, data: bytes) -> Path:
        """落盘上传原件并登记文件名；素材名为空时用原件茎补上。"""
        self.require(asset_id)
        path = self.raw_path(asset_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        row = self.get_row(asset_id) or {}
        fields: Dict[str, Any] = {"filename": path.name}
        if not row.get("name"):
            fields["name"] = safe_stem(filename, "asset")
        self.update_asset(asset_id, **fields)
        logger.info("素材 %s 已收原件 %s（%d 字节）", asset_id, path.name, len(data))
        return path

    def save_align(self, asset_id: str, align: Dict[str, Any]) -> None:
        """写 align.json 侧车 + 落库 ``align_json``（DB 为 API 的权威来源）。"""
        path = self.align_path(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(align, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update_asset(asset_id, align_json=align)

    def load_align(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """优先读 DB，缺失时回退磁盘侧车（便于手工修过文件后恢复）。"""
        row = self.get_row(asset_id)
        if row is None:
            raise NotFoundError(asset_id)
        align = parse_json(row["align_json"], None)
        if align is not None:
            return align
        path = self.align_path(asset_id)
        if not path.exists():
            return None
        return parse_json(path.read_text(encoding="utf-8"), None)

    def save_meta(self, asset_id: str, meta: Dict[str, Any]) -> None:
        path = self.meta_path(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update_asset(asset_id, meta_json=meta)

    def require_align_ready(self, asset_id: str) -> Dict[str, Any]:
        """S2 准入门槛：单位/轴系未矫正落库前禁止绑定。

        ``skin.weld_epsilon=2e-3``、``retarget.foot_contact_height=0.12`` 都是米制绝对
        阈值，跳过 S1 直接绑定会得到完全错误的焊接半径与脚部接触判定。
        """
        row = self.require(asset_id)
        state = AssetState(row["state"])
        if state not in (AssetState.ALIGN_READY, AssetState.READY):
            raise StateError(f"素材 {asset_id} 处于 {state.value}，须先完成并确认 S1 导入矫正")
        if not self.glb_path(asset_id).exists():
            raise StateError(f"素材 {asset_id} 缺少 {GLB_FILE}，无法绑定")
        return row

    # ------------------------------------------------------------------ #
    # bindings（S2 结果缓存，模型素材跨作业复用的关键）
    # ------------------------------------------------------------------ #
    def get_binding(self, asset_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM bindings WHERE asset_id=?", (asset_id,))
        return dict(row) if row else None

    def ensure_binding(self, asset_id: str,
                       state: BindingState | str = BindingState.NONE) -> Dict[str, Any]:
        """取绑定行，不存在则建（``ON CONFLICT DO NOTHING`` 语义）。"""
        existing = self.get_binding(asset_id)
        if existing is not None:
            return existing
        self.db.insert("bindings", {
            "asset_id": asset_id, "state": BindingState(state).value,
            "confidence": None, "revision": 0, "rig_path": None,
            "skin_path": None, "report_path": None, "updated_at": now_iso(),
        })
        return self.get_binding(asset_id) or {}

    def update_binding(self, asset_id: str, **fields: Any) -> None:
        self.ensure_binding(asset_id)
        values: Dict[str, Any] = {}
        for k, v in fields.items():
            if k == "state" and hasattr(v, "value"):
                v = v.value
            values[k] = v
        values["updated_at"] = now_iso()
        self.db.update("bindings", values, "asset_id=?", (asset_id,))

    def binding_info(self, asset_id: str) -> BindingInfo:
        """S2 摘要：状态 + 子步骤进度 + 置信度 + 产物是否就绪 + 蒙皮报告全文。"""
        row = self.get_binding(asset_id)
        if row is None:
            return BindingInfo()
        return BindingInfo(
            state=BindingState(row["state"]) if row["state"] else BindingState.NONE,
            stage=row["stage"], message=row["message"], error=row["error"],
            confidence=row["confidence"], revision=row["revision"] or 0,
            has_rig=self.binding_path(asset_id, "rig").exists(),
            has_skin=self.binding_path(asset_id, "skin").exists(),
            views=self.list_views(asset_id),
            report=self.load_binding_json(asset_id, "skin_report"),
        )

    def load_binding_json(self, asset_id: str, name: str) -> Optional[Dict[str, Any]]:
        """读 S2 的 JSON 产物（``rig`` / ``skin_report``）；缺失或损坏返回 ``None``。"""
        path = self.binding_path(asset_id, name)
        if not path.exists():
            return None
        try:
            return parse_json(path.read_text(encoding="utf-8"), None)
        except OSError as exc:
            logger.warning("读 %s 失败（%s），按缺失处理", path.name, exc)
            return None

    def set_binding_progress(self, asset_id: str, stage: str, message: str = "",
                             state: BindingState | str = BindingState.RUNNING) -> None:
        """推进 S2 子步骤（前端轮询据此画进度）。"""
        self.update_binding(asset_id, state=state, stage=stage,
                            message=message or None, error=None)

    def mark_binding_done(self, asset_id: str, confidence: Optional[float] = None,
                          revision: Optional[int] = None, message: str = "") -> None:
        """S2 完成：登记三个产物路径 + 置信度，状态置 ``READY``。

        ``revision`` 只在**重新生成了 rig.json** 时传（从头重绑时 rig 的 revision 归 0，
        不跟着改会让下一次 PATCH /rig 的乐观锁对不上）；仅重算蒙皮时不传。
        """
        fields: Dict[str, Any] = {
            "state": BindingState.READY, "stage": None, "error": None,
            "message": message or None,
            "rig_path": str(self.rig_path(asset_id)),
            "skin_path": str(self.skin_path(asset_id)),
            "report_path": str(self.skin_report_path(asset_id)),
        }
        if confidence is not None:
            fields["confidence"] = round(float(confidence), 4)
        if revision is not None:
            fields["revision"] = int(revision)
        self.update_binding(asset_id, **fields)

    def mark_binding_failed(self, asset_id: str, reason: str,
                            stage: Optional[str] = None) -> None:
        """S2 失败：把原因写进 ``bindings.error``（与 S1 的 ``assets.error`` 互不覆盖）。"""
        self.update_binding(asset_id, state=BindingState.FAILED, error=reason,
                            message=None, **({"stage": stage} if stage else {}))

    def save_rig(self, asset_id: str, expected_revision: int, rig: Dict[str, Any]) -> int:
        """人工微调关节：写 rig.json + revision 乐观锁（移植 task_store 同款语义）。"""
        self.require(asset_id)
        path = self.binding_path(asset_id, "rig")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rig, ensure_ascii=False, indent=2), encoding="utf-8")
        self.ensure_binding(asset_id)
        new_rev = expected_revision + 1
        affected = self.db.execute(
            "UPDATE bindings SET revision=?, rig_path=?, updated_at=? "
            "WHERE asset_id=? AND revision=?",
            (new_rev, str(path), now_iso(), asset_id, expected_revision))
        if affected == 0:
            row = self.get_binding(asset_id) or {}
            raise ConflictError(
                f"revision 冲突：期望 {expected_revision}，实际 {row.get('revision')}"
                f"（请刷新后重试）")
        return new_rev

    # ------------------------------------------------------------------ #
    # S2 多视角图（前端离屏渲染后回传）
    # ------------------------------------------------------------------ #
    def save_view_spec(self, asset_id: str, spec: Dict[str, Any]) -> Path:
        """持久化下发的渲染规格（重启后 SOLVE_RIG 仍按同一份相机参数三角化）。"""
        path = self.view_spec_path(asset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_view_spec(self, asset_id: str) -> Optional[Dict[str, Any]]:
        path = self.view_spec_path(asset_id)
        return parse_json(path.read_text(encoding="utf-8"), None) if path.exists() else None

    def save_views(self, asset_id: str, files: Iterable[Tuple[str, bytes]]) -> List[str]:
        """落盘前端回传的多视角图，返回实际写入的文件名（已排序）。

        文件名主干即 ``view_id``，必须与 ``view_spec.json`` 里的相机一一对应，否则
        三角化会把某个视角的 2D 关键点配到错误的相机上——故这里**先清空目录**再写，
        不让上一次回传的残留视角混进来。
        """
        self.require(asset_id)
        vdir = self.views_dir(asset_id)
        if vdir.exists():
            shutil.rmtree(vdir, ignore_errors=True)
        vdir.mkdir(parents=True, exist_ok=True)
        saved: List[str] = []
        for filename, data in files:
            name = Path(str(filename or "view.png")).name      # 去目录成分，防穿越
            if Path(name).suffix.lower() not in VIEW_IMAGE_EXTS:
                logger.warning("忽略非图片的视角文件：%s", name)
                continue
            (vdir / name).write_bytes(data)
            saved.append(name)
        if not saved:
            raise StateError("回传内容里没有可用图片（仅支持 "
                             + " / ".join(sorted(VIEW_IMAGE_EXTS)) + "）")
        logger.info("素材 %s 收到 %d 张视角图：%s", asset_id, len(saved), ", ".join(saved))
        return sorted(saved)

    def list_views(self, asset_id: str) -> List[str]:
        """已回传的视角图文件名（主干即 view_id）。"""
        vdir = self.views_dir(asset_id)
        if not vdir.exists():
            return []
        return sorted(p.name for p in vdir.iterdir()
                      if p.is_file() and p.suffix.lower() in VIEW_IMAGE_EXTS)

    # ------------------------------------------------------------------ #
    # S2 UniRig 执行日志（前端增量轮询看「跑到哪了」）
    # ------------------------------------------------------------------ #
    def read_unirig_log(self, asset_id: str, offset: int = 0,
                        max_bytes: int = LOG_CHUNK_BYTES) -> Dict[str, Any]:
        """从字节 ``offset`` 起读一段 UniRig 执行输出（增量轮询的唯一入口）。

        返回体里的 ``offset`` 就是**下次请求该带的值**，前端不必自己累加字节数。
        三个边界：

        - 日志不存在（没跑过 UniRig）→ ``exists=false`` + 200，不当错：前端是在
          绑定刚起步时就开始轮的，那时文件可能还没建；
        - ``offset > size``（重跑把文件覆盖了）→ ``reset=true`` 并从头返回，
          前端据此清空已渲染的旧输出；
        - 剩余超过 ``max_bytes`` → 只返回一片、``truncated=true``，offset 照样往前推，
          下轮接着拉（不丢中间输出，只是多轮几次）。

        按**字节**读再 ``decode(errors="replace")``：offset 可能正切在一个中文字的
        多字节序列中间，用文本模式 seek 会抛错或丢字。
        """
        self.require(asset_id)
        path = self.unirig_log_path(asset_id)
        row = self.get_binding(asset_id) or {}
        state = row.get("state") or BindingState.NONE.value
        base: Dict[str, Any] = {
            "asset_id": asset_id,
            "state": state,
            "stage": row.get("stage"),
            "message": row.get("message"),
            "error": row.get("error"),
            "running": state in (BindingState.RUNNING.value, BindingState.PENDING.value),
        }
        try:
            size = path.stat().st_size
        except OSError:
            return {**base, "exists": False, "size": 0, "offset": 0,
                    "text": "", "reset": False, "truncated": False}
        reset = offset > size or offset < 0
        start = 0 if reset else int(offset)
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                chunk = fh.read(max(1, int(max_bytes)))
        except OSError as exc:
            logger.warning("读 UniRig 日志失败 %s：%s", path, exc)
            return {**base, "exists": True, "size": size, "offset": start,
                    "text": "", "reset": reset, "truncated": False}
        return {**base, "exists": True, "size": size, "offset": start + len(chunk),
                "text": chunk.decode("utf-8", "replace"),
                "reset": reset, "truncated": start + len(chunk) < size}

    # ------------------------------------------------------------------ #
    # 作业引用 / 删除
    # ------------------------------------------------------------------ #
    def count_jobs(self, asset_id: str) -> Dict[str, int]:
        """该素材被多少作业引用（删除前给前端提示用）。"""
        return {
            "as_model": int(self.db.scalar(
                "SELECT COUNT(*) FROM jobs WHERE model_asset_id=?", (asset_id,)) or 0),
            "as_animation": int(self.db.scalar(
                "SELECT COUNT(*) FROM jobs WHERE anim_asset_id=?", (asset_id,)) or 0),
        }

    def delete_asset(self, asset_id: str, remove_files: bool = True) -> bool:
        """删素材：连带 bindings（FK CASCADE）与引用它的作业目录。

        模型被作业引用时 ``jobs`` 走 ON DELETE CASCADE，磁盘目录不会自动消失，故先把
        这些 job_id 捞出来一并清目录；动画被引用时 FK 是 SET NULL，作业保留。
        """
        row = self.get_row(asset_id)
        if row is None:
            return False
        job_rows = self.db.query(
            "SELECT job_id FROM jobs WHERE model_asset_id=?", (asset_id,))
        job_dirs = [self.settings.job_dir(r["job_id"]) for r in job_rows]
        self.db.execute("DELETE FROM assets WHERE asset_id=?", (asset_id,))
        if remove_files:
            shutil.rmtree(self.asset_dir(asset_id), ignore_errors=True)
            for d in job_dirs:
                shutil.rmtree(d, ignore_errors=True)
        logger.info("已删除素材 %s（连带 %d 个作业目录）", asset_id, len(job_dirs))
        return True


# ---- 单例 ----
_store: Optional[AssetStore] = None


def get_asset_store(reload: bool = False) -> AssetStore:
    global _store
    if _store is None or reload:
        _store = AssetStore()
    return _store
