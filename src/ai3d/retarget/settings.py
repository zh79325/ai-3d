"""集中式配置：dataclass + PyYAML，风格对齐 ``ai3d.config``。

加载优先级：内置默认(dataclass) < ``src/ai3d/config/retarget.yaml`` < 环境变量(``RETARGET_*``)。
所有相对路径按项目根 ``ai3d.config.PROJECT_ROOT`` 解析，不受启动工作目录影响。
配置加载后做校验，缺失/非法即抛错（不用隐式默认掩盖问题）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import ai3d.config as _ai3d_config
from ai3d.config import PROJECT_ROOT, ModelConfig

logger = logging.getLogger(__name__)

# 配置文件固定放在 ai3d 包下的 config/ 目录
_AI3D_DIR = Path(_ai3d_config.__file__).resolve().parent
CONFIG_DIR = _AI3D_DIR / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "retarget.yaml"

ENV_PREFIX = "RETARGET_"


def _resolve(p: str | os.PathLike[str]) -> Path:
    """把配置里的路径解析为绝对路径：相对路径按项目根解析。"""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8790
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class DatabaseConfig:
    # 仅 SQLite；相对路径按项目根解析
    sqlite_path: str = "output/data/retarget.db"


@dataclass
class PathsConfig:
    # 所有落盘的根；相对路径按项目根解析
    project_dir: str = "output"


@dataclass
class AssimpConfig:
    # assimp CLI 名称或绝对路径；仅 FBX 输入时需要
    path: str = "assimp"


@dataclass
class AxisConfig:
    # 导入时自动把模型轴系归一到规范系（+Y up / +Z forward / +X left）
    enabled: bool = True


@dataclass
class GatingConfig:
    auto_pass: float = 0.80       # >= 自动通过
    review_min: float = 0.55      # [review_min, auto_pass) 需人工审核
    core_joint_min: float = 0.50  # 核心关节低于此值强制审核


@dataclass
class SolveConfig:
    min_confidence: float = 0.30       # 参与三角化的最低 2D 置信度
    symmetry_weight: float = 0.5       # 左右对称软约束权重
    proportion_weight: float = 0.3     # 人体比例软约束权重
    bone_length_stability: float = 0.5 # 骨长稳定化权重


@dataclass
class SkinConfig:
    max_influences: int = 4            # 每顶点最大骨骼影响数
    weld_epsilon: float = 2e-3         # 求解域焊接合并半径(m)；=0 关闭合并退化为逐岛
    handle_margin: float = 0.25        # handle 选骨容差 = d_min + margin*分量 bbox 对角
    max_handles: int = 22              # 单分量 handle 骨上限；全身大分量需全骨 handle，小甲片由 handle_margin 自动收敛
    bbw_max_verts: int = 30000         # 分量顶点超阈走 igl.decimate 代理求解+最近邻传回


@dataclass
class MapConfig:
    name_weight: float = 1.0
    hierarchy_weight: float = 0.6
    geometry_weight: float = 0.8
    motion_weight: float = 0.4
    min_match_score: float = 0.35


@dataclass
class RetargetConfig:
    bake_fps: int = 0                  # 0 = 沿用源动画 fps
    root_scale_by_height: bool = True  # root motion 按身高缩放
    foot_lock: bool = True             # 脚部接触锁定 + 双骨 IK
    spine_distribute_by_length: bool = True
    foot_contact_height: float = 0.12  # 脚踝低于 (最低点 + frac*身高) 视为可能接触
    foot_contact_speed: float = 0.30   # 支撑相水平速度上限(m/s)，超过视为摆动相
    foot_blend_frames: int = 2         # 接触窗口边缘平滑混入/混出帧数
    foot_min_contact_frames: int = 2   # 少于该帧数的接触窗口忽略


@dataclass
class RetargetSettings:
    """顶层配置。"""
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    assimp: AssimpConfig = field(default_factory=AssimpConfig)
    axis: AxisConfig = field(default_factory=AxisConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    solve: SolveConfig = field(default_factory=SolveConfig)
    skin: SkinConfig = field(default_factory=SkinConfig)
    mapping: MapConfig = field(default_factory=MapConfig)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    config_path: Optional[Path] = None

    # ---- 解析后的绝对路径 ----
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def project_dir(self) -> Path:
        return _resolve(self.paths.project_dir)

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def tasks_dir(self) -> Path:
        return self.project_dir / "tasks"

    @property
    def sqlite_file(self) -> Path:
        return _resolve(self.database.sqlite_path)

    def task_dir(self, task_id: str) -> Path:
        return self.tasks_dir / task_id

    def ensure_dirs(self) -> None:
        """创建项目目录结构与数据库父目录。"""
        for d in (self.project_dir, self.data_dir, self.tasks_dir,
                  self.sqlite_file.parent):
            d.mkdir(parents=True, exist_ok=True)

    # ---- 加载 ----
    @classmethod
    def load(cls, config_path: str | os.PathLike[str] | None = None) -> "RetargetSettings":
        path = Path(
            config_path
            or os.environ.get(f"{ENV_PREFIX}CONFIG")
            or DEFAULT_CONFIG_PATH
        )
        if not path.exists():
            raise FileNotFoundError(
                f"配置文件不存在：{path}。请参考 {CONFIG_DIR / 'retarget.example.yaml'} 创建。"
            )
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        settings = cls._from_dict(raw)
        settings.config_path = path
        cls._apply_env(settings)
        settings._validate()
        return settings

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "RetargetSettings":
        s = cls()
        s.server = _merge_dc(s.server, raw.get("server"))
        s.database = _merge_dc(s.database, raw.get("database"))
        s.paths = _merge_dc(s.paths, raw.get("paths"))
        s.assimp = _merge_dc(s.assimp, raw.get("assimp"))
        s.axis = _merge_dc(s.axis, raw.get("axis"))
        s.gating = _merge_dc(s.gating, raw.get("gating"))
        s.solve = _merge_dc(s.solve, raw.get("solve"))
        s.skin = _merge_dc(s.skin, raw.get("skin"))
        s.mapping = _merge_dc(s.mapping, raw.get("mapping"))
        s.retarget = _merge_dc(s.retarget, raw.get("retarget"))
        s.model = _merge_dc(s.model, raw.get("model"))
        return s

    @classmethod
    def _apply_env(cls, s: "RetargetSettings") -> None:
        env = os.environ.get
        if (v := env(f"{ENV_PREFIX}SERVER_HOST")):
            s.server.host = v
        if (v := env(f"{ENV_PREFIX}SERVER_PORT")):
            s.server.port = int(v)
        if (v := env(f"{ENV_PREFIX}DB_PATH")):
            s.database.sqlite_path = v
        if (v := env(f"{ENV_PREFIX}PROJECT_DIR")):
            s.paths.project_dir = v
        if (v := env(f"{ENV_PREFIX}ASSIMP_PATH")):
            s.assimp.path = v

    def _validate(self) -> None:
        if not (0 <= self.server.port <= 65535):
            raise ValueError(f"server.port 非法：{self.server.port}")
        g = self.gating
        if not (0.0 <= g.core_joint_min <= g.review_min <= g.auto_pass <= 1.0):
            raise ValueError(
                f"gating 阈值须满足 core_joint_min<=review_min<=auto_pass 且在[0,1]："
                f"{g.core_joint_min}/{g.review_min}/{g.auto_pass}"
            )
        if self.skin.max_influences < 1:
            raise ValueError("skin.max_influences 必须 >= 1")
        # 路径可解析（不强制存在，运行时会创建）
        _ = self.project_dir
        _ = self.sqlite_file

    def to_public_dict(self) -> Dict[str, Any]:
        """脱敏后的生效配置，用于 GET /v1/config 排查。"""
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "server": {"host": self.server.host, "port": self.server.port,
                       "cors_origins": self.server.cors_origins},
            "database": {"sqlite_file": str(self.sqlite_file)},
            "paths": {
                "project_dir": str(self.project_dir),
                "data_dir": str(self.data_dir),
                "tasks_dir": str(self.tasks_dir),
            },
            "assimp": {"path": self.assimp.path},
            "gating": {
                "auto_pass": self.gating.auto_pass,
                "review_min": self.gating.review_min,
                "core_joint_min": self.gating.core_joint_min,
            },
        }


def _merge_dc(dc: Any, override: Optional[Dict[str, Any]]) -> Any:
    """用 override 字典中「已存在于 dataclass 的字段」覆盖 dc，返回新实例。"""
    if not override:
        return dc
    valid = {f for f in dc.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(override) - valid
    if unknown:
        raise ValueError(f"配置项含未知字段 {sorted(unknown)}（{type(dc).__name__}）")
    return replace(dc, **{k: v for k, v in override.items() if k in valid})


# ---- 单例 ----
_settings: Optional[RetargetSettings] = None


def get_settings(reload: bool = False) -> RetargetSettings:
    global _settings
    if _settings is None or reload:
        _settings = RetargetSettings.load()
        _settings.ensure_dirs()
        logger.info(
            "配置已加载：%s | project_dir=%s | sqlite=%s",
            _settings.config_path, _settings.project_dir, _settings.sqlite_file,
        )
    return _settings
