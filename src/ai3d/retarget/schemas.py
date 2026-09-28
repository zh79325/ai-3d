"""接口层 DTO 与枚举（Pydantic v2）。仅用于 API 契约，与 DB 层分离。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 阶段 / 状态
# --------------------------------------------------------------------------- #
class Stage(str, Enum):
    """流水线阶段（有序）。"""
    NORMALIZE = "NORMALIZE"        # 输入(FBX/GLB) → 归一化 GLB
    PRECHECK = "PRECHECK"          # 目标网格/源骨架基本校验
    RENDER_VIEWS = "RENDER_VIEWS"  # 等待前端回传多视角图
    POSE_INFER = "POSE_INFER"      # DWPose/YOLO 2D 关键点
    SOLVE_RIG = "SOLVE_RIG"        # 三角化 + 约束 → 三维关节
    BUILD_RIG = "BUILD_RIG"        # 语义骨架 + 自动蒙皮
    MAP_SOURCE = "MAP_SOURCE"      # 源骨 → 语义骨架映射
    RETARGET = "RETARGET"          # 重定向 + 逐帧烘焙
    EXPORT_VERIFY = "EXPORT_VERIFY"  # 导出 result.glb 并回读验证


# 阶段执行顺序
STAGE_ORDER: List[Stage] = [
    Stage.NORMALIZE,
    Stage.PRECHECK,
    Stage.RENDER_VIEWS,
    Stage.POSE_INFER,
    Stage.SOLVE_RIG,
    Stage.BUILD_RIG,
    Stage.MAP_SOURCE,
    Stage.RETARGET,
    Stage.EXPORT_VERIFY,
]


class JobState(str, Enum):
    """任务生命周期状态。"""
    CREATED = "CREATED"        # 已建任务但尚未上传齐文件/未启动
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"          # 等待外部输入（如前端回传视角图）
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DONE = "DONE"
    FAILED = "FAILED"


class StageStatus(str, Enum):
    """单个阶段的执行状态（写入 stages 表）。"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ArtifactKind(str, Enum):
    SOURCE = "source"            # 归一化后的源 GLB
    TARGET = "target"            # 归一化后的目标 GLB
    SOURCE_RAW = "source_raw"    # 原始上传源文件
    TARGET_RAW = "target_raw"    # 原始上传目标文件
    VIEWS = "views"              # 前端回传的多视角图目录
    RIG = "rig"                  # rig.json
    MAPPING = "mapping"          # mapping.json
    RESULT = "result"            # result.glb
    REPORT = "report"            # report.json


# --------------------------------------------------------------------------- #
# 配置 / 请求
# --------------------------------------------------------------------------- #
class JobConfig(BaseModel):
    """上传时随 multipart 传入的处理配置（JSON）。"""
    action: str = "retarget"
    fps: int = 0                       # 0 = 沿用源动画 fps
    enable_pose_ai: bool = True        # 关闭时走 P1 确定性/人工路径
    foot_lock: bool = True
    auto_continue_on_review: bool = False


class RerunRequest(BaseModel):
    from_stage: Stage = Stage.NORMALIZE
    force: bool = False                # 强制跳过审核门控


class JointPatch(BaseModel):
    head: Optional[List[float]] = None       # [x, y, z]
    tail: Optional[List[float]] = None
    locked: Optional[bool] = None
    confidence: Optional[float] = None


class RigPatch(BaseModel):
    revision: int
    joints: Dict[str, JointPatch] = Field(default_factory=dict)


class MappingPatch(BaseModel):
    revision: int
    # semantic_id -> source_bone_name（人工覆盖映射）
    overrides: Dict[str, str] = Field(default_factory=dict)


class CameraSpec(BaseModel):
    view_id: str
    name: str
    azimuth: float                 # 绕 up 轴方位角（度）
    elevation: float = 0.0         # 仰角（度）


class ViewSpec(BaseModel):
    """后端下发给前端的渲染指令：对 target.glb 做 N 个正交视角离屏渲染。"""
    model_url: str
    cameras: List[CameraSpec]
    width: int = 512
    height: int = 512
    up_axis: str = "Y"             # Three.js/GLB 世界 up
    background: str = "#ffffff"
    ortho: bool = True


# 默认多视角（正交，绕 up 轴均布）：(view_id, azimuth_deg, elevation_deg)。
# 前端按此离屏渲染并回传，后端 solve 按同一参数重建相机做三角化——必须保持一致。
DEFAULT_CAMERAS: List[tuple] = [
    ("front", 0.0, 0.0),
    ("three_quarter", 45.0, 8.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
]


# --------------------------------------------------------------------------- #
# 响应
# --------------------------------------------------------------------------- #
class StageInfo(BaseModel):
    stage: Stage
    status: StageStatus
    message: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class ArtifactInfo(BaseModel):
    kind: str
    path: str
    created_at: Optional[str] = None


class JobInfo(BaseModel):
    """GET /v1/jobs/{task_id} 响应。"""
    task_id: str
    state: JobState
    current_stage: Optional[Stage] = None
    name: Optional[str] = None
    source_filename: Optional[str] = None
    target_filename: Optional[str] = None
    config: JobConfig = Field(default_factory=JobConfig)
    rig_revision: int = 0
    mapping_revision: int = 0
    overall_confidence: Optional[float] = None
    review_reasons: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    stages: List[StageInfo] = Field(default_factory=list)
    artifacts: List[ArtifactInfo] = Field(default_factory=list)


class JobSummary(BaseModel):
    task_id: str
    state: JobState
    current_stage: Optional[Stage] = None
    name: Optional[str] = None
    source_filename: Optional[str] = None
    target_filename: Optional[str] = None
    overall_confidence: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobListResponse(BaseModel):
    tasks: List[JobSummary]
    total: int
    limit: int
    offset: int


class CreateJobResponse(BaseModel):
    task_id: str
    state: JobState
    current_stage: Optional[Stage] = None
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    assimp: Optional[str] = None
    counts: Dict[str, Any] = Field(default_factory=dict)
