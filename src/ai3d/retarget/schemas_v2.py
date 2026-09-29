"""``/v2`` 四阶段流程的接口层 DTO 与枚举（Pydantic v2）。

与 :mod:`schemas`（``/v1`` 一对一任务）严格分离：过渡期两套 API 并存，各自只读写
自己的表与 DTO，避免字段语义互相污染。``/v1`` 下线后本模块成为唯一契约。

四阶段（模型侧）：**S1 导入矫正 → S2 绑定 → S3 重定向 → S4 导出**；动画素材只走 S1，
入库后可被任意模型复用。九阶段（:class:`ai3d.retarget.schemas.Stage`）作为 S1~S4 的
内部子步骤保留。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas import ArtifactInfo, Stage, StageInfo


# --------------------------------------------------------------------------- #
# 素材库
# --------------------------------------------------------------------------- #
class AssetKind(str, Enum):
    """素材类型：模型走四阶段，动画只走 S1 后进入素材库供复用。"""
    MODEL = "model"
    ANIMATION = "animation"


class AssetState(str, Enum):
    """素材生命周期状态。

    ``ALIGN_READY`` 是 S1 的完成态（等人工确认），也是 S2 的准入门槛：
    ``skin.weld_epsilon`` / ``foot_contact_height`` 都是米制绝对阈值，单位未矫正就
    绑定会得到完全错误的焊接半径。
    """
    CREATED = "CREATED"              # 已建素材但尚未上传文件
    NORMALIZING = "NORMALIZING"      # S1 进行中（格式归一 + OBB 对齐 + 单位推断）
    ALIGN_READY = "ALIGN_READY"      # S1 完成，等人工核对/修正后确认
    READY = "READY"                  # 已确认（animation 到此为止；model 表示 S2 也完成）
    FAILED = "FAILED"


class BindingState(str, Enum):
    """S2 绑定状态（缓存于 bindings 表，供模型素材跨作业复用）。"""
    NONE = "NONE"            # 尚未开始
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAIT_VIEWS = "WAIT_VIEWS"    # 等前端回传多视角图
    READY = "READY"
    FAILED = "FAILED"


class StageGroup(str, Enum):
    """四阶段分组（``jobs.stage_group``）。"""
    S1 = "S1"    # 导入矫正
    S2 = "S2"    # 绑定
    S3 = "S3"    # 重定向
    S4 = "S4"    # 导出


# --------------------------------------------------------------------------- #
# 请求
# --------------------------------------------------------------------------- #
class AssetCreate(BaseModel):
    """POST /v2/assets：先建素材再分步上传（也允许一步到位带 name）。"""
    kind: AssetKind
    name: Optional[str] = None


class AlignPatch(BaseModel):
    """PATCH /v2/assets/{id}/align：只改 ``manual`` 段，后端重算 ``final``。

    ``front_flipped`` = 前后方向是否相反（True 即绕规范系 Y 轴转 180°，对应人工
    二选确认的「前后方向相反」）；校准基与外切盒由后端自动重测，不接受人工欧拉角。
    字段留 ``None`` 表示不改该项。

    两项覆盖的**清除**用哨兵值（``None`` 已被占用为「不改」）：
    ``height_override <= 0`` 取消真实身高直填，``unit_override = ""`` 取消强制单位。
    """
    front_flipped: Optional[bool] = None
    unit_override: Optional[str] = None
    height_override: Optional[float] = None
    reset: bool = False          # True = 清空全部人工修正，回到自动探测


# --------------------------------------------------------------------------- #
# 响应
# --------------------------------------------------------------------------- #
class BindingInfo(BaseModel):
    """S2 绑定摘要（``GET /v2/assets/{id}`` 内嵌）。

    ``stage`` / ``message`` 是子步骤级进度（RENDER_VIEWS → POSE_INFER → SOLVE_RIG →
    BUILD_RIG），蒙皮一跑就是几秒到十几秒，前端靠轮询这两个字段画进度；
    ``error`` 只在 ``FAILED`` 时非空，与 S1 的 ``AssetInfo.error`` 互不覆盖。
    """
    state: BindingState = BindingState.NONE
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    confidence: Optional[float] = None
    revision: int = 0
    has_rig: bool = False
    has_skin: bool = False
    views: List[str] = Field(default_factory=list)
    report: Optional[Dict[str, Any]] = None


class AssetSummary(BaseModel):
    """列表项：不含 align/meta 大字段。"""
    asset_id: str
    kind: AssetKind
    name: Optional[str] = None
    filename: Optional[str] = None
    state: AssetState
    binding_state: BindingState = BindingState.NONE
    height_m: Optional[float] = None     # 取自 align.unit.height_m，列表直接展示
    unit: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AssetInfo(BaseModel):
    """GET /v2/assets/{id} 响应。"""
    asset_id: str
    kind: AssetKind
    name: Optional[str] = None
    filename: Optional[str] = None
    state: AssetState
    align: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    binding: BindingInfo = Field(default_factory=BindingInfo)
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AssetListResponse(BaseModel):
    assets: List[AssetSummary]
    total: int
    limit: int
    offset: int


class CreateAssetResponse(BaseModel):
    asset_id: str
    state: AssetState
    message: str = ""


class AlignResponse(BaseModel):
    """S1 产物回执（上传后自动跑、PATCH 重算共用）。

    ``align`` 即 align.json 全文；``state`` 让前端无需再拉一次详情就能判断是
    ``ALIGN_READY``（可确认）还是 ``FAILED``（看 ``error``）。
    """
    asset_id: str
    state: AssetState
    filename: Optional[str] = None
    align: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    message: str = ""


class JobCreate(BaseModel):
    """POST /v2/jobs：作业 = 模型素材 × （稍后选定的）动画素材。"""
    model_asset_id: str
    name: Optional[str] = None


class AnimationSelect(BaseModel):
    """PATCH /v2/jobs/{id}/animation：从素材库选动画并触发 S3。"""
    anim_asset_id: Optional[str] = None


class ExportRequest(BaseModel):
    """POST /v2/jobs/{id}/export：``force`` 越过 S4 门控。"""
    force: bool = False


class JobInfoV2(BaseModel):
    """GET /v2/jobs/{id} 响应。"""
    job_id: str
    name: Optional[str] = None
    state: str
    stage_group: Optional[StageGroup] = None
    current_stage: Optional[Stage] = None
    model_asset_id: str
    anim_asset_id: Optional[str] = None
    mapping_revision: int = 0
    review_reasons: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    stages: List[StageInfo] = Field(default_factory=list)
    artifacts: List[ArtifactInfo] = Field(default_factory=list)


class JobSummaryV2(BaseModel):
    job_id: str
    name: Optional[str] = None
    state: str
    stage_group: Optional[StageGroup] = None
    model_asset_id: str
    anim_asset_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobListResponseV2(BaseModel):
    jobs: List[JobSummaryV2]
    total: int
    limit: int
    offset: int
