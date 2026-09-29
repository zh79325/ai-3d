"""FastAPI 路由：``/v2`` 四阶段流程（素材库 + 作业）。

与 :mod:`server`（``/v1`` 一对一任务）**并存**：过渡期两套路由挂在同一应用上，
``/v1`` 代码路径零改动，``/v2`` 只读写 ``assets``/``bindings``/``jobs`` 三组新表与
``output/assets/``、``output/jobs/`` 新目录，旧 ``output/tasks/`` 原地封存不读不删。

四阶段（模型侧）：**S1 导入矫正 → S2 绑定 → S3 重定向 → S4 导出**；动画素材只走
S1，入库后可被任意模型复用。当前已实现 S1（素材库全套）与 S2（多视角回传、关节
微调、绑定/重绑），S3~S4 的路由在 P4~P5 接入。

S1 是**同步**跑的（实测单个模型 0.15~0.32s，含 align.json 写盘），上传即可拿到
自动对齐提案，前端立刻能渲染外切盒与六面标签；FBX 走 assimp 会慢一些，仍按同步
处理，超时问题在真机验证后再决定是否改后台线程。

S2 恰好相反：蒙皮（libigl BBW）万级顶点就要数秒到十几秒，一律后台线程跑，进度
落 ``bindings.stage`` / ``message``，前端轮询 ``GET /v2/assets/{id}`` 画进度条。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__, glb_io
from .asset_store import (
    AssetStore,
    ConflictError,
    NotFoundError,
    StateError,
    get_asset_store,
)
from .asset_worker import (
    AlignError,
    AlignInputError,
    AssetWorker,
    get_asset_worker,
)
from .axis_norm import ALIGN_VERSION
from .job_worker import BindingError, BindingInputError, JobWorker, get_job_worker
from .obb import FACE_LABELS
from .schemas import DEFAULT_CAMERAS, CameraSpec, HealthResponse, RigPatch, ViewSpec
from .schemas_v2 import (
    AlignPatch,
    AlignResponse,
    AssetCreate,
    AssetInfo,
    AssetKind,
    AssetListResponse,
    AssetState,
    BindingState,
    CreateAssetResponse,
)
from .settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["retarget-v2"])

GLB_MEDIA = "model/gltf-binary"
# 上传扩展名白名单：与 glb_io.classify 支持的输入一致（先看魔数，这里是快速拒绝）
ALLOWED_EXT = frozenset({".fbx", ".glb", ".gltf"})
# 视角图的 Content-Type（与 asset_store.VIEW_IMAGE_EXTS 对齐）
VIEW_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".bmp": "image/bmp"}
VIEW_SIZE = 512


def _store() -> AssetStore:
    return get_asset_store()


def _worker() -> AssetWorker:
    return get_asset_worker()


def _job_worker() -> JobWorker:
    return get_job_worker()


def _require_asset(asset_id: str) -> Dict[str, Any]:
    row = _store().get_row(asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"素材不存在：{asset_id}")
    return row


def _asset_info(asset_id: str) -> AssetInfo:
    info = _store().get_asset_info(asset_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"素材不存在：{asset_id}")
    return info


def _align_response(asset_id: str, align: Dict[str, Any], message: str = "") -> AlignResponse:
    row = _require_asset(asset_id)
    return AlignResponse(asset_id=asset_id, state=AssetState(row["state"]),
                         filename=row["filename"], align=align, error=row["error"],
                         message=message)


def _run_align(asset_id: str, patch: Optional[AlignPatch]) -> Dict[str, Any]:
    """跑 S1 并把领域异常翻成 HTTP 状态码。

    ``AlignInputError``（缺文件/格式不支持/无网格/面映射非法）→ 400，用户改输入就能修；
    其余 :class:`AlignError`（assimp 失败、写盘失败）→ 500。两种情况素材都已落
    ``FAILED`` 且 ``error`` 入库，前端刷新详情即可看到原因。
    """
    try:
        return _worker().run_align(asset_id, patch)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"素材不存在：{exc}") from exc
    except AlignInputError as exc:
        raise HTTPException(status_code=400, detail=f"导入矫正失败：{exc}") from exc
    except AlignError as exc:
        logger.exception("S1 服务端失败 asset=%s", asset_id)
        raise HTTPException(status_code=500, detail=f"导入矫正失败：{exc}") from exc


def _start_binding(asset_id: str, pose_ai: bool) -> JSONResponse:
    """启动 S2 并把领域异常翻成 HTTP 状态码。

    准入校验（素材存在 / 是模型 / S1 已落库 / 未在跑）在 :meth:`JobWorker.start_binding`
    里是**同步**做的，故这里能把错误当场回给前端：``BindingInputError``（动画素材、
    缺产物）→ 400，``StateError``（S1 未完成、已有 S2 在跑）→ 409。
    """
    try:
        _job_worker().start_binding(asset_id, pose_ai=pose_ai)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"素材不存在：{exc}") from exc
    except BindingInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    info = _store().binding_info(asset_id)
    return JSONResponse({"status": "started", "asset_id": asset_id, "pose_ai": pose_ai,
                         "binding": info.model_dump(mode="json"),
                         "message": "S2 绑定已启动，轮询素材详情看进度"})


def _resume_binding(asset_id: str, pose_ai: bool = True) -> bool:
    """尝试起/续跑 S2，**失败不抛**（返回是否真的起来了）。

    用于「主作业已成功」的场景：确认 S1、回传视角图。这时 S2 起不来（上一轮还在
    跑、素材已删）不该把已成功的请求翻成错误，否则会误导用户重做主作业。
    """
    try:
        _job_worker().start_binding(asset_id, pose_ai=pose_ai)
        return True
    except (BindingError, NotFoundError, StateError) as exc:
        logger.warning("素材 %s 的 S2 未启动：%s", asset_id, exc)
        return False


# --------------------------------------------------------------------------- #
# 健康 / 约定
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """``/v2`` 健康检查：counts 为素材按状态计数（与 ``/v1`` 的任务计数区分）。"""
    return HealthResponse(status="ok", version=__version__,
                          assimp=glb_io.resolve_assimp(get_settings()) or None,
                          counts=_store().count_by_state())


@router.get("/conventions")
async def conventions() -> JSONResponse:
    """下发前后端**共享的固定约定**，避免面标签/语义轴在两侧各写一份走偏。

    面编号约定（规范系米制外切盒）：``1=+X left, 2=-X right, 3=+Y up, 4=-Y down,
    5=+Z front, 6=-Z back``，标签见 ``face_labels``。
    """
    return JSONResponse({
        "align_version": ALIGN_VERSION,
        "face_labels": list(FACE_LABELS),
        "canonical_axes": {"up": "+Y", "front": "+Z", "left": "+X"},
        "asset_states": [s.value for s in AssetState],
        "asset_kinds": [k.value for k in AssetKind],
    })


# --------------------------------------------------------------------------- #
# 素材库：创建 / 列表 / 详情 / 删除
# --------------------------------------------------------------------------- #
@router.post("/assets", response_model=CreateAssetResponse, status_code=201)
async def create_asset(req: AssetCreate) -> CreateAssetResponse:
    """建素材（先建后传）：``kind=model`` 走四阶段，``kind=animation`` 只走 S1。"""
    asset_id = _store().create_asset(req.kind, (req.name or "").strip() or None)
    return CreateAssetResponse(asset_id=asset_id, state=AssetState.CREATED,
                               message="素材已创建，请上传文件")


@router.get("/assets", response_model=AssetListResponse)
async def list_assets(
    kind: Optional[AssetKind] = Query(None, description="按类型过滤"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AssetListResponse:
    summaries, total = _store().list_assets(kind, limit=limit, offset=offset)
    return AssetListResponse(assets=summaries, total=total, limit=limit, offset=offset)


@router.get("/assets/{asset_id}", response_model=AssetInfo)
async def get_asset(asset_id: str) -> AssetInfo:
    """详情：state / align（align.json 全文）/ meta（结构概要）/ binding（S2 摘要）。"""
    return _asset_info(asset_id)


@router.delete("/assets/{asset_id}")
async def delete_asset(
    asset_id: str,
    force: bool = Query(False, description="被作业引用时也删"),
) -> JSONResponse:
    """删素材：连带 bindings（FK CASCADE）与引用它的作业目录。

    模型被作业引用时默认拒绝（409），``force=true`` 才连带删作业；动画被引用时 FK 是
    ``SET NULL``，作业保留、只是失去动画来源。
    """
    store = _store()
    _require_asset(asset_id)
    refs = store.count_jobs(asset_id)
    if refs["as_model"] and not force:
        raise HTTPException(
            status_code=409,
            detail=f"该模型素材被 {refs['as_model']} 个作业引用，确认要连带删除请加 force=true")
    store.delete_asset(asset_id)
    return JSONResponse({"status": "deleted", "asset_id": asset_id, "references": refs})


# --------------------------------------------------------------------------- #
# S1 导入矫正：上传 → 自动跑 → 人工修正 → 确认
# --------------------------------------------------------------------------- #
@router.post("/assets/{asset_id}/upload", response_model=AlignResponse)
async def upload_asset(
    asset_id: str,
    file: UploadFile = File(..., description="FBX / GLB / glTF"),
) -> AlignResponse:
    """上传原件并**同步跑完 S1**，返回自动校准提案（PCA / 外切盒 / 单位 / auto / final）。"""
    _require_asset(asset_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext or '(无扩展名)'}，仅支持 FBX / GLB / glTF")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    try:
        raw = _store().save_upload(asset_id, file.filename or "asset", data)
    except (OSError, NotFoundError) as exc:
        raise HTTPException(status_code=500, detail=f"保存上传文件失败：{exc}") from exc
    align = _run_align(asset_id, None)
    unit = align.get("unit") or {}
    logger.info("素材 %s 上传 %s 并完成 S1：unit=%s height=%.4fm",
                asset_id, raw.name, unit.get("detected"), unit.get("height_m") or 0.0)
    return _align_response(asset_id, align, "S1 导入矫正完成，请核对角色朝向与单位后确认")


@router.get("/assets/{asset_id}/align")
async def get_align(asset_id: str) -> JSONResponse:
    """align.json（S1 唯一产物）。"""
    _require_asset(asset_id)
    align = _store().load_align(asset_id)
    if align is None:
        raise HTTPException(status_code=404, detail="S1 尚未完成，align.json 不存在")
    return JSONResponse(align)


@router.patch("/assets/{asset_id}/align", response_model=AlignResponse)
async def patch_align(asset_id: str, patch: AlignPatch) -> AlignResponse:
    """人工修正 S1：只改 ``manual`` 段（前后翻转二选 / 单位 / 真实身高）。

    后端在**原始模型坐标**下重测 PCA / 校准基 / 包围盒与身高（``axis_norm._reset_canon``
    先复位旧的 canon 节点），重算 ``final`` 并原地改写 canon 节点的旋转与缩放；反复
    修正不会累积。``reset=true`` 清空全部人工修正，回到自动探测结果。
    """
    _require_asset(asset_id)
    if _store().load_align(asset_id) is None:
        raise HTTPException(status_code=400, detail="请先上传文件并完成 S1，再修正对齐")
    align = _run_align(asset_id, patch)
    msg = "已恢复自动探测" if patch.reset else "已按人工修正重算对齐"
    return _align_response(asset_id, align, msg)


@router.post("/assets/{asset_id}/confirm", response_model=AssetInfo)
async def confirm_asset(
    asset_id: str,
    pose_ai: bool = Query(True, description="false = 跳过 DWPose，直接按人体比例生骨架"),
    auto_bind: bool = Query(True, description="false = 只确认 S1，不自动起 S2"),
) -> AssetInfo:
    """确认 S1 结果。

    动画素材到此即 ``READY``（可被任意模型复用）；模型素材置 ``READY`` 并**立即起 S2**
    —— 对应「上传完模型后自动先跑一遍」：前端拿到响应后先渲染多视角图回传，S2 则先
    挂在 ``WAIT_VIEWS`` 等图。``auto_bind=false`` 只确认不起 S2（批量入库、脚本、单测用）。

    S2 的触发放在路由层而不是 ``AssetWorker.confirm`` 里：两个 worker 互相依赖会让
    单测起真线程，而且直接调 ``confirm()`` 的调用方不该被塞一个后台任务。

    ``ALIGN_READY`` 是 S2 的硬性准入门槛：``skin.weld_epsilon=2e-3``、
    ``retarget.foot_contact_height=0.12`` 都是米制绝对阈值，单位未矫正就绑定会得到
    完全错误的焊接半径。
    """
    try:
        info = _worker().confirm(asset_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"素材不存在：{exc}") from exc
    except AlignInputError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if auto_bind and info["kind"] == AssetKind.MODEL.value:
        _resume_binding(asset_id, pose_ai)      # S1 已确认成功，S2 起不来也不报错
    return _asset_info(asset_id)


@router.post("/assets/{asset_id}/realign", response_model=AlignResponse)
async def realign(asset_id: str) -> AlignResponse:
    """重跑 S1（重做格式归一 + 对齐），保留既有 ``manual`` 修正。

    与 PATCH 的区别：PATCH 只在既有 ``asset.glb`` 上重算，本接口从上传原件重新归一，
    用于原件被替换或产物损坏的场景。
    """
    row = _require_asset(asset_id)
    if not row["filename"]:
        raise HTTPException(status_code=400, detail="素材尚未上传文件")
    glb = _store().glb_path(asset_id)
    if glb.exists():
        glb.unlink()          # 强制走 normalize_to_glb 全路径
    align = _run_align(asset_id, None)
    return _align_response(asset_id, align, "已重新归一并对齐")


# --------------------------------------------------------------------------- #
# 产物下载
# --------------------------------------------------------------------------- #
def _file_response(path: Path, media_type: str, download_name: str,
                   missing: str) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail=missing)
    # 产物会被重跑覆盖：禁浏览器缓存，否则修正对齐后前端拿到旧字节
    return FileResponse(path, media_type=media_type, filename=download_name,
                        headers={"Cache-Control": "no-store"})


@router.get("/assets/{asset_id}/glb")
async def get_asset_glb(asset_id: str) -> FileResponse:
    """归一化 + 对齐后的 GLB（前端预览与后续阶段的唯一输入）。"""
    _require_asset(asset_id)
    return _file_response(_store().glb_path(asset_id), GLB_MEDIA, "asset.glb",
                          "归一化 GLB 尚未生成，请先上传文件")


@router.get("/assets/{asset_id}/raw")
async def get_asset_raw(asset_id: str) -> FileResponse:
    """上传原件（复查用；重跑 S1 的输入）。"""
    row = _require_asset(asset_id)
    if not row["filename"]:
        raise HTTPException(status_code=404, detail="素材尚未上传文件")
    path = _store().asset_dir(asset_id) / row["filename"]
    return _file_response(path, "application/octet-stream", row["filename"],
                          "上传原件已丢失，请重新上传")


@router.get("/assets/{asset_id}/views")
async def list_views(asset_id: str) -> JSONResponse:
    """S2 已回传的多视角图清单（文件名主干即 ``view_id``）。"""
    _require_asset(asset_id)
    files = _store().list_views(asset_id)
    return JSONResponse({"asset_id": asset_id, "views": files, "total": len(files),
                         "urls": [f"/v2/assets/{asset_id}/views/{n}" for n in files]})


@router.get("/assets/{asset_id}/views/{name}")
async def get_view_image(asset_id: str, name: str) -> FileResponse:
    """单张视角图（前端复核回传质量、排查推理失败时用）。"""
    _require_asset(asset_id)
    safe = Path(name).name                       # 去目录成分，防 ../ 穿越
    if safe != name or not safe:
        raise HTTPException(status_code=400, detail=f"非法的视角图文件名：{name}")
    path = _store().views_dir(asset_id) / safe
    media = VIEW_MEDIA.get(path.suffix.lower(), "application/octet-stream")
    return _file_response(path, media, safe, f"视角图不存在：{safe}")


# --------------------------------------------------------------------------- #
# S2 绑定：多视角回传 → 推理/三角化 → 骨架 + 蒙皮
# --------------------------------------------------------------------------- #
@router.get("/assets/{asset_id}/view_spec", response_model=ViewSpec)
async def get_view_spec(asset_id: str) -> ViewSpec:
    """下发多视角渲染规格（前端离屏渲染后回传，S2 按同一份相机参数三角化）。

    下发的同时把规格持久化到 ``view_spec.json``：三角化必须用**前端实际渲染时**的
    方位角/仰角/画布尺寸，两侧各写一套默认值会让重建出来的关节整体错位——而这类
    错误在预览里只表现为「骨架略微偏移」，极难排查。
    """
    _require_asset(asset_id)
    spec = ViewSpec(
        model_url=f"/v2/assets/{asset_id}/glb",
        cameras=[CameraSpec(view_id=vid, name=vid, azimuth=az, elevation=el)
                 for (vid, az, el) in DEFAULT_CAMERAS],
        width=VIEW_SIZE, height=VIEW_SIZE,
        up_axis="Y", background="#ffffff", ortho=True,
    )
    try:
        _store().save_view_spec(asset_id, spec.model_dump(mode="json"))
    except OSError as exc:
        logger.warning("持久化 view_spec 失败（三角化会回落默认相机）：%s", exc)
    return spec


@router.post("/assets/{asset_id}/views")
async def upload_views(
    asset_id: str,
    files: List[UploadFile] = File(..., description="多视角渲染图，文件名主干 = view_id"),
) -> JSONResponse:
    """接收前端回传的多视角图；若 S2 正挂在 ``WAIT_VIEWS`` 则自动续跑。

    文件名必须与 :func:`get_view_spec` 下发的 ``view_id`` 对应（``front.png`` /
    ``right.png`` / …）：对不上时该视角在三角化里只是不参与，**不会报错**，骨架会
    静默退化到比例先验。落盘前会先清空 ``views/``，不让上一次回传的残留视角混进来。
    """
    _require_asset(asset_id)
    payloads: List[Tuple[str, bytes]] = []
    for f in files:
        payloads.append((Path(f.filename or "view.png").name, await f.read()))
    try:
        saved = _store().save_views(asset_id, payloads)
    except StateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, NotFoundError) as exc:
        raise HTTPException(status_code=500, detail=f"保存视角图失败：{exc}") from exc
    was_waiting = _store().binding_info(asset_id).state == BindingState.WAIT_VIEWS
    resumed = _resume_binding(asset_id, pose_ai=True) if was_waiting else False
    return JSONResponse({
        "status": "ok", "asset_id": asset_id, "views": saved, "total": len(saved),
        "resumed": resumed,
        "message": (f"已收 {len(saved)} 张视角图，S2 继续跑推理与三角化" if resumed
                    else f"已收 {len(saved)} 张视角图"),
        "binding": _store().binding_info(asset_id).model_dump(mode="json"),
    })


@router.get("/assets/{asset_id}/binding")
async def get_binding(asset_id: str) -> JSONResponse:
    """S2 绑定摘要：state / stage / message / confidence / revision / 产物 / 蒙皮报告。"""
    _require_asset(asset_id)
    info = _store().binding_info(asset_id)
    return JSONResponse(info.model_dump(mode="json"))


@router.post("/assets/{asset_id}/bind")
async def bind(
    asset_id: str,
    pose_ai: bool = Query(True, description="false = 跳过 DWPose，直接按人体比例生骨架"),
) -> JSONResponse:
    """启动 S2 绑定（后台线程）。

    ``pose_ai=true`` 时若视角图还没回传，S2 会停在 ``WAIT_VIEWS``（**不是失败**），等
    :func:`upload_views` 到达后自动续跑；``pose_ai=false`` 适合非人形道具，或 DWPose
    检不到人时的兜底（不需要视角图，直接跑完）。
    """
    _require_asset(asset_id)
    return _start_binding(asset_id, pose_ai)


@router.post("/assets/{asset_id}/rebind")
async def rebind(
    asset_id: str,
    pose_ai: bool = Query(True, description="同 /bind"),
) -> JSONResponse:
    """重跑 S2（换视角图、或上一次失败后重试），与 ``/bind`` 同一实现。

    保留独立路径是为了语义清晰：``rebind`` 会**重写 ``rig.json``**（人工微调的关节与
    revision 归零），前端应给二次确认；只想换蒙皮请用 ``PATCH /rig?reskin=true``。
    重跑失败时上一次的绑定产物**保留**（见 ``JobWorker._fail_binding``），素材不会退回
    未绑定状态。
    """
    _require_asset(asset_id)
    return _start_binding(asset_id, pose_ai)


# --------------------------------------------------------------------------- #
# S2 产物：rig.json（revision 乐观锁）
# --------------------------------------------------------------------------- #
@router.get("/assets/{asset_id}/rig")
async def get_rig(asset_id: str) -> JSONResponse:
    """rig.json：22 语义关节的 head/tail + revision + confidence + source。"""
    _require_asset(asset_id)
    rig = _store().load_binding_json(asset_id, "rig")
    if rig is None:
        raise HTTPException(status_code=404, detail="rig.json 尚未生成，请先跑完 S2")
    return JSONResponse(rig)


@router.patch("/assets/{asset_id}/rig")
async def patch_rig(
    asset_id: str,
    patch: RigPatch,
    reskin: bool = Query(True, description="改完关节后重算蒙皮（拖拽中可传 false）"),
) -> JSONResponse:
    """人工微调关节（revision 乐观锁），默认顺带重算蒙皮。

    关节一动，BBW 的 handle 选骨与权重全部失效，不重算就会看到「骨架对了但网格还是
    旧姿势」；重算走 :meth:`JobWorker.run_reskin`，**不重写 rig.json**，人工版本号与
    ``source="manual"`` 标记都留着。连着调好几根时用 ``reskin=false`` 只存骨架
    （秒回），调完再单独 POST ``/reskin`` 一次性重算。
    """
    _require_asset(asset_id)
    try:
        new_rev = _job_worker().patch_rig(asset_id, patch)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"素材不存在：{exc}") from exc
    except BindingInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    started = False
    if reskin:
        try:
            _job_worker().start_reskin(asset_id)
            started = True
        except (BindingError, NotFoundError, StateError) as exc:
            # 骨架已存下；蒙皮重算起不来（多半是上一轮还在跑）不该让 PATCH 报错
            logger.warning("素材 %s 的蒙皮重算未启动：%s", asset_id, exc)
    return JSONResponse({"status": "ok", "asset_id": asset_id, "rig_revision": new_rev,
                         "reskin": "started" if started else "skipped",
                         "binding": _store().binding_info(asset_id).model_dump(mode="json")})


@router.get("/assets/{asset_id}/skin_report")
async def get_skin_report(asset_id: str) -> JSONResponse:
    """skin_report.json：蒙皮质量指标（零权重行、应变、连通分量兜底数）。"""
    _require_asset(asset_id)
    report = _store().load_binding_json(asset_id, "skin_report")
    if report is None:
        raise HTTPException(status_code=404, detail="蒙皮报告尚未生成，请先跑完 S2")
    return JSONResponse(report)


@router.post("/assets/{asset_id}/reskin")
async def reskin_asset(asset_id: str) -> JSONResponse:
    """**只**重算蒙皮，不动骨架。

    两个场景需要它：连着微调了好几根关节（每次 ``PATCH /rig?reskin=false``）后一次
    性重算；以及上一次蒙皮失败（BBW 崩了、进程被杀）后重试。走 PATCH 带空补丁也能
    达到同样效果，但那会白白把 revision +1，还会把没动过的关节标记成人工改过。
    """
    try:
        _job_worker().start_reskin(asset_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"素材不存在：{exc}") from exc
    except BindingInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"status": "started", "asset_id": asset_id, "reskin": True,
                         "binding": _store().binding_info(asset_id).model_dump(mode="json"),
                         "message": "已排队重算蒙皮，轮询素材详情看进度"})


# --------------------------------------------------------------------------- #
# 应用装配（独立运行用；主应用见 ai3d/viewer/server.py::_mount_retarget）
# --------------------------------------------------------------------------- #
def create_app():
    """仅含 ``/v2`` 的最小应用（联调 React 开发服务器时用）。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    settings = get_settings()
    settings.ensure_dirs()
    get_asset_store()          # 触发建表
    app = FastAPI(title="3D 骨骼动画迁移工具 /v2", version=__version__,
                  description="四阶段流程：S1 导入矫正 → S2 绑定 → S3 重定向 → S4 导出")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins or ["*"],
        allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(router)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    settings = get_settings()
    settings.ensure_dirs()
    app = create_app()
    port = settings.server.port + 1     # 与 /v1 服务错开，便于并存联调
    logger.info("🎯 /v2 服务： http://%s:%s/docs  项目目录=%s 数据库=%s",
                settings.server.host, port, settings.project_dir, settings.sqlite_file)
    uvicorn.run(app, host=settings.server.host, port=port)


# 便于 pytest 与主应用复用异常映射
__all__ = ["router", "create_app", "main", "ALLOWED_EXT",
           "NotFoundError", "ConflictError", "StateError",
           "BindingError", "BindingInputError"]


if __name__ == "__main__":
    main()
