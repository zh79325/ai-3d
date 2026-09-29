"""FastAPI 服务：/v1 接口 + 前端页面。

参照 ``ai3d/viewer/server.py`` 的 router/HTML/后台线程模式，但任务状态落 SQLite，
服务重启后凭 task_id 仍可找回历史任务与产物。

独立运行：``python -m ai3d.retarget.server``（或项目根 ``run_retarget_server.py``）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import __version__, glb_io
from .schemas import (
    ArtifactKind,
    CreateJobResponse,
    DEFAULT_CAMERAS,
    HealthResponse,
    JobConfig,
    JobInfo,
    JobListResponse,
    JobState,
    MappingPatch,
    RerunRequest,
    RigPatch,
    Stage,
    StageStatus,
    ViewSpec,
    CameraSpec,
)
from .settings import get_settings
from .task_store import ConflictError, NotFoundError, TaskStore, get_store
from .worker import get_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["retarget"])

VIEWER_DIR = Path(__file__).parent / "viewer"
GLB_MEDIA = "model/gltf-binary"


def _store() -> TaskStore:
    return get_store()


def _safe_suffix(filename: Optional[str], default: str) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext else default


def _require_task(task_id: str) -> None:
    if not _store().exists(task_id):
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")


# --------------------------------------------------------------------------- #
# 健康 / 配置
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = _store()
    return HealthResponse(
        status="ok",
        version=__version__,
        assimp=glb_io.resolve_assimp(get_settings()) or None,
        counts=s.count_by_state(),
    )


@router.get("/config")
async def get_config() -> JSONResponse:
    return JSONResponse(get_settings().to_public_dict())


# --------------------------------------------------------------------------- #
# 任务：创建 / 列表 / 详情
# --------------------------------------------------------------------------- #
@router.post("/jobs", response_model=CreateJobResponse)
async def create_job(
    source_file: Optional[UploadFile] = File(None, description="源：已绑定骨骼动画的 FBX/GLB"),
    target_file: Optional[UploadFile] = File(None, description="目标：无骨骼网格 FBX/GLB"),
    name: str = Form("", description="任务名称（可选）"),
    config: str = Form("{}", description="JobConfig JSON"),
) -> CreateJobResponse:
    try:
        cfg = JobConfig.model_validate_json(config or "{}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"config 非法 JSON：{exc}") from exc

    s = _store()
    # 仅名称：先建空任务，进入详情后再分别上传模型/动画
    if source_file is None and target_file is None:
        task_id = s.create_task("", "", cfg, name=name.strip() or "未命名任务",
                                state=JobState.CREATED)
        return CreateJobResponse(
            task_id=task_id, state=JobState.CREATED, current_stage=None,
            message="任务已创建，请进入后上传模型与动画",
        )
    if source_file is None or target_file is None:
        raise HTTPException(status_code=400,
                            detail="源文件与目标文件需同时提供，或都不提供（仅建空任务）")

    task_id = s.create_task(source_file.filename or "source", target_file.filename or "target",
                            cfg, name=name.strip() or (source_file.filename or "source"))
    try:
        up = s.upload_dir(task_id)
        up.mkdir(parents=True, exist_ok=True)
        src_raw = up / f"source_raw{_safe_suffix(source_file.filename, '.glb')}"
        tgt_raw = up / f"target_raw{_safe_suffix(target_file.filename, '.glb')}"
        src_raw.write_bytes(await source_file.read())
        tgt_raw.write_bytes(await target_file.read())
        s.register_artifact(task_id, ArtifactKind.SOURCE_RAW, src_raw)
        s.register_artifact(task_id, ArtifactKind.TARGET_RAW, tgt_raw)
    except Exception as exc:  # noqa: BLE001
        s.update_task(task_id, state="FAILED", error=f"保存上传文件失败：{exc}")
        raise HTTPException(status_code=500, detail=f"保存上传文件失败：{exc}") from exc

    get_worker().start_async(task_id, from_stage=Stage.NORMALIZE)
    return CreateJobResponse(
        task_id=task_id, state="QUEUED", current_stage=Stage.NORMALIZE,
        message="任务已创建并开始归一化",
    )


def _has_artifact(task_id: str, kind: ArtifactKind) -> bool:
    p = _store().get_artifact_path(task_id, kind)
    return bool(p) and Path(p).exists()


@router.post("/jobs/{task_id}/upload/target")
async def upload_target(task_id: str, file: UploadFile = File(...)) -> JSONResponse:
    """上传目标模型（无骨骼网格）。上传即归一化，便于立即预览；不自动跑流水线。"""
    _require_task(task_id)
    s = _store()
    up = s.upload_dir(task_id)
    up.mkdir(parents=True, exist_ok=True)
    tgt_raw = up / f"target_raw{_safe_suffix(file.filename, '.glb')}"
    tgt_raw.write_bytes(await file.read())
    s.register_artifact(task_id, ArtifactKind.TARGET_RAW, tgt_raw)
    s.update_task(task_id, target_filename=file.filename or "target")
    try:
        tgt_glb = s.path_for(task_id, ArtifactKind.TARGET)
        glb_io.normalize_to_glb(tgt_raw, tgt_glb)
        s.register_artifact(task_id, ArtifactKind.TARGET, tgt_glb)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"目标归一化失败：{exc}") from exc
    return JSONResponse({"status": "ok", "task_id": task_id,
                         "preview_ready": True, "pipeline_started": False})


@router.post("/jobs/{task_id}/upload/source")
async def upload_source(task_id: str, file: UploadFile = File(...)) -> JSONResponse:
    """上传源（已绑定骨骼动画）。上传即归一化；不自动跑流水线，由前端按钮触发。"""
    _require_task(task_id)
    s = _store()
    up = s.upload_dir(task_id)
    up.mkdir(parents=True, exist_ok=True)
    src_raw = up / f"source_raw{_safe_suffix(file.filename, '.glb')}"
    src_raw.write_bytes(await file.read())
    s.register_artifact(task_id, ArtifactKind.SOURCE_RAW, src_raw)
    s.update_task(task_id, source_filename=file.filename or "source")
    try:
        src_glb = s.path_for(task_id, ArtifactKind.SOURCE)
        glb_io.normalize_to_glb(src_raw, src_glb)
        s.register_artifact(task_id, ArtifactKind.SOURCE, src_glb)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"源归一化失败：{exc}") from exc
    return JSONResponse({"status": "ok", "task_id": task_id, "pipeline_started": False})


@router.post("/jobs/{task_id}/detect")
async def detect(task_id: str) -> JSONResponse:
    """目标侧骨骼检测：NORMALIZE→…→BUILD_RIG（YOLO/姿态 + 三维骨骼），不需源。"""
    _require_task(task_id)
    if not _has_artifact(task_id, ArtifactKind.TARGET_RAW):
        raise HTTPException(status_code=400, detail="请先上传模型")
    s = _store()
    s.set_state(task_id, JobState.QUEUED, current_stage=Stage.NORMALIZE)
    get_worker().start_async(task_id, from_stage=Stage.NORMALIZE, to_stage=Stage.BUILD_RIG)
    return JSONResponse({"status": "ok", "task_id": task_id, "to_stage": "BUILD_RIG"})


@router.post("/jobs/{task_id}/migrate")
async def migrate(task_id: str) -> JSONResponse:
    """动画迁移：MAP_SOURCE→…→EXPORT_VERIFY，需源+已建骨骼。"""
    _require_task(task_id)
    if not _has_artifact(task_id, ArtifactKind.SOURCE_RAW):
        raise HTTPException(status_code=400, detail="请先上传动画（源）")
    s = _store()
    if s.load_json_artifact(task_id, "rig.json") is None:
        raise HTTPException(status_code=400, detail="请先完成骨骼检测（绑定骨骼）")
    s.set_state(task_id, JobState.QUEUED, current_stage=Stage.MAP_SOURCE)
    get_worker().start_async(task_id, from_stage=Stage.MAP_SOURCE)
    return JSONResponse({"status": "ok", "task_id": task_id, "from_stage": "MAP_SOURCE"})


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> JobListResponse:
    summaries, total = _store().list_jobs(limit=limit, offset=offset)
    return JobListResponse(tasks=summaries, total=total, limit=limit, offset=offset)


@router.get("/jobs/{task_id}", response_model=JobInfo)
async def get_job(task_id: str) -> JobInfo:
    info = _store().get_job_info(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    return info


@router.delete("/jobs/{task_id}")
async def delete_job(task_id: str) -> JSONResponse:
    _require_task(task_id)
    _store().delete_task(task_id)
    return JSONResponse({"status": "deleted", "task_id": task_id})


@router.get("/jobs/{task_id}/artifacts")
async def get_artifacts(task_id: str) -> JSONResponse:
    _require_task(task_id)
    return JSONResponse([a.model_dump() for a in _store().get_artifacts(task_id)])


# --------------------------------------------------------------------------- #
# GLB 产物
# --------------------------------------------------------------------------- #
def _glb_response(task_id: str, kind: ArtifactKind, download_name: str) -> FileResponse:
    path = _store().get_artifact_path(task_id, kind)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"{kind.value} 尚未生成")
    # 产物会被流水线/重跑覆盖：禁浏览器缓存，否则重上传/重跑后前端拿到旧字节
    return FileResponse(path, media_type=GLB_MEDIA, filename=download_name,
                        headers={"Cache-Control": "no-store"})


@router.get("/jobs/{task_id}/target.glb")
async def get_target_glb(task_id: str) -> FileResponse:
    _require_task(task_id)
    return _glb_response(task_id, ArtifactKind.TARGET, "target.glb")


@router.get("/jobs/{task_id}/source.glb")
async def get_source_glb(task_id: str) -> FileResponse:
    _require_task(task_id)
    return _glb_response(task_id, ArtifactKind.SOURCE, "source.glb")


@router.get("/jobs/{task_id}/result.glb")
async def get_result_glb(task_id: str) -> FileResponse:
    _require_task(task_id)
    return _glb_response(task_id, ArtifactKind.RESULT, "result.glb")


# --------------------------------------------------------------------------- #
# 多视角渲染（前端渲染，后端提供规格 + 接收回传）
# --------------------------------------------------------------------------- #
@router.get("/jobs/{task_id}/view_spec", response_model=ViewSpec)
async def get_view_spec(task_id: str) -> ViewSpec:
    _require_task(task_id)
    cameras = [CameraSpec(view_id=vid, name=vid, azimuth=az, elevation=el)
               for (vid, az, el) in DEFAULT_CAMERAS]
    spec = ViewSpec(
        model_url=f"/v1/jobs/{task_id}/target.glb",
        cameras=cameras,
        width=512, height=512, up_axis="Y", background="#ffffff", ortho=True,
    )
    # 持久化下发规格，供 SOLVE_RIG 按同一相机参数三角化（重启后仍一致）
    try:
        vs_path = _store().task_dir(task_id) / "view_spec.json"
        vs_path.parent.mkdir(parents=True, exist_ok=True)
        vs_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:  # noqa: PERF203
        logger.warning("持久化 view_spec 失败：%s", exc)
    return spec


@router.post("/jobs/{task_id}/views")
async def upload_views(
    task_id: str,
    files: List[UploadFile] = File(..., description="多视角渲染图（PNG/JPG）"),
) -> JSONResponse:
    _require_task(task_id)
    s = _store()
    vdir = s.views_dir(task_id)
    vdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename or "view.png").name  # 去路径，防穿越
        dest = vdir / name
        dest.write_bytes(await f.read())
        saved.append(name)
    s.register_artifact(task_id, ArtifactKind.VIEWS, vdir)
    s.set_stage(task_id, Stage.RENDER_VIEWS, StageStatus.DONE, f"已接收 {len(saved)} 张视角图")
    # 视角图就绪，继续下游；若尚无源文件则只跑到 BUILD_RIG（目标侧骨骼检测）
    to_stage = None if _has_artifact(task_id, ArtifactKind.SOURCE_RAW) else Stage.BUILD_RIG
    get_worker().start_async(task_id, from_stage=Stage.POSE_INFER, to_stage=to_stage)
    return JSONResponse({"status": "ok", "task_id": task_id, "views": saved})


# --------------------------------------------------------------------------- #
# rig / mapping（revision 乐观锁）
# --------------------------------------------------------------------------- #
@router.get("/jobs/{task_id}/rig")
async def get_rig(task_id: str) -> JSONResponse:
    _require_task(task_id)
    rig = _store().load_json_artifact(task_id, "rig.json")
    if rig is None:
        raise HTTPException(status_code=404, detail="rig 尚未生成")
    return JSONResponse(rig)


@router.patch("/jobs/{task_id}/rig")
async def patch_rig(task_id: str, patch: RigPatch) -> JSONResponse:
    _require_task(task_id)
    s = _store()
    rig = s.load_json_artifact(task_id, "rig.json")
    if rig is None:
        raise HTTPException(status_code=404, detail="rig 尚未生成，无法校正")
    joints = rig.setdefault("joints", {})
    for jid, jp in patch.joints.items():
        cur = joints.setdefault(jid, {})
        upd = jp.model_dump(exclude_none=True)
        cur.update(upd)
    rig["revision"] = patch.revision
    try:
        new_rev = s.save_rig(task_id, patch.revision, rig)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"status": "ok", "rig_revision": new_rev})


@router.get("/jobs/{task_id}/mapping")
async def get_mapping(task_id: str) -> JSONResponse:
    _require_task(task_id)
    mapping = _store().load_json_artifact(task_id, "mapping.json")
    if mapping is None:
        raise HTTPException(status_code=404, detail="mapping 尚未生成")
    return JSONResponse(mapping)


@router.patch("/jobs/{task_id}/mapping")
async def patch_mapping(task_id: str, patch: MappingPatch) -> JSONResponse:
    _require_task(task_id)
    s = _store()
    mapping = s.load_json_artifact(task_id, "mapping.json")
    if mapping is None:
        raise HTTPException(status_code=404, detail="mapping 尚未生成，无法覆盖")
    overrides = mapping.setdefault("overrides", {})
    overrides.update(patch.overrides)
    mapping["revision"] = patch.revision
    try:
        new_rev = s.save_mapping(task_id, patch.revision, mapping)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"status": "ok", "mapping_revision": new_rev})


# --------------------------------------------------------------------------- #
# rerun / report
# --------------------------------------------------------------------------- #
@router.post("/jobs/{task_id}/rerun")
async def rerun(task_id: str, req: RerunRequest) -> JSONResponse:
    _require_task(task_id)
    s = _store()
    s.reset_from_stage(task_id, req.from_stage)
    get_worker().start_async(task_id, from_stage=req.from_stage, force=req.force)
    return JSONResponse({"status": "ok", "task_id": task_id,
                         "from_stage": req.from_stage.value, "force": req.force})


@router.get("/jobs/{task_id}/report")
async def get_report(task_id: str) -> JSONResponse:
    _require_task(task_id)
    report = _store().load_json_artifact(task_id, "report.json")
    if report is None:
        raise HTTPException(status_code=404, detail="report 尚未生成")
    return JSONResponse(report)


# --------------------------------------------------------------------------- #
# 应用装配
# --------------------------------------------------------------------------- #
def _serve_viewer() -> HTMLResponse:
    html = VIEWER_DIR / "index.html"
    if not html.exists():
        return JSONResponse(status_code=404, content={"error": "viewer/index.html 未找到"})
    # 禁用缓存：前端 HTML 改动后刷新页面立即生效
    return HTMLResponse(html.read_text(encoding="utf-8"), headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()
    get_store()  # 触发建表
    app = FastAPI(
        title="3D 骨骼动画迁移工具",
        description="前端 WebGL 渲染 + Python 算法后端（FBX/GLB → GLB），SQLite 持久化任务。",
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    # /v2 四阶段路由（素材库 + 作业）与 /v1 并存；导入失败时降级为仅 /v1
    try:
        from .server_v2 import router as v2_router
        app.include_router(v2_router)
    except ImportError as exc:
        logger.warning("/v2 路由未挂载（仅 /v1 可用）：%s", exc)

    @app.get("/", include_in_schema=False)
    async def root() -> HTMLResponse:
        return _serve_viewer()

    @app.get("/viewer", include_in_schema=False)
    async def viewer() -> HTMLResponse:
        return _serve_viewer()

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    import uvicorn
    app = create_app()
    print("=" * 64)
    print("🎯 3D 骨骼动画迁移工具 (retarget)")
    print("=" * 64)
    print(f"  页面:     http://{settings.server.host}:{settings.server.port}/")
    print(f"  API 文档: http://{settings.server.host}:{settings.server.port}/docs")
    print(f"  项目目录: {settings.project_dir}")
    print(f"  数据库:   {settings.sqlite_file}")
    print(f"  assimp:   {glb_io.resolve_assimp(settings) or '未找到（仅 FBX 输入需要）'}")
    print("=" * 64)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
