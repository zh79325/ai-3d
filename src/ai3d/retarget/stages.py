"""S2~S4 的阶段纯函数：只认产物路径，不碰 store / DB / HTTP。

``/v1`` 的 :class:`worker.Worker` 与 ``/v2`` 的 :class:`job_worker.JobWorker` 共用这里
的算法编排。两套流程只在「产物落哪个目录、状态写哪张表」上分道；算法调用顺序与参数
必须一致，否则同一个模型从两个入口进来会得到不同的骨架与蒙皮权重。

文件名常量沿用既有约定（``rig.json`` / ``skin.npz`` / ``skin_report.json`` /
``detections.json`` / ``view_spec.json`` / ``mapping.json`` / ``anim.npz`` /
``retarget_meta.json``），:mod:`asset_store`、:mod:`job_store` 与 :mod:`task_store`
三侧同名，便于人工比对产物。

约定：本模块函数**只抛 RuntimeError / ValueError**（算法层面的失败），不定义领域异常；
调用方按自己的状态机把它们翻成 ``FAILED`` 与 HTTP 状态码。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import foot_ik, glb_io, inference, solve
from .mapping import build_mapping
from .retarget import (
    _global_rest_positions,
    retarget_animation,
    stack_rotations,
    with_fps,
)
from .schemas import DEFAULT_CAMERAS
from .settings import RetargetSettings, get_settings
from .skeleton import JOINTS, Rig, generate_rig_from_bbox
from .skinning import compute_skin_weights

logger = logging.getLogger(__name__)

DETECTIONS_FILE = "detections.json"
VIEW_SPEC_FILE = "view_spec.json"
RIG_FILE = "rig.json"
SKIN_FILE = "skin.npz"
SKIN_REPORT_FILE = "skin_report.json"
MAPPING_FILE = "mapping.json"
ANIM_FILE = "anim.npz"
RETARGET_META_FILE = "retarget_meta.json"
RESULT_FILE = "result.glb"
REPORT_FILE = "report.json"

DEFAULT_VIEW_SIZE = 512


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> Optional[Any]:
    """读 JSON 产物；文件不存在或内容损坏一律返回 ``None``（不抛，交由调用方判定）。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("产物 %s 无法解析（%s），按缺失处理", p.name, exc)
        return None


def write_json(path: Path, payload: Any, indent: Optional[int] = 2) -> Path:
    """写 JSON 产物（``indent=None`` 即紧凑单行，detections.json 这种大字典用它）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return p


def load_mesh(target_glb: Path) -> Dict[str, Any]:
    """加载并合并网格（按节点全局矩阵变换，故顶点已是**米制**，S1 的 canon 缩放在内）。"""
    g = glb_io._load(Path(target_glb))
    mesh = glb_io.merge_mesh(g)
    if len(mesh["positions"]) == 0:
        raise RuntimeError("目标网格无顶点，无法蒙皮")
    mesh["gltf"] = g
    return mesh


def cameras_from_view_spec(
        view_spec: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
    """从 view_spec 还原三角化用的相机参数；缺失时回落 :data:`DEFAULT_CAMERAS` + 512²。

    相机参数必须与前端离屏渲染时**完全一致**（正交、绕 up 轴的方位角 + 仰角、画布尺寸），
    否则射线三角化出来的三维关节会整体错位——这是 S2 最隐蔽的一类 bug，故 view_spec
    在下发时就持久化到产物目录，求解时读同一份而不是各写一套默认值。
    """
    if not view_spec:
        return ({vid: (float(az), float(el)) for (vid, az, el) in DEFAULT_CAMERAS},
                DEFAULT_VIEW_SIZE, DEFAULT_VIEW_SIZE)
    width = int(view_spec.get("width") or DEFAULT_VIEW_SIZE)
    height = int(view_spec.get("height") or DEFAULT_VIEW_SIZE)
    cameras = {str(c["view_id"]): (float(c["azimuth"]), float(c["elevation"]))
               for c in (view_spec.get("cameras") or [])}
    if not cameras:
        cameras = {vid: (float(az), float(el)) for (vid, az, el) in DEFAULT_CAMERAS}
    return cameras, width, height


# --------------------------------------------------------------------------- #
# S2：POSE_INFER → SOLVE_RIG → BUILD_RIG
# --------------------------------------------------------------------------- #
def infer_views(views_dir: Path, detections_path: Path,
                settings: Optional[RetargetSettings] = None) -> Dict[str, Dict[str, Dict]]:
    """多视角图逐张推理 → ``{view_id: {kp_name: {"uv": [u, v], "conf": c}}}``。

    写 ``detections.json`` 并返回。视角图为空/读不出来直接抛错：三角化缺输入时
    继续往下走只会得到一份全是比例回退的骨架，白白浪费一次蒙皮。
    """
    settings = settings or get_settings()
    estimator = inference.load_estimator(settings)
    detections = inference.infer_views(Path(views_dir), estimator=estimator)
    if not detections:
        raise RuntimeError("视角图为空或无法读取，无法推理")
    write_json(Path(detections_path), detections, indent=None)
    n_kp = sum(len(d) for d in detections.values())
    logger.info("多视角推理完成：%d 个视角 / %d 个 2D 关键点", len(detections), n_kp)
    return detections


def solve_rig(target_glb: Path, detections_path: Path, rig_path: Path,
              view_spec_path: Optional[Path] = None,
              settings: Optional[RetargetSettings] = None) -> Rig:
    """多视角加权三角化 + 对称/比例/骨长约束 → 22 语义关节，写 ``rig.json``。

    ``target_glb`` 只用来取包围盒（重建正交相机与比例先验），不读蒙皮。
    """
    settings = settings or get_settings()
    detections = read_json(Path(detections_path))
    if not detections:
        raise RuntimeError(f"缺少 {DETECTIONS_FILE}，无法求解三维关节")
    view_spec = read_json(Path(view_spec_path)) if view_spec_path else None
    cameras, width, height = cameras_from_view_spec(view_spec)
    bbox = glb_io.mesh_bbox(glb_io._load(Path(target_glb)))
    rig = solve.solve_rig(detections, cameras, bbox["min"], bbox["max"],
                          settings.solve, width, height)
    write_json(Path(rig_path), rig.to_dict())
    solved = sum(1 for j in JOINTS if rig.source.get(j) == "solved")
    logger.info("三角化求解 %d/%d 关节（overall=%.2f）", solved, len(JOINTS),
                rig.overall_confidence())
    return rig


def compute_skin(mesh: Dict[str, Any], rig: Rig, out_dir: Path,
                 settings: Optional[RetargetSettings] = None) -> Tuple[int, Dict[str, Any]]:
    """自动蒙皮（libigl BBW，按连通分量 + 焊接求解域）→ ``skin.npz`` + ``skin_report.json``。

    返回 ``(顶点数, 蒙皮报告)``。**不写 rig.json**：人工微调过的骨架不该被蒙皮步骤覆盖。
    """
    settings = settings or get_settings()
    out = Path(out_dir)
    joints_u16, weights_f32, report = compute_skin_weights(
        mesh["positions"], mesh["indices"], rig, settings.skin)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / SKIN_FILE, joints=joints_u16, weights=weights_f32)
    write_json(out / SKIN_REPORT_FILE, report)
    return len(joints_u16), report


def build_rig(target_glb: Path, out_dir: Path, rig: Optional[Rig] = None,
              settings: Optional[RetargetSettings] = None) -> Tuple[Rig, int, Dict[str, Any]]:
    """骨架 + 蒙皮：``rig`` 为空时按网格包围盒的人体比例生成（无 AI 的确定性路径）。

    写 ``rig.json`` / ``skin.npz`` / ``skin_report.json``，返回 ``(rig, 顶点数, 报告)``。
    """
    mesh = load_mesh(target_glb)
    if rig is None:
        p = mesh["positions"]
        rig = generate_rig_from_bbox(p.min(0).tolist(), p.max(0).tolist(), confidence=0.9)
    write_json(Path(out_dir) / RIG_FILE, rig.to_dict())
    n_verts, report = compute_skin(mesh, rig, out_dir, settings)
    return rig, n_verts, report


def reskin(target_glb: Path, out_dir: Path,
           settings: Optional[RetargetSettings] = None) -> Tuple[Rig, int, Dict[str, Any]]:
    """人工微调关节后**只重算蒙皮**：读 ``out_dir/rig.json``，不改它（保留 revision）。

    关节一动，BBW 的 handle 与权重全都失效，必须重算；但骨架本身是人工确认过的，
    重写会把 ``revision`` 与 ``source="manual"`` 标记冲掉。
    """
    rig_dict = read_json(Path(out_dir) / RIG_FILE)
    if rig_dict is None:
        raise RuntimeError(f"缺少 {RIG_FILE}，无法重算蒙皮")
    rig = Rig.from_dict(rig_dict)
    mesh = load_mesh(target_glb)
    n_verts, report = compute_skin(mesh, rig, out_dir, settings)
    return rig, n_verts, report


def skin_summary(report: Dict[str, Any]) -> str:
    """把蒙皮报告压成一行日志/阶段消息（zero_rows 与 strain.max 是最要紧的两个数）。"""
    strain = report.get("strain") or {}
    comp = report.get("components") or {}
    return (f"zero={report.get('zero_rows')}, strain_max={strain.get('max')}, "
            f"fallback={comp.get('fallback')}, proxy={comp.get('proxy')}")


def list_view_images(views_dir: Path) -> List[Path]:
    """已回传的视角图（按文件名排序，主干即 view_id）。"""
    return inference.list_view_images(Path(views_dir))


# --------------------------------------------------------------------------- #
# S3：MAP_SOURCE → RETARGET
# --------------------------------------------------------------------------- #
def source_joints(anim_glb: Path) -> List[Dict[str, Any]]:
    """从动画素材的 GLB 里提取源骨架关节（节点索引 / 名字 / 父节点 / rest 位置）。

    ``position`` 是**原始单位**下的全局 rest 坐标：:func:`retarget._global_rest_positions`
    刻意不乘 canon 缩放（``delta = t_anim - t_rest`` 取自动画通道的局部 translation，
    永远是原始单位）。映射的几何补全只在同一素材内部比距离，单位统一即可。
    """
    g = glb_io._load(Path(anim_glb))
    skin = glb_io.extract_skin(g)
    if skin is None:
        raise RuntimeError("动画素材不含骨架(skin)，无法映射")
    nodes = glb_io.extract_nodes(g)
    parent_of: Dict[int, Optional[int]] = {n["index"]: None for n in nodes}
    for n in nodes:
        for c in n.get("children") or []:
            parent_of[c] = n["index"]
    positions = _global_rest_positions(skin, nodes)
    joints: List[Dict[str, Any]] = []
    for ni, nm in zip(skin["joints"], skin["names"]):
        pj = positions.get(ni)
        joints.append({"node": ni, "name": nm, "parent": parent_of.get(ni),
                       "position": pj.tolist() if pj is not None else None})
    return joints


def map_source(anim_glb: Path, mapping_path: Path,
               overrides: Optional[Dict[str, str]] = None,
               settings: Optional[RetargetSettings] = None) -> Dict[str, Any]:
    """源骨 → 22 语义关节映射（别名库 + 层级/几何补全），写 ``mapping.json``。

    ``overrides`` 是人工指定的 ``{semantic_id: 源骨名}``，优先级最高且不受
    ``min_match_score`` 门限约束。重跑时把上一版 mapping 的 overrides 带进来，
    人工校正才不会被自动匹配冲掉。

    一个关节都没匹配上时仍然**先写盘再抛错**：前端需要 mapping.json 里的源骨清单
    才能让人工指定映射，不落盘就把唯一的自救途径堵死了。
    """
    settings = settings or get_settings()
    joints = source_joints(anim_glb)
    mapping = build_mapping(joints, overrides=overrides, cfg=settings.mapping)
    write_json(Path(mapping_path), mapping)
    logger.info("源骨映射 %d/%d（coverage=%s）", mapping["matched"], mapping["total"],
                mapping["coverage"])
    if mapping["matched"] == 0:
        raise RuntimeError("源骨架无任何关节匹配到语义骨架（检查命名或人工指定映射）")
    return mapping


def retarget(anim_glb: Path, mapping: Dict[str, Any], rig: Rig, out_dir: Path,
             fps: int = 0, foot_lock: bool = True,
             settings: Optional[RetargetSettings] = None) -> Tuple[Dict[str, Any], str]:
    """重定向源动画到目标语义骨架并逐帧烘焙，写 ``anim.npz`` + ``retarget_meta.json``。

    返回 ``(meta, 一行进度描述)``。脚部 IK 锁定只在**作业与全局配置都启用**时生效：
    ``foot_contact_height=0.12`` 是米制绝对阈值，S1 未矫正单位时它会整体失效。
    """
    settings = settings or get_settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = retarget_animation(Path(anim_glb), mapping, rig,
                                with_fps(settings.retarget, fps))
    ik_msg = ""
    if foot_lock and settings.retarget.foot_lock:
        result["rotations"], ik_info = foot_ik.apply_foot_lock(
            rig, result["rotations"], result["root_translations"],
            result["times"], settings.retarget)
        result["meta"]["foot_ik"] = ik_info
        if ik_info.get("applied"):
            ik_msg = (f", 脚部锁定 L={ik_info['windows']['l']}"
                      f"/R={ik_info['windows']['r']} 窗口")
    meta = result["meta"]
    write_json(out / RETARGET_META_FILE, meta)
    np.savez(out / ANIM_FILE, times=result["times"],
             rotations=stack_rotations(result["rotations"], meta["frames"]),
             root_translations=result["root_translations"])
    message = (f"烘焙 {meta['frames']} 帧（duration={meta['duration']}s, "
               f"height_scale={meta['height_scale']}{ik_msg}）")
    logger.info("重定向完成：%s", message)
    return meta, message


def mapping_summary(mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """mapping.json 的紧凑摘要（列表/详情接口用，不带 22 行 items）。"""
    if not mapping:
        return {}
    methods: Dict[str, int] = {}
    for item in mapping.get("items") or []:
        if item.get("source_node") is not None:
            methods[item["method"]] = methods.get(item["method"], 0) + 1
    return {"matched": mapping.get("matched"), "total": mapping.get("total"),
            "coverage": mapping.get("coverage"), "revision": mapping.get("revision", 0),
            "methods": methods,
            "overrides": dict(mapping.get("overrides") or {})}
