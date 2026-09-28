"""多视角 2D 关键点 → 三维关节求解（P2）。

流程：按 view_spec 重建正交相机 → 置信度加权射线最小二乘三角化 → 关键点映射到
22 语义关节（含派生：骨盆/胸/脊柱/颈/头/锁骨/脚趾）→ 左右对称 + 比例先验融合 +
骨长稳定 → Rig（含每关节置信度，供 P3 门控）。

相机模型与前端 ``viewer/index.html`` 的 ``captureViews`` 严格一致：
    half = maxDim*0.62, dist = maxDim*3,
    OrthographicCamera(-half, half, half, -half, ...), up=+Y,
    pos = center + dir*dist, dir = [sinθcosφ, sinφ, cosθcosφ], lookAt(center)。
正交投影下每个视角给出 2 个线性方程，多视角堆叠成超定方程组用最小二乘求解。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .skeleton import (
    JOINTS,
    MIRROR,
    PARENTS,
    Rig,
    generate_rig_from_bbox,
)
from .settings import SolveConfig

logger = logging.getLogger(__name__)

# 语义关节 ← 直接检测的关键点（COCO17 / DWPose 前 17 点同名）
_DIRECT: Dict[str, str] = {
    "upper_arm_l": "left_shoulder", "lower_arm_l": "left_elbow", "hand_l": "left_wrist",
    "upper_arm_r": "right_shoulder", "lower_arm_r": "right_elbow", "hand_r": "right_wrist",
    "upper_leg_l": "left_hip", "lower_leg_l": "left_knee", "foot_l": "left_ankle",
    "upper_leg_r": "right_hip", "lower_leg_r": "right_knee", "foot_r": "right_ankle",
}
_TOE: Dict[str, str] = {"toe_l": "left_big_toe", "toe_r": "right_big_toe"}
_EAR = {"left_ear", "right_ear"}


# --------------------------------------------------------------------------- #
# 正交相机（与前端一致）
# --------------------------------------------------------------------------- #
def cam_basis(az_deg: float, el_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    th, ph = np.radians(az_deg), np.radians(el_deg)
    z_cam = np.array([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)])
    up = np.array([0.0, 1.0, 0.0])
    x_cam = np.cross(up, z_cam)
    nx = np.linalg.norm(x_cam)
    x_cam = x_cam / nx if nx > 1e-6 else np.array([1.0, 0.0, 0.0])
    y_cam = np.cross(z_cam, x_cam)
    return x_cam, y_cam, z_cam


def build_camera(az_deg: float, el_deg: float, center: np.ndarray, max_dim: float,
                 width: int, height: int) -> Dict[str, object]:
    half = float(max_dim) * 0.62
    dist = float(max_dim) * 3.0
    x_cam, y_cam, z_cam = cam_basis(az_deg, el_deg)
    pos = np.asarray(center, np.float64) + z_cam * dist
    return {"x": x_cam, "y": y_cam, "z": z_cam, "pos": pos,
            "half": half, "W": float(width), "H": float(height)}


def project(X: np.ndarray, cam: Dict) -> np.ndarray:
    """世界点 → 像素 (u, v)（前端渲染的逆变换，用于合成测试与自检）。"""
    v = np.asarray(X, np.float64) - cam["pos"]
    xc = float(v @ cam["x"])
    yc = float(v @ cam["y"])
    u = (xc / cam["half"] + 1.0) * 0.5 * cam["W"]
    vv = (1.0 - yc / cam["half"]) * 0.5 * cam["H"]
    return np.array([u, vv])


def triangulate(obs: List[np.ndarray], cams: List[Dict],
                weights: Optional[List[float]] = None) -> np.ndarray:
    """置信度加权最小二乘三角化：每视角 2 个正交投影方程。"""
    m = len(obs)
    if m == 0:
        return np.zeros(3)
    w = np.ones(m) if weights is None else np.asarray(weights, np.float64)
    A = np.zeros((2 * m, 3))
    b = np.zeros(2 * m)
    ww = np.zeros(2 * m)
    for i, (uv, cam) in enumerate(zip(obs, cams)):
        ndc_x = 2.0 * float(uv[0]) / cam["W"] - 1.0
        ndc_y = 1.0 - 2.0 * float(uv[1]) / cam["H"]
        xc, yc = ndc_x * cam["half"], ndc_y * cam["half"]
        A[2 * i] = cam["x"]
        b[2 * i] = xc + float(cam["pos"] @ cam["x"])
        A[2 * i + 1] = cam["y"]
        b[2 * i + 1] = yc + float(cam["pos"] @ cam["y"])
        ww[2 * i] = ww[2 * i + 1] = w[i]
    Aw = A * ww[:, None]
    bw = b * ww
    X, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    return X


# --------------------------------------------------------------------------- #
# 关键点三角化
# --------------------------------------------------------------------------- #
def _triangulate_keypoints(detections: Dict[str, Dict[str, Dict]], cams: Dict[str, Dict],
                           cfg: SolveConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """对每个关键点名跨视角三角化 → {name: (X3, conf)}。"""
    names = set()
    for det in detections.values():
        names.update(det.keys())
    pts: Dict[str, np.ndarray] = {}
    confs: Dict[str, float] = {}
    for name in names:
        obs, camlist, wts = [], [], []
        for vid, det in detections.items():
            kp = det.get(name)
            cam = cams.get(vid)
            if kp is None or cam is None:
                continue
            c = float(kp.get("conf", 0.0))
            if c < cfg.min_confidence:
                continue
            obs.append(np.asarray(kp["uv"], np.float64))
            camlist.append(cam)
            wts.append(c)
        if len(obs) < 2:
            continue
        X = triangulate(obs, camlist, wts)
        if not np.all(np.isfinite(X)):
            continue
        coverage = min(1.0, len(obs) / 3.0)
        pts[name] = X
        confs[name] = float(np.mean(wts)) * coverage
    return pts, confs


def _mid(pts, confs, a: str, b: str) -> Tuple[Optional[np.ndarray], float]:
    if a in pts and b in pts:
        return (pts[a] + pts[b]) * 0.5, min(confs[a], confs[b])
    if a in pts:
        return pts[a], confs[a] * 0.6
    if b in pts:
        return pts[b], confs[b] * 0.6
    return None, 0.0


# --------------------------------------------------------------------------- #
# 主求解
# --------------------------------------------------------------------------- #
def solve_rig(detections: Dict[str, Dict[str, Dict]],
              cameras: Dict[str, Tuple[float, float]],
              bbox_min, bbox_max,
              cfg: Optional[SolveConfig] = None,
              width: int = 512, height: int = 512) -> Rig:
    """多视角检测 → 22 语义关节 Rig（含置信度）。

    detections: {view_id: {kp_name: {"uv":[u,v],"conf":c}}}
    cameras:    {view_id: (azimuth_deg, elevation_deg)}
    bbox_min/max: 目标网格包围盒（与前端 Box3 一致，用于重建相机与比例先验）。
    """
    cfg = cfg or SolveConfig()
    mn = np.asarray(bbox_min, np.float64)
    mx = np.asarray(bbox_max, np.float64)
    center = (mn + mx) * 0.5
    size = mx - mn
    max_dim = float(max(size[0], size[1], size[2], 1e-6))
    body_height = float(size[1]) or 1.0

    cams = {vid: build_camera(az, el, center, max_dim, width, height)
            for vid, (az, el) in cameras.items()}
    pts, confs = _triangulate_keypoints(detections, cams, cfg)

    # 比例先验（无检测时的回退，也作为软约束锚点）
    prior = generate_rig_from_bbox(mn.tolist(), mx.tolist(), confidence=0.9)

    heads: Dict[str, np.ndarray] = {}
    conf: Dict[str, float] = {}
    src: Dict[str, str] = {}

    def put(j: str, X: Optional[np.ndarray], c: float) -> None:
        if X is not None and c >= cfg.min_confidence and np.all(np.isfinite(X)):
            heads[j] = np.asarray(X, np.float64)
            conf[j] = float(c)
            src[j] = "solved"
        else:
            heads[j] = prior.heads[j].copy()
            conf[j] = 0.2  # 低置信：标记为比例回退，供门控审核
            src[j] = "proportion"

    # 直接检测关节
    for j, kp in _DIRECT.items():
        put(j, pts.get(kp), confs.get(kp, 0.0))
    # 脚趾（DWPose 有 big_toe；YOLO 无则回退比例）
    for j, kp in _TOE.items():
        put(j, pts.get(kp), confs.get(kp, 0.0))

    # 派生关节
    pelvis, c_pelvis = _mid(pts, confs, "left_hip", "right_hip")
    put("pelvis", pelvis, c_pelvis)
    chest, c_chest = _mid(pts, confs, "left_shoulder", "right_shoulder")
    put("chest", chest, c_chest)

    ear_mid, c_ear = _mid(pts, confs, "left_ear", "right_ear")
    nose = pts.get("nose")
    if ear_mid is not None:
        head_ref, c_head = ear_mid, c_ear
    elif nose is not None:
        head_ref, c_head = nose, confs.get("nose", 0.0) * 0.8
    else:
        head_ref, c_head = None, 0.0
    put("head", head_ref, c_head)

    # 颈/脊柱/锁骨：在已确定的解剖锚点间按比例内插
    P = heads  # 便于读取（此时 pelvis/chest/head 已就位）
    if "pelvis" in P and "chest" in P:
        d = P["chest"] - P["pelvis"]
        base_c = min(conf.get("pelvis", 0), conf.get("chest", 0))
        for name, t in (("spine_01", 1.0 / 3.0), ("spine_02", 2.0 / 3.0)):
            put(name, P["pelvis"] + d * t, base_c * 0.9)
    if "chest" in P and "head" in P:
        d = P["head"] - P["chest"]
        base_c = min(conf.get("chest", 0), conf.get("head", 0))
        put("neck", P["chest"] + d * 0.556, base_c * 0.9)
    if "chest" in P and "upper_arm_l" in P:
        put("clavicle_l", P["chest"] + (P["upper_arm_l"] - P["chest"]) * 0.4,
            min(conf.get("chest", 0), conf.get("upper_arm_l", 0)) * 0.85)
    if "chest" in P and "upper_arm_r" in P:
        put("clavicle_r", P["chest"] + (P["upper_arm_r"] - P["chest"]) * 0.4,
            min(conf.get("chest", 0), conf.get("upper_arm_r", 0)) * 0.85)

    # 补齐任何仍缺失的关节（比例回退）
    for j in JOINTS:
        if j not in heads:
            put(j, None, 0.0)

    # 左右对称软约束（仅对 solved 的镜像对）；左右轴自动识别（镜像对差异最大的轴），
    # 不能硬编码 X：有的模型左右在 Z（脚趾朝 +X）
    axis = _lateral_axis(heads)
    heads, conf = _symmetrize(heads, conf, float(center[axis]), cfg.symmetry_weight, axis)
    # 骨长稳定：按层级把骨长轻拉向比例先验
    heads = _stabilize_bones(heads, prior, cfg.bone_length_stability)
    # 人体比例软约束：把身高归一到包围盒高度（三角化已在世界尺度，通常无需再缩放，
    # 但对整体偏移做对齐——脚底落到 bbox 底）
    heads = _align_to_bbox(heads, prior, cfg.proportion_weight)

    return Rig(heads={j: np.asarray(heads[j], np.float64) for j in JOINTS},
               height=body_height, revision=0,
               confidence={j: float(conf.get(j, 0.0)) for j in JOINTS},
               source=src)


def _lateral_axis(heads: Dict[str, np.ndarray]) -> int:
    """取镜像关节对坐标差异最大的轴作为左右轴（0=X,1=Y,2=Z）。"""
    diffs = np.zeros(3)
    n = 0
    for j, m in MIRROR.items():
        if m == j or j not in heads or m not in heads:
            continue
        diffs += np.abs(np.asarray(heads[j]) - np.asarray(heads[m]))
        n += 1
    if n == 0:
        return 0
    return int(np.argmax(diffs))


def _symmetrize(heads: Dict[str, np.ndarray], conf: Dict[str, float],
                center_lat: float, weight: float, axis: int) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """对镜像关节对绕矢状面做置信度加权对称化（axis 为左右轴）。"""
    if weight <= 0:
        return heads, conf

    def mirror(p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, np.float64).copy()
        q[axis] = 2 * center_lat - q[axis]
        return q

    seen: set = set()
    for j in JOINTS:
        m = MIRROR.get(j)
        if m == j or m not in heads or j in seen:
            continue
        seen.add(j)
        seen.add(m)
        a, b = heads[j], heads[m]
        ca, cb = conf.get(j, 0.0), conf.get(m, 0.0)
        if ca < 1e-6 and cb < 1e-6:
            continue
        tot = ca + cb
        wa, wb = ca / tot, cb / tot
        blended = a * wa + mirror(b) * wb
        heads[j] = a * (1 - weight) + blended * weight
        blended_m = b * wb + mirror(a) * wa
        heads[m] = b * (1 - weight) + blended_m * weight
        avg = (ca + cb) * 0.5
        conf[j] = conf.get(j, 0.0) * (1 - weight) + avg * weight
        conf[m] = conf.get(m, 0.0) * (1 - weight) + avg * weight
    return heads, conf


def _stabilize_bones(heads: Dict[str, np.ndarray], prior: Rig, weight: float) -> Dict[str, np.ndarray]:
    """按层级顺序把每根骨长轻拉向比例先验骨长（JOINTS 已保证父在子前）。"""
    if weight <= 0:
        return heads
    w = float(weight) * 0.5  # 温和
    for j in JOINTS:
        p = PARENTS[j]
        if p is None or p not in heads or j not in heads:
            continue
        d = heads[j] - heads[p]
        L = float(np.linalg.norm(d))
        pd = prior.heads[j] - prior.heads[p]
        pL = float(np.linalg.norm(pd))
        if L < 1e-6:
            continue
        target_L = (1 - w) * L + w * pL
        heads[j] = heads[p] + d / L * target_L
    return heads


def _align_to_bbox(heads: Dict[str, np.ndarray], prior: Rig, weight: float) -> Dict[str, np.ndarray]:
    """把求解骨架的脚底对齐到包围盒底（消除整体竖直偏移），按 weight 混合。"""
    if weight <= 0:
        return heads
    feet = [heads[j] for j in ("foot_l", "foot_r") if j in heads]
    if not feet:
        return heads
    solved_min_y = min(f[1] for f in feet)
    prior_min_y = min(prior.heads[j][1] for j in ("foot_l", "foot_r"))
    dy = (prior_min_y - solved_min_y) * float(weight)
    if abs(dy) > 1e-9:
        shift = np.array([0.0, dy, 0.0])
        for j in JOINTS:
            heads[j] = heads[j] + shift
    return heads
