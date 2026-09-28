"""坐标轴归一化：把不同 DCC/引擎工具的轴系统一到全局规范坐标系。

规范约定（右手系）：``+Y`` 向上、``+Z`` 为角色朝向（front）、``+X`` 为角色左手侧。
导入时（``glb_io.normalize_to_glb``）自动探测模型的 up/forward 轴并符号定向，
在场景根插入一个纯旋转校正节点（不改顶点/不改动画通道），使下游所有阶段
（多视角渲染、三角化求解、对称化、映射、retarget、前端展示）都在同一坐标系工作。

探测策略：
- 有骨架：up = pelvis→head 吸附坐标轴；forward = foot→ball/toe 吸附轴；
  再用 ``*_l / *_r`` 命名对的左右位置校验 handedness，不一致则翻转 forward（防镜像）。
- 无骨架（纯网格）：up = bbox 最长轴（符号用「脚端双簇/头端单簇」间隙判定）；
  forward = 剩余两轴中最短轴（人物前后最薄），符号用头带前凸（鼻尖）判定；
  left 由 handedness ``cross(up, forward)`` 导出。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

AXES = np.eye(3)


# --------------------------------------------------------------------------- #
# 小工具：TRS / 四元数 / 矩阵
# --------------------------------------------------------------------------- #
def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n ** 0.5, y / n ** 0.5, z / n ** 0.5, w / n ** 0.5
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) or 1.0)


def _local_matrix(n) -> np.ndarray:
    m = getattr(n, "matrix", None)
    if m is not None and list(m) != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] and list(m):
        return np.array(m, dtype=np.float64).reshape(4, 4).T
    t = np.array(n.translation or [0, 0, 0], dtype=np.float64)
    r = np.array(n.rotation or [0, 0, 0, 1], dtype=np.float64)
    s = np.array(n.scale or [1, 1, 1], dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = _quat_to_mat(r) * s
    out[:3, 3] = t
    return out


def global_matrices(g) -> List[np.ndarray]:
    nodes = getattr(g, "nodes", []) or []
    mats: List[Optional[np.ndarray]] = [None] * len(nodes)
    parent = [None] * len(nodes)
    for i, n in enumerate(nodes):
        for c in (n.children or []):
            parent[c] = i

    def rec(i: int, p: np.ndarray) -> np.ndarray:
        if mats[i] is not None:
            return mats[i]
        m = p @ _local_matrix(nodes[i])
        mats[i] = m
        for c in (nodes[i].children or []):
            rec(c, m)
        return m

    for i in range(len(nodes)):
        if parent[i] is None:
            rec(i, np.eye(4))
    for i in range(len(nodes)):
        if mats[i] is None:
            mats[i] = _local_matrix(nodes[i])
    return mats


def _snap_axis(v: np.ndarray, forbid: Optional[int] = None) -> np.ndarray:
    """把方向吸附到最近的坐标轴（带符号单位向量）。"""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    v = v / n
    scores = [abs(v[i]) for i in range(3)]
    if forbid is not None:
        scores[forbid] = -1.0
    i = int(np.argmax(scores))
    out = np.zeros(3)
    out[i] = 1.0 if v[i] >= 0 else -1.0
    return out


def _find(names: List[str], *keys: str, exclude: tuple = ()) -> Optional[int]:
    for i, nm in enumerate(names):
        low = (nm or "").lower()
        if any(e in low for e in exclude):
            continue
        if all(k in low for k in keys):
            return i
    return None


# --------------------------------------------------------------------------- #
# 朝向探测
# --------------------------------------------------------------------------- #
def _mesh_points(g, mats: List[np.ndarray]) -> np.ndarray:
    pts: List[np.ndarray] = []
    from ai3d.retarget.glb_io import read_accessor
    for i, node in enumerate(g.nodes or []):
        if node.mesh is None:
            continue
        mesh = g.meshes[node.mesh]
        for prim in (mesh.primitives or []):
            if prim.attributes.POSITION is None:
                continue
            p = read_accessor(g, prim.attributes.POSITION).astype(np.float64)
            pts.append(p @ mats[i][:3, :3].T + mats[i][:3, 3])
    if not pts:
        return np.zeros((0, 3))
    return np.concatenate(pts, 0)


def _band_gap_score(p_band: np.ndarray, axis: int) -> float:
    """带内沿 axis 坐标的双簇间隙得分：脚端（两脚分开）> 头端（单簇）。"""
    if len(p_band) < 8:
        return -1.0
    v = p_band[:, axis]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return -1.0
    bins = np.clip(((v - lo) / (hi - lo) * 9).astype(int), 0, 8)
    occ = sorted(set(bins.tolist()))
    gap = max((b - a - 1 for a, b in zip(occ, occ[1:])), default=0)
    return float(gap)


def estimate_frame(g) -> Dict[str, Any]:
    """探测模型当前轴系，返回规范基向量（模型坐标表示）：left/up/forward + 诊断信息。"""
    mats = global_matrices(g)
    names = [n.name or f"node{i}" for i, n in enumerate(g.nodes or [])]
    notes: List[str] = []
    up = fwd = None
    method = "mesh"
    skin = (g.skins or [None])[0]
    if skin is not None and skin.joints:
        jw = {names[j]: mats[j][:3, 3] for j in skin.joints if j < len(names)}
        low = {k.lower(): k for k in jw}

        def jget(*keys: str, exclude: tuple = ()):
            for lk, orig in low.items():
                if any(e in lk for e in exclude):
                    continue
                if all(k in lk for k in keys):
                    return jw[orig]
            return None

        def jfirst(*candidates) -> Optional[np.ndarray]:
            for keys, exclude in candidates:
                v = jget(*keys, exclude=exclude)
                if v is not None:
                    return v
            return None

        pelvis = jfirst((("pelvis",), ()), (("hips",), ()),
                        (("spine_01",), ()), (("spine01",), ()))
        head = jget("head", exclude=("headtop",))
        if pelvis is not None and head is not None and np.linalg.norm(head - pelvis) > 1e-6:
            up = _snap_axis(head - pelvis)
            method = "skeleton"
        fl = jfirst((("foot_l",), ()), (("foot", "l"), ()))
        fr = jfirst((("foot_r",), ()), (("foot", "r"), ()))
        bl = jfirst((("ball_l",), ()), (("toe_l",), ()))
        br = jfirst((("ball_r",), ()), (("toe_r",), ()))
        vecs = []
        if fl is not None and bl is not None:
            vecs.append(bl - fl)
        if fr is not None and br is not None:
            vecs.append(br - fr)
        if vecs and up is not None:
            fv = np.mean(vecs, axis=0)
            fv = fv - up * np.dot(fv, up)
            fwd = _snap_axis(fv)
        # 左右命名对校验 handedness（防前后判反导致镜像）
        if up is not None and fwd is not None:
            for a, b in (("hand_l", "hand_r"), ("upperarm_l", "upperarm_r"),
                         ("thigh_l", "thigh_r"), ("foot_l", "foot_r")):
                pa, pb = jget(a), jget(b)
                if pa is not None and pb is not None and np.linalg.norm(pa - pb) > 1e-6:
                    lv = pa - pb
                    if np.dot(lv, np.cross(up, fwd)) < 0:
                        fwd = -fwd
                        notes.append(f"左右命名对 {a}/{b} 校验翻转了 forward 符号")
                    break
    if up is None or fwd is None:
        p = _mesh_points(g, mats)
        if len(p) < 16:
            return {"left": [1, 0, 0], "up": [0, 1, 0], "forward": [0, 0, 1],
                    "method": "identity", "notes": ["顶点不足，保持原轴系"]}
        ext = p.max(0) - p.min(0)
        up_a = int(np.argmax(ext))
        rest = [i for i in range(3) if i != up_a]
        fwd_a = rest[int(np.argmin(ext[rest]))]
        left_a = rest[int(np.argmax(ext[rest]))]
        # up 符号：脚端带双簇间隙 > 头端带
        span = (p.min(0), p.max(0))
        size = span[1][up_a] - span[0][up_a]
        lo_band = p[p[:, up_a] < span[0][up_a] + 0.15 * size]
        hi_band = p[p[:, up_a] > span[1][up_a] - 0.15 * size]
        gap_lo = _band_gap_score(lo_band, left_a)
        gap_hi = _band_gap_score(hi_band, left_a)
        up_sign = 1.0 if gap_lo >= gap_hi else -1.0
        if gap_lo < 0 and gap_hi < 0:
            notes.append("脚/头双簇判定失效，up 符号取正轴")
        up = np.zeros(3)
        up[up_a] = up_sign
        # forward 符号：脚端带前凸尾（脚趾长于脚跟）比头带鼻尖信号更强
        foot_band = lo_band if up_sign > 0 else hi_band
        fv = foot_band[:, fwd_a] if len(foot_band) >= 8 else p[:, fwd_a]
        mu = float(fv.mean())
        fwd_sign = 1.0 if (fv.max() - mu) >= (mu - fv.min()) else -1.0
        notes.append("无骨架：forward 符号由脚带前凸尾判定，如镜像请在审核视图核对")
        fwd = np.zeros(3)
        fwd[fwd_a] = fwd_sign
        method = "mesh"
    if np.linalg.norm(fwd) < 0.5:  # 骨架路径缺 toe/ball：退化为最薄轴
        p = _mesh_points(g, mats)
        ext = p.max(0) - p.min(0)
        ua = int(np.argmax(np.abs(up)))
        rest = [i for i in range(3) if i != ua]
        fa = rest[int(np.argmin(ext[rest]))]
        fwd = np.zeros(3)
        fwd[fa] = 1.0
        notes.append("缺少 toe/ball 骨：forward 取最薄轴，符号默认正")
    left = np.cross(up, fwd)
    return {"left": left.tolist(), "up": up.tolist(), "forward": fwd.tolist(),
            "method": method, "notes": notes}


def frame_rotation(frame: Dict[str, Any]) -> np.ndarray:
    """模型坐标 → 规范坐标 的旋转矩阵：行 = [left, up, forward]。"""
    l = np.array(frame["left"], float)
    u = np.array(frame["up"], float)
    f = np.array(frame["forward"], float)
    return np.stack([l, u, f])


# --------------------------------------------------------------------------- #
# 应用：场景根插入纯旋转校正节点
# --------------------------------------------------------------------------- #
def apply_canonical(g, R: np.ndarray) -> bool:
    if np.allclose(R, np.eye(3), atol=1e-6):
        return False
    from pygltflib import Node
    scene_idx = g.scene or 0
    scene = g.scenes[scene_idx]
    roots = list(scene.nodes or [])
    if not roots:
        return False
    q = _mat_to_quat(R)
    node = Node(name="canon_axis_root", rotation=[float(v) for v in q], children=roots)
    g.nodes.append(node)
    new_idx = len(g.nodes) - 1
    scene.nodes = [new_idx]
    return True


def normalize_axes(g) -> Dict[str, Any]:
    """探测 + 应用规范轴系；返回诊断信息（是否改动、旋转矩阵、方法）。"""
    frame = estimate_frame(g)
    R = frame_rotation(frame)
    changed = apply_canonical(g, R)
    return {"frame": frame, "rotation": R.tolist(), "changed": changed,
            "canonical": {"up": "+Y", "forward": "+Z", "left": "+X"}}


def normalize_axes_file(path) -> Dict[str, Any]:
    """对已落盘的 GLB 文件做轴系归一（原地保存），返回诊断信息。"""
    from ai3d.retarget.glb_io import _load, save_glb
    g = _load(path)
    info = normalize_axes(g)
    if info["changed"]:
        save_glb(g, path)
    fr = info["frame"]
    logger.info("轴系归一 %s：%s up=%s fwd=%s left=%s changed=%s %s",
                Path(path).name,
                fr["method"], fr["up"], fr["forward"], fr["left"],
                info["changed"], ";".join(fr["notes"]) or "")
    return info
