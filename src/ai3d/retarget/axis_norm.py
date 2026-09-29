"""坐标轴与单位归一化：把不同 DCC/引擎工具的轴系、尺度统一到全局规范系。

规范约定（右手系）：``+Y`` 向上、``+Z`` 为角色朝向（front）、``+X`` 为角色左手侧；
尺度为米（``skin.weld_epsilon=2e-3``、``retarget.foot_contact_height=0.12`` 等米制
绝对阈值均依赖此前提）。

两套入口：
- :func:`normalize_axes_file`（旧，``/v1`` 依赖）：自动探测 up/forward 并**吸附到世界
  坐标轴**，在场景根插入纯旋转节点 ``canon_axis_root``，不改顶点与动画通道。
- :func:`align_asset`（S1 导入矫正）：改用最小体积 OBB（:mod:`obb`）给出的紧凑三轴
  框架 + 面编号映射，除旋转外还做**单位缩放**（同一 canon 节点带 scale），产出
  ``align.json`` 供前端立方体可视化与人工修正。

探测策略（两套共用 :func:`estimate_directions`）：
- 有骨架：up = pelvis→head；forward = foot→ball/toe；再用 ``*_l / *_r`` 命名对的左右
  位置校验 handedness，不一致则翻转 forward（防镜像）。
- 无骨架（纯网格）：up = bbox 最长轴（符号用「脚端双簇/头端单簇」间隙判定）；
  forward = 剩余两轴中最短轴（人物前后最薄），符号用头带前凸（鼻尖）判定；
  left 由 handedness ``cross(up, forward)`` 导出。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai3d.retarget.obb import FACE_AXES, FACE_LABELS, OBB, assign_faces, min_volume_obb

logger = logging.getLogger(__name__)

AXES = np.eye(3)
ALIGN_VERSION = 1               # align.json 结构版本，前端按它兼容读取

# 候选单位 → 换算到米的系数；推断时取「换算后身高最接近成人中值」者，
# 比固定数量级区间鲁棒（inch 与 cm 的区间在 50~120 会重叠）
_UNIT_SCALES: Tuple[Tuple[str, float], ...] = (
    ("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("inch", 0.0254), ("ft", 0.3048))
_REF_HEIGHT = 1.75           # 成人身高中值(m)，单位推断的参考目标
HUMAN_HEIGHT_RANGE = (1.4, 2.1)   # 越界则给人工提示（归一化资产/非人形/儿童）
CANON_NODE = "canon_axis_root"    # 场景根校正节点名（幂等更新靠它识别）


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


def _skeleton_dirs(g, mats: List[np.ndarray],
                   names: List[str]) -> Optional[Dict[str, Any]]:
    """骨架路径：返回**未吸附**的 up / 脚掌 / 左右参考向量；无骨架时 None。

    ``up_raw`` = pelvis→head；``foot_vec`` = mean(ball/toe − foot)；``left_vec`` = 首个
    可用的 ``*_l − *_r`` 命名对（handedness 校验用，防前后判反导致镜像）。
    不做任何坐标轴吸附，吸附与否由调用方决定。
    """
    skin = (g.skins or [None])[0]
    if skin is None or not skin.joints:
        return None
    jw = {names[j]: mats[j][:3, 3] for j in skin.joints if j < len(names)}
    low = {k.lower(): k for k in jw}

    def jget(*keys: str, exclude: tuple = ()) -> Optional[np.ndarray]:
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
    up_raw = None
    if pelvis is not None and head is not None:
        d = head - pelvis
        if float(np.linalg.norm(d)) > 1e-6:
            up_raw = d
    fl = jfirst((("foot_l",), ()), (("foot", "l"), ()))
    fr = jfirst((("foot_r",), ()), (("foot", "r"), ()))
    bl = jfirst((("ball_l",), ()), (("toe_l",), ()))
    br = jfirst((("ball_r",), ()), (("toe_r",), ()))
    vecs = []
    if fl is not None and bl is not None:
        vecs.append(bl - fl)
    if fr is not None and br is not None:
        vecs.append(br - fr)
    foot_vec = np.mean(vecs, axis=0) if vecs else None
    left_vec, pair = None, None
    for a, b in (("hand_l", "hand_r"), ("upperarm_l", "upperarm_r"),
                 ("thigh_l", "thigh_r"), ("foot_l", "foot_r")):
        pa, pb = jget(a), jget(b)
        if pa is not None and pb is not None and float(np.linalg.norm(pa - pb)) > 1e-6:
            left_vec, pair = pa - pb, f"{a}/{b}"
            break
    if up_raw is None and foot_vec is None:
        return None
    return {"up_raw": up_raw, "foot_vec": foot_vec,
            "left_vec": left_vec, "pair": pair}


def _mesh_dirs(g, mats: List[np.ndarray],
               notes: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """纯网格路径：up = bbox 最长轴（脚端双簇间隙定符号）、forward = 最薄轴。

    forward 符号用脚带前凸尾（脚趾长于脚跟）判定；顶点不足时返回 (None, None)，
    由调用方退化为 identity。
    """
    p = _mesh_points(g, mats)
    if len(p) < 16:
        notes.append("顶点不足，保持原轴系")
        return None, None
    ext = p.max(0) - p.min(0)
    up_a = int(np.argmax(ext))
    rest = [i for i in range(3) if i != up_a]
    fwd_a = rest[int(np.argmin(ext[rest]))]
    left_a = rest[int(np.argmax(ext[rest]))]
    # up 符号：脚端带双簇间隙 > 头端带
    lo, hi = p.min(0), p.max(0)
    size = hi[up_a] - lo[up_a]
    lo_band = p[p[:, up_a] < lo[up_a] + 0.15 * size]
    hi_band = p[p[:, up_a] > hi[up_a] - 0.15 * size]
    gap_lo = _band_gap_score(lo_band, left_a)
    gap_hi = _band_gap_score(hi_band, left_a)
    up_sign = 1.0 if gap_lo >= gap_hi else -1.0
    if gap_lo < 0 and gap_hi < 0:
        notes.append("脚/头双簇判定失效，up 符号取正轴")
    up = np.zeros(3)
    up[up_a] = up_sign
    # forward 符号：脚端带前凸尾比头带鼻尖信号更强
    foot_band = lo_band if up_sign > 0 else hi_band
    fv = foot_band[:, fwd_a] if len(foot_band) >= 8 else p[:, fwd_a]
    mu = float(fv.mean())
    fwd_sign = 1.0 if (fv.max() - mu) >= (mu - fv.min()) else -1.0
    notes.append("无骨架：forward 符号由脚带前凸尾判定，如镜像请在审核视图核对")
    fwd = np.zeros(3)
    fwd[fwd_a] = fwd_sign
    return up, fwd


def _joint_points(g, mats: List[np.ndarray]) -> np.ndarray:
    """所有 skin 关节的全局位置 (n,3)；无骨架时返回空数组。

    动画素材常见形态是“只有骨架没有网格”，此时顶点集为空，OBB 与身高只能拿
    关节位置量（否则单位推断拿不到 up 轴跨度，只能保留原尺度）。
    """
    pts: List[np.ndarray] = []
    for skin in (getattr(g, "skins", []) or []):
        for ji in (skin.joints or []):
            if 0 <= ji < len(mats) and mats[ji] is not None:
                pts.append(np.asarray(mats[ji][:3, 3], dtype=np.float64))
    if not pts:
        return np.zeros((0, 3))
    return np.stack(pts)


def estimate_frame(g) -> Dict[str, Any]:
    """探测模型当前轴系，返回规范基向量（模型坐标表示）：left/up/forward + 诊断信息。

    结果**吸附到世界坐标轴**（旧行为，``/v1`` 链路依赖，不得改）；S1 导入矫正改用
    :func:`estimate_directions` 拿未吸附方向去贴合 OBB 面。
    """
    mats = global_matrices(g)
    names = [n.name or f"node{i}" for i, n in enumerate(g.nodes or [])]
    notes: List[str] = []
    up = fwd = None
    method = "mesh"
    sd = _skeleton_dirs(g, mats, names)
    if sd is not None:
        if sd["up_raw"] is not None:
            up = _snap_axis(sd["up_raw"])
            method = "skeleton"
        if sd["foot_vec"] is not None and up is not None:
            fv = sd["foot_vec"] - up * np.dot(sd["foot_vec"], up)
            fwd = _snap_axis(fv)
        # 左右命名对校验 handedness（防前后判反导致镜像）
        if up is not None and fwd is not None and sd["left_vec"] is not None:
            if np.dot(sd["left_vec"], np.cross(up, fwd)) < 0:
                fwd = -fwd
                notes.append(f"左右命名对 {sd['pair']} 校验翻转了 forward 符号")
    if up is None or fwd is None:
        mu, mf = _mesh_dirs(g, mats, notes)
        if mu is None or mf is None:
            return {"left": [1, 0, 0], "up": [0, 1, 0], "forward": [0, 0, 1],
                    "method": "identity", "notes": notes}
        up, fwd, method = mu, mf, "mesh"
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


def estimate_directions(g) -> Dict[str, Any]:
    """探测 up / forward 的**未吸附**参考方向（模型坐标），供 OBB 面指派使用。

    与 :func:`estimate_frame` 的差别：不吸附到世界坐标轴（斜放模型吸附会丢掉真实
    朝向），改由 :func:`obb.assign_faces` 贴合到最接近的 OBB 面。骨架结果优先，
    网格路径只补缺失的那一个方向（网格本身只能给出轴向）。
    """
    mats = global_matrices(g)
    names = [n.name or f"node{i}" for i, n in enumerate(g.nodes or [])]
    notes: List[str] = []
    up_dir = fwd_dir = None
    method = "mesh"
    sd = _skeleton_dirs(g, mats, names)
    if sd is not None and sd["up_raw"] is not None:
        u = np.asarray(sd["up_raw"], dtype=np.float64)
        nu = float(np.linalg.norm(u))
        if nu > 1e-9:
            up_dir = u / nu
            method = "skeleton"
            if sd["foot_vec"] is not None:
                fv = sd["foot_vec"] - up_dir * float(np.dot(sd["foot_vec"], up_dir))
                nf = float(np.linalg.norm(fv))
                if nf > 1e-9:
                    fwd_dir = fv / nf
            if fwd_dir is not None and sd["left_vec"] is not None:
                if float(np.dot(sd["left_vec"], np.cross(up_dir, fwd_dir))) < 0:
                    fwd_dir = -fwd_dir
                    notes.append(f"左右命名对 {sd['pair']} 校验翻转了 forward 符号")
    if up_dir is None or fwd_dir is None:
        mark = len(notes)
        mu, mf = _mesh_dirs(g, mats, notes)
        if up_dir is not None:
            del notes[mark:]      # 骨架已定 up，网格路径的「顶点不足」提示是噪音
        if up_dir is None and mu is not None:
            up_dir = mu
            method = "mesh"
        if fwd_dir is None and mf is not None:
            fwd_dir = mf
    if up_dir is None and fwd_dir is None:
        return {"up_dir": None, "forward_dir": None,
                "method": "identity", "notes": notes}
    if fwd_dir is None:
        # 纯骨架动画（无网格、又缺 toe/ball）的常见形态：up 已由 pelvis→head 定住，
        # forward 交给 assign_faces 按最薄面兜底，**不得**把已知的 up 一并丢掉
        notes.append("缺 forward 先验（无网格且无 toe/ball 骨），由最薄面兜底指派")
    return {"up_dir": up_dir.tolist() if up_dir is not None else None,
            "forward_dir": fwd_dir.tolist() if fwd_dir is not None else None,
            "method": method, "notes": notes}


def frame_rotation(frame: Dict[str, Any]) -> np.ndarray:
    """模型坐标 → 规范坐标 的旋转矩阵：行 = [left, up, forward]。"""
    l = np.array(frame["left"], float)
    u = np.array(frame["up"], float)
    f = np.array(frame["forward"], float)
    return np.stack([l, u, f])


# --------------------------------------------------------------------------- #
# 单位推断与应用：场景根插入校正节点
# --------------------------------------------------------------------------- #
def estimate_unit(extent_up: float, target_height: Optional[float] = None,
                  unit: Optional[str] = None) -> Dict[str, Any]:
    """按 up 轴跨度推断原始单位，返回换算到米的系数与米制身高。

    优先级 ``target_height``（人工填真实身高，scale = 身高 / 跨度）> ``unit``（人工
    强制单位）> 自动推断。自动推断取「换算后身高与成人中值 1.75m 的对数距离最小」
    的单位，比固定数量级区间鲁棒（inch 与 cm 的区间在 50~120 重叠，单靠区间会误判）。
    """
    ext = float(extent_up) if extent_up is not None else 0.0
    if not np.isfinite(ext) or ext <= 1e-9:
        return {"detected": "m", "scale": 1.0, "height_m": 0.0,
                "candidates": [], "hint": "无法测量高度，保留原尺度"}
    cands = []
    for name, sc in _UNIT_SCALES:
        h = ext * sc
        cands.append({"unit": name, "scale": sc, "height_m": round(h, 6),
                      "log_dist": round(float(abs(np.log(h / _REF_HEIGHT))), 4)})
    cands.sort(key=lambda c: c["log_dist"])
    hint = ""
    if target_height is not None and float(target_height) > 1e-6:
        detected = "custom"
        scale = float(target_height) / ext
        height = float(target_height)
    else:
        want = str(unit).strip().lower() if unit else ""
        forced = next((c for c in cands if c["unit"] == want), None) if want else None
        if want and forced is None:
            hint = (f"未知单位 {unit!r}，已回退自动推断；可选 "
                    f"{'/'.join(n for n, _ in _UNIT_SCALES)}")
        best = forced or cands[0]
        detected, scale, height = best["unit"], best["scale"], best["height_m"]
        lo, hi = HUMAN_HEIGHT_RANGE
        if not (lo <= height <= hi):
            extra = (f"身高 {height:.3f}m 不在成人区间 [{lo}, {hi}]，可能是归一化资产"
                     f"或非人形；若知道真实身高请在阶段一填写")
            hint = f"{hint}；{extra}" if hint else extra
    return {"detected": detected, "scale": float(scale), "height_m": float(height),
            "candidates": cands, "hint": hint}


def apply_canonical(g, R: np.ndarray, scale: float = 1.0) -> bool:
    """在场景根插入（或原地更新）校正节点：纯旋转 + 均匀缩放。

    返回是否改动了场景。两个关键约束：

    1. 缩放挂在同一节点是安全的：``glb_io.merge_mesh`` 与
       ``retarget._global_rest_positions`` 都按节点全局矩阵（含 S）取值，numpy 侧直接
       得到米制；而 ``glb_build.build_result_glb`` 用 merge 顶点 + rig.heads 重建，
       产物不含本节点，故 identity-rest（inverseBindMatrix = translate(-head)）不错位。
    2. 人工修正会反复调用本函数，**不得删旧节点重建**（glTF 按索引引用节点，
       删除中间节点会使 skin.joints / scene.nodes 集体错位），而是原地改 TRS。
    """
    s = float(scale) if (scale is not None and float(scale) > 0.0) else 1.0
    scene_idx = g.scene or 0
    scene = g.scenes[scene_idx]
    roots = list(scene.nodes or [])
    if not roots:
        return False
    new_rot = [float(v) for v in _mat_to_quat(R)]
    new_scl = [s, s, s]
    exist = next((i for i, n in enumerate(g.nodes or [])
                  if (n.name or "") == CANON_NODE), None)
    if exist is not None:
        node = g.nodes[exist]
        old_rot = [float(v) for v in (node.rotation or [0.0, 0.0, 0.0, 1.0])]
        old_scl = [float(v) for v in (node.scale or [1.0, 1.0, 1.0])]
        node.rotation = new_rot
        node.scale = new_scl
        return not (np.allclose(old_rot, new_rot, atol=1e-9)
                    and np.allclose(old_scl, new_scl, atol=1e-12))
    if np.allclose(R, np.eye(3), atol=1e-6) and abs(s - 1.0) < 1e-9:
        return False                      # 无需校正，不插多余节点（保持旧行为）
    # 函数内 import：本模块的几何计算部分不依赖 pygltflib，便于单测
    from pygltflib import Node
    node = Node(name=CANON_NODE, rotation=new_rot, scale=new_scl, children=roots)
    g.nodes.append(node)
    scene.nodes = [len(g.nodes) - 1]
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


# --------------------------------------------------------------------------- #
# S1 导入矫正：OBB 面映射 → 规范系（新路径，/v2 资产库使用）
# --------------------------------------------------------------------------- #
# 语义 token → (R 的行索引, 符号)；行序与 frame_rotation 一致：
# 0=left(+X)、1=up(+Y)、2=forward(+Z)
_SEMANTIC_ROWS: Dict[str, Tuple[int, float]] = {
    "up+": (1, 1.0), "up-": (1, -1.0),
    "front+": (2, 1.0), "front-": (2, -1.0),
    "left+": (0, 1.0), "left-": (0, -1.0),
}
# 常见写法归一（前端下拉直接用 FACE_SEMANTICS，这里只为兼容手写与 API 调用）
_SEM_ALIASES: Dict[str, str] = {
    "up": "up+", "y": "up+", "y+": "up+", "y-": "up-",
    "front": "front+", "forward": "front+", "forward+": "front+", "forward-": "front-",
    "fwd": "front+", "fwd+": "front+", "fwd-": "front-",
    "z": "front+", "z+": "front+", "z-": "front-",
    "left": "left+", "x": "left+", "x+": "left+", "x-": "left-",
}
FACE_SEMANTICS: Tuple[str, ...] = tuple(_SEMANTIC_ROWS)
_ROW_NAMES: Tuple[str, ...] = ("left(+X)", "up(+Y)", "front(+Z)")
_MANUAL_KEYS: Tuple[str, ...] = ("face_map", "trim_euler", "unit_override",
                                 "height_override")


def _parse_semantic(token: Any) -> Tuple[int, float]:
    """把人工填写的语义 token 归一为 (R 行索引, 符号)；无法识别抛 ``ValueError``。"""
    t = str(token or "").strip().lower().replace(" ", "")
    t = _SEM_ALIASES.get(t, t)
    if t not in _SEMANTIC_ROWS:
        raise ValueError(f"无法识别的轴向语义 {token!r}，可选 "
                         f"{'/'.join(FACE_SEMANTICS)}")
    return _SEMANTIC_ROWS[t]


def _euler_xyz_mat(deg: Any) -> np.ndarray:
    """绕规范系固定轴 X→Y→Z 的外旋矩阵（度），即 ``Rz · Ry · Rx``。"""
    src = np.asarray(list(deg) if deg is not None else [], dtype=np.float64).ravel()
    a = np.zeros(3, dtype=np.float64)
    a[:min(3, len(src))] = src[:3]
    rx, ry, rz = np.deg2rad(a)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _trim_list(raw: Any) -> List[float]:
    """``trim_euler`` 归一为三个 float（度），缺失或非法项按 0 处理。"""
    out = [0.0, 0.0, 0.0]
    try:
        src = [float(v) for v in (raw or [])][:3]
    except (TypeError, ValueError):
        return out
    for i, v in enumerate(src):
        out[i] = v if np.isfinite(v) else 0.0
    return out


def _pos_float(raw: Any) -> Optional[float]:
    """人工填写的正数（真实身高 / 覆盖值）；非法或非正一律当未填。"""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0.0 else None


def frame_from_face_map(obb: OBB, face_map: Dict[Any, Any],
                        trim_euler: Any = None) -> np.ndarray:
    """由「面序号 → 语义轴」映射组装 模型坐标 → 规范坐标 的旋转矩阵。

    ``face_map`` 形如 ``{"1": "up+", "5": "front+"}``（键可为 int/str，值语义见
    :data:`FACE_SEMANTICS`）：面外法向即该语义轴在模型坐标下的方向。三轴缺其一时
    由右手系导出（``left = cross(up, forward)``）；``trim_euler`` 为绕规范系
    X/Y/Z 的外旋微调（度）后置叠加：``R = Rz·Ry·Rx · R_face``。

    面序号越界、语义 token 不识别、同一语义轴被两面重复指派、或少于两个不同语义轴
    时抛 ``ValueError``（由 ``/v2`` 路由转成 400）。
    """
    normals = obb.face_normals()
    rows = np.zeros((3, 3), dtype=np.float64)
    filled = np.zeros(3, dtype=bool)
    seen: Dict[int, int] = {}
    for key, token in dict(face_map or {}).items():
        fid = int(key)
        if not 1 <= fid <= 6:
            raise ValueError(f"面序号须在 1..6：{key!r}")
        row, sign = _parse_semantic(token)
        if row in seen:
            raise ValueError(f"{_ROW_NAMES[row]} 被面 {seen[row]} 与面 {fid} 重复指派")
        seen[row] = fid
        rows[row] = sign * normals[fid - 1]
        filled[row] = True
    missing = [i for i in range(3) if not filled[i]]
    if len(missing) > 1:
        raise ValueError("至少要在六面中指派两个不同的语义轴（第三个由右手系导出）")
    for i in missing:
        # 规范系为右手：X=left, Y=up, Z=forward，故 row_i = cross(row_i+1, row_i+2)
        rows[i] = np.cross(rows[(i + 1) % 3], rows[(i + 2) % 3])
    if not np.allclose(rows @ rows.T, np.eye(3), atol=1e-6) \
            or float(np.linalg.det(rows)) <= 0.0:
        raise ValueError("面映射得到的三轴不正交或构成左手系，"
                         "请检查是否把对偶面（如 1 与 2）指派成了两个语义轴")
    r_trim = _euler_xyz_mat(trim_euler)
    if not np.allclose(r_trim, np.eye(3), atol=1e-12):
        rows = r_trim @ rows
    return rows


def _reset_canon(g) -> None:
    """把已存在的 canon 节点复位为单位变换。

    OBB 与身高都必须在**原始模型坐标**下测量；人工修正会反复重跑，不先复位就会在
    上一次的旋转结果上再测一次（旋转累积 + 身高量错）。也因此，``normalize_to_glb``
    旧路径插入的吸附旋转会被复位后由 OBB 结果接管，不会叠加。
    """
    for n in (g.nodes or []):
        if (n.name or "") == CANON_NODE:
            n.matrix = None
            n.rotation = [0.0, 0.0, 0.0, 1.0]
            n.scale = [1.0, 1.0, 1.0]
            return


def _canon_trs(g) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """当前 canon 节点的 ``(rotation, scale)``；无节点返回 ``None``。

    :func:`build_align` 会先复位节点，故 ``apply_canonical`` 自己的返回值在这里恒为
    “有差异”；要判断文件是否真被改动，必须在复位**前**先存一份 TRS 对比。
    """
    for n in (g.nodes or []):
        if (n.name or "") == CANON_NODE:
            return (np.asarray(n.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    np.asarray(n.scale or [1.0, 1.0, 1.0], dtype=np.float64))
    return None


def _same_trs(a: Optional[Tuple[np.ndarray, np.ndarray]],
              b: Optional[Tuple[np.ndarray, np.ndarray]]) -> bool:
    """两份 canon TRS 是否等价（均无节点也算等价）。"""
    if (a is None) != (b is None):
        return False
    if a is None or b is None:
        return True
    return bool(np.allclose(a[0], b[0], atol=1e-9) and np.allclose(a[1], b[1], atol=1e-12))


def build_align(g, align: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """在已加载的 GLB 上算出 align.json 结构（不改场景变换、不落盘）。

    ``align`` 传入上一次的完整 align.json 或直接传 ``manual`` 段均可；只有 ``manual``
    影响结果，OBB 与自动指派每次重测。``final.changed`` 留给 :func:`align_asset` 填。
    """
    _reset_canon(g)
    mats = global_matrices(g)
    pts = _mesh_points(g, mats)
    from_joints = False
    if len(pts) < 16:
        pts = _joint_points(g, mats)
        from_joints = len(pts) >= 4
    dirs = estimate_directions(g)
    raw_manual = dict((align or {}).get("manual") or {})
    if not raw_manual and align:
        raw_manual = {k: v for k, v in align.items() if k in _MANUAL_KEYS}
    override = raw_manual.get("unit_override")
    manual: Dict[str, Any] = {
        "face_map": {str(k): str(v) for k, v in (raw_manual.get("face_map") or {}).items()},
        "trim_euler": _trim_list(raw_manual.get("trim_euler")),
        "unit_override": str(override).strip().lower() if override else None,
        "height_override": _pos_float(raw_manual.get("height_override")),
    }
    notes: List[str] = list(dirs.get("notes") or [])
    if from_joints:
        notes.append("无网格：OBB 与身高按骨架关节位置测量（动画素材的常见形态）")
    auto: Dict[str, Any] = {"up_face": None, "forward_face": None, "left_face": None,
                            "method": dirs.get("method"), "notes": notes}
    faces: List[Dict[str, Any]] = []
    obb = min_volume_obb(pts)
    if obb is None:
        notes.append("OBB 求解失败（顶点不足或共面），退化为世界轴吸附")
    else:
        normals = obb.face_normals()
        ext = np.asarray(obb.extents, dtype=np.float64)
        faces = [{"id": i + 1,
                  "normal": [round(float(v), 6) for v in normals[i]],
                  "extent_face": FACE_LABELS[i],
                  "extent": round(float(ext[FACE_AXES[i][0]]), 6)} for i in range(6)]
        up_dir = np.asarray(dirs["up_dir"], dtype=np.float64) if dirs.get("up_dir") else None
        fwd_dir = (np.asarray(dirs["forward_dir"], dtype=np.float64)
                   if dirs.get("forward_dir") else None)
        got = assign_faces(obb, up_dir, fwd_dir)
        auto.update({"up_face": got["up_face"], "forward_face": got["forward_face"],
                     "left_face": got["left_face"]})
        notes.extend(got["notes"])
    # manual 非空即整体采用（left 可缺省）；为空则用 auto 指派的 up/front 两面
    face_map = dict(manual["face_map"])
    if not face_map:
        if auto["up_face"]:
            face_map[str(auto["up_face"])] = "up+"
        if auto["forward_face"]:
            face_map[str(auto["forward_face"])] = "front+"
    if obb is not None:
        rot = frame_from_face_map(obb, face_map, manual["trim_euler"])
    else:
        rot = _euler_xyz_mat(manual["trim_euler"]) @ frame_rotation(estimate_frame(g))
    if obb is not None and auto["up_face"]:
        extent_up = float(obb.extents[FACE_AXES[auto["up_face"] - 1][0]])
    else:
        span = (pts.max(0) - pts.min(0)) if len(pts) else np.zeros(3)
        extent_up = float(span.max())
    unit = estimate_unit(extent_up, target_height=manual["height_override"],
                         unit=manual["unit_override"])
    return {
        "version": ALIGN_VERSION,
        "obb": obb.to_dict() if obb is not None else None,
        "faces": faces,
        "unit": unit,
        "auto": auto,
        "manual": manual,
        "final": {"rotation": [[round(float(v), 9) for v in row] for row in rot],
                  "scale": unit["scale"], "changed": False},
    }


def align_asset(path, align: Optional[Dict[str, Any]] = None,
                json_path=None) -> Dict[str, Any]:
    """S1 导入矫正：对已归一为 GLB 的资产做 OBB + 单位 + 面映射矫正（原地保存）。

    与 :func:`normalize_axes_file` 的区别：旋转不限于世界轴吸附（OBB 给出最紧凑摆放），
    且附带单位缩放。传入上一次的 align.json 即走人工修正重算路径：``PATCH`` 只改
    ``manual``，本函数重测 OBB 并重算 ``final`` + 重写 canon 节点。

    返回完整 align.json 结构；``json_path`` 给定时同时写出侧车文件。
    """
    from ai3d.retarget.glb_io import _load, save_glb

    g = _load(path)
    before = _canon_trs(g)
    info = build_align(g, align)
    apply_canonical(g, np.asarray(info["final"]["rotation"], dtype=np.float64),
                    float(info["final"]["scale"]))
    changed = not _same_trs(before, _canon_trs(g))
    info["final"]["changed"] = changed
    if changed:
        save_glb(g, path)
    if json_path is not None:
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    auto = info["auto"]
    logger.info("S1 对齐 %s：%s up_face=%s front_face=%s unit=%s scale=%.6g "
                "height=%.4fm changed=%s %s",
                Path(path).name, auto["method"], auto["up_face"], auto["forward_face"],
                info["unit"]["detected"], info["final"]["scale"],
                info["unit"]["height_m"], changed, ";".join(auto["notes"]) or "")
    return info
