"""动画重定向：把源骨架动画烘焙到 22 语义骨架（P1 确定性路径）。

核心思路（局部增量传递 + rest 朝向共轭）：
- 目标骨架采用 identity-rest 设计（关节 rest 局部旋转=单位、局部系即世界系，骨向由平移编码）。
- 源关节动画增量 D = R_src_rest^-1 * R_src_anim 表达在源骨 rest 局部系，须共轭到世界系再输出：
      R_tgt[f] = A * D * A^-1，A = 源关节 rest 全局旋转（含 canon_axis_root 校正）。
  缺这步共轭会把源骨 rest 朝向差当成动画叠加、沿链累积成严重扭曲；
  源为 identity-rest（合成源）时 A=单位，退化为 R_tgt[f]=R_src_anim[f] 精确传递。
- root motion：骨盆平移增量先乘父链 rest 旋转转到世界系，再按身高比例缩放叠加到目标骨盆 head。
- 逐帧采样：rotation 用 slerp、translation 用 lerp，统一到输出时间轴（bake_fps>0 时重采样）。

P3 再补：脊柱骨数不匹配按骨长分配、脚部接触检测 + 双骨 IK、rest 姿态对齐校正。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import glb_io
from .axis_norm import CANON_NODE
from .skeleton import JOINTS, Rig
from .settings import RetargetConfig

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# 四元数工具（glTF 约定 [x, y, z, w]）
# --------------------------------------------------------------------------- #
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    t = np.asarray(t, np.float64)
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0)[..., None], -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    st_safe = np.where(small, 1.0, sin_t)
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / st_safe)
    w1 = np.where(small, t, np.sin(t * theta) / st_safe)
    return quat_normalize(w0[..., None] * q0 + w1[..., None] * q1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(np.asarray(q, np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return quat_normalize(np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                                        (R[1, 0] - R[0, 1]) / s, 0.25 * s]))
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return quat_normalize(np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                                        (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]))
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return quat_normalize(np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                                        (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]))
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return quat_normalize(np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                                    0.25 * s, (R[1, 0] - R[0, 1]) / s]))


# --------------------------------------------------------------------------- #
# 通道采样
# --------------------------------------------------------------------------- #
def sample_rotation(times: np.ndarray, values: np.ndarray, out_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, np.float64).ravel()
    values = np.asarray(values, np.float64).reshape(-1, 4)
    n = len(out_times)
    if len(times) == 0 or len(values) == 0:
        return np.tile(_IDENTITY, (n, 1))
    if len(times) == 1:
        return np.tile(quat_normalize(values[0]), (n, 1))
    idx = np.clip(np.searchsorted(times, out_times, side="right") - 1, 0, len(times) - 2)
    t0, t1 = times[idx], times[idx + 1]
    span = np.where((t1 - t0) < 1e-9, 1e-9, t1 - t0)
    alpha = np.clip((out_times - t0) / span, 0.0, 1.0)
    return quat_slerp(values[idx], values[idx + 1], alpha)


def sample_translation(times: np.ndarray, values: np.ndarray, out_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, np.float64).ravel()
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    n = len(out_times)
    d = values.shape[1] if values.size else 3
    if len(times) == 0 or len(values) == 0:
        return np.zeros((n, d))
    if len(times) == 1:
        return np.tile(values[0], (n, 1))
    idx = np.clip(np.searchsorted(times, out_times, side="right") - 1, 0, len(times) - 2)
    t0, t1 = times[idx], times[idx + 1]
    span = np.where((t1 - t0) < 1e-9, 1e-9, t1 - t0)
    alpha = np.clip((out_times - t0) / span, 0.0, 1.0)
    return values[idx] * (1 - alpha[:, None]) + values[idx + 1] * alpha[:, None]


# --------------------------------------------------------------------------- #
# 源节点 rest 变换 / 全局位置
# --------------------------------------------------------------------------- #
def _node_local_matrix(node: Dict[str, Any]) -> np.ndarray:
    m = node.get("matrix")
    if m is not None:
        return np.asarray(m, np.float64).reshape(4, 4).T  # glTF 列主序 → 行主序
    T = np.eye(4)
    t = node.get("translation")
    if t is not None:
        T[:3, 3] = np.asarray(t, np.float64)
    r = node.get("rotation")
    R = quat_to_matrix(np.asarray(r, np.float64)) if r is not None else np.eye(3)
    s = node.get("scale")
    S = np.diag(np.asarray(s, np.float64)) if s is not None else np.eye(3)
    T[:3, :3] = R @ S
    return T


def _node_rest_quat(node: Dict[str, Any]) -> np.ndarray:
    r = node.get("rotation")
    if r is not None:
        return quat_normalize(np.asarray(r, np.float64))
    m = node.get("matrix")
    if m is not None:
        return matrix_to_quat(np.asarray(m, np.float64).reshape(4, 4).T[:3, :3])
    return _IDENTITY.copy()


def _node_rest_translation(node: Dict[str, Any]) -> np.ndarray:
    t = node.get("translation")
    if t is not None:
        return np.asarray(t, np.float64)
    m = node.get("matrix")
    if m is not None:
        return np.asarray(m, np.float64).reshape(4, 4).T[:3, 3]
    return np.zeros(3)


def _node_parent_map(nodes: List[Dict[str, Any]]) -> Dict[int, Optional[int]]:
    parent: Dict[int, Optional[int]] = {n["index"]: None for n in nodes}
    for n in nodes:
        for c in n.get("children") or []:
            parent[c] = n["index"]
    return parent


def _global_rest_rotations(nodes: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """每节点全局 rest 旋转（3x3），自场景根累加（含 canon_axis_root 校正节点）。"""
    parent = _node_parent_map(nodes)
    cache: Dict[int, np.ndarray] = {}

    def rot(ni: int, guard: int = 0) -> np.ndarray:
        if ni in cache:
            return cache[ni]
        if guard > 4096 or not (0 <= ni < len(nodes)):
            return np.eye(3)
        R = quat_to_matrix(_node_rest_quat(nodes[ni]))
        p = parent.get(ni)
        M = R if p is None else rot(p, guard + 1) @ R
        cache[ni] = M
        return M

    for n in nodes:
        rot(n["index"])
    return cache


def _canon_unit_scale(nodes: List[Dict[str, Any]]) -> float:
    """场景根校正节点 ``canon_axis_root`` 的均匀缩放（无节点则 1.0）。

    该缩放是 S1 的**单位换算**（cm/mm/inch → m），不是资产自带比例，故不能计入
    :func:`_global_rest_positions` 的源身高（见其 docstring）。
    """
    for n in nodes:
        if (n.get("name") or "") == CANON_NODE:
            s = np.asarray(n.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64)
            return float(np.mean(s)) if s.size and float(np.mean(s)) > 0.0 else 1.0
    return 1.0


def _global_rest_positions(skin: Dict[str, Any],
                           nodes: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """关节全局 rest 位置：优先用 inverseBindMatrix 求逆，缺失则沿层级累加。

    **返回值的单位必须与动画通道的 translation 增量一致（即原始单位，非米制）**，
    因为 ``retarget_animation`` 靠 ``height_scale = rig.height / src_height`` 同时承担
    「源→目标身高比例」与「原始单位→米」两重换算：IBM 是资产导出时烘焙的，天然为
    原始单位；层级累加路径会把 S1 插入的 canon 缩放算进去，故需显式除掉。
    """
    joints = skin.get("joints") or []
    ibm = skin.get("inverse_bind_matrices")
    positions: Dict[int, np.ndarray] = {}
    if ibm is not None:
        ibm = np.asarray(ibm, np.float64)
        for k, ni in enumerate(joints):
            if k >= len(ibm):
                break
            M = ibm[k].T  # 行主序 IBM
            try:
                positions[ni] = np.linalg.inv(M)[:3, 3]
            except np.linalg.LinAlgError:
                continue
        if positions:
            return positions
    parent: Dict[int, int] = {}
    for n in nodes:
        for c in n.get("children") or []:
            parent[c] = n["index"]

    def gmat(ni: int) -> np.ndarray:
        M = _node_local_matrix(nodes[ni])
        p, guard = parent.get(ni), 0
        while p is not None and p < len(nodes) and guard < 4096:
            M = _node_local_matrix(nodes[p]) @ M
            p, guard = parent.get(p), guard + 1
        return M

    unit = _canon_unit_scale(nodes)
    for ni in joints:
        if 0 <= ni < len(nodes):
            positions[ni] = gmat(ni)[:3, 3] / unit
    return positions


def _height_from_positions(positions: List[np.ndarray]) -> float:
    if not positions:
        return 0.0
    P = np.stack([np.asarray(p, np.float64) for p in positions])
    return float(P[:, 1].max() - P[:, 1].min())


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def retarget_animation(source_glb: str | Path, mapping: Dict[str, Any], rig: Rig,
                       cfg: Optional[RetargetConfig] = None) -> Dict[str, Any]:
    """把 source_glb 的动画重定向到 rig，返回 {times, rotations, root_translations, meta}。"""
    cfg = cfg or RetargetConfig()
    g = glb_io._load(source_glb)
    skin = glb_io.extract_skin(g)
    if skin is None:
        raise RuntimeError("源文件不含骨架(skin)，无法重定向")
    nodes = glb_io.extract_nodes(g)
    anims = glb_io.extract_animations(g)
    if not anims or not anims[0].get("channels"):
        raise RuntimeError("源文件不含动画(animation)，无法重定向")
    anim = anims[0]

    node_ch: Dict[int, Dict[str, Dict]] = {}
    for ch in anim["channels"]:
        node_ch.setdefault(ch["target_node"], {})[ch["path"]] = ch

    # 输出时间轴
    all_t = [ch["times"] for ch in anim["channels"] if len(ch["times"])]
    duration = float(max(t.max() for t in all_t)) if all_t else 0.0
    fps = int(getattr(cfg, "bake_fps", 0) or 0)
    if fps > 0 and duration > 0:
        out_times = np.arange(0.0, duration + 1e-6, 1.0 / fps, dtype=np.float64)
        if len(out_times) < 2:
            out_times = np.array([0.0, duration])
    elif all_t:
        out_times = np.unique(np.concatenate(all_t))
    else:
        out_times = np.array([0.0])
    n_frames = len(out_times)

    # 身高比例（root motion 缩放）
    src_pos = _global_rest_positions(skin, nodes)
    src_height = _height_from_positions(list(src_pos.values()))
    height_scale = (rig.height / src_height) if (
        cfg.root_scale_by_height and src_height > 1e-6) else 1.0

    sem_to_node: Dict[str, Optional[int]] = mapping.get("semantic_to_source_node", {}) or {}

    # 每关节局部旋转增量，再共轭到世界轴：A 为源关节 rest 全局旋转，把「源骨 rest
    # 局部系」下的增量转到世界系（目标 identity-rest 的局部系即世界系）
    rest_rot = _global_rest_rotations(nodes)
    parent_of = _node_parent_map(nodes)
    rotations: Dict[str, np.ndarray] = {}
    for j in JOINTS:
        sn = sem_to_node.get(j)
        if sn is None or not (0 <= sn < len(nodes)):
            rotations[j] = np.tile(_IDENTITY, (n_frames, 1))
            continue
        r_rest = _node_rest_quat(nodes[sn])
        rc = node_ch.get(sn, {}).get("rotation")
        if rc is None:
            r_anim = np.tile(r_rest, (n_frames, 1))
        else:
            r_anim = sample_rotation(rc["times"], rc["values"], out_times)
        d = quat_mul(quat_inv(r_rest), r_anim)
        a = matrix_to_quat(rest_rot[sn])
        rotations[j] = quat_mul(a, quat_mul(d, quat_inv(a)))

    # root motion（骨盆平移）
    pelvis_head = np.asarray(rig.heads["pelvis"], np.float64)
    pn = sem_to_node.get("pelvis")
    if pn is not None and 0 <= pn < len(nodes):
        t_rest = _node_rest_translation(nodes[pn])
        tc = node_ch.get(pn, {}).get("translation")
        if tc is not None:
            t_anim = sample_translation(tc["times"], tc["values"], out_times)
        else:
            t_anim = np.tile(t_rest, (n_frames, 1))
        # 平移增量在父节点局部系：乘父链 rest 旋转转到世界系（轴系校正在此生效）
        delta = t_anim - t_rest
        pr = parent_of.get(pn)
        if pr is not None and pr in rest_rot:
            delta = delta @ rest_rot[pr].T
        root_t = pelvis_head + delta * height_scale
    else:
        root_t = np.tile(pelvis_head, (n_frames, 1))

    return {
        "times": out_times.astype(np.float32),
        "rotations": rotations,
        "root_translations": np.asarray(root_t, np.float32).reshape(n_frames, 3),
        "meta": {
            "frames": int(n_frames),
            "duration": round(duration, 6),
            "bake_fps": fps,
            "source_height": round(float(src_height), 6),
            "target_height": round(float(rig.height), 6),
            "height_scale": round(float(height_scale), 6),
            "mapped_joints": sum(1 for j in JOINTS if sem_to_node.get(j) is not None),
        },
    }


def stack_rotations(rotations: Dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    """dict joint→(F,4) 按 JOINTS 顺序堆叠为 (F, 22, 4)。"""
    out = np.zeros((n_frames, len(JOINTS), 4), dtype=np.float32)
    out[..., 3] = 1.0
    for i, j in enumerate(JOINTS):
        q = rotations.get(j)
        if q is None:
            continue
        q = np.asarray(q, np.float32).reshape(-1, 4)
        m = min(len(q), n_frames)
        out[:m, i, :] = q[:m]
    return out


def unstack_rotations(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """(F, 22, 4) → dict joint→(F,4)。"""
    arr = np.asarray(arr)
    return {j: arr[:, i, :] for i, j in enumerate(JOINTS)}


def with_fps(cfg: RetargetConfig, fps: Optional[int]) -> RetargetConfig:
    """用 JobConfig.fps 覆盖 bake_fps（fps>0 时）。"""
    if fps and int(fps) > 0:
        return replace(cfg, bake_fps=int(fps))
    return cfg
