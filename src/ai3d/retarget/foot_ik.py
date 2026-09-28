"""脚部接触检测 + 双骨 IK 锁定（P3）。

在重定向烘焙后的「局部旋转 + root 平移」上做后处理，消除支撑相滑步：
1. FK 复原每帧关节世界位置/全局旋转（identity-rest 约定，与 ``glb_build`` 一致）；
2. 接触检测：脚踝高度接近地面 **且** 水平速度低 → 支撑相（摆动相不锁，避免拖拽）；
3. 每个接触窗口把脚踝世界位置锁定到窗口均值，窗口边缘按 ``foot_blend_frames`` 平滑混入/混出；
4. 双骨 IK（大腿 L1 - 小腿 L2）以 FK 膝盖为极向量求解膝点，再反解髋/膝**局部**旋转；
5. 仅改写 upper_leg/lower_leg 通道；无接触窗口时原样返回（不影响既有动画）。

关键：脚踝位置只取决于髋/膝旋转（foot 自身旋转只影响脚趾朝向），故锁定脚踝无需改 foot 通道。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .retarget import (
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_to_matrix,
)
from .settings import RetargetConfig
from .skeleton import JOINTS, PARENTS, Rig

logger = logging.getLogger(__name__)

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# 四元数 / 向量工具
# --------------------------------------------------------------------------- #
def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q 旋转向量 v。"""
    return quat_to_matrix(q) @ np.asarray(v, float)


def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """从向量 a 到 b 的最小旋转四元数 [x,y,z,w]（无需预先归一化）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return _IDENTITY.copy()
    a, b = a / na, b / nb
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return _IDENTITY.copy()
    if dot < -1.0 + 1e-9:  # 反向：取任一垂直轴转 180°
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - (axis @ a) * a
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    return quat_normalize(np.array([axis[0], axis[1], axis[2], 1.0 + dot]))


def _fk(rig: Rig, rotations: Dict[str, np.ndarray],
        root_t: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """正向运动学 → 每关节世界位置 (F,3) 与全局旋转四元数 (F,4)。"""
    F = len(root_t)
    rest_off = {j: (rig.heads[j] if PARENTS[j] is None
                    else rig.heads[j] - rig.heads[PARENTS[j]]) for j in JOINTS}
    P: Dict[str, np.ndarray] = {j: np.zeros((F, 3)) for j in JOINTS}
    Rg: Dict[str, np.ndarray] = {j: np.zeros((F, 4)) for j in JOINTS}
    for f in range(F):
        for j in JOINTS:  # JOINTS 拓扑有序：父在子前
            p = PARENTS[j]
            q = np.asarray(rotations[j][f], float)
            if p is None:
                Rg[j][f] = quat_normalize(q)
                P[j][f] = root_t[f]
            else:
                Rg[j][f] = quat_normalize(quat_mul(Rg[p][f], q))
                P[j][f] = P[p][f] + quat_rotate(Rg[p][f], rest_off[j])
    return P, Rg


def _contact_windows(contact: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
    """把布尔接触序列切成 [start, end) 窗口，丢弃短于 min_frames 的。"""
    wins: List[Tuple[int, int]] = []
    n = len(contact)
    i = 0
    while i < n:
        if contact[i]:
            j = i
            while j < n and contact[j]:
                j += 1
            if j - i >= min_frames:
                wins.append((i, j))
            i = j
        else:
            i += 1
    return wins


def _solve_two_bone(hip: np.ndarray, ankle: np.ndarray, knee_pole: np.ndarray,
                    L1: float, L2: float) -> Optional[np.ndarray]:
    """双骨 IK：髋/踝世界位置 + 膝极向量 → 膝世界位置。"""
    d_vec = ankle - hip
    d = float(np.linalg.norm(d_vec))
    if d < 1e-6:
        return None
    lo, hi = abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4
    if hi <= lo:
        return None
    dc = float(np.clip(d, lo, hi))
    u = d_vec / d
    a = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc)
    h = np.sqrt(max(0.0, L1 * L1 - a * a))
    mid = hip + a * u
    pv = knee_pole - hip
    pv = pv - (pv @ u) * u  # 垂直于 hip→ankle 轴的分量
    if np.linalg.norm(pv) < 1e-6:  # 退化（腿伸直）：用世界前方作为弯曲方向
        for fallback in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])):
            pv = fallback - (fallback @ u) * u
            if np.linalg.norm(pv) > 1e-6:
                break
    v = pv / np.linalg.norm(pv)
    return mid + h * v


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def apply_foot_lock(rig: Rig, rotations: Dict[str, np.ndarray],
                    root_translations: np.ndarray, times: np.ndarray,
                    cfg: Optional[RetargetConfig] = None
                    ) -> Tuple[Dict[str, np.ndarray], Dict]:
    """对支撑相脚踝做锁定 IK，返回（可能修改后的）rotations 与信息 dict。"""
    cfg = cfg or RetargetConfig()
    root_translations = np.asarray(root_translations, float)
    times = np.asarray(times, float)
    F = len(root_translations)
    info: Dict = {"applied": False, "windows": {"l": 0, "r": 0},
                  "contact_frames": {"l": 0, "r": 0}, "max_correction": 0.0}
    if F < 2:
        return rotations, info

    P, Rg = _fk(rig, rotations, root_translations)
    height = float(rig.height) or 1.0
    ground_y = float(min(P["foot_l"][:, 1].min(), P["foot_r"][:, 1].min()))
    h_thresh = ground_y + cfg.foot_contact_height * height

    dt = np.gradient(times)
    dt = np.where(np.abs(dt) < 1e-6, 1e-6, dt)
    rot_out: Dict[str, np.ndarray] = {j: np.asarray(rotations[j], float).copy()
                                      for j in JOINTS}
    blend = max(1, int(cfg.foot_blend_frames))
    any_applied = False
    max_corr = 0.0

    for side in ("l", "r"):
        hip_j, knee_j, ankle_j = f"upper_leg_{side}", f"lower_leg_{side}", f"foot_{side}"
        ankle = P[ankle_j]
        # 3D 速度：只看 xz 会把原地踏步（踝垂直运动大、水平几乎不动）误判为支撑相
        speed = np.linalg.norm(np.gradient(ankle, axis=0) / dt[:, None], axis=1)
        contact = (ankle[:, 1] < h_thresh) & (speed < cfg.foot_contact_speed)
        info["contact_frames"][side] = int(contact.sum())
        wins = _contact_windows(contact, int(cfg.foot_min_contact_frames))
        info["windows"][side] = len(wins)
        if not wins:
            continue

        # rest 骨向量（局部）
        thigh = rig.heads[knee_j] - rig.heads[hip_j]
        shin = rig.heads[ankle_j] - rig.heads[knee_j]
        L1, L2 = float(np.linalg.norm(thigh)), float(np.linalg.norm(shin))
        u1 = thigh / L1
        u2 = shin / L2
        parent_hip = PARENTS[hip_j]

        for (a, b) in wins:
            target = ankle[a:b].mean(axis=0)  # 锁定到窗口均值
            for f in range(a, b):
                w = min(min(1.0, (f - a + 1) / blend), min(1.0, (b - f) / blend))
                if w <= 0.0:
                    continue
                goal = ankle[f] * (1.0 - w) + target * w
                max_corr = max(max_corr, float(np.linalg.norm(goal - ankle[f])))
                hip_pos = P[hip_j][f]
                knee = _solve_two_bone(hip_pos, goal, P[knee_j][f], L1, L2)
                if knee is None:
                    continue
                # --- 髋：把大腿 rest 方向转到 hip→knee ---
                R_pelvis = Rg[parent_hip][f]
                R_upper_old = Rg[hip_j][f]
                thigh_cur = quat_rotate(R_upper_old, u1)
                q_delta_hip = quat_between(thigh_cur, knee - hip_pos)
                R_upper_new = quat_normalize(quat_mul(q_delta_hip, R_upper_old))
                rot_out[hip_j][f] = quat_normalize(quat_mul(quat_inv(R_pelvis), R_upper_new))
                # --- 膝：把小腿 rest 方向转到 knee→goal（相对新髋）---
                q_knee_old = np.asarray(rotations[knee_j][f], float)
                R_lower_cur = quat_normalize(quat_mul(R_upper_new, q_knee_old))
                shin_cur = quat_rotate(R_lower_cur, u2)
                q_delta_knee = quat_between(shin_cur, goal - knee)
                R_lower_new = quat_normalize(quat_mul(q_delta_knee, R_lower_cur))
                rot_out[knee_j][f] = quat_normalize(quat_mul(quat_inv(R_upper_new), R_lower_new))
                any_applied = True

    info["applied"] = any_applied
    info["max_correction"] = round(max_corr, 6)
    if any_applied:
        logger.info("脚部锁定：窗口 L=%d R=%d，最大校正 %.4fm",
                    info["windows"]["l"], info["windows"]["r"], max_corr)
    return rot_out, info
