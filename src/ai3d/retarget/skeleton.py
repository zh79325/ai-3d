"""22 节点统一语义骨架（glTF/Three 约定：+Y up，右手系）。

设计要点（P1）：
- 绑定姿态用「纯平移」定义：每个关节节点 rest 局部旋转为单位四元数，
  局部平移 = 本关节 head − 父关节 head。因此 inverseBindMatrix = translate(−head_global)。
- 动画即对每个关节施加局部旋转四元数（+ 骨盆平移做 root motion）。
- 目标关节在无 AI 时按人体比例从网格包围盒生成（首版限定标准双足人形、A/T Pose）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 关节顺序（索引即 skin.joints 中的顺序）
JOINTS: List[str] = [
    "pelvis", "spine_01", "spine_02", "chest", "neck", "head",
    "clavicle_l", "upper_arm_l", "lower_arm_l", "hand_l",
    "clavicle_r", "upper_arm_r", "lower_arm_r", "hand_r",
    "upper_leg_l", "lower_leg_l", "foot_l", "toe_l",
    "upper_leg_r", "lower_leg_r", "foot_r", "toe_r",
]

PARENTS: Dict[str, Optional[str]] = {
    "pelvis": None,
    "spine_01": "pelvis", "spine_02": "spine_01", "chest": "spine_02",
    "neck": "chest", "head": "neck",
    "clavicle_l": "chest", "upper_arm_l": "clavicle_l",
    "lower_arm_l": "upper_arm_l", "hand_l": "lower_arm_l",
    "clavicle_r": "chest", "upper_arm_r": "clavicle_r",
    "lower_arm_r": "upper_arm_r", "hand_r": "lower_arm_r",
    "upper_leg_l": "pelvis", "lower_leg_l": "upper_leg_l",
    "foot_l": "lower_leg_l", "toe_l": "foot_l",
    "upper_leg_r": "pelvis", "lower_leg_r": "upper_leg_r",
    "foot_r": "lower_leg_r", "toe_r": "foot_r",
}

JOINT_INDEX: Dict[str, int] = {j: i for i, j in enumerate(JOINTS)}

# 比例（相对身高 H，脚底 y=0）：(y_frac, x_frac, z_frac)，+x 为角色左侧
PROPORTIONS: Dict[str, Tuple[float, float, float]] = {
    "pelvis": (0.50, 0.0, 0.0),
    "spine_01": (0.58, 0.0, 0.0), "spine_02": (0.66, 0.0, 0.0),
    "chest": (0.74, 0.0, 0.0), "neck": (0.84, 0.0, 0.0), "head": (0.92, 0.0, 0.0),
    "clavicle_l": (0.80, 0.05, 0.0), "upper_arm_l": (0.80, 0.18, 0.0),
    "lower_arm_l": (0.80, 0.36, 0.0), "hand_l": (0.80, 0.52, 0.0),
    "clavicle_r": (0.80, -0.05, 0.0), "upper_arm_r": (0.80, -0.18, 0.0),
    "lower_arm_r": (0.80, -0.36, 0.0), "hand_r": (0.80, -0.52, 0.0),
    "upper_leg_l": (0.48, 0.10, 0.0), "lower_leg_l": (0.27, 0.10, 0.0),
    "foot_l": (0.06, 0.10, 0.0), "toe_l": (0.02, 0.10, 0.10),
    "upper_leg_r": (0.48, -0.10, 0.0), "lower_leg_r": (0.27, -0.10, 0.0),
    "foot_r": (0.06, -0.10, 0.0), "toe_r": (0.02, -0.10, 0.10),
}

MIRROR: Dict[str, str] = {}
for _j in JOINTS:
    if _j.endswith("_l"):
        MIRROR[_j] = _j[:-2] + "_r"
    elif _j.endswith("_r"):
        MIRROR[_j] = _j[:-2] + "_l"
    else:
        MIRROR[_j] = _j

CORE_JOINTS = ("pelvis", "chest", "upper_leg_l", "upper_leg_r",
               "lower_leg_l", "lower_leg_r", "upper_arm_l", "upper_arm_r",
               "lower_arm_l", "lower_arm_r")


def children_of(joint: str) -> List[str]:
    return [j for j in JOINTS if PARENTS[j] == joint]


@dataclass
class Rig:
    """语义骨架实例：每关节的全局 head 位置（米，Y-up）。"""
    heads: Dict[str, np.ndarray] = field(default_factory=dict)  # id -> (3,) float64
    height: float = 1.0
    revision: int = 0
    confidence: Dict[str, float] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)       # id -> 'proportion'|'solved'|'manual'
    locked: Dict[str, bool] = field(default_factory=dict)

    # ---- 派生 ----
    def order(self) -> List[str]:
        return list(JOINTS)

    def head_array(self) -> np.ndarray:
        """按 JOINTS 顺序返回 (22,3)。"""
        return np.stack([self.heads[j] for j in JOINTS]).astype(np.float64)

    def local_offsets(self) -> np.ndarray:
        """每关节相对父关节的局部平移 (22,3)；根关节为其全局 head。"""
        offs = np.zeros((len(JOINTS), 3), dtype=np.float64)
        for j in JOINTS:
            i = JOINT_INDEX[j]
            p = PARENTS[j]
            offs[i] = self.heads[j] if p is None else (self.heads[j] - self.heads[p])
        return offs

    def bone_vectors(self) -> Dict[str, np.ndarray]:
        """每根骨头方向（head→子 head；叶子用比例估计的 tail）。"""
        vecs: Dict[str, np.ndarray] = {}
        for j in JOINTS:
            ch = children_of(j)
            if ch:
                vecs[j] = self.heads[ch[0]] - self.heads[j]
            else:
                vecs[j] = self._leaf_tail(j) - self.heads[j]
        return vecs

    def _leaf_tail(self, j: str) -> np.ndarray:
        h = self.heads[j]
        if j == "head":
            return h + np.array([0.0, 0.10 * self.height, 0.0])
        if j.startswith("toe"):
            return h + np.array([0.0, 0.0, 0.06 * self.height])
        if j.startswith("hand"):
            sign = 1.0 if j.endswith("_l") else -1.0
            return h + np.array([sign * 0.06 * self.height, 0.0, 0.0])
        return h + np.array([0.0, 0.05 * self.height, 0.0])

    def overall_confidence(self) -> float:
        if not self.confidence:
            return 0.0
        return float(np.mean([self.confidence.get(j, 0.0) for j in JOINTS]))

    def core_failure(self, threshold: float) -> List[str]:
        return [j for j in CORE_JOINTS if self.confidence.get(j, 1.0) < threshold]

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "revision": self.revision,
            "height": self.height,
            "joints": {
                j: {
                    "semantic_id": j,
                    "parent": PARENTS[j],
                    "head": [round(float(x), 6) for x in self.heads[j]],
                    "confidence": float(self.confidence.get(j, 0.0)),
                    "source": self.source.get(j, "unknown"),
                    "locked": bool(self.locked.get(j, False)),
                } for j in JOINTS
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rig":
        heads, conf, src, locked = {}, {}, {}, {}
        for j, jd in (d.get("joints") or {}).items():
            if j not in JOINT_INDEX:
                continue
            heads[j] = np.asarray(jd["head"], dtype=np.float64)
            conf[j] = float(jd.get("confidence", 0.0))
            src[j] = jd.get("source", "unknown")
            locked[j] = bool(jd.get("locked", False))
        for j in JOINTS:  # 补齐缺失关节（容错）
            heads.setdefault(j, np.zeros(3))
            conf.setdefault(j, 0.0)
        return cls(heads=heads, height=float(d.get("height", 1.0)),
                   revision=int(d.get("revision", 0)), confidence=conf,
                   source=src, locked=locked)


def generate_rig_from_bbox(min_xyz: Sequence[float], max_xyz: Sequence[float],
                           confidence: float = 0.9) -> Rig:
    """按人体比例从目标网格包围盒生成 22 关节骨架（无 AI 的确定性路径）。"""
    mn = np.asarray(min_xyz, dtype=np.float64)
    mx = np.asarray(max_xyz, dtype=np.float64)
    height = float(mx[1] - mn[1]) or 1.0
    base_y = float(mn[1])
    cx = float((mn[0] + mx[0]) * 0.5)
    cz = float((mn[2] + mx[2]) * 0.5)
    heads: Dict[str, np.ndarray] = {}
    for j in JOINTS:
        fy, fx, fz = PROPORTIONS[j]
        heads[j] = np.array([cx + fx * height, base_y + fy * height, cz + fz * height])
    return Rig(
        heads=heads, height=height, revision=0,
        confidence={j: confidence for j in JOINTS},
        source={j: "proportion" for j in JOINTS},
    )


def apply_patch(rig: Rig, patch: dict) -> Rig:
    """把前端 PATCH 的 joints 覆盖应用到 rig（head/locked/confidence）。"""
    for j, jd in (patch or {}).items():
        if j not in JOINT_INDEX:
            continue
        if jd.get("head") is not None:
            rig.heads[j] = np.asarray(jd["head"], dtype=np.float64)
            rig.source[j] = "manual"
        if jd.get("locked") is not None:
            rig.locked[j] = bool(jd["locked"])
        if jd.get("confidence") is not None:
            rig.confidence[j] = float(jd["confidence"])
    rig.revision += 1
    return rig
