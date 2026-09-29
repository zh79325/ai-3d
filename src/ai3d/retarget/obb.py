"""包围盒：轴对齐包围盒（AABB）数据与角点/棱线几何。

S1 轴校准 V2 用：校准完成后在**规范系（米制）**下量 AABB 写入 ``align.bbox``，与
asset.glb 同坐标系，前端直接画、六面标签即语义名（+Y up / -Y down / +Z front …）。
底面与模型最低点齐平（``foot_ik.foot_contact_height=0.12`` 等米制阈值依赖脚底在
原点附近的前提，故 canon 节点只带旋转+scale，不做中心化平移）。

V1 的「模型坐标 AABB + 六面号映射」与最小体积 OBB 已随方案 V2 删除：实测动态姿势
（飞踢）模型的最小体积 OBB 斜轴几何上正确却会把站立模型转斜 45°；六面号映射的
人工交互也被「朝向正确 / 前后相反」二选确认替代（见 :mod:`axis_norm`）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12

# 规范系六面语义标签，行序 = 面编号 1..6（+X left / -X right / +Y up / -Y down /
# +Z front / -Z back）；前端立方体标签以此为准，不得重排
FACE_LABELS: Tuple[str, ...] = ("+X left", "-X right", "+Y up",
                                "-Y down", "+Z front", "-Z back")


@dataclass
class OBB:
    """轴对齐包围盒：中心、三轴（恒为单位阵，按行）、三轴全长。"""

    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray

    def face_normals(self) -> np.ndarray:
        """六面外法向 (6,3)，行序对应面编号 1..6（规范系 ±X/±Y/±Z）。"""
        return np.stack([self.axes[0], -self.axes[0],
                         self.axes[1], -self.axes[1],
                         self.axes[2], -self.axes[2]])

    def corners(self) -> np.ndarray:
        """八个角点 (8,3)，供前端画线框。"""
        half = np.asarray(self.extents, dtype=np.float64) * 0.5
        signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        # axes 按行是轴，须每行乘自己的半长（half[:, None]），写成 axes * half
        # 会按列广播得到错误角点
        return self.center + signs @ (self.axes * half[:, None])

    def edges(self) -> List[Tuple[int, int]]:
        """十二条棱的角点索引对（配合 :meth:`corners` 画线框）。"""
        out: List[Tuple[int, int]] = []
        for a in range(8):
            for b in range(a + 1, 8):
                # 角点按 (±,±,±) 顺序生成，恰差一位符号即共棱
                if bin(a ^ b).count("1") == 1:
                    out.append((a, b))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [round(float(v), 6) for v in np.asarray(self.center, dtype=np.float64)],
            "extents": [round(float(v), 6) for v in np.asarray(self.extents, dtype=np.float64)],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OBB":
        return cls(np.asarray(d["center"], dtype=np.float64),
                   np.eye(3),
                   np.asarray(d["extents"], dtype=np.float64))


def axis_aligned_bbox(V: np.ndarray) -> OBB:
    """点集的轴对齐包围盒（``axes`` 恒为单位阵）。

    底面与点集最低点齐平。顶点不足或全部重合返回 ``None``。
    """
    V = np.asarray(V, dtype=np.float64)
    if len(V) < 4:
        logger.warning("axis_aligned_bbox：顶点数 %d < 4，跳过", len(V))
        return None
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    span = hi - lo
    if not np.all(np.isfinite(span)) or float(span.max()) <= _EPS:
        logger.warning("axis_aligned_bbox：退化点集（span=%s），跳过", span.tolist())
        return None
    return OBB(center=(lo + hi) * 0.5, axes=np.eye(3), extents=span)
