"""最小体积有向包围盒（OBB）与面编号指派。

S1 导入矫正用：OBB 给出模型的最紧凑三轴框架，六个外表面按固定约定编号
1..6（``1=+axis0, 2=-axis0, 3=+axis1, 4=-axis1, 5=+axis2, 6=-axis2``），
前端立方体标签与人工「面 → 语义轴」映射表都以此编号为准。

轴向语义（哪个面是 up / forward）不由 OBB 几何决定：实测 A-pose 人形
``extents=[0.93, 0.98, 0.22]``，臂展已接近身高，完全水平张臂的 T-pose 会让
最长轴落到 X，故「最长轴 = up」不成立。语义指派走 :mod:`axis_norm` 的人形
先验（有骨架 pelvis→head；纯网格脚端双簇间隙），再由 :func:`assign_faces`
把先验方向吸附到最近的 OBB 面。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

logger = logging.getLogger(__name__)

# 面编号 1..6 → (轴索引, 符号)；与前端立方体标签一致，不得重排
FACE_AXES: Tuple[Tuple[int, float], ...] = (
    (0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0))
FACE_LABELS: Tuple[str, ...] = ("u+", "u-", "v+", "v-", "w+", "w-")
# 面编号的对偶面（1↔2 / 3↔4 / 5↔6），指派 forward 时须排除 up 的对偶面
FACE_OPPOSITE: Tuple[int, ...] = (2, 1, 4, 3, 6, 5)

DEFAULT_MAX_PTS = 6000   # 凸包采样上限：实测 12000 点凸包仅 254 面 / 0.001s
_MAX_RECT_PTS = 96       # 单面旋转卡尺的二维凸包点数上限（控总耗时）
_EPS = 1e-12


@dataclass
class OBB:
    """有向包围盒：中心、三轴（模型坐标表示的单位正交基，按行）、三轴全长。"""

    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray

    def face_normals(self) -> np.ndarray:
        """六面外法向 (6,3)，行序对应面编号 1..6。"""
        return np.stack([s * self.axes[i] for i, s in FACE_AXES])

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
            "center": np.asarray(self.center, dtype=np.float64).tolist(),
            "axes": np.asarray(self.axes, dtype=np.float64).tolist(),
            "extents": np.asarray(self.extents, dtype=np.float64).tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OBB":
        return cls(np.asarray(d["center"], dtype=np.float64),
                   np.asarray(d["axes"], dtype=np.float64),
                   np.asarray(d["extents"], dtype=np.float64))


def convex_hull_sample(V: np.ndarray, max_pts: int = DEFAULT_MAX_PTS,
                       seed: int = 0) -> np.ndarray:
    """顶点降采样：凸包只由外轮廓决定，随机采样不损精度而面数大幅下降。"""
    V = np.asarray(V, dtype=np.float64)
    if len(V) <= max_pts:
        return V
    idx = np.random.default_rng(seed).choice(len(V), max_pts, replace=False)
    return V[idx]


def _min_area_rect(P: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """二维点集的最小面积包围矩形（旋转卡尺）。

    返回 ``(面积, 单位轴a, 单位轴b, [a_min, a_max, b_min, b_max])``。
    """
    hull = ConvexHull(P)
    hp = P[hull.vertices]
    if len(hp) > _MAX_RECT_PTS:
        hp = hp[np.linspace(0, len(hp), _MAX_RECT_PTS, endpoint=False).astype(int)]
    n = len(hp)
    best = (np.inf, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.zeros(4))
    for i in range(n):
        e = hp[(i + 1) % n] - hp[i]
        le = float(np.linalg.norm(e))
        if le < _EPS:
            continue
        a = e / le
        b = np.array([-a[1], a[0]])
        q = hp @ np.stack([a, b]).T                      # (n,2) 旋转到 ab 系
        lo = q.min(axis=0)
        hi = q.max(axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if area < best[0]:
            best = (area, a, b, np.array([lo[0], hi[0], lo[1], hi[1]]))
    return best


def min_volume_obb(V: np.ndarray, max_pts: int = DEFAULT_MAX_PTS) -> Optional[OBB]:
    """最小体积有向包围盒。

    依据「最小体积盒必有一个面贴合点集凸包的某个面」，只需遍历凸包面：以面法向
    为第一轴，把点投影到该面平面求最小面积矩形得另两轴，体积 = 面积 × 法向跨度。
    顶点不足或全部共面时返回 ``None``。
    """
    V = np.asarray(V, dtype=np.float64)
    if len(V) < 4:
        logger.warning("min_volume_obb：顶点数 %d < 4，跳过", len(V))
        return None
    P = convex_hull_sample(V, max_pts)
    try:
        hull = ConvexHull(P)
    except QhullError as exc:            # 共面/重合等退化点集
        logger.warning("min_volume_obb：凸包求解失败（%s），跳过", exc)
        return None
    c0 = P.mean(axis=0)
    best: Optional[Tuple[float, OBB]] = None
    for tri in hull.simplices:
        p0, p1, p2 = P[tri[0]], P[tri[1]], P[tri[2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(nrm))
        if ln < _EPS:
            continue
        a0 = nrm / ln                                     # 候选第一轴 = 面法向
        ref = np.array([1.0, 0.0, 0.0]) if abs(a0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        a1 = ref - a0 * float(ref @ a0)
        n1 = float(np.linalg.norm(a1))
        if n1 < _EPS:
            continue
        a1 /= n1
        a2 = np.cross(a0, a1)
        q = (P - c0) @ np.stack([a1, a2]).T               # (m,2) 平面投影
        area, u2, v2, rect = _min_area_rect(q)
        if not np.isfinite(area):
            continue
        ax1 = u2[0] * a1 + u2[1] * a2
        ax2 = v2[0] * a1 + v2[1] * a2
        lo_b, hi_b = rect[2], rect[3]
        axes = np.stack([a0, ax1, ax2])
        if float(np.linalg.det(axes)) < 0.0:              # 规范成右手基，稳定面编号
            ax2, lo_b, hi_b = -ax2, -hi_b, -lo_b
            axes = np.stack([a0, ax1, ax2])
        d = (P - c0) @ a0
        lo_a, hi_a = float(d.min()), float(d.max())
        center = c0 + ((lo_a + hi_a) * 0.5) * a0 \
            + ((rect[0] + rect[1]) * 0.5) * ax1 + ((lo_b + hi_b) * 0.5) * ax2
        extents = np.array([hi_a - lo_a, rect[1] - rect[0], hi_b - lo_b])
        vol = float(abs(extents[0] * extents[1] * extents[2]))
        if best is None or vol < best[0]:
            best = (vol, OBB(center=center, axes=axes, extents=extents))
    if best is None:
        return None
    return best[1]


def assign_faces(obb: OBB, up_dir: Optional[np.ndarray],
                 forward_dir: Optional[np.ndarray]) -> Dict[str, Any]:
    """把人形先验方向吸附到 OBB 面，给出自动指派的面编号（1..6）。

    ``up_dir`` / ``forward_dir`` 为模型坐标下的未吸附方向（可为 ``None``）。
    up 优先，forward 在排除 up 及其对偶面后取剩余面中最贴合者，left 由右手系
    ``cross(up, forward)`` 导出。两者皆缺时按最长轴兜底并记入 ``notes``。
    """
    normals = obb.face_normals()
    notes: List[str] = []
    ext = np.asarray(obb.extents, dtype=np.float64)
    up_face: Optional[int] = None
    if up_dir is not None and float(np.linalg.norm(up_dir)) > _EPS:
        u = np.asarray(up_dir, dtype=np.float64)
        u = u / float(np.linalg.norm(u))
        up_face = int(np.argmax(normals @ u)) + 1
    else:
        # 无任何先验：只能按最长轴兜底，对张臂 T-pose 会判错，须人工核对
        up_face = int(2 * int(np.argmax(ext)) + 1)
        notes.append("缺少 up 先验，按最长轴兜底指派，请在阶段一核对顶面")
    fwd_face: Optional[int] = None
    if forward_dir is not None and float(np.linalg.norm(forward_dir)) > _EPS:
        f = np.asarray(forward_dir, dtype=np.float64)
        f = f / float(np.linalg.norm(f))
        cand = [i for i in range(6)
                if (i + 1) != up_face and (i + 1) != FACE_OPPOSITE[up_face - 1]]
        scores = normals[cand] @ f
        fwd_face = cand[int(np.argmax(scores))] + 1
    else:
        cand = [i for i in range(6)
                if (i + 1) != up_face and (i + 1) != FACE_OPPOSITE[up_face - 1]]
        # 无 forward 先验：取剩余面中厚度最小者（人形前后最薄）
        axis_of = [FACE_AXES[i][0] for i in cand]
        fwd_face = cand[int(np.argmin([ext[a] for a in axis_of]))] + 1
        notes.append("缺少 forward 先验，按最薄面兜底指派，请在阶段一核对前面")
    up_n = normals[up_face - 1]
    fwd_n = normals[fwd_face - 1]
    left_n = np.cross(up_n, fwd_n)
    left_face = int(np.argmax(normals @ left_n)) + 1
    return {"up_face": up_face, "forward_face": fwd_face,
            "left_face": left_face, "notes": notes}
