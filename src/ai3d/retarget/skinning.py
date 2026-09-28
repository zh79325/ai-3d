"""无 Bone Heat 的确定性自动蒙皮。

流程：骨段胶囊距离初始权重 → 左右软屏蔽 → 网格邻接 Laplacian 扩散平滑 →
行归一化 → 每顶点保留 ≤max_influences 个骨骼 → 输出 JOINTS_0(u16)/WEIGHTS_0(f32)。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve  # noqa: F401  (保留以备后续隐式扩散)

from .skeleton import JOINTS, JOINT_INDEX, Rig, children_of
from .settings import SkinConfig


def _bone_segments(rig: Rig) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
    """返回 [(joint_index, head, tail, radius)]。"""
    segs = []
    for j in JOINTS:
        i = JOINT_INDEX[j]
        head = np.asarray(rig.heads[j], np.float64)
        ch = children_of(j)
        tail = np.asarray(rig.heads[ch[0]], np.float64) if ch else rig._leaf_tail(j)
        length = float(np.linalg.norm(tail - head)) or (0.05 * rig.height)
        radius = max(length * 0.5, 0.02 * rig.height)
        segs.append((i, head, tail, radius))
    return segs


def _segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点集 P(N,3) 到线段 ab 的最近距离 (N,)。"""
    ab = b - a
    denom = float(ab @ ab) or 1e-8
    t = np.clip(((P - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab[None, :]
    return np.linalg.norm(P - proj, axis=1)


def _build_adjacency(n_verts: int, indices: np.ndarray) -> sparse.csr_matrix:
    """从三角面构建对称归一化邻接矩阵 A_norm（行和=1）。"""
    tri = np.asarray(indices).reshape(-1, 3).astype(np.int64)
    r = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    c = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    rows = np.concatenate([r, c])   # 对称化：每条边同时加 (a,b) 与 (b,a)
    cols = np.concatenate([c, r])
    data = np.ones(len(rows), np.float32)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0
    A.sum_duplicates()
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sparse.diags(1.0 / deg)
    return (Dinv @ A).tocsr()


def compute_skin_weights(positions: np.ndarray, indices: np.ndarray, rig: Rig,
                         cfg: SkinConfig | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (joints_u16 (N,4), weights_f32 (N,4))。"""
    cfg = cfg or SkinConfig()
    P = np.ascontiguousarray(positions, dtype=np.float64)
    n = len(P)
    nj = len(JOINTS)
    segs = _bone_segments(rig)

    center_x = float(np.asarray(rig.heads["pelvis"])[0])
    W = np.zeros((n, nj), dtype=np.float64)
    for (ji, head, tail, radius) in segs:
        d = _segment_distance(P, head, tail)
        w = np.exp(-2.0 * (d / radius) ** 2)
        # 左右软屏蔽：抑制跨中线影响
        name = JOINTS[ji]
        if name.endswith("_l") or name.endswith("_r"):
            side = 1.0 if name.endswith("_l") else -1.0
            s = side * (P[:, 0] - center_x) / (0.06 * rig.height + 1e-6)
            mask = 1.0 / (1.0 + np.exp(-s))          # sigmoid
            w *= (0.12 + 0.88 * mask)
        W[:, ji] = w

    # 行归一化（避免全零）
    rs = W.sum(1, keepdims=True)
    rs[rs < 1e-9] = 1.0
    W = W / rs

    # Laplacian 扩散平滑
    if cfg.diffusion_iters > 0 and n > 0 and len(indices) >= 3:
        A = _build_adjacency(n, indices)
        lam = float(cfg.diffusion_lambda)
        for _ in range(int(cfg.diffusion_iters)):
            W = (1.0 - lam) * W + lam * (A @ W)
        rs = W.sum(1, keepdims=True); rs[rs < 1e-9] = 1.0
        W = W / rs

    # 附件岛刚性化：装甲片/挂饰等独立连通壳若聚合权重集中于单骨，
    # 整岛 one-hot，避免刚性壳被多骨混合权重拉弯/半途滞留成碎片；
    # 分散岛（体表马赛克）保持平滑权重以维持缝连续
    if cfg.rigid_island_share > 0 and n > 0 and len(indices) >= 3:
        from scipy.sparse.csgraph import connected_components
        tri = np.asarray(indices).reshape(-1, 3).astype(np.int64)
        r = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
        c = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
        A = sparse.coo_matrix((np.ones(len(r), np.int8), (r, c)),
                              shape=(n, n)).tocsr()
        ncomp, lab = connected_components(A, directed=False)
        if ncomp > 1:
            for k in range(ncomp):
                sel = lab == k
                agg = W[sel].sum(0)
                tot = agg.sum()
                if tot <= 1e-9:
                    continue
                agg = agg / tot
                j = int(np.argmax(agg))
                if agg[j] >= float(cfg.rigid_island_share):
                    W[sel] = 0.0
                    W[sel, j] = 1.0

    # top-K 影响 + 归一化
    k = max(1, min(int(cfg.max_influences), nj))
    joints = np.zeros((n, 4), dtype=np.uint16)
    weights = np.zeros((n, 4), dtype=np.float32)
    if n:
        top = np.argpartition(-W, k - 1, axis=1)[:, :k]
        row = np.arange(n)[:, None]
        vals = W[row, top]
        order = np.argsort(-vals, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        vs = vals.sum(1, keepdims=True); vs[vs < 1e-9] = 1.0
        vals = vals / vs
        m = min(k, 4)
        joints[:, :m] = top[:, :m].astype(np.uint16)
        weights[:, :m] = vals[:, :m].astype(np.float32)
    return joints, weights
