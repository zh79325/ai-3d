"""基于 libigl BBW（Bounded Biharmonic Weights）的确定性自动蒙皮。

流程：求解域焊接合并（位置近接触聚簇，导出网格不动）→ 连通分量 →
主分量（身体壳）骨段距离选 handle →（高模分量经 igl.decimate 代理）igl.bbw
求解 → 其余分量（衣服/附件）最近邻继承主分量权重（同变形不穿模）→
按簇散射回原顶点 → 退化兜底最近骨 one-hot → 行归一 + top-K →
输出 JOINTS_0(u16)/WEIGHTS_0(f32)，并附质量报告（岛统计/权重跳变/合成姿势应变）。

蒙皮与源动画无关：输入仅 rest 网格与语义骨架；合成姿势应变为动画无关自检。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .skeleton import JOINTS, JOINT_INDEX, Rig, children_of
from .settings import SkinConfig

logger = logging.getLogger(__name__)

_SYN_ANGLE = np.deg2rad(20.0)   # 合成自检姿势：每关节绕各世界轴的旋转角
_SYN_MAX_EDGES = 50000          # 应变度量的采样边上限（高模控耗时）
# 中轴链：躯干与头颈由脊柱驱动，皮肤归属按沿中轴的环形分带（高度区间）而非
# “离哪根骨最近”；顺序即 pelvis→head 的解剖链，不得重排
_AXIS_CHAIN = ("pelvis", "spine_01", "spine_02", "chest", "neck", "head")


def _bone_segments(rig: Rig) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """返回 [(joint_index, head, tail)]。"""
    segs = []
    for j in JOINTS:
        i = JOINT_INDEX[j]
        head = np.asarray(rig.heads[j], np.float64)
        ch = children_of(j)
        tail = np.asarray(rig.heads[ch[0]], np.float64) if ch else rig._leaf_tail(j)
        segs.append((i, head, tail))
    return segs


def _segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点集 P(N,3) 到线段 ab 的最近距离 (N,)。"""
    ab = b - a
    denom = float(ab @ ab) or 1e-8
    t = np.clip(((P - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab[None, :]
    return np.linalg.norm(P - proj, axis=1)


def _all_segment_distances(P: np.ndarray, segs) -> np.ndarray:
    """(N, nj) 每顶点到各骨段距离。"""
    return np.stack([_segment_distance(P, h, t) for (_, h, t) in segs], axis=1)


def _weld_solve_domain(P: np.ndarray, tri: np.ndarray, eps: float):
    """位置近接触聚簇合并，返回 (V', F', cluster_of)。eps<=0 为恒等。

    仅求解域合并：簇内原顶点共享同一行权重；导出网格与法线/UV 不动。
    """
    n = len(P)
    if eps <= 0 or len(tri) == 0:
        return P, tri, np.arange(n, dtype=np.int64)
    key = np.round(P / eps).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.astype(np.int64)
    np_ = int(inv.max()) + 1
    rep = np.empty(np_, np.int64)
    rep[inv[::-1]] = np.arange(n)[::-1]      # 簇内最早顶点作代表
    Fp = inv[tri]
    keep = (Fp[:, 0] != Fp[:, 1]) & (Fp[:, 1] != Fp[:, 2]) & (Fp[:, 0] != Fp[:, 2])
    return P[rep], Fp[keep], inv


def _components(F: np.ndarray, n: int) -> Tuple[int, np.ndarray]:
    """三角边连通分量。"""
    if len(F) == 0:
        return n, np.arange(n, dtype=np.int32)
    r = F.reshape(-1)
    c = np.roll(F, -1, axis=1).reshape(-1)
    A = sparse.coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    return connected_components(A, directed=False)


def _mean_edge_len(Vk: np.ndarray, Fk: np.ndarray) -> float:
    """网格边长中位数（减面网格分布偏斜，中位数比均值稳）。"""
    if len(Fk) == 0 or len(Vk) < 2:
        return 0.0
    e = np.concatenate([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]], axis=0)
    return float(np.median(np.linalg.norm(Vk[e[:, 0]] - Vk[e[:, 1]], axis=1)))


def _axis_claim(Vk: np.ndarray, segs, D: np.ndarray, axis_idx: List[int],
                gap: float) -> Dict[int, int]:
    """中轴链（脊柱+头颈）沿中轴环形分带归属，返回 {顶点: 关节索引}。

    躯干/头颈皮肤按沿中轴的高度区间整块归属对应骨（环形分界面保证前后左右
    一致），横向由“离中轴不远离任何四肢骨”限定，肩部/四肢交给四肢骨。
    不能用“离哪根骨最近”或骨段中段带：脊柱骨段短（实测 4~9cm），中段带仅
    2~4cm 且在与锁骨竞争时大量败给 clear 判据，躯干会被锁骨/骨盆瓜分（实测
    上背 clavicle 0.86、chest 0.06，表现为背部随肩臂拉扯扭曲）。
    head 分界面下探半个头骨段长以覆盖下巴（下巴属头骨）；不用等距面（head/neck
    两段共享端点，下巴到两段距离几乎相等、归属由列序决定）也不用前后半空间
    （会把颈部前后劈开）；分界处留 gap 窄带由 BBW 谐波过渡。
    """
    a0 = segs[axis_idx[0]][1]                        # pelvis 关节
    u = segs[axis_idx[-1]][1] - a0                    # pelvis→head 关节
    lu = float(np.linalg.norm(u)) or 1e-9
    u = u / lu
    s = (Vk - a0) @ u                                 # 沿中轴高度坐标
    bnd = np.array([(segs[j][1] - a0) @ u for j in axis_idx])
    len_head = float(np.linalg.norm(segs[axis_idx[-1]][2] - segs[axis_idx[-1]][1])) or 1e-9
    bnd[-1] -= 0.5 * len_head                         # head 下探盖下巴
    d_axis = D[:, axis_idx].min(1)
    limb = [j for j in range(D.shape[1]) if j not in set(axis_idx)]
    d_limb = D[:, limb].min(1) if limb else np.full(len(Vk), np.inf)
    inside = d_axis <= d_limb                         # 躯干/头颈区（非肩肢）
    out: Dict[int, int] = {}
    n_ax = len(axis_idx)
    for i, ji in enumerate(axis_idx):
        lo_b = bnd[i] if i > 0 else -np.inf
        hi_b = bnd[i + 1] if i + 1 < n_ax else np.inf
        span = hi_b - lo_b
        # 过渡带不得超过环带长的 1/3，否则短环带（实测 chest 段仅 6.5cm）被上下
        # gap 挤空，自由区过大时 head 谐波会向躯干扩散（实测胸前 head 0.31）；
        # span 为负（head 下探越过 neck 关节）时自然得出空区间，该骨被跳过
        g = min(gap, 0.3 * span) if np.isfinite(span) else gap
        m = inside & (s >= (lo_b + g if i > 0 else -np.inf)) \
            & (s < (hi_b - g if i + 1 < n_ax else np.inf))
        for v in np.nonzero(m)[0]:
            out[int(v)] = int(ji)
    return out


def _component_handles(Vk: np.ndarray, Fk: np.ndarray, segs, cfg: SkinConfig):
    """按骨段距离选 handle 骨，返回 [(约束顶点数组, 关节索引)]。

    H = d_j <= d_min + handle_margin*分量对角 的骨（上限 max_handles，中轴链必含）。
    约束顶点分两类：
    ①中轴链（_AXIS_CHAIN）由 _axis_claim 沿中轴环形分带整块归属，优先锁定；
    ②四肢骨：末端骨（手/脚）取骨段外侧延伸的整块刚性带（t∈[0.2,1.35]）免 clear，
    其余取骨段中段带（t∈[0.3,0.7]）+ 明确胜出（d1 <= 0.8*d2），胯/腋等同级骨
    等距区留白由 BBW 谐波混合，避免约束带相邻硬碰硬产生权重断崖。
    无归属区的骨直接跳过（孤立硬点会制造断崖）；全部骨皆空时退化为质心最近骨单点。
    """
    D = _all_segment_distances(Vk, segs)                 # (nv, nj)
    dmin = D.min(axis=0)
    diag = float(np.linalg.norm(Vk.max(0) - Vk.min(0))) if len(Vk) > 1 else 0.0
    thr = float(dmin.min()) + float(cfg.handle_margin) * diag
    order = np.argsort(dmin)
    picked = [int(j) for j in order if dmin[j] <= thr][: int(cfg.max_handles)]
    axis_idx = [JOINT_INDEX[k] for k in _AXIS_CHAIN
                if JOINT_INDEX[k] < len(segs) and JOINT_INDEX[k] not in picked]
    picked += axis_idx                                   # 中轴链不得被 max_handles 截断
    axis_set = {JOINT_INDEX[k] for k in _AXIS_CHAIN if JOINT_INDEX[k] < len(segs)}
    axis_set_max = max(axis_set) if axis_set else 0
    if not picked:
        picked = [int(order[0])]
    Dp = D[:, picked]                                    # (nv, ns)
    if len(picked) >= 2:
        srt = np.sort(Dp, axis=1)
        clear = srt[:, 0] <= 0.8 * srt[:, 1]
        nearest_sel = np.argmin(Dp, axis=1)
    else:
        clear = np.ones(len(Vk), bool)
        nearest_sel = np.zeros(len(Vk), np.int64)
    band_r = dmin + 0.05 * diag                          # 中段带的贴骨距离上限
    all_true = np.ones(len(Vk), bool)
    col_of = {ji: c for c, ji in enumerate(picked)}
    claim: Dict[int, Tuple[float, int, bool]] = {}       # 顶点 -> (距离, 列, 中轴锁定)
    # 分界过渡带按网格边长自适应：窄于边长时环带两侧顶点直接相邻，两个硬 1.0
    # 行形成权重断崖（实测 pelvis 1.0 邻接 spine_01 1.0、weight_jump max 2.0）
    lu = float(np.linalg.norm(segs[axis_set_max][1] - segs[min(axis_set)][1])) or 1e-9
    gap = max(2.0 * _mean_edge_len(Vk, Fk), 0.02 * lu)
    for v, ji in _axis_claim(Vk, segs, D, sorted(axis_set), gap).items():
        c = col_of.get(int(ji))
        if c is not None:
            claim[v] = (0.0, c, True)
    for col, ji in enumerate(picked):
        if ji in axis_set:                               # 中轴骨已由环形分带处理
            continue
        a, b = segs[ji][1], segs[ji][2]
        ab = b - a
        l2 = float(ab @ ab)
        t = ((Vk - a) @ ab) / l2 if l2 > 1e-12 else np.zeros(len(Vk))
        if children_of(JOINTS[ji]):                      # 非末端骨：中段带 + clear
            zone, rad, ok_clear = (t >= 0.3) & (t <= 0.7) & (nearest_sel == col), band_r[ji], clear
        else:                                            # 末端骨：整块刚性带，免 clear
            zone = (t >= 0.2) & (t <= 1.35) & (nearest_sel == col)
            rad, ok_clear = band_r[ji] + 0.03 * diag, all_true
        cand = np.nonzero(zone & (D[:, ji] <= rad) & ok_clear)[0]
        if len(cand) == 0:
            # 该骨在本分量无归属区：不作 handle。切勿退化为“离骨段中点最近的单个
            # 顶点”，孤立硬点会与邻骨硬带相邻形成权重断崖（实测 weight_jump max 2.0）；
            # 该骨旋转经 FK 累积仍作用于子骨
            continue
        for v in cand:
            v = int(v)
            if v in claim and claim[v][2]:                # 中轴已锁定，不得抢占
                continue
            d = float(D[v, ji])
            if v not in claim or d < claim[v][0]:
                claim[v] = (d, col, False)
    if not claim:                                    # 极端退化：分量质心最近骨单点 handle
        j0 = int(np.argmin(_all_segment_distances(Vk.mean(0, keepdims=True), segs)[0]))
        return [(np.array([int(np.argmin(D[:, j0]))], np.int64), j0)]
    by_col: Dict[int, List[int]] = {}
    for v, (_d, col, _lk) in claim.items():
        by_col.setdefault(col, []).append(v)
    return [(np.array(sorted(vs), np.int64), picked[col])
            for col, vs in sorted(by_col.items())]


def _bbw_component(Vk: np.ndarray, Fk: np.ndarray, handles, cfg: SkinConfig):
    """单分量 BBW 求解，返回 (Wk (nv,#handles), proxy_used, transferred)。

    超 bbw_max_verts 时 igl.decimate 坍缩代理求解 + cKDTree 最近邻传回。
    解为未归一化权重，负值 clamp 仅作保险（BBW 有界保证非负）。
    """
    import igl  # 延迟导入：libigl 缺失时在求解点报错而非导入期
    V = np.ascontiguousarray(Vk, np.float64)
    F = np.ascontiguousarray(Fk, np.int64)
    b = np.concatenate([h[0] for h in handles]).astype(np.int64)
    bc = np.zeros((len(b), len(handles)))
    off = 0
    for col, (verts, _ji) in enumerate(handles):
        bc[off:off + len(verts), col] = 1.0
        off += len(verts)
    proxy_used = False
    transferred = 0
    if len(V) > int(cfg.bbw_max_verts):
        max_faces = max(64, int(cfg.bbw_max_verts) // 2)
        U, G = igl.decimate(V, F, max_faces)[:2]
        U = np.asarray(U, np.float64)
        G = np.asarray(G, np.int64)
        _, hidx = cKDTree(U).query(V[b])
        b2: List[int] = []
        bc2: List[np.ndarray] = []
        seen: Dict[int, int] = {}
        for i, hv in enumerate(hidx):
            hv = int(hv)
            if hv in seen:
                continue
            seen[hv] = len(b2)
            b2.append(hv)
            bc2.append(bc[i])
        if not b2:
            raise RuntimeError("decimate 代理丢失全部 handle")
        Wp = igl.bbw(U, G, np.array(b2, np.int64), np.asarray(bc2, np.float64),
                     partition_unity=True)
        _, idx = cKDTree(U).query(V)
        Wk = np.asarray(Wp, np.float64)[idx]
        proxy_used = True
        transferred = len(V)
    else:
        Wk = np.asarray(igl.bbw(V, F, b, bc, partition_unity=True), np.float64)
    return np.clip(Wk, 0.0, None), proxy_used, transferred


def _axis_rot(ax: int, ang: float) -> np.ndarray:
    R = np.eye(3)
    c, s = np.cos(ang), np.sin(ang)
    i, j = (ax + 1) % 3, (ax + 2) % 3
    R[i, i], R[j, j] = c, c
    R[i, j], R[j, i] = -s, s
    return R


def _topk(W: np.ndarray, cfg: SkinConfig, nj: int) -> Tuple[np.ndarray, np.ndarray]:
    """每顶点保留 ≤max_influences 个骨骼，输出 (N,4) u16 / (N,4) f32。"""
    n = len(W)
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
        vs = vals.sum(1, keepdims=True)
        vs[vs < 1e-9] = 1.0
        vals = vals / vs
        m = min(k, 4)
        joints[:, :m] = top[:, :m].astype(np.uint16)
        weights[:, :m] = vals[:, :m].astype(np.float32)
    return joints, weights


def _synthetic_strain(P: np.ndarray, edges: np.ndarray, joints: np.ndarray,
                      weights: np.ndarray, rig: Rig) -> Dict[str, float]:
    """合成姿势应变：每关节绕 3 世界轴各旋转 20°，量采样边长变化率（动画无关自检）。"""
    if len(edges) == 0:
        return {"med": 0.0, "max": 0.0, "gt20pct": 0, "edges": 0}
    e0, e1 = edges[:, 0], edges[:, 1]
    L0 = np.linalg.norm(P[e0] - P[e1], axis=1)
    ok = L0 > 1e-9
    e0, e1, L0 = e0[ok], e1[ok], L0[ok]
    H = np.asarray([rig.heads[j] for j in JOINTS], np.float64)
    W4 = weights.astype(np.float64)
    J4 = joints.astype(np.int64)
    Hs = H[J4]                                   # (n,4,3)
    offs = P[:, None, :] - Hs                    # 相对各影响骨 head 的 rest 偏移
    worst = np.zeros(len(e0))
    for ji in range(len(JOINTS)):
        mask = (J4 == ji)                        # (n,4) 本姿势受旋转的影响槽
        if not mask.any():
            continue
        for ax in range(3):
            R = _axis_rot(ax, _SYN_ANGLE)
            rot = np.where(mask[..., None], offs @ R.T, offs)
            Vp = (W4[..., None] * (rot + Hs)).sum(1)
            ratio = np.linalg.norm(Vp[e0] - Vp[e1], axis=1) / L0
            np.maximum(worst, np.abs(ratio - 1.0), out=worst)
    return {"med": round(float(np.median(worst)), 4),
            "max": round(float(worst.max()), 4),
            "gt20pct": int((worst > 0.2).sum()),
            "edges": int(len(worst))}


def _skin_report(P, tri, joints, weights, W, rig, ncomp, fallback,
                 proxy_comps, transferred, island_stats, n_clusters,
                 elapsed) -> Dict:
    zero_rows = int((weights.sum(1) == 0).sum())
    if len(tri):
        r = tri[:, [0, 1, 2, 0]].reshape(-1)
        c = tri[:, [1, 2, 0, 1]].reshape(-1)
        jump = np.abs(W[r] - W[c]).sum(1)
        jump_stat = {"mean": round(float(jump.mean()), 4),
                     "max": round(float(jump.max()), 4)}
        pairs = np.sort(np.stack([r, c], 1), axis=1)
        edges = np.unique(pairs, axis=0)
    else:
        jump_stat = {"mean": 0.0, "max": 0.0}
        edges = np.zeros((0, 2), np.int64)
    if len(edges) > _SYN_MAX_EDGES:
        sel = np.random.default_rng(0).choice(len(edges), _SYN_MAX_EDGES, replace=False)
        edges = edges[sel]
    return {
        "zero_rows": zero_rows,
        "island_stats": island_stats,
        "weld": {"clusters": int(n_clusters), "merged_verts": int(len(P) - n_clusters)},
        "components": {"count": int(ncomp), "fallback": int(fallback),
                       "proxy": int(proxy_comps), "transferred": int(transferred)},
        "weight_jump": jump_stat,
        "strain": _synthetic_strain(P, edges, joints, weights, rig),
        "seconds": round(elapsed, 2),
    }


def compute_skin_weights(positions: np.ndarray, indices: np.ndarray, rig: Rig,
                         cfg: SkinConfig | None = None
                         ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """返回 (joints_u16 (N,4), weights_f32 (N,4), report)。"""
    cfg = cfg or SkinConfig()
    t0 = time.time()
    P = np.ascontiguousarray(positions, dtype=np.float64)
    n = len(P)
    nj = len(JOINTS)
    tri = (np.asarray(indices).reshape(-1, 3).astype(np.int64)
           if len(indices) else np.zeros((0, 3), np.int64))
    segs = _bone_segments(rig)

    # 原网格岛统计（输入质量诊断：正常资产应有主导主岛，马赛克 AI 网格无）
    n_orig, lab_orig = _components(tri, n)
    sizes = np.bincount(lab_orig, minlength=n_orig)
    island_stats = {"components": int(n_orig),
                    "largest_share": round(float(sizes.max() / max(n, 1)), 4),
                    "tiny_lt16": int((sizes < 16).sum())}

    W = np.zeros((n, nj), dtype=np.float64)
    Vp, Fp, cluster_of = _weld_solve_domain(P, tri, float(cfg.weld_epsilon))
    ncomp, lab = _components(Fp, len(Vp))
    comp_of_orig = lab[cluster_of]
    fallback = proxy_comps = transferred = 0
    primary = 0
    if ncomp > 1:
        # 主分量 = 多数关节最近顶点所在分量（身体壳）；其余分量（衣服/附件）
        # 不独立求解，最近邻继承主分量权重，与身体同变形避免穿模
        votes = np.bincount(lab[np.argmin(_all_segment_distances(Vp, segs), axis=0)],
                            minlength=ncomp)
        primary = int(np.argmax(votes))
    prim_V = prim_W = None
    for k in [primary] + [q for q in range(ncomp) if q != primary]:
        sk = np.nonzero(lab == k)[0]
        ov = np.nonzero(comp_of_orig == k)[0]
        if len(ov) == 0:
            continue
        loc = -np.ones(len(Vp), np.int64)
        loc[sk] = np.arange(len(sk))
        if k != primary:
            if prim_W is None:                           # 主分量求解失败落兜底
                fallback += 1
                W[ov, np.argmin(_all_segment_distances(P[ov], segs), axis=1)] = 1.0
                continue
            _, idx = cKDTree(prim_V).query(Vp[sk], k=min(8, len(prim_V)))
            idx = np.atleast_2d(idx)
            dd = 1.0 / (np.linalg.norm(
                Vp[sk][:, None, :] - prim_V[idx], axis=2) + 1e-4)
            dd /= dd.sum(1, keepdims=True)
            W[ov] = np.einsum("nk,nkj->nj", dd, prim_W[idx])[loc[cluster_of[ov]]]
            transferred += len(sk)
            continue
        Vk = Vp[sk]
        Fk = loc[Fp[np.nonzero(lab[Fp[:, 0]] == k)[0]]] if len(Fp) else Fp
        Wk = None
        handles = _component_handles(Vk, Fk, segs, cfg) if len(sk) >= 4 else []
        if handles:
            try:
                Wk, pu, tr = _bbw_component(Vk, Fk, handles, cfg)
                proxy_comps += int(pu)
                transferred += tr
            except Exception as exc:  # noqa: BLE001 求解失败一律落最近骨兜底
                logger.warning("分量 %d（%d 顶点）BBW 求解失败，落最近骨兜底：%s",
                               k, len(sk), exc)
        if Wk is None or Wk.shape[0] != len(sk) or (Wk.sum(1) < 1e-6).any():
            fallback += 1
            W[ov, np.argmin(_all_segment_distances(P[ov], segs), axis=1)] = 1.0
            continue
        rs = Wk.sum(1, keepdims=True)
        rs[rs < 1e-9] = 1.0
        Wk /= rs
        col_to_joint = np.array([h[1] for h in handles], np.int64)
        Wc = np.zeros((len(sk), nj))
        Wc[:, col_to_joint] = Wk
        prim_V, prim_W = Vk, Wc
        W[ov] = Wc[loc[cluster_of[ov]]]

    rs = W.sum(1, keepdims=True)
    rs[rs < 1e-9] = 1.0
    W /= rs
    joints, weights = _topk(W, cfg, nj)
    report = _skin_report(P, tri, joints, weights, W, rig, ncomp, fallback,
                          proxy_comps, transferred, island_stats, len(Vp),
                          time.time() - t0)
    return joints, weights, report
