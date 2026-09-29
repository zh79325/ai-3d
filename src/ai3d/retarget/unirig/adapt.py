"""UniRig 预测结果 → 本项目 22 语义关节骨架 + 蒙皮（计划 §4，决策定死）。

输入 ``unirig_raw.npz``：joints(J,3 规范系米制) / parents / names / skin(Ns,J)
（采样顶点上的权重）/ sampled_vertices(Ns,3) / vertices(N,3) / faces(F,3)。

算法（全部几何规则，无学习成分）：
1. **pelvis**：拥有 ≥2 条「首骨方向 y<-0.7」子链（腿）与 ≥1 条「y>0.7」子链（脊柱）
   的节点；找不到 → 非人形，抛错（绑定落 FAILED，提示改用比例骨架/旧链路）。
2. **腿链**：按序映射 upper_leg/lower_leg/foot/toe（沿链折线按比例重采样）；
   链不足时缺失的远端关节按 ``skeleton.PROPORTIONS`` 相对身高合成。
3. **脊柱链**：pelvis→最高叶节点折线，按 PROPORTIONS 高度（0.58/0.66/0.74/0.84/0.92H）
   等高度重采样得 spine_01/spine_02/chest/neck/head（与节点数无关）。
4. **臂链**：chest 节点上「首骨 |dir·x|>0.5」的两条子链，按 PROPORTIONS 横向偏移
   （0.05/0.18/0.36/0.52 × 链长）重采样得 clavicle/upper_arm/lower_arm/hand；
   左右按 canonical +X=left 的符号判定；缺臂按 PROPORTIONS 合成（低置信度）。
5. **权重聚合**：语义关节 s 对应其重采样区间覆盖的 UniRig 骨集合 B(s)，
   ``W[:,s] = Σ_{j∈B(s)} skin[:,j]``；再最近邻回传到全分辨率网格，复用
   ``skinning._topk`` / ``_skin_report`` 产出与旧链路同契约的三产物。

这是「学习模型预测权重的骨合并聚合」，非自研几何蒙皮；旧链路与 /reskin 仍严格走
libigl BBW（UniRig 指令下的显式例外）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .. import stages
from ..settings import RetargetSettings, get_settings
from ..skeleton import JOINTS, JOINT_INDEX, PARENTS, PROPORTIONS, Rig
from ..skinning import (
    _all_segment_distances,
    _bone_segments,
    _components,
    _skin_report,
    _topk,
    _weld_solve_domain,
)

logger = logging.getLogger(__name__)

# 链首骨方向阈值（单位向量）：腿明显向下；脊柱用「净上升>0.2H且
# 横向位移<0.25H」判定（见 _classify_child_chains，容忍髋部短骨近水平的真骨架）。
# 臂不用首骨方向判：真人形锁骨上扬（首骨 dir≈(±0.37,+0.93)），改用整条子链
# 的**净横向伸展** > _ARM_LATERAL_MIN*H 判定（臂外展到 ~0.5H，脊柱续段横向≈0）。
_LEG_DIR_Y = -0.7
_ARM_LATERAL_MIN = 0.15

# 语义关节在链折线上的重采样位置（腿/脊柱用身高比例，臂用链长比例）
_LEG_CUTS = (("upper_leg", 0.48), ("lower_leg", 0.27), ("foot", 0.06), ("toe", 0.02))
_SPINE_CUTS = (("spine_01", 0.58), ("spine_02", 0.66), ("chest", 0.74),
               ("neck", 0.84), ("head", 0.92))
_ARM_CUTS = (("clavicle", 0.05), ("upper_arm", 0.18), ("lower_arm", 0.36), ("hand", 0.52))

# 身高与 S1 校准值的允许偏差（超出则按对齐身高缩放关节）
_HEIGHT_TOL = 0.10


class UniRigAdaptError(RuntimeError):
    """UniRig 骨架无法映射到 22 语义关节（非人形 / 标注失败）。"""


# --------------------------------------------------------------------------- #
# 树工具
# --------------------------------------------------------------------------- #
def _children_map(parents: np.ndarray) -> Dict[int, List[int]]:
    kids: Dict[int, List[int]] = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p is not None and int(p) >= 0 and int(p) != i:
            kids.setdefault(int(p), []).append(i)
    return kids


def _chain_down(root: int, kids: Dict[int, List[int]], joints: np.ndarray,
                used: Optional[set] = None) -> List[int]:
    """从 root 沿「最向下」的子节点走到叶子，返回节点链（含 root）。"""
    chain = [root]
    cur = root
    while True:
        cands = [c for c in kids.get(cur, []) if used is None or c not in used]
        if not cands:
            return chain
        cur = min(cands, key=lambda c: joints[c][1])
        chain.append(cur)
        if used is not None:
            used.add(cur)


def _chain_highest(root: int, kids: Dict[int, List[int]], joints: np.ndarray,
                   used: Optional[set] = None) -> List[int]:
    """从 root 沿「最高叶节点」方向走（DFS 取终点 y 最大的分支），返回节点链。"""
    def _best_leaf(n: int) -> Tuple[float, List[int]]:
        cands = [c for c in kids.get(n, []) if used is None or c not in used]
        if not cands:
            return float(joints[n][1]), [n]
        best: Optional[Tuple[float, List[int]]] = None
        for c in cands:
            h, path = _best_leaf(c)
            if best is None or h > best[0]:
                best = (h, path)
        assert best is not None
        return best[0], [n] + best[1]

    return _best_leaf(root)[1]


def _dir(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else d


def _resample_polyline(points: Sequence[np.ndarray], cuts: Sequence[Tuple[str, float]],
                       frac_of: str, scale: float, base_y: float = 0.0
                       ) -> Tuple[Dict[str, np.ndarray], float]:
    """沿折线按目标位置重采样语义关节点。

    ``frac_of='height'``：目标 y = base_y + frac*scale（腿/脊柱；base_y 是**模型
    脚底**而非链首点 y，PROPORTIONS 的身高分数都以脚底为基准）；
    ``frac_of='length'``：目标弧长 = frac*scale（臂，PROPORTIONS 的 x_frac 即链长比例）。
    目标超出折线范围时钳到链端点（腿链缺 foot/toe 节点时由调用方另合成）。
    返回 ``({semantic: (3,)}, 总弧长)``。
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    seg = np.linalg.norm(np.diff(np.stack(pts), axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    out: Dict[str, np.ndarray] = {}
    if total < 1e-9:
        for name, _ in cuts:
            out[name] = pts[0].copy()
        return out, total
    if frac_of == "height":
        for name, frac in cuts:
            out[name] = _point_at_height(pts, cum, base_y + frac * scale)
    else:
        for name, frac in cuts:
            out[name] = _point_at_arclen(pts, cum, frac * scale)
    return out, total


def _point_at_arclen(pts, cum: np.ndarray, s_target: float) -> np.ndarray:
    s_target = float(np.clip(s_target, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s_target, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    span = cum[i + 1] - cum[i]
    t = (s_target - cum[i]) / span if span > 1e-12 else 0.0
    return pts[i] + t * (pts[i + 1] - pts[i])


def _point_at_height(pts, cum: np.ndarray, target_y: float) -> np.ndarray:
    """折线上 y 首次到达 target_y 的点（超出范围钳到 y 最接近的链端点）。"""
    ys = np.array([p[1] for p in pts])
    if target_y >= ys.max():
        return pts[int(np.argmax(ys))].copy()
    if target_y <= ys.min():
        return pts[int(np.argmin(ys))].copy()
    for i in range(len(pts) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if (y0 - target_y) * (y1 - target_y) <= 0 and abs(y1 - y0) > 1e-12:
            t = (target_y - y0) / (y1 - y0)
            return pts[i] + t * (pts[i + 1] - pts[i])
    return pts[int(np.argmin(np.abs(ys - target_y)))].copy()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def load_prediction(npz_path: Path) -> Dict[str, Any]:
    data = np.load(Path(npz_path), allow_pickle=True)
    joints = np.asarray(data["joints"], dtype=np.float64)
    parents_raw = data["parents"]
    parents: List[Optional[int]] = []
    for p in (parents_raw.tolist() if parents_raw.ndim else parents_raw[()].tolist()):
        parents.append(None if p is None else int(p))
    skin = np.asarray(data["skin"], dtype=np.float64)
    if skin.ndim != 2 or skin.shape[1] != len(joints):
        skin = skin.T                     # 容错：上游注释出现过 (J,N) 写法
    names = data["names"]
    names = [str(n) for n in (names.tolist() if names.ndim else names[()].tolist())]
    return {
        "joints": joints,
        "parents": parents,
        "names": names,
        "skin": np.clip(skin, 0.0, None),
        "sampled_vertices": np.asarray(data["sampled_vertices"], dtype=np.float64),
        "vertices": np.asarray(data["vertices"], dtype=np.float64),
        "faces": np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3),
    }


def _classify_child_chains(n: int, joints: np.ndarray, kids: Dict[int, List[int]],
                           model_h: float) -> Tuple[List[List[int]], Optional[List[int]]]:
    """把节点 n 的每条子链分为腿（≥ 2 条）与脊柱（≤ 1 条，取净上升最大者）。

    腿：首骨方向 y<-0.7，**或**整链净下降 > 0.2H（髋部短骨近水平的真骨架也命中）；
    脊柱：净上升 > 0.2H 且横向位移 < 0.25H（排除从 pelvis 直接长出的臂/尾链）。
    """
    legs: List[List[int]] = []
    spine: Optional[List[int]] = None
    spine_rise = 0.0
    for c in kids.get(n, []):
        d = _dir(joints[n], joints[c])
        # 腿用「最向下」链量净下降；脊柱候选用「最高叶」链量上升（_chain_down
        # 在等高分叉处会误入臂链，导致脊柱被判成横向链）
        down_end = joints[_chain_down(c, kids, joints)[-1]]
        drop = float(joints[n][1] - down_end[1])
        up_chain = _chain_highest(c, kids, joints)
        end = joints[up_chain[-1]]
        rise = float(end[1] - joints[n][1])
        lateral = float(np.hypot(end[0] - joints[n][0], end[2] - joints[n][2]))
        if d[1] < _LEG_DIR_Y or drop > 0.2 * model_h:
            legs.append(_chain_down(c, kids, joints))
        elif rise > 0.2 * model_h and lateral < 0.25 * model_h and rise > spine_rise:
            spine, spine_rise = up_chain, rise
    return legs, spine


def _find_pelvis(joints: np.ndarray, kids: Dict[int, List[int]],
                 model_h: float) -> Tuple[int, List[List[int]], List[int]]:
    """返回 (pelvis, 腿链列表, 脊柱链)；找不到合格 pelvis 抛 UniRigAdaptError。"""
    best: Optional[Tuple[int, int, List[List[int]], List[int]]] = None
    for n in range(len(joints)):
        legs, spine = _classify_child_chains(n, joints, kids, model_h)
        if len(legs) >= 2 and spine is not None:
            score = len(legs) + (2 if len(legs) == 2 else 0)
            if best is None or score > best[0]:
                best = (score, n, legs, spine)
    if best is None:
        raise UniRigAdaptError(
            "UniRig 骨架无法映射到 22 语义关节（未找到具备双腿+脊柱的 pelvis），"
            "请改用比例骨架或旧 AI 链路")
    _, pelvis, legs, spine = best
    return pelvis, legs, spine


def _assign_sides(chains: List[List[int]], joints: np.ndarray,
                  origin: np.ndarray) -> Tuple[List[int], List[int]]:
    """按链首节点相对原点的 x 符号分左右（canonical +X = 角色左侧）。

    返回 (left_indices, right_indices)——chains 的下标；无法判定时按文件序对半分。
    """
    signs = [float(joints[c[0]][0] - origin[0]) for c in chains]
    left = [i for i, s in enumerate(signs) if s >= 0]
    right = [i for i, s in enumerate(signs) if s < 0]
    if not left or not right:
        order = sorted(range(len(chains)), key=lambda i: -signs[i])
        half = len(order) // 2
        left, right = order[:half], order[half:]
    return left, right


def _synthesize(side: str, part: str, height: float, base_y: float,
                cx: float, cz: float) -> np.ndarray:
    fy, fx, fz = PROPORTIONS[f"{part}_{side}"]
    return np.array([cx + fx * height, base_y + fy * height, cz + fz * height])


def adapt_prediction(pred: Dict[str, Any], height_m: Optional[float],
                     settings: Optional[RetargetSettings] = None
                     ) -> Tuple[Rig, np.ndarray, Dict[str, Any]]:
    """语义适配主函数：返回 ``(rig, skin 摘要用 W (N,22), 适配信息)``。"""
    settings = settings or get_settings()
    joints = pred["joints"]
    parents = pred["parents"]
    kids = _children_map(np.array([-1 if p is None else p for p in parents]))
    verts = pred["vertices"]
    v_min, v_max = verts.min(axis=0), verts.max(axis=0)
    base_y = float(v_min[1])
    mesh_h = float(v_max[1] - v_min[1]) or 1.0
    cx = float((v_min[0] + v_max[0]) * 0.5)
    cz = float((v_min[2] + v_max[2]) * 0.5)

    # 身高校验：与 S1 校准值偏差 >10% 时按对齐身高**等比**缩放关节位置（脚底为锚）。
    # 绝不能只缩 Y：那是各向异性变形，A-pose 的手臂会被压平接近 T-pose、
    # 腿的膝踝高度全错（ UniRig 空间未统一时这里曾掩盖过真 bug）
    height = mesh_h
    if height_m and height_m > 1e-6 and abs(mesh_h - height_m) / height_m > _HEIGHT_TOL:
        scale = height_m / mesh_h
        joints = v_min + (joints - v_min) * scale
        height = height_m
        logger.info("UniRig 网格身高 %.3fm 与对齐身高 %.3fm 偏差超 %.0f%%，等比缩放关节",
                    mesh_h, height_m, _HEIGHT_TOL * 100)

    pelvis, leg_chains, spine_chain = _find_pelvis(joints, kids, height)
    leg_roots = [c[0] for c in leg_chains]

    heads: Dict[str, np.ndarray] = {}
    source: Dict[str, str] = {}
    groups: Dict[str, List[int]] = {j: [] for j in JOINTS}   # 语义关节 → UniRig 骨集合
    synthesized: List[str] = []                              # 按比例合成的关节名（低置信）

    # ---- 脊柱链：pelvis→最高叶节点（_chain_highest 已含起点），等高度重采样
    #（基准 = 模型脚底，PROPORTIONS 的身高分数全部以脚底为基准，不是 pelvis 高度）----
    spine_chain = _chain_highest(pelvis, kids, joints)
    spine_pts, _ = _resample_polyline(
        [joints[n] for n in spine_chain], _SPINE_CUTS, "height", height,
        base_y=base_y)
    for name, pt in spine_pts.items():
        heads[name] = pt
        source[name] = "unirig"
    heads["pelvis"] = joints[pelvis].copy()
    source["pelvis"] = "unirig"
    # 脊柱骨归属：本项目骨架里「关节 j 的骨 = j→其子」，故 pelvis→spine_01 的骨属于
    # spine_01：骨中点在第一个高于它的 cut 处归属（区间 (pelvis,0.58] → spine_01 …）
    spine_nodes = set(spine_chain)
    for b in range(len(joints)):
        p = parents[b]
        if p is None or b not in spine_nodes or p not in spine_nodes:
            continue
        mid_y = float((joints[b][1] + joints[p][1]) * 0.5)
        frac = (mid_y - base_y) / height if height > 1e-9 else 0.0
        owner = "head"
        for name, cut in _SPINE_CUTS:
            if frac <= cut:
                owner = name
                break
        groups[owner].append(b)

    # ---- 腿链：等高重采样 + 缺失合成 ----
    l_idx, r_idx = _assign_sides(leg_chains, joints, joints[pelvis])
    for side, idxs in (("l", l_idx), ("r", r_idx)):
        for li in idxs[:1]:                      # 每侧只取一条主腿链
            chain = leg_chains[li]
            pts, _ = _resample_polyline(
                [joints[n] for n in chain],
                [(f"{part}_{side}", frac) for part, frac in _LEG_CUTS],
                "height", height, base_y=base_y)
            # 链短于 4 节时远端切点会被钳到链尾（y 明显高于目标）→ 按比例合成
            for (part, frac), (name, pt) in zip(_LEG_CUTS, pts.items()):
                if pt[1] - (base_y + frac * height) > 0.05 * height:
                    heads[name] = _synthesize(side, part, height, base_y, cx, cz)
                    source[name] = "unirig_synth"
                    synthesized.append(name)
                else:
                    heads[name] = pt
                    source[name] = "unirig"
            # 骨归属：骨属于链上方关节（upper_leg 的骨 = upper_leg→lower_leg，
            # 中点在 [0.27H, 0.48H) 区间）：首个满足 mid >= frac 的 cut 即归属
            chain_set = set(chain)
            for b in chain_set:
                p = parents[b]
                if p is None or p not in chain_set:
                    continue
                mid_y = float((joints[b][1] + joints[p][1]) * 0.5)
                owner = f"toe_{side}"
                for part, frac in _LEG_CUTS:      # 自上而下：upper_leg → toe
                    if mid_y >= frac * height + base_y:
                        owner = f"{part}_{side}"
                        break
                groups[owner].append(b)

    # ---- 臂链：胸腔区间（chest~neck 高度）脊柱节点上「整条子链净横向伸展>0.15H」
    # 的子链。不用首骨方向：真人形锁骨上扬（首骨 dir≈(±0.37,+0.93)），但整条臂外展
    # 到 ~0.5H；脊柱续段横向≈0。也不锁定单一 chest 节点：重采样的 chest 点可能落在
    # 两节点之间，而真骨架的臂可能挂在 chest 上下一节的任意节点上 ----
    y_lo = base_y + _SPINE_CUTS[2][1] * height      # chest 0.74H
    y_hi = base_y + _SPINE_CUTS[3][1] * height      # neck  0.84H
    arm_chains: List[List[int]] = []
    axis_xs: List[float] = []                       # 每条臂链挂载的脊柱节点 x（左右中轴）
    for sn in spine_chain:
        if not (y_lo - 0.05 * height <= float(joints[sn][1]) <= y_hi):
            continue
        axis_x_sn = float(joints[sn][0])
        for c in kids.get(sn, []):
            if c in spine_nodes or any(c == lr for lr in leg_roots):
                continue
            chain = _chain_down(c, kids, joints)
            lateral = max(abs(float(joints[n][0]) - axis_x_sn) for n in chain)
            if lateral > _ARM_LATERAL_MIN * height:
                arm_chains.append(chain)
                axis_xs.append(axis_x_sn)
    if arm_chains:
        # 左右判定：链首 x 相对其挂载脊柱节点 x 的符号（canonical +X=left），
        # 不用 mesh bbox 中心——网格偏心时后者会误判。每侧取最长链（排尾巴/饰物）
        by_side: Dict[str, List[List[int]]] = {"l": [], "r": []}
        for chain, ax in zip(arm_chains, axis_xs):
            side_key = "l" if float(joints[chain[0]][0] - ax) >= 0 else "r"
            by_side[side_key].append(chain)
        side_chains = {s: max(cs, key=lambda ch: _polyline_len([joints[n] for n in ch]))
                       for s, cs in by_side.items() if cs}
        axis_x = float(heads["chest"][0])       # 矢状面（脊柱中轴），与镜像合成同轴
        for side, chain in side_chains.items():
            sign = 1.0 if side == "l" else -1.0
            # 按绝对横向偏移重采样（与脊柱按绝对高度同理）：clavicle/upper_arm/
            # lower_arm/hand 分别在 axis_x ± (0.05/0.18/0.36/0.52)*H 处
            chain_pts = [joints[n] for n in chain]
            pts = {f"{part}_{side}": _point_at_x(chain_pts, axis_x + sign * frac * height)
                   for part, frac in _ARM_CUTS}
            for name, pt in pts.items():
                heads[name] = pt
                source[name] = "unirig"
            # 骨归属：按骨中点的横向偏移（与关节定位同口径），骨属于链上方
            # 关节：signed_x 在 [cut_k, cut_{k+1}) 区间的骨归第 k 个关节；链首骨
            # （近脊柱，signed_x < 0.05H）也归 clavicle
            chain_set = set(chain)
            for b in chain_set:
                p = parents[b]
                if p is None or p not in chain_set:
                    continue
                mid = (joints[b] + joints[p]) * 0.5
                signed = sign * (float(mid[0]) - axis_x)
                owner = f"{_ARM_CUTS[-1][0]}_{side}"
                for part, frac in _ARM_CUTS:      # 自内向外：clavicle → hand
                    if signed <= frac * height:
                        owner = f"{part}_{side}"
                        break
                groups[owner].append(b)
        # 只找到一侧臂链时，另一侧绕矢状面（x = chest x）镜像合成
        if len(side_chains) == 1:
            have = next(iter(side_chains))
            miss = "r" if have == "l" else "l"
            mirror_x = float(heads["chest"][0])
            for part, _ in _ARM_CUTS:
                pt = heads[f"{part}_{have}"].copy()
                pt[0] = 2 * mirror_x - pt[0]
                heads[f"{part}_{miss}"] = pt
                source[f"{part}_{miss}"] = "unirig_synth"
                synthesized.append(f"{part}_{miss}")
    else:
        for side in ("l", "r"):
            for part, _ in _ARM_CUTS:
                heads[f"{part}_{side}"] = _synthesize(
                    side, part, height, base_y, cx, cz)
                source[f"{part}_{side}"] = "unirig_synth"
                synthesized.append(f"{part}_{side}")

    for j in JOINTS:               # 兜底：任何漏掉的关节按比例合成
        if j not in heads:
            fy, fx, fz = PROPORTIONS[j]
            heads[j] = np.array([cx + fx * height, base_y + fy * height, cz + fz * height])
            source[j] = "unirig_synth"
            synthesized.append(j)

    # ---- 权重聚合：W[:,s] = Σ_{j∈B(s)} skin[:,j] ----
    skin = pred["skin"]
    n_s = skin.shape[0]
    Ws = np.zeros((n_s, len(JOINTS)), dtype=np.float64)
    for j, bones in groups.items():
        if bones:
            Ws[:, JOINT_INDEX[j]] = skin[:, bones].sum(axis=1)

    rig = Rig(heads={j: heads[j].astype(np.float64) for j in JOINTS},
              height=float(height), revision=0,
              confidence={}, source=source)

    # 聚合后零权重的采样顶点 → 最近语义骨段兜底
    zero = Ws.sum(axis=1) < 1e-9
    if zero.any():
        segs = _bone_segments(rig)
        nearest = np.argmin(_all_segment_distances(pred["sampled_vertices"][zero], segs),
                            axis=1)
        Ws[zero, nearest] = 1.0
    rs = Ws.sum(axis=1, keepdims=True)
    rs[rs < 1e-9] = 1.0
    Ws /= rs

    info = {"synthesized": synthesized, "groups": {j: len(b) for j, b in groups.items()},
            "pelvis": int(pelvis), "height": float(height), "mesh_height": float(mesh_h)}
    return rig, Ws, info


def _polyline_len(pts: Sequence[np.ndarray]) -> float:
    pts = np.stack([np.asarray(p) for p in pts])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def _point_at_x(pts, target_x: float) -> np.ndarray:
    """折线上 x 首次到达 target_x 的点（超出范围钳到 x 最接近的链端点）。"""
    xs = np.array([p[0] for p in pts])
    if target_x >= xs.max():
        return pts[int(np.argmax(xs))].copy()
    if target_x <= xs.min():
        return pts[int(np.argmin(xs))].copy()
    for i in range(len(pts) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if (x0 - target_x) * (x1 - target_x) <= 0 and abs(x1 - x0) > 1e-12:
            t = (target_x - x0) / (x1 - x0)
            return pts[i] + t * (pts[i + 1] - pts[i])
    return pts[int(np.argmin(np.abs(xs - target_x)))].copy()


def _transfer_to_full(Ws: np.ndarray, sampled: np.ndarray, full: np.ndarray) -> np.ndarray:
    """采样顶点权重 → 全分辨率网格（k=7 近邻反距离加权，与上游 reskin 同参）。"""
    if len(sampled) == len(full) and np.allclose(sampled, full):
        return Ws
    k = min(7, len(sampled))
    _, idx = cKDTree(sampled).query(full, k=k)
    idx = np.atleast_2d(idx)
    dd = 1.0 / (np.linalg.norm(
        full[:, None, :] - sampled[idx], axis=2) + 1e-4)
    dd /= dd.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkj->nj", dd, Ws[idx])


def build_binding(glb_path: Path, work_dir: Path, out_dir: Path,
                  height_m: Optional[float] = None,
                  settings: Optional[RetargetSettings] = None
                  ) -> Tuple[Rig, int, Dict[str, Any]]:
    """适配 + 回传 + 落盘：写 ``rig.json`` / ``skin.npz`` / ``skin_report.json``。

    返回 ``(rig, 顶点数, report)``，report 兼容 ``stages.skin_summary`` 并额外带
    ``method='unirig'`` / ``unirig`` 详情 / ``coverage``（顶点最大权重均值）。
    顶点坐标以 ``stages.load_mesh``（与旧链路/导出共用同一 glb 读取）为准，
    UniRig 采样顶点与其做最近邻对应，故对 GLB 节点变换的差异不敏感。
    """
    t0 = time.time()
    settings = settings or get_settings()
    pred = load_prediction(Path(work_dir) / "unirig_raw.npz")
    rig, Ws, info = adapt_prediction(pred, height_m, settings)

    mesh = stages.load_mesh(glb_path)
    P = np.ascontiguousarray(mesh["positions"], dtype=np.float64)
    tri = np.asarray(mesh["indices"], dtype=np.int64).reshape(-1, 3)

    W = _transfer_to_full(Ws, pred["sampled_vertices"], P)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs < 1e-9] = 1.0
    W /= rs
    # 回传后仍零权重的顶点（理论上无）→ 最近语义骨段兜底
    zero = W.sum(axis=1) < 1e-6
    if zero.any():
        segs = _bone_segments(rig)
        nearest = np.argmin(_all_segment_distances(P[zero], segs), axis=1)
        W[zero] = 0.0
        W[zero, nearest] = 1.0
    coverage = float(W.max(axis=1).mean())

    joints_u16, weights_f32 = _topk(W, settings.skin, len(JOINTS))

    # 报告：网格诊断字段与旧链路同口径（island/weld/components），求解字段置零
    n_orig, lab_orig = _components(tri, len(P))
    sizes = np.bincount(lab_orig, minlength=n_orig)
    island_stats = {"components": int(n_orig),
                    "largest_share": round(float(sizes.max() / max(len(P), 1)), 4),
                    "tiny_lt16": int((sizes < 16).sum())}
    _, _, cluster_of = _weld_solve_domain(P, tri, float(settings.skin.weld_epsilon))
    report = _skin_report(P, tri, joints_u16, weights_f32, W, rig,
                          ncomp=1, fallback=int(zero.sum()), proxy_comps=0,
                          transferred=0, island_stats=island_stats,
                          n_clusters=int(len(np.unique(cluster_of))),
                          elapsed=time.time() - t0)
    report["method"] = "unirig"
    report["coverage"] = round(coverage, 4)
    report["unirig"] = info

    # 置信度：coverage 为主，合成关节打折
    for j in JOINTS:
        rig.confidence[j] = round(
            coverage * (0.3 if source_is_synth(rig, j) else 1.0), 4)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stages.write_json(out / stages.RIG_FILE, rig.to_dict())
    np.savez(out / stages.SKIN_FILE, joints=joints_u16, weights=weights_f32)
    stages.write_json(out / stages.SKIN_REPORT_FILE, report)
    logger.info("UniRig 适配完成：%d 顶点，coverage=%.3f，合成关节=%s",
                len(P), coverage, info["synthesized"] or "无")
    return rig, len(P), report


def source_is_synth(rig: Rig, joint: str) -> bool:
    return rig.source.get(joint, "") == "unirig_synth"
