"""源骨 → 22 语义关节映射。

匹配策略（优先级从高到低）：
1. ``overrides``：人工指定 semantic → 源骨名称（PATCH /mapping 覆盖）。
2. 精确别名：归一化骨名（去分隔符/前缀、小写）命中 ``ALIASES``（method="alias", score=1.0）。
3. 模糊别名：归一化骨名包含某个已知别名子串（取最长者），覆盖带编号/twist 等变体
   （method="fuzzy", score<1.0）。
4. 层级补全：仍未匹配的语义关节，用源骨架层级 + 归一化几何从已匹配父关节推断
   （method="hierarchy", score<1.0；需提供 parent/position 特征）。

别名库覆盖 Mixamo / Unreal(Mannequin) / Unity(Humanoid) / Blender(Rigify) / VRM / Rokoko /
CMU 等常见命名体系；肢体系别名由「部位关键词 × 左右标记 × 前/后缀」自动组合生成，
中轴关节（骨盆/脊柱/胸/颈/头）按各体系习惯手工维护。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .skeleton import JOINTS, PARENTS

# 归一化时剥离的常见前缀（Rigify 的 DEF/MCH/ORG/WGT/FK/IK、VRM 的 J_Bip_C 等）
_PREFIXES = (
    "mixamorig", "mixamohand", "valvebiped", "bip001", "bip01", "bip02",
    "armature", "jbipc", "jbip", "def", "mch", "org", "wgt", "fk", "ik",
)

# 肢体系：语义部位 ← 关键词（英文习惯名，归一化后无分隔符）
_LIMB_PARTS: Dict[str, Tuple[str, ...]] = {
    "clavicle": ("shoulder", "clavicle", "collar"),
    "upper_arm": ("arm", "upperarm", "shldr", "humerus", "bicep"),
    "lower_arm": ("forearm", "lowerarm", "elbow", "radius"),
    "hand": ("hand", "wrist", "palm"),
    "upper_leg": ("upleg", "upperleg", "thigh", "femur"),
    "lower_leg": ("leg", "lowerleg", "calf", "shin", "tibia"),
    "foot": ("foot", "ankle", "feet"),
    "toe": ("toe", "toebase", "ball", "phalanges"),
}
# 左右标记（前缀式 leftarm / 后缀式 arml 都覆盖）
_SIDE_TOKENS: Dict[str, Tuple[str, ...]] = {"l": ("left", "l"), "r": ("right", "r")}

# 中轴关节：手工维护（各体系习惯差异大，不做左右组合）
# 注意：别名必须全局唯一（一个别名只归一个语义），且遵循 Mixamo 惯例
# Hips/Spine/Spine1/Spine2 → pelvis/spine_01/spine_02/chest。
_MIDLINE_ALIASES: Dict[str, set] = {
    "pelvis": {"hips", "hip", "pelvis", "root", "center", "cog", "sacrum",
               "body", "base", "hipcenter"},
    "spine_01": {"spine", "spine01", "spine0", "spinea", "abdomen", "waist",
                 "belly", "torso"},
    "spine_02": {"spine1", "spine02", "spineb", "middlespine"},
    "chest": {"spine2", "spine03", "spine3", "spinec", "chest", "upperchest",
              "thorax", "pectorals"},
    "neck": {"neck", "neck01", "neck1", "necka", "cervical"},
    "head": {"head", "head1", "skull", "cranium", "headtop"},
}


def _build_aliases() -> Dict[str, set]:
    als: Dict[str, set] = {j: set(v) for j, v in _MIDLINE_ALIASES.items()}
    for part, kws in _LIMB_PARTS.items():
        for side, toks in _SIDE_TOKENS.items():
            sem = f"{part}_{side}"
            s = als.setdefault(sem, set())
            for kw in kws:
                for tok in toks:
                    s.add(f"{tok}{kw}")   # leftarm / larm
                    s.add(f"{kw}{tok}")   # armleft / arml
    return als


ALIASES: Dict[str, set] = _build_aliases()

# 反向索引：归一化别名 → 语义（一个别名只归属一个语义）
_ALIAS_TO_SEMANTIC: Dict[str, str] = {}
for _sem, _als in ALIASES.items():
    for _a in _als:
        _ALIAS_TO_SEMANTIC.setdefault(_a, _sem)

# 模糊匹配用的别名（长度 >= 5 才做子串匹配，避免误命中）
_FUZZY_ALIASES = [(a, s) for a, s in _ALIAS_TO_SEMANTIC.items() if len(a) >= 5]


def normalize_name(raw: Optional[str]) -> str:
    """归一化骨名：小写、去非字母数字、循环剥离已知前缀。"""
    if not raw:
        return ""
    n = re.sub(r"[^a-z0-9]", "", raw.lower())
    for _ in range(3):  # 最多剥 3 层前缀（如 DEF-fk_upper_arm）
        stripped = False
        for p in _PREFIXES:
            if n.startswith(p) and len(n) - len(p) >= 3:
                n = n[len(p):]
                stripped = True
                break
        if not stripped:
            break
    return n


def match_semantic(bone_name: Optional[str], fuzzy: bool = True
                   ) -> Tuple[Optional[str], float, str]:
    """返回 (semantic_id, score, method)；未命中返回 (None, 0.0, "none")。"""
    n = normalize_name(bone_name)
    if not n:
        return None, 0.0, "none"
    if n in _ALIAS_TO_SEMANTIC:
        return _ALIAS_TO_SEMANTIC[n], 1.0, "alias"
    if fuzzy:
        best_sem, best_len = None, 0
        for alias, sem in _FUZZY_ALIASES:
            if alias in n and len(alias) > best_len:
                best_sem, best_len = sem, len(alias)
        if best_sem is not None:
            score = round(min(0.9, 0.5 + 0.4 * best_len / max(len(n), 1)), 3)
            return best_sem, score, "fuzzy"
    return None, 0.0, "none"


def _match_features(sem_to_node: Dict[str, Optional[int]],
                    source_joints: List[Dict],
                    used: set) -> None:
    """层级 + 归一化几何补全：仅当提供 parent/position 特征时启用。

    对未匹配的语义关节 j：在其父语义 p 已匹配的源节点 np 的直接子节点里，选出
    归一化位置最接近 j 期望位置（由 p 位置 + 语义方向/长度比例估计）的未用子节点。
    """
    has_pos = all(j.get("position") is not None for j in source_joints) and len(source_joints) > 0
    if not has_pos:
        return
    pos = {j["node"]: np.asarray(j["position"], float) for j in source_joints}
    children_of: Dict[int, List[int]] = {}
    for j in source_joints:
        p = j.get("parent")
        if p is not None:
            children_of.setdefault(p, []).append(j["node"])
    # 归一化尺度：用已匹配关节位置估算身高
    matched_pos = [pos[n] for n in sem_to_node.values() if n is not None and n in pos]
    if len(matched_pos) < 3:
        return
    P = np.stack(matched_pos)
    height = float(P[:, 1].max() - P[:, 1].min()) or 1.0
    # 语义方向单位向量（相对父）与相对长度（用标准比例骨架估计）
    from .skeleton import PROPORTIONS
    dir_unit: Dict[str, np.ndarray] = {}
    rel_len: Dict[str, float] = {}
    for j in JOINTS:
        p = PARENTS[j]
        if p is None:
            continue
        pj = np.asarray(PROPORTIONS[j], float)
        pp = np.asarray(PROPORTIONS[p], float)
        d = np.array([pj[1] - pp[1], pj[0] - pp[0], pj[2] - pp[2]])  # (y,x,z)
        L = float(np.linalg.norm(d))
        rel_len[j] = L
        dir_unit[j] = d / L if L > 1e-6 else np.array([0.0, 1.0, 0.0])

    for _ in range(len(JOINTS)):  # 迭代传播（父先于子）
        progressed = False
        for j in JOINTS:
            if sem_to_node.get(j) is not None:
                continue
            p = PARENTS[j]
            if p is None:
                continue
            np_node = sem_to_node.get(p)
            if np_node is None:
                continue
            cands = [c for c in children_of.get(np_node, []) if c not in used and c in pos]
            if not cands:
                continue
            base = pos[np_node]
            expected = base + dir_unit[j] * (rel_len[j] * height)
            best_c, best_d = None, float("inf")
            for c in cands:
                d = float(np.linalg.norm(pos[c] - expected))
                if d < best_d:
                    best_c, best_d = c, d
            # 几何门限：偏差超过 0.35*身高视为不可靠，跳过
            if best_c is not None and best_d < 0.35 * height:
                sem_to_node[j] = best_c
                used.add(best_c)
                progressed = True
        if not progressed:
            break


def build_mapping(source_joints: List[Dict], overrides: Optional[Dict[str, str]] = None,
                  cfg=None) -> Dict:
    """source_joints: [{'node': int, 'name': str, 'parent'?: int, 'position'?: [x,y,z]}, ...]。

    返回 mapping.json 结构：semantic → 源节点索引 + 每项 score/method。overrides 优先。
    """
    overrides = overrides or {}
    min_score = float(getattr(cfg, "min_match_score", 0.0)) if cfg is not None else 0.0
    name_to_node = {normalize_name(j.get("name")): j.get("node")
                    for j in source_joints if j.get("name")}
    sem_to_node: Dict[str, Optional[int]] = {j: None for j in JOINTS}
    sem_method: Dict[str, str] = {j: "none" for j in JOINTS}
    sem_score: Dict[str, float] = {j: 0.0 for j in JOINTS}
    used: set = set()

    def assign(sem: str, node: Optional[int], method: str, score: float) -> None:
        if node is None or sem_to_node.get(sem) is not None or node in used:
            return
        if method != "override" and score < min_score:
            return
        sem_to_node[sem] = node
        used.add(node)
        sem_method[sem] = method
        sem_score[sem] = score

    # 1) 人工覆盖优先
    for sem, src_name in overrides.items():
        if sem in sem_to_node:
            assign(sem, name_to_node.get(normalize_name(src_name)), "override", 1.0)

    # 2) 精确别名，3) 模糊别名（两轮，确保精确全局优先于模糊）
    for want in ("alias", "fuzzy"):
        for j in source_joints:
            node = j.get("node")
            if node in used:
                continue
            sem, score, method = match_semantic(j.get("name"), fuzzy=(want == "fuzzy"))
            if method != want:
                continue
            assign(sem, node, method, score)

    # 4) 层级 + 几何补全
    _match_features(sem_to_node, source_joints, used)
    for j in JOINTS:
        if sem_to_node.get(j) is not None and sem_method[j] == "none":
            sem_method[j] = "hierarchy"
            sem_score[j] = 0.6

    node_to_name = {j.get("node"): j.get("name") for j in source_joints}
    items = []
    for sem in JOINTS:
        node = sem_to_node.get(sem)
        items.append({
            "semantic": sem, "source_node": node,
            "source_name": node_to_name.get(node) if node is not None else None,
            "score": round(float(sem_score[sem]), 3), "method": sem_method[sem],
        })

    matched = sum(1 for s in JOINTS if sem_to_node.get(s) is not None)
    return {
        "revision": 0,
        "matched": matched,
        "total": len(JOINTS),
        "coverage": round(matched / len(JOINTS), 3),
        "semantic_to_source_node": {s: sem_to_node[s] for s in JOINTS},
        "items": items,
        "overrides": overrides,
    }
