"""P3 定向验证：别名库/模糊匹配 + 层级补全、脚部 IK 锁定、门控审核、质量报告富化。

复用 P1 的构造器与 fixture（env/client/_wait/_build_source_glb/_build_target_glb）。
不做真实模型推理、不联网、只写 tmp_path。
运行： <venv>/bin/python -m pytest tests/test_retarget_p3.py -v
"""

from __future__ import annotations

import json

import numpy as np

from ai3d.retarget import foot_ik
from ai3d.retarget.mapping import build_mapping, match_semantic, normalize_name
from ai3d.retarget.schemas import JobConfig
from ai3d.retarget.settings import RetargetConfig
from ai3d.retarget.skeleton import (
    CORE_JOINTS,
    JOINTS,
    JOINT_INDEX,
    PARENTS,
    PROPORTIONS,
    Rig,
    generate_rig_from_bbox,
)

from test_retarget_p1 import (  # 复用 fixture 与构造器
    _build_source_glb,
    _build_target_glb,
    _wait,
    client,  # noqa: F401  (pytest fixture 需被导入以在本模块生效)
    env,  # noqa: F401
)


# --------------------------------------------------------------------------- #
# 别名库 / 模糊匹配 / 前缀剥离
# --------------------------------------------------------------------------- #
def test_normalize_name_strips_prefixes():
    assert normalize_name("mixamorig:Hips") == "hips"
    assert normalize_name("DEF-upper_arm.L") == "upperarmL".lower()
    assert normalize_name("Bip001 L Thigh") == "lthigh"
    assert normalize_name("J_Bip_C_Hips") == "hips"
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


def test_match_semantic_across_naming_systems():
    cases = {
        # Mixamo
        "mixamorig:Hips": "pelvis", "mixamorigLeftUpLeg": "upper_leg_l",
        "mixamorig:RightForeArm": "lower_arm_r", "mixamorigLeftToeBase": "toe_l",
        # Unreal / Mannequin
        "thigh_l": "upper_leg_l", "calf_r": "lower_leg_r", "clavicle_l": "clavicle_l",
        # Blender / Rigify（带 DEF 前缀）
        "DEF-upper_arm.L": "upper_arm_l", "DEF-foot.R": "foot_r",
        # VRM
        "J_Bip_C_Hips": "pelvis", "J_Bip_C_LeftUpperLeg": "upper_leg_l",
        # 后缀式左右标记
        "armL": "upper_arm_l", "legR": "lower_leg_r", "handL": "hand_l",
    }
    for raw, want in cases.items():
        sem, score, method = match_semantic(raw)
        assert sem == want, (raw, sem, want)
        assert score == 1.0 and method == "alias", (raw, score, method)


def test_match_semantic_fuzzy_variants():
    # twist/编号等变体走模糊子串匹配（score<1.0, method="fuzzy"）
    for raw, want in {
        "mixamorigLeftArmTwist": "upper_arm_l",
        "LeftForeArmTwist": "lower_arm_l",
        "RightUpLegRoll1": "upper_leg_r",
    }.items():
        sem, score, method = match_semantic(raw)
        assert sem == want, (raw, sem)
        assert method == "fuzzy", (raw, method)
        assert 0.5 <= score < 1.0, (raw, score)
    # 完全未知 → 无匹配
    assert match_semantic("zzz_unknown_bone") == (None, 0.0, "none")


def test_build_mapping_hierarchy_fills_unknown_name():
    """一个几何正确但命名未知的骨，应由层级+几何补全（method="hierarchy"）。"""
    height = 1.8
    # 中轴用可识别的 Mixamo 名（根关节无法靠层级补全，必须命名命中）；
    # 肢体走下方 fallback 的 Mixamo 名；foot_l 故意用未知名以验证层级补全。
    names = {
        "pelvis": "mixamorig:Hips", "spine_01": "mixamorig:Spine",
        "spine_02": "mixamorig:Spine1", "chest": "mixamorig:Spine2",
        "neck": "mixamorig:Neck", "head": "mixamorig:Head",
    }
    source_joints = []
    for i, j in enumerate(JOINTS):
        fy, fx, fz = PROPORTIONS[j]
        pos = [fx * height, fy * height, fz * height]
        p = PARENTS[j]
        parent = JOINT_INDEX[p] if p is not None else None
        # 中轴用可识别名，肢体用 Mixamo 风格名；foot_l 故意用未知名
        if j in names:
            nm = names[j]
        elif j == "foot_l":
            nm = "zzz_unnamed"
        else:
            side = "Left" if j.endswith("_l") else "Right"
            base = {"clavicle": "Shoulder", "upper_arm": "Arm", "lower_arm": "ForeArm",
                    "hand": "Hand", "upper_leg": "UpLeg", "lower_leg": "Leg",
                    "foot": "Foot", "toe": "ToeBase"}[j.rsplit("_", 1)[0]]
            nm = f"mixamorig:{side}{base}"
        source_joints.append({"node": i, "name": nm, "parent": parent, "position": pos})

    mapping = build_mapping(source_joints)
    by_sem = {it["semantic"]: it for it in mapping["items"]}
    # 其余关节全部匹配
    assert mapping["matched"] == len(JOINTS), mapping["items"]
    # foot_l 由层级补全命中（几何正确）
    assert by_sem["foot_l"]["source_node"] == JOINT_INDEX["foot_l"]
    assert by_sem["foot_l"]["method"] == "hierarchy"


# --------------------------------------------------------------------------- #
# 脚部 IK：接触检测 + 双骨锁定
# --------------------------------------------------------------------------- #
def _identity_rotations(n_frames: int) -> dict:
    return {j: np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_frames, 1)) for j in JOINTS}


def test_foot_lock_stabilizes_planted_ankle():
    """支撑相（脚踝低 + 水平速度慢）：锁定后脚踝世界漂移显著减小，且只改髋/膝通道。"""
    rig = generate_rig_from_bbox([-.3, 0, -.3], [.3, 1.8, .3], confidence=0.9)
    F = 12
    times = np.arange(F) * 0.033
    pelvis = np.asarray(rig.heads["pelvis"], float)
    root_t = np.tile(pelvis, (F, 1))
    root_t[:, 2] += np.arange(F) * 0.004  # 缓慢前移（≈0.12 m/s < 阈值 0.30）
    rot = _identity_rotations(F)

    cfg = RetargetConfig(foot_lock=True, foot_contact_speed=0.30,
                         foot_contact_height=0.12, foot_blend_frames=2,
                         foot_min_contact_frames=2)
    # FK 得到锁定前脚踝世界位置
    P_before, _ = foot_ik._fk(rig, rot, root_t)
    ank_l_before = P_before["foot_l"][:, [0, 2]].std(axis=0).sum()

    rot_out, info = foot_ik.apply_foot_lock(rig, rot, root_t, times, cfg)

    assert info["applied"] is True, info
    assert info["windows"]["l"] >= 1 and info["windows"]["r"] >= 1, info
    assert info["max_correction"] > 0.0
    # 锁定后脚踝漂移更小
    P_after, _ = foot_ik._fk(rig, rot_out, root_t)
    ank_l_after = P_after["foot_l"][:, [0, 2]].std(axis=0).sum()
    assert ank_l_after < ank_l_before * 0.6, (ank_l_before, ank_l_after)
    # 仅髋/膝通道被改写，foot 通道保持不变（脚踝位置只取决于髋/膝）
    for j in JOINTS:
        if j in ("upper_leg_l", "lower_leg_l", "upper_leg_r", "lower_leg_r"):
            continue
        assert np.allclose(rot_out[j], rot[j], atol=1e-9), j


def test_foot_lock_noop_on_fast_swing():
    """摆动相（水平速度快）：无接触窗口，动画原样返回（P1 合成动画即此情形）。"""
    rig = generate_rig_from_bbox([-.3, 0, -.3], [.3, 1.8, .3], confidence=0.9)
    F = 10
    times = np.arange(F) * 0.033
    pelvis = np.asarray(rig.heads["pelvis"], float)
    root_t = np.tile(pelvis, (F, 1))
    root_t[:, 2] += np.arange(F) * 0.05  # 快速移动（≈1.5 m/s > 阈值）
    rot = _identity_rotations(F)

    rot_out, info = foot_ik.apply_foot_lock(rig, rot, root_t, times, RetargetConfig())
    assert info["applied"] is False
    assert info["windows"]["l"] == 0 and info["windows"]["r"] == 0
    for j in JOINTS:
        assert np.allclose(rot_out[j], rot[j], atol=1e-12), j


# --------------------------------------------------------------------------- #
# 门控审核（_gate 单元 + rerun force 端到端）
# --------------------------------------------------------------------------- #
def _make_task(store, conf: float, core_conf=None, auto_continue=False) -> str:
    tid = store.create_task(
        "s.glb", "t.glb",
        JobConfig(enable_pose_ai=False, auto_continue_on_review=auto_continue))
    rig = generate_rig_from_bbox([-.3, 0, -.3], [.3, 1.8, .3], confidence=conf)
    if core_conf is not None:
        for j in CORE_JOINTS:
            rig.confidence[j] = core_conf
    store.save_rig(tid, 0, rig.to_dict())
    return tid


def test_gate_regimes(client, env):  # noqa: F811 - client/env 提供隔离的 store/worker
    from ai3d.retarget.task_store import get_store
    from ai3d.retarget.worker import get_worker
    store, w = get_store(), get_worker()

    # ≥auto_pass 且核心关节达标 → 放行
    assert w._gate(_make_task(store, 0.9)) is None
    # [review_min, auto_pass) → 审核（非阻断）
    g = w._gate(_make_task(store, 0.65))
    assert g is not None and g[1] is False and g[0]
    # <review_min → 阻断
    g = w._gate(_make_task(store, 0.40))
    assert g is not None and g[1] is True
    # 整体高但核心关节 <core_joint_min → 强制审核（非阻断）
    g = w._gate(_make_task(store, 0.9, core_conf=0.3))
    assert g is not None and g[1] is False
    assert any("核心关节" in r for r in g[0])
    # auto_continue_on_review 放行非阻断审核，但不放行阻断
    assert w._gate(_make_task(store, 0.65, auto_continue=True)) is None
    g = w._gate(_make_task(store, 0.40, auto_continue=True))
    assert g is not None and g[1] is True


def test_gate_blocks_export_then_force_rerun(client, env, tmp_path):  # noqa: F811
    src_bytes = _build_source_glb(tmp_path / "src.glb")
    tgt_bytes = _build_target_glb(tmp_path / "tgt.glb")
    r = client.post("/v1/jobs", files={
        "source_file": ("src.glb", src_bytes, "model/gltf-binary"),
        "target_file": ("tgt.glb", tgt_bytes, "model/gltf-binary"),
    }, data={"config": json.dumps({"enable_pose_ai": False})})
    task_id = r.json()["task_id"]
    info = _wait(client, task_id, {"DONE", "FAILED", "WAITING"})
    assert info["state"] == "DONE", info
    result_glb = env.task_dir(task_id) / "result.glb"
    assert result_glb.exists()

    # 把 rig.json 置信度压到阻断区间，从 EXPORT_VERIFY 重跑（不 force）
    rig_path = env.task_dir(task_id) / "rig.json"
    rig = Rig.from_dict(json.loads(rig_path.read_text(encoding="utf-8")))
    for j in JOINTS:
        rig.confidence[j] = 0.40
    rig_path.write_text(json.dumps(rig.to_dict(), ensure_ascii=False), encoding="utf-8")

    rr = client.post(f"/v1/jobs/{task_id}/rerun", json={"from_stage": "EXPORT_VERIFY"})
    assert rr.status_code == 200, rr.text
    info2 = _wait(client, task_id, {"NEEDS_REVIEW", "DONE", "FAILED"})
    assert info2["state"] == "NEEDS_REVIEW", info2
    assert info2["review_reasons"], info2
    assert not result_glb.exists(), "阻断时不应产出 result.glb"
    by_stage = {s["stage"]: s["status"] for s in info2["stages"]}
    assert by_stage["EXPORT_VERIFY"] == "WAITING", by_stage

    # force=True 强制跳过门控 → DONE，result.glb 重新产出
    rr2 = client.post(f"/v1/jobs/{task_id}/rerun",
                      json={"from_stage": "EXPORT_VERIFY", "force": True})
    assert rr2.status_code == 200, rr2.text
    info3 = _wait(client, task_id, {"DONE", "FAILED", "NEEDS_REVIEW"})
    assert info3["state"] == "DONE", info3
    assert result_glb.exists()


# --------------------------------------------------------------------------- #
# 质量报告富化
# --------------------------------------------------------------------------- #
def test_report_enriched_fields(client, env, tmp_path):  # noqa: F811
    src_bytes = _build_source_glb(tmp_path / "src.glb")
    tgt_bytes = _build_target_glb(tmp_path / "tgt.glb")
    r = client.post("/v1/jobs", files={
        "source_file": ("src.glb", src_bytes, "model/gltf-binary"),
        "target_file": ("tgt.glb", tgt_bytes, "model/gltf-binary"),
    }, data={"config": json.dumps({"enable_pose_ai": False})})
    task_id = r.json()["task_id"]
    info = _wait(client, task_id, {"DONE", "FAILED", "WAITING"})
    assert info["state"] == "DONE", info

    report = json.loads((env.task_dir(task_id) / "report.json").read_text(encoding="utf-8"))
    # rig 富化：逐关节置信度 + 来源 + 核心关节失败列表
    assert set(report["rig"]["per_joint_confidence"]) == set(JOINTS)
    assert set(report["rig"]["sources"]) == set(JOINTS)
    assert report["rig"]["core_failures"] == []
    assert report["rig"]["confidence"] >= 0.8
    # mapping 富化：方法计数（全 alias 命中）
    assert report["mapping"]["matched"] == len(JOINTS)
    assert report["mapping"]["methods"].get("alias") == len(JOINTS)
    # retarget 元数据（含脚部 IK 信息）
    assert report["retarget"]["frames"] >= 2
    assert set(report["retarget"]["foot_ik"]) >= {
        "applied", "windows", "contact_frames", "max_correction"}
    # 门控决策
    assert report["gating"]["decision"] == "PASS"
    assert report["gating"]["auto_pass"] == 0.80
