"""P1 定向验证：确定性闭环（比例骨架 → 自动蒙皮 → 别名映射 → 增量重定向 → result.glb）。

用程序生成的「带骨架+动画的源 GLB（Mixamo 命名）」与「无骨骼目标 GLB（人形网格）」，
在 enable_pose_ai=False 下跑通到 result.glb 并回读验证。
不做真实模型推理、不联网、只写 tmp_path（RETARGET_PROJECT_DIR / RETARGET_DB_PATH 重定向）。
运行： <venv>/bin/python -m pytest tests/test_retarget_p1.py -v
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Attributes,
    Buffer,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
    Skin,
)

from ai3d.retarget import glb_io
from ai3d.retarget.glb_build import _Builder
from ai3d.retarget.mapping import build_mapping
from ai3d.retarget.retarget import retarget_animation
from ai3d.retarget.skeleton import (
    JOINTS,
    JOINT_INDEX,
    PARENTS,
    PROPORTIONS,
    Rig,
    children_of,
    generate_rig_from_bbox,
)
from ai3d.retarget.skinning import compute_skin_weights

# 语义关节 → Mixamo 骨名（用于验证别名库映射）
SEMANTIC_TO_MIXAMO = {
    "pelvis": "mixamorig:Hips", "spine_01": "mixamorig:Spine",
    "spine_02": "mixamorig:Spine1", "chest": "mixamorig:Spine2",
    "neck": "mixamorig:Neck", "head": "mixamorig:Head",
    "clavicle_l": "mixamorig:LeftShoulder", "upper_arm_l": "mixamorig:LeftArm",
    "lower_arm_l": "mixamorig:LeftForeArm", "hand_l": "mixamorig:LeftHand",
    "clavicle_r": "mixamorig:RightShoulder", "upper_arm_r": "mixamorig:RightArm",
    "lower_arm_r": "mixamorig:RightForeArm", "hand_r": "mixamorig:RightHand",
    "upper_leg_l": "mixamorig:LeftUpLeg", "lower_leg_l": "mixamorig:LeftLeg",
    "foot_l": "mixamorig:LeftFoot", "toe_l": "mixamorig:LeftToeBase",
    "upper_leg_r": "mixamorig:RightUpLeg", "lower_leg_r": "mixamorig:RightLeg",
    "foot_r": "mixamorig:RightFoot", "toe_r": "mixamorig:RightToeBase",
}

_SRC_HEIGHT = 1.8


def _source_heads(height: float = _SRC_HEIGHT) -> dict:
    heads = {}
    for j in JOINTS:
        fy, fx, fz = PROPORTIONS[j]
        heads[j] = np.array([fx * height, fy * height, fz * height], dtype=np.float64)
    return heads


def _build_source_glb(path, n_frames: int = 3) -> bytes:
    """identity-rest 源骨架（Mixamo 命名）+ 动画：upper_arm_l 绕 Z 摆动 + 骨盆前移。"""
    heads = _source_heads()
    b = _Builder()

    ibm = np.zeros((len(JOINTS), 16), dtype=np.float32)
    for i, j in enumerate(JOINTS):
        h = heads[j]
        ibm[i] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -h[0], -h[1], -h[2], 1]
    acc_ibm = b.add(ibm, "MAT4", FLOAT, count=len(JOINTS))

    times = np.linspace(0.0, 1.0, n_frames).astype(np.float32)
    acc_time = b.add(times, "SCALAR", FLOAT, minmax=True)

    samplers: list[AnimationSampler] = []
    channels: list[AnimationChannel] = []

    def add_ch(node: int, path_: str, acc: int) -> None:
        si = len(samplers)
        samplers.append(AnimationSampler(input=acc_time, interpolation="LINEAR", output=acc))
        channels.append(AnimationChannel(
            target=AnimationChannelTarget(node=node, path=path_), sampler=si))

    for j in JOINTS:
        if j == "upper_arm_l":
            qs = np.zeros((n_frames, 4), dtype=np.float32)
            for f in range(n_frames):
                ang = (np.pi / 2) * (f / max(1, n_frames - 1))  # 0 → 90°
                qs[f] = [0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)]
        else:
            qs = np.tile(np.array([0, 0, 0, 1], np.float32), (n_frames, 1))
        add_ch(JOINT_INDEX[j], "rotation", b.add(qs, "VEC4", FLOAT))

    ph = heads["pelvis"]
    ts = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        t = f / max(1, n_frames - 1)
        ts[f] = [ph[0], ph[1], ph[2] + 0.3 * t]  # 前移 +Z
    add_ch(JOINT_INDEX["pelvis"], "translation", b.add(ts, "VEC3", FLOAT))

    nodes = []
    for j in JOINTS:
        p = PARENTS[j]
        ph2 = np.zeros(3) if p is None else heads[p]
        local = (heads[j] - ph2).tolist()
        kids = [JOINT_INDEX[c] for c in children_of(j)]
        nodes.append(Node(name=SEMANTIC_TO_MIXAMO[j], translation=local,
                          children=kids or None))

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-source"),
        scene=0, scenes=[Scene(nodes=[0])], nodes=nodes,
        skins=[Skin(name="src_rig", inverseBindMatrices=acc_ibm,
                    skeleton=0, joints=list(range(len(JOINTS))))],
        animations=[Animation(name="src_anim", samplers=samplers, channels=channels)],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))
    return path.read_bytes()


def _humanoid_mesh(height: float = 1.8, radius: float = 0.22,
                   rings: int = 18, seg: int = 12):
    """程序生成的人形（花瓶状）网格：沿 Y 分布多圈顶点，便于检验蒙皮。"""
    positions = []
    for r in range(rings + 1):
        y = height * r / rings
        rad = radius * (0.55 + 0.45 * np.sin(np.pi * (r / rings)))
        for s in range(seg):
            th = 2 * np.pi * s / seg
            positions.append([rad * np.cos(th), y, rad * np.sin(th)])
    positions = np.array(positions, dtype=np.float32)
    indices = []
    for r in range(rings):
        for s in range(seg):
            s2 = (s + 1) % seg
            a, bb = r * seg + s, r * seg + s2
            c, d = (r + 1) * seg + s, (r + 1) * seg + s2
            indices += [a, c, bb, bb, c, d]
    return positions, np.array(indices, dtype=np.uint32)


def _build_target_glb(path) -> bytes:
    """无骨骼目标网格 GLB（仅 POSITION + indices）。"""
    positions, indices = _humanoid_mesh()
    b = _Builder()
    acc_pos = b.add(positions, "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    acc_idx = b.add(indices, "SCALAR", UNSIGNED_INT, ELEMENT_ARRAY_BUFFER)
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-target"),
        scene=0, scenes=[Scene(nodes=[0])],
        nodes=[Node(name="target_mesh", mesh=0)],
        meshes=[Mesh(primitives=[Primitive(
            attributes=Attributes(POSITION=acc_pos), indices=acc_idx)])],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))
    return path.read_bytes()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import ai3d.retarget.settings as S
    import ai3d.retarget.task_store as TS
    import ai3d.retarget.worker as W

    monkeypatch.setenv("RETARGET_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.setenv("RETARGET_DB_PATH", str(tmp_path / "proj" / "data" / "test.db"))
    S.get_settings(reload=True)
    TS.get_store(reload=True)
    W.get_worker(reload=True)
    return S.get_settings()


@pytest.fixture()
def client(env):
    from ai3d.retarget.server import create_app
    with TestClient(create_app()) as c:
        yield c


def _wait(client, task_id, states, timeout=20.0):
    end = time.time() + timeout
    info = None
    while time.time() < end:
        info = client.get(f"/v1/jobs/{task_id}").json()
        if info["state"] in states:
            return info
        time.sleep(0.05)
    return info


# --------------------------------------------------------------------------- #
# 单元测试：重定向数学（identity-rest 源 → 精确增量传递）
# --------------------------------------------------------------------------- #
def test_retarget_animation_identity_source(tmp_path):
    src = tmp_path / "src.glb"
    _build_source_glb(src)
    g = glb_io._load(src)
    skin = glb_io.extract_skin(g)
    source_joints = [{"node": ni, "name": nm}
                     for ni, nm in zip(skin["joints"], skin["names"])]
    mapping = build_mapping(source_joints)
    assert mapping["matched"] == len(JOINTS), mapping["items"]

    rig = generate_rig_from_bbox([-.3, 0, -.3], [.3, 1.8, .3], confidence=0.9)
    res = retarget_animation(src, mapping, rig)
    assert res["meta"]["mapped_joints"] == len(JOINTS)
    # identity-rest：upper_arm_l 末帧应为绕 Z 90°（w≈cos45°≈0.707）
    q_last = res["rotations"]["upper_arm_l"][-1]
    assert abs(q_last[2] - np.sin(np.pi / 4)) < 1e-3
    assert abs(q_last[3] - np.cos(np.pi / 4)) < 1e-3
    # 静止关节保持单位四元数
    q_head = res["rotations"]["head"][-1]
    assert abs(q_head[3] - 1.0) < 1e-5
    # root motion：骨盆沿 +Z 前移（按身高缩放）
    rt = res["root_translations"]
    assert rt[-1][2] > rt[0][2] + 1e-3
    assert np.allclose(rt[0][:2], np.asarray(rig.heads["pelvis"])[:2], atol=1e-4)


def test_skin_weights_normalized(tmp_path):
    positions, indices = _humanoid_mesh()
    rig = generate_rig_from_bbox(positions.min(0), positions.max(0))
    joints, weights, _report = compute_skin_weights(positions, indices, rig)
    assert joints.shape == (len(positions), 4)
    assert weights.shape == (len(positions), 4)
    sums = weights.sum(1)
    assert np.allclose(sums, 1.0, atol=1e-4), sums[:5]
    assert joints.max() < len(JOINTS)
    assert (weights >= 0).all()


# --------------------------------------------------------------------------- #
# 端到端：上传 → 确定性闭环 → result.glb 回读验证
# --------------------------------------------------------------------------- #
def test_p1_deterministic_retarget_e2e(client, env, tmp_path):
    src_bytes = _build_source_glb(tmp_path / "src.glb")
    tgt_bytes = _build_target_glb(tmp_path / "tgt.glb")
    r = client.post(
        "/v1/jobs",
        files={
            "source_file": ("src.glb", src_bytes, "model/gltf-binary"),
            "target_file": ("tgt.glb", tgt_bytes, "model/gltf-binary"),
        },
        data={"config": json.dumps({"enable_pose_ai": False})},
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]

    info = _wait(client, task_id, {"DONE", "FAILED", "WAITING"})
    assert info["state"] == "DONE", info
    by_stage = {s["stage"]: s["status"] for s in info["stages"]}
    # 未启用 AI：依赖视角图/推理/求解的阶段被跳过
    for st in ("RENDER_VIEWS", "POSE_INFER", "SOLVE_RIG"):
        assert by_stage[st] == "SKIPPED", (st, info)
    for st in ("NORMALIZE", "PRECHECK", "BUILD_RIG", "MAP_SOURCE",
               "RETARGET", "EXPORT_VERIFY"):
        assert by_stage[st] == "DONE", (st, info)

    # result.glb 可下载且为合法 GLB
    rr = client.get(f"/v1/jobs/{task_id}/result.glb")
    assert rr.status_code == 200
    assert rr.content[:4] == b"glTF"

    result_path = env.task_dir(task_id) / "result.glb"
    assert result_path.exists()
    summary = glb_io.read_summary(result_path)
    assert summary["meshes"] >= 1
    assert summary["has_skeleton"] and summary["has_animation"]
    assert summary["skin_joint_counts"][0] == len(JOINTS)
    # 22 关节 rotation + 1 骨盆 translation
    assert sum(summary["anim_channel_counts"]) == len(JOINTS) + 1

    # 回读验证报告
    report = json.loads((env.task_dir(task_id) / "report.json").read_text(encoding="utf-8"))
    assert report["verify"]["ok"] is True, report["verify"]["problems"]
    assert report["mapping"]["matched"] == len(JOINTS)

    # 动画确实把 upper_arm_l 的旋转迁移进了结果
    g = glb_io._load(result_path)
    anims = glb_io.extract_animations(g)
    rot = {ch["target_node"]: ch for ch in anims[0]["channels"] if ch["path"] == "rotation"}
    q_last = rot[JOINT_INDEX["upper_arm_l"]]["values"][-1]
    assert abs(q_last[3]) < 0.99, q_last  # 明显非单位旋转


def test_p1_rerun_from_retarget(client, env, tmp_path):
    """rerun：从 RETARGET 重跑，复用已生成的 rig/mapping，仍产出 result.glb。"""
    src_bytes = _build_source_glb(tmp_path / "src.glb")
    tgt_bytes = _build_target_glb(tmp_path / "tgt.glb")
    r = client.post(
        "/v1/jobs",
        files={
            "source_file": ("src.glb", src_bytes, "model/gltf-binary"),
            "target_file": ("tgt.glb", tgt_bytes, "model/gltf-binary"),
        },
        data={"config": json.dumps({"enable_pose_ai": False})},
    )
    task_id = r.json()["task_id"]
    info = _wait(client, task_id, {"DONE", "FAILED", "WAITING"})
    assert info["state"] == "DONE", info

    rr = client.post(f"/v1/jobs/{task_id}/rerun", json={"from_stage": "RETARGET"})
    assert rr.status_code == 200, rr.text
    info2 = _wait(client, task_id, {"DONE", "FAILED", "WAITING"})
    assert info2["state"] == "DONE", info2
    assert (env.task_dir(task_id) / "result.glb").exists()
    # rig/mapping 在 RETARGET 上游，未被清理
    assert (env.task_dir(task_id) / "rig.json").exists()
    assert (env.task_dir(task_id) / "mapping.json").exists()
