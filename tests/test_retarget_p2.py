"""P2 定向验证：多视角 2D 关键点 → 加权三角化 + 约束 → 三维关节（solve）+ worker 接线。

不加载真实 DWPose/YOLO 模型：用 ``solve.project`` 把已知 3D 姿态按与后端一致的相机投影成
合成 2D 检测（mock 推理），验证：
  1) 三角化数学可逆（project→triangulate 精确还原）；
  2) solve_rig 从多视角检测恢复任意 3D 姿态（关掉软约束时精确）；
  3) 左右对称软约束确实降低不对称（且为“软”而非强制）；
  4) worker 闭环：上传→RENDER_VIEWS(等图)→回传视角图→POSE_INFER→SOLVE_RIG→…→DONE。
不联网、只写 tmp_path（RETARGET_PROJECT_DIR / RETARGET_DB_PATH 重定向）。
运行： <venv>/bin/python -m pytest tests/test_retarget_p2.py -v
"""

from __future__ import annotations

import json

import numpy as np

from ai3d.retarget import glb_io, solve
from ai3d.retarget.schemas import DEFAULT_CAMERAS, ArtifactKind
from ai3d.retarget.settings import SolveConfig
from ai3d.retarget.skeleton import JOINTS, generate_rig_from_bbox

# 复用 P1 的构造器与 fixture（同目录，pytest 会把 tests/ 加入 sys.path）
from test_retarget_p1 import (  # noqa: E402
    _build_source_glb,
    _build_target_glb,
    _wait,
    client,  # fixture
    env,     # fixture
)

# 前端回传的视角图仅需文件名（stem=view_id）；mock 不解码像素，占位字节即可
_IMG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 48


# --------------------------------------------------------------------------- #
# 合成检测：把 3D 关键点按 DEFAULT_CAMERAS 投影成多视角 2D
# --------------------------------------------------------------------------- #
def _cams(center, max_dim, width=512, height=512):
    return {vid: solve.build_camera(az, el, center, max_dim, width, height)
            for vid, az, el in DEFAULT_CAMERAS}


def _cameras_arg():
    return {vid: (az, el) for vid, az, el in DEFAULT_CAMERAS}


def _synthesize(gt, cams, conf=0.95):
    """{name: xyz} → {view_id: {name: {"uv":[u,v],"conf":c}}}。"""
    return {vid: {name: {"uv": solve.project(np.asarray(X, float), cam).tolist(),
                         "conf": conf}
                  for name, X in gt.items()}
            for vid, cam in cams.items()}


# 一个“非比例”的已知 3D 姿态（COCO/DWPose 命名），用于验证三角化不是回退到先验
_GT = {
    "left_shoulder": [0.18, 1.40, 0.00], "right_shoulder": [-0.18, 1.40, 0.00],
    "left_elbow": [0.45, 1.40, 0.02], "right_elbow": [-0.45, 1.40, 0.02],
    "left_wrist": [0.72, 1.40, 0.05], "right_wrist": [-0.72, 1.40, 0.05],
    "left_hip": [0.10, 0.95, 0.00], "right_hip": [-0.10, 0.95, 0.00],
    "left_knee": [0.11, 0.50, 0.01], "right_knee": [-0.11, 0.50, 0.01],
    "left_ankle": [0.10, 0.08, 0.00], "right_ankle": [-0.10, 0.08, 0.00],
    "left_big_toe": [0.10, 0.02, 0.14], "right_big_toe": [-0.10, 0.02, 0.14],
    "nose": [0.00, 1.62, 0.06],
    "left_ear": [0.07, 1.60, 0.00], "right_ear": [-0.07, 1.60, 0.00],
}
_BBOX_MIN = [-0.8, 0.0, -0.2]
_BBOX_MAX = [0.8, 1.7, 0.2]
# 语义关节 ← 直接检测关键点（solve._DIRECT 的反查）
_DIRECT_CHECK = {
    "upper_arm_l": "left_shoulder", "lower_arm_l": "left_elbow", "hand_l": "left_wrist",
    "upper_arm_r": "right_shoulder", "lower_arm_r": "right_elbow", "hand_r": "right_wrist",
    "upper_leg_l": "left_hip", "lower_leg_l": "left_knee", "foot_l": "left_ankle",
    "upper_leg_r": "right_hip", "lower_leg_r": "right_knee", "foot_r": "right_ankle",
    "toe_l": "left_big_toe", "toe_r": "right_big_toe",
}


def _bbox_center_maxdim(mn, mx):
    mn = np.asarray(mn, float)
    mx = np.asarray(mx, float)
    return (mn + mx) * 0.5, float(max(mx - mn))


# --------------------------------------------------------------------------- #
# 1) 三角化数学：project → triangulate 精确还原
# --------------------------------------------------------------------------- #
def test_triangulate_roundtrip_exact():
    center, max_dim = _bbox_center_maxdim(_BBOX_MIN, _BBOX_MAX)
    cams = _cams(center, max_dim)
    X_true = np.array([0.17, 1.23, -0.09])
    obs = [solve.project(X_true, cams[vid]) for vid in cams]
    X_rec = solve.triangulate(obs, list(cams.values()))
    assert np.allclose(X_rec, X_true, atol=1e-9), (X_rec, X_true)


# --------------------------------------------------------------------------- #
# 2) solve_rig 恢复任意 3D 姿态（关软约束 → 直接/派生关节精确）
# --------------------------------------------------------------------------- #
def test_solve_rig_recovers_synthetic_pose():
    center, max_dim = _bbox_center_maxdim(_BBOX_MIN, _BBOX_MAX)
    cams = _cams(center, max_dim)
    det = _synthesize(_GT, cams)
    cfg = SolveConfig(symmetry_weight=0.0, proportion_weight=0.0,
                      bone_length_stability=0.0)
    rig = solve.solve_rig(det, _cameras_arg(), _BBOX_MIN, _BBOX_MAX, cfg, 512, 512)

    for j, kp in _DIRECT_CHECK.items():
        assert rig.source[j] == "solved", (j, rig.source[j])
        assert np.allclose(rig.heads[j], np.array(_GT[kp]), atol=1e-6), \
            (j, rig.heads[j], _GT[kp])
    # 派生关节：骨盆=髋中点，胸=肩中点，头=双耳中点
    assert np.allclose(rig.heads["pelvis"], [0.0, 0.95, 0.0], atol=1e-6)
    assert np.allclose(rig.heads["chest"], [0.0, 1.40, 0.0], atol=1e-6)
    assert np.allclose(rig.heads["head"], [0.0, 1.60, 0.0], atol=1e-6)
    # 22 关节齐备且整体置信度高（5 视角全命中）
    assert set(rig.heads.keys()) == set(JOINTS)
    assert rig.overall_confidence() > 0.8, rig.overall_confidence()


# --------------------------------------------------------------------------- #
# 3) 左右对称软约束：降低不对称，但为“软”（不完全塌缩到镜像）
# --------------------------------------------------------------------------- #
def test_solve_rig_symmetry_is_soft_constraint():
    gt = {k: list(v) for k, v in _GT.items()}
    # 制造不对称：左臂上举、右臂下垂
    gt["left_elbow"] = [0.45, 1.70, 0.0]
    gt["left_wrist"] = [0.60, 1.95, 0.0]
    gt["right_elbow"] = [-0.45, 1.10, 0.0]
    gt["right_wrist"] = [-0.60, 0.85, 0.0]
    center, max_dim = _bbox_center_maxdim(_BBOX_MIN, _BBOX_MAX)
    cams = _cams(center, max_dim)
    det = _synthesize(gt, cams)

    off = SolveConfig(symmetry_weight=0.0, proportion_weight=0.0, bone_length_stability=0.0)
    on = SolveConfig(symmetry_weight=0.6, proportion_weight=0.0, bone_length_stability=0.0)
    r_off = solve.solve_rig(det, _cameras_arg(), _BBOX_MIN, _BBOX_MAX, off, 512, 512)
    r_on = solve.solve_rig(det, _cameras_arg(), _BBOX_MIN, _BBOX_MAX, on, 512, 512)

    def asym(rig):
        # center_x=0 → 镜像即 x 取反；累加左臂三关节与其镜像的偏差
        tot = 0.0
        for j in ("upper_arm_l", "lower_arm_l", "hand_l"):
            m = j[:-2] + "_r"
            a = rig.heads[j]
            b = rig.heads[m]
            b_mirror = np.array([-b[0], b[1], b[2]])
            tot += float(np.linalg.norm(a - b_mirror))
        return tot

    a_off, a_on = asym(r_off), asym(r_on)
    assert a_on < a_off * 0.55, (a_on, a_off)   # 明显收敛
    assert a_on > a_off * 0.15, (a_on, a_off)   # 仍是软约束（未强制对称）


# --------------------------------------------------------------------------- #
# 4) worker 闭环：mock 推理 → POSE_INFER/SOLVE_RIG → … → DONE
# --------------------------------------------------------------------------- #
def test_infer_image_formats_skeleton(tmp_path, monkeypatch):
    """infer_image 封装：Skeleton(joints) → {name: {"uv":[u,v], "conf":c}}（不加载真实模型）。"""
    import ai3d.retarget.inference as inference

    class _FakeSkeleton:
        joints = [
            {"name": "nose", "position": [10.0, 20.0, 0.0], "confidence": 0.9},
            {"name": "left_shoulder", "position": [30.0, 40.0, 0.0], "confidence": 0.12},
        ]

    class _FakeEst:
        def estimate_frame(self, bgr):
            return _FakeSkeleton()

    # infer_image 内部用 cv2 读图；此处替换为假图，聚焦验证输出结构
    monkeypatch.setattr(inference, "_read_image",
                        lambda p: np.zeros((8, 8, 3), np.uint8))
    out = inference.infer_image(tmp_path / "x.png", _FakeEst())
    assert out["nose"] == {"uv": [10.0, 20.0], "conf": 0.9}
    assert out["left_shoulder"]["uv"] == [30.0, 40.0]
    assert out["left_shoulder"]["conf"] == 0.12


def test_p2_pose_ai_pipeline_e2e(client, env, tmp_path, monkeypatch):
    import ai3d.retarget.inference as inference
    from ai3d.retarget.task_store import get_store

    src_bytes = _build_source_glb(tmp_path / "src.glb")
    tgt_bytes = _build_target_glb(tmp_path / "tgt.glb")
    r = client.post(
        "/v1/jobs",
        files={
            "source_file": ("src.glb", src_bytes, "model/gltf-binary"),
            "target_file": ("tgt.glb", tgt_bytes, "model/gltf-binary"),
        },
        data={"config": json.dumps({"enable_pose_ai": True})},
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]

    # 启用 AI：先跑到 RENDER_VIEWS 挂起，等前端回传视角图
    info = _wait(client, task_id, {"WAITING", "DONE", "FAILED"})
    assert info["state"] == "WAITING", info
    by_stage = {s["stage"]: s["status"] for s in info["stages"]}
    assert by_stage["RENDER_VIEWS"] == "WAITING", info

    # 用归一化目标的真实 bbox 构造相机与 GT 姿态（与后端 SOLVE_RIG 完全一致）
    store = get_store()
    tgt_path = store.get_artifact_path(task_id, ArtifactKind.TARGET)
    bbox = glb_io.mesh_bbox(glb_io._load(tgt_path))
    center, max_dim = _bbox_center_maxdim(bbox["min"], bbox["max"])
    prior = generate_rig_from_bbox(bbox["min"], bbox["max"], 0.9)
    head = prior.heads["head"]
    gt = {
        "left_shoulder": prior.heads["upper_arm_l"], "right_shoulder": prior.heads["upper_arm_r"],
        "left_elbow": prior.heads["lower_arm_l"], "right_elbow": prior.heads["lower_arm_r"],
        "left_wrist": prior.heads["hand_l"], "right_wrist": prior.heads["hand_r"],
        "left_hip": prior.heads["upper_leg_l"], "right_hip": prior.heads["upper_leg_r"],
        "left_knee": prior.heads["lower_leg_l"], "right_knee": prior.heads["lower_leg_r"],
        "left_ankle": prior.heads["foot_l"], "right_ankle": prior.heads["foot_r"],
        "left_big_toe": prior.heads["toe_l"], "right_big_toe": prior.heads["toe_r"],
        "nose": head,
        "left_ear": head + np.array([0.05, 0.0, 0.0]),
        "right_ear": head - np.array([0.05, 0.0, 0.0]),
    }
    cams = _cams(center, max_dim)

    # mock 推理：绝不加载真实模型；按 view_id 投影 GT
    def fake_load_estimator(settings=None, reload=False):
        return object()

    def fake_infer_views(views_dir, estimator=None, settings=None):
        out = {}
        for p in inference.list_view_images(views_dir):
            cam = cams.get(p.stem)
            if cam is None:
                continue
            out[p.stem] = {name: {"uv": solve.project(np.asarray(X, float), cam).tolist(),
                                  "conf": 0.95}
                           for name, X in gt.items()}
        return out

    monkeypatch.setattr(inference, "load_estimator", fake_load_estimator)
    monkeypatch.setattr(inference, "infer_views", fake_infer_views)

    # 取下发规格（同时持久化 view_spec.json），按其视角回传占位图
    vs = client.get(f"/v1/jobs/{task_id}/view_spec").json()
    view_ids = [c["view_id"] for c in vs["cameras"]]
    files = [("files", (f"{vid}.png", _IMG, "image/png")) for vid in view_ids]
    rv = client.post(f"/v1/jobs/{task_id}/views", files=files)
    assert rv.status_code == 200, rv.text

    info = _wait(client, task_id, {"DONE", "FAILED"}, timeout=30.0)
    assert info["state"] == "DONE", info
    by_stage = {s["stage"]: s["status"] for s in info["stages"]}
    assert by_stage["POSE_INFER"] == "DONE", info
    assert by_stage["SOLVE_RIG"] == "DONE", info
    for st in ("RENDER_VIEWS", "BUILD_RIG", "MAP_SOURCE", "RETARGET", "EXPORT_VERIFY"):
        assert by_stage[st] == "DONE", (st, info)

    # detections.json 落盘且视角齐全
    det = json.loads((env.task_dir(task_id) / "detections.json").read_text(encoding="utf-8"))
    assert set(det.keys()) == set(view_ids)

    # rig.json：直接检测关节标记 solved 且接近 GT
    rig = json.loads((env.task_dir(task_id) / "rig.json").read_text(encoding="utf-8"))
    jl = rig["joints"]["upper_arm_l"]
    assert jl["source"] == "solved", jl
    assert np.allclose(jl["head"], gt["left_shoulder"], atol=0.10), (jl["head"], gt["left_shoulder"])

    # result.glb 合法且含骨架+动画（22 关节）
    rr = client.get(f"/v1/jobs/{task_id}/result.glb")
    assert rr.status_code == 200 and rr.content[:4] == b"glTF"
    summary = glb_io.read_summary(env.task_dir(task_id) / "result.glb")
    assert summary["has_skeleton"] and summary["has_animation"]
    assert summary["skin_joint_counts"][0] == len(JOINTS)
