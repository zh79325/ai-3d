"""S2 绑定定向验证：确认 S1 → 绑定 → 多视角回传 → 关节微调 → 蒙皮重算。

覆盖两条绑定路径与三处易错点：

- ``pose_ai=false``（人体比例骨架 + libigl BBW 蒙皮）能一路跑到 ``READY``，产物落
  ``output/assets/<id>/``（素材级复用，换动画不必重算）；
- ``pose_ai=true`` 但视角图未回传时停在 ``WAIT_VIEWS``（**不是失败**），回传后自动续跑；
- ``PATCH /rig`` 的 revision 乐观锁、``source="manual"`` 标记、以及重算蒙皮**不重写**
  rig.json（人工版本号必须留着）；
- ``POST /reskin`` 只重算蒙皮，不动 revision，且准入校验（缺 rig.json / 动画素材）
  在当前线程同步抛错而不是让素材在后台默默落 FAILED；
- 重跑失败时保留上一次的绑定产物，素材不退回未绑定状态。

不做真实推理（DWPose 的 ``stages.infer_views`` 被打桩）、不联网、只写 tmp_path
（RETARGET_PROJECT_DIR / RETARGET_DB_PATH 重定向）。
运行： <venv>/bin/python -m pytest tests/test_binding.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
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
from scipy.spatial import ConvexHull

from ai3d.retarget import stages
from ai3d.retarget.job_worker import get_job_worker
from ai3d.retarget.skeleton import JOINTS
from ai3d.retarget.glb_build import _Builder

# 1.75m 长方体按厘米落盘 → S1 判 cm，canon 节点带 scale=0.01，S2 拿到的是米制网格
_BOX_CM = (50.0, 175.0, 30.0)
# 1×1 PNG 头 + 填充：内容不重要（推理被打桩），扩展名决定它是否被当成视角图
_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 48


# --------------------------------------------------------------------------- #
# 构造输入文件
# --------------------------------------------------------------------------- #
def _box_surface(size, n: int = 8) -> np.ndarray:
    """长方体表面点（底面贴地：Y ∈ [0, size[1]]），凸包即精确长方体。"""
    sx, sy, sz = (float(v) for v in size)
    g = np.linspace(0.0, 1.0, n)
    pts = []
    for u in g:
        for v in g:
            pts += [[sx * (u - 0.5), 0.0, sz * (v - 0.5)],
                    [sx * (u - 0.5), sy, sz * (v - 0.5)],
                    [sx * (u - 0.5), sy * v, -sz * 0.5],
                    [sx * (u - 0.5), sy * v, sz * 0.5],
                    [-sx * 0.5, sy * u, sz * (v - 0.5)],
                    [sx * 0.5, sy * u, sz * (v - 0.5)]]
    return np.asarray(pts, dtype=np.float32)


def _build_box_glb(path) -> bytes:
    """带索引的闭合长方体 GLB：蒙皮（BBW 四面体化）需要真实三角面，点云不行。"""
    pts = _box_surface(_BOX_CM)
    tri = ConvexHull(pts).simplices.astype(np.uint32)
    b = _Builder()
    acc_pos = b.add(pts, "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    acc_idx = b.add(tri.ravel(), "SCALAR", UNSIGNED_INT, ELEMENT_ARRAY_BUFFER,
                    count=len(tri) * 3)
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-binding"),
        scene=0, scenes=[Scene(nodes=[0])],
        nodes=[Node(name="body", mesh=0)],
        meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=acc_pos),
                                           indices=acc_idx)])],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))
    return path.read_bytes()


def _build_skeleton_glb(path, height: float = 1.75) -> bytes:
    """纯骨架 GLB（无网格）：动画素材的常见形态，S2 应拒绝给它绑骨骼。"""
    b = _Builder()
    b.add(np.zeros((1, 3), dtype=np.float32), "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    h = float(height)
    joints = [("pelvis", [0.0, h * 0.5, 0.0]), ("head", [0.0, h, 0.02]),
              ("arm_l", [-0.2, h * 0.7, 0.05]), ("arm_r", [0.2, h * 0.7, -0.05]),
              ("foot_l", [-0.1, 0.0, 0.06]), ("foot_r", [0.1, 0.0, -0.06])]
    nodes = [Node(name="armature", children=list(range(1, len(joints) + 1)))]
    nodes += [Node(name=n, translation=t) for n, t in joints]
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-binding-skel"),
        scene=0, scenes=[Scene(nodes=[0])], nodes=nodes,
        skins=[Skin(name="rig", skeleton=0, joints=list(range(1, len(joints) + 1)))],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def env(tmp_path, monkeypatch):
    """把项目目录与 SQLite 重定向到 tmp_path，并重载 /v2 的全部单例。"""
    import ai3d.retarget.asset_store as AS
    import ai3d.retarget.asset_worker as AW
    import ai3d.retarget.job_worker as JW
    import ai3d.retarget.settings as S

    monkeypatch.setenv("RETARGET_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.setenv("RETARGET_DB_PATH", str(tmp_path / "proj" / "data" / "test.db"))
    S.get_settings(reload=True)
    AS.get_asset_store(reload=True)
    AW.get_asset_worker(reload=True)
    JW.get_job_worker(reload=True)      # JobWorker 缓存了 store，不重载会写上一个 tmp 库
    return S.get_settings()


@pytest.fixture()
def client(env):
    from ai3d.retarget.server_v2 import create_app
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def box_glb(tmp_path):
    return _build_box_glb(tmp_path / "hero.glb")


@pytest.fixture()
def stub_infer(monkeypatch):
    """打桩 DWPose 推理：不载入 315MB ONNX，只按真实契约写一份空关键点 detections。

    关键点为空时 ``solve.solve_rig`` 全部落到比例先验（``source="proportion"``），
    正好用来验证「视角图 → 推理 → 三角化 → 蒙皮」的编排而不动真实模型。
    """
    calls = []

    def fake_infer(views_dir, detections_path, settings=None):
        names = [p.stem for p in stages.list_view_images(Path(views_dir))]
        calls.append(names)
        detections = {vid: {} for vid in names}
        stages.write_json(Path(detections_path), detections, indent=None)
        return detections

    monkeypatch.setattr(stages, "infer_views", fake_infer)
    return calls


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ready_model(client, box_glb, auto_bind: bool = False) -> str:
    """建模型素材 → 上传跑完 S1 → 确认，返回 asset_id。"""
    r = client.post("/v2/assets", json={"kind": "model", "name": "hero"})
    assert r.status_code == 201, r.text
    aid = r.json()["asset_id"]
    up = client.post(f"/v2/assets/{aid}/upload",
                     files={"file": ("hero.glb", box_glb, "model/gltf-binary")})
    assert up.status_code == 200, up.text
    assert up.json()["state"] == "ALIGN_READY"
    c = client.post(f"/v2/assets/{aid}/confirm", params={"auto_bind": str(auto_bind).lower()})
    assert c.status_code == 200, c.text
    return aid


def _wait_binding(client, aid, want, timeout: float = 90.0) -> dict:
    """轮询 binding 直到状态命中 ``want``（S2 在后台线程跑，同步断言会竞态）。"""
    info: dict = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v2/assets/{aid}/binding")
        assert r.status_code == 200, r.text
        info = r.json()
        if info["state"] in want:
            return info
        if info["state"] == "FAILED":
            pytest.fail(f"S2 意外失败：{info.get('error')}（stage={info.get('stage')}）")
        time.sleep(0.05)
    pytest.fail(f"binding 未在 {timeout}s 内到达 {want}，最后状态：{info}")


def _bind(client, aid, pose_ai: bool = False) -> dict:
    r = client.post(f"/v2/assets/{aid}/bind", params={"pose_ai": str(pose_ai).lower()})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "started"
    return _wait_binding(client, aid, {"READY"})


# --------------------------------------------------------------------------- #
# pose_ai=false：人体比例骨架 + BBW 蒙皮
# --------------------------------------------------------------------------- #
def test_bind_without_pose_ai_produces_rig_and_skin(client, env, box_glb):
    """``pose_ai=false`` 一路跑到 READY：22 关节 + 蒙皮产物 + 报告，全落素材目录。"""
    aid = _ready_model(client, box_glb)
    info = _bind(client, aid, pose_ai=False)

    assert info["state"] == "READY"
    assert info["stage"] is None and info["error"] is None
    assert info["has_rig"] and info["has_skin"]
    assert info["revision"] == 0
    assert 0.0 < info["confidence"] <= 1.0
    assert "骨架" in info["message"] and "蒙皮" in info["message"]

    adir = env.asset_dir(aid)
    assert (adir / "rig.json").exists() and (adir / "skin.npz").exists()
    assert (adir / "skin_report.json").exists()
    # 素材级复用：产物写在 output/assets/<id>/，不是某个作业目录下
    assert adir.parent.name == "assets"

    rig = client.get(f"/v2/assets/{aid}/rig").json()
    assert set(rig["joints"]) == set(JOINTS)
    assert rig["joints"]["pelvis"]["source"] == "proportion"
    assert rig["height"] == pytest.approx(1.75, abs=1e-2)      # S1 的 cm→m 已在网格里

    skin = np.load(adir / "skin.npz")
    assert skin["joints"].shape[0] == skin["weights"].shape[0]
    assert skin["joints"].shape[1] == 4 and skin["weights"].shape[1] == 4
    assert np.allclose(skin["weights"].sum(1), 1.0, atol=1e-5)   # 权重归一，无零权重行

    report = client.get(f"/v2/assets/{aid}/skin_report").json()
    assert report["zero_rows"] == 0
    assert report["components"]["count"] >= 1
    # 焊接后的求解域不可能多于原顶点（长方体表面点有重复，会被 weld 合掉）
    assert 0 < report["weld"]["clusters"] <= skin["joints"].shape[0]


def test_binding_is_cached_and_reused_across_jobs(client, box_glb):
    """换动画不重算蒙皮：绑定完成后再 ``GET /binding`` 结果稳定，重跑才覆盖。"""
    aid = _ready_model(client, box_glb)
    first = _bind(client, aid, pose_ai=False)
    again = client.get(f"/v2/assets/{aid}/binding").json()
    assert again["confidence"] == first["confidence"]
    assert again["revision"] == first["revision"]
    assert again["state"] == "READY"
    # 素材详情内嵌同一份 binding（前端一次拉取即可）
    assert client.get(f"/v2/assets/{aid}").json()["binding"]["state"] == "READY"


# --------------------------------------------------------------------------- #
# pose_ai=true：视角图回传前后
# --------------------------------------------------------------------------- #
def test_confirm_auto_starts_binding_and_waits_for_views(client, box_glb):
    """确认 S1 即自动起 S2；无视角图时停在 WAIT_VIEWS（不是失败），并下发渲染规格。"""
    aid = _ready_model(client, box_glb, auto_bind=True)
    info = _wait_binding(client, aid, {"WAIT_VIEWS"})
    assert info["stage"] == "RENDER_VIEWS"
    assert info["error"] is None
    assert "视角图" in info["message"]
    assert info["has_rig"] is False           # 挂起时不该产出半成品

    spec = client.get(f"/v2/assets/{aid}/view_spec")
    assert spec.status_code == 200, spec.text
    body = spec.json()
    assert body["model_url"] == f"/v2/assets/{aid}/glb"
    assert body["ortho"] is True and body["width"] == body["height"] == 512
    assert [c["view_id"] for c in body["cameras"]] == [
        "front", "three_quarter", "right", "back", "left"]
    # 下发即持久化：SOLVE_RIG 必须读同一份相机参数，否则三角化整体错位
    assert client.get(f"/v2/assets/{aid}/glb").status_code == 200


def test_upload_views_resumes_binding(client, env, box_glb, stub_infer):
    """回传视角图 → 自动续跑 POSE_INFER → SOLVE_RIG → BUILD_RIG，直到 READY。"""
    aid = _ready_model(client, box_glb, auto_bind=True)
    _wait_binding(client, aid, {"WAIT_VIEWS"})
    client.get(f"/v2/assets/{aid}/view_spec")       # 落 view_spec.json

    r = client.post(f"/v2/assets/{aid}/views",
                    files=[("files", ("front.png", _FAKE_PNG, "image/png")),
                           ("files", ("right.png", _FAKE_PNG, "image/png"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["views"] == ["front.png", "right.png"]
    assert body["resumed"] is True

    info = _wait_binding(client, aid, {"READY"})
    assert info["has_rig"] and info["has_skin"]
    assert stub_infer == [["front", "right"]]        # 打桩确实被调用，且按 view_id 命名
    assert (env.asset_dir(aid) / "detections.json").exists()
    # 空关键点 → 全部落比例先验，但骨架结构与蒙皮照常产出
    rig = client.get(f"/v2/assets/{aid}/rig").json()
    assert set(rig["joints"]) == set(JOINTS)
    assert rig["joints"]["head"]["source"] == "proportion"


def test_upload_views_replaces_previous_and_rejects_non_images(client, box_glb):
    """回传前先清空 ``views/``（残留视角会配错相机）；非图片文件一律 400。

    故意用 ``auto_bind=false``：本用例只验证图文件的落盘/清单/下载，不要让它顺手
    触发一轮真的绑定（后台线程会与第二次回传抢 ``views/`` 目录）。
    """
    aid = _ready_model(client, box_glb)
    first = client.post(f"/v2/assets/{aid}/views",
                        files=[("files", ("front.png", _FAKE_PNG, "image/png")),
                               ("files", ("back.png", _FAKE_PNG, "image/png"))])
    assert first.status_code == 200, first.text
    assert first.json()["resumed"] is False          # 未进入 WAIT_VIEWS，不自动续跑
    assert client.get(f"/v2/assets/{aid}/views").json()["views"] == [
        "back.png", "front.png"]

    second = client.post(f"/v2/assets/{aid}/views",
                         files=[("files", ("left.png", _FAKE_PNG, "image/png"))])
    assert second.status_code == 200, second.text
    listed = client.get(f"/v2/assets/{aid}/views").json()
    assert listed["views"] == ["left.png"]           # 上一轮的两张已被清掉
    assert listed["urls"] == [f"/v2/assets/{aid}/views/left.png"]

    img = client.get(f"/v2/assets/{aid}/views/left.png")
    assert img.status_code == 200 and img.headers["Cache-Control"] == "no-store"
    assert client.get(f"/v2/assets/{aid}/views/nope.png").status_code == 404
    assert client.post(f"/v2/assets/{aid}/views",
                       files=[("files", ("a.txt", b"x", "text/plain"))]).status_code == 400


# --------------------------------------------------------------------------- #
# 关节微调：revision 乐观锁 + 蒙皮重算
# --------------------------------------------------------------------------- #
def test_patch_rig_uses_optimistic_lock_and_marks_manual(client, box_glb):
    """PATCH /rig 递增 revision 并把动过的关节标 ``manual``；旧 revision 再改 → 409。"""
    aid = _ready_model(client, box_glb)
    _bind(client, aid, pose_ai=False)
    before = client.get(f"/v2/assets/{aid}/rig").json()["joints"]["pelvis"]["head"]

    patch = {"revision": 0, "joints": {"pelvis": {"head": [0.0, 0.95, 0.0]}}}
    r = client.patch(f"/v2/assets/{aid}/rig", json=patch,
                     params={"reskin": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rig_revision"] == 1
    assert body["reskin"] == "skipped"

    rig = client.get(f"/v2/assets/{aid}/rig").json()
    assert rig["revision"] == 1
    assert rig["joints"]["pelvis"]["head"] == [0.0, 0.95, 0.0]
    assert rig["joints"]["pelvis"]["source"] == "manual"
    assert rig["joints"]["head"]["source"] == "proportion"      # 没动的关节不标记
    assert rig["joints"]["pelvis"]["head"] != before
    assert client.get(f"/v2/assets/{aid}/binding").json()["revision"] == 1

    # 乐观锁：拿旧 revision 再改 → 409，且 rig.json 不被覆盖
    stale = client.patch(f"/v2/assets/{aid}/rig", json=patch, params={"reskin": "false"})
    assert stale.status_code == 409
    assert "revision" in stale.json()["detail"]
    assert client.get(f"/v2/assets/{aid}/rig").json()["revision"] == 1


def test_reskin_keeps_manual_rig_and_revision(client, env, box_glb):
    """改关节后重算蒙皮：``rig.json`` 与人工 revision 保持不动，只换 skin.npz。"""
    aid = _ready_model(client, box_glb)
    _bind(client, aid, pose_ai=False)
    skin = env.asset_dir(aid) / "skin.npz"
    skin_mtime = skin.stat().st_mtime_ns

    client.patch(f"/v2/assets/{aid}/rig",
                 json={"revision": 0, "joints": {"head": {"head": [0.0, 1.7, 0.0]}}},
                 params={"reskin": "false"})
    r = client.patch(f"/v2/assets/{aid}/rig",
                     json={"revision": 1, "joints": {"pelvis": {"head": [0.0, 0.9, 0.0]}}},
                     params={"reskin": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["reskin"] == "started"
    assert r.json()["binding"]["state"] == "RUNNING"      # 路由返回时已置忙，不会误读 READY
    rig_after_patch = (env.asset_dir(aid) / "rig.json").read_bytes()
    info = _wait_binding(client, aid, {"READY"})

    assert info["revision"] == 2                 # 只有 PATCH 递增，reskin 不动它
    assert "重算蒙皮" in info["message"]
    assert (env.asset_dir(aid) / "rig.json").read_bytes() == rig_after_patch
    rig = client.get(f"/v2/assets/{aid}/rig").json()
    assert rig["revision"] == 2
    assert rig["joints"]["head"]["head"] == [0.0, 1.7, 0.0]     # 上一次的人工值还在
    assert rig["joints"]["head"]["source"] == "manual"
    assert rig["joints"]["pelvis"]["source"] == "manual"
    assert skin.stat().st_mtime_ns > skin_mtime  # 蒙皮确实被重写


def test_reskin_endpoint_recomputes_skin_only(client, env, box_glb):
    """``POST /reskin``：只重算蒙皮，revision 与 rig.json 一个字节都不动。

    连着微调几根关节（每次都 ``reskin=false`` 只存骨架）后用它一次性重算。走 PATCH
    带空补丁也能达到同样效果，但那会白白把 revision +1，还会把没动过的关节标成人工。
    """
    aid = _ready_model(client, box_glb)
    _bind(client, aid, pose_ai=False)
    client.patch(f"/v2/assets/{aid}/rig",
                 json={"revision": 0, "joints": {"head": {"head": [0.0, 1.7, 0.0]}}},
                 params={"reskin": "false"})
    skin = env.asset_dir(aid) / "skin.npz"
    skin_mtime = skin.stat().st_mtime_ns
    rig_before = (env.asset_dir(aid) / "rig.json").read_bytes()

    r = client.post(f"/v2/assets/{aid}/reskin")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reskin"] is True
    assert body["binding"]["state"] == "RUNNING"     # 同步置忙，前端不会误读上一轮 READY
    info = _wait_binding(client, aid, {"READY"})

    assert info["revision"] == 1                 # PATCH 加的那一次，reskin 不再动
    assert "重算蒙皮" in info["message"]
    assert info["has_rig"] and info["has_skin"]
    assert (env.asset_dir(aid) / "rig.json").read_bytes() == rig_before
    assert skin.stat().st_mtime_ns > skin_mtime
    rig = client.get(f"/v2/assets/{aid}/rig").json()
    assert rig["revision"] == 1
    assert rig["joints"]["head"]["source"] == "manual"
    assert rig["joints"]["pelvis"]["source"] == "proportion"   # 没动过的不被标人工


def test_reskin_gates(client, tmp_path, box_glb):
    """缺 rig.json / 动画素材 → 400（同步抛，不会让素材莫名落 FAILED）；不存在 → 404。"""
    aid = _ready_model(client, box_glb)          # 只跑完 S1，还没有 rig.json
    r = client.post(f"/v2/assets/{aid}/reskin")
    assert r.status_code == 400, r.text
    assert "rig.json 尚未生成" in r.json()["detail"]
    # 同步抛错就不能碰 bindings 行：confirm 预置的 PENDING 与 error=None 原封不动
    after = client.get(f"/v2/assets/{aid}/binding").json()
    assert after["state"] == "PENDING" and after["error"] is None

    anim_id = client.post("/v2/assets", json={"kind": "animation", "name": "run"}
                          ).json()["asset_id"]
    client.post(f"/v2/assets/{anim_id}/upload",
                files={"file": ("run.glb", _build_skeleton_glb(tmp_path / "run.glb"),
                                "model/gltf-binary")})
    client.post(f"/v2/assets/{anim_id}/confirm")
    bad = client.post(f"/v2/assets/{anim_id}/reskin")
    assert bad.status_code == 400, bad.text
    assert "模型素材" in bad.json()["detail"]
    assert client.post("/v2/assets/nope123/reskin").status_code == 404


def test_rebind_failure_keeps_previous_artifacts(client, env, box_glb, monkeypatch):
    """重跑 S2 失败时保留上一次的 rig/skin：素材不能退回「必须重绑才能进 S3」。"""
    aid = _ready_model(client, box_glb)
    good = _bind(client, aid, pose_ai=False)
    assert good["state"] == "READY"
    rig_before = json.loads((env.asset_dir(aid) / "rig.json").read_text(encoding="utf-8"))

    def boom(*args, **kwargs):
        raise RuntimeError("蒙皮求解器炸了（测试注入）")

    monkeypatch.setattr(stages, "build_rig", boom)
    r = client.post(f"/v2/assets/{aid}/rebind", params={"pose_ai": "false"})
    assert r.status_code == 200, r.text
    info = _wait_binding(client, aid, {"READY"})
    assert info["error"] and "蒙皮求解器炸了" in info["error"]
    assert info["has_rig"] and info["has_skin"]
    assert info["confidence"] == good["confidence"]
    assert json.loads(
        (env.asset_dir(aid) / "rig.json").read_text(encoding="utf-8")) == rig_before


# --------------------------------------------------------------------------- #
# 门槛与错误码
# --------------------------------------------------------------------------- #
def test_binding_gates_and_error_codes(client, tmp_path, box_glb, monkeypatch):
    """S1 未确认 → 409；动画素材 → 400；S2 在跑 → 409；缺产物 → 404。"""
    from ai3d.retarget.job_worker import JobWorker

    aid = _ready_model(client, box_glb)
    # 动画素材不需要（也不允许）绑骨骼
    anim = client.post("/v2/assets", json={"kind": "animation", "name": "run"}).json()
    anim_id = anim["asset_id"]
    skel = _build_skeleton_glb(tmp_path / "run.glb")
    client.post(f"/v2/assets/{anim_id}/upload",
                files={"file": ("run.glb", skel, "model/gltf-binary")})
    client.post(f"/v2/assets/{anim_id}/confirm")
    bad = client.post(f"/v2/assets/{anim_id}/bind")
    assert bad.status_code == 400, bad.text
    assert "模型素材" in bad.json()["detail"]
    assert client.patch(f"/v2/assets/{anim_id}/rig",
                        json={"revision": 0, "joints": {}}).status_code == 400

    # 未上传/未确认 S1 的素材禁止绑定（米制绝对阈值依赖单位已矫正）
    raw = client.post("/v2/assets", json={"kind": "model"}).json()["asset_id"]
    assert client.post(f"/v2/assets/{raw}/bind").status_code == 409
    uploaded = client.post("/v2/assets", json={"kind": "model"}).json()["asset_id"]
    client.post(f"/v2/assets/{uploaded}/upload",
                files={"file": ("hero.glb", box_glb, "model/gltf-binary")})
    assert client.post(f"/v2/assets/{uploaded}/bind").status_code == 200

    # 素材不存在
    assert client.post("/v2/assets/nope123/bind").status_code == 404
    assert client.get("/v2/assets/nope123/rig").status_code == 404
    # rig / 报告未生成
    assert client.get(f"/v2/assets/{aid}/rig").status_code == 404
    assert client.get(f"/v2/assets/{aid}/skin_report").status_code == 404
    assert client.patch(f"/v2/assets/{aid}/rig",
                        json={"revision": 0, "joints": {}}).status_code == 400

    # 已有 S2 在跑 → 409（per-asset 锁，避免两个线程互相覆盖 rig.json）
    monkeypatch.setattr(JobWorker, "is_busy", lambda self, asset_id: True)
    busy = client.post(f"/v2/assets/{aid}/rebind", params={"pose_ai": "false"})
    assert busy.status_code == 409, busy.text
    assert "正在执行" in busy.json()["detail"]
    monkeypatch.undo()
    assert get_job_worker().is_busy(aid) is False


def test_binding_survives_asset_delete(client, box_glb):
    """删素材连带 bindings（FK CASCADE）与产物目录，不留孤儿蒙皮。"""
    aid = _ready_model(client, box_glb)
    _bind(client, aid, pose_ai=False)
    assert client.get(f"/v2/assets/{aid}/binding").status_code == 200
    assert client.delete(f"/v2/assets/{aid}").status_code == 200
    assert client.get(f"/v2/assets/{aid}/binding").status_code == 404
    assert client.get(f"/v2/assets/{aid}/rig").status_code == 404
