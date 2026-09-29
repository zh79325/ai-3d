"""``/v2`` 素材库接口定向验证：建资产 → 上传 → S1 提案 → 人工修正 → 确认 → 删除。

用程序生成的长方体网格 GLB 与纯骨架 GLB 覆盖：cm 尺度自动推断、canon 节点旋转/缩放
落盘、面映射 PATCH 重算、单位与真实身高覆盖、动画素材走关节回退、各错误码，以及
``/v1`` 四条常用路由在同一应用内仍 200（两套 API 并存不互相污染）。

不做真实模型推理、不联网、只写 tmp_path（RETARGET_PROJECT_DIR / RETARGET_DB_PATH 重定向）。
运行： <venv>/bin/python -m pytest tests/test_server_v2.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pygltflib import (
    ARRAY_BUFFER,
    FLOAT,
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

from ai3d.retarget import axis_norm, glb_io
from ai3d.retarget.axis_norm import CANON_NODE
from ai3d.retarget.glb_build import _Builder

# 1.75m 长方体按厘米落盘 → 应被推断为 cm，canon 节点带 scale=0.01
_BOX_CM = (50.0, 175.0, 30.0)


# --------------------------------------------------------------------------- #
# 构造输入文件
# --------------------------------------------------------------------------- #
def _box_surface(size, n: int = 12) -> np.ndarray:
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


def _build_mesh_glb(path, size=_BOX_CM) -> bytes:
    """无骨骼网格 GLB（仅 POSITION）：走纯网格探测路径。"""
    b = _Builder()
    acc = b.add(_box_surface(size), "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-v2-mesh"),
        scene=0, scenes=[Scene(nodes=[0])],
        nodes=[Node(name="body", mesh=0)],
        meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=acc))])],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))
    return path.read_bytes()


def _build_skeleton_glb(path, height: float = 1.75) -> bytes:
    """纯骨架 GLB（6 关节、无网格）：动画素材的常见形态，走关节位置回退量 OBB。

    关节带微小 Z 偏移：全部共面时凸包无法构造，OBB 会退化。这里靠 pelvis→head
    的人形先验定 up，身高（Y 跨度）就是 ``height``。
    """
    b = _Builder()
    b.add(np.zeros((1, 3), dtype=np.float32), "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    h = float(height)
    joints = [
        ("pelvis", [0.0, h * 0.5, 0.0]),
        ("head", [0.0, h, 0.02]),
        ("arm_l", [-0.2, h * 0.7, 0.05]),
        ("arm_r", [0.2, h * 0.7, -0.05]),
        ("foot_l", [-0.1, 0.0, 0.06]),
        ("foot_r", [0.1, 0.0, -0.06]),
    ]
    nodes = [Node(name="armature", children=list(range(1, len(joints) + 1)))]
    nodes += [Node(name=n, translation=t) for n, t in joints]
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-v2-skin"),
        scene=0, scenes=[Scene(nodes=[0])], nodes=nodes,
        skins=[Skin(name="rig", skeleton=0,
                    joints=list(range(1, len(joints) + 1)))],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))
    return path.read_bytes()


def _canon_node(path):
    """产物 GLB 里的 canon 节点（不存在返回 None）。"""
    g = glb_io._load(path)
    return next((n for n in (g.nodes or []) if (n.name or "") == CANON_NODE), None)


def _metric_span(path) -> np.ndarray:
    """对齐后 merge_mesh 的 AABB 跨度（米制）。"""
    p = glb_io.merge_mesh(glb_io._load(path))["positions"]
    return (p.max(0) - p.min(0)).astype(np.float64)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def env(tmp_path, monkeypatch):
    """把项目目录与 SQLite 重定向到 tmp_path，并重载 /v1 + /v2 全部单例。"""
    import ai3d.retarget.asset_store as AS
    import ai3d.retarget.asset_worker as AW
    import ai3d.retarget.job_worker as JW
    import ai3d.retarget.settings as S
    import ai3d.retarget.task_store as TS
    import ai3d.retarget.worker as W

    monkeypatch.setenv("RETARGET_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.setenv("RETARGET_DB_PATH", str(tmp_path / "proj" / "data" / "test.db"))
    S.get_settings(reload=True)
    TS.get_store(reload=True)
    W.get_worker(reload=True)
    AS.get_asset_store(reload=True)
    AW.get_asset_worker(reload=True)
    JW.get_job_worker(reload=True)      # JobWorker 缓存了 store，不重载会写上一个 tmp 库
    return S.get_settings()


@pytest.fixture()
def client(env):
    """主应用（``/v1`` 与 ``/v2`` 并存），与 ai3d/viewer/server.py 的挂载方式一致。"""
    from ai3d.retarget.server import create_app
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def mesh_glb(tmp_path):
    return _build_mesh_glb(tmp_path / "hero.glb")


def _create(client, kind="model", name="hero") -> str:
    r = client.post("/v2/assets", json={"kind": kind, "name": name})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["state"] == "CREATED"
    return body["asset_id"]


def _upload(client, asset_id, data, filename="hero.glb") -> dict:
    r = client.post(f"/v2/assets/{asset_id}/upload",
                    files={"file": (filename, data, "model/gltf-binary")})
    assert r.status_code == 200, r.text
    return r.json()


def _get_align(client, asset_id) -> dict:
    r = client.get(f"/v2/assets/{asset_id}/align")
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# 约定 / 健康
# --------------------------------------------------------------------------- #
def test_conventions_and_health(client):
    """面编号与语义轴约定由后端单点下发，前端不各写一份。"""
    c = client.get("/v2/conventions")
    assert c.status_code == 200, c.text
    body = c.json()
    assert body["face_labels"] == ["u+", "u-", "v+", "v-", "w+", "w-"]
    assert body["face_semantics"] == ["up+", "up-", "front+", "front-", "left+", "left-"]
    assert body["canonical_axes"] == {"up": "+Y", "front": "+Z", "left": "+X"}
    assert set(body["asset_kinds"]) == {"model", "animation"}

    h = client.get("/v2/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"


# --------------------------------------------------------------------------- #
# S1 主流程
# --------------------------------------------------------------------------- #
def test_upload_runs_s1_and_proposes_align(client, env, mesh_glb):
    """上传即同步跑完 S1：返回 OBB / 六面 / 单位 / auto / final，并落 ALIGN_READY。"""
    aid = _create(client)
    body = _upload(client, aid, mesh_glb)
    align = body["align"]
    assert body["state"] == "ALIGN_READY"
    assert set(align) >= {"version", "obb", "faces", "unit", "auto", "manual", "final"}
    assert len(align["faces"]) == 6
    assert [f["id"] for f in align["faces"]] == [1, 2, 3, 4, 5, 6]
    # 175cm 长方体 → 判 cm，换算回 1.75m
    assert align["unit"]["detected"] == "cm"
    assert align["unit"]["scale"] == pytest.approx(0.01)
    assert align["unit"]["height_m"] == pytest.approx(1.75, abs=1e-3)
    assert align["auto"]["method"] == "mesh"
    assert align["auto"]["up_face"] in range(1, 7)
    assert align["manual"] == {"face_map": {}, "trim_euler": [0.0, 0.0, 0.0],
                               "unit_override": None, "height_override": None}

    # 目录产物齐全，且 canon 节点带上了单位缩放
    adir = env.asset_dir(aid)
    assert (adir / "asset.glb").exists()
    assert (adir / "align.json").exists()
    assert (adir / "meta.json").exists()
    node = _canon_node(adir / "asset.glb")
    assert node is not None
    assert np.allclose(node.scale, [0.01, 0.01, 0.01], atol=1e-9)
    assert np.allclose(axis_norm._quat_to_mat(np.asarray(node.rotation)),
                       np.asarray(align["final"]["rotation"]), atol=1e-6)
    assert _metric_span(adir / "asset.glb")[1] == pytest.approx(1.75, abs=1e-3)

    # DB 为 align 的权威来源，GET 与上传回执一致
    got = client.get(f"/v2/assets/{aid}/align")
    assert got.status_code == 200
    assert got.json() == align
    info = client.get(f"/v2/assets/{aid}").json()
    assert info["state"] == "ALIGN_READY"
    assert info["meta"]["meshes"] == 1
    assert info["binding"]["state"] == "NONE"


def test_patch_face_map_rewrites_canon_rotation(client, env, mesh_glb):
    """PATCH 面映射（up↔front 互换）→ 重算 final 并原地改写 canon 节点旋转。"""
    aid = _create(client)
    auto = _upload(client, aid, mesh_glb)["align"]["auto"]
    before = np.asarray(_get_align(client, aid)["final"]["rotation"])

    patch = {"face_map": {str(auto["up_face"]): "front+",
                          str(auto["forward_face"]): "up+"}}
    r = client.patch(f"/v2/assets/{aid}/align", json=patch)
    assert r.status_code == 200, r.text
    align = r.json()["align"]
    rot = np.asarray(align["final"]["rotation"])
    assert not np.allclose(rot, before, atol=1e-6)
    # 人工指派后，被指为 up+ 的面法向必须落到规范 +Y
    obb = axis_norm.OBB.from_dict(align["obb"])
    normals = obb.face_normals()
    assert np.allclose(rot @ normals[auto["forward_face"] - 1], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(rot @ normals[auto["up_face"] - 1], [0.0, 0.0, 1.0], atol=1e-9)
    assert align["manual"]["face_map"] == {str(k): v for k, v in patch["face_map"].items()}
    # canon 节点只有一个，且旋转与 final 一致（原地改，不重建）
    node = _canon_node(env.asset_dir(aid) / "asset.glb")
    assert np.allclose(axis_norm._quat_to_mat(np.asarray(node.rotation)), rot, atol=1e-6)
    assert sum(1 for n in (glb_io._load(env.asset_dir(aid) / "asset.glb").nodes or [])
               if (n.name or "") == CANON_NODE) == 1


def test_patch_trim_euler_and_unit_overrides(client, env, mesh_glb):
    """trim_euler 后置叠加；真实身高覆盖优先于单位推断，产物落米制。"""
    aid = _create(client)
    _upload(client, aid, mesh_glb)
    base = np.asarray(_get_align(client, aid)["final"]["rotation"])

    trimmed = client.patch(f"/v2/assets/{aid}/align",
                           json={"trim_euler": [0.0, 90.0, 0.0]})
    assert trimmed.status_code == 200, trimmed.text
    rot = np.asarray(trimmed.json()["align"]["final"]["rotation"])
    ry90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(rot @ base.T, ry90, atol=1e-6)

    # 真实身高直填优先于一切推断（scale = 1.8 / 175cm），产物立即变米制
    overridden = client.patch(f"/v2/assets/{aid}/align", json={"height_override": 1.8})
    assert overridden.status_code == 200, overridden.text
    unit = overridden.json()["align"]["unit"]
    assert unit["detected"] == "custom"
    assert unit["height_m"] == pytest.approx(1.8)
    assert _metric_span(env.asset_dir(aid) / "asset.glb")[1] == pytest.approx(1.8, abs=1e-3)

    # height_override<=0 为清除哨兵（JSON 里 null 已表示「不改」），清除后强制单位生效
    forced = client.patch(f"/v2/assets/{aid}/align",
                          json={"unit_override": "mm", "height_override": 0})
    assert forced.status_code == 200, forced.text
    manual = forced.json()["align"]["manual"]
    assert manual["height_override"] is None
    assert manual["unit_override"] == "mm"
    assert forced.json()["align"]["unit"]["detected"] == "mm"
    # unit_override="" 同理清除，回到自动推断（cm）
    auto_again = client.patch(f"/v2/assets/{aid}/align", json={"unit_override": ""})
    assert auto_again.json()["align"]["unit"]["detected"] == "cm"


def test_patch_reset_returns_to_auto(client, mesh_glb):
    """reset=true 清空全部人工修正，回到自动探测结果。"""
    aid = _create(client)
    auto_align = _upload(client, aid, mesh_glb)["align"]
    client.patch(f"/v2/assets/{aid}/align",
                 json={"height_override": 1.8, "trim_euler": [10.0, 0.0, 0.0]})
    r = client.patch(f"/v2/assets/{aid}/align", json={"reset": True})
    assert r.status_code == 200, r.text
    align = r.json()["align"]
    assert align["manual"]["height_override"] is None
    assert align["manual"]["trim_euler"] == [0.0, 0.0, 0.0]
    assert align["final"]["rotation"] == auto_align["final"]["rotation"]
    assert align["unit"]["detected"] == auto_align["unit"]["detected"]


def test_realign_is_idempotent_and_keeps_manual(client, env, mesh_glb):
    """重跑 S1 不累积旋转（每次在原始坐标下重测），且保留既有人工修正。"""
    aid = _create(client)
    _upload(client, aid, mesh_glb)
    client.patch(f"/v2/assets/{aid}/align", json={"height_override": 1.8})
    first = _get_align(client, aid)

    r = client.post(f"/v2/assets/{aid}/realign")
    assert r.status_code == 200, r.text
    second = r.json()["align"]
    assert second["final"]["rotation"] == first["final"]["rotation"]
    assert second["final"]["scale"] == pytest.approx(first["final"]["scale"])
    assert second["unit"]["height_m"] == pytest.approx(first["unit"]["height_m"])
    assert second["manual"]["height_override"] == pytest.approx(1.8)
    # 从原件重新归一（canon 节点不在）→ 确实改写了文件
    assert second["final"]["changed"] is True

    again = client.patch(f"/v2/assets/{aid}/align", json={})   # 空 patch 也应幂等
    assert again.status_code == 200, again.text
    third = again.json()["align"]
    assert third["final"]["rotation"] == second["final"]["rotation"]
    assert third["final"]["scale"] == pytest.approx(second["final"]["scale"])
    # 结果已经与文件里的一致 → changed=False，不重写产物
    assert third["final"]["changed"] is False
    assert sum(1 for n in (glb_io._load(env.asset_dir(aid) / "asset.glb").nodes or [])
               if (n.name or "") == CANON_NODE) == 1


def test_confirm_model_enters_binding_pending(client, mesh_glb):
    """模型确认 S1 后 READY，并建 bindings 行（``auto_bind=false`` 不自动起 S2）。"""
    aid = _create(client)
    _upload(client, aid, mesh_glb)
    r = client.post(f"/v2/assets/{aid}/confirm", params={"auto_bind": "false"})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["state"] == "READY"
    assert info["binding"]["state"] == "PENDING"
    assert info["binding"]["revision"] == 0
    assert info["binding"]["has_rig"] is False

    b = client.get(f"/v2/assets/{aid}/binding")
    assert b.status_code == 200
    assert b.json()["state"] == "PENDING"


def test_animation_asset_reuses_joint_fallback(client, tmp_path):
    """动画素材（纯骨架、无网格）走关节位置量 OBB，确认后入库可被复用。"""
    aid = _create(client, kind="animation", name="run")
    skeleton = _build_skeleton_glb(tmp_path / "run.glb")
    body = _upload(client, aid, skeleton, "run.glb")
    align = body["align"]
    assert body["state"] == "ALIGN_READY"
    assert align["auto"]["method"] == "skeleton"
    assert any("无网格" in n for n in align["auto"]["notes"]), align["auto"]["notes"]
    assert align["unit"]["detected"] == "m"
    assert align["unit"]["height_m"] == pytest.approx(1.75, abs=0.1)
    assert align["unit"]["hint"] == ""

    r = client.post(f"/v2/assets/{aid}/confirm")
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["state"] == "READY"
    assert info["binding"]["state"] == "NONE"      # 动画不绑定骨骼


# --------------------------------------------------------------------------- #
# 列表 / 产物 / 删除
# --------------------------------------------------------------------------- #
def test_list_filter_by_kind_and_summary_fields(client, mesh_glb):
    aid = _create(client, name="hero")
    _upload(client, aid, mesh_glb)
    _create(client, kind="animation", name="run")

    all_assets = client.get("/v2/assets").json()
    assert all_assets["total"] == 2
    models = client.get("/v2/assets", params={"kind": "model"}).json()
    assert models["total"] == 1
    assert models["assets"][0]["asset_id"] == aid
    assert models["assets"][0]["height_m"] == pytest.approx(1.75, abs=1e-3)
    assert models["assets"][0]["unit"] == "cm"
    anims = client.get("/v2/assets", params={"kind": "animation"}).json()
    assert [a["name"] for a in anims["assets"]] == ["run"]
    assert anims["assets"][0]["state"] == "CREATED"

    paged = client.get("/v2/assets", params={"limit": 1, "offset": 1}).json()
    assert paged["total"] == 2 and len(paged["assets"]) == 1


def test_glb_and_raw_download(client, env, mesh_glb):
    aid = _create(client)
    _upload(client, aid, mesh_glb)
    g = client.get(f"/v2/assets/{aid}/glb")
    assert g.status_code == 200
    assert g.content[:4] == b"glTF"
    assert g.headers.get("Cache-Control") == "no-store"
    raw = client.get(f"/v2/assets/{aid}/raw")
    assert raw.status_code == 200
    assert raw.content[:4] == b"glTF"
    # 上传原件按 <name>_raw.<ext> 落盘
    assert (env.asset_dir(aid) / "hero_raw.glb").exists()
    assert client.get(f"/v2/assets/{aid}/views").json() == {
        "asset_id": aid, "views": [], "total": 0, "urls": []}


def test_delete_asset_removes_row_and_dir(client, env, mesh_glb):
    aid = _create(client)
    _upload(client, aid, mesh_glb)
    client.post(f"/v2/assets/{aid}/confirm", params={"auto_bind": "false"})
    assert env.asset_dir(aid).exists()

    r = client.delete(f"/v2/assets/{aid}")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "deleted"
    assert client.get(f"/v2/assets/{aid}").status_code == 404
    assert not env.asset_dir(aid).exists()
    # bindings 靠 FK CASCADE 一起消失
    assert client.get(f"/v2/assets/{aid}/binding").status_code == 404
    assert client.delete(f"/v2/assets/{aid}").status_code == 404


# --------------------------------------------------------------------------- #
# 错误码
# --------------------------------------------------------------------------- #
def test_error_paths(client, mesh_glb):
    aid = _create(client)
    # 未跑 S1 时的门槛
    assert client.get(f"/v2/assets/{aid}/align").status_code == 404
    assert client.get(f"/v2/assets/{aid}/glb").status_code == 404
    assert client.patch(f"/v2/assets/{aid}/align", json={"reset": True}).status_code == 400
    assert client.post(f"/v2/assets/{aid}/confirm").status_code == 409
    assert client.post(f"/v2/assets/{aid}/realign").status_code == 400
    assert client.post(f"/v2/assets/{aid}/rebind").status_code == 409
    # 素材不存在
    assert client.get("/v2/assets/nope123").status_code == 404
    assert client.get("/v2/assets/nope123/align").status_code == 404
    assert client.post("/v2/assets/nope123/confirm").status_code == 404
    assert client.delete("/v2/assets/nope123").status_code == 404
    # 扩展名白名单 / 空内容
    bad = client.post(f"/v2/assets/{aid}/upload",
                      files={"file": ("hero.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
    empty = client.post(f"/v2/assets/{aid}/upload",
                        files={"file": ("hero.glb", b"", "model/gltf-binary")})
    assert empty.status_code == 400
    # 上传成功后仍可正常跑
    assert _upload(client, aid, mesh_glb)["state"] == "ALIGN_READY"


def test_model_without_mesh_is_rejected(client, tmp_path):
    """模型素材必须有网格：纯骨架文件上传为 model → 400，素材落 FAILED 且原因入库。"""
    aid = _create(client, kind="model")
    skeleton = _build_skeleton_glb(tmp_path / "run.glb")
    r = client.post(f"/v2/assets/{aid}/upload",
                    files={"file": ("run.glb", skeleton, "model/gltf-binary")})
    assert r.status_code == 400, r.text
    assert "网格" in r.json()["detail"]
    info = client.get(f"/v2/assets/{aid}").json()
    assert info["state"] == "FAILED"
    assert info["error"] and "网格" in info["error"]
    # 失败素材不能被确认
    assert client.post(f"/v2/assets/{aid}/confirm").status_code == 409


def test_bad_face_map_is_rejected(client, mesh_glb):
    """非法面映射（面号越界 / 语义不识别 / 重复指派 / 对偶面）→ 400。

    关键：修正填错不能把已算好的 S1 结果废掉——align 与状态都要保持上一次的正确值。
    """
    aid = _create(client)
    good = _upload(client, aid, mesh_glb)["align"]
    for patch in ({"face_map": {"9": "up+", "1": "front+"}},
                  {"face_map": {"1": "sideways", "2": "up+"}},
                  {"face_map": {"1": "up+", "2": "front+"}},          # 对偶面 → 左手系
                  {"face_map": {"1": "up+"}}):                         # 只给一个语义轴
        r = client.patch(f"/v2/assets/{aid}/align", json=patch)
        assert r.status_code == 400, (patch, r.text)
    assert _get_align(client, aid)["final"] == good["final"]
    info = client.get(f"/v2/assets/{aid}").json()
    assert info["state"] == "ALIGN_READY"
    assert info["error"] is None
    # 保留上一次结果后仍能确认
    assert client.post(f"/v2/assets/{aid}/confirm",
                        params={"auto_bind": "false"}).status_code == 200


# --------------------------------------------------------------------------- #
# /v1 与 /v2 并存
# --------------------------------------------------------------------------- #
def test_v1_routes_still_work_and_are_isolated(client, mesh_glb, env):
    """同一应用内 /v1 四条常用路由仍 200，且两套 API 各读写自己的表与目录。"""
    aid = _create(client)
    _upload(client, aid, mesh_glb)

    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/config").status_code == 200
    jobs = client.get("/v1/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["total"] == 0                 # /v2 素材不写 tasks 表
    assert client.get("/").status_code == 200        # 旧单页 HTML 仍可访问

    assert not env.tasks_dir.exists() or not any(env.tasks_dir.iterdir())
    assert (env.assets_dir / aid).exists()
    # /v1 的任务计数与 /v2 的素材计数互不干扰
    assert client.get("/v1/health").json()["counts"] == {}
    assert client.get("/v2/health").json()["counts"] == {"ALIGN_READY": 1}
