"""P0 定向验证：ai3d.retarget 后端（配置/SQLite 持久化/GLB 归一化/接口）。

不做真实模型推理、不联网、只写 tmp_path（通过 RETARGET_PROJECT_DIR / RETARGET_DB_PATH 重定向）。
运行： <venv>/bin/python -m pytest tests/test_retarget_p0.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    SCALAR,
    UNSIGNED_SHORT,
    VEC3,
    Accessor,
    Asset,
    Attributes,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)


def _build_min_glb(path) -> bytes:
    """构造一个仅含单三角网格的最小合法 GLB（目标模型用）。"""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint16)
    vb, ib = verts.tobytes(), idx.tobytes()
    blob = vb + ib
    g = GLTF2(
        asset=Asset(version="2.0"),
        scenes=[Scene(nodes=[0])], scene=0,
        nodes=[Node(mesh=0, name="mesh0")],
        meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), indices=1)])],
        accessors=[
            Accessor(bufferView=0, componentType=FLOAT, count=3, type=VEC3,
                     min=[0, 0, 0], max=[1, 1, 0]),
            Accessor(bufferView=1, componentType=UNSIGNED_SHORT, count=3, type=SCALAR),
        ],
        bufferViews=[
            BufferView(buffer=0, byteOffset=0, byteLength=len(vb), target=ARRAY_BUFFER),
            BufferView(buffer=0, byteOffset=len(vb), byteLength=len(ib),
                       target=ELEMENT_ARRAY_BUFFER),
        ],
        buffers=[Buffer(byteLength=len(blob))],
    )
    g.set_binary_blob(blob)
    g.save_binary(str(path))
    return path.read_bytes()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """把项目目录与 SQLite 文件重定向到 tmp_path，并重载所有单例。"""
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


@pytest.fixture()
def glb_bytes(tmp_path):
    return _build_min_glb(tmp_path / "min.glb")


def _wait(client, task_id, states, timeout=10.0):
    end = time.time() + timeout
    info = None
    while time.time() < end:
        info = client.get(f"/v1/jobs/{task_id}").json()
        if info["state"] in states:
            return info
        time.sleep(0.05)
    return info


def _create_job(client, glb_bytes):
    r = client.post(
        "/v1/jobs",
        files={
            "source_file": ("source.glb", glb_bytes, "model/gltf-binary"),
            "target_file": ("target.glb", glb_bytes, "model/gltf-binary"),
        },
        data={"config": "{}"},
    )
    assert r.status_code == 200, r.text
    return r.json()["task_id"]


def test_health_and_config(client):
    h = client.get("/v1/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    c = client.get("/v1/config")
    assert c.status_code == 200
    assert c.json()["server"]["port"] == 8790


def test_create_job_normalizes_and_waits_for_views(client, glb_bytes, env):
    task_id = _create_job(client, glb_bytes)
    info = _wait(client, task_id, {"WAITING", "DONE", "FAILED"})
    assert info["state"] == "WAITING", info
    assert info["current_stage"] == "RENDER_VIEWS", info
    by_stage = {s["stage"]: s["status"] for s in info["stages"]}
    assert by_stage["NORMALIZE"] == "DONE"
    assert by_stage["PRECHECK"] == "DONE"
    assert by_stage["RENDER_VIEWS"] == "WAITING"
    # 归一化产物落盘在项目目录下
    assert (env.task_dir(task_id) / "target.glb").exists()
    assert (env.task_dir(task_id) / "source.glb").exists()


def test_target_glb_and_view_spec(client, glb_bytes):
    task_id = _create_job(client, glb_bytes)
    _wait(client, task_id, {"WAITING", "DONE", "FAILED"})
    r = client.get(f"/v1/jobs/{task_id}/target.glb")
    assert r.status_code == 200
    assert r.content[:4] == b"glTF"
    vs = client.get(f"/v1/jobs/{task_id}/view_spec")
    assert vs.status_code == 200
    body = vs.json()
    assert body["model_url"].endswith("/target.glb")
    assert len(body["cameras"]) >= 4


def test_persistence_across_restart(client, glb_bytes):
    """模拟重启：重载单例 + 新建 app，仍能凭 task_id 找回历史任务。"""
    import ai3d.retarget.settings as S
    import ai3d.retarget.task_store as TS
    import ai3d.retarget.worker as W
    from ai3d.retarget.server import create_app

    task_id = _create_job(client, glb_bytes)
    _wait(client, task_id, {"WAITING", "DONE", "FAILED"})

    S.get_settings(reload=True)
    TS.get_store(reload=True)
    W.get_worker(reload=True)
    with TestClient(create_app()) as client2:
        r = client2.get(f"/v1/jobs/{task_id}")
        assert r.status_code == 200, r.text
        assert r.json()["task_id"] == task_id
        lst = client2.get("/v1/jobs").json()
        assert any(t["task_id"] == task_id for t in lst["tasks"])


def test_upload_views_resumes(client, glb_bytes, env):
    task_id = _create_job(client, glb_bytes)
    _wait(client, task_id, {"WAITING", "DONE", "FAILED"})
    png = b"\x89PNG\r\n\x1a\n_fakebytes_"
    r = client.post(
        f"/v1/jobs/{task_id}/views",
        files=[("files", ("front.png", png, "image/png")),
               ("files", ("right.png", png, "image/png"))],
    )
    assert r.status_code == 200, r.text
    info = _wait(client, task_id, {"WAITING", "DONE", "FAILED"})
    by_stage = {s["stage"]: s["status"] for s in info["stages"]}
    assert by_stage["RENDER_VIEWS"] == "DONE"
    vdir = env.task_dir(task_id) / "views"
    assert (vdir / "front.png").exists()
    assert (vdir / "right.png").exists()


def test_missing_task_404(client):
    assert client.get("/v1/jobs/nope123").status_code == 404
