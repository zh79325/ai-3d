"""``unirig.log`` 的增量读取语义（``AssetStore.read_unirig_log`` + ``GET`` 路由）。

前端靠字节 offset 分片轮询这个文件看实时进度，几个边界必须钉死：文件还没建、
一次读不完、重跑把文件截短（offset 落到 size 之外）。这些都不是异常路径，
是绑定刚起步 / 推理刷屏 / 二次重跑时的**常态**。
"""

import asyncio
import json

import pytest

from ai3d.retarget.asset_store import AssetStore
from ai3d.retarget.schemas_v2 import BindingState
from ai3d.retarget.settings import get_settings


@pytest.fixture
def store(tmp_path, monkeypatch):
    """指向 tmp 的素材库：不碰真实 ``output/``（规则要求测试只写 tmp_path）。"""
    monkeypatch.setenv("RETARGET_PROJECT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("RETARGET_DB_PATH", str(tmp_path / "output" / "data" / "t.db"))
    settings = get_settings(reload=True)
    settings.ensure_dirs()
    try:
        yield AssetStore(settings)
    finally:
        get_settings(reload=True)          # 还原单例，别把 tmp 配置带给后续测试


@pytest.fixture
def model_asset(store):
    return store.create_asset("model", "hero")


def write_log(store, asset_id, text):
    path = store.unirig_log_path(asset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_log_is_not_an_error(store, model_asset):
    """没跑过 UniRig 时返回 exists=false 而不是抛：前端从绑定起步就在轮询。"""
    out = store.read_unirig_log(model_asset)
    assert out["exists"] is False
    assert out["text"] == ""
    assert out["offset"] == 0
    assert out["asset_id"] == model_asset


def test_incremental_offsets_reassemble_full_text(store, model_asset):
    """边跑边读：多次增量拼接必须等于全文（不丢字、不重复）。"""
    write_log(store, model_asset, "")
    collected, offset = "", 0
    for chunk in ["第一行\n", "第二行（骨架预测中）\n", "第三行：蒙皮 100%\n"]:
        path = store.unirig_log_path(model_asset)
        with path.open("a", encoding="utf-8") as fh:      # 模拟推理线程持续追加
            fh.write(chunk)
        out = store.read_unirig_log(model_asset, offset=offset)
        collected += out["text"]
        offset = out["offset"]

    full = store.unirig_log_path(model_asset).read_text(encoding="utf-8")
    assert collected == full
    assert "第三行：蒙皮 100%" in collected
    assert offset == len(full.encode("utf-8"))


def test_max_bytes_slices_and_reports_truncated(store, model_asset):
    """一次读不完时只返回一片、offset 照样前进，下轮接着拉。"""
    write_log(store, model_asset, "x" * 1000 + "\n")
    first = store.read_unirig_log(model_asset, offset=0, max_bytes=100)
    assert first["truncated"] is True
    assert len(first["text"]) == 100
    assert first["offset"] == 100

    second = store.read_unirig_log(model_asset, offset=first["offset"], max_bytes=4096)
    assert second["truncated"] is False
    assert second["offset"] == first["size"]


def test_offset_beyond_size_resets_to_head(store, model_asset):
    """重跑会覆盖（截短）文件：offset 落到 size 之外要从头返回并置 reset。"""
    write_log(store, model_asset, "上一轮的很长很长的输出" * 20)
    stale_offset = store.unirig_log_path(model_asset).stat().st_size
    write_log(store, model_asset, "新一轮\n")            # 覆盖后变短

    out = store.read_unirig_log(model_asset, offset=stale_offset)
    assert out["reset"] is True
    assert out["text"].startswith("新一轮")
    assert out["offset"] == len("新一轮\n".encode("utf-8"))


def test_multibyte_split_offset_does_not_raise(store, model_asset):
    """offset 正切在中文的多字节序列中间也不能抛（按字节读 + replace 解码）。"""
    write_log(store, model_asset, "骨架预测中\n")
    raw_size = store.unirig_log_path(model_asset).stat().st_size
    out = store.read_unirig_log(model_asset, offset=0, max_bytes=4)   # 「骨」占 3 字节
    assert isinstance(out["text"], str)                # 切在多字节中间也不报错
    assert out["offset"] == 4
    assert "骨" in out["text"]

    rest = store.read_unirig_log(model_asset, offset=out["offset"])
    assert rest["offset"] == raw_size                  # 剩下的一次拉完
    assert "预测中" in rest["text"]


def test_binding_progress_is_reported_alongside(store, model_asset):
    """日志响应顺带带上 stage / message / running，前端一次请求就够画进度。"""
    store.update_binding(model_asset, state=BindingState.RUNNING,
                         stage="UNIRIG_SKIN", message="蒙皮预测中")
    write_log(store, model_asset, "Predicting: 42%\n")

    out = store.read_unirig_log(model_asset)
    assert out["running"] is True
    assert out["stage"] == "UNIRIG_SKIN"
    assert out["message"] == "蒙皮预测中"

    store.mark_binding_done(model_asset, message="完成")
    assert store.read_unirig_log(model_asset)["running"] is False


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(store, monkeypatch):
    """只挂 /v2 路由的最小应用；store 换成 tmp 版，不动真实库。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ai3d.retarget import server_v2

    monkeypatch.setattr(server_v2, "_store", lambda: store)
    app = FastAPI()
    app.include_router(server_v2.router)
    return TestClient(app)


def test_route_returns_log_with_no_store_header(client, store, model_asset):
    write_log(store, model_asset, "==== RUN ====\n骨架预测中\n")
    res = client.get(f"/v2/assets/{model_asset}/unirig/log")
    assert res.status_code == 200
    # 日志边跑边写，绝不能让浏览器用旧副本
    assert res.headers["Cache-Control"] == "no-store"
    body = res.json()
    assert body["exists"] is True
    assert "骨架预测中" in body["text"]
    assert body["offset"] == len("==== RUN ====\n骨架预测中\n".encode("utf-8"))


def test_route_honours_offset_query(client, store, model_asset):
    """路由按 offset / max_bytes 分片：第二片从上次返回的 offset 接着走。"""
    body = "A" * 2000 + "\n" + "B" * 500 + "\n"
    write_log(store, model_asset, body)
    url = f"/v2/assets/{model_asset}/unirig/log"

    first = client.get(url, params={"offset": 0, "max_bytes": 1024})
    assert first.status_code == 200
    assert len(first.json()["text"]) == 1024
    assert first.json()["truncated"] is True

    second = client.get(url, params={"offset": first.json()["offset"]})
    assert second.json()["text"] == body[1024:]
    assert second.json()["truncated"] is False


def test_route_404_for_unknown_asset(client):
    res = client.get("/v2/assets/nope/unirig/log")
    assert res.status_code == 404
    assert json.loads(res.content)["detail"]


def test_route_rejects_negative_offset(client, model_asset):
    assert client.get(f"/v2/assets/{model_asset}/unirig/log",
                      params={"offset": -1}).status_code == 422


def test_route_rejects_too_small_max_bytes(client, model_asset):
    """分片下限 1KB：拉太小只会把一轮输出拆成几十次请求。"""
    assert client.get(f"/v2/assets/{model_asset}/unirig/log",
                      params={"max_bytes": 4}).status_code == 422


def test_route_handler_is_usable_without_testclient(store, model_asset, monkeypatch):
    """路由函数本身可直接 await（不依赖 httpx 的兜底验证）。"""
    from ai3d.retarget import server_v2

    monkeypatch.setattr(server_v2, "_store", lambda: store)
    write_log(store, model_asset, "直接调用\n")
    res = asyncio.run(server_v2.get_unirig_log(model_asset, offset=0, max_bytes=4096))
    assert res.status_code == 200
    assert "直接调用" in json.loads(res.body.decode("utf-8"))["text"]
