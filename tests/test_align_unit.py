"""S1 导入矫正定向验证：单位推断、面映射旋转、canon 节点缩放与米制一致性。

用程序生成的「长方体网格 GLB」与「两关节骨架 GLB」按 m/cm/mm/inch/ft 五种尺度各建
一份，验证 estimate_unit / frame_from_face_map / align_asset 的行为，以及 canon 节点的
单位缩放不污染源 rest 位置的单位（height_scale 要靠它做单位换算）。
不做真实模型推理、不联网、只写 tmp_path。
运行： <venv>/bin/python -m pytest tests/test_align_unit.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest
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
from ai3d.retarget.axis_norm import CANON_NODE, align_asset, estimate_unit
from ai3d.retarget.glb_build import _Builder
from ai3d.retarget.obb import FACE_AXES, assign_faces, min_volume_obb
from ai3d.retarget.retarget import _global_rest_positions

# (单位名, 该单位换算到米的系数)；同一具 1.75m × 0.5m × 0.3m 的长方体按五种单位落盘
_UNITS = [("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("inch", 0.0254), ("ft", 0.3048)]
_BOX_M = (0.5, 1.75, 0.3)


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


def _build_mesh_glb(path, size) -> None:
    """无骨骼网格 GLB（仅 POSITION）：走 axis_norm 的纯网格探测路径。"""
    b = _Builder()
    acc = b.add(_box_surface(size), "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-align"),
        scene=0, scenes=[Scene(nodes=[0])],
        nodes=[Node(name="body", mesh=0)],
        meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=acc))])],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))


def _build_skin_glb(path, height: float, canon_scale: float = 1.0) -> None:
    """两关节骨架 GLB（pelvis/head），**不含 inverseBindMatrices** 以强制走层级累加路径。"""
    b = _Builder()
    acc = b.add(_box_surface((0.2, height, 0.2), n=4), "VEC3", FLOAT,
                ARRAY_BUFFER, minmax=True)
    s = float(canon_scale)
    nodes = [
        Node(name=CANON_NODE, rotation=[0.0, 0.0, 0.0, 1.0], scale=[s, s, s], children=[1]),
        Node(name="armature", children=[2, 3]),
        Node(name="pelvis", translation=[0.0, height * 0.5, 0.0], mesh=0),
        Node(name="head", translation=[0.0, height, 0.0]),
    ]
    gltf = GLTF2(
        asset=Asset(version="2.0", generator="test-align-skin"),
        scene=0, scenes=[Scene(nodes=[0])], nodes=nodes,
        meshes=[Mesh(primitives=[Primitive(attributes=Attributes(POSITION=acc))])],
        skins=[Skin(name="rig", skeleton=1, joints=[2, 3])],
        accessors=b.accessors, bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    gltf.save_binary(str(path))


def _metric_span(path) -> np.ndarray:
    """对齐后 merge_mesh 的 AABB 跨度（米制）。"""
    p = glb_io.merge_mesh(glb_io._load(path))["positions"]
    return (p.max(0) - p.min(0)).astype(np.float64)


# --------------------------------------------------------------------------- #
# estimate_unit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,factor", _UNITS)
def test_estimate_unit_detects_scale_of_same_geometry(name, factor):
    """同一具 1.75m 长方体按五种单位落盘，都应判对该单位且换算回 1.75m。"""
    got = estimate_unit(_BOX_M[1] / factor)
    assert got["detected"] == name, got
    assert got["scale"] == pytest.approx(factor)
    assert got["height_m"] == pytest.approx(_BOX_M[1], abs=1e-6)
    assert got["hint"] == ""


def test_estimate_unit_hint_out_of_human_range():
    """身高不在成人区间时必须给人工提示（归一化资产/非人形的典型信号）。"""
    small = estimate_unit(0.9776)
    assert small["detected"] == "m"
    assert "不在成人区间" in small["hint"]
    assert estimate_unit(1.75)["hint"] == ""


def test_estimate_unit_height_override_wins():
    """人工填真实身高优先于一切推断，scale = 身高 / up 轴跨度。"""
    got = estimate_unit(100.0, target_height=1.78)
    assert got["detected"] == "custom"
    assert got["scale"] == pytest.approx(0.0178)
    assert got["height_m"] == pytest.approx(1.78)


def test_estimate_unit_forced_and_unknown_unit():
    """人工指定单位直接采用；写错单位名回退自动推断并提示。"""
    forced = estimate_unit(100.0, unit="cm")
    assert (forced["detected"], forced["scale"]) == ("cm", 0.01)
    assert "不在成人区间" in forced["hint"]          # 100cm = 1.0m，确实要提示
    bad = estimate_unit(100.0, unit="parsec")
    assert bad["detected"] != "parsec"
    assert "未知单位" in bad["hint"]


def test_estimate_unit_degenerate_extent():
    got = estimate_unit(0.0)
    assert got["scale"] == 1.0 and got["candidates"] == [] and got["hint"]


# --------------------------------------------------------------------------- #
# frame_from_face_map
# --------------------------------------------------------------------------- #
def _axis_aligned_obb(size=_BOX_M):
    obb = min_volume_obb(_box_surface(size).astype(np.float64))
    assert obb is not None
    return obb


def _auto_faces(obb):
    """用标准人形先验（+Y up / +Z front）拿自动指派的面编号。"""
    return assign_faces(obb, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def test_frame_from_face_map_maps_faces_to_canonical():
    """指派面 → 规范轴后，面法向必须落到对应的规范单位轴上。"""
    obb = _axis_aligned_obb()
    faces = _auto_faces(obb)
    rot = axis_norm.frame_from_face_map(
        obb, {faces["up_face"]: "up+", faces["forward_face"]: "front+"})
    normals = obb.face_normals()
    assert np.allclose(rot @ normals[faces["up_face"] - 1], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(rot @ normals[faces["forward_face"] - 1], [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(rot @ normals[faces["left_face"] - 1], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(rot, np.eye(3), atol=1e-9)     # 本就规范摆放 → 单位旋转


def test_frame_from_face_map_left_derived_from_right_hand_rule():
    """只给 up/front 两面时，left 由 cross(up, forward) 导出（规范系 +X = left）。"""
    obb = _axis_aligned_obb()
    up_face = _auto_faces(obb)["up_face"]
    # 取一个与 up 不同轴的面作 front（排除 up 及其对偶面）
    up_axis = FACE_AXES[up_face - 1][0]
    fwd_face = next(i for i in range(1, 7) if FACE_AXES[i - 1][0] != up_axis)
    rot = axis_norm.frame_from_face_map(obb, {up_face: "up+", fwd_face: "front+"})
    assert np.allclose(rot[0], np.cross(rot[1], rot[2]), atol=1e-12)
    assert np.linalg.det(rot) > 0.0
    assert np.allclose(rot @ obb.face_normals()[fwd_face - 1], [0.0, 0.0, 1.0], atol=1e-9)


def test_frame_from_face_map_trim_euler_is_post_rotation():
    """trim_euler 绕规范系后置叠加：R = Rz·Ry·Rx · R_face。"""
    obb = _axis_aligned_obb()
    faces = _auto_faces(obb)
    fm = {faces["up_face"]: "up+", faces["forward_face"]: "front+"}
    base = axis_norm.frame_from_face_map(obb, fm)
    trimmed = axis_norm.frame_from_face_map(obb, fm, [0.0, 90.0, 0.0])
    ry90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(trimmed @ base.T, ry90, atol=1e-9)
    assert np.allclose(axis_norm.frame_from_face_map(obb, fm, [0, 0, 0]), base)


def test_frame_from_face_map_rejects_bad_input():
    """面号越界、语义不识别、重复指派、少于两个语义轴都要报错（由路由转 400）。"""
    obb = _axis_aligned_obb()
    ok = _auto_faces(obb)
    with pytest.raises(ValueError, match="1..6"):
        axis_norm.frame_from_face_map(obb, {7: "up+"})
    with pytest.raises(ValueError, match="无法识别"):
        axis_norm.frame_from_face_map(obb, {1: "sideways+"})
    with pytest.raises(ValueError, match="重复指派"):
        axis_norm.frame_from_face_map(obb, {ok["up_face"]: "up+", 2: "up-"})
    with pytest.raises(ValueError, match="两个不同的语义轴"):
        axis_norm.frame_from_face_map(obb, {ok["up_face"]: "up+"})
    with pytest.raises(ValueError, match="不正交"):
        # 把对偶面指派成 up 与 front（法向共线）→ 三轴退化
        axis_norm.frame_from_face_map(obb, {1: "up+", 2: "front+"})


def test_face_semantics_covers_six_tokens():
    assert set(axis_norm.FACE_SEMANTICS) == {"up+", "up-", "front+", "front-",
                                            "left+", "left-"}
    assert len(FACE_AXES) == 6


# --------------------------------------------------------------------------- #
# align_asset：端到端 + 幂等 + 人工修正
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,factor", _UNITS)
def test_align_asset_normalizes_units_to_meters(tmp_path, name, factor):
    """五种单位落盘的同一长方体，对齐后 merge_mesh 都应是同一份米制尺寸。"""
    path = tmp_path / f"box_{name}.glb"
    _build_mesh_glb(path, tuple(v / factor for v in _BOX_M))
    info = align_asset(path)
    assert info["unit"]["detected"] == name, info["unit"]
    assert info["final"]["scale"] == pytest.approx(factor)
    assert info["auto"]["method"] == "mesh"
    # 本就规范摆放且已是米制时不插 canon 节点（changed=False 是预期行为）
    assert info["final"]["changed"] is (factor != 1.0)
    span = _metric_span(path)
    assert span[1] == pytest.approx(_BOX_M[1], abs=0.02)   # +Y = 身高
    assert span[2] == pytest.approx(_BOX_M[2], abs=0.02)   # +Z = 最薄（前后）
    assert span[0] == pytest.approx(_BOX_M[0], abs=0.02)   # +X = 左右


def test_align_asset_writes_sidecar_and_full_structure(tmp_path):
    path = tmp_path / "box.glb"
    _build_mesh_glb(path, _BOX_M)
    side = tmp_path / "align.json"
    info = align_asset(path, json_path=side)
    assert set(info) == {"version", "obb", "faces", "unit", "auto", "manual", "final"}
    assert [f["id"] for f in info["faces"]] == [1, 2, 3, 4, 5, 6]
    assert info["manual"] == {"face_map": {}, "trim_euler": [0.0, 0.0, 0.0],
                              "unit_override": None, "height_override": None}
    on_disk = json.loads(side.read_text(encoding="utf-8"))
    assert on_disk["final"]["rotation"] == info["final"]["rotation"]


def test_align_asset_is_idempotent_and_keeps_single_canon_node(tmp_path):
    """重跑不得累积旋转，也不得多插 canon 节点（否则索引错位）。"""
    path = tmp_path / "box.glb"
    _build_mesh_glb(path, tuple(v / 0.01 for v in _BOX_M))     # cm 尺度
    first = align_asset(path)
    span = _metric_span(path)
    second = align_asset(path, first)
    assert second["final"]["changed"] is False
    assert second["final"]["rotation"] == first["final"]["rotation"]
    assert second["final"]["scale"] == first["final"]["scale"]
    assert _metric_span(path) == pytest.approx(span, abs=1e-9)
    nodes = [n.name for n in glb_io._load(path).nodes]
    assert nodes.count(CANON_NODE) == 1


def test_align_asset_over_old_canon_rotation_does_not_accumulate(tmp_path):
    """文件里已有别的 canon 旋转（如旧 /v1 吸附结果）时，重测必须回到原始坐标。"""
    a = tmp_path / "a.glb"
    b = tmp_path / "b.glb"
    _build_mesh_glb(a, tuple(v / 0.01 for v in _BOX_M))
    _build_mesh_glb(b, tuple(v / 0.01 for v in _BOX_M))
    align_asset(a)                                            # 先自动对齐一次
    manual = {"face_map": {"1": "up+", "3": "front-"}, "trim_euler": [0.0, 30.0, 0.0]}
    patched = align_asset(a, {"manual": manual})
    fresh = align_asset(b, {"manual": manual})                # 未被污染过的对照
    assert patched["final"]["rotation"] == fresh["final"]["rotation"]
    assert patched["obb"]["extents"] == fresh["obb"]["extents"]


def test_align_asset_manual_unit_and_height_override(tmp_path):
    """人工覆盖单位/身高后 scale 生效，且 merge_mesh 立即变成对应米制尺寸。"""
    path = tmp_path / "box.glb"
    _build_mesh_glb(path, tuple(v / 0.01 for v in _BOX_M))    # cm，自动会判 cm
    info = align_asset(path, {"manual": {"height_override": 1.9}})
    assert info["unit"]["detected"] == "custom"
    assert info["final"]["scale"] == pytest.approx(1.9 / _BOX_M[1] * 0.01)
    assert _metric_span(path)[1] == pytest.approx(1.9, abs=0.02)
    forced = align_asset(path, {"manual": {"unit_override": "m"}})
    assert forced["unit"]["detected"] == "m"
    assert _metric_span(path)[1] == pytest.approx(_BOX_M[1] / 0.01, abs=2.0)


def test_align_asset_accepts_bare_manual_section(tmp_path):
    """PATCH 只传 manual 段（不含完整 align.json）也要能识别。"""
    path = tmp_path / "box.glb"
    _build_mesh_glb(path, _BOX_M)
    info = align_asset(path, {"trim_euler": [0.0, 45.0, 0.0]})
    assert info["manual"]["trim_euler"] == [0.0, 45.0, 0.0]
    assert not np.allclose(np.asarray(info["final"]["rotation"]), np.eye(3))


# --------------------------------------------------------------------------- #
# canon 缩放的下游一致性
# --------------------------------------------------------------------------- #
def test_canon_scale_keeps_rest_positions_in_raw_units(tmp_path):
    """canon 缩放是单位换算：merge 后是米制，而源 rest 位置必须留在原始单位。

    ``retarget_animation`` 用 ``height_scale = rig.height / src_height`` 同时承担
    「源→目标身高比例」与「原始单位→米」，故 src_height 必须与动画通道的 translation
    增量同单位，否则 cm 资产的 root motion 会放大 100 倍。
    """
    raw = tmp_path / "raw.glb"
    scaled = tmp_path / "scaled.glb"
    _build_skin_glb(raw, 175.0, canon_scale=1.0)
    _build_skin_glb(scaled, 175.0, canon_scale=0.01)

    def rest(path):
        g = glb_io._load(path)
        return _global_rest_positions(glb_io.extract_skin(g), glb_io.extract_nodes(g))

    pos_raw, pos_scaled = rest(raw), rest(scaled)
    assert np.allclose(pos_scaled[2], pos_raw[2], atol=1e-9)
    assert np.allclose(pos_scaled[3], pos_raw[3], atol=1e-9)
    assert pos_scaled[3][1] == pytest.approx(175.0, abs=1e-6)   # 原始单位，未被缩放

    # 全局矩阵（含 canon）仍是米制：蒙皮/求解侧读到的是 1.75m
    mats = axis_norm.global_matrices(glb_io._load(scaled))
    assert mats[3][:3, 3][1] == pytest.approx(1.75, abs=1e-6)
    assert _metric_span(scaled)[1] == pytest.approx(1.75, abs=1e-6)
