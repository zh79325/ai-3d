"""OBB 定向验证：最小体积有向包围盒求解与「面 → 语义轴」指派。

不做真实模型推理、不联网、不读 output/（真实资产目视验证手动跑）。
运行： <venv>/bin/python -m pytest tests/test_obb.py -v
"""

from __future__ import annotations

import time

import numpy as np

from ai3d.retarget.obb import (
    FACE_AXES,
    FACE_OPPOSITE,
    OBB,
    assign_faces,
    convex_hull_sample,
    min_volume_obb,
)


def _rot_zy(deg_z: float, deg_y: float) -> np.ndarray:
    """绕 Z 再绕 Y 的旋转矩阵（度）。"""
    cz, sz = np.cos(np.deg2rad(deg_z)), np.sin(np.deg2rad(deg_z))
    cy, sy = np.cos(np.deg2rad(deg_y)), np.sin(np.deg2rad(deg_y))
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return ry @ rz


def _box_points(size, n: int = 20000, seed: int = 0, rot=None) -> np.ndarray:
    """均匀填充的长方体点云，可整体旋转。"""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(size, dtype=np.float64)
    return p if rot is None else p @ rot.T


def _tpose_points(n: int = 6000, seed: int = 1) -> np.ndarray:
    """T-pose 人形点云：臂展 2.0 > 身高 1.8，最长轴落在 X（陷阱用例）。"""
    rng = np.random.default_rng(seed)
    torso = rng.uniform([-0.2, 0.0, -0.12], [0.2, 1.8, 0.12], (n, 3))
    arm_r = rng.uniform([0.2, 1.5, -0.08], [1.0, 1.7, 0.08], (n // 2, 3))
    arm_l = rng.uniform([-1.0, 1.5, -0.08], [-0.2, 1.7, 0.08], (n // 2, 3))
    return np.concatenate([torso, arm_r, arm_l])


# --------------------------------------------------------------------------- #
# min_volume_obb
# --------------------------------------------------------------------------- #
def test_min_volume_obb_recovers_rotated_box_extents():
    """随机旋转的已知长方体，OBB 三轴全长应恢复原尺寸（采样误差内）。"""
    size = (2.0, 0.5, 1.0)
    obb = min_volume_obb(_box_points(size, rot=_rot_zy(30.0, 20.0)))
    assert obb is not None
    got = np.sort(np.asarray(obb.extents))
    want = np.sort(np.asarray(size, dtype=np.float64))
    assert np.allclose(got, want, atol=0.08 * want.max()), (got, want)


def test_min_volume_obb_beats_aabb_on_rotated_box():
    """旋转后的长方体，OBB 体积必须小于轴对齐包围盒体积（否则等于没求解）。"""
    p = _box_points((2.0, 0.5, 1.0), rot=_rot_zy(35.0, 25.0))
    obb = min_volume_obb(p)
    assert obb is not None
    v_obb = float(np.prod(obb.extents))
    v_aabb = float(np.prod(p.max(0) - p.min(0)))
    assert v_obb < v_aabb * 0.8, (v_obb, v_aabb)


def test_obb_axes_orthonormal_and_right_handed():
    """三轴须单位正交且构成右手基（面编号稳定性依赖此约定）。"""
    obb = min_volume_obb(_box_points((1.0, 3.0, 0.4), rot=_rot_zy(15.0, -40.0)))
    assert obb is not None
    a = np.asarray(obb.axes)
    assert np.allclose(a @ a.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(a) > 0.0


def test_corners_span_matches_extents():
    """八角点在 OBB 自身坐标系下的跨度应等于 extents。"""
    obb = min_volume_obb(_box_points((1.2, 2.4, 0.6), rot=_rot_zy(50.0, 10.0)))
    assert obb is not None
    local = (obb.corners() - obb.center) @ np.asarray(obb.axes).T
    span = local.max(0) - local.min(0)
    assert np.allclose(span, np.asarray(obb.extents), atol=1e-9)
    assert len(obb.edges()) == 12


def test_obb_roundtrip_dict():
    obb = min_volume_obb(_box_points((1.0, 2.0, 0.5)))
    assert obb is not None
    back = OBB.from_dict(obb.to_dict())
    assert np.allclose(back.center, obb.center)
    assert np.allclose(back.axes, obb.axes)
    assert np.allclose(back.extents, obb.extents)


def test_min_volume_obb_degenerate_returns_none():
    """顶点不足或全部共面时返回 None，不得抛异常。"""
    assert min_volume_obb(np.zeros((3, 3))) is None
    planar = np.zeros((200, 3))
    planar[:, :2] = np.random.default_rng(0).uniform(-1, 1, (200, 2))
    assert min_volume_obb(planar) is None


def test_convex_hull_sample_caps_points():
    p = np.random.default_rng(0).uniform(-1, 1, (50000, 3))
    assert len(convex_hull_sample(p, 6000)) == 6000
    assert len(convex_hull_sample(p[:100], 6000)) == 100


def test_min_volume_obb_large_pointcloud_under_5s():
    """20 万顶点级模型走降采样后须在秒级完成（阶段一同步跑，不能挂住请求）。"""
    p = _tpose_points(n=60000, seed=3)
    t = time.time()
    obb = min_volume_obb(p)
    dt = time.time() - t
    assert obb is not None
    assert dt < 5.0, f"耗时 {dt:.2f}s"


# --------------------------------------------------------------------------- #
# assign_faces
# --------------------------------------------------------------------------- #
def test_assign_faces_up_ignores_longest_axis():
    """T-pose 臂展 > 身高时，up 必须由先验方向决定而非最长轴。"""
    p = _tpose_points()
    obb = min_volume_obb(p)
    assert obb is not None
    ext = np.asarray(obb.extents)
    longest = int(np.argmax(ext))                     # 应为臂展所在轴
    got = assign_faces(obb, up_dir=np.array([0.0, 1.0, 0.0]),
                       forward_dir=np.array([0.0, 0.0, 1.0]))
    up_axis = FACE_AXES[got["up_face"] - 1][0]
    assert up_axis != longest, (up_axis, longest, ext)
    normal = obb.face_normals()[got["up_face"] - 1]
    assert abs(float(normal @ np.array([0.0, 1.0, 0.0]))) > 0.99
    assert got["notes"] == []


def test_assign_faces_forward_is_thinnest_axis():
    """forward 先验给定时应吸附到最薄轴（人形前后最薄）。"""
    obb = min_volume_obb(_tpose_points())
    assert obb is not None
    ext = np.asarray(obb.extents)
    got = assign_faces(obb, up_dir=np.array([0.0, 1.0, 0.0]),
                       forward_dir=np.array([0.0, 0.0, 1.0]))
    fwd_axis = FACE_AXES[got["forward_face"] - 1][0]
    assert fwd_axis == int(np.argmin(ext))
    assert got["forward_face"] != got["up_face"]
    assert got["forward_face"] != FACE_OPPOSITE[got["up_face"] - 1]


def test_assign_faces_left_from_right_hand_rule():
    """left 面须满足 cross(up, forward) 右手关系（规范系 +X left）。"""
    obb = min_volume_obb(_tpose_points())
    assert obb is not None
    got = assign_faces(obb, up_dir=np.array([0.0, 1.0, 0.0]),
                       forward_dir=np.array([0.0, 0.0, 1.0]))
    n = obb.face_normals()
    left = np.cross(n[got["up_face"] - 1], n[got["forward_face"] - 1])
    assert float(left @ n[got["left_face"] - 1]) > 0.99
    assert len({got["up_face"], got["forward_face"], got["left_face"]}) == 3


def test_assign_faces_fallback_records_note():
    """无任何先验时按最长轴兜底，且必须写 notes 提示人工核对。"""
    obb = min_volume_obb(_tpose_points())
    assert obb is not None
    got = assign_faces(obb, up_dir=None, forward_dir=None)
    assert any("兜底" in s for s in got["notes"])
    assert 1 <= got["up_face"] <= 6 and 1 <= got["forward_face"] <= 6
