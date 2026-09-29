"""``unirig`` 的空间契约：合并 npz 里 joints / 采样顶点必须是规范系米制。

实测踩过的坑：UniRig 的 inference transform 把网格归一到 [-1,1] 立方体，
``predict_skeleton.npz`` / ``predict_skin.npz`` 的 joints 与采样顶点都在该空间，
而 ``raw_data.npz`` 的顶点是米制。``_merge_raw`` 曾把两者直接拼进同一个 npz，
adapt 于是把归一坐标当米制 —— 手臂链检测全部落空、8 个手臂关节被按比例合成成
**水平 T-pose**（模型明明是 A-pose），kNN 权重回传也对着错位点云查最近邻。

这里用合成小人钉死三件事：合并后坐标回到米制、归一方式变了要 loudly 报错、
adapt 出来的手臂跟着网格姿势走而不是合成横臂。
"""

import numpy as np
import pytest

from ai3d.retarget.unirig import adapt, predict

H = 1.0                      # 合成小人身高
CENTER = np.array([0.0, 0.5, 0.0])
HALF = 0.5                   # bbox 最长轴半跨度（y: 0..1）


def _canon_skeleton():
    """21 骨合成小人：脊柱 5 + 双臂各 4（A-pose 下倾）+ 双腿各 4。"""
    joints = [
        (0.0, 0.50, 0.0),    # 0 pelvis
        (0.0, 0.60, 0.0),    # 1
        (0.0, 0.75, 0.0),    # 2 chest（臂挂这里）
        (0.0, 0.90, 0.0),    # 3
        (0.0, 1.00, 0.0),    # 4 head
        (0.05, 0.74, 0.0), (0.20, 0.66, 0.0), (0.35, 0.58, 0.0), (0.50, 0.50, 0.0),   # 5-8 左臂
        (-0.05, 0.74, 0.0), (-0.20, 0.66, 0.0), (-0.35, 0.58, 0.0), (-0.50, 0.50, 0.0),  # 9-12 右臂
        (0.10, 0.45, 0.0), (0.12, 0.25, 0.0), (0.13, 0.05, 0.0), (0.15, 0.00, 0.0),   # 13-16 左腿
        (-0.10, 0.45, 0.0), (-0.12, 0.25, 0.0), (-0.13, 0.05, 0.0), (-0.15, 0.00, 0.0),  # 17-20 右腿
    ]
    parents = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    return np.array(joints, dtype=np.float64), np.array(parents, dtype=np.int64)


def _canon_mesh():
    """粗糙盒状点云：y 0..1、x ±0.5、z ±0.1（bbox 最长轴 = y）。"""
    ys = np.linspace(0.0, 1.0, 12)
    xs = np.array([-0.5, 0.0, 0.5])
    pts = np.stack(np.meshgrid(xs, ys, np.array([-0.1, 0.1]), indexing="ij"), axis=-1)
    return pts.reshape(-1, 3).astype(np.float64)


def _write_npzs(tmp_path, anisotropic: bool = False):
    """按 UniRig 的约定写三份 npz：skeleton/skin 在归一空间，raw 在米制。"""
    joints, parents = _canon_skeleton()
    verts = _canon_mesh()
    to_norm = lambda a: (a - CENTER) / HALF                      # noqa: E731
    if anisotropic:                                              # 伪造「逐轴归一」的上游
        scale = np.array([HALF, HALF, HALF * 0.4])
        to_norm = lambda a: (a - CENTER) / scale                 # noqa: E731
    n_j = len(joints)
    skin = np.zeros((len(joints), n_j))                          # 采样点=关节，one-hot 自重
    np.fill_diagonal(skin, 1.0)
    np.savez(tmp_path / predict.RAW_NPZ, vertices=verts, faces=np.zeros((0, 3), np.int64))
    np.savez(tmp_path / predict.SKELETON_NPZ,
             vertices=to_norm(verts), joints=to_norm(joints), parents=parents,
             names=np.array([f"bone_{i}" for i in range(n_j)]))
    np.savez(tmp_path / predict.SKIN_NPZ,
             vertices=to_norm(joints), joints=to_norm(joints), skin=skin)
    return joints, verts


def test_merge_puts_joints_back_into_metres(tmp_path):
    joints, verts = _write_npzs(tmp_path)
    merged = np.load(predict._merge_raw(tmp_path), allow_pickle=True)
    assert np.allclose(merged["joints"], joints, atol=1e-9)
    assert np.allclose(merged["sampled_vertices"], joints, atol=1e-9)
    assert np.allclose(merged["vertices"], verts, atol=1e-9)


def test_merge_rejects_non_uniform_normalisation(tmp_path):
    """上游归一方式一旦变了必须报错：静默产出错空间的骨架比崩掉更糟。"""
    _write_npzs(tmp_path, anisotropic=True)
    with pytest.raises(RuntimeError, match="归一空间还原自检失败"):
        predict._merge_raw(tmp_path)


def test_adapt_follows_mesh_pose_instead_of_synthesising_arms(tmp_path):
    """手臂关节取自预测链（A-pose 下倾），不再合成水平 T-pose。"""
    _write_npzs(tmp_path)
    pred = adapt.load_prediction(predict._merge_raw(tmp_path))
    rig, Ws, info = adapt.adapt_prediction(pred, H)

    assert info["synthesized"] == []
    for side in ("l", "r"):
        src = [rig.source[f"{p}_{side}"] for p in ("clavicle", "upper_arm", "lower_arm", "hand")]
        assert src == ["unirig"] * 4
    # A-pose：hand 必须明显低于 upper_arm（合成 T-pose 时两者 y 相同）
    for side in ("l", "r"):
        drop = rig.heads[f"upper_arm_{side}"][1] - rig.heads[f"hand_{side}"][1]
        assert drop > 0.10 * H, f"{side} 臂没有下倾：{drop}"
    # 左右对称且手在身体外侧
    assert rig.heads["hand_l"][0] > 0.4 and rig.heads["hand_r"][0] < -0.4
    assert np.allclose(rig.heads["hand_l"] * [-1, 1, 1], rig.heads["hand_r"], atol=1e-9)
    assert (Ws.sum(axis=1) < 1e-9).sum() == 0
