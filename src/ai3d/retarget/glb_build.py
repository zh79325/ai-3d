"""组装 result.glb：目标网格 + 骨架(22 关节) + 蒙皮权重 + 逐帧动画。

绑定姿态用纯平移定义（关节 rest 局部旋转=单位），故：
- 关节节点 local translation = head − parent_head（根=pelvis 全局 head）
- inverseBindMatrix = translate(−head_global)（列主序 16 float）
- 动画 = 每关节 rotation 通道（局部四元数）+ pelvis translation 通道（root motion，绝对局部平移）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
    UNSIGNED_SHORT,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Attributes,
    Buffer,
    BufferView,
    GLTF2,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Scene,
    Skin,
)

from .skeleton import JOINTS, JOINT_INDEX, children_of


class _Builder:
    """累积二进制 blob，并按 4 字节对齐创建 bufferView/accessor。"""

    def __init__(self):
        self.blob = bytearray()
        self.buffer_views: List[BufferView] = []
        self.accessors: List[Accessor] = []

    def _pad(self, align: int = 4) -> None:
        pad = (-len(self.blob)) % align
        if pad:
            self.blob.extend(b"\x00" * pad)

    def add(self, arr: np.ndarray, gltf_type: str, comp: int,
            target: Optional[int] = None, minmax: bool = False,
            count: Optional[int] = None) -> int:
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        self._pad(4)
        off = len(self.blob)
        self.blob.extend(raw)
        bv_idx = len(self.buffer_views)
        self.buffer_views.append(
            BufferView(buffer=0, byteOffset=off, byteLength=len(raw), target=target))
        n = arr.shape[0] if count is None else count
        kw = dict(bufferView=bv_idx, componentType=comp, count=n, type=gltf_type)
        if minmax and n > 0:
            flat = arr.reshape(n, -1)
            kw["min"] = flat.min(0).tolist()
            kw["max"] = flat.max(0).tolist()
        acc_idx = len(self.accessors)
        self.accessors.append(Accessor(**kw))
        return acc_idx


def _compute_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = np.zeros_like(positions, dtype=np.float32)
    tri = positions[indices.reshape(-1, 3)]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, norm, out=np.zeros_like(fn), where=norm > 1e-8)
    for k in range(3):
        np.add.at(n, indices.reshape(-1, 3)[:, k], fn)
    vn = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, vn, out=n, where=vn > 1e-8).astype(np.float32)


def build_result_glb(
    mesh: Dict[str, np.ndarray],
    heads: Dict[str, np.ndarray],
    joints_u16: np.ndarray,      # (N,4) uint16，取值 0..21（JOINTS 顺序）
    weights_f32: np.ndarray,     # (N,4) float32，和≈1
    times: np.ndarray,           # (F,) float32，递增
    rotations: Dict[str, np.ndarray],   # joint -> (F,4) 四元数(x,y,z,w)
    root_translations: Optional[np.ndarray],  # (F,3) pelvis 绝对局部平移
    out_path: str | Path,
    anim_name: str = "retarget",
) -> Path:
    positions = np.ascontiguousarray(mesh["positions"], dtype=np.float32)
    indices = np.ascontiguousarray(mesh["indices"].astype(np.uint32).ravel())
    normals = mesh.get("normals")
    if normals is None or not np.any(normals):
        normals = _compute_normals(positions, indices)
    normals = np.ascontiguousarray(normals, dtype=np.float32)

    n_verts = len(positions)
    use_u16_idx = n_verts < 65536
    idx_arr = indices.astype(np.uint16 if use_u16_idx else np.uint32)

    b = _Builder()
    acc_pos = b.add(positions, "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
    acc_nor = b.add(normals, "VEC3", FLOAT, ARRAY_BUFFER)
    acc_idx = b.add(idx_arr, "SCALAR", UNSIGNED_SHORT if use_u16_idx else UNSIGNED_INT,
                    ELEMENT_ARRAY_BUFFER)
    acc_joints = b.add(np.ascontiguousarray(joints_u16, dtype=np.uint16), "VEC4",
                       UNSIGNED_SHORT, ARRAY_BUFFER)
    acc_weights = b.add(np.ascontiguousarray(weights_f32, dtype=np.float32), "VEC4",
                        FLOAT, ARRAY_BUFFER)

    # 逆绑定矩阵（列主序 16 float）：translate(-head)
    nj = len(JOINTS)
    ibm = np.zeros((nj, 16), dtype=np.float32)
    for i, j in enumerate(JOINTS):
        h = np.asarray(heads[j], dtype=np.float64)
        ibm[i] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -h[0], -h[1], -h[2], 1]
    acc_ibm = b.add(ibm, "MAT4", FLOAT, count=nj)

    # 动画：时间轴 + 每关节 rotation + pelvis translation
    times = np.ascontiguousarray(times.astype(np.float32).ravel())
    n_frames = len(times)
    acc_time = b.add(times, "SCALAR", FLOAT, minmax=True)

    samplers: List[AnimationSampler] = []
    channels: List[AnimationChannel] = []

    def _add_channel(node_idx: int, path: str, out_acc: int) -> None:
        s_idx = len(samplers)
        samplers.append(AnimationSampler(input=acc_time, interpolation="LINEAR", output=out_acc))
        channels.append(AnimationChannel(
            target=AnimationChannelTarget(node=node_idx, path=path), sampler=s_idx))

    for j in JOINTS:
        q = rotations.get(j)
        if q is None:
            q = np.tile(np.array([0, 0, 0, 1], np.float32), (n_frames, 1))
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(n_frames, 4))
        acc_q = b.add(q, "VEC4", FLOAT)
        _add_channel(JOINT_INDEX[j], "rotation", acc_q)

    pelvis_t = root_translations
    if pelvis_t is None:
        pelvis_t = np.tile(np.asarray(heads["pelvis"], np.float32), (n_frames, 1))
    pelvis_t = np.ascontiguousarray(np.asarray(pelvis_t, dtype=np.float32).reshape(n_frames, 3))
    acc_t = b.add(pelvis_t, "VEC3", FLOAT)
    _add_channel(JOINT_INDEX["pelvis"], "translation", acc_t)

    # 节点：0..21 关节，22 网格
    nodes: List[Node] = []
    for j in JOINTS:
        i = JOINT_INDEX[j]
        parent = None
        for pj in JOINTS:
            if j in children_of(pj):
                parent = pj
                break
        ph = np.zeros(3) if parent is None else np.asarray(heads[parent], np.float64)
        local = (np.asarray(heads[j], np.float64) - ph).tolist()
        kids = [JOINT_INDEX[c] for c in children_of(j)]
        nodes.append(Node(name=j, translation=local, children=kids or None))
    mesh_node_idx = len(nodes)
    nodes.append(Node(name="target_mesh", mesh=0, skin=0))

    material = Material(
        name="mat", doubleSided=True,
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.82, 0.84, 0.88, 1.0], metallicFactor=0.0, roughnessFactor=0.75))

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="ai3d.retarget"),
        scene=0,
        scenes=[Scene(name="Scene", nodes=[0, mesh_node_idx])],
        nodes=nodes,
        meshes=[Mesh(name="target", primitives=[Primitive(
            attributes=Attributes(POSITION=acc_pos, NORMAL=acc_nor,
                                  JOINTS_0=acc_joints, WEIGHTS_0=acc_weights),
            indices=acc_idx, material=0)])],
        materials=[material],
        skins=[Skin(name="rig", inverseBindMatrices=acc_ibm,
                    skeleton=JOINT_INDEX["pelvis"], joints=list(range(nj)))],
        animations=[Animation(name=anim_name, samplers=samplers, channels=channels)],
        accessors=b.accessors,
        bufferViews=b.buffer_views,
        buffers=[Buffer(byteLength=len(b.blob))],
    )
    gltf.set_binary_blob(bytes(b.blob))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out_path))
    return out_path
