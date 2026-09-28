"""组装 result.glb：目标网格 + 骨架(22 关节) + 蒙皮权重 + 逐帧动画。

绑定姿态用纯平移定义（关节 rest 局部旋转=单位），故：
- 关节节点 local translation = head − parent_head（根=pelvis 全局 head）
- inverseBindMatrix = translate(−head_global)（列主序 16 float）
- 动画 = 每关节 rotation 通道（局部四元数）+ pelvis translation 通道（root motion，绝对局部平移）
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    Sampler,
    Scene,
    Skin,
    Texture,
)

from . import glb_io
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
        bv_idx = self.add_raw(raw, target)
        n = arr.shape[0] if count is None else count
        kw = dict(bufferView=bv_idx, componentType=comp, count=n, type=gltf_type)
        if minmax and n > 0:
            flat = arr.reshape(n, -1)
            kw["min"] = flat.min(0).tolist()
            kw["max"] = flat.max(0).tolist()
        acc_idx = len(self.accessors)
        self.accessors.append(Accessor(**kw))
        return acc_idx

    def add_raw(self, raw: bytes, target: Optional[int] = None) -> int:
        """裸字节 bufferView（image 等无 accessor 数据用）。"""
        self._pad(4)
        off = len(self.blob)
        self.blob.extend(raw)
        bv_idx = len(self.buffer_views)
        self.buffer_views.append(
            BufferView(buffer=0, byteOffset=off, byteLength=len(raw), target=target))
        return bv_idx


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


def _src_chunks(src_gltf, n_verts: int) -> List[Tuple[Any, int, int, int, int]]:
    """按 merge_mesh 的 chunk 顺序（meshes×prims×节点实例）切片。

    返回 [(prim, vcount, voff, icount, ioff)]：顶点与索引两套偏移（索引空间
    与顶点空间长度不同，不可混用）；与合并顶点数不匹配时返回 []（回退单 primitive）。
    """
    if src_gltf is None or not getattr(src_gltf, "meshes", None):
        return []
    inst: Dict[int, int] = {}
    for nd in (getattr(src_gltf, "nodes", []) or []):
        if getattr(nd, "mesh", None) is not None:
            inst[nd.mesh] = inst.get(nd.mesh, 0) + 1
    chunks: List[Tuple[Any, int, int, int, int]] = []
    voff = ioff = 0
    for mi, sm in enumerate(src_gltf.meshes):
        for prim in (sm.primitives or []):
            if prim.attributes.POSITION is None:
                continue
            cnt = src_gltf.accessors[prim.attributes.POSITION].count
            icnt = (src_gltf.accessors[prim.indices].count
                    if prim.indices is not None else cnt)
            for _ in range(inst.get(mi, 1)):
                chunks.append((prim, cnt, voff, icnt, ioff))
                voff += cnt
                ioff += icnt
    return chunks if voff == n_verts else []


def _image_bytes(g, img) -> Optional[bytes]:
    """取源 glb 内嵌图片字节（bufferView 或 data URI）。"""
    if img.bufferView is not None:
        blob = g.binary_blob()
        if blob:
            bv = g.bufferViews[img.bufferView]
            off = bv.byteOffset or 0
            return bytes(blob[off:off + bv.byteLength])
    uri = getattr(img, "uri", None)
    if uri and uri.startswith("data:"):
        import base64
        _, _, b64 = uri.partition(",")
        try:
            return base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            return None
    return None


def _copy_materials(g, b: "_Builder", used_mats: set):
    """把源 glb 的材质/纹理/采样器/图片拷入新 blob，返回新列表与材质索引 remap。"""
    mats_src = g.materials or []
    texs_src = g.textures or []
    sams_src = g.samplers or []
    imgs_src = g.images or []
    new_mats: List[Material] = []
    new_texs: List[Texture] = []
    new_sams: List[Sampler] = []
    new_imgs = []
    sam_remap: Dict[int, int] = {}
    tex_remap: Dict[int, Optional[int]] = {}
    mat_remap: Dict[int, int] = {}

    def remap_sampler(si):
        if si is None or si >= len(sams_src):
            return None
        if si not in sam_remap:
            new_sams.append(copy.deepcopy(sams_src[si]))
            sam_remap[si] = len(new_sams) - 1
        return sam_remap[si]

    def remap_tex(ti):
        if ti is None or ti >= len(texs_src):
            return None
        if ti not in tex_remap:
            src = texs_src[ti]
            new_img = None
            if src.source is not None and src.source < len(imgs_src):
                data = _image_bytes(g, imgs_src[src.source])
                if data:
                    ni = copy.deepcopy(imgs_src[src.source])
                    ni.uri = None
                    ni.bufferView = b.add_raw(data)
                    new_imgs.append(ni)
                    new_img = len(new_imgs) - 1
            if new_img is None:
                tex_remap[ti] = None
            else:
                new_texs.append(Texture(sampler=remap_sampler(src.sampler),
                                        source=new_img))
                tex_remap[ti] = len(new_texs) - 1
        return tex_remap[ti]

    def fix_ref(holder, attr):
        ref = getattr(holder, attr, None)
        if ref is None or getattr(ref, "index", None) is None:
            return
        ni = remap_tex(ref.index)
        if ni is None:
            setattr(holder, attr, None)
        else:
            ref.index = ni

    for mi in sorted(used_mats):
        if mi is None or mi >= len(mats_src):
            continue
        m = copy.deepcopy(mats_src[mi])
        pbr = m.pbrMetallicRoughness
        if pbr is not None:
            fix_ref(pbr, "baseColorTexture")
            fix_ref(pbr, "metallicRoughnessTexture")
        fix_ref(m, "normalTexture")
        fix_ref(m, "occlusionTexture")
        fix_ref(m, "emissiveTexture")
        new_mats.append(m)
        mat_remap[mi] = len(new_mats) - 1
    return new_mats, new_texs, new_sams, new_imgs, mat_remap


def build_result_glb(
    mesh: Dict[str, np.ndarray],
    heads: Dict[str, np.ndarray],
    joints_u16: np.ndarray,      # (N,4) uint16，取值 0..21（JOINTS 顺序）
    weights_f32: np.ndarray,     # (N,4) float32，和≈1
    times: np.ndarray,           # (F,) float32，递增
    rotations: Dict[str, np.ndarray],   # joint -> (F,4) 四元数(x,y,z,w)
    root_translations: Optional[np.ndarray],  # (F,3) pelvis 绝对局部平移
    out_path: str | Path,
    src_gltf=None,                       # 源目标 glb：提供时保留原材质/UV/纹理
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
    chunks = _src_chunks(src_gltf, n_verts)
    if not chunks:
        use_u16_idx = n_verts < 65536
        acc_pos = b.add(positions, "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)
        acc_nor = b.add(normals, "VEC3", FLOAT, ARRAY_BUFFER)
        acc_idx = b.add(idx_arr, "SCALAR",
                        UNSIGNED_SHORT if use_u16_idx else UNSIGNED_INT,
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

    if chunks:
        used_mats = {p.material for (p, _, _, _, _) in chunks if p.material is not None}
        materials_out, textures_out, samplers_out, images_out, mat_remap = \
            _copy_materials(src_gltf, b, used_mats)
        primitives: List[Primitive] = []
        uv_acc: Dict[int, int] = {}
        for (prim, cnt, off, icnt, ioff) in chunks:
            sl = slice(off, off + cnt)
            attrs = Attributes(
                POSITION=b.add(positions[sl], "VEC3", FLOAT, ARRAY_BUFFER, minmax=True),
                NORMAL=b.add(normals[sl], "VEC3", FLOAT, ARRAY_BUFFER),
                JOINTS_0=b.add(np.ascontiguousarray(joints_u16[sl], dtype=np.uint16),
                               "VEC4", UNSIGNED_SHORT, ARRAY_BUFFER),
                WEIGHTS_0=b.add(np.ascontiguousarray(weights_f32[sl], dtype=np.float32),
                                "VEC4", FLOAT, ARRAY_BUFFER))
            if prim.attributes.TEXCOORD_0 is not None:
                key = prim.attributes.TEXCOORD_0
                if key not in uv_acc:
                    uv = glb_io.read_accessor(src_gltf, key).astype(np.float32)
                    uv_acc[key] = b.add(np.ascontiguousarray(uv), "VEC2", FLOAT,
                                        ARRAY_BUFFER)
                attrs.TEXCOORD_0 = uv_acc[key]
            ci = indices[ioff:ioff + icnt] - off
            u16 = cnt < 65536
            primitives.append(Primitive(
                attributes=attrs,
                indices=b.add(np.ascontiguousarray(
                    ci.astype(np.uint16 if u16 else np.uint32)),
                    "SCALAR", UNSIGNED_SHORT if u16 else UNSIGNED_INT,
                    ELEMENT_ARRAY_BUFFER),
                material=mat_remap.get(prim.material, 0)))
    else:
        materials_out = [Material(
            name="mat", doubleSided=True,
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0], metallicFactor=0.0,
                roughnessFactor=0.75))]
        textures_out, samplers_out, images_out = [], [], []
        primitives = [Primitive(
            attributes=Attributes(POSITION=acc_pos, NORMAL=acc_nor,
                                  JOINTS_0=acc_joints, WEIGHTS_0=acc_weights),
            indices=acc_idx, material=0)]

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="ai3d.retarget"),
        scene=0,
        scenes=[Scene(name="Scene", nodes=[0, mesh_node_idx])],
        nodes=nodes,
        meshes=[Mesh(name="target", primitives=primitives)],
        materials=materials_out,
        textures=textures_out,
        samplers=samplers_out,
        images=images_out,
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
