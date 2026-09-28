"""GLB/FBX I/O。

- FBX → GLB：调用 assimp CLI（仅 FBX 输入需要；缺失时给出安装提示）。
- GLB 读：pygltflib 解析 mesh/skin/skeleton/animation 概要。
- GLB 写：pygltflib save_binary（P1 组装结果时使用）。

前后端 3D 数据统一 GLB；输入若为 FBX 由后端统一归一化为 GLB。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import RetargetSettings, get_settings

logger = logging.getLogger(__name__)

GLB_MAGIC = b"glTF"
_FBX_EXT = {".fbx"}
_GLTF_EXT = {".gltf"}
_GLB_EXT = {".glb"}


class AssimpMissing(RuntimeError):
    """需要 assimp 但未找到（仅 FBX 输入触发）。"""


class ConversionError(RuntimeError):
    """FBX→GLB 转换失败。"""


# --------------------------------------------------------------------------- #
# 格式判定
# --------------------------------------------------------------------------- #
def classify(path: str | Path) -> str:
    """返回 'glb' | 'gltf' | 'fbx' | 'unknown'（先看魔数，再看扩展名）。"""
    p = Path(path)
    if not p.exists():
        return "unknown"
    try:
        with p.open("rb") as f:
            if f.read(4) == GLB_MAGIC:
                return "glb"
    except OSError:
        pass
    ext = p.suffix.lower()
    if ext in _GLB_EXT:
        return "glb"
    if ext in _GLTF_EXT:
        return "gltf"
    if ext in _FBX_EXT:
        return "fbx"
    return "unknown"


# --------------------------------------------------------------------------- #
# assimp
# --------------------------------------------------------------------------- #
def resolve_assimp(settings: Optional[RetargetSettings] = None) -> Optional[str]:
    """解析 assimp 可执行文件；找不到返回 None。"""
    settings = settings or get_settings()
    candidate = settings.assimp.path
    if candidate and Path(candidate).is_file() and os_access_executable(candidate):
        return candidate
    found = shutil.which(candidate or "assimp")
    return found


def os_access_executable(path: str) -> bool:
    import os
    return os.access(path, os.X_OK)


def assimp_available(settings: Optional[RetargetSettings] = None) -> bool:
    return resolve_assimp(settings) is not None


def convert_fbx_to_glb(fbx_path: str | Path, glb_path: str | Path,
                       settings: Optional[RetargetSettings] = None) -> Path:
    """用 assimp 把 FBX 导出为二进制 glTF(.glb)，保留骨架/蒙皮/动画。"""
    fbx_path, glb_path = Path(fbx_path), Path(glb_path)
    assimp = resolve_assimp(settings)
    if not assimp:
        raise AssimpMissing(
            "输入为 FBX，需要 assimp 做转换，但未找到。请安装：`brew install assimp`，"
            "或在 src/ai3d/config/retarget.yaml 的 assimp.path 指定绝对路径；"
            "也可直接上传 GLB（无需 assimp）。")
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [assimp, "export", str(fbx_path), str(glb_path)]
    logger.info("assimp 转换：%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not glb_path.exists():
        raise ConversionError(
            f"assimp 转换失败(code={proc.returncode})：{proc.stderr.strip() or proc.stdout.strip()}")
    return glb_path


def normalize_to_glb(src: str | Path, dst_glb: str | Path,
                     settings: Optional[RetargetSettings] = None) -> Path:
    """把任意输入（FBX/GLB/glTF）归一化为单文件 GLB 写到 dst_glb。

    格式转换后统一做坐标轴归一（axis.enabled）：把不同工具的轴系旋到
    规范系（+Y up / +Z forward / +X left），探测结果写 <stem>_axis_frame.json
    侧车文件（与产物同目录，如 source_axis_frame.json）。
    """
    src, dst_glb = Path(src), Path(dst_glb)
    kind = classify(src)
    dst_glb.parent.mkdir(parents=True, exist_ok=True)
    if kind == "fbx":
        out = convert_fbx_to_glb(src, dst_glb, settings)
    elif kind == "glb":
        if src.resolve() != dst_glb.resolve():
            shutil.copyfile(src, dst_glb)
        out = dst_glb
    elif kind == "gltf":
        out = convert_gltf_to_glb(src, dst_glb)
    else:
        raise ConversionError(f"不支持的输入格式：{src.name}（仅支持 FBX / GLB / glTF）")
    settings = settings or get_settings()
    if settings.axis.enabled:
        from ai3d.retarget import axis_norm
        info = axis_norm.normalize_axes_file(out)
        try:
            (Path(out).parent / f"{Path(out).stem}_axis_frame.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
    return out


def convert_gltf_to_glb(gltf_path: str | Path, glb_path: str | Path) -> Path:
    """.gltf(+.bin) → .glb（用 pygltflib 内嵌缓冲）。"""
    from pygltflib import GLTF2
    gltf_path, glb_path = Path(gltf_path), Path(glb_path)
    g = GLTF2().load(str(gltf_path))
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(glb_path))
    return glb_path


# --------------------------------------------------------------------------- #
# 读取 / 概要
# --------------------------------------------------------------------------- #
def _load(path: str | Path):
    """加载 .glb/.gltf 为 pygltflib GLTF2 对象。"""
    from pygltflib import GLTF2
    p = Path(path)
    if classify(p) == "glb":
        return GLTF2().load_binary(str(p))
    return GLTF2().load(str(p))


def read_summary(path: str | Path) -> Dict[str, Any]:
    """解析 GLB/glTF 结构概要，用于 PRECHECK 与报告。"""
    p = Path(path)
    g = _load(p)
    nodes = getattr(g, "nodes", []) or []
    meshes = getattr(g, "meshes", []) or []
    skins = getattr(g, "skins", []) or []
    anims = getattr(g, "animations", []) or []
    node_names = [n.name for n in nodes if getattr(n, "name", None)]
    skin_joint_counts = [len(s.joints or []) for s in skins]
    anim_channel_counts = [len(a.channels or []) for a in anims]
    anim_names = [a.name for a in anims if getattr(a, "name", None)]
    # 顶点/图元统计
    prim_count = sum(len(m.primitives or []) for m in meshes)
    return {
        "format": classify(p),
        "file_size": p.stat().st_size if p.exists() else 0,
        "scenes": len(getattr(g, "scenes", []) or []),
        "nodes": len(nodes),
        "node_names": node_names[:64],
        "meshes": len(meshes),
        "primitives": prim_count,
        "materials": len(getattr(g, "materials", []) or []),
        "images": len(getattr(g, "images", []) or []),
        "skins": len(skins),
        "skin_joint_counts": skin_joint_counts,
        "animations": len(anims),
        "animation_names": anim_names,
        "anim_channel_counts": anim_channel_counts,
        "accessors": len(getattr(g, "accessors", []) or []),
        "has_skeleton": len(skins) > 0,
        "has_animation": len(anims) > 0,
    }


def list_bones(path: str | Path) -> List[str]:
    """列出所有 skin 引用的关节节点名（源骨架分析用）。"""
    g = _load(path)
    nodes = getattr(g, "nodes", []) or []
    names: List[str] = []
    for skin in (getattr(g, "skins", []) or []):
        for ji in (skin.joints or []):
            if 0 <= ji < len(nodes) and nodes[ji].name:
                names.append(nodes[ji].name)
    return names


def save_glb(g, glb_path: str | Path) -> Path:
    """把 GLTF2 对象保存为 .glb。"""
    glb_path = Path(glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(glb_path))
    return glb_path


# --------------------------------------------------------------------------- #
# 读取：accessor → numpy，及 mesh/skin/skeleton/animation 提取
# --------------------------------------------------------------------------- #
import numpy as np  # noqa: E402

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
         5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
          "MAT2": 4, "MAT3": 9, "MAT4": 16}


def read_accessor(g, accessor_index: int) -> np.ndarray:
    """把 accessor 读为 numpy 数组 (count, ncomp)；MAT4 返回 (count,4,4)。支持 byteStride。"""
    acc = g.accessors[accessor_index]
    bv = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    dtype = _COMP[acc.componentType]
    n = _NCOMP[acc.type]
    count = acc.count
    base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    itemsize = np.dtype(dtype).itemsize
    stride = getattr(bv, "byteStride", None)
    if stride and stride != n * itemsize:
        out = np.empty((count, n), dtype=dtype)
        for i in range(count):
            out[i] = np.frombuffer(blob, dtype, n, base + i * stride)
    else:
        out = np.frombuffer(blob, dtype, count * n, base).reshape(count, n).copy()
    if acc.type == "MAT4":
        # glTF mat4 为列主序；(count,16) → (count,4,4) 每行是一个列向量
        return out.reshape(count, 4, 4)
    return out


def merge_mesh(g) -> Dict[str, Any]:
    """合并所有 mesh/primitive 的顶点与索引（单一 POSITION/NORMAL/indices）。"""
    positions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    voff = 0
    has_normals = True
    for mesh in (getattr(g, "meshes", []) or []):
        for prim in (mesh.primitives or []):
            attrs = prim.attributes
            if attrs.POSITION is None:
                continue
            pos = read_accessor(g, attrs.POSITION).astype(np.float32)
            positions.append(pos)
            if attrs.NORMAL is not None:
                normals.append(read_accessor(g, attrs.NORMAL).astype(np.float32))
            else:
                has_normals = False
                normals.append(np.zeros_like(pos))
            if prim.indices is not None:
                idx = read_accessor(g, prim.indices).astype(np.uint32).ravel() + voff
            else:
                idx = np.arange(voff, voff + len(pos), dtype=np.uint32)
            indices.append(idx)
            voff += len(pos)
    if not positions:
        return {"positions": np.zeros((0, 3), np.float32),
                "normals": np.zeros((0, 3), np.float32),
                "indices": np.zeros((0,), np.uint32), "has_normals": False}
    P = np.concatenate(positions, 0)
    N = np.concatenate(normals, 0) if normals else np.zeros_like(P)
    I = np.concatenate(indices, 0) if indices else np.arange(len(P), dtype=np.uint32)
    return {"positions": P, "normals": N, "indices": I, "has_normals": has_normals}


def mesh_bbox(g) -> Dict[str, List[float]]:
    m = merge_mesh(g)
    p = m["positions"]
    if len(p) == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {"min": p.min(0).tolist(), "max": p.max(0).tolist()}


def extract_nodes(g) -> List[Dict[str, Any]]:
    """所有节点的层级/变换概要。"""
    out = []
    for i, n in enumerate(getattr(g, "nodes", []) or []):
        out.append({
            "index": i, "name": n.name,
            "children": list(n.children or []),
            "translation": list(n.translation) if n.translation is not None else None,
            "rotation": list(n.rotation) if n.rotation is not None else None,
            "scale": list(n.scale) if n.scale is not None else None,
            "matrix": list(n.matrix) if n.matrix is not None else None,
            "mesh": n.mesh, "skin": n.skin,
        })
    return out


def extract_skin(g, skin_index: int = 0) -> Optional[Dict[str, Any]]:
    """提取 skin：关节节点索引/名称、逆绑定矩阵、skeleton 根。"""
    skins = getattr(g, "skins", []) or []
    if not skins:
        return None
    skin = skins[skin_index]
    nodes = getattr(g, "nodes", []) or []
    joints = list(skin.joints or [])
    names = [nodes[j].name if 0 <= j < len(nodes) else None for j in joints]
    ibm = read_accessor(g, skin.inverseBindMatrices) if skin.inverseBindMatrices is not None else None
    return {"joints": joints, "names": names, "inverse_bind_matrices": ibm,
            "skeleton": skin.skeleton}


def extract_animations(g) -> List[Dict[str, Any]]:
    """提取动画：每个 channel 的 target 节点、path、时间轴与采样值。"""
    anims = []
    for a in (getattr(g, "animations", []) or []):
        channels = []
        for ch in (a.channels or []):
            sampler = a.samplers[ch.sampler]
            times = read_accessor(g, sampler.input).astype(np.float32).ravel()
            values = read_accessor(g, sampler.output).astype(np.float32)
            channels.append({
                "target_node": ch.target.node,
                "path": ch.target.path,           # translation|rotation|scale|weights
                "interpolation": sampler.interpolation or "LINEAR",
                "times": times, "values": values,  # values: (F, ncomp)
            })
        anims.append({"name": a.name, "channels": channels,
                      "duration": float(max((c["times"].max() for c in channels
                                             if len(c["times"])), default=0.0))})
    return anims
