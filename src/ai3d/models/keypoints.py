"""
关键点定义（单一真值源）

后端估计器、标注绘制、JSON 下发与前端渲染的关键点名称/骨骼拓扑均以此模块为准，
任何一方都不应再自行硬编码索引表。

支持两种格式:
- ``coco_17``: YOLO-pose 输出的 COCO 17 点（仅身体）
- ``coco_wholebody_133``: DWPose/RTMW 输出的 COCO-WholeBody 133 点
"""

from typing import Dict, List, Optional, Tuple

FORMAT_COCO_17 = "coco_17"
FORMAT_WHOLEBODY_133 = "coco_wholebody_133"

# ---------------------------------------------------------------- 关键点名称

# COCO 17 点身体关键点，DWPose 的前 17 点与之完全一致
COCO_17_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 脚部 6 点（索引 17-22）
FEET_NAMES: List[str] = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]

# 面部 68 点（索引 23-90）遵循 300W landmark 顺序，此处不臆测语义命名，仅编号
FACE_NAMES: List[str] = ["face_%d" % i for i in range(68)]


def _hand_names(side: str) -> List[str]:
    """生成单手 21 点名称（腕根 + 5 指 × 4 关节）"""
    names = ["%s_hand_root" % side]
    for finger in ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger"):
        names.extend("%s_%s%d" % (side, finger, i) for i in range(1, 5))
    return names


LEFT_HAND_NAMES: List[str] = _hand_names("left")    # 索引 91-111
RIGHT_HAND_NAMES: List[str] = _hand_names("right")  # 索引 112-132

WHOLEBODY_133_NAMES: List[str] = (
    COCO_17_NAMES + FEET_NAMES + FACE_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES
)

# ---------------------------------------------------------------- 分组区间

# 分组区间为左闭右开，与 numpy 切片一致
WHOLEBODY_GROUPS: Dict[str, Tuple[int, int]] = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "left_hand": (91, 112),
    "right_hand": (112, 133),
}

# ---------------------------------------------------------------- 骨骼拓扑

COCO_17_BONES: List[List[int]] = [
    [0, 1], [1, 3],    # 鼻子 -> 左眼 -> 左耳
    [0, 2], [2, 4],    # 鼻子 -> 右眼 -> 右耳
    [5, 6],            # 左肩 -> 右肩
    [5, 7], [7, 9],    # 左肩 -> 左肘 -> 左手腕
    [6, 8], [8, 10],   # 右肩 -> 右肘 -> 右手腕
    [5, 11], [6, 12],  # 肩膀 -> 髋部
    [11, 12],          # 左髋 -> 右髋
    [11, 13], [13, 15],  # 左髋 -> 左膝 -> 左脚踝
    [12, 14], [14, 16],  # 右髋 -> 右膝 -> 右脚踝
]

# 脚踝 -> 脚趾/脚跟
FEET_BONES: List[List[int]] = [
    [15, 17], [15, 18], [15, 19],
    [16, 20], [16, 21], [16, 22],
]

# 单手骨骼（手内相对索引 0-20，0 为腕根）
HAND_BONES_LOCAL: List[List[int]] = [
    [0, 1], [1, 2], [2, 3], [3, 4],        # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],        # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],   # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20],  # 小指
]


def _offset_bones(bones: List[List[int]], offset: int) -> List[List[int]]:
    return [[a + offset, b + offset] for a, b in bones]


_LEFT_HAND_START = WHOLEBODY_GROUPS["left_hand"][0]
_RIGHT_HAND_START = WHOLEBODY_GROUPS["right_hand"][0]

# 手腕 -> 手掌根，把手接回身体骨架
WRIST_TO_HAND_BONES: List[List[int]] = [
    [9, _LEFT_HAND_START],
    [10, _RIGHT_HAND_START],
]

# 133 点骨骼拓扑：身体 + 脚 + 双手。面部 68 点只作散点绘制，不参与连线
WHOLEBODY_133_BONES: List[List[int]] = (
    COCO_17_BONES
    + FEET_BONES
    + WRIST_TO_HAND_BONES
    + _offset_bones(HAND_BONES_LOCAL, _LEFT_HAND_START)
    + _offset_bones(HAND_BONES_LOCAL, _RIGHT_HAND_START)
)

# ---------------------------------------------------------------- 3D 渲染子集

# 3D 视图渲染的关键点索引：身体 17 + 脚 6 + 双手 42 = 65 点。
# 面部 68 点密度过高，在 3D 里只会糊成一团，仅在标注视频中呈现
WHOLEBODY_RENDER_3D_INDICES: List[int] = (
    list(range(*WHOLEBODY_GROUPS["body"]))
    + list(range(*WHOLEBODY_GROUPS["feet"]))
    + list(range(*WHOLEBODY_GROUPS["left_hand"]))
    + list(range(*WHOLEBODY_GROUPS["right_hand"]))
)

COCO_17_RENDER_3D_INDICES: List[int] = list(range(17))

_FORMATS = {
    FORMAT_COCO_17: {
        "names": COCO_17_NAMES,
        "bones": COCO_17_BONES,
        "render_3d": COCO_17_RENDER_3D_INDICES,
        "groups": {"body": (0, 17)},
    },
    FORMAT_WHOLEBODY_133: {
        "names": WHOLEBODY_133_NAMES,
        "bones": WHOLEBODY_133_BONES,
        "render_3d": WHOLEBODY_RENDER_3D_INDICES,
        "groups": WHOLEBODY_GROUPS,
    },
}


def _spec(keypoint_format: str) -> dict:
    spec = _FORMATS.get(keypoint_format)
    if spec is None:
        raise ValueError(
            "未知的关键点格式: %s，可选: %s" % (keypoint_format, ", ".join(_FORMATS))
        )
    return spec


def get_keypoint_names(keypoint_format: str) -> List[str]:
    """返回指定格式的关键点名称列表"""
    return list(_spec(keypoint_format)["names"])


def get_bone_connections(keypoint_format: str) -> List[List[int]]:
    """返回指定格式的骨骼连接（关键点索引对）"""
    return [list(bone) for bone in _spec(keypoint_format)["bones"]]


def get_render_3d_indices(keypoint_format: str) -> List[int]:
    """返回 3D 视图应渲染的关键点索引子集"""
    return list(_spec(keypoint_format)["render_3d"])


def get_groups(keypoint_format: str) -> Dict[str, List[int]]:
    """返回分组区间 {组名: [start, end)}，供前端按部位着色/开关显示"""
    return {name: [start, end] for name, (start, end) in _spec(keypoint_format)["groups"].items()}


def format_from_keypoint_count(count: int) -> Optional[str]:
    """按关键点数量推断格式，用于兼容不带格式字段的历史数据"""
    for name, spec in _FORMATS.items():
        if len(spec["names"]) == count:
            return name
    return None


def describe_format(keypoint_format: str) -> Dict[str, object]:
    """打包给前端的一份拓扑描述，作为 JSON 中的 keypoint 元数据"""
    return {
        "format": keypoint_format,
        "keypoint_names": get_keypoint_names(keypoint_format),
        "bone_connections": get_bone_connections(keypoint_format),
        "render_3d_indices": get_render_3d_indices(keypoint_format),
        "groups": get_groups(keypoint_format),
    }


__all__ = [
    "FORMAT_COCO_17",
    "FORMAT_WHOLEBODY_133",
    "COCO_17_NAMES",
    "FEET_NAMES",
    "FACE_NAMES",
    "LEFT_HAND_NAMES",
    "RIGHT_HAND_NAMES",
    "WHOLEBODY_133_NAMES",
    "WHOLEBODY_GROUPS",
    "COCO_17_BONES",
    "FEET_BONES",
    "HAND_BONES_LOCAL",
    "WHOLEBODY_133_BONES",
    "WHOLEBODY_RENDER_3D_INDICES",
    "COCO_17_RENDER_3D_INDICES",
    "get_keypoint_names",
    "get_bone_connections",
    "get_render_3d_indices",
    "get_groups",
    "format_from_keypoint_count",
    "describe_format",
]
