"""Data models for skeleton, animation, and weapon structures."""

from ai3d.models.keypoints import (
    FORMAT_COCO_17,
    FORMAT_WHOLEBODY_133,
    WHOLEBODY_GROUPS,
    describe_format,
    format_from_keypoint_count,
    get_bone_connections,
    get_groups,
    get_keypoint_names,
    get_render_3d_indices,
)
from ai3d.models.skeleton import Joint, Skeleton

__all__ = [
    "Joint",
    "Skeleton",
    "FORMAT_COCO_17",
    "FORMAT_WHOLEBODY_133",
    "WHOLEBODY_GROUPS",
    "describe_format",
    "format_from_keypoint_count",
    "get_bone_connections",
    "get_groups",
    "get_keypoint_names",
    "get_render_3d_indices",
]
