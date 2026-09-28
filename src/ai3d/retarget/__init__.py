"""FBX/GLB 骨骼动画迁移工具（ai3d 子包）。

前端 Three.js(WebGL) 只负责渲染；本包（后端）负责文件归一化（FBX→GLB）与全部算法：
姿态推理、三维关节求解、自动蒙皮、源骨映射、动画重定向、GLB 导出。

任务用 SQLite 持久化：上传即自动建任务并返回 task_id，服务重启后仍可凭 task_id 找回。
配置文件位于 ``src/ai3d/config/retarget.yaml``。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
