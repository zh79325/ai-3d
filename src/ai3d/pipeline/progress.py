"""
处理进度追踪 - 步骤定义与状态快照。

步骤表是唯一真值来源：后端上报、前端渲染都以此为准（label/order 均由后端下发），
避免前后端各自硬编码步骤清单导致的顺序错乱与名称不一致。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# 步骤状态
STATUS_PENDING = "pending"        # 尚未开始
STATUS_PROCESSING = "processing"  # 正在执行
STATUS_COMPLETED = "completed"    # 已完成
STATUS_SKIPPED = "skipped"        # 按配置跳过（不计入总进度）
STATUS_FAILED = "failed"          # 执行失败


@dataclass
class StepDef:
    """步骤定义

    weight: 在总进度中的权重占比，所有步骤权重之和为 100
    """
    key: str
    label: str
    weight: int


# 处理步骤清单（顺序即前端展示顺序）
STEP_DEFINITIONS: List[StepDef] = [
    StepDef("video_read", "加载视频", 3),
    StepDef("frame_extract", "提取帧", 7),
    StepDef("bg_remove", "背景移除", 15),
    StepDef("pose_estimate", "姿态估计", 35),
    StepDef("annotate_video", "标注视频合成", 10),
    StepDef("depth_convert", "深度转换", 20),
    StepDef("validation", "数据校验", 3),
    StepDef("export_json", "导出动画数据", 7),
]


@dataclass
class StepState:
    """单个步骤的运行时状态"""
    definition: StepDef
    order: int
    status: str = STATUS_PENDING
    current: int = 0
    total: int = 0
    message: str = ""

    @property
    def percent(self) -> int:
        if self.status in (STATUS_COMPLETED, STATUS_SKIPPED):
            return 100
        if self.total <= 0:
            return 0
        return min(100, int(self.current / self.total * 100))

    def to_dict(self) -> dict:
        return {
            "key": self.definition.key,
            "label": self.definition.label,
            "order": self.order,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "percent": self.percent,
            "message": self.message,
        }


class ProgressTracker:
    """步骤进度追踪器

    每次状态变化都通过 callback 下发**全量**步骤快照，
    使前端拿到的步骤列表长度与顺序恒定，每条进度都归属明确的步骤。

    典型用法::

        tracker = ProgressTracker(callback)
        tracker.skip("bg_remove")                    # 按配置不执行的步骤
        tracker.start("pose_estimate", total=100)
        tracker.update("pose_estimate", 20, "姿态估计中 20/100 帧...")
        tracker.finish("pose_estimate", "姿态估计完成")
    """

    def __init__(self, callback: Optional[Callable[[dict], None]] = None,
                 definitions: Optional[List[StepDef]] = None,
                 verbose: bool = True):
        self.callback = callback
        self.verbose = verbose
        defs = definitions or STEP_DEFINITIONS
        self._steps: Dict[str, StepState] = {
            d.key: StepState(definition=d, order=i + 1)
            for i, d in enumerate(defs)
        }
        # 最后一次活动的步骤，用于给出面板外的整体提示语
        self._active_key: Optional[str] = None
        # 已打印过的 (step, message)，避免同一条消息重复刷屏
        self._last_logged: Optional[tuple] = None

    # ---------- 状态变更 ----------

    def skip(self, key: str, message: str = "") -> None:
        """标记步骤为跳过（按配置不执行），不计入总进度分母"""
        step = self._steps.get(key)
        if step is None:
            return
        step.status = STATUS_SKIPPED
        step.message = message
        step.current = 0
        step.total = 0
        self._emit()

    def start(self, key: str, total: int = 0, message: str = "") -> None:
        step = self._steps.get(key)
        if step is None:
            return
        step.status = STATUS_PROCESSING
        step.current = 0
        step.total = total
        step.message = message
        self._active_key = key
        self._emit()

    def update(self, key: str, current: int, message: str = "",
               total: Optional[int] = None) -> None:
        step = self._steps.get(key)
        if step is None:
            return
        step.status = STATUS_PROCESSING
        step.current = current
        if total is not None:
            step.total = total
        if message:
            step.message = message
        self._active_key = key
        self._emit()

    def finish(self, key: str, message: str = "") -> None:
        step = self._steps.get(key)
        if step is None:
            return
        step.status = STATUS_COMPLETED
        if step.total > 0:
            step.current = step.total
        step.message = message
        self._active_key = key
        self._emit()

    def fail(self, key: str, message: str = "") -> None:
        step = self._steps.get(key)
        if step is None:
            return
        step.status = STATUS_FAILED
        step.message = message
        self._active_key = key
        self._emit()

    # ---------- 快照 ----------

    @property
    def overall_percent(self) -> float:
        """总进度：按权重加权，跳过的步骤不计入分母"""
        counted = [s for s in self._steps.values() if s.status != STATUS_SKIPPED]
        total_weight = sum(s.definition.weight for s in counted)
        if total_weight <= 0:
            return 100.0
        done = sum(s.definition.weight * s.percent / 100 for s in counted)
        return round(min(100.0, done / total_weight * 100), 1)

    def snapshot(self) -> dict:
        steps = sorted((s for s in self._steps.values()), key=lambda s: s.order)
        active = self._steps.get(self._active_key) if self._active_key else None
        return {
            "overall_percent": self.overall_percent,
            "current_step": self._active_key,
            "current_step_label": active.definition.label if active else "",
            "message": active.message if active else "",
            "current_frame": active.current if active else 0,
            "total_frames": active.total if active else 0,
            "steps": [s.to_dict() for s in steps],
        }

    def _emit(self) -> None:
        if self.verbose and self._active_key:
            step = self._steps[self._active_key]
            entry = (self._active_key, step.status, step.message)
            if entry != self._last_logged:
                print(f"[{self._active_key}] {step.message}")
                self._last_logged = entry
        if self.callback:
            self.callback(self.snapshot())
