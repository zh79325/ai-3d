"""UniRig 骨架蒙皮新链路（method='unirig'）。

进程内 import vendored UniRig（``third_party/UniRig``，由 ``scripts/setup_unirig.py``
clone + patch），CPU 推理。两个子模块：

- :mod:`predict`：extract（trimesh 读 GLB → raw_data.npz）→ 骨架预测 → 蒙皮预测，
  合并产物写 ``unirig_raw.npz``；
- :mod:`adapt`：把 UniRig 的任意骨骼树语义适配到本项目 22 关节骨架，聚合蒙皮权重
  并回传到全分辨率网格，产出与旧链路同契约的 rig.json / skin.npz / skin_report.json。

旧 S2 链路（DWPose 多视角 / 比例骨架 + libigl BBW）零改动；本包只被
``job_worker.run_binding_unirig`` 调用。
"""
