"""spconv 的纯 torch CPU shim（仅供 UniRig PTv3 蒙皮模型在 macOS/CPU 推理使用）。

真实 spconv（PyPI）无 macOS arm64 wheel 且依赖 CUDA；UniRig 的 PTv3Object 仅用到
三个入口：`import spconv.pytorch as spconv` 后的
  - spconv.SparseConvTensor（structure.Point.sparsify 构造 / modules.PointSequential isinstance）
  - spconv.SubMConv3d（Block.cpe 与 Embedding.stem 的条件位置编码卷积）
  - spconv.modules.is_spconv_module（PointSequential 分派）
本 shim 以朴素 gather-gemm 实现子流形稀疏卷积，权重布局与 spconv 2.x GPU
implicit-gemm 保存的 ckpt 一致：weight shape = (out_channels, kx, ky, kz, in_channels)。
"""

from . import pytorch  # noqa: F401  （令 `import spconv.pytorch` 生效）
from . import modules  # noqa: F401
from .pytorch import SparseConvTensor, SubMConv3d  # noqa: F401
