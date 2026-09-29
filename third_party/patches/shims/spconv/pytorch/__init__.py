"""spconv.pytorch 的纯 torch CPU 实现（子流形稀疏卷积）。

覆盖 UniRig PTv3Object 用到的全部 API：
  - SparseConvTensor(features, indices, spatial_shape, batch_size) / .replace_feature()
  - SubMConv3d(in, out, kernel_size, bias, indice_key)：朴素 gather-gemm，
    对 kernel 的每个偏移做一次「坐标哈希配对 + index_add_」，
    权重布局 (out_channels, kx, ky, kz, in_channels)，与 spconv 2.x GPU
    implicit-gemm 训练保存的 ckpt 一致（若 ckpt 实为 RSKC (kx,ky,kz,out,in)
    会在 load_state_dict 时 shape 报错，届时按报错调整 permute）。
"""

from typing import Sequence

import torch
import torch.nn as nn


class SparseConvTensor:
    """spconv.SparseConvTensor 的轻量替身（只做特征容器，不做惰性求值）。"""

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Sequence[int],
        batch_size: int,
    ):
        self.features = features
        self.indices = indices  # (N, 4) int: [batch, x, y, z]
        self.spatial_shape = list(spatial_shape)
        self.batch_size = batch_size

    def replace_feature(self, feat: torch.Tensor) -> "SparseConvTensor":
        return SparseConvTensor(feat, self.indices, self.spatial_shape, self.batch_size)

    @property
    def device(self):
        return self.features.device


class SubMConv3d(nn.Module):
    """子流形稀疏卷积：输出位置集合与输入相同，仅对占用体素求值。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        indice_key=None,
        **kwargs,
    ):
        super().__init__()
        assert groups == 1, "shim SubMConv3d 不支持 groups"
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = tuple(int(k) for k in kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.indice_key = indice_key
        # KRSC 布局，匹配 spconv 2.x implicit-gemm ckpt
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, *self.kernel_size, self.in_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, sct: SparseConvTensor) -> SparseConvTensor:
        feats = sct.features  # (N, in)
        n = feats.shape[0]
        out = torch.zeros(
            n, self.out_channels, dtype=feats.dtype, device=feats.device
        )
        if n == 0:
            return sct.replace_feature(out)

        idx = sct.indices.to(torch.int64)  # (N, 4): [b, x, y, z]
        # 单基编码：S 大于任何坐标跨度，保证 (b,x,y,z) 与 key 一一对应
        s = max(int(v) for v in sct.spatial_shape) + 8
        key = ((idx[:, 0] * s + idx[:, 1]) * s + idx[:, 2]) * s + idx[:, 3]
        skey, perm = torch.sort(key)

        center = [k // 2 for k in self.kernel_size]
        kx, ky, kz = self.kernel_size
        for dx in range(kx):
            for dy in range(ky):
                for dz in range(kz):
                    # 卷积 out[j] += W[dk]·in[c_j + dk - center]
                    # 等价于配对 (i, j): c_j = c_i + (center - dk)
                    off = (center[0] - dx, center[1] - dy, center[2] - dz)
                    delta = (off[0] * s + off[1]) * s + off[2]
                    tgt = key + delta
                    pos = torch.searchsorted(skey, tgt)
                    pos.clamp_(max=n - 1)
                    hit = skey[pos] == tgt
                    if not bool(hit.any()):
                        continue
                    src = torch.nonzero(hit, as_tuple=True)[0]
                    dst = perm[pos[src]]
                    wmat = self.weight[:, dx, dy, dz, :]  # (out, in)
                    out.index_add_(0, dst, feats[src] @ wmat.t())

        if self.bias is not None:
            out = out + self.bias
        return sct.replace_feature(out)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"kernel={self.kernel_size}, bias={self.bias is not None}"
        )


def is_spconv_module(module) -> bool:
    """真实 spconv.pytorch.modules.is_spconv_module 的同名替身。"""
    return isinstance(module, SubMConv3d)


class _ModulesShim:
    """`spconv.pytorch.modules` 命名空间替身。

    pointcept 以 `import spconv.pytorch as spconv` 后调 `spconv.modules.is_spconv_module`，
    故本包（spconv.pytorch）必须暴露一个带该函数的 `modules` 属性。
    """

    is_spconv_module = staticmethod(is_spconv_module)


modules = _ModulesShim()
