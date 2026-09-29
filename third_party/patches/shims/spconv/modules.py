"""spconv.modules 的 shim：PointSequential 用它分派稀疏卷积模块。"""


def is_spconv_module(module) -> bool:
    from .pytorch import SubMConv3d

    return isinstance(module, SubMConv3d)
