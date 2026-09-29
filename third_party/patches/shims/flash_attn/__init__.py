"""flash_attn 的纯 torch CPU shim。

UniRig 用到两处：
  - flash_attn.flash_attn_varlen_qkvpacked_func（pointcept SerializedAttention，
    enable_flash=True 路径）→ 以 F.scaled_dot_product_attention 分段等价实现；
  - flash_attn.modules.mha.MHA（unirig_skin.ResidualCrossAttn，cross_attn=True）
    → 见 modules/mha.py，参数名 Wq/Wqkv/Wout 与 flash-attn 一致以匹配 ckpt。

数值说明：输入常为 .half()（CPU 上 fp16 算子支持差），内部一律升 float32 计算并
返回 float32；调用方随后 .to(qkv.dtype) 不受影响。
"""

import torch
import torch.nn.functional as F

from . import modules  # noqa: F401


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """qkv: (total, 3, H, D)；cu_seqlens: (num_segs+1,) int32。返回 (total, H, D)。"""
    qkv = qkv.float()
    total, _, n_heads, head_dim = qkv.shape
    n_segs = cu_seqlens.shape[0] - 1
    if n_segs <= 0:
        return qkv.new_zeros(total, n_heads, head_dim)

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    k = int(seqlens[0].item())
    uniform = k > 0 and bool((seqlens == k).all()) and total == k * n_segs

    if uniform:
        # 等长段（PTv3 的 get_padding_and_inverse 保证按 patch_size 对齐）→ 一次 batched sdpa
        seg = qkv.reshape(n_segs, k, 3, n_heads, head_dim)
        q, kk, v = seg.unbind(2)  # (B', K, H, D)
        q = q.transpose(1, 2)
        kk = kk.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, kk, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        return o.transpose(1, 2).reshape(total, n_heads, head_dim)

    # 变长回退：逐段 sdpa
    outs = []
    for i in range(n_segs):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        q, kk, v = qkv[s:e].unbind(1)  # (L, H, D)
        q = q.transpose(0, 1).unsqueeze(0)
        kk = kk.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(
            q, kk, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        outs.append(o.squeeze(0).transpose(0, 1))  # (L, H, D)
    return torch.cat(outs, dim=0)
