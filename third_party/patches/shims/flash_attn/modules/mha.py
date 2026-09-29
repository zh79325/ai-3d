"""flash_attn.modules.mha.MHA 的纯 torch CPU shim。

参数名与 flash-attn 官方 MHA 严格一致，确保 UniRig 蒙皮 ckpt 的 state_dict
可原样加载（load_state_dict strict）：
  - cross_attn=True：Wq (embed←q_dim)，Wkv (2*embed←kv_dim)，out_proj
  - cross_attn=False：Wqkv (3*embed)，out_proj
UniRig 仅以 `MHA(embed_dim, num_heads, cross_attn=True)` + `attention(q, x_kv=kv)`
使用；rotary/norm 等可选项未启用（ckpt 中无对应权重）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cross_attn: bool = False,
        qkv_dim: int = None,
        kv_dim: int = None,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cross_attn = cross_attn
        q_dim = qkv_dim if qkv_dim is not None else embed_dim
        k_dim = kv_dim if kv_dim is not None else embed_dim
        if cross_attn:
            self.Wq = nn.Linear(q_dim, embed_dim, bias=bias)
            self.Wkv = nn.Linear(k_dim, 2 * embed_dim, bias=bias)
        else:
            self.Wqkv = nn.Linear(q_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, L, embed) → (B, H, L, D)
        b, seq, _ = t.shape
        return t.view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        qkv: torch.Tensor = None,
        x: torch.Tensor = None,
        x_kv: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.cross_attn:
            q_in = x if x is not None else qkv
            kv_in = x_kv if x_kv is not None else q_in
            assert q_in is not None, "cross_attn MHA 需要 q（x= 或位置参数）"
            q = self.Wq(q_in)
            k, v = self.Wkv(kv_in).chunk(2, dim=-1)
        else:
            q_in = x if x is not None else qkv
            q, k, v = self.Wqkv(q_in).chunk(3, dim=-1)

        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        o = F.scaled_dot_product_attention(qh, kh, vh)
        # (B, H, Lq, D) → (B, Lq, embed)
        o = o.transpose(1, 2).reshape(o.shape[0], o.shape[2], self.embed_dim)
        return self.out_proj(o)
