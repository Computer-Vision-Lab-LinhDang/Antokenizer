"""Shared building blocks: 3-axis RoPE (t, x, y), the stream block, a window read and an MLP.

A stream block is pre-norm attention + MLP whose key set is assembled by the caller:
    keys = [own tokens ; keys handed in by another stream ; read-only context (DINOv3 memory, residual)]
Own tokens and context are projected with this block's weights; keys from another stream arrive already
projected by that stream (K_C, V_C of block C go into block D, never the other way).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dino_v0.camera import ZeroLinear
from ..dino_v0.rope import apply_rope, rope_tables


class Rope3(nn.Module):
    """RoPE over (t, x, y) with dims rope_dims per head (e.g. 16 / 24 / 24 of 64). coords [..., N, 3],
    mask [..., N, 3] in {0, 1} (an axis at 0 does not rotate) -> (cos, sin) [..., 1, N, hd]."""

    def __init__(self, rope_dims: Sequence[int], base: float = 100.0):
        super().__init__()
        self.dims = tuple(rope_dims)
        for a, d in enumerate(self.dims):
            inv = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
            self.register_buffer(f"inv{a}", inv, persistent=False)

    def tables(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None, dtype=None) -> tuple:
        c = coords.float() if mask is None else coords.float() * mask.float()
        ang = torch.cat([c[..., a:a + 1] * getattr(self, f"inv{a}") for a in range(len(self.dims))], -1)
        cos, sin = rope_tables(ang.unsqueeze(-3), dtype)
        return cos, sin


def rotate(x: torch.Tensor, rope: Optional[tuple]) -> torch.Tensor:
    if rope is None:
        return x
    return apply_rope(x, rope[0].to(x.dtype), rope[1].to(x.dtype))


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """[..., N, H * hd] -> [..., H, N, hd]."""
    return x.unflatten(-1, (heads, -1)).transpose(-3, -2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-3, -2).flatten(-2)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """SDPA on [..., H, N, hd] with any number of batch dims (flattened to 4-D so the fused kernels apply);
    mask bool, broadcastable to [..., H, Nq, Nk], True = may attend."""
    if q.dim() == 4:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    batch = q.shape[:-3]
    if mask is not None:
        mask = mask.expand(*batch, *mask.shape[-3:]).reshape(-1, *mask.shape[-3:])
    out = F.scaled_dot_product_attention(q.reshape(-1, *q.shape[-3:]), k.reshape(-1, *k.shape[-3:]),
                                         v.reshape(-1, *v.shape[-3:]), attn_mask=mask)
    return out.view(*batch, *out.shape[-3:])


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, out: Optional[int] = None):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hid), nn.Linear(hid, out or dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class StreamBlock(nn.Module):
    """Pre-norm attention + MLP. Use: q, k, v = own(h, rope); kc, vc = ctx(m, rope_m);
    h = finish(h, q, [k, ...], [v, ...], mask)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, ctx: bool = True):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        if ctx:
            self.norm_ctx = nn.LayerNorm(dim)
            self.kv_ctx = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def own(self, h: torch.Tensor, rope: Optional[tuple]) -> tuple:
        q, k, v = self.qkv(self.norm1(h)).chunk(3, -1)
        q, k, v = (split_heads(t, self.heads) for t in (q, k, v))
        return rotate(q, rope), rotate(k, rope), v

    def ctx(self, m: torch.Tensor, rope: Optional[tuple]) -> tuple:
        k, v = self.kv_ctx(self.norm_ctx(m)).chunk(2, -1)
        k, v = split_heads(k, self.heads), split_heads(v, self.heads)
        return rotate(k, rope), v

    def finish(self, h: torch.Tensor, q: torch.Tensor, ks: list, vs: list,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """mask: bool, broadcastable to [..., H, Nq, Nk], True = may attend."""
        k, v = (ks[0], vs[0]) if len(ks) == 1 else (torch.cat(ks, -2), torch.cat(vs, -2))
        a = attention(q, k, v, mask)
        h = h + self.out(merge_heads(a))
        return h + self.mlp(self.norm2(h))

    def forward(self, h: torch.Tensor, rope: Optional[tuple] = None, m: Optional[torch.Tensor] = None,
                rope_m: Optional[tuple] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-attention over h, plus read-only context m when given."""
        q, k, v = self.own(h, rope)
        ks, vs = [k], [v]
        if m is not None:
            kc, vc = self.ctx(m, rope_m)
            ks.append(kc)
            vs.append(vc)
        return self.finish(h, q, ks, vs, mask)


def window_mask(grid: int, r: int, device=None) -> torch.Tensor:
    """[N, N] bool: query cell n may read key cell n' when both rows and columns differ by <= r."""
    idx = torch.arange(grid * grid, device=device)
    row, col = idx // grid, idx % grid
    return ((row[:, None] - row[None]).abs() <= r) & ((col[:, None] - col[None]).abs() <= r)


class WindowRead(nn.Module):
    """Cross-attention of each query cell to a (2r+1)^2 window of another token grid (the previous decoded
    frame), output through a zero-init projection so it changes nothing at init."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_k = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)
        self.out = ZeroLinear(dim, dim)

    def forward(self, h: torch.Tensor, ref: torch.Tensor, rope: tuple, mask: torch.Tensor) -> torch.Tensor:
        q = rotate(split_heads(self.q(self.norm_q(h)), self.heads), rope)
        k, v = self.kv(self.norm_k(ref)).chunk(2, -1)
        k, v = rotate(split_heads(k, self.heads), rope), split_heads(v, self.heads)
        return self.out(merge_heads(attention(q, k, v, mask)))
