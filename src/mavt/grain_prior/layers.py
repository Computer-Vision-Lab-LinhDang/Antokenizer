"""The attention operator and blocks of section 6.1, shared by every stage.

    CA(Q, E; B) = Concat_h [ softmax_j( Q W_Q^h (E W_K^h)^T / sqrt(d_h) + B^h ) E W_V^h ] W_O

Padding / causal masks enter B as -inf. A query row without any valid evidence returns zero evidence (its
row is never softmaxed over -inf only). Pre-LN block: U = H + CA(LN(H), E; B) for each evidence source, then
an optional self-attention sublayer over the queries, then H' = U + MLP(LN(U)), MLP d -> 4d -> d.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BIAS_FLOOR = -50.0          # biases are clamped here (exp(-50) is no weight; keeps bf16 resolution)


def mlp2(n_in: int, n_out: int, hidden: int) -> nn.Sequential:
    """The 2-layer heads f(.) of the design: Linear -> GELU -> Linear."""
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


def zero_last(m: nn.Module) -> nn.Module:
    """Zero the last Linear of a head: identity / zero proposals at initialisation."""
    last = [x for x in m.modules() if isinstance(x, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return m


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, bias: Optional[torch.Tensor] = None,
           valid: Optional[torch.Tensor] = None, *, strict: bool = False) -> torch.Tensor:
    """q [B, nq, d], k / v [B, ne, d]; bias [B, H or 1, nq, ne] (finite); valid [B, ne] or [B, nq, ne] bool.
    -> [B, nq, d]; rows without a valid key are 0."""
    B, nq, d = q.shape
    ne = k.shape[1]
    sh = lambda t, n: t.reshape(B, n, heads, -1).transpose(1, 2)
    qh, kh, vh = sh(q, nq), sh(k, ne), sh(v, ne)
    m, has = None, None
    if valid is not None:
        ok = valid.view(B, 1, 1 if valid.dim() == 2 else nq, ne).expand(B, 1, nq, ne)
        has = ok.any(-1, keepdim=True)
        ok = ok | ~has
        m = torch.zeros(ok.shape, device=q.device, dtype=q.dtype).masked_fill(~ok, float("-inf"))
    if bias is not None:
        b = (bias.float() if strict else bias.float().clamp_min(BIAS_FLOOR)).to(q.dtype)
        if strict and has is not None:
            b = torch.where(has, b, torch.zeros_like(b))
        m = b if m is None else m + b
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=m)
    if has is not None:
        out = out * has.to(out.dtype)
    return out.transpose(1, 2).reshape(B, nq, d)


class _EntmaxBisect(torch.autograd.Function):
    """alpha-entmax over the last axis by bisection on tau (Peters et al. 2019, entmax_bisect):
    p = [(alpha - 1) x - tau]_+^(1 / (alpha - 1)), sum p = 1. Closed-form backward for x and alpha; -inf entries
    get 0; alpha (> 1) broadcasts against x."""

    @staticmethod
    def forward(ctx, x, alpha, n_iter):
        am1 = alpha - 1
        X = x * am1
        mx = X.max(-1, keepdim=True).values
        tau_lo, tau_hi = mx - 1, mx - (1.0 / x.shape[-1]) ** am1
        p = lambda tau: (X - tau).clamp_min(0) ** (1 / am1)
        f_lo = p(tau_lo).sum(-1, keepdim=True) - 1
        dm = tau_hi - tau_lo
        for _ in range(n_iter):
            dm = dm / 2
            tau_m = tau_lo + dm
            Y = p(tau_m)
            tau_lo = torch.where((Y.sum(-1, keepdim=True) - 1) * f_lo >= 0, tau_m, tau_lo)
        Y = Y / Y.sum(-1, keepdim=True)
        ctx.save_for_backward(Y, alpha)
        return Y

    @staticmethod
    def backward(ctx, dY):
        Y, alpha = ctx.saved_tensors
        pos = Y > 0
        gppr = torch.where(pos, Y.clamp_min(1e-30) ** (2 - alpha), torch.zeros_like(Y))
        dX = dY * gppr
        dX = dX - gppr * dX.sum(-1, keepdim=True) / gppr.sum(-1, keepdim=True)
        d_alpha = None
        if ctx.needs_input_grad[1]:
            S = torch.where(pos, Y * Y.clamp_min(1e-30).log(), torch.zeros_like(Y))
            Ys = gppr / gppr.sum(-1, keepdim=True)
            am1 = alpha - 1
            d_alpha = (dY * (Y - Ys) / am1 ** 2 - dY * (S - Ys * S.sum(-1, keepdim=True)) / am1).sum(-1, keepdim=True)
            d_alpha = d_alpha.sum_to_size(alpha.shape)
        return dX, d_alpha, None


def entmax_bisect(x: torch.Tensor, alpha: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """alpha-entmax over the last axis (alpha > 1, broadcastable to x[..., :1]); 2 = sparsemax, -> 1 softmax."""
    return _EntmaxBisect.apply(x, alpha.to(x.dtype), n_iter)


class CrossAttn(nn.Module):
    def __init__(self, dim: int, heads: int, d_kv: Optional[int] = None):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(d_kv or dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(d_kv or dim, 2 * dim), nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor, bias=None, valid=None, *, strict=False) -> torch.Tensor:
        """Evidence read of LN(h) (not added to h): [B, nq, dim]."""
        k, v = self.kv(self.norm_kv(e)).chunk(2, -1)
        out = self.out(attend(self.q(self.norm_q(h)), k, v, self.heads, bias, valid, strict=strict))
        if strict and valid is not None:
            has = valid.any(-1)
            if has.ndim == 1:
                has = has[:, None]
            out = out * has.unsqueeze(-1).to(out.dtype)
        return out


class SelfAttn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm, self.qkv, self.out = nn.LayerNorm(dim), nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.qkv(self.norm(h)).chunk(3, -1)
        return self.out(attend(q, k, v, self.heads, None, valid))


class Block(nn.Module):
    """Pre-LN block: one CA sublayer per evidence source, optional query self-attention, MLP."""

    def __init__(self, dim: int, heads: int, n_ctx: int = 1, self_attn: bool = True, mlp_ratio: float = 4.0,
                 d_kv: Sequence[Optional[int]] = ()):
        super().__init__()
        d_kv = list(d_kv) + [None] * (n_ctx - len(d_kv))
        self.ca = nn.ModuleList([CrossAttn(dim, heads, d_kv[j]) for j in range(n_ctx)])
        self.sa = SelfAttn(dim, heads) if self_attn else None
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, h: torch.Tensor, ctx: Sequence[tuple] = (), self_valid: Optional[torch.Tensor] = None):
        """ctx: one (e, bias, valid) per CA sublayer; self_valid [B, n, n] bool (e.g. causal) or None."""
        for ca, (e, b, v) in zip(self.ca, ctx):
            h = h + ca(h, e, b, v)
        if self.sa is not None:
            h = h + self.sa(h, self_valid)
        return h + self.mlp(self.norm(h))


def run(blocks: nn.ModuleList, h: torch.Tensor, ctx: Sequence[tuple] = (), self_valid=None) -> torch.Tensor:
    for blk in blocks:
        h = blk(h, ctx, self_valid)
    return h
