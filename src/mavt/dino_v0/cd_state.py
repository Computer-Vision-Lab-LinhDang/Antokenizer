"""C / D state of MAVT on DINOv3, v0.

A video is cut into segments of K = cfg.keyframe_every frames; the first frame of each
segment is its keyframe k.

C_k [N, dim]: one token per patch, built FRESH at each keyframe from the trunk output X_k only
             (self-attn over C + cross-attn to the 3 x 3 cells of X_k around each cell).
D_t [M, d]:  M learned query tokens of width d = cfg.d_dim, updated at EVERY frame relative
             to the current C_k (predict from D_{t-1} and C_k, then correct with X_t).

The mask is asymmetric and enforced by construction (no attention mask needed):
    C_k reads X_k and C            (never X_t for t != k, never D)
    D_t reads X_t, C_k and D
Images (T = 1) have only C.

v1 (track edge C_{k-K} -> C_k, event memory S) is not built: the flags raise NotImplementedError.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import ZeroLinear, relative_pose
from .config import DinoV0Config
from .rope import AxialRoPE


def _check_v0(cfg: DinoV0Config) -> None:
    if cfg.use_track or cfg.use_event_memory:
        raise NotImplementedError("v1")


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head attention, q from xq and k/v from xkv (same width), optional RoPE angles and bool mask.

    q_ang / k_ang: [Nq, head_dim/2] / [Nk, head_dim/2] from AxialRoPE.angles, or None (no rotation).
    attn_mask: [Nq, Nk] bool, True = may attend.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor, rope: Optional[AxialRoPE] = None,
                q_ang: Optional[torch.Tensor] = None, k_ang: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, D = xq.shape
        Nk = xkv.shape[1]
        q = self.q(xq).view(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(xkv).view(B, Nk, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if rope is not None and q_ang is not None:
            q = rope.rotate(q, q_ang)
        if rope is not None and k_ang is not None:
            k = rope.rotate(k, k_ang)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(o.transpose(1, 2).reshape(B, Nq, D))


def neighbourhood_mask(grid: int, radius: int = 1, device=None) -> torch.Tensor:
    """[N, N] bool: cell i may read cell j iff |row_i - row_j| <= r and |col_i - col_j| <= r."""
    r = torch.arange(grid, device=device)
    row, col = r.repeat_interleave(grid), r.repeat(grid)
    return ((row[:, None] - row[None, :]).abs() <= radius) & ((col[:, None] - col[None, :]).abs() <= radius)


# ----------------------------------------------------------------------------------------
# C: built at each keyframe from X_k only
# ----------------------------------------------------------------------------------------

class CBlock(nn.Module):
    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        d = cfg.dim
        self.n1, self.self_attn = nn.LayerNorm(d), Attention(d, cfg.heads)
        self.n2, self.nx, self.cross = nn.LayerNorm(d), nn.LayerNorm(d), Attention(d, cfg.heads)
        self.n3, self.mlp = nn.LayerNorm(d), MLP(d, cfg.mlp_ratio)

    def forward(self, c, x_k, rope, ang, nbr):
        h = self.n1(c)
        c = c + self.self_attn(h, h, rope, ang, ang)
        c = c + self.cross(self.n2(c), self.nx(x_k), rope, ang, ang, attn_mask=nbr)
        return c + self.mlp(self.n3(c))


class CBuilder(nn.Module):
    """C_k = f(C0, X_k): learned per-cell init, c_blocks x (self-attn over C, 3 x 3 cross-attn to X_k, MLP)."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        _check_v0(cfg)
        self.cfg = cfg
        self.c0 = nn.Parameter(torch.randn(cfg.n_patches, cfg.dim) * 0.02)
        self.rope = AxialRoPE(cfg.dim // cfg.heads)
        self.blocks = nn.ModuleList([CBlock(cfg) for _ in range(cfg.c_blocks)])
        self.norm = nn.LayerNorm(cfg.dim)
        # additive float mask (0 / -inf): bool masks gave zero gradients on some SDPA backends
        nbr = torch.zeros(cfg.n_patches, cfg.n_patches).masked_fill(~neighbourhood_mask(cfg.grid), float("-inf"))
        self.register_buffer("nbr", nbr, persistent=False)

    def forward(self, x_k: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x_k [B, N, dim], coords / mask [N, 4] -> C_k [B, N, dim]."""
        B = x_k.shape[0]
        ang = self.rope.angles(coords, mask)                     # cells share the patch coords of X_k
        c = self.c0.unsqueeze(0).expand(B, -1, -1).to(x_k.dtype)
        for blk in self.blocks:
            c = blk(c, x_k, self.rope, ang, self.nbr.to(x_k.dtype))
        return self.norm(c)


# ----------------------------------------------------------------------------------------
# D: updated every frame relative to C_k
# ----------------------------------------------------------------------------------------

class DBlock(nn.Module):
    """Pre-LN: D <- D + attn(D, [ctx ; D]) + MLP, ctx (width dim) enters through Linear(dim -> d)."""

    def __init__(self, cfg: DinoV0Config, n_ctx: int, self_first: bool):
        super().__init__()
        d, h = cfg.d_dim, max(1, cfg.d_dim // 32)
        self.self_first = self_first
        if self_first:
            self.ns, self.self_attn = nn.LayerNorm(d), Attention(d, h)
        self.ctx_norm = nn.ModuleList([nn.LayerNorm(cfg.dim) for _ in range(n_ctx)])
        self.ctx_proj = nn.ModuleList([nn.Linear(cfg.dim, d) for _ in range(n_ctx)])
        self.nq, self.cross = nn.LayerNorm(d), Attention(d, h)
        self.nm, self.mlp = nn.LayerNorm(d), MLP(d, cfg.mlp_ratio)

    def forward(self, D: torch.Tensor, ctx: list, include_self: bool) -> torch.Tensor:
        if self.self_first:
            h = self.ns(D)
            D = D + self.self_attn(h, h)
        h = self.nq(D)
        kv = [p(n(x)) for x, n, p in zip(ctx, self.ctx_norm, self.ctx_proj)]
        if include_self:
            kv.append(h)
        D = D + self.cross(h, torch.cat(kv, 1))
        return D + self.mlp(self.nm(D))


class DState(nn.Module):
    """D tokens: predict(D_prev, C_k) -> D_hat; correct(D_hat, X_t, C_k) -> D_t; up(D) -> width dim. No RoPE (mask 0000)."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        _check_v0(cfg)
        self.cfg = cfg
        self.d0 = nn.Parameter(torch.randn(cfg.d_tokens, cfg.d_dim) * 0.02)
        self.pred = DBlock(cfg, n_ctx=1, self_first=True)        # self-attn over D, cross-attn D -> C_k, MLP
        self.corr = nn.ModuleList([DBlock(cfg, n_ctx=2, self_first=False) for _ in range(cfg.d_blocks)])
        self.up_proj = nn.Linear(cfg.d_dim, cfg.dim)
        # target-view pose relative to the keyframe view, an 'action' for the prediction; zero at init
        self.pose_proj = nn.Sequential(nn.Linear(12, cfg.d_dim), nn.GELU(), ZeroLinear(cfg.d_dim, cfg.d_dim))

    def init(self, B: int, dtype=None) -> torch.Tensor:
        D = self.d0.unsqueeze(0).expand(B, -1, -1)
        return D if dtype is None else D.to(dtype)

    def predict(self, D_prev: Optional[torch.Tensor], C_k: torch.Tensor,
                pose: Optional[torch.Tensor] = None) -> torch.Tensor:
        """D_prev [B, M, d] | None, C_k [B, N, dim], pose [B, 12] (target view vs keyframe) -> D_hat [B, M, d]."""
        if D_prev is None:
            D_prev = self.init(C_k.shape[0], C_k.dtype)
        if pose is not None:
            D_prev = D_prev + self.pose_proj(pose.to(D_prev.dtype)).unsqueeze(1)
        return self.pred(D_prev, [C_k], include_self=False)

    def correct(self, D_hat: torch.Tensor, x_t: torch.Tensor, C_k: torch.Tensor) -> torch.Tensor:
        """D_hat [B, M, d], x_t / C_k [B, N, dim] -> D_t [B, M, d] (attn over [X_t ; C_k ; D])."""
        D = D_hat
        for blk in self.corr:
            D = blk(D, [x_t, C_k], include_self=True)
        return D

    def up(self, D: torch.Tensor) -> torch.Tensor:
        return self.up_proj(D)


# ----------------------------------------------------------------------------------------
# C + D over a clip
# ----------------------------------------------------------------------------------------

class CDState(nn.Module):
    @staticmethod
    def _pose(c2w: Optional[torch.Tensor], ref: int, t: int) -> Optional[torch.Tensor]:
        """Pose of view t in keyframe view ref's frame [B, 12]; None without poses or past the clip."""
        if c2w is None or t >= c2w.shape[1]:
            return None
        return relative_pose(c2w[:, ref], c2w[:, t])

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        _check_v0(cfg)
        self.cfg = cfg
        self.c_builder = CBuilder(cfg)
        self.d_state = DState(cfg)

    def forward(self, X: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor,
                c2w: Optional[torch.Tensor] = None) -> dict:
        """X [B, T, N, dim], coords / mask [T, N, 4], c2w [B, T, 4, 4] for posed views (else None) ->
        C [B, S, N, dim], seg [T] long, D / D_hat / D_next [B, T, M, d] (None when T == 1)."""
        B, T = X.shape[:2]
        K = self.cfg.keyframe_every
        S = math.ceil(T / K)
        seg = torch.arange(T, device=X.device) // K
        Cs = [self.c_builder(X[:, s * K], coords[s * K], mask[s * K]) for s in range(S)]
        C = torch.stack(Cs, 1)
        if T == 1:
            return dict(C=C, seg=seg, D=None, D_hat=None, D_next=None)
        Ds, D_hats, D_nexts = [], [], []
        D_prev = None
        for t in range(T):
            C_k = Cs[t // K]
            # the prediction is made before X_t arrives, so it uses the C of frame t-1's segment;
            # at a keyframe that is the previous segment's C, never the C built from X_t
            D_hat = (self.d_state.init(B, X.dtype) if t == 0
                     else self.d_state.predict(D_prev, Cs[(t - 1) // K], self._pose(c2w, ((t - 1) // K) * K, t)))
            D_t = self.d_state.correct(D_hat, X[:, t], C_k)
            D_nexts.append(self.d_state.predict(D_t, C_k, self._pose(c2w, (t // K) * K, t + 1)))
            D_hats.append(D_hat)
            Ds.append(D_t)
            D_prev = D_t
        return dict(C=C, seg=seg, D=torch.stack(Ds, 1), D_hat=torch.stack(D_hats, 1),
                    D_next=torch.stack(D_nexts, 1))
