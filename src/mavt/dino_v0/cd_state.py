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
    """Multi-head attention, q from xq and k/v from xkv (same width), optional RoPE angles and additive mask.

    q_ang / k_ang: [Nq, head_dim/2] / [Nk, head_dim/2] from AxialRoPE.angles, or None (no rotation).
    attn_mask: [Nq, Nk] float, 0 = may attend, -inf = blocked.
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
        if attn_mask is None:
            o = F.scaled_dot_product_attention(q, k, v)
        else:                                                    # explicit math: same result on every backend
            a = (q.float() @ k.float().transpose(-2, -1)) * self.head_dim ** -0.5 + attn_mask.float()
            o = (a.softmax(-1) @ v.float()).to(v.dtype)
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
    """D tokens: predict(D_prev, C_k, pose, h) -> D_hat; correct(D_hat, X_t, C_k) -> D_t; up(D) -> width dim.
    No RoPE (mask 0000).

    One predictor serves both the in-loop prediction of D_t made before X_t arrives (h = 1, from D_{t-1})
    and the forecasts D_hat(t + h) made at t for each h in cfg.forecast_horizons. Inputs: D_prev,
    C_k of D_prev's segment, the target view's pose relative to that segment's keyframe view (posed views)
    and, when present, a horizon embedding. v1 parameters exist only when their flag is on, so with
    default flags the parameters (and the math) are the v0 ones:
      h_emb   [max_h + 1, d], zero-init, added to the input; when len(forecast_horizons) > 1 or forecast_residual
      fut_out ZeroLinear(d, d); forecast_residual: D_hat = D_prev + fut_out(pred(.)), so every
              prediction starts as copy-last (D_hat = D_prev)
    """

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        _check_v0(cfg)
        self.cfg = cfg
        self.horizons = tuple(sorted({int(h) for h in cfg.forecast_horizons}))   # YAML gives a list
        assert self.horizons and self.horizons[0] >= 1, f"forecast_horizons must be >= 1: {cfg.forecast_horizons}"
        self.residual = bool(cfg.forecast_residual)
        self.d0 = nn.Parameter(torch.randn(cfg.d_tokens, cfg.d_dim) * 0.02)
        self.pred = DBlock(cfg, n_ctx=1, self_first=True)        # self-attn over D, cross-attn D -> C_k, MLP
        self.corr = nn.ModuleList([DBlock(cfg, n_ctx=2, self_first=False) for _ in range(cfg.d_blocks)])
        self.up_proj = nn.Linear(cfg.d_dim, cfg.dim)
        # target-view pose relative to the keyframe view, an 'action' for the prediction; zero at init
        self.pose_proj = nn.Sequential(nn.Linear(12, cfg.d_dim), nn.GELU(), ZeroLinear(cfg.d_dim, cfg.d_dim))
        if len(self.horizons) > 1 or self.residual:              # index h; row 1 serves the in-loop step
            self.h_emb = nn.Parameter(torch.zeros(max(self.horizons) + 1, cfg.d_dim))
        else:
            self.register_parameter("h_emb", None)
        self.fut_out = ZeroLinear(cfg.d_dim, cfg.d_dim) if self.residual else None

    def init(self, B: int, dtype=None) -> torch.Tensor:
        D = self.d0.unsqueeze(0).expand(B, -1, -1)
        return D if dtype is None else D.to(dtype)

    def predict(self, D_prev: Optional[torch.Tensor], C_k: torch.Tensor,
                pose: Optional[torch.Tensor] = None, h=1) -> torch.Tensor:
        """D_prev [B, M, d] | None, C_k [B, N, dim] (D_prev's segment), pose [B, 12] (target view vs that
        keyframe view) | None, h: horizon in frames, int or long [B] -> D_hat [B, M, d]."""
        if D_prev is None:
            D_prev = self.init(C_k.shape[0], C_k.dtype)
        x = D_prev
        if pose is not None:
            x = x + self.pose_proj(pose.to(x.dtype)).unsqueeze(1)
        if self.h_emb is not None:
            e = self.h_emb[h].to(x.dtype)                        # [d] or [B, d]
            x = x + (e.unsqueeze(1) if e.dim() == 2 else e)
        y = self.pred(x, [C_k], include_self=False)
        return D_prev + self.fut_out(y) if self.residual else y

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
    """C at keyframes, D every frame, and the forecasts of D.

    The prediction of D_t is made before X_t arrives: from D_{t-1}, the C of frame t-1's segment (at a
    keyframe that is the previous segment's C, never the C built from X_t) and the pose of view t
    relative to that segment's keyframe view. It is the one-step forecast made at t-1, so it is computed
    once: D_hat_t = D_next_{t-1}. Forecasts for h > 1 (t + h < T) are made after the loop, all (t, h)
    pairs in one predictor call, from (D_t, C of t's segment, pose of view t + h vs t's keyframe view, h).
    """

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
        C [B, S, N, dim], seg [T] long, D / D_hat / D_next [B, T, M, d] and
        D_fut {h: [B, T - h, M, d]} (index t = forecast made at t for frame t + h; horizons h < T only),
        D_fut[1] = D_next[:, :T - 1]. D / D_hat / D_next / D_fut are None when T == 1."""
        B, T = X.shape[:2]
        K = self.cfg.keyframe_every
        S = math.ceil(T / K)
        seg = torch.arange(T, device=X.device) // K
        Cs = [self.c_builder(X[:, s * K], coords[s * K], mask[s * K]) for s in range(S)]
        C = torch.stack(Cs, 1)
        if T == 1:
            return dict(C=C, seg=seg, D=None, D_hat=None, D_next=None, D_fut=None)
        Ds, D_hats, D_nexts = [], [], []
        D_hat = self.d_state.init(B, X.dtype)
        for t in range(T):
            C_k = Cs[t // K]
            D_t = self.d_state.correct(D_hat, X[:, t], C_k)
            # forecast made at t before X_{t+1} arrives = the in-loop prediction of D_{t+1}
            D_next = self.d_state.predict(D_t, C_k, self._pose(c2w, (t // K) * K, t + 1))
            D_hats.append(D_hat)
            Ds.append(D_t)
            D_nexts.append(D_next)
            D_hat = D_next
        D, D_next = torch.stack(Ds, 1), torch.stack(D_nexts, 1)
        return dict(C=C, seg=seg, D=D, D_hat=torch.stack(D_hats, 1), D_next=D_next,
                    D_fut=self._forecasts(D, D_next, C, seg, c2w))

    def _forecasts(self, D: torch.Tensor, D_next: torch.Tensor, C: torch.Tensor, seg: torch.Tensor,
                   c2w: Optional[torch.Tensor]) -> dict:
        """{h: D_hat(t + h) made at t, [B, T - h, M, d]} for every forecast horizon h < T."""
        B, T = D.shape[:2]
        K = self.cfg.keyframe_every
        hs = [h for h in self.d_state.horizons if h < T]
        out = {1: D_next[:, :T - 1]} if 1 in hs else {}
        hs = [h for h in hs if h > 1]
        if not hs:
            return out
        src = torch.cat([torch.arange(T - h, device=D.device) for h in hs])            # [P] frame t
        hv = torch.cat([torch.full((T - h,), h, device=D.device, dtype=torch.long) for h in hs])
        pose = None
        if c2w is not None:                                       # view t + h vs t's keyframe view
            pose = relative_pose(c2w[:, (src // K) * K], c2w[:, src + hv]).flatten(0, 1)
        P = src.shape[0]
        D_f = self.d_state.predict(D[:, src].flatten(0, 1), C[:, seg[src]].flatten(0, 1), pose,
                                   hv.repeat(B)).unflatten(0, (B, P))
        out.update(zip(hs, D_f.split([T - h for h in hs], 1)))
        return out
