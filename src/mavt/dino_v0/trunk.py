"""Tier B: trunk. R0 = plain attention (default); R2 = learned relations (cfg.rgat_relations > 0).

Each layer, a token of frame t attends to all N tokens of frame t plus the tokens of frame t-1
in the (2r+1)^2 grid window around its own cell. Causal and parallel over T: keys are
[x_t ; x_{t-1}] (frame -1 = zeros, fully masked) with one boolean mask [N, 2N] built from the grid.
RoPE uses each key's own frame coords, so Delta t is visible. T = 1 is plain full attention.

R2 (rgat_relations = R > 0): position stays with RoPE; each edge (i, j) also picks a relation
r_ij in {1..R} from MLP([a_i * b_j ; |a_i - b_j|]) with a = LN(Linear_q(h)), b = LN(Linear_k(h))
(d -> 64 per side, so the pair MLP is 128 -> 256 -> R). r is a straight-through Gumbel top-1 in
training (argmax, no noise, in eval). The edge carries v_j + U_r V_r^T v_j (per head, rank
rgat_rank) and adds b_{r,h} + b_{type,h} (type = same frame / frame t-1) to its logit. U and the
biases start at zero, so R2 at init equals R0. This path uses explicit attention [B, T, H, N, 2N].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DinoV0Config
from .rope import AxialRoPE


def window_mask(grid: int, r: int, device=None) -> torch.Tensor:
    """[N, N] bool: query cell i may read previous-frame cell j (Chebyshev distance <= r)."""
    idx = torch.arange(grid * grid, device=device)
    row, col = idx // grid, idx % grid
    return ((row[:, None] - row[None]).abs() <= r) & ((col[:, None] - col[None]).abs() <= r)


def _shift(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Shift by one along dim (frame t gets frame t-1); frame 0 gets zeros."""
    pad = torch.zeros_like(x.narrow(dim, 0, 1))
    return torch.cat([pad, x.narrow(dim, 0, x.shape[dim] - 1)], dim)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


class RelationHead(nn.Module):
    """Per-edge relation logits [B, T, N, 2N, R] from the layer input h [B, T, N, d]."""

    def __init__(self, d: int, R: int, proj: int = 64, hidden: int = 256):
        super().__init__()
        self.pq, self.pk = nn.Linear(d, proj), nn.Linear(d, proj)
        self.nq, self.nk = nn.LayerNorm(proj), nn.LayerNorm(proj)
        self.mlp = nn.Sequential(nn.Linear(2 * proj, hidden), nn.GELU(), nn.Linear(hidden, R))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.nq(self.pq(h))                                            # [B,T,N,p]
        b = self.nk(self.pk(h))
        b = torch.cat([b, _shift(b, 1)], 2)                                # keys [x_t ; x_{t-1}]: [B,T,2N,p]
        a, b = a.unsqueeze(3), b.unsqueeze(2)                              # [B,T,N,1,p], [B,T,1,2N,p]
        return self.mlp(torch.cat([(a * b), (a - b).abs()], -1))           # [B,T,N,2N,R]


class TrunkLayer(nn.Module):
    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        d, self.h = cfg.dim, cfg.heads
        self.hd = d // cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.rope = AxialRoPE(self.hd)
        self.R, self.tau = cfg.rgat_relations, cfg.rgat_tau
        if self.R > 0:                                                     # R2; created after the R0 params
            R, H, hd, rk = self.R, self.h, self.hd, cfg.rgat_rank
            self.rel = RelationHead(d, R)
            self.rel_V = nn.Parameter(torch.randn(R, H, hd, rk) * hd ** -0.5)
            self.rel_U = nn.Parameter(torch.zeros(R, H, rk, hd))           # zero: starts equal to R0
            self.rel_bias = nn.Parameter(torch.zeros(R, H))                # b_{r,h}
            self.type_bias = nn.Parameter(torch.zeros(2, H))               # b_{type,h}: same frame / t-1

    def _qkv(self, x, ang, ang_prev):
        B, T, N, D = x.shape
        q, k, v = self.qkv(x).view(B, T, N, 3, self.h, self.hd).permute(3, 0, 1, 4, 2, 5)  # [B,T,H,N,hd]
        rot = lambda z, a: self.rope.rotate(z.reshape(B * T, self.h, N, self.hd),
                                            a.repeat(B, 1, 1)).view(B, T, self.h, N, self.hd)
        q, k_cur, k_prev = rot(q, ang), rot(k, ang), rot(_shift(k, 1), ang_prev)
        k = torch.cat([k_cur, k_prev], 3)
        v = torch.cat([v, _shift(v, 1)], 3)
        return q, k, v

    def attn(self, x: torch.Tensor, ang: torch.Tensor, ang_prev: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        q, k, v = self._qkv(x, ang, ang_prev)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)       # mask [T,1,N,2N]
        return self.out(o.transpose(2, 3).reshape(B, T, N, D))

    def attn_rel(self, x: torch.Tensor, ang: torch.Tensor, ang_prev: torch.Tensor,
                 mask: torch.Tensor) -> tuple:
        """R2 attention. Returns (out [B,T,N,D], stats [3 + R]: marginal H (with grad), edge H, usage)."""
        B, T, N, D = x.shape
        q, k, v = self._qkv(x, ang, ang_prev)                              # q [B,T,H,N,hd], k/v [B,T,H,2N,hd]
        logits = self.rel(x).float()                                       # [B,T,N,2N,R]
        probs = (logits / self.tau).softmax(-1)
        idx = logits.argmax(-1)
        if self.training:
            g = -torch.empty_like(logits).exponential_().log()
            soft = ((logits + g) / self.tau).softmax(-1)
            idx = soft.argmax(-1)
        else:
            soft = probs
        hard = F.one_hot(idx, self.R).to(soft.dtype)
        r = hard - soft.detach() + soft if self.training else hard        # straight-through top-1
        r = r.to(q.dtype)

        valid = mask.squeeze(1)                                            # [T,N,2N]
        s = (q @ k.transpose(-1, -2)) * self.hd ** -0.5                    # [B,T,H,N,2N]
        s = s + torch.einsum("btijr,rh->bthij", r, self.rel_bias.to(r.dtype))
        tb = self.type_bias.to(s.dtype)
        s = s + torch.cat([tb[0].view(1, 1, -1, 1, 1).expand(-1, -1, -1, N, N),
                           tb[1].view(1, 1, -1, 1, 1).expand(-1, -1, -1, N, N)], -1)
        s = s.masked_fill(~valid[None, :, None], float("-inf"))
        a = s.softmax(-1)                                                  # [B,T,H,N,2N]
        # sum_j a_ij U_r(ij) V_r(ij)^T v_j, contracted in rank space: [B,T,H,2N,R,rk] -> [B,T,H,N,R,rk]
        vv = torch.einsum("bthjd,rhdk->bthjrk", v, self.rel_V.to(v.dtype))
        z = torch.einsum("bthij,btijr,bthjrk->bthirk", a, r, vv)
        o = a @ v + torch.einsum("bthirk,rhke->bthie", z, self.rel_U.to(v.dtype))
        out = self.out(o.transpose(2, 3).reshape(B, T, N, D))

        w = valid.to(probs.dtype)[None, ..., None]                         # [1,T,N,2N,1]
        cnt = w.sum() * B
        marg = (probs * w).sum((0, 1, 2, 3)) / cnt
        with torch.no_grad():
            edge_h = (_entropy(probs) * w[..., 0]).sum() / cnt
            usage = (hard * w).sum((0, 1, 2, 3)) / cnt
        stats = torch.cat([_entropy(marg).view(1), edge_h.view(1), cnt.view(1).detach(), usage])
        return out, stats

    def forward(self, x, ang, ang_prev, mask):
        if self.R > 0:
            a, stats = self.attn_rel(self.norm1(x), ang, ang_prev, mask)
            x = x + a
            return x + self.mlp(self.norm2(x)), stats
        x = x + self.attn(self.norm1(x), ang, ang_prev, mask)
        return x + self.mlp(self.norm2(x))


class Trunk(nn.Module):
    """x [B, T, N, dim], coords [T, N, 4], mask [T, N, 4] -> [B, T, N, dim]."""

    def __init__(self, cfg: DinoV0Config, use_checkpoint: bool = False):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.r, self.use_checkpoint = cfg.temporal_window, use_checkpoint
        self.layers = nn.ModuleList(TrunkLayer(cfg) for _ in range(cfg.trunk_layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.rgat = cfg.rgat_relations > 0
        self._rel_stats = None

    def relation_stats(self) -> dict | None:
        """R2 stats of the last forward, averaged over layers: marginal_entropy (with grad, for the
        anti-collapse loss), edge_entropy (mean per-edge entropy), usage (relation histogram [R])."""
        if self._rel_stats is None:
            return None
        st = torch.stack(self._rel_stats)                                  # [L, 3 + R]
        return dict(marginal_entropy=st[:, 0].mean(), edge_entropy=st[:, 1].detach().mean(),
                    usage=st[:, 3:].detach().mean(0))

    def attn_mask(self, T: int, N: int, device=None) -> torch.Tensor:
        grid = int(round(N ** 0.5))
        assert grid * grid == N, "tokens must form a square grid"
        prev = window_mask(grid, self.r, device).expand(T, N, N).clone()
        prev[0] = False                                                    # frame 0 has no past
        cur = torch.ones(T, N, N, dtype=torch.bool, device=device)
        return torch.cat([cur, prev], -1).unsqueeze(1)                      # [T, 1, N, 2N]

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        ang = self.layers[0].rope.angles(coords, mask)                     # [T, N, hd/2], same for all layers
        ang_prev = _shift(ang, 0)
        am = self.attn_mask(T, N, x.device)
        stats = []
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                y = checkpoint(layer, x, ang, ang_prev, am, use_reentrant=False)
            else:
                y = layer(x, ang, ang_prev, am)
            if self.rgat:
                x, st = y
                stats.append(st)
            else:
                x = y
        self._rel_stats = stats if self.rgat else None
        return self.norm(x)
