"""Tier B: trunk. R0 = plain attention (default); R2 = learned relations (cfg.rgat_relations > 0).

Each layer, a token of frame t attends to all N tokens of frame t plus the tokens of frame t-1
in the (2r+1)^2 grid window around its own cell. Causal and parallel over T. RoPE uses each key's
own frame coords, so Delta t is visible. T = 1 is plain full attention.

R0 runs as two 4-D SDPA calls (4-D so that the fused kernels apply: SDPA sends 5-D inputs to its
math path on every backend, as v0 did), the same function as v0's one masked call over [x_t ; x_{t-1}]:
  frame 0 (all of an image): its own N tokens only, no mask (v0 masked out a zero frame -1);
  frames 1..T-1, batch B(T-1): keys [x_{t-1} ; x_t] and ONE additive float mask [N, 2N] (window on the
                             x_{t-1} half), the same for every frame: cached, broadcast over batch and heads.
q / k are rotated once per layer (frame t-1's keys reuse their own rotation) from cos / sin tables
built once per forward (rope_tables). Why this split: an exact log-sum-exp merge of in-frame + window
attention needs a differentiable LSE, which SDPA does not return; one 4-D call over all frames needs
the per-frame mask materialised over the batch ([B T, 1, N, 2N], read once per head); on CPU (fp32,
fwd + bwd, T = 8, N = 256) the split ran 1.6-3.2x faster than v0's 5-D call, on par with that
one-call variant, and an unfold view of the keys (no cat copy) was not faster.

R2 (rgat_relations = R > 0): position stays with RoPE; each edge (i, j) also picks a relation
r_ij in {1..R} from MLP([a_i * b_j ; |a_i - b_j|]) with a = LN(Linear_q(h)), b = LN(Linear_k(h))
(d -> 64 per side, so the pair MLP is 128 -> 256 -> R). r is a straight-through Gumbel top-1 in
training (argmax, no noise, in eval). The edge carries v_j + U_r V_r^T v_j (per head, rank
rgat_rank) and adds b_{r,h} + b_{type,h} (type = same frame / frame t-1) to its logit. U and the
biases start at zero, so R2 at init equals R0. This path uses explicit attention [B, T, H, N, 2N]
over keys [x_t ; x_{t-1}] (frame -1 = zeros, masked by the bool mask of attn_mask).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DinoV0Config
from .rope import AxialRoPE, apply_rope, compute_dtype, rope_tables


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
    """forward(x [B,T,N,D], cs, mask): cs = rope (cos, sin) tables [T, 1, N, hd] (Trunk builds them once);
    mask = R0: float bias [N, 2N] for frames 1..T-1 (None when T == 1); R2: bool [T, 1, N, 2N]."""

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

    def _qkv(self, x: torch.Tensor, cs: tuple) -> tuple:
        """q, k (rotated, each frame by its own coords) and v, each [B, T, H, N, hd] (token-major views)."""
        B, T, N, D = x.shape
        qkv = self.qkv(x).view(B, T, N, 3, self.h, self.hd).permute(3, 0, 1, 4, 2, 5)  # [3,B,T,H,N,hd]
        qk, v = qkv.split([2, 1])                                          # split / unbind: one cat in backward
        q, k = apply_rope(qk, *cs).to(qk.dtype).unbind(0)                  # q and k in one pass
        return q, k, v.squeeze(0)

    def attn(self, x: torch.Tensor, cs: tuple, bias) -> torch.Tensor:
        B, T, N, D = x.shape
        q, k, v = self._qkv(x, cs)
        # frame 0 (all of an image): its own frame only, no mask
        o = F.scaled_dot_product_attention(q[:, 0], k[:, 0], v[:, 0]).transpose(1, 2).unsqueeze(1)   # [B,1,N,H,hd]
        if T > 1:                                                          # frames 1..T-1: keys [x_{t-1} ; x_t]
            kk = torch.cat([k[:, :-1], k[:, 1:]], 3).flatten(0, 1)        # [B(T-1),H,2N,hd]
            vv = torch.cat([v[:, :-1], v[:, 1:]], 3).flatten(0, 1)
            o1 = F.scaled_dot_product_attention(q[:, 1:].flatten(0, 1), kk, vv, attn_mask=bias)
            o = torch.cat([o, o1.unflatten(0, (B, T - 1)).transpose(2, 3)], 1)   # [B,T,N,H,hd]
        return self.out(o.reshape(B, T, N, D))

    def _qkv_rel(self, x: torch.Tensor, cs: tuple) -> tuple:
        """q [B,T,H,N,hd]; keys / values [x_t ; x_{t-1}] [B,T,H,2N,hd] with frame -1 = zeros (v0 layout).
        k of frame t-1 rotates with its own coords either way, so it is rotated once and then shifted."""
        q, k, v = self._qkv(x, cs)
        return q, torch.cat([k, _shift(k, 1)], 3), torch.cat([v, _shift(v, 1)], 3)

    def attn_rel(self, x: torch.Tensor, cs: tuple, mask: torch.Tensor) -> tuple:
        """R2 attention. Returns (out [B,T,N,D], stats [3 + R]: marginal H (with grad), edge H, usage)."""
        q, k, v = self._qkv_rel(x, cs)                                     # q [B,T,H,N,hd], k/v [B,T,H,2N,hd]
        return self._rel_core(x, q, k, v, mask)

    def _rel_core(self, x, q, k, v, mask) -> tuple:
        B, T, N, D = x.shape
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

    def forward(self, x, cs, mask):
        if self.R > 0:
            a, stats = self.attn_rel(self.norm1(x), cs, mask)
            x = x + a
            return x + self.mlp(self.norm2(x)), stats
        x = x + self.attn(self.norm1(x), cs, mask)
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
        self._masks = {}                                                   # (kind, T, N, device, dtype) -> mask

    def relation_stats(self) -> dict | None:
        """R2 stats of the last forward, averaged over layers: marginal_entropy (with grad, for the
        anti-collapse loss), edge_entropy (mean per-edge entropy), usage (relation histogram [R])."""
        if self._rel_stats is None:
            return None
        st = torch.stack(self._rel_stats)                                  # [L, 3 + R]
        return dict(marginal_entropy=st[:, 0].mean(), edge_entropy=st[:, 1].detach().mean(),
                    usage=st[:, 3:].detach().mean(0))

    def attn_mask(self, T: int, N: int, device=None) -> torch.Tensor:
        """[T, 1, N, 2N] bool over keys [x_t ; x_{t-1}] (v0 layout; the R2 path and the reference in tests)."""
        grid = int(round(N ** 0.5))
        assert grid * grid == N, "tokens must form a square grid"
        prev = window_mask(grid, self.r, device).expand(T, N, N).clone()
        prev[0] = False                                                    # frame 0 has no past
        cur = torch.ones(T, N, N, dtype=torch.bool, device=device)
        return torch.cat([cur, prev], -1).unsqueeze(1)                      # [T, 1, N, 2N]

    def prev_bias(self, N: int, device=None, dtype=torch.float32) -> torch.Tensor:
        """[N, 2N] additive mask over keys [x_{t-1} ; x_t]: -inf outside the window on the x_{t-1} half, else 0
        (float, not bool: bool masks gave zero gradients on some SDPA backends)."""
        grid = int(round(N ** 0.5))
        assert grid * grid == N, "tokens must form a square grid"
        ok = torch.cat([window_mask(grid, self.r, device), torch.ones(N, N, dtype=torch.bool, device=device)], -1)
        return torch.zeros(N, 2 * N, device=device, dtype=dtype).masked_fill(~ok, float("-inf"))

    def _cached(self, kind: str, T: int, N: int, device, dtype) -> torch.Tensor:
        key = (kind, T, N, torch.device(device), dtype)
        m = self._masks.get(key)
        if m is None:
            m = self.attn_mask(T, N, device) if kind == "bool" else self.prev_bias(N, device, dtype)
            self._masks[key] = m
        return m

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        dt = compute_dtype(x)
        ang = self.layers[0].rope.angles(coords, mask)                     # [T, N, hd/2] float32, same for all layers
        cs = rope_tables(ang.unsqueeze(1), dt)                             # cos / sin [T, 1, N, hd], once per forward
        if self.rgat:
            am = self._cached("bool", T, N, x.device, torch.bool)
        else:
            am = self._cached("bias", 0, N, x.device, dt) if T > 1 else None
        stats = []
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                y = checkpoint(layer, x, cs, am, use_reentrant=False)
            else:
                y = layer(x, cs, am)
            if self.rgat:
                x, st = y
                stats.append(st)
            else:
                x = y
        self._rel_stats = stats if self.rgat else None
        return self.norm(x)
