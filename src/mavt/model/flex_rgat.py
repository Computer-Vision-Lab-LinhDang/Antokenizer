"""FlexRGAT4D — SparseRGAT4D maths on PyTorch FlexAttention block-sparse kernels.
Measured on MI325X (ROCm 7.0), N=2048, batch 8: 22 ms / 0.57 GB versus
60 ms / 15.6 GB for the dense RGAT4DBlock. Falls back to the gather path on CPU."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from mavt.model.graph_attention import GraphStructure, SparseRGAT4D, _dense_masks, graph_from_dense

_FLEX = None
try:  # a few distinct graphs (image/video/3D layouts) each compile once
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
except Exception:  # noqa: BLE001
    pass


def _compiled_flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


@dataclass
class FlexGraph:
    masks: List[torch.Tensor]
    positions: torch.Tensor
    r_s: int = 2
    r_t: int = 1
    _adj: Optional[torch.Tensor] = None
    _etype: Optional[torch.Tensor] = None
    _block_masks: Dict[str, object] = field(default_factory=dict)
    _sparse: Optional[GraphStructure] = None

    @property
    def num_tokens(self) -> int:
        return self.positions.shape[0]

    def adj(self) -> torch.Tensor:
        if self._adj is None:
            a = self.masks[0].clone()
            for m in self.masks[1:]:
                a |= m
            self._adj = a
        return self._adj

    def etype(self) -> torch.Tensor:
        if self._etype is None:
            N = self.num_tokens
            et = torch.zeros(N, N, dtype=torch.long, device=self.masks[0].device)
            taken = torch.zeros(N, N, dtype=torch.bool, device=self.masks[0].device)
            for e, m in enumerate(self.masks):
                new = m & ~taken
                et[new] = e
                taken |= m
            self._etype = et
        return self._etype

    def block_mask(self, device: torch.device):
        key = str(device)
        if key not in self._block_masks:
            from torch.nn.attention.flex_attention import create_block_mask
            adj = self.adj().to(device)
            N = self.num_tokens
            def mask_mod(b, h, q, k):
                return adj[q, k]
            self._block_masks[key] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device)
        return self._block_masks[key]

    def sparse(self) -> GraphStructure:
        if self._sparse is None:
            self._sparse = graph_from_dense(self.masks, self.positions, self.r_s, self.r_t)
        return self._sparse


@torch.no_grad()
def build_flex_graph(positions, plane_ids, modality, r_s=2, r_t=1, plane_local_spatial=False,
                     cross_mode="shared_axis") -> FlexGraph:
    masks = _dense_masks(positions, plane_ids, modality, r_s, r_t, plane_local_spatial, cross_mode)
    return FlexGraph(list(masks), positions.long(), r_s, r_t)


class FlexRGAT4D(SparseRGAT4D):
    """Graph attention block executed with FlexAttention (shared K/V)."""

    def __init__(self, dim=1152, num_heads=16, mlp_ratio=4.0, dropout=0.0, r_s=2, r_t=1):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                         per_type_kv=False, r_s=r_s, r_t=r_t)
        self._score_mods: dict = {}          # (id(graph), device) -> closure (not a submodule)

    def _score_mod_for(self, g, dev):
        """One score_mod closure per (graph, device), reused across forward calls.

        torch.compile guards on the score_mod function object: a fresh closure
        every forward would trigger a recompile per step and, past dynamo's
        cache limit, silently fall back to the eager (very slow) path.
        Captured tensors are the module's own Parameters (stable identity) and
        the graph's cached tables, so one compile per graph is enough.
        """
        key = (id(g), str(dev))
        sm = self._score_mods.get(key)
        if sm is not None:
            return sm
        etype = g.etype().to(dev)
        pos = g.positions.to(dev)
        pt, px, py, pz = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
        eb, rt, rx, ry, rz = self.edge_bias, self.rel_t, self.rel_x, self.rel_y, self.rel_z
        Rt, Rs = self.r_t, self.r_s

        def score_mod(score, b, h, qi, ki):
            e = etype[qi, ki]
            dt = torch.clamp(pt[ki] - pt[qi], -Rt, Rt) + Rt
            dx = torch.clamp(px[ki] - px[qi], -Rs, Rs) + Rs
            dy = torch.clamp(py[ki] - py[qi], -Rs, Rs) + Rs
            dz = torch.clamp(pz[ki] - pz[qi], -Rs, Rs) + Rs
            return score + eb[e, h] + rt[dt, h] + rx[dx, h] + ry[dy, h] + rz[dz, h]

        self._score_mods[key] = score_mod
        return score_mod

    def forward(self, x: torch.Tensor, g) -> torch.Tensor:
        if isinstance(g, GraphStructure):
            return super().forward(x, g)
        if not x.is_cuda:
            return super().forward(x, g.sparse())
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        dev = x.device
        xn = self.norm1(x)
        q = self.q_proj(xn).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(xn).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(xn).view(B, N, H, d).transpose(1, 2)
        score_mod = self._score_mod_for(g, dev)
        out = _compiled_flex()(q, k, v, score_mod=score_mod, block_mask=g.block_mask(dev), scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, D)
        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


__all__ = ["FlexGraph", "FlexRGAT4D", "build_flex_graph"]
