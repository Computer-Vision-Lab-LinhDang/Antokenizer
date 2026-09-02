"""Sparse relational graph attention on the 4D token lattice.

Replaces the dense (N, N) masked formulation of RGAT4DBlock with attention over
an explicit neighbour set per token:
  * cost O(N·K·D) instead of O(N²·D)  (K ≈ 22 for images/video, ≈ 50 for 3D)
  * every edge carries a TYPE (spatial / temporal / depth / cross-plane) and a
    clipped 4-D relative offset (Δt, Δx, Δy, Δz); both enter the attention
    logit as learned per-head biases (Graphormer / Swin-style).
  * edge sets are geometric: spatial windows are measured in each plane's own
    (row, col) frame, and cross-plane edges follow the shared projection axis.
The block is a zero-initialised residual adapter like RGAT4DBlock and can be
loaded from one (per_type_kv=True) for exact equivalence on the same edge set.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mavt.model.rgat import RGAT4DBlock, build_adjacency

E_SPATIAL, E_TEMPORAL, E_DEPTH, E_CROSS = 0, 1, 2, 3
NUM_EDGE_TYPES = 4


@dataclass
class GraphStructure:
    """Padded neighbour lists for one token sequence (shared across the batch)."""
    nbr_idx: torch.Tensor    # (N, K) long — neighbour token index (0 on pads)
    nbr_type: torch.Tensor   # (N, K) long — edge type (0 on pads; check `valid`)
    nbr_rel: torch.Tensor    # (N, K, 4) long — clipped offsets shifted to [0, 2R_axis]
    valid: torch.Tensor      # (N, K) bool
    r_s: int = 2
    r_t: int = 1

    def to(self, device) -> "GraphStructure":
        return GraphStructure(self.nbr_idx.to(device), self.nbr_type.to(device),
                              self.nbr_rel.to(device), self.valid.to(device), self.r_s, self.r_t)

    @property
    def num_tokens(self) -> int:
        return self.nbr_idx.shape[0]

    @property
    def max_degree(self) -> int:
        return self.nbr_idx.shape[1]


def _plane_frame(positions: torch.Tensor, plane_ids: torch.Tensor):
    """(row, col) of each token in its own plane's 2-D frame.
    image/video/XY (plane -1 or 0): (x, y); XZ (1): (x, z); YZ (2): (y, z)."""
    x, y, z = positions[:, 1], positions[:, 2], positions[:, 3]
    row = torch.where(plane_ids == 2, y, x)
    col = torch.where(plane_ids <= 0, y, z)
    return row, col


@torch.no_grad()
def _dense_masks(positions, plane_ids, modality, r_s, r_t,
                 plane_local_spatial, cross_mode) -> List[torch.Tensor]:
    if not plane_local_spatial and cross_mode == "shared_axis":
        _, masks = build_adjacency(positions, plane_ids, modality, r_s, r_t)
        return list(masks)
    N = positions.shape[0]
    dev = positions.device
    p = positions.float()
    d = p.unsqueeze(1) - p.unsqueeze(0)
    dt, dx, dy, dz = d[..., 0], d[..., 1], d[..., 2], d[..., 3]
    same_plane = plane_ids.unsqueeze(1) == plane_ids.unsqueeze(0)
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    if plane_local_spatial:
        row, col = _plane_frame(positions, plane_ids)
        drow = (row.unsqueeze(1) - row.unsqueeze(0)).abs()
        dcol = (col.unsqueeze(1) - col.unsqueeze(0)).abs()
        spatial = same_plane & (dt == 0) & (drow <= r_s) & (dcol <= r_s)
    else:
        spatial = same_plane & (dt == 0) & (dz == 0) & (dx.abs() <= r_s) & (dy.abs() <= r_s)
    temporal = (dx == 0) & (dy == 0) & (dz == 0) & same_plane & (dt.abs() <= r_t)
    depth = torch.zeros(N, N, dtype=torch.bool, device=dev)
    if modality == "threed":
        if cross_mode == "projection":
            pid = plane_ids
            def pair(a, b, axis):
                m = ((pid.unsqueeze(1) == a) & (pid.unsqueeze(0) == b)) | \
                    ((pid.unsqueeze(1) == b) & (pid.unsqueeze(0) == a))
                same = positions[:, axis].unsqueeze(1) == positions[:, axis].unsqueeze(0)
                return m & same
            cross = pair(0, 1, 1) | pair(0, 2, 2) | pair(1, 2, 3)
        else:
            shares = ((positions[:, 1].unsqueeze(1) == positions[:, 1].unsqueeze(0)) |
                      (positions[:, 2].unsqueeze(1) == positions[:, 2].unsqueeze(0)) |
                      (positions[:, 3].unsqueeze(1) == positions[:, 3].unsqueeze(0)))
            cross = (~same_plane) & shares
    else:
        cross = torch.zeros(N, N, dtype=torch.bool, device=dev)
    return [spatial & ~eye, temporal & ~eye, depth, cross & ~eye]


@torch.no_grad()
def graph_from_dense(masks: List[torch.Tensor], positions: Optional[torch.Tensor] = None,
                     r_s: int = 2, r_t: int = 1) -> GraphStructure:
    """Convert per-type (N, N) boolean masks into padded neighbour lists."""
    N = masks[0].shape[0]
    dev = masks[0].device
    etype = torch.full((N, N), -1, dtype=torch.long, device=dev)
    for e, m in enumerate(masks):
        etype = torch.where(m & (etype < 0), torch.full_like(etype, e), etype)
    has = etype >= 0
    deg = has.sum(1)
    K = max(int(deg.max().item()) if N > 0 else 0, 1)
    order = torch.argsort((~has).to(torch.int8), dim=1, stable=True)[:, :K]
    valid = torch.gather(has, 1, order)
    nbr_idx = torch.where(valid, order, torch.zeros_like(order))
    nbr_type = torch.where(valid, torch.gather(etype, 1, order), torch.zeros_like(order))
    if positions is None:
        rel = torch.zeros(N, K, 4, dtype=torch.long, device=dev)
        rel[..., 0] = r_t; rel[..., 1:] = r_s
    else:
        off = positions[nbr_idx] - positions.unsqueeze(1)
        lim = torch.tensor([r_t, r_s, r_s, r_s], device=dev)
        rel = (off.clamp(-lim, lim) + lim).long()
    return GraphStructure(nbr_idx, nbr_type, rel, valid, r_s, r_t)


@torch.no_grad()
def build_graph(positions: torch.Tensor, plane_ids: torch.Tensor, modality: str,
                r_s: int = 2, r_t: int = 1, plane_local_spatial: bool = False,
                cross_mode: str = "shared_axis") -> GraphStructure:
    masks = _dense_masks(positions, plane_ids, modality, r_s, r_t, plane_local_spatial, cross_mode)
    return graph_from_dense(masks, positions, r_s, r_t)


class SparseRGAT4D(nn.Module):
    """Relational graph attention over gathered neighbour sets.
    logit(i→j) = q_i·k_j/√d + b_type[e_ij] + b_t[Δt] + b_x[Δx] + b_y[Δy] + b_z[Δz]
    softmax over j ∈ N(i); out_i = Σ α_ij v_j."""

    def __init__(self, dim: int = 1152, num_heads: int = 16, num_edge_types: int = NUM_EDGE_TYPES,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, per_type_kv: bool = False,
                 r_s: int = 2, r_t: int = 1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_edge_types = num_edge_types
        self.per_type_kv = per_type_kv
        self.r_s, self.r_t = r_s, r_t
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        if per_type_kv:
            self.k_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_edge_types)])
            self.v_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_edge_types)])
        else:
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(num_edge_types, num_heads))
        self.rel_t = nn.Parameter(torch.zeros(2 * r_t + 1, num_heads))
        self.rel_x = nn.Parameter(torch.zeros(2 * r_s + 1, num_heads))
        self.rel_y = nn.Parameter(torch.zeros(2 * r_s + 1, num_heads))
        self.rel_z = nn.Parameter(torch.zeros(2 * r_s + 1, num_heads))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    @torch.no_grad()
    def load_from_dense(self, dense: RGAT4DBlock) -> None:
        assert self.per_type_kv, "load_from_dense needs per_type_kv=True"
        self.norm1.load_state_dict(dense.norm1.state_dict())
        self.norm2.load_state_dict(dense.norm2.state_dict())
        self.q_proj.load_state_dict(dense.q_proj.state_dict())
        for e in range(self.num_edge_types):
            self.k_projs[e].load_state_dict(dense.k_projs[e].state_dict())
            self.v_projs[e].load_state_dict(dense.v_projs[e].state_dict())
        self.edge_bias.copy_(dense.edge_bias)
        self.out_proj.load_state_dict(dense.out_proj.state_dict())
        self.mlp.load_state_dict(dense.mlp.state_dict())

    def _gather_kv(self, xn: torch.Tensor, g: GraphStructure):
        B, N, D = xn.shape
        H, d = self.num_heads, self.head_dim
        if self.per_type_kv:
            K_all = torch.stack([p(xn) for p in self.k_projs], 1).view(B, self.num_edge_types, N, H, d)
            V_all = torch.stack([p(xn) for p in self.v_projs], 1).view(B, self.num_edge_types, N, H, d)
            kg = K_all[:, g.nbr_type, g.nbr_idx]
            vg = V_all[:, g.nbr_type, g.nbr_idx]
        else:
            Kt = self.k_proj(xn).view(B, N, H, d)
            Vt = self.v_proj(xn).view(B, N, H, d)
            kg = Kt[:, g.nbr_idx]
            vg = Vt[:, g.nbr_idx]
        return kg, vg

    def _relation_bias(self, g: GraphStructure) -> torch.Tensor:
        b = self.edge_bias[g.nbr_type]
        return (b + self.rel_t[g.nbr_rel[..., 0]] + self.rel_x[g.nbr_rel[..., 1]]
                  + self.rel_y[g.nbr_rel[..., 2]] + self.rel_z[g.nbr_rel[..., 3]])

    def forward(self, x: torch.Tensor, g: GraphStructure) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        g = g.to(x.device)
        xn = self.norm1(x)
        q = self.q_proj(xn).view(B, N, H, d)
        kg, vg = self._gather_kv(xn, g)
        logits = (q.unsqueeze(2) * kg).sum(-1) * self.scale
        logits = logits + self._relation_bias(g).unsqueeze(0).to(logits.dtype)
        logits = logits.masked_fill(~g.valid.view(1, N, -1, 1), float("-inf"))
        attn = torch.nan_to_num(torch.softmax(logits, dim=2), nan=0.0)
        out = (attn.unsqueeze(-1) * vg).sum(2).reshape(B, N, D)
        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


__all__ = ["GraphStructure", "SparseRGAT4D", "build_graph", "graph_from_dense",
           "E_SPATIAL", "E_TEMPORAL", "E_DEPTH", "E_CROSS"]
