"""Stage 2 (partial): RGAT4D block and adjacency mask construction.

No dependency on torch_geometric — all graph ops are dense masked attention.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Adjacency mask construction                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def build_adjacency(
    positions: torch.Tensor,   # (N, 4) long
    plane_ids: torch.Tensor,   # (N,) long  (-1 for non-3D)
    modality: str,
    r_s: int = 2,
    r_t: int = 1,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Construct per-type edge masks for a single sequence of tokens.

    Returns
    -------
    adj_mask : (N, N) bool  — any edge exists
    edge_type_masks : list of 4 (N, N) bool tensors
    """
    N = positions.shape[0]
    device = positions.device

    # All pairwise coordinate deltas: (N, N, 4)
    delta = positions.float().unsqueeze(1) - positions.float().unsqueeze(0)
    dt = delta[..., 0]
    dx = delta[..., 1]
    dy = delta[..., 2]
    dz = delta[..., 3]

    # Type 0 — SPATIAL: same t and z, within r_s in x and y
    spatial = (dt == 0) & (dz == 0) & (dx.abs() <= r_s) & (dy.abs() <= r_s)

    # Type 1 — TEMPORAL: same x, y, z; within r_t in t
    temporal = (dx == 0) & (dy == 0) & (dz == 0) & (dt.abs() <= r_t)

    # Type 2 — DEPTH: reserved, unused
    depth = torch.zeros(N, N, dtype=torch.bool, device=device)

    # Type 3 — CROSS-PLANE: different plane, shares ≥ 1 coordinate value
    if modality == 'threed':
        same_plane = (plane_ids.unsqueeze(1) == plane_ids.unsqueeze(0))
        diff_plane = ~same_plane
        shares = (
            (positions[:, 1].unsqueeze(1) == positions[:, 1].unsqueeze(0)) |
            (positions[:, 2].unsqueeze(1) == positions[:, 2].unsqueeze(0)) |
            (positions[:, 3].unsqueeze(1) == positions[:, 3].unsqueeze(0))
        )
        cross_plane = diff_plane & shares
        spatial = spatial & same_plane   # restrict spatial to same plane for 3D
    else:
        cross_plane = torch.zeros(N, N, dtype=torch.bool, device=device)

    # Remove self-loops
    eye = torch.eye(N, dtype=torch.bool, device=device)
    spatial    = spatial    & ~eye
    temporal   = temporal   & ~eye
    cross_plane = cross_plane & ~eye

    edge_type_masks = [spatial, temporal, depth, cross_plane]
    adj_mask = spatial | temporal | depth | cross_plane
    return adj_mask, edge_type_masks


# --------------------------------------------------------------------------- #
#  RGAT4D block                                                                 #
# --------------------------------------------------------------------------- #

class RGAT4DBlock(nn.Module):
    """4D Relational Graph Attention Transformer block.

    Injects typed geometric edge information via per-type K and V projections.
    The output projection is zero-initialised so the block starts as identity.
    Masks are passed at forward time (precomputed and cached by the backbone).
    """

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        num_edge_types: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_edge_types)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_edge_types)])

        # Additive per-type bias in attention logit space: (E, H)
        self.edge_bias = nn.Parameter(torch.zeros(num_edge_types, num_heads))

        # Zero-init → identity at step 0
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                     # (B, N, D)
        adj_mask: torch.Tensor,              # (N, N) bool
        edge_type_masks: List[torch.Tensor], # list[4] of (N, N) bool
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        residual = x
        xn = self.norm1(x)

        Q = self.q_proj(xn).reshape(B, N, H, d).transpose(1, 2)  # (B, H, N, d)

        # Accumulate attention logits: for each edge (i,j), sum scores from
        # all active edge types. Non-active types contribute 0 (not -inf).
        attn_logits = torch.zeros(B, H, N, N, dtype=x.dtype, device=x.device)
        V_list: List[torch.Tensor | None] = []

        for etype in range(self.num_edge_types):
            mask_e = edge_type_masks[etype]   # (N, N)
            if not mask_e.any():
                V_list.append(None)
                continue

            K_e = self.k_projs[etype](xn).reshape(B, N, H, d).transpose(1, 2)
            V_e = self.v_projs[etype](xn).reshape(B, N, H, d).transpose(1, 2)
            V_list.append(V_e)

            score_e = (Q @ K_e.transpose(-2, -1)) * self.scale   # (B, H, N, N)
            score_e = score_e + self.edge_bias[etype].view(1, H, 1, 1)

            # Add score only where this edge type is active
            m = mask_e.view(1, 1, N, N).float()
            attn_logits = attn_logits + score_e * m

        # Mask disconnected pairs → -inf
        attn_logits = attn_logits.masked_fill(
            ~adj_mask.view(1, 1, N, N), float('-inf')
        )
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # isolated → 0

        # Aggregate values per edge type weighted by attention
        V_out = torch.zeros(B, H, N, d, dtype=x.dtype, device=x.device)
        for etype in range(self.num_edge_types):
            if V_list[etype] is None:
                continue
            mask_e = edge_type_masks[etype].view(1, 1, N, N).float()
            type_weights = attn_weights * mask_e    # (B, H, N, N)
            V_out = V_out + type_weights @ V_list[etype]

        out = V_out.transpose(1, 2).reshape(B, N, D)
        x = residual + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x
gitig