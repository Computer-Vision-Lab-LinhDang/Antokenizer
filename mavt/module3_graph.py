"""Module 3: Frequency-Informed 4D Graph Construction.

Builds a sparse k-NN graph where edge weights combine:
  w_ij = α·sim_spatial + β·sim_freq + γ·sim_feature
  (α, β, γ learnable via softmax; initialized ≈ 0.3, 0.4, 0.3)

4D spatial distance uses per-dimension learnable weights [t, x, y, z].
Graph is symmetrized (undirected) after top-K sparsification.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GraphConfig
from .types import GraphOutput, PatchifyOutput, STFOutput


class FrequencyInformed4DGraphBuilder(nn.Module):
    """Build sparse k-NN graph informed by frequency profiles.

    Learnable parameters:
        log_alpha, log_beta, log_gamma: edge weight balance (via softmax)
        w_4d:    per-dimension spatial distance weights [t, x, y, z]
        log_sigma: Gaussian RBF bandwidth
    """

    def __init__(self, cfg: Optional[GraphConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or GraphConfig()

        # Edge weight balance: softmax([log_alpha, log_beta, log_gamma]) ≈ [0.3, 0.4, 0.3]
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.3)))
        self.log_beta  = nn.Parameter(torch.tensor(math.log(0.4)))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(0.3)))

        # Per-dimension spatial distance weights [t, x, y, z]
        self.w_4d = nn.Parameter(torch.tensor([0.3, 1.0, 1.0, 0.3]))
        # Gaussian RBF bandwidth
        self.log_sigma = nn.Parameter(torch.tensor(math.log(self.cfg.sigma_init)))

    @property
    def edge_balance(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return softmax-normalized (alpha, beta, gamma)."""
        w = F.softmax(
            torch.stack([self.log_alpha, self.log_beta, self.log_gamma]), dim=0
        )
        return w[0], w[1], w[2]

    def compute_affinity(
        self,
        f_spatial: torch.Tensor,   # (B, N, D)
        freq_raw: torch.Tensor,    # (B, N, 15)
        positions: torch.Tensor,   # (B, N, 4)
    ) -> torch.Tensor:             # (B, N, N)
        """Compute combined pairwise affinity matrix."""
        alpha, beta, gamma = self.edge_balance

        # Spatial similarity: weighted Gaussian RBF
        w = F.softplus(self.w_4d).to(positions.dtype)
        pos_w = positions.float() * w                        # (B, N, 4)
        dist = torch.cdist(pos_w, pos_w, p=2)               # (B, N, N)
        sigma = F.softplus(self.log_sigma)
        sim_spatial = torch.exp(-dist.pow(2) / (2 * sigma ** 2))

        # Frequency cosine similarity
        fn = F.normalize(freq_raw.float(), dim=-1, eps=1e-8)
        sim_freq = (torch.bmm(fn, fn.transpose(1, 2)) + 1) * 0.5   # [0, 1]

        # Feature cosine similarity
        ff = F.normalize(f_spatial.float(), dim=-1, eps=1e-8)
        sim_feat = (torch.bmm(ff, ff.transpose(1, 2)) + 1) * 0.5

        return alpha * sim_spatial + beta * sim_freq + gamma * sim_feat

    def forward(
        self,
        patch_out: PatchifyOutput,
        stf_out: STFOutput,
    ) -> GraphOutput:
        """Build sparse k-NN graph.

        Returns GraphOutput with adj (B,N,N), edge_weights (B,N,K), neighbor_idx (B,N,K).
        """
        affinity = self.compute_affinity(
            patch_out.f_spatial, stf_out.freq_raw, patch_out.positions
        )
        B, N, _ = affinity.shape
        K = min(self.cfg.k, N - 1)

        # Remove self-loops
        diag = torch.eye(N, device=affinity.device, dtype=torch.bool).unsqueeze(0)
        affinity = affinity.masked_fill(diag, 0.0)

        # Top-K neighbors
        edge_weights, neighbor_idx = affinity.topk(K, dim=-1)  # (B, N, K)

        # Build adjacency and symmetrize
        adj = torch.zeros_like(affinity)
        adj.scatter_(-1, neighbor_idx, edge_weights)
        adj = torch.maximum(adj, adj.transpose(1, 2))

        return GraphOutput(adj=adj, edge_weights=edge_weights, neighbor_idx=neighbor_idx)

    def debug_neighbors(self, graph_out: GraphOutput, token_idx: int) -> dict:
        """Return top neighbors for a specific token (debug utility)."""
        return {
            "neighbors": graph_out.neighbor_idx[:, token_idx],
            "weights":   graph_out.edge_weights[:, token_idx],
        }


__all__ = ["FrequencyInformed4DGraphBuilder"]
