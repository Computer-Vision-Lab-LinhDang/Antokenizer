"""Stage 3: Content-Detail Split via slot cross-attention.

ContentExtractor   → N_c content tokens  (0.25·N by default)
DynamicsPooler     → N_d detail tokens   (0.10·N by default)

Monitoring signals (logged during training):
  slot_diversity       : mean pairwise cosine sim of content slots (target ≤ 0.5)
  residual_ratio       : ||R|| / ||x||                            (target 0.3–0.5)
  detail_contribution  : variance fraction from detail branch
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """Single cross-attention + FFN layer (pre-LN)."""

    def __init__(self, dim: int, num_heads: int = 8, kv_dim: Optional[int] = None,
                 mlp_ratio: float = 4.0):
        super().__init__()
        kv_dim = kv_dim or dim
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.norm_ff = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            kdim=kv_dim, vdim=kv_dim,
            batch_first=True, bias=True,
        )
        mlp_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Nq, D), kv: (B, Nkv, D_kv)
        q = self.norm_q(q)
        k = self.norm_kv(kv)
        out, _ = self.attn(q, k, k)
        q = q + out
        q = q + self.ff(self.norm_ff(q))
        return q


class SlotPooler(nn.Module):
    """Slot cross-attention pooler: learns to pool N tokens into num_slots tokens."""

    def __init__(self, num_slots: int, dim: int, num_heads: int = 8,
                 num_layers: int = 2):
        super().__init__()
        self.num_slots = num_slots
        # Learnable slot initialisation
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim) * (dim ** -0.5))
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → slots: (B, num_slots, D)"""
        B = x.shape[0]
        slots = self.slots.expand(B, -1, -1)
        for layer in self.layers:
            slots = layer(slots, x)
        return slots


class ContentDetailSplit(nn.Module):
    """Content-Detail Split module.

    Separates tokens into a content channel (semantic, low-frequency) and a
    detail channel (residual, high-frequency) using learned slot attention.
    """

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 8,
        num_slot_layers: int = 2,
    ):
        super().__init__()
        self.dim = dim
        # Slots are built dynamically based on (N, content_ratio, detail_ratio);
        # we cache poolers per (N_c, N_d) key.
        self._content_poolers: nn.ModuleDict = nn.ModuleDict()
        self._detail_poolers:  nn.ModuleDict = nn.ModuleDict()
        self._num_heads = num_heads
        self._num_slot_layers = num_slot_layers

    def _get_poolers(self, N_c: int, N_d: int) -> Tuple[SlotPooler, SlotPooler]:
        key = f"{N_c}_{N_d}"
        if key not in self._content_poolers:
            self._content_poolers[key] = SlotPooler(
                N_c, self.dim, self._num_heads, self._num_slot_layers)
            self._detail_poolers[key] = SlotPooler(
                N_d, self.dim, self._num_heads, self._num_slot_layers)
        return self._content_poolers[key], self._detail_poolers[key]

    def forward(
        self,
        x: torch.Tensor,       # (B, N, D)
        content_ratio: float = 0.25,
        detail_ratio: float = 0.10,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        compressed : (B, N_c + N_d, D)
        metrics    : dict with slot_diversity, residual_ratio keys
        """
        B, N, D = x.shape
        N_c = max(1, int(N * content_ratio))
        N_d = max(1, int(N * detail_ratio))

        content_pooler, detail_pooler = self._get_poolers(N_c, N_d)
        content_pooler = content_pooler.to(x.device)
        detail_pooler  = detail_pooler.to(x.device)

        # Stage 3a: ContentExtractor
        C = content_pooler(x)   # (B, N_c, D)

        # Stage 3b: Residual via inverse (broadcast) attention
        # weights[b, c, n] = softmax over n: sim(C[b,c], x[b,n])
        weights = F.softmax(
            (C @ x.transpose(-1, -2)) / (D ** 0.5), dim=-1
        )  # (B, N_c, N)
        x_approx = weights.transpose(-1, -2) @ C   # (B, N, D)
        R = x - x_approx                            # (B, N, D)

        # Stage 3c: DynamicsPooler on residual
        D_tokens = detail_pooler(R)   # (B, N_d, D)

        compressed = torch.cat([C, D_tokens], dim=1)  # (B, N_c + N_d, D)

        # Monitoring signals
        metrics = self._compute_metrics(C, R, x)
        return compressed, metrics

    @staticmethod
    def _compute_metrics(C: torch.Tensor, R: torch.Tensor,
                         x: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # slot_diversity: mean pairwise cosine similarity of content slots
            C_n = F.normalize(C, dim=-1)  # (B, N_c, D)
            sim = (C_n @ C_n.transpose(-1, -2))  # (B, N_c, N_c)
            N_c = C.shape[1]
            # exclude diagonal
            mask = ~torch.eye(N_c, dtype=torch.bool, device=C.device)
            slot_div = sim[:, mask].mean() if mask.any() else sim.mean()

            # residual_ratio: ||R|| / ||x||
            res_ratio = (R.norm(dim=-1) / (x.norm(dim=-1) + 1e-8)).mean()

        return {'slot_diversity': slot_div, 'residual_ratio': res_ratio}
