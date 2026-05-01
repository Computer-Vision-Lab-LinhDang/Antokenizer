"""Stage 3: Content-Detail Split via slot cross-attention.

ContentExtractor   → N_c content tokens  (0.25·N by default)
DynamicsPooler     → N_d detail tokens   (0.10·N by default)

Monitoring signals (logged during training):
  slot_diversity       : mean pairwise cosine sim of content slots (target ≤ 0.5)
  residual_ratio       : ||R|| / ||x||                            (target 0.3–0.5)
  detail_contribution  : variance fraction from detail branch
"""

from __future__ import annotations
import math
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
    """Slot cross-attention pooler with optional spatial position anchoring.

    When grid_h/grid_w are provided, each slot query is initialised with a
    fixed 2-D sinusoidal position bias so the cross-attention learns to pool
    spatially localised information rather than permutation-invariant slots.
    """

    def __init__(self, num_slots: int, dim: int, num_heads: int = 8,
                 num_layers: int = 2,
                 grid_h: Optional[int] = None, grid_w: Optional[int] = None):
        super().__init__()
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim) * (dim ** -0.5))
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])
        if grid_h is not None and grid_w is not None:
            pos = SlotPooler._make_2d_sinpos(grid_h, grid_w, dim)
            self.register_buffer('slot_pos', pos)   # (1, num_slots, dim), fixed
        else:
            self.slot_pos = None

    @staticmethod
    def _make_2d_sinpos(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
        """Fixed 2-D sinusoidal encoding → (1, grid_h*grid_w, dim)."""
        assert dim % 4 == 0, "dim must be divisible by 4"
        d = dim // 4
        den = 10000 ** (torch.arange(0, d, dtype=torch.float) / d)
        rows = torch.arange(grid_h, dtype=torch.float).unsqueeze(1) / den   # (H, d)
        cols = torch.arange(grid_w, dtype=torch.float).unsqueeze(1) / den   # (W, d)
        row_enc = torch.stack([rows.sin(), rows.cos()], dim=-1).flatten(1)  # (H, 2d)
        col_enc = torch.stack([cols.sin(), cols.cos()], dim=-1).flatten(1)  # (W, 2d)
        row_exp = row_enc.unsqueeze(1).expand(grid_h, grid_w, 2 * d)
        col_exp = col_enc.unsqueeze(0).expand(grid_h, grid_w, 2 * d)
        pos = torch.cat([row_exp, col_exp], dim=-1)                          # (H, W, dim)
        return pos.reshape(1, grid_h * grid_w, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → slots: (B, num_slots, D)"""
        B = x.shape[0]
        slots = self.slots
        if self.slot_pos is not None:
            slots = slots + self.slot_pos
        slots = slots.expand(B, -1, -1)
        for layer in self.layers:
            slots = layer(slots, x)
        return slots


class ContentDetailSplit(nn.Module):
    """Content-Detail Split module.

    Separates tokens into a content channel (semantic, low-frequency) and a
    detail channel (residual, high-frequency) using learned slot attention.

    Note on parameter registration:
      Slot poolers depend on (N_c, N_d) which depend on (modality, resolution).
      Call ``prepare_poolers(N_c, N_d)`` for every combo that will appear at
      training time BEFORE the optimizer is built — otherwise the pooler
      params are not in any param_group and never receive updates. The lazy
      fallback in ``_get_poolers`` only exists to keep smoke tests and
      one-off inference paths functional; it emits a ``RuntimeWarning``.
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

    @staticmethod
    def _grid_dims(n: int) -> Tuple[Optional[int], Optional[int]]:
        """Return (H, W) with H*W == n, H closest to sqrt(n). None if not found."""
        h = round(math.sqrt(n))
        for delta in range(6):
            for h_try in sorted({max(1, h - delta), h + delta}):
                if n % h_try == 0:
                    return h_try, n // h_try
        return None, None

    def prepare_poolers(self, N_c: int, N_d: int) -> None:
        """Eagerly create poolers for a known (N_c, N_d) combo.

        Call once per expected combo BEFORE ``configure_optimizers`` runs so
        that the new params are picked up by the optimizer's param_groups.
        """
        key = f"{N_c}_{N_d}"
        if key in self._content_poolers:
            return
        c_h, c_w = self._grid_dims(N_c)
        d_h, d_w = self._grid_dims(N_d)
        self._content_poolers[key] = SlotPooler(
            N_c, self.dim, self._num_heads, self._num_slot_layers,
            grid_h=c_h, grid_w=c_w)
        self._detail_poolers[key] = SlotPooler(
            N_d, self.dim, self._num_heads, self._num_slot_layers,
            grid_h=d_h, grid_w=d_w)

    def _get_poolers(self, N_c: int, N_d: int) -> Tuple[SlotPooler, SlotPooler]:
        key = f"{N_c}_{N_d}"
        if key not in self._content_poolers:
            import warnings
            warnings.warn(
                f"ContentDetailSplit: lazy pooler creation for "
                f"(N_c={N_c}, N_d={N_d}); their params are NOT in the "
                f"optimizer and will stay at random init. Call "
                f"prepare_poolers() in setup() before configure_optimizers().",
                RuntimeWarning,
                stacklevel=2,
            )
            self.prepare_poolers(N_c, N_d)
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
