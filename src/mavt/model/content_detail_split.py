"""Stage 3: Content-Detail Split via slot cross-attention.

ContentExtractor   → N_c content tokens  (0.25·N by default)
LocalDetailPooler  → local residual detail tokens

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


def content_reconstruction_error(x: torch.Tensor, x_approx: torch.Tensor) -> torch.Tensor:
    """Relative error of the content reconstruction, normalised by the *centred* energy.

    Backbone features carry ~99.9 % of their energy in a mean vector shared by every patch,
    so normalising by ``||x||^2`` makes the metric trivially satisfiable: predicting that mean
    alone scores 0.001 while capturing none of the per-patch variation (measured 2026-09-03 —
    the first version of this loss learned exactly that degenerate solution). Dividing by the
    variance around the per-sample mean makes "predict the mean" score 1.0, so only real
    structure lowers it. 0 = slots reproduce x, 1 = slots carry nothing beyond the mean.
    """
    dev = x - x.mean(dim=1, keepdim=True)
    num = (x - x_approx).pow(2).sum(-1).mean()
    den = dev.pow(2).sum(-1).mean().clamp_min(1e-8)
    return num / den


class ContentDetailSplit(nn.Module):
    """Content-Detail Split module.

    Separates tokens into a content channel (semantic, low-frequency) and a
    detail channel (residual, high-frequency).

    Content stays global: learned slot attention pools the full token sequence
    into semantic / low-frequency slots. Detail is local: residual tokens are
    pooled inside small coordinate windows, preserving a window-center position
    for each detail token. The decoder can then prefer nearby detail tokens
    instead of reconstructing texture from positionless global slots.

    Note on parameter registration:
      Content slot poolers depend on N_c which depends on modality / resolution.
      Call ``prepare_poolers(N_c, N_d)`` for every combo that will appear at
      training time BEFORE the optimizer is built — otherwise the pooler
      params are not in any param_group and never receive updates. The lazy
      fallback in ``_get_content_pooler`` only exists to keep smoke tests and
      one-off inference paths functional; it emits a ``RuntimeWarning``.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_slot_layers: int = 2,
        local_detail_window_size: int = 1,
        local_detail_temporal_window_size: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.local_detail_window_size = local_detail_window_size
        self.local_detail_temporal_window_size = local_detail_temporal_window_size
        # Content slots are built dynamically based on (N, content_ratio);
        # the key keeps N_d for backward-compatible checkpoint naming.
        self._content_poolers: nn.ModuleDict = nn.ModuleDict()
        self._num_heads = num_heads
        self._num_slot_layers = num_slot_layers
        self.detail_norm = nn.LayerNorm(dim)
        self.detail_proj = nn.Linear(dim, dim)

    def prepare_poolers(self, N_c: int, N_d: int) -> None:
        """Eagerly create content poolers for a known (N_c, N_d) combo.

        Call once per expected combo BEFORE ``configure_optimizers`` runs so
        that the new params are picked up by the optimizer's param_groups.
        """
        key = f"{N_c}_{N_d}"
        if key in self._content_poolers:
            return
        self._content_poolers[key] = SlotPooler(
            N_c, self.dim, self._num_heads, self._num_slot_layers)

    def _get_content_pooler(self, N_c: int, N_d: int) -> SlotPooler:
        key = f"{N_c}_{N_d}"
        if key not in self._content_poolers:
            import warnings
            warnings.warn(
                f"ContentDetailSplit: lazy pooler creation for "
                f"(N_c={N_c}, N_d={N_d}); its params are NOT in the "
                f"optimizer and will stay at random init. Call "
                f"prepare_poolers() in setup() before configure_optimizers().",
                RuntimeWarning,
                stacklevel=2,
            )
            self.prepare_poolers(N_c, N_d)
        return self._content_poolers[key]

    @staticmethod
    def _default_positions(N: int, device: torch.device) -> torch.Tensor:
        """Fallback positions for direct unit tests without patch metadata."""
        side = int(N ** 0.5)
        pos = torch.zeros(N, 4, dtype=torch.long, device=device)
        if side * side == N:
            i = torch.arange(side, device=device)
            j = torch.arange(side, device=device)
            gi, gj = torch.meshgrid(i, j, indexing='ij')
            pos[:, 1] = gi.reshape(-1)
            pos[:, 2] = gj.reshape(-1)
        else:
            pos[:, 1] = torch.arange(N, device=device)
        return pos

    @staticmethod
    def _content_weights(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Assignment of each patch to the content slots: softmax over the SLOT axis.

        (B, N_c, N), with ``w[:, :, n]`` summing to 1 so ``x_approx[n]`` is a convex
        combination of slots and can reach the scale of ``x[n]``. Normalising over the
        patch axis instead (the 2026-09-02 code) made ``x_approx`` an unnormalised
        mixture with ~0.2x the norm of ``x`` and cos(x, x_approx) = 0.027, so the
        "residual" R was as large as x itself and the detail branch carried everything.
        """
        D = x.shape[-1]
        return F.softmax((C @ x.transpose(-1, -2)) / (D ** 0.5), dim=-2)

    def _local_detail_pool(
        self,
        residual: torch.Tensor,
        positions: Optional[torch.Tensor],
        plane_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool residual tokens in local coordinate windows.

        Returns
        -------
        detail_tokens    : (B, N_d_local, D)
        detail_positions : (N_d_local, 4), rounded window centers
        detail_counts    : (N_d_local,), number of source tokens per window
        """
        B, N, D = residual.shape
        device = residual.device
        if positions is None:
            positions = self._default_positions(N, device)
        positions = positions.to(device=device, dtype=torch.long)

        if plane_ids is None:
            plane_ids = torch.full((N,), -1, dtype=torch.long, device=device)
        else:
            plane_ids = plane_ids.to(device=device, dtype=torch.long)

        grouped = positions.clone()
        t_win = max(1, int(self.local_detail_temporal_window_size))
        s_win = max(1, int(self.local_detail_window_size))
        grouped[:, 0] = grouped[:, 0] // t_win
        grouped[:, 1] = grouped[:, 1] // s_win
        grouped[:, 2] = grouped[:, 2] // s_win
        grouped[:, 3] = grouped[:, 3] // s_win

        group_coords = torch.cat([plane_ids.unsqueeze(1), grouped], dim=1)
        _, inverse = torch.unique(group_coords, dim=0, sorted=True, return_inverse=True)
        num_groups = int(inverse.max().item()) + 1

        idx = inverse.view(1, N, 1).expand(B, N, D)
        pooled = residual.new_zeros(B, num_groups, D)
        pooled.scatter_add_(1, idx, residual)

        counts = torch.bincount(inverse, minlength=num_groups).to(device=device)
        pooled = pooled / counts.view(1, num_groups, 1).clamp_min(1).to(residual.dtype)
        detail_tokens = self.detail_proj(self.detail_norm(pooled))

        pos_sum = torch.zeros(num_groups, 4, device=device, dtype=torch.float32)
        pos_sum.scatter_add_(0, inverse.view(N, 1).expand(N, 4), positions.float())
        detail_positions = (
            pos_sum / counts.view(num_groups, 1).clamp_min(1).float() + 0.5
        ).floor().long()

        return detail_tokens, detail_positions, counts

    def forward(
        self,
        x: torch.Tensor,       # (B, N, D)
        positions: Optional[torch.Tensor] = None,
        plane_ids: Optional[torch.Tensor] = None,
        content_ratio: float = 0.25,
        detail_ratio: float = 0.25,
        return_metadata: bool = False,
    ):
        """
        Returns
        -------
        compressed : (B, N_c + N_d_local, D)
        metrics    : dict with slot_diversity, residual_ratio keys

        If return_metadata=True, also returns:
        latent_positions  : (N_c + N_d_local, 4)
        latent_token_type : (N_c + N_d_local,), 0=content, 1=detail
        """
        B, N, D = x.shape
        N_c = max(1, int(N * content_ratio))
        # Kept for pooler-key stability. Detail tokens are now determined by
        # local coordinate windows rather than by global slot count.
        N_d_key = max(1, int(N * detail_ratio))

        content_pooler = self._get_content_pooler(N_c, N_d_key)
        content_pooler = content_pooler.to(x.device)

        # Stage 3a: ContentExtractor
        C = content_pooler(x)   # (B, N_c, D)

        # Stage 3b: residual = x minus its reconstruction from the content slots.
        weights = self._content_weights(x, C)       # (B, N_c, N), convex per patch
        x_approx = weights.transpose(-1, -2) @ C    # (B, N, D)
        R = x - x_approx                            # (B, N, D)

        # Stage 3c: local residual detail tokens with explicit positions
        D_tokens, D_positions, detail_counts = self._local_detail_pool(
            R, positions, plane_ids
        )

        compressed = torch.cat([C, D_tokens], dim=1)  # (B, N_c + N_d, D)

        # Monitoring signals
        metrics = self._compute_metrics(C, R, x)
        # Relative error of the content reconstruction: 0 = slots span x, 1 = slots useless.
        metrics['content_recon_error'] = content_reconstruction_error(x, x_approx)
        metrics['detail_token_count'] = torch.tensor(
            D_tokens.shape[1], device=x.device, dtype=x.dtype)
        metrics['detail_avg_window_tokens'] = detail_counts.float().mean().to(
            device=x.device, dtype=x.dtype)

        if not return_metadata:
            return compressed, metrics

        content_positions = torch.zeros(N_c, 4, dtype=torch.long, device=x.device)
        latent_positions = torch.cat([content_positions, D_positions], dim=0)
        latent_token_type = torch.cat([
            torch.zeros(N_c, dtype=torch.long, device=x.device),
            torch.ones(D_tokens.shape[1], dtype=torch.long, device=x.device),
        ], dim=0)
        return compressed, metrics, latent_positions, latent_token_type

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
