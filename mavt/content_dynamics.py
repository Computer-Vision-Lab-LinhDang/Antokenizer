"""Unified Content-Detail Split for all visual modalities.

Works on the 4D token space directly — no modality branching.
The module sees only tokens (B, N, D) + positions (B, N, 4) and learns
to select content vs detail purely from the data.

Pipeline:  Patchify → **UnifiedContentDetailSplit** → Encoder

Components
----------
ContentExtractor           Slot-attention: learnable queries → content tokens.
DynamicsPooler             Cross-attention: compress residual → detail tokens.
UnifiedContentDetailSplit  Orchestrator for any modality.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ContentDynamicsConfig
from .types import ContentDynamicsOutput, PatchifyOutput


# ── Building blocks ──────────────────────────────────────────────────


class ContentExtractor(nn.Module):
    """Learnable slot-attention queries extract content from ANY 4D token set.

    Queries are modality-agnostic — they learn what to keep purely from
    features + 4D positions seen during training.
    """

    def __init__(
        self,
        d_model: int = 1152,
        n_heads: int = 16,
        max_queries: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.max_queries = max_queries
        self.query_bank = nn.Parameter(torch.randn(max_queries, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    d_model, n_heads, batch_first=True,
                ),
                "norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm_ffn": nn.LayerNorm(d_model),
            }))

    def forward(
        self,
        tokens: torch.Tensor,      # (B, N, D)
        positions: torch.Tensor,    # (B, N, 4)
        Nc: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract *Nc* content tokens via slot cross-attention.

        Returns:
            content:      (B, Nc, D)
            content_pos:  (B, Nc, 4)   attention-weighted position centroids
            attn_weights: (B, Nc, N)   last-layer attention (for residual computation)
        """
        B = tokens.shape[0]
        Q = self.query_bank[:Nc].unsqueeze(0).expand(B, -1, -1)

        out = Q
        attn_w = None
        for layer in self.layers:
            residual = out
            attn_out, attn_w = layer["cross_attn"](out, tokens, tokens)
            out = layer["norm"](attn_out + residual)
            out = layer["norm_ffn"](layer["ffn"](out) + out)

        # Position = attention-weighted centroid of input positions.
        content_pos = torch.bmm(attn_w, positions)  # (B, Nc, 4)

        return out, content_pos, attn_w


class DynamicsPooler(nn.Module):
    """Compress residual features into *K* detail tokens.

    Re-used from video C-D Split — works identically for any modality.
    KV = concat(content, residual) so queries see both global context
    and the sparse difference signal.
    """

    def __init__(
        self,
        d_model: int = 1152,
        n_heads: int = 16,
        max_K: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.max_K = max_K
        self.queries = nn.Parameter(torch.randn(max_K, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    d_model, n_heads, batch_first=True,
                ),
                "norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm_ffn": nn.LayerNorm(d_model),
            }))

    def forward(
        self,
        content_feat: torch.Tensor,   # (B, Nc, D)
        residual_feat: torch.Tensor,  # (B, N, D)
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress residual into *K* detail tokens.

        Returns:
            detail:      (B, K, D)
            attn_weights (B, K, Nc+N)  last-layer attention
        """
        B = content_feat.shape[0]
        Q = self.queries[:K].unsqueeze(0).expand(B, -1, -1)
        KV = torch.cat([content_feat, residual_feat], dim=1)

        out = Q
        attn_w = None
        for layer in self.layers:
            residual = out
            attn_out, attn_w = layer["cross_attn"](out, KV, KV)
            out = layer["norm"](attn_out + residual)
            out = layer["norm_ffn"](layer["ffn"](out) + out)

        return out, attn_w


# ── Main module ──────────────────────────────────────────────────────


class UnifiedContentDetailSplit(nn.Module):
    """Unified content-detail split for image, video, and 3D.

    No modality-specific branching — operates entirely on
    tokens (B, N, D) + positions (B, N, 4).

    The module learns:
      * Which tokens are informative → keep as **content**
      * What information is missing → compress as **detail**

    through end-to-end training with reconstruction loss.
    """

    def __init__(self, cfg: ContentDynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.content_extractor = ContentExtractor(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_queries=cfg.max_content_queries,
            n_layers=cfg.n_pooler_layers,
        )
        self.pooler = DynamicsPooler(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            max_K=cfg.max_content_queries,
            n_layers=cfg.n_pooler_layers,
        )

    def forward(self, patch_out: PatchifyOutput) -> ContentDynamicsOutput:
        B, N, D = patch_out.f_spatial.shape
        tokens = patch_out.f_spatial
        positions = patch_out.positions

        Nc = max(self.cfg.min_tokens, int(N * self.cfg.content_ratio))
        Nd = max(self.cfg.min_tokens, int(N * self.cfg.detail_ratio))

        # Don't expand: if compressed would exceed input, pass through.
        if Nc + Nd >= N:
            return ContentDynamicsOutput(
                tokens=tokens,
                positions=positions,
                token_types=torch.zeros(B, N, device=tokens.device),
                modality=patch_out.modality,
            )

        # 1. Content extraction — slot attention.
        content, content_pos, content_attn = self.content_extractor(
            tokens, positions, Nc,
        )

        # 2. Residual = input − reconstruction_from_content.
        #    Near-zero where content already captures the information.
        recon_w = content_attn.transpose(1, 2)  # (B, N, Nc)
        recon_w = recon_w / recon_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        reconstructed = torch.bmm(recon_w, content)  # (B, N, D)
        residual = tokens - reconstructed

        # 3. Detail compression — pool residual into Nd tokens.
        detail, detail_attn = self.pooler(content, residual, Nd)

        #    Detail positions from residual attention (skip content portion).
        Nc_kv = content.shape[1]
        residual_attn = detail_attn[:, :, Nc_kv:]  # (B, Nd, N)
        residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        detail_pos = torch.bmm(residual_attn, positions)  # (B, Nd, 4)

        # 4. Assemble.
        out_tokens = torch.cat([content, detail], dim=1)
        out_positions = torch.cat([content_pos, detail_pos], dim=1)
        token_types = torch.cat([
            torch.zeros(B, Nc, device=tokens.device),
            torch.ones(B, Nd, device=tokens.device),
        ], dim=1)

        return ContentDynamicsOutput(
            tokens=out_tokens,
            positions=out_positions,
            token_types=token_types,
            modality=patch_out.modality,
            cd_metadata={
                "n_content": Nc,
                "n_detail": Nd,
                "n_original": N,
                "original_positions": positions,
            },
        )


__all__ = [
    "ContentExtractor",
    "DynamicsPooler",
    "UnifiedContentDetailSplit",
]
