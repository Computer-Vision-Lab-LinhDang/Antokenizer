"""Stage 2: Hybrid Transformer-RGAT4D Backbone.

Block layout (12 blocks):
  0-3:  StandardTransformerBlock  (SigLIP2 init)
  4:    RGAT4DBlock               (zero-init output)
  5-7:  StandardTransformerBlock  (SigLIP2 init)
  8:    RGAT4DBlock               (zero-init output)
  9-11: StandardTransformerBlock  (SigLIP2 init)

Adjacency masks are precomputed once per (modality, resolution) and cached.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mavt.model.transformer import StandardTransformerBlock
from mavt.model.rgat import RGAT4DBlock, build_adjacency


RGAT_POSITIONS = {4, 8}   # which block indices are RGAT4D


class HybridBackbone(nn.Module):
    """12-block hybrid Transformer-RGAT backbone."""

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        num_blocks: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        r_s: int = 2,
        r_t: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.r_s = r_s
        self.r_t = r_t
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i in RGAT_POSITIONS:
                self.blocks.append(RGAT4DBlock(dim, num_heads,
                                               mlp_ratio=mlp_ratio, dropout=dropout))
            else:
                self.blocks.append(StandardTransformerBlock(dim, num_heads,
                                                             mlp_ratio=mlp_ratio, dropout=dropout))

        # Mask cache: key = (modality, N) → (adj_mask, edge_type_masks)
        self._mask_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, List[torch.Tensor]]] = {}

    # ------------------------------------------------------------------ #

    def _get_masks(
        self,
        positions: torch.Tensor,  # (N, 4)
        plane_ids: torch.Tensor,  # (N,)
        modality: str,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        key = (modality, positions.shape[0])
        if key not in self._mask_cache:
            adj, etype_masks = build_adjacency(positions, plane_ids, modality,
                                               self.r_s, self.r_t)
            self._mask_cache[key] = (adj, etype_masks)
        return self._mask_cache[key]

    # ------------------------------------------------------------------ #

    def _run_transformer(self, block: StandardTransformerBlock, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return cp.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _run_rgat(
        self,
        block: RGAT4DBlock,
        x: torch.Tensor,
        adj_mask: torch.Tensor,
        edge_type_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return cp.checkpoint(block, x, adj_mask, edge_type_masks, use_reentrant=False)
        return block(x, adj_mask, edge_type_masks)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,          # (B, N, D)
        positions: torch.Tensor,  # (N, 4)
        plane_ids: torch.Tensor,  # (N,)
        modality: str,
    ) -> torch.Tensor:
        adj_mask, edge_type_masks = self._get_masks(positions, plane_ids, modality)
        # Move cached masks to current device if needed
        adj_mask = adj_mask.to(x.device)
        edge_type_masks = [m.to(x.device) for m in edge_type_masks]

        for i, block in enumerate(self.blocks):
            if i in RGAT_POSITIONS:
                x = self._run_rgat(block, x, adj_mask, edge_type_masks)
            else:
                x = self._run_transformer(block, x)
        return x

    # ------------------------------------------------------------------ #
    #  SigLIP2 weight loading utility                                     #
    # ------------------------------------------------------------------ #

    def load_siglip2_weights(self, model_name: str = "google/siglip2-base-patch16-224",
                              freeze_stages: int = 0) -> None:
        """Load SigLIP2 backbone weights into Transformer blocks (best-effort).

        freeze_stages: number of initial Transformer blocks to freeze (stage 1: all,
        stage 2: leave last 4 unfrozen, stage 3: none frozen).
        """
        try:
            from transformers import AutoModel
            import re
            siglip = AutoModel.from_pretrained(model_name)
            siglip_blocks = siglip.vision_model.encoder.layers

            transformer_idx = 0  # index into siglip_blocks
            for block_idx, block in enumerate(self.blocks):
                if block_idx in RGAT_POSITIONS:
                    continue
                if transformer_idx >= len(siglip_blocks):
                    break
                src = siglip_blocks[transformer_idx]
                _copy_siglip2_block(src, block)
                transformer_idx += 1

            # Freeze early blocks
            frozen = 0
            for block_idx, block in enumerate(self.blocks):
                if block_idx in RGAT_POSITIONS:
                    continue
                if frozen < freeze_stages:
                    for p in block.parameters():
                        p.requires_grad_(False)
                    frozen += 1

        except Exception as exc:  # noqa: BLE001
            print(f"[backbone] SigLIP2 weight loading skipped: {exc}")


def _copy_siglip2_block(src: nn.Module, dst: StandardTransformerBlock) -> None:
    """Best-effort copy from a SigLIP2 encoder layer to our StandardTransformerBlock."""
    state = dst.state_dict()
    # SigLIP2 uses self_attn.{q,k,v,out}_proj; we use fused qkv + out_proj
    try:
        Q = src.self_attn.q_proj.weight.data
        K = src.self_attn.k_proj.weight.data
        V = src.self_attn.v_proj.weight.data
        state['qkv.weight'] = torch.cat([Q, K, V], dim=0)
        if src.self_attn.q_proj.bias is not None:
            Qb = src.self_attn.q_proj.bias.data
            Kb = src.self_attn.k_proj.bias.data
            Vb = src.self_attn.v_proj.bias.data
            state['qkv.bias'] = torch.cat([Qb, Kb, Vb], dim=0)
        state['out_proj.weight'] = src.self_attn.out_proj.weight.data
        if src.self_attn.out_proj.bias is not None:
            state['out_proj.bias'] = src.self_attn.out_proj.bias.data
        # LayerNorm
        state['norm1.weight'] = src.layer_norm1.weight.data
        state['norm1.bias']   = src.layer_norm1.bias.data
        state['norm2.weight'] = src.layer_norm2.weight.data
        state['norm2.bias']   = src.layer_norm2.bias.data
        # MLP
        state['mlp.0.weight'] = src.mlp.fc1.weight.data
        state['mlp.0.bias']   = src.mlp.fc1.bias.data
        state['mlp.3.weight'] = src.mlp.fc2.weight.data
        state['mlp.3.bias']   = src.mlp.fc2.bias.data
        dst.load_state_dict(state)
    except (AttributeError, RuntimeError):
        pass  # dimension mismatch or different naming — skip silently
