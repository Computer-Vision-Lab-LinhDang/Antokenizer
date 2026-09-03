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
from mavt.model.graph_attention import SparseRGAT4D, build_graph
from mavt.model.flex_rgat import FlexRGAT4D, build_flex_graph


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
        rgat_impl: str = "dense",          # dense | sparse | flex
        edge_plane_local: bool = False,    # spatial edges in each plane's own frame
        edge_cross_mode: str = "shared_axis",  # shared_axis | projection
    ):
        super().__init__()
        self.r_s = r_s
        self.r_t = r_t
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if rgat_impl not in ("dense", "sparse", "flex"):
            raise ValueError(f"rgat_impl must be dense|sparse|flex, got {rgat_impl!r}")
        self.rgat_impl = rgat_impl
        self.edge_plane_local = edge_plane_local
        self.edge_cross_mode = edge_cross_mode

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i in RGAT_POSITIONS:
                if rgat_impl == "dense":
                    blk = RGAT4DBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                elif rgat_impl == "sparse":
                    blk = SparseRGAT4D(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, r_s=r_s, r_t=r_t)
                else:
                    blk = FlexRGAT4D(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, r_s=r_s, r_t=r_t)
                self.blocks.append(blk)
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
            if self.rgat_impl == "dense":
                self._mask_cache[key] = build_adjacency(positions, plane_ids, modality, self.r_s, self.r_t)
            elif self.rgat_impl == "sparse":
                self._mask_cache[key] = build_graph(positions, plane_ids, modality, self.r_s, self.r_t,
                                                    self.edge_plane_local, self.edge_cross_mode)
            else:
                self._mask_cache[key] = build_flex_graph(positions, plane_ids, modality, self.r_s, self.r_t,
                                                         self.edge_plane_local, self.edge_cross_mode)
        return self._mask_cache[key]

    # ------------------------------------------------------------------ #

    def _run_transformer(self, block: StandardTransformerBlock, x: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return cp.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _run_rgat(self, block: nn.Module, x: torch.Tensor, graph) -> torch.Tensor:
        if self.rgat_impl == "dense":
            adj_mask, edge_type_masks = graph
            if self.use_gradient_checkpointing and self.training:
                return cp.checkpoint(block, x, adj_mask, edge_type_masks, use_reentrant=False)
            return block(x, adj_mask, edge_type_masks)
        if self.use_gradient_checkpointing and self.training:
            return cp.checkpoint(block, x, graph, use_reentrant=False)
        return block(x, graph)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,          # (B, N, D)
        positions: torch.Tensor,  # (N, 4)
        plane_ids: torch.Tensor,  # (N,)
        modality: str,
    ) -> torch.Tensor:
        graph = self._get_masks(positions, plane_ids, modality)
        if self.rgat_impl == "dense":
            adj_mask, edge_type_masks = graph
            graph = (adj_mask.to(x.device), [m.to(x.device) for m in edge_type_masks])

        for i, block in enumerate(self.blocks):
            if i in RGAT_POSITIONS:
                x = self._run_rgat(block, x, graph)
            else:
                x = self._run_transformer(block, x)
        return x

    # ------------------------------------------------------------------ #
    #  SigLIP2 weight loading utility                                     #
    # ------------------------------------------------------------------ #

    def load_siglip2_weights(self, model_name: str = "google/siglip2-so400m-patch16-384",
                              freeze_stages: int = 0, strict: bool = True) -> None:
        """Load SigLIP2 encoder layers into the Transformer blocks.

        strict=True (default): any failure — model unloadable, shape mismatch, zero
        blocks copied — raises RuntimeError instead of silently training from
        random init (which is what happened in every earlier run).
        freeze_stages: number of initial Transformer blocks to freeze.
        """
        copied = 0
        try:
            from transformers import AutoModel
            siglip = AutoModel.from_pretrained(model_name)
            siglip_blocks = siglip.vision_model.encoder.layers

            transformer_idx = 0  # index into siglip_blocks
            for block_idx, block in enumerate(self.blocks):
                if block_idx in RGAT_POSITIONS:
                    continue
                if transformer_idx >= len(siglip_blocks):
                    break
                src = siglip_blocks[transformer_idx]
                if _copy_siglip2_block(src, block, strict=strict):
                    copied += 1
                transformer_idx += 1
            n_tf = sum(1 for i in range(len(self.blocks)) if i not in RGAT_POSITIONS)
            print(f"[backbone] SigLIP2 init: copied {copied}/{n_tf} transformer blocks from {model_name}")
            if strict and copied == 0:
                raise RuntimeError("SigLIP2 init copied zero blocks")

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
            if strict:
                raise RuntimeError(f"SigLIP2 weight loading failed for {model_name!r}: {exc}") from exc
            print(f"[backbone] SigLIP2 weight loading skipped: {exc}")


def _copy_siglip2_block(src: nn.Module, dst: StandardTransformerBlock, strict: bool = False) -> bool:
    """Copy a SigLIP2 encoder layer into our StandardTransformerBlock. Returns True on success."""
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
        return True
    except (AttributeError, RuntimeError) as exc:
        if strict:
            raise RuntimeError(f"SigLIP2 block copy failed (dim/name mismatch): {exc}") from exc
        return False
