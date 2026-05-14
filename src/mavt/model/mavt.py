"""MAVT: Memory-Augmented Vision Tokenizer.

Unified 7-stage pipeline:
  1. Patchify (Conv3d, modality-specific)
  2. Hybrid Transformer-RGAT Backbone (12 blocks)
  3. Content-Detail Split (slot attention)
  4. Dual Latent Projection (VAE + Semantic)
  5. Modality-Specific Decoder
  6. Losses (handled by LightningModule)
  7. Outputs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from mavt.model.patchify import PatchifyEncoder
from mavt.model.backbone import HybridBackbone
from mavt.model.content_detail_split import ContentDetailSplit
from mavt.model.latent_heads import VAEHead
from mavt.model.decoder import AsymmetricDecoder, UnderstandingDecoder


# Compression ratios per modality (content, detail)
_MODALITY_RATIOS = {
    'image':  (0.25, 0.25),
    'video':  (0.25, 0.25),
    'threed': (0.35, 0.25),
}


@dataclass
class MAVTOutput:
    reconstruction: torch.Tensor      # pixel-space reconstruction
    z: torch.Tensor                    # VAE latent
    mu: torch.Tensor
    logvar: torch.Tensor
    latent_positions: torch.Tensor     # (N_z, 4), content zeros + local detail centers
    latent_token_types: torch.Tensor    # (N_z,), 0=content, 1=detail
    semantic: torch.Tensor             # (B, semantic_dim)
    loss_kl: torch.Tensor
    cd_metrics: Dict[str, torch.Tensor]  # slot_diversity, residual_ratio


class MAVT(nn.Module):
    """Full MAVT model.

    All hyper-parameters are configurable via YAML (Lightning CLI).
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        num_heads: int = 16,
        num_blocks: int = 12,
        patch_size: int = 16,
        t_patch: int = 2,
        # C-D Split
        num_slot_heads: int = 8,
        num_slot_layers: int = 2,
        local_detail_window_size: int = 1,
        local_detail_temporal_window_size: int = 1,
        # VAE
        latent_dim: int = 32,
        kl_weight: float = 1e-4,
        # Semantic
        semantic_dim: int = 768,
        # Decoder
        dec_dim: int = 768,
        num_dec_attn_blocks: int = 4,
        # RGAT
        r_s: int = 2,
        r_t: int = 1,
        # Training
        use_gradient_checkpointing: bool = False,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        # Stage 1
        self.patchify = PatchifyEncoder(embed_dim, patch_size, t_patch)

        # Stage 2
        self.backbone = HybridBackbone(
            dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
            mlp_ratio=mlp_ratio, dropout=dropout,
            r_s=r_s, r_t=r_t,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Stage 3
        self.cd_split = ContentDetailSplit(
            dim=embed_dim, num_heads=num_slot_heads, num_slot_layers=num_slot_layers,
            local_detail_window_size=local_detail_window_size,
            local_detail_temporal_window_size=local_detail_temporal_window_size,
        )

        # Stage 4 — VAE bottleneck only (semantic moved downstream of z)
        self.vae_head = VAEHead(embed_dim, latent_dim, kl_weight)

        # Stage 5 — two heads decoding from the shared latent z
        # 5a. Reconstruction head: z → pixel
        self.decoder = AsymmetricDecoder(
            latent_dim=latent_dim, dec_dim=dec_dim,
            num_attn_blocks=num_dec_attn_blocks, num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # 5b. Understanding head: z → semantic vector aligned with vision teacher
        self.understanding_decoder = UnderstandingDecoder(
            latent_dim=latent_dim, dec_dim=dec_dim,
            semantic_dim=semantic_dim, num_heads=8, num_layers=2,
            mlp_ratio=mlp_ratio,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _grid_shape(self, modality: str, x: torch.Tensor) -> tuple:
        """Return (H_grid, W_grid) or (Tp, Hg, Wg) based on input shape."""
        if modality == 'image':
            _, _, H, W = x.shape
            return (H // self.patch_size, W // self.patch_size)
        elif modality == 'video':
            _, _, T, H, W = x.shape
            return (T // 2, H // self.patch_size, W // self.patch_size)
        elif modality == 'threed':
            _, _, _, S, _ = x.shape   # (B, 3planes, 3ch, S, S)
            return (S // self.patch_size, S // self.patch_size)
        raise ValueError(modality)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        decode: bool = True,
    ) -> MAVTOutput:
        """
        x : raw input tensor (see patchify.py for shapes per modality)
        modality : 'image' | 'video' | 'threed'
        decode : if False, skip decoder (encoder-only mode for downstream tasks)
        """
        grid_shape = self._grid_shape(modality, x)

        # Stage 1 — Patchify
        tokens, positions, plane_ids = self.patchify(x, modality)
        # tokens: (B, N, D), positions: (N, 4), plane_ids: (N,)

        # Stage 2 — Hybrid backbone
        features = self.backbone(tokens, positions, plane_ids, modality)

        # Stage 3 — Content-Detail Split
        content_ratio, detail_ratio = _MODALITY_RATIOS[modality]
        compressed, cd_metrics, latent_positions, latent_token_types = self.cd_split(
            features,
            positions=positions,
            plane_ids=plane_ids,
            content_ratio=content_ratio,
            detail_ratio=detail_ratio,
            return_metadata=True,
        )  # (B, N_c + N_d, D)

        # Stage 4 — VAE bottleneck (semantic now derives from z, not compressed)
        z, mu, logvar, loss_kl = self.vae_head(compressed)

        # Stage 5a — Understanding head: z → semantic
        # Always run (cheap, gives semantic supervision signal even when decode=False)
        semantic = self.understanding_decoder(z)

        # Stage 5b — Reconstruction head: z → pixel
        if decode:
            recon = self.decoder(
                z, positions, modality, grid_shape,
                latent_positions=latent_positions,
                latent_token_types=latent_token_types,
            )
        else:
            recon = torch.zeros(1, device=x.device)  # placeholder

        return MAVTOutput(
            reconstruction=recon,
            z=z,
            mu=mu,
            logvar=logvar,
            latent_positions=latent_positions,
            latent_token_types=latent_token_types,
            semantic=semantic,
            loss_kl=loss_kl,
            cd_metrics=cd_metrics,
        )

    def encode(self, x: torch.Tensor, modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return (z, semantic) without decoding."""
        out = self.forward(x, modality, decode=False)
        return out.z, out.semantic

    def load_siglip2_weights(self, model_name: str = "google/siglip2-base-patch16-224",
                              freeze_stages: int = 10) -> None:
        self.backbone.load_siglip2_weights(model_name, freeze_stages)

    # ------------------------------------------------------------------ #
    #  Eager pre-creation of slot poolers                                 #
    # ------------------------------------------------------------------ #

    def prepare_for_modalities(self, specs: Iterable[Dict[str, Any]]) -> None:
        """Pre-create every SlotPooler the trainer will need.

        Must be called BEFORE the optimizer is built (e.g. from
        LightningModule.setup) so the pooler params are picked up by the
        optimizer's param_groups. Without this, poolers are created lazily
        in ContentDetailSplit.forward and their parameters never receive
        gradient updates.

        Each spec dict has key 'modality' plus modality-specific shape keys:
            image  : {'modality': 'image',  'resolution': H}
            video  : {'modality': 'video',  'resolution': H, 'frames': T,
                      't_patch': 2}                # t_patch optional
            threed : {'modality': 'threed', 'resolution': S}
        """
        for spec in specs:
            modality = spec['modality']
            if modality == 'image':
                H = spec['resolution']
                Hp = H // self.patch_size
                N  = Hp * Hp
            elif modality == 'video':
                H  = spec['resolution']
                T  = spec['frames']
                tp = spec.get('t_patch', 2)
                Tp = T // tp
                Hp = H // self.patch_size
                N  = Tp * Hp * Hp
            elif modality == 'threed':
                S  = spec['resolution']
                Sp = S // self.patch_size
                N  = 3 * Sp * Sp                  # 3 planes
            else:
                raise ValueError(f"Unknown modality in spec: {modality!r}")
            c_r, d_r = _MODALITY_RATIOS[modality]
            N_c = max(1, int(N * c_r))
            N_d = max(1, int(N * d_r))
            self.cd_split.prepare_poolers(N_c, N_d)
