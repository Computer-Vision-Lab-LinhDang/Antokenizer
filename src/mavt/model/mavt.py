"""MAVT: Memory-Augmented Vision Tokenizer with Matryoshka head.

Pipeline:
  1. Patchify (Conv3d, modality-shared)
  2. Hybrid Transformer-RGAT backbone (12 blocks)
  3. Content-Detail Split (slot attention) → compressed (B, N, D_max)
  4. Matryoshka head: per-prefix VAE + understanding pool over nested
     channel widths {d_1, …, d_K = D_max}
  5a. Reconstruction:   per-prefix z_k → shared AsymmetricDecoder → pixels
  5b. Understanding:    per-prefix global vector g_k + multi-task heads
  6.  Loss aggregation lives in MAVTLoss (Σ_k α_k · L_task_k).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mavt.model.patchify import PatchifyEncoder
from mavt.model.backbone import HybridBackbone
from mavt.model.content_detail_split import ContentDetailSplit
from mavt.model.matryoshka_head import MatryoshkaHead
from mavt.model.decoder import AsymmetricDecoder


# Compression ratios per modality (content, detail)
_MODALITY_RATIOS = {
    'image':  (0.25, 0.10),
    'video':  (0.25, 0.10),
    'threed': (0.35, 0.15),
}


@dataclass
class MAVTOutput:
    """Per-prefix outputs of the Matryoshka head plus shared metrics."""

    # Reconstruction branch — only populated for prefixes in ``recon_prefixes``.
    reconstruction: Dict[int, torch.Tensor]
    # Understanding branch — populated for every prefix.
    z: Dict[int, torch.Tensor]
    mu: Dict[int, torch.Tensor]
    logvar: Dict[int, torch.Tensor]
    loss_kl: Dict[int, torch.Tensor]
    g: Dict[int, torch.Tensor]
    semantic: Dict[int, torch.Tensor]
    classification: Optional[Dict[int, torch.Tensor]]
    # Slot diversity / residual ratio metrics from C-D split.
    cd_metrics: Dict[str, torch.Tensor]
    # Which prefixes ran the decoder this forward pass.
    recon_prefixes: Tuple[int, ...]


class MAVT(nn.Module):
    """Full MAVT model with Matryoshka representation learning."""

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
        # Matryoshka
        matryoshka_dims: Optional[Sequence[int]] = None,
        latent_dim: int = 32,
        semantic_dim: int = 768,
        num_classes: Optional[int] = None,
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
        self.matryoshka_dims = self._normalise_mrl_dims(matryoshka_dims, embed_dim)

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
            dim=embed_dim, num_heads=num_slot_heads, num_slot_layers=num_slot_layers)

        # Stage 4 — Matryoshka head (per-prefix VAE + understanding pool)
        self.matryoshka_head = MatryoshkaHead(
            dims=self.matryoshka_dims,
            latent_dim=latent_dim,
            semantic_dim=semantic_dim,
            num_classes=num_classes,
        )

        # Stage 5a — Reconstruction decoder shared across prefixes (input is
        # always ``latent_dim``-d after the per-prefix VAE).
        self.decoder = AsymmetricDecoder(
            latent_dim=latent_dim, dec_dim=dec_dim,
            num_attn_blocks=num_dec_attn_blocks, num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_mrl_dims(
        dims: Optional[Sequence[int]],
        embed_dim: int,
    ) -> Tuple[int, ...]:
        """Validate the Matryoshka prefix list. Last entry must equal ``embed_dim``."""
        if dims is None:
            return (embed_dim,)
        cleaned = sorted({int(d) for d in dims})
        if not cleaned:
            return (embed_dim,)
        invalid = [d for d in cleaned if d <= 0 or d > embed_dim]
        if invalid:
            raise ValueError(
                f"matryoshka_dims must be in [1, embed_dim={embed_dim}], got {invalid}"
            )
        if cleaned[-1] != embed_dim:
            cleaned.append(embed_dim)
        return tuple(cleaned)

    def _grid_shape(self, modality: str, x: torch.Tensor) -> tuple:
        if modality == 'image':
            _, _, H, W = x.shape
            return (H // self.patch_size, W // self.patch_size)
        if modality == 'video':
            _, _, T, H, W = x.shape
            return (T // 2, H // self.patch_size, W // self.patch_size)
        if modality == 'threed':
            S = x.shape[-1]
            return (S // self.patch_size, S // self.patch_size)
        raise ValueError(modality)

    def _resolve_recon_prefixes(
        self, recon_prefixes: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        if recon_prefixes is None:
            return self.matryoshka_dims
        chosen = tuple(int(d) for d in recon_prefixes)
        unknown = [d for d in chosen if d not in self.matryoshka_dims]
        if unknown:
            raise ValueError(
                f"recon_prefixes {unknown} not in matryoshka_dims={self.matryoshka_dims}"
            )
        return chosen

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        decode: bool = True,
        recon_prefixes: Optional[Sequence[int]] = None,
    ) -> MAVTOutput:
        """Run patchify → backbone → C-D split → Matryoshka head, then
        optionally decode a chosen subset of prefixes (default: all)."""
        grid_shape = self._grid_shape(modality, x)

        tokens, positions, plane_ids = self.patchify(x, modality)
        features = self.backbone(tokens, positions, plane_ids, modality)

        content_ratio, detail_ratio = _MODALITY_RATIOS[modality]
        compressed, cd_metrics = self.cd_split(features, content_ratio, detail_ratio)

        mrl_out = self.matryoshka_head(compressed)

        chosen_prefixes = self._resolve_recon_prefixes(recon_prefixes) if decode else ()
        recon: Dict[int, torch.Tensor] = {}
        for d in chosen_prefixes:
            recon[d] = self.decoder(
                mrl_out[d]['z'], positions, modality, grid_shape
            )

        return MAVTOutput(
            reconstruction=recon,
            z={d: mrl_out[d]['z']      for d in self.matryoshka_dims},
            mu={d: mrl_out[d]['mu']     for d in self.matryoshka_dims},
            logvar={d: mrl_out[d]['logvar'] for d in self.matryoshka_dims},
            loss_kl={d: mrl_out[d]['kl']    for d in self.matryoshka_dims},
            g={d: mrl_out[d]['g']      for d in self.matryoshka_dims},
            semantic={d: mrl_out[d]['sem']    for d in self.matryoshka_dims},
            classification=(
                {d: mrl_out[d]['cls'] for d in self.matryoshka_dims}
                if self.matryoshka_head.cls_heads is not None else None
            ),
            cd_metrics=cd_metrics,
            recon_prefixes=tuple(chosen_prefixes),
        )

    def encode(
        self, x: torch.Tensor, modality: str, prefix: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: return ``(z_d, semantic_d)`` for the requested prefix."""
        out = self.forward(x, modality, decode=False)
        d = prefix if prefix is not None else self.matryoshka_dims[-1]
        if d not in out.z:
            raise ValueError(f"prefix {d} not in matryoshka_dims={self.matryoshka_dims}")
        return out.z[d], out.semantic[d]

    def load_siglip2_weights(self, model_name: str = "google/siglip2-base-patch16-224",
                              freeze_stages: int = 10) -> None:
        self.backbone.load_siglip2_weights(model_name, freeze_stages)

    # ------------------------------------------------------------------ #
    #  Eager pre-creation of slot poolers                                 #
    # ------------------------------------------------------------------ #

    def prepare_for_modalities(self, specs: Iterable[Dict[str, Any]]) -> None:
        """Pre-create every SlotPooler the trainer will need so its params
        are picked up by ``configure_optimizers``."""
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
                N  = 3 * Sp * Sp
            else:
                raise ValueError(f"Unknown modality in spec: {modality!r}")
            c_r, d_r = _MODALITY_RATIOS[modality]
            N_c = max(1, int(N * c_r))
            N_d = max(1, int(N * d_r))
            self.cd_split.prepare_poolers(N_c, N_d)
