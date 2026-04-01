"""MAVTokenizer: MAVT pipeline orchestrator.

Pipeline:
  Input → Patchify → [UnifiedContentDetailSplit] → Encoder → Latent → Decoder
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import MAVTConfig
from .module1_patchify import Unified4DPatchify
from .module5_encoder import MAVTEncoder
from .module6_latent import ContinuousLatentProjection
from .module7_decoder import AsymmetricDecoder
from .types import (
    DecoderOutput,
    LatentOutput,
    Modality,
    PatchifyOutput,
)


class MAVTokenizer(nn.Module):
    """Memory-Augmented Vision Tokenizer.

    When ``content_dynamics`` is set in the config, the unified
    Content-Detail Split is applied to **all** modalities — no
    modality-specific branching.
    """

    def __init__(self, cfg: Optional[MAVTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MAVTConfig()

        self.patchify = Unified4DPatchify(self.cfg.patchify)
        self.encoder = MAVTEncoder(self.cfg.encoder)
        self.latent = ContinuousLatentProjection(self.cfg.latent)

        # Unified Content-Detail Split (optional — active for ALL modalities).
        cd_cfg = self.cfg.content_dynamics
        self.cd_split = None
        if cd_cfg is not None and cd_cfg.enabled:
            from .content_dynamics import UnifiedContentDetailSplit

            self.cd_split = UnifiedContentDetailSplit(cd_cfg)

        self.decoder = AsymmetricDecoder(self.cfg.decoder, cd_cfg=cd_cfg)

    # ── public API ───────────────────────────────────────────────────

    def encode(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> LatentOutput:
        patch_out = self.patchify(x, modality)
        enc_input = self._maybe_cd_split(patch_out)
        return self.latent(self.encoder(enc_input))

    def forward(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> tuple[DecoderOutput, LatentOutput]:
        patch_out = self.patchify(x, modality)
        mod = modality or patch_out.modality

        enc_input, cd_metadata = self._maybe_cd_split_meta(patch_out)
        enc_out = self.encoder(enc_input)
        latent_out = self.latent(enc_out)

        decoder_out = self.decoder(
            latent_out, enc_out.positions_out,
            tuple(x.shape), mod, cd_metadata=cd_metadata,
        )
        return decoder_out, latent_out

    def compute_loss(
        self,
        x: torch.Tensor,
        decoder_out: DecoderOutput,
        latent_out: LatentOutput,
        kl_weight: float = 1e-4,
    ) -> dict[str, torch.Tensor]:
        recon = decoder_out.reconstruction
        target = x

        if target.dim() == 5 and recon.dim() == 5:
            T_tgt, T_rec = target.shape[2], recon.shape[2]
            if T_tgt > T_rec > 0:
                stride = T_tgt // T_rec
                target = target[:, :, ::stride][:, :, :T_rec]
            elif T_rec > T_tgt:
                recon = recon[:, :, :T_tgt]
        elif target.dim() == 5 and recon.dim() == 4:
            target = target[:, :, 0]

        recon_loss = torch.nn.functional.l1_loss(recon, target)
        kl_loss = self.latent.kl_loss(latent_out)
        total = recon_loss + kl_weight * kl_loss

        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total,
        }

    # ── internal helpers ─────────────────────────────────────────────

    def _maybe_cd_split(self, patch_out):
        """Apply unified Content-Detail Split when configured."""
        if self.cd_split is not None:
            cd_out = self.cd_split(patch_out)
            return PatchifyOutput(
                f_spatial=cd_out.tokens,
                positions=cd_out.positions,
                modality=patch_out.modality,
            )
        return patch_out

    def _maybe_cd_split_meta(self, patch_out):
        """Like :meth:`_maybe_cd_split` but also returns *cd_metadata*."""
        if self.cd_split is not None:
            cd_out = self.cd_split(patch_out)
            enc_input = PatchifyOutput(
                f_spatial=cd_out.tokens,
                positions=cd_out.positions,
                modality=patch_out.modality,
            )
            # Pass-through (small inputs) returns empty cd_metadata → treat as None.
            meta = cd_out.cd_metadata if cd_out.cd_metadata else None
            return enc_input, meta
        return patch_out, None


__all__ = ["MAVTokenizer"]
