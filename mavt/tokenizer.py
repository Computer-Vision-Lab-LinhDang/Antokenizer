"""MAVTokenizer: MAVT pipeline orchestrator.

Simplified pipeline (graph, STF, spectral PE removed):
  Input → Patchify → Encoder (Transformer + 4D RoPE) → Latent → Decoder
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
    EncoderOutput,
    LatentOutput,
    Modality,
    PatchifyOutput,
)


class MAVTokenizer(nn.Module):
    """Memory-Augmented Vision Tokenizer.

    Usage:
        model = MAVTokenizer()
        latent = model.encode(image)
        recon  = model(image).reconstruction
    """

    def __init__(self, cfg: Optional[MAVTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MAVTConfig()

        self.patchify = Unified4DPatchify(self.cfg.patchify)
        self.encoder = MAVTEncoder(self.cfg.encoder)
        self.latent = ContinuousLatentProjection(self.cfg.latent)
        self.decoder = AsymmetricDecoder(self.cfg.decoder)

    def encode(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> LatentOutput:
        patch_out, enc_out = self._run_encoder(x, modality)
        return self.latent(enc_out)

    def forward(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> tuple[DecoderOutput, LatentOutput]:
        patch_out, enc_out = self._run_encoder(x, modality)
        latent_out = self.latent(enc_out)
        decoder_out = self.decoder(
            latent_out, enc_out.positions_out,
            tuple(x.shape), modality or patch_out.modality,
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
            T_min = min(target.shape[2], recon.shape[2])
            target, recon = target[:, :, :T_min], recon[:, :, :T_min]
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

    def _run_encoder(
        self, x: torch.Tensor, modality: Optional[Modality],
    ) -> tuple[PatchifyOutput, EncoderOutput]:
        patch_out = self.patchify(x, modality)
        enc_out = self.encoder(patch_out)
        return patch_out, enc_out


__all__ = ["MAVTokenizer"]
