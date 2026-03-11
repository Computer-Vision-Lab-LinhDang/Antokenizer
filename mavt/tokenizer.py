"""MAVTokenizer: full MAVT pipeline orchestrator."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import MAVTConfig
from .module1_patchify import Unified4DPatchify
from .module2_stf import SpaceTimeFrequencyTransform
from .module3_graph import FrequencyInformed4DGraphBuilder
from .module4_pos_enc import SpectralPositionEncoding
from .module5_encoder import MAVTEncoder
from .module6_latent import ContinuousLatentProjection
from .module7_decoder import AsymmetricDecoder
from .types import (
    DecoderOutput,
    EncoderOutput,
    GraphOutput,
    LatentOutput,
    Modality,
    PatchifyOutput,
    PosEncOutput,
    STFOutput,
)


class MAVTokenizer(nn.Module):
    """Memory-Augmented Vision Tokenizer.

    Usage:
        model = MAVTokenizer()
        latent = model.encode(image)           # (B, N, 32) + (B, 768)
        recon  = model(image).reconstruction   # (B, 3, H, W)
        losses = model.compute_loss(image, recon_out, latent_out)
    """

    def __init__(self, cfg: Optional[MAVTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MAVTConfig()

        self.patchify  = Unified4DPatchify(self.cfg.patchify)
        self.stf       = SpaceTimeFrequencyTransform(self.cfg.stf)
        self.graph     = FrequencyInformed4DGraphBuilder(self.cfg.graph)
        self.pos_enc   = SpectralPositionEncoding(
            self.cfg.pos_enc,
            d_model=self.cfg.d_model,
            d_freq=self.cfg.stf.d_freq_embed,
        )
        self.encoder   = MAVTEncoder(self.cfg.encoder)
        self.latent    = ContinuousLatentProjection(self.cfg.latent)
        self.decoder   = AsymmetricDecoder(self.cfg.decoder)

        self._memory_state: Optional[torch.Tensor] = None

    # ─── public API ──────────────────────────────────────────────────────────

    def encode(
        self,
        x: torch.Tensor,
        modality: Optional[Modality] = None,
    ) -> LatentOutput:
        """Input → continuous latent (z, z_understand)."""
        _, _, _, _, enc_out = self._run_encoder(x, modality)
        return self.latent(enc_out)

    def decode(
        self,
        latent_out: LatentOutput,
        positions: torch.Tensor,
        target_shape: tuple[int, ...],
        modality: Modality = "image",
    ) -> DecoderOutput:
        return self.decoder(latent_out, positions, target_shape, modality)

    def forward(
        self,
        x: torch.Tensor,
        modality: Optional[Modality] = None,
    ) -> tuple[DecoderOutput, LatentOutput]:
        """Full autoencoder round-trip. Returns (decoder_out, latent_out)."""
        patch_out, _, _, _, enc_out = self._run_encoder(x, modality)
        latent_out = self.latent(enc_out)
        decoder_out = self.decoder(
            latent_out,
            enc_out.positions_out,
            tuple(x.shape),
            modality or patch_out.modality,
        )
        return decoder_out, latent_out

    def compute_loss(
        self,
        x: torch.Tensor,
        decoder_out: DecoderOutput,
        latent_out: LatentOutput,
        kl_weight: float = 1e-4,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction + KL loss.

        Returns dict with keys: recon_loss, kl_loss, total_loss.
        """
        recon = decoder_out.reconstruction

        # Handle shape mismatch for video (5D) vs image (4D)
        target = x
        if target.dim() == 5 and recon.dim() == 5:
            # Both video — truncate T if needed
            T_min = min(target.shape[2], recon.shape[2])
            target = target[:, :, :T_min]
            recon  = recon[:, :, :T_min]
        elif target.dim() == 5 and recon.dim() == 4:
            # Video input but image recon (3D path or mismatch)
            target = target[:, :, 0]  # use first frame as target

        recon_loss = torch.nn.functional.l1_loss(recon, target)
        kl_loss = self.latent.kl_loss(latent_out)
        total = recon_loss + kl_weight * kl_loss

        return {
            "recon_loss": recon_loss,
            "kl_loss":    kl_loss,
            "total_loss": total,
        }

    def reset_memory(self) -> None:
        self._memory_state = None

    # ─── internal ────────────────────────────────────────────────────────────

    def _run_encoder(
        self,
        x: torch.Tensor,
        modality: Optional[Modality],
    ) -> tuple[PatchifyOutput, STFOutput, GraphOutput, PosEncOutput, EncoderOutput]:
        patch_out  = self.patchify(x, modality)
        stf_out    = self.stf(patch_out)
        graph_out  = self.graph(patch_out, stf_out)
        pos_out    = self.pos_enc(graph_out, stf_out)
        enc_out    = self.encoder(patch_out, pos_out, graph_out, self._memory_state)
        return patch_out, stf_out, graph_out, pos_out, enc_out


__all__ = ["MAVTokenizer"]
