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
from .module8_cls import LinearClassifier, FastFormerClassifier
from .types import (
    DecoderOutput,
    EncoderOutput,
    LatentOutput,
    Modality,
    PatchifyOutput,
)


class MAVTClassifier(nn.Module):
    """Memory-Augmented Vision Tokenizer.

    Usage:
        model = MAVTokenizer()
        latent = model.encode(image)
        recon  = model(image).reconstruction
    """

    def __init__(self, num_classes: int, classifier_name: str, cfg: Optional[MAVTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MAVTConfig()

        self.patchify = Unified4DPatchify(self.cfg.patchify)
        self.encoder = MAVTEncoder(self.cfg.encoder)
        self.latent = ContinuousLatentProjection(self.cfg.latent)
        self.decoder = AsymmetricDecoder(self.cfg.decoder)
        if classifier_name == "linear":
            self.classifier = LinearClassifier(num_classes=num_classes, in_channel=self.cfg.latent.d_understand)
        else:
            self.classifier = FastFormerClassifier(
                num_classes=num_classes, in_channel=self.cfg.latent.d_understand, out_channel=self.cfg.latent.latent_dim,
                latent_dim=self.cfg.latent.latent_dim, num_heads=8, num_blocks=2
            )

    def encode(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> LatentOutput:
        patch_out, enc_out = self._run_encoder(x, modality)
        return self.latent(enc_out)
    
    def classify(
        self,
        latent_out: LatentOutput
    ):
        return self.classifier.forward(latent_output=latent_out)

    def forward(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> tuple[DecoderOutput, LatentOutput]:
        patch_out, enc_out = self._run_encoder(x, modality)
        latent_out = self.latent(enc_out)
        decoder_out = self.decoder(
            latent_out, enc_out.positions_out,
            tuple(x.shape), modality or patch_out.modality,
        )
        cls_out = self.classify(latent_out=latent_out)
        return decoder_out, latent_out, cls_out

    def compute_loss(
        self,
        x: torch.Tensor,
        decoder_out: DecoderOutput,
        latent_out: LatentOutput,
        cls_out,
        cls_target,
        cls_weight: float = 1e-4,
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

        # Multi-label (float target) vs single-label (long target)
        if cls_target.dtype == torch.float32:
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_out, cls_target)
        else:
            cls_loss = torch.nn.functional.cross_entropy(cls_out, cls_target)

        total = (recon_loss + kl_weight * kl_loss) * 0.5 + cls_weight * cls_loss

        return {
            "recon_loss": recon_loss,
            "kl_loss":    kl_loss,
            "cls_loss": cls_loss,
            "total_loss": total,
        }

    def _run_encoder(
        self, x: torch.Tensor, modality: Optional[Modality],
    ) -> tuple[PatchifyOutput, EncoderOutput]:
        patch_out = self.patchify(x, modality)
        enc_out = self.encoder(patch_out)
        return patch_out, enc_out


__all__ = ["MAVTokenizer"]
