# mavt/mavt_recon.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mavt.tokenizer import MAVTokenizer
from mavt.module8_recon import ReconstructionHead, ReconstructionCriterion


# Output container (match classifier style)

@dataclass
class MavtReconOutput:
    reconstruction: torch.Tensor
    aux: Optional[list[torch.Tensor]] = None


# Main Model

class MAVTRecon(nn.Module):
    """
    MAVT Reconstruction Model

    Pipeline:
        x → MAVTokenizer → coarse reconstruction
          → ReconstructionHead → refined reconstruction
          → ReconstructionCriterion (optional)

    Returns:
        (DecoderOutput, LatentOutput, logs_dict)
    """

    def __init__(
        self,
        tokenizer: Optional[MAVTokenizer] = None,
        *,
        lambda_l1: float = 1.0,
        lambda_lpips: float = 10.0,
        lambda_gram: float = 1e3,
        lambda_clip: float = 1.0,
    ) -> None:
        super().__init__()

        # Core tokenizer (FULL pipeline)
        self.tokenizer = tokenizer or MAVTokenizer()

        # Refinement head
        self.recon_head = ReconstructionHead(in_channels=3)

        # Loss
        self.criterion = ReconstructionCriterion(
            lambda_l1=lambda_l1,
            lambda_lpips=lambda_lpips,
            lambda_gram=lambda_gram,
            lambda_clip=lambda_clip,
        )

    # Forward

    def forward(
        self,
        x: torch.Tensor,
        *,
        compute_loss: bool = True,
    ) -> Tuple[MavtReconOutput, object, Dict[str, torch.Tensor]]:
        """
        Args:
            x:
                image → (B, C, H, W)
                video → (B, C, T, H, W)

        Returns:
            recon_out: MavtReconOutput
            latent_out: LatentOutput (from tokenizer)
            logs: dict (loss + metrics)
        """

        # ─── Core pipeline (correct usage) ───
        decoder_out, latent_out = self.tokenizer(x)

        coarse_recon = decoder_out.reconstruction  # (B, C, H, W) or video

        # ─── Refinement ───
        recon, recon_feats, target_feats = self.recon_head(
            coarse_recon,
            x if compute_loss else None,
        )

        logs: Dict[str, torch.Tensor] = {}
        total_loss = None

        # ─── Loss ───
        if compute_loss:
            total_loss, loss_logs = self.criterion(
                recon,
                x,
                recon_features=recon_feats,
                target_features=target_feats,
                num_patches=latent_out.z.shape[1],
            )
            logs.update(loss_logs)

        # ─── Metrics ───
        with torch.no_grad():
            metrics = self._compute_metrics(recon, x)
            logs.update(metrics)

        recon_out = MavtReconOutput(
            reconstruction=recon,
            aux=[coarse_recon],  # keep aux for compatibility/debug
        )

        return recon_out, latent_out, logs

    # Metrics

    def _compute_metrics(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Flatten video if needed
        if recon.dim() == 5:
            B, C, T, H, W = recon.shape
            recon = recon.view(B * T, C, H, W)
            target = target.view(B * T, C, H, W)

        mse = torch.mean((recon - target) ** 2)
        psnr = -10 * torch.log10(mse + 1e-8)
        l1 = torch.mean(torch.abs(recon - target))

        return {
            "psnr": psnr,
            "l1": l1,
        }

    # Inference API

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        recon_out, _, _ = self.forward(x, compute_loss=False)
        return recon_out.reconstruction

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        latent_out = self.tokenizer.encode(x)
        return latent_out.z

    @torch.no_grad()
    def decode(self, latent_out) -> torch.Tensor:
        """
        Decode from latent (for generative tasks later)
        """
        self.eval()

        decoder_out = self.tokenizer.decode(
            latent_out,
            positions=None,   # requires external handling if used standalone
            target_shape=None,
        )

        recon, _, _ = self.recon_head(decoder_out.reconstruction, None)
        return recon


__all__ = [
    "MAVTRecon",
    "MavtReconOutput",
]