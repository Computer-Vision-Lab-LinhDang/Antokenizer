"""DDTokenizer: full DDT pipeline orchestrator."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import DDTConfig
from .decoder import AsymmetricDecoder
from .encoder import DDTEncoder
from .latent import ContinuousLatentProjection
from .patchify import Unified4DPatchify
from .types import DecoderOutput, EncoderOutput, LatentOutput, Modality, WaveletOutput
from .wavelet_enrich import WaveletEnrich


class DDTokenizer(nn.Module):
    """Dual-Domain Tokenizer: SigLIP2 semantic + wavelet frequency enrichment.

    Usage:
        model = DDTokenizer()
        dec_out, lat_out = model(image)
        z = model.encode(image)
    """

    def __init__(self, cfg: Optional[DDTConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DDTConfig()

        self.patchify = Unified4DPatchify(self.cfg.patchify)
        self.wavelet = WaveletEnrich(self.cfg.wavelet, d_model=self.cfg.d_model)
        self.encoder = DDTEncoder(self.cfg.encoder)
        self.latent = ContinuousLatentProjection(self.cfg.latent)
        self.decoder = AsymmetricDecoder(self.cfg.decoder)

    def encode(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> LatentOutput:
        wav_out = self._run_encoder_pipeline(x, modality)
        enc_out = self.encoder(wav_out)
        return self.latent(enc_out)

    def forward(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> tuple[DecoderOutput, LatentOutput]:
        patch_out = self.patchify(x, modality)
        wav_out = self.wavelet(patch_out)
        enc_out = self.encoder(wav_out)
        lat_out = self.latent(enc_out)
        dec_out = self.decoder(
            lat_out, enc_out.positions_out,
            tuple(x.shape), modality or patch_out.modality,
        )
        return dec_out, lat_out

    def _run_encoder_pipeline(
        self, x: torch.Tensor, modality: Optional[Modality],
    ) -> WaveletOutput:
        patch_out = self.patchify(x, modality)
        return self.wavelet(patch_out)


__all__ = ["DDTokenizer"]
