# mavt/mavt_recon.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mavt.config import MAVTConfig
from mavt.module1_patchify import Unified4DPatchify
from mavt.module2_stf import SpaceTimeFrequencyTransform
from mavt.module3_graph import FrequencyInformed4DGraphBuilder
from mavt.module4_pos_enc import SpectralPositionEncoding
from mavt.module5_encoder import MAVTEncoder
from mavt.module6_latent import ContinuousLatentProjection
from mavt.module8_recon import ReconstructionDecoder
from mavt.types import (
    Modality,
    PatchifyOutput,
    STFOutput,
    GraphOutput,
    PosEncOutput,
    EncoderOutput,
    LatentOutput,
    DecoderOutput,
)
from mavt.module8_recon import ReconstructionHead, ReconstructionCriterion


class MAVTRecon(nn.Module):
    """
    """
    def __init__(self, cfg: Optional[MAVTConfig]) -> None:
        super().__init__()
        self.cfg = cfg or MAVTConfig()

        self.patchify = Unified4DPatchify(self.cfg.patchify)
        self.stf = SpaceTimeFrequencyTransform(self.cfg.stf)
        self.graph = FrequencyInformed4DGraphBuilder(self.cfg.graph)
        self.pos_enc = SpectralPositionEncoding(
            self.cfg.pos_enc,
            d_model=self.cfg.d_model,
            d_freq=self.cfg.stf.d_freq_embed,
        )
        self.encoder   = MAVTEncoder(self.cfg.encoder)
        self.latent = ContinuousLatentProjection(self.cfg.latent)
        self.decoder = ReconstructionDecoder(self.cfg.decoder)

        self._memory_cache: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, modality: Optional[Modality]) -> DecoderOutput:
        pass

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pass

    def _reset_memory(self) -> None:
        self._memory_cache = None


__all__ = [
    "MAVTRecon",
    "DecoderOutput",
]