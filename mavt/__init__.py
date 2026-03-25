"""MAVT: Memory-Augmented Vision Tokenizer.

Simplified pipeline:
  Module 1: Unified4DPatchify  — image/video/3D → 4D tokens + SigLIP2 embeddings
  Module 5: MAVTEncoder — Transformer + 4D RoPE
  Module 6: ContinuousLatentProjection — continuous VAE-style latent
  Module 7: AsymmetricDecoder — Attention + CNN → reconstruction

Entry point: MAVTokenizer composes all modules.
"""
from .config import (
    MAVTConfig,
    DecoderConfig,
    EncoderConfig,
    LatentConfig,
    PatchifyConfig,
)
from .module1_patchify import SigLIP2PatchEmbed, Unified4DPatchify
from .module5_encoder import MAVTEncoder
from .module6_latent import ContinuousLatentProjection, VariationalProjection
from .module7_decoder import AsymmetricDecoder, PixelShuffleUpsample
from .tokenizer import MAVTokenizer
from .types import (
    DecoderOutput,
    EncoderOutput,
    LatentOutput,
    Modality,
    PatchifyOutput,
)
from .mavt_cls import MAVTClassifier

__all__ = [
    "MAVTokenizer",
    "MAVTConfig",
    "MAVTClassifier",
    "PatchifyConfig",
    "EncoderConfig",
    "LatentConfig",
    "DecoderConfig",
    "SigLIP2PatchEmbed",
    "Unified4DPatchify",
    "MAVTEncoder",
    "ContinuousLatentProjection",
    "VariationalProjection",
    "AsymmetricDecoder",
    "PixelShuffleUpsample",
    "Modality",
    "PatchifyOutput",
    "EncoderOutput",
    "LatentOutput",
    "DecoderOutput",
]
