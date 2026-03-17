"""MAVT: Memory-Augmented Vision Tokenizer.

Architecture: 7-module pipeline
  Module 1: Unified4DPatchify  — image/video/3D → 4D tokens + SigLIP2 embeddings
  Module 2: SpaceTimeFrequencyTransform — DWT + STFT frequency profiles
  Module 3: FrequencyInformed4DGraphBuilder — sparse k-NN token graph
  Module 4: SpectralPositionEncoding — Laplacian PE + 4D RoPE + freq embed
  Module 5: MAVTEncoder — CNN + MambaVision + SpectralGNN + Titans memory
  Module 6: ContinuousLatentProjection — continuous VAE-style latent (no VQ)
  Module 7: AsymmetricDecoder — Attention + Mamba + CNN → reconstruction

Entry point: MAVTokenizer composes all modules.
Config:      MAVTConfig controls all hyperparameters.
Types:       PatchifyOutput, STFOutput, GraphOutput, etc. define module I/O.
"""
from .config import (
    MAVTConfig,
    DecoderConfig,
    EncoderConfig,
    GraphConfig,
    LatentConfig,
    PatchifyConfig,
    PosEncConfig,
    STFConfig,
)
from .module1_patchify import SigLIP2PatchEmbed, Unified4DPatchify
from .module2_stf import HaarDWT2D, SpaceTimeFrequencyTransform
from .module3_graph import FrequencyInformed4DGraphBuilder
from .module4_pos_enc import SignNet, SpectralPositionEncoding
from .module5_encoder import ChebyshevGraphConv, MAVTEncoder, MambaVisionMixer, TitansMemory
from .module6_latent import ContinuousLatentProjection, VariationalProjection
from .module7_decoder import AsymmetricDecoder, PixelShuffleUpsample
from .tokenizer import MAVTokenizer
from .mavt_cls import MAVTClassifier
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

__all__ = [
    # Tokenizer (entry point)
    "MAVTokenizer",
    "MAVTClassifier",
    # Config
    "MAVTConfig",
    "PatchifyConfig",
    "STFConfig",
    "GraphConfig",
    "PosEncConfig",
    "EncoderConfig",
    "LatentConfig",
    "DecoderConfig",
    # Modules
    "SigLIP2PatchEmbed",
    "Unified4DPatchify",
    "HaarDWT2D",
    "SpaceTimeFrequencyTransform",
    "FrequencyInformed4DGraphBuilder",
    "SpectralPositionEncoding",
    "SignNet",
    "MAVTEncoder",
    "ChebyshevGraphConv",
    "MambaVisionMixer",
    "TitansMemory",
    "ContinuousLatentProjection",
    "VariationalProjection",
    "AsymmetricDecoder",
    "PixelShuffleUpsample",
    # I/O types
    "Modality",
    "PatchifyOutput",
    "STFOutput",
    "GraphOutput",
    "PosEncOutput",
    "EncoderOutput",
    "LatentOutput",
    "DecoderOutput",
]
