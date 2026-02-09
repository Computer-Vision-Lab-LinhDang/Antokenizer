"""Discrete diffusion modules for AToken generator.

This module implements a dual-branch discrete diffusion model:
- Semantic branch: MaskGIT-style masked modeling on semantic tokens (zC)
- Detail branch: D3PM discrete diffusion on detail tokens (zD)

Key components:
- D3PM: Core discrete diffusion process with absorbing/uniform transitions
- TransitionMatrix: Manages discrete token transition probabilities
- NoiseSchedule: Beta schedules (cosine, linear, sigmoid)
- SemanticBranch: Parallel decoding for semantic token completion
- DetailBranch: D3PM denoiser for high-frequency detail synthesis
- DiffusionGenerator: Unified generator combining both branches

Example:
    >>> from atoken.diffusion import DiffusionGenerator
    >>> generator = DiffusionGenerator(d_model=768, detail_timesteps=50)
    >>> output = generator.generate(degraded_input, degradation_params)
"""

from __future__ import annotations

from .conditioning import (
    AdaLayerNorm,
    AdaLayerNormZero,
    CrossAttentionConditioner,
    FiLMConditioner,
    GatedCrossAttention,
    SinusoidalPositionEmbedding,
    TimestepEmbedding,
)
from .d3pm import D3PM
from .detail_branch import DetailBranch, DetailTransformerBlock
from .generator import DiffusionGenerator
from .quantizer import EMAVectorQuantizer, ResidualVectorQuantizer, VectorQuantizer
from .schedule import NoiseSchedule
from .semantic_branch import SemanticBranch, SemanticTransformerBlock
from .shallow_encoder import DegradationEncoder, ShallowFeatureEncoder, ShallowViTEncoder
from .transition import TransitionMatrix

__all__ = [
    # Core D3PM
    "D3PM",
    "TransitionMatrix",
    "NoiseSchedule",
    # Conditioning
    "AdaLayerNorm",
    "AdaLayerNormZero",
    "CrossAttentionConditioner",
    "FiLMConditioner",
    "GatedCrossAttention",
    "SinusoidalPositionEmbedding",
    "TimestepEmbedding",
    # Branches
    "SemanticBranch",
    "SemanticTransformerBlock",
    "DetailBranch",
    "DetailTransformerBlock",
    # Encoders
    "ShallowFeatureEncoder",
    "ShallowViTEncoder",
    "DegradationEncoder",
    # Generator
    "DiffusionGenerator",
    # Quantization
    "VectorQuantizer",
    "EMAVectorQuantizer",
    "ResidualVectorQuantizer",
]
