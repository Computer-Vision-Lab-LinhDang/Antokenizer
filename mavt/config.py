from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class PatchifyConfig:
    patch_size_spatial: int = 16
    patch_size_temporal: int = 2
    embed_dim: int = 1152
    siglip2_model: str = "google/siglip2-so400m-patch16-384"
    freeze_siglip2: bool = True


@dataclass
class EncoderConfig:
    d_model: int = 1152
    n_blocks_stage3: int = 6
    n_blocks_stage4: int = 6


@dataclass
class LatentConfig:
    d_encoder: int = 1152
    latent_dim: int = 32
    d_understand: int = 768
    noise_std: float = 0.01
    kl_weight: float = 1e-4


@dataclass
class DecoderConfig:
    latent_dim: int = 32
    d_model: int = 768
    n_attn_blocks: int = 4
    n_attn_heads: int = 12
    cnn_channels: Tuple[int, ...] = (512, 256, 128, 64)
    out_channels: int = 3
    patch_size: int = 16
    dropout: float=0.2


@dataclass
class MAVTConfig:
    modality: Literal["image", "video", "3d", "auto"] = "auto"

    patchify: PatchifyConfig = field(default_factory=PatchifyConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    use_ema: bool = True
    gradient_checkpointing: bool = False

    @property
    def d_model(self) -> int:
        return self.patchify.embed_dim


__all__ = [
    "MAVTConfig",
    "PatchifyConfig",
    "EncoderConfig",
    "LatentConfig",
    "DecoderConfig",
]
