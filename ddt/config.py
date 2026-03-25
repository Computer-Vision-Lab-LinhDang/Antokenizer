from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class PatchifyConfig:
    patch_size_spatial: int = 16
    patch_size_temporal: int = 2
    embed_dim: int = 1152  # SigLIP2-SO400M
    siglip2_model: str = "google/siglip2-so400m-patch16-384"
    freeze_siglip2: bool = True


@dataclass
class WaveletConfig:
    dwt_levels: int = 2
    d_freq: int = 64       # frequency embedding dim before projection to d_model
    max_keep_ratio: float = 0.3   # keep at most 30% of dynamics patches per frame
    min_keep_ratio: float = 0.05  # keep at least 5%


@dataclass
class EncoderConfig:
    d_model: int = 1152
    n_blocks: int = 12
    dropout: float = 0.0


@dataclass
class LatentConfig:
    d_encoder: int = 1152
    latent_dim: int = 32
    d_understand: int = 768
    noise_std: float = 0.01


@dataclass
class DecoderConfig:
    latent_dim: int = 32
    d_model: int = 768
    n_attn_blocks: int = 4
    n_attn_heads: int = 12
    cnn_channels: Tuple[int, ...] = (512, 256, 128, 64)
    out_channels: int = 3
    patch_size: int = 16


@dataclass
class LossConfig:
    w_pixel: float = 1.0
    w_structure: float = 1.0
    w_detail: float = 2.0
    w_kl: float = 1e-4
    w_lpips: float = 0.1
    use_lpips: bool = False


@dataclass
class DDTConfig:
    modality: Literal["image", "video", "3d", "auto"] = "auto"

    patchify: PatchifyConfig = field(default_factory=PatchifyConfig)
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    use_ema: bool = True
    gradient_checkpointing: bool = False

    @property
    def d_model(self) -> int:
        return self.patchify.embed_dim


__all__ = [
    "DDTConfig",
    "PatchifyConfig",
    "WaveletConfig",
    "EncoderConfig",
    "LatentConfig",
    "DecoderConfig",
    "LossConfig",
]
