from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class PatchifyConfig:
    patch_size_spatial: int = 16
    patch_size_temporal: int = 2
    embed_dim: int = 1152  # SigLIP2-SO400M output dim
    siglip2_model: str = "google/siglip2-so400m-patch16-384"
    freeze_siglip2: bool = True


@dataclass
class STFConfig:
    wavelet: str = "haar"
    dwt_levels: int = 2  # 7 spatial freq features
    stft_window: int = 4
    stft_hop: int = 2
    stft_n_fft: int = 4  # 3 positive freq bins
    d_freq_embed: int = 128
    # Derived
    n_spatial_feats: int = 7   # 1 + 3*dwt_levels
    n_temporal_feats: int = 4  # mean_DC, mean_F1, mean_F2, var_DC
    n_depth_feats: int = 4     # mean_DC, mean_F1, mean_F2, max_F2
    freq_raw_dim: int = 15     # 7 + 4 + 4


@dataclass
class GraphConfig:
    k: int = 9               # k-NN neighbors per node
    alpha_init: float = 0.3  # spatial similarity weight
    beta_init: float = 0.4   # freq similarity weight
    gamma_init: float = 0.3  # feature similarity weight
    sigma_init: float = 1.0  # Gaussian RBF bandwidth


@dataclass
class PosEncConfig:
    n_eigvecs: int = 32      # top-k Laplacian eigenvectors
    d_spectral: int = 128    # spectral PE output dim
    rope_base: float = 10000.0


@dataclass
class EncoderConfig:
    d_model: int = 1152
    # CNN stages (1-2): high-res spatial processing
    cnn_channels: Tuple[int, ...] = (256, 512)
    # Mamba stages (3-4)
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    # Spectral Graph Conv
    cheby_order: int = 3     # Chebyshev polynomial order
    # Titans memory (stage 4)
    memory_slots: int = 2048
    memory_dim: int = 256
    # Number of blocks per stage
    n_blocks_stage3: int = 6
    n_blocks_stage4: int = 6


@dataclass
class LatentConfig:
    d_encoder: int = 1152
    latent_dim: int = 32        # z dimension
    d_understand: int = 768     # understanding path (CLS token)
    noise_std: float = 0.01     # regularization noise
    kl_weight: float = 1e-4


@dataclass
class DecoderConfig:
    latent_dim: int = 32
    d_model: int = 768
    # Attention stage
    n_attn_blocks: int = 4
    n_attn_heads: int = 12
    # Mamba + Graph stage
    n_mamba_blocks: int = 4
    # CNN upsampling stage
    cnn_channels: Tuple[int, ...] = (512, 256, 128, 64)
    out_channels: int = 3
    patch_size: int = 16


@dataclass
class MAVTConfig:
    """Single source of truth for all MAVT hyperparameters."""

    modality: Literal["image", "video", "3d", "auto"] = "auto"

    patchify: PatchifyConfig = field(default_factory=PatchifyConfig)
    stf: STFConfig = field(default_factory=STFConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    pos_enc: PosEncConfig = field(default_factory=PosEncConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Training flags
    use_ema: bool = True
    gradient_checkpointing: bool = False

    @property
    def d_model(self) -> int:
        return self.patchify.embed_dim


__all__ = [
    "MAVTConfig",
    "PatchifyConfig",
    "STFConfig",
    "GraphConfig",
    "PosEncConfig",
    "EncoderConfig",
    "LatentConfig",
    "DecoderConfig",
]
