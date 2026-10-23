"""Hyper-parameters of MAVT on DINOv3, v0.

v0 = frozen DINOv3 + pixel stem + plain trunk (R0) + C at keyframes / D every frame + query decoder.
v1 adds the track edge C_{k-K} -> C_k and the event memory S; both are flags here and
raise NotImplementedError until v1 is built.

Budget: image 256 x 48 = 12 288; video segment of K = 4 frames 256 x 40 + 4 x 16 x 32 = 12 288.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DinoV0Config:
    # ---- input -------------------------------------------------------------------------
    image_size: int = 256
    patch: int = 16                      # DINOv3 ViT-L/16 patch -> 16 x 16 grid at 256^2
    # ---- tier A: frozen encoder + pixel stem ---------------------------------------------
    dino_repo: Optional[str] = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # None -> tiny random stub (tests)
    dino_dim: int = 1024
    dino_registers: int = 4
    stub_dino_dim: int = 64              # width of the stub encoder used when dino_repo is None
    pix_dim: int = 256                   # pixel stem output channels per patch
    # ---- tier B: trunk -----------------------------------------------------------------
    dim: int = 1024
    heads: int = 16
    trunk_layers: int = 8
    mlp_ratio: float = 4.0
    temporal_window: int = 1             # radius of the (2r+1)^2 window each token reads at frame t-1
    rgat_relations: int = 0              # 0 = R0 plain attention; > 0 (e.g. 4) = R2 learned relations per edge
    rgat_rank: int = 32                  # rank of each relation's value transform U_r V_r^T (per head)
    rgat_entropy_w: float = 0.01         # weight of -H(batch-marginal of r): anti-collapse of relation usage
    rgat_tau: float = 1.0                # Gumbel-softmax temperature of the relation choice
    # ---- tier C: C / D -------------------------------------------------------------------
    keyframe_every: int = 4              # K
    c_blocks: int = 2                    # C_k: self-attn over C + cross-attn to X_k
    d_tokens: int = 16
    d_dim: int = 128                     # D inner width; Up -> dim when it enters the latent
    d_blocks: int = 2                    # D_t: attn over [X_t ; C_k ; D]
    use_track: bool = False              # v1: C_k reads C_{k-K} through a DINOv3 track edge
    use_event_memory: bool = False       # v1: S (KDA), only with clips >= 64 frames
    # ---- tier D: latent ------------------------------------------------------------------
    zr_c_image: int = 48                 # z_r channels of C for images  (256 x 48 = 12 288)
    zr_c_video: int = 40                 # z_r channels of C for video   (256 x 40)
    zr_d: int = 32                       # z_r channels of each D token  (16 x 32 per frame)
    zs_dim: int = 1024                   # understanding vector, matched to DINOv3 CLS
    # ---- tier E: decoder -----------------------------------------------------------------
    dec_dim: int = 768
    dec_heads: int = 12
    dec_blocks: int = 4
    read_window: int = 2                 # (2r+1)^2 = 5 x 5 cells around the shifted centre
    gate_init_bias: float = 4.0          # o = sigmoid(.) starts at ~0.98: read from C first
    train_queries_per_frame: int = 64    # random subset of the 256 queries per frame in training
    # ---- losses --------------------------------------------------------------------------
    w_rgb: float = 1.0
    w_lpips: float = 0.0                 # needs the lpips package; full-frame decode of one frame
    w_feat: float = 1.0
    w_fut: float = 0.5
    w_var: float = 0.1
    w_kl: float = 1e-6
    w_sem: float = 0.5
    w_gate: float = 0.01                 # keeps o from collapsing to 0 (D carrying everything)
    # ---- v1 phase 1 (defaults keep v0 behaviour, so v0 checkpoints load and evaluate unchanged) ----
    delta_mode: str = "mean"             # "mean": Δ from the mean of the D tokens (v0); "query": each query attends to D for its own Δ
    rgb_head: str = "linear"             # "linear": per-patch Linear (v0); "conv": full-grid conv upsampler (needs all N queries)
    conv_channels: tuple = (512, 256, 128, 64, 64)   # conv upsampler: width at grid resolution, then one per x2 stage
    conv_res_blocks: int = 2             # ResBlocks per conv stage
    gate_loss: str = "mean"              # "mean": (1 - o).mean() (v0, pushes o to 1); "hinge": relu(gate_floor - mean(o))
    gate_floor: float = 0.5
    w_warp: float = 0.0                  # warp-only loss: decode with the gate locked to 1 (C read at (u, v) + Δ only)
    forecast_horizons: tuple = (1,)      # D_hat forecasts frame t + h for each h (v0: t + 1 only)
    forecast_residual: bool = False      # D_hat(t + h) = D_t + f(D_t, C_k, h), f zero-init: starts as copy-last
    w_fut_rgb: float = 0.0               # rgb L1 on forecast frames
    w_gram: float = 0.0                  # Gram loss on the LPIPS VGG features (AToken-style)
    lpips_max_frames: int = 64           # frames per step that get LPIPS / Gram (random subset)
    extra: dict = field(default_factory=dict)

    @property
    def grid(self) -> int:
        return self.image_size // self.patch

    @property
    def n_patches(self) -> int:
        return self.grid * self.grid

    def budget(self, modality: str) -> int:
        """Latent values per image (image) or per K-frame segment (video)."""
        if modality == "image":
            return self.n_patches * self.zr_c_image
        return self.n_patches * self.zr_c_video + self.keyframe_every * self.d_tokens * self.zr_d


def tiny_config(**kw) -> DinoV0Config:
    """Small CPU config for unit tests: stub encoder, 64^2 images (4 x 4 grid)."""
    base = dict(dino_repo=None, image_size=64, stub_dino_dim=32, pix_dim=16, dim=64, heads=4,
                trunk_layers=2, c_blocks=1, d_tokens=4, d_dim=32, d_blocks=1, zr_c_image=8,
                zr_c_video=6, zr_d=4, zs_dim=32, dec_dim=64, dec_heads=4, dec_blocks=1,
                train_queries_per_frame=8)
    base.update(kw)
    return DinoV0Config(**base)
