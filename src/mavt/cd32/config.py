"""Hyper-parameters of the C-D tokenizer, version 3.2, with the two-stream backbone.

C_k   = (c_k, rho_k, f_k) at every keyframe (every K frames): content c, log inverse depth rho per cell,
        one focal length f (estimated only when the camera intrinsics are unknown).
D_t   = (D_geom_t, D_innov_t[:L]) at the other frames: pose xi (6) + coarse 3D flow M (g x g x 3),
        then a variable-length prefix of residual tokens (nested dropout).
link  = 7 numbers per keyframe k >= 1: pose of keyframe k against keyframe k - 1 and a log scale ratio.

Budget (default config, 16 x 16 grid, K = 4):
  image          256 x (47 + 1)                                    = 12 288
  video segment  256 x (23 + 1) + 1 + 7 + 3 x (54 + L_max x 24)    <= 12 288  ->  L_max = 82
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CD32Config:
    # ---- input -------------------------------------------------------------------------------
    image_size: int = 256
    patch: int = 16                       # DINOv3 ViT-L/16 -> 16 x 16 grid at 256^2
    # ---- frozen DINOv3 + pixel stem -------------------------------------------------------------
    dino_repo: Optional[str] = "facebook/dinov3-vitl16-pretrain-lvd1689m"   # None -> small random stub (tests)
    dino_dim: int = 1024
    dino_registers: int = 4
    dino_layers: tuple = (6, 12, 18, 24)  # DINOv3 blocks re-read as read-only keys / values in every layer
    stub_dino_dim: int = 64               # width of the stub used when dino_repo is None
    stub_dino_blocks: int = 4             # depth of the stub (dino_layers index its blocks, 1-based)
    pix_dim: int = 256                    # pixel stem channels per patch
    # ---- two-stream backbone ---------------------------------------------------------------------
    dim: int = 512
    heads: int = 8
    layers: int = 8
    mlp_ratio: float = 4.0
    rope_dims: tuple = (16, 24, 24)       # head dims rotated by t (frame index), x, y (cells); sum = dim // heads
    rope_base: float = 100.0
    shared_streams: bool = False          # option B of the design: C and D blocks share weights
    keyframe_every: int = 4               # K
    # ---- C ---------------------------------------------------------------------------------------
    d_c_image: int = 47                   # content channels per cell (+ 1 rho) for images
    d_c_video: int = 23                   # ... for video and posed views
    fov_min: float = 30.0                 # focal length as a bounded field of view (degrees)
    fov_max: float = 120.0
    fov_init: float = 50.0
    rho_refine_heads: int = 4             # cross-view attention that refines rho only (posed views)
    # ---- D ---------------------------------------------------------------------------------------
    flow_grid: int = 4                    # M_t: flow_grid^2 x 3 numbers, convex-upsampled guided by c_k
    d_r: int = 24                         # numbers per innovation token
    l_max: int = 0                        # 0: largest that fits the budget (82 for the default config)
    budget: int = 12288                   # numbers per image and per K-frame segment
    p_l0: float = 0.15                    # nested dropout: P(L = 0); otherwise L ~ U{1, ..., L_max}
    enc_r_layers: int = 4                 # residual encoder (slots read e = X - sg[X~], sg[X~], c_k)
    # ---- decoder ---------------------------------------------------------------------------------
    dec_dim: int = 512
    dec_heads: int = 8
    knn: int = 9                          # neighbours read per query after forward projection
    softmin_beta: float = 1.0             # soft z-buffer: softmin_j(log z_j + beta d_j^2), windowed by w_cut
    cut_eps: float = 1e-4                 # w_cut = log max(eps, 1 - d^2 / R^2), R = (k+1)-th distance
    z_eps: float = 1e-2                   # cells with depth below this (behind the camera) are not read
    backface_kappa: float = 10.0          # soft back-face cull log sigmoid(-kappa cos); 0 turns it off
    dec_pred_blocks: int = 2              # self-attention blocks after the geometric read
    dec_innov_blocks: int = 2             # cross-attention blocks of the innovation branch
    prev_read: bool = True                # predicted frame also reads the previous decoded frame (3 x 3 window)
    n_freq: int = 6                       # Fourier bands of offsets / positions in the decoder
    conv_channels: tuple = (256, 128, 64, 64, 64)   # rgb conv upsampler: width at grid resolution, then per x2
    conv_res_blocks: int = 1
    # ---- losses ----------------------------------------------------------------------------------
    w_rgb: float = 1.0
    w_feat: float = 1.0
    w_geo: float = 0.5                    # geometry-only decode of non-key frames (Charbonnier)
    w_pred: float = 0.5                   # sg[h_geom] + previous-frame read (Charbonnier, feature only)
    geo_rgb: bool = True                  # L_geo also on rgb (one more conv pass per non-key frame)
    w_flow: float = 0.01                  # ||m||_1: explain motion by pose + depth first
    w_smooth: float = 0.01                # edge-aware smoothness of rho at keyframes
    w_kl_c: float = 1e-6
    w_kl_d: float = 1e-6
    w_link: float = 0.25                  # keyframe k decoded from C_{k-1} through the link pose
    w_depth_prior: float = 0.0            # scale/shift-invariant pull of rho to a depth probe (needs depth_probe)
    depth_probe: Optional[str] = None     # torch.save'd state_dict of Linear(dino_dim, 1) -> log inverse depth
    w_lpips: float = 0.0
    charbonnier_eps: float = 1e-3
    detach_rho_image: bool = True         # still images have no parallax: rho gets no gradient from L_rec
    extra: dict = field(default_factory=dict)

    # ---- derived ---------------------------------------------------------------------------------
    @property
    def grid(self) -> int:
        return self.image_size // self.patch

    @property
    def n_cells(self) -> int:
        return self.grid * self.grid

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def n_flow(self) -> int:
        return self.flow_grid * self.flow_grid

    @property
    def geom_numbers(self) -> int:
        """Numbers of D_geom per frame: pose (6) + coarse 3D flow."""
        return 6 + 3 * self.n_flow

    @property
    def link_numbers(self) -> int:
        return 7

    def c_numbers(self, modality: str) -> int:
        """Numbers of C per keyframe: cells x (content + rho), + 1 focal length for video."""
        if modality == "image":
            return self.n_cells * (self.d_c_image + 1)
        return self.n_cells * (self.d_c_video + 1) + (1 if modality == "video" else 0)

    @property
    def L_max(self) -> int:
        if self.l_max > 0:
            return self.l_max
        K = self.keyframe_every
        free = self.budget - self.c_numbers("video") - self.link_numbers - (K - 1) * self.geom_numbers
        return max(1, free // ((K - 1) * self.d_r))

    def segment_numbers(self, modality: str, L: Optional[int] = None) -> int:
        """Numbers per image, per K-frame video segment (link included), or per object of K posed views
        (posed views: D_geom costs 0, each view after the first has up to L_max innovation tokens)."""
        L = self.L_max if L is None else L
        if modality == "image":
            return self.c_numbers("image")
        n = self.keyframe_every - 1
        if modality == "video":
            return self.c_numbers("video") + self.link_numbers + n * (self.geom_numbers + L * self.d_r)
        return self.c_numbers("threed") + n * L * self.d_r

    def check(self) -> None:
        assert self.grid % self.flow_grid == 0, "flow_grid must divide the cell grid"
        assert sum(self.rope_dims) == self.head_dim and all(d % 2 == 0 for d in self.rope_dims), \
            "rope_dims must be even and sum to dim // heads"
        assert self.dec_dim // self.dec_heads == self.head_dim, "decoder heads must have the backbone head dim"
        assert self.knn + 1 <= self.n_cells, "knn + 1 neighbours must exist"
        assert len(self.conv_channels) - 1 == (self.patch.bit_length() - 1), "one conv stage per x2 up to the patch"
        assert self.segment_numbers("image") <= self.budget, "image C exceeds the budget"
        assert self.segment_numbers("video") <= self.budget, "video segment exceeds the budget"


def tiny_config(**kw) -> CD32Config:
    """Small CPU config for unit tests: stub encoder, 64^2 images (4 x 4 grid), 2 x 2 flow grid."""
    base = dict(dino_repo=None, image_size=64, stub_dino_dim=32, stub_dino_blocks=2, dino_layers=(1, 2),
                pix_dim=16, dim=48, heads=4, layers=2, rope_dims=(4, 4, 4), d_c_image=11, d_c_video=5,
                flow_grid=2, d_r=4, budget=192, enc_r_layers=1, dec_dim=48, dec_heads=4, knn=9,
                dec_pred_blocks=1, dec_innov_blocks=1, n_freq=4, conv_channels=(32, 16, 16, 8, 8),
                rho_refine_heads=2)
    base.update(kw)
    return CD32Config(**base)
