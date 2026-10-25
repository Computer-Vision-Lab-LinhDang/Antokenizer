"""Hyper-parameters and payload ledger of the C-D tokenizer, version 4.1 (geometric transport + Δ-Tucker).

State of one segment of n <= 4 observations (anchor i = 0, targets i = 1 .. n - 1):

    Z = (h, C = (c, rho, kappa_0), {z_i = (xi_i, M_i, kappa_i, a_i, r_i)})

Ledger at the default config (16 x 16 cells, L_i = 80):
    header                                             16
    anchor   256 x 23 + 256 + 4                     6 148
    target   6 + 16 x 3 + 4 + 32 + 80 x 24          2 010
    segment  6 148 + 16 + 3 x 2 010                12 194   (94 under the 12 288 ceiling, unallocated)
    image    6 148 + 16                             6 164

Scalars are counted as transmitted FP32 numbers, not as a compressed bitrate. Known cameras are counted
like estimated ones. `tucker: false` drops a_i (ablation) and can take one more slot (n_slots 81) to stay
within 8 scalars of the full model.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from ..cd32.config import CD32Config

VERSION = 41
MODES = ("image", "video", "static_views")


@dataclass
class CD41Config(CD32Config):
    """CD32Config supplies the frozen DINOv3 reader and pixel stem fields (dino_*, pix_dim, image_size, patch)
    and the trunk width (dim, heads, layers, mlp_ratio). The 3.2 latent and decoder fields are not used."""
    # ---- anchor C
    d_c: int = 23                         # content channels per cell
    # ---- Δ-Tucker on the space x observation x channel tensor of the variation
    tucker: bool = True                   # False: no a_i, no U B Q (ablation)
    rank_xy: int = 32                     # R_xy: spatial basis U(c) [N, R_xy]
    rank_a: int = 32                      # R_a: a_i per target
    rank_c: int = 32                      # R_c: core G [R_xy, R_a, R_c], channel map Q [R_c, d]
    # ---- target z_i
    n_motion: int = 16                    # M_i: 16 3D displacement vectors, blended per cell by beta(c)
    n_slots: int = 80                     # L_i innovation slots of d_r numbers (d_r from CD32Config: 24)
    nested: bool = False                  # train with nested prefixes L in nested_L (ablation)
    nested_L: tuple = (0, 8, 16, 32, 64, 80)
    header_slots: int = 16
    # ---- cameras estimated on video: kappa = (fx / W, fy / H, cx / W, cy / H)
    pp_range: float = 0.05                # principal point offset bound around the centre (normalised)
    # ---- transport (sparse, 9 nearest projected anchor cells within r_cut)
    geometry: bool = True                 # False: identity transport (ablation)
    r_cut: float = 1.5                    # patch units
    sigma_u: float = 0.35                 # patch units
    tau_z: float = 0.1                    # depth temperature, in the gauge (video: anchor median depth 1)
    # ---- decoder side (reads only Z and the shared weights)
    syn_blocks: int = 2                   # F_C / F_U generator from c
    rgb_blocks: int = 4                   # Dec_RGB over the cells
    dec_r_blocks: int = 2                 # Dec_R: cells cross-attend to the innovation slots
    pix_out: int = 8
    # ---- encoder side
    enc_r_layers: int = 2                 # Enc_R over [residual cells ; slot queries]
    world_camera: str = "estimated"       # world clips: estimated (video mode, GT only in the loss) | given
    # ---- losses
    w_rgb: float = 1.0
    w_lpips: float = 0.5
    lpips_max_frames: int = 64
    w_feat: float = 1.0                   # cosine + MSE on the teacher features (LN of the last DINO layer)
    w_base: float = 0.5                   # same on H_base, cells with support only
    w_cam_gt: float = 1.0                 # pose and intrinsics against the given cameras (world clips)
    w_depth_gt: float = 0.0               # SILog of rho against measured depth, when the batch has it
    w_flow: float = 0.01                  # |M|: explain motion by the camera first
    w_smooth: float = 0.0

    # ---- ledger -----------------------------------------------------------------------------------
    @property
    def anchor_numbers(self) -> int:
        return self.n_cells * self.d_c + self.n_cells + 4

    def target_numbers(self, L: Optional[int] = None) -> int:
        L = self.n_slots if L is None else L
        return 6 + 3 * self.n_motion + 4 + (self.rank_a if self.tucker else 0) + L * self.d_r

    def segment_numbers(self, modality: str = "video", L: Optional[int] = None, n_obs: Optional[int] = None) -> int:
        """Scalars of one image (anchor + header) or one segment of n_obs observations (default K)."""
        if modality == "image":
            return self.header_slots + self.anchor_numbers
        n = self.keyframe_every if n_obs is None else n_obs
        return self.header_slots + self.anchor_numbers + (n - 1) * self.target_numbers(L)

    @property
    def L_max(self) -> int:
        return self.n_slots

    def check(self) -> None:
        assert self.patch == 16, "the pixel stem downsamples x16"
        assert self.dim % self.heads == 0 and self.dec_dim % self.dec_heads == 0
        assert 1 <= self.knn <= self.n_cells
        assert 2 <= self.keyframe_every <= 4, "a segment holds the anchor and at most three targets"
        assert self.header_slots == 16, "the header has exactly 16 slots"
        assert self.world_camera in ("estimated", "given")
        assert all(0 <= L <= self.n_slots for L in self.nested_L)
        assert self.segment_numbers("video") <= self.budget, \
            f"segment {self.segment_numbers('video')} exceeds the budget {self.budget}"


def tiny_cd41_config(**kw) -> CD41Config:
    """CPU config for unit tests: stub DINO, 64^2 images (4 x 4 cells), no LPIPS."""
    base = dict(dino_repo=None, image_size=64, stub_dino_dim=32, stub_dino_blocks=2, dino_layers=(1, 2),
                pix_dim=16, dim=48, heads=4, layers=2, dec_dim=48, dec_heads=4, d_c=5, rank_xy=4, rank_a=4,
                rank_c=4, n_motion=4, d_r=4, n_slots=6, nested_L=(0, 2, 6), budget=400, syn_blocks=1,
                rgb_blocks=1, dec_r_blocks=1, enc_r_layers=1, w_lpips=0.0, lpips_max_frames=0)
    base.update(kw)
    return CD41Config(**base)


def config_from_dict(d: dict, **overrides) -> CD41Config:
    """Checkpoint / yaml dict -> CD41Config (unknown keys dropped, lists of tuple fields -> tuples)."""
    fields = {f.name: f for f in dataclasses.fields(CD41Config)}
    out = {k: v for k, v in dict(d).items() if k in fields}
    out.update(overrides)
    for k, v in list(out.items()):
        if isinstance(v, list) and isinstance(fields[k].default, tuple):
            out[k] = tuple(v)
    return CD41Config(**out)
