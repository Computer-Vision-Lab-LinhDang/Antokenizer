"""Hyper-parameters and payload ledger of MAVT Grain-Prior C-D v1.

State after observation n (anchor j = 0, at most `queue` target records in the window W_n):

    S_n = (h, C = (G_0, c_0, kappa_0), z_b, {D_i}_{i in W_n})
    G_0[k] = (u_k 2, rho_k 1, sigma_k 1, f_k 60)                     64 grains, one per 2 x 2 stratum
    c_0, z_b in R^{16 x 128}                                         coarse anchor, boundary memory
    D_i = (xi_i 6, M_i 16 x 3, kappa_i 4, b_i 16 x 2, r_i L_i x 24)  r_ij = (u 2, rho 1, sigma 1, e 20)

Ledger at the reference config (L_i = 52):
    header                                    32
    grains        64 x 64                  4 096
    coarse c_0    16 x 128                 2 048
    kappa_0                                    4
    boundary z_b  16 x 128                 2 048     (sent as soon as the state has a target record)
    target        6 + 48 + 4 + 32 + 52 x 24   1 338
    full state    three targets           12 242     (46 under the 12 288 ceiling)
    image         header + C               6 180

Scalars are transmitted FP32 numbers, not an entropy-coded bitrate. Known cameras are copied into D_i and
counted like estimated ones.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..cd32.config import CD32Config

VERSION = 1
MODES = ("image", "video", "static_views", "calibrated_video")
HEADER = 32


@dataclass
class GrainPriorConfig(CD32Config):
    """CD32Config supplies the frozen DINOv3 reader fields (dino_*, image_size, patch, stub_*) and the width of
    the attention blocks (dim 512, heads 8, mlp_ratio 4: FFN 2048). Its 3.2 latent and decoder fields are not
    used here."""
    layers: int = 6                       # two-stream trunk depth (stream C and stream D, own weights)
    # ---- stems
    stem_channels: tuple = (64, 128, 256, 512)    # four Conv2d stride 2, kernel 3, GELU between
    pe_freq: int = 8                      # Fourier sin / cos bands per scalar coordinate
    mlp_hidden: int = 512                 # the 2-layer MLPs of f(.) heads
    # ---- anchor C
    stratum: int = 2                      # stratum side in patches: (16 / 2)^2 = 64 grains
    d_f: int = 60                         # grain feature
    coarse_grid: int = 4                  # 4 x 4 = 16 coarse slots
    d_coarse: int = 128                   # c_0, z
    grain_blocks: int = 2                 # read-refine blocks of the grain queries
    grain_read_s: float = 1.0             # Gaussian width (patches) of the grain read around u_k
    sigma_min: float = 0.4                # grain radius in latent cells: 0.4 + 1.1 sigmoid
    sigma_span: float = 1.1
    rho_lo: float = -7.0                  # bounds of the depth logits (log inverse depth before the gauge)
    rho_hi: float = 5.0
    temp_s: float = 1.0                   # T_s of the within-stratum soft location
    # ---- target record D_i
    n_motion: int = 16                    # M_i: 16 controls in R^3, blended per grain by beta
    d_b: int = 2                          # memory update scalars per coarse slot (16 x 2 = 32)
    n_slots: int = 52                     # innovation slots
    d_e: int = 20                         # innovation feature (+ u 2, rho 1, sigma 1 = 24)
    queue: int = 3                        # target records kept in the state
    nested_L: tuple = (0, 8, 16, 32, 52)  # prefix lengths drawn uniformly in training
    # ---- cameras: kappa = (fx / W, fy / H, cx / W, cy / H), edge convention
    fov_min: float = 30.0                 # focal bounds as a field of view (degrees), locked in the run config
    fov_max: float = 120.0
    fov_init: float = 50.0
    image_fov_fixed: bool = False         # images: kappa_0 = fov_init, centred (no focal signal in a single view)
    pp_range: float = 0.05                # principal point offset bound around the centre (normalised)
    z_eps: float = 1e-2
    # ---- refinement (stage 4B)
    refine_passes: int = 2
    upd_blocks: int = 2
    corr_eps: float = 0.05                # exploration mass of the correspondence prior
    corr_s: tuple = (3.0, 1.5)            # its Gaussian width per pass (patches)
    # ---- memory (diagonal recurrence)
    frame_seconds: float = 0.125          # seconds per sampled frame step (the loaders carry no timestamps)
    # ---- residual encoder (stage 5)
    res_layers: int = 4
    # ---- public decoder (stage 7)
    read_layers: int = 2
    tau_d: float = 0.1                    # depth temperature of the read bias
    tau_d_mode: str = "median"            # median: depth in units of the anchor median | absolute
    rgb_channels: tuple = (256, 128, 64, 32)
    rgb_from: str = "feature"             # feature: RGB from the predicted DINO feature | hidden: from the decoder state
    query_grid: int = 0                   # > 0 (legacy, rgb_from=hidden): RGB from a second read on this finer grid
    read_qk_norm: bool = False            # per-head LayerNorm on q, k of the grain read (gp3: |W_q|, |W_k| grew, softmax saturated)
    readout_version: str = "legacy"      # old checkpoints retain exactly the original decoder
    # CRSR reader (shared_ray_v1 only): coarse residual init, zero-value sink, radius projected to the target
    read_strict: bool = False             # legacy readout with the strict reader (q/k normalised, bounded temperature)
    read_coarse_init: bool = False        # a^(0) from the bilinear coarse grid at u_q (no memory cross-attention)
    read_sink: bool = False               # learned-logit sink with zero value; grain bias carries proximity
    read_radius_scale: bool = False       # grain radius x (d_anchor / z_target) x (f_target / f_anchor)
    sigma_param: str = "sigmoid"          # sigmoid: sigma_min + span * sigmoid | log: exp, floor sigma_lo
    sigma_lo: float = 0.02
    query_version: int = 1
    measurement_version: int = 1         # 1 historical; 2 fixed GT denominator/local support
    render_profile: str = "uniform_midpoint_v1"
    innovation_semantics: str = "legacy_image"
    render_samples: int = 32
    query_chunk: int = 512
    render_near: float = 0.1             # source gauge units, never fitted to target labels
    render_far: float = 4.0
    background_rgb: tuple = (0.0, 0.0, 0.0)
    # ---- causal prior (stage 8)
    prior_layers: int = 6
    horizons: tuple = (1, 4)
    # ---- arms (section 15.1)
    query_init: str = "cond"              # cond: coarse + memory + geometry pooled | uncond: learned queries
    geom_bias: bool = True                # False: grain read without the geometry bias
    memory: bool = True                   # False: z_i = c_0 at every observation
    # ---- training
    phase: str = "codec"                  # codec (T1 / T2) | prior (T3: tokenizer frozen)
    world_given_p: float = 0.5            # world clips: share coded with given cameras (rest: L_pose label)
    w_lpips: float = 0.1
    w_logn: float = 0.01
    w_base: float = 0.5
    w_init: float = 0.25
    w_geom: float = 1.0
    w_pose: float = 0.1
    w_motion: float = 1e-3
    w_code: float = 1e-4
    depth_delta: float = 0.1              # Huber delta of the log-depth loss
    lpips_max_frames: int = 64

    def __post_init__(self):
        self.d_r = 4 + self.d_e           # CD32Config field, fixed here by the slot layout (u 2, rho, sigma, e)

    # ------------------------------------------------------------------------------ derived
    @property
    def n_strata(self) -> int:
        return self.grid // self.stratum

    @property
    def n_grains(self) -> int:
        return self.n_strata * self.n_strata

    @property
    def n_coarse(self) -> int:
        return self.coarse_grid * self.coarse_grid

    @property
    def d_grain(self) -> int:
        return 4 + self.d_f

    @property
    def n_memb(self) -> int:
        return self.n_coarse * self.d_b

    def anchor_numbers(self) -> int:
        return self.n_grains * self.d_grain + self.n_coarse * self.d_coarse + 4

    def target_numbers(self, L: int) -> int:
        return 6 + 3 * self.n_motion + 4 + self.n_memb + L * self.d_r

    def state_numbers(self, Ls=()) -> int:
        """Scalars of a state with target records of lengths Ls (image: Ls empty)."""
        n = HEADER + self.anchor_numbers()
        if len(Ls):
            n += self.n_coarse * self.d_coarse + sum(self.target_numbers(int(L)) for L in Ls)
        return n

    def check(self) -> None:
        if self.readout_version not in ("legacy", "shared_ray_v1", "shared_field_v1"):
            raise ValueError("unknown readout_version")
        crsr = self.read_coarse_init or self.read_sink or self.read_radius_scale
        if self.read_strict and self.readout_version != "legacy":
            raise ValueError("read_strict applies to the legacy readout (shared readouts are strict already)")
        if crsr and not (self.readout_version == "shared_ray_v1" or self.read_strict):
            raise ValueError("read_coarse_init / read_sink / read_radius_scale need shared_ray_v1 or read_strict")
        if self.sigma_param not in ("sigmoid", "log") or not 0 < self.sigma_lo < self.sigma_min + self.sigma_span:
            raise ValueError("sigma_param: sigmoid | log, with 0 < sigma_lo < sigma_min + sigma_span")
        if self.query_version != 1 or self.measurement_version not in (1, 2):
            raise ValueError("unsupported query/measurement version")
        if self.render_profile != "uniform_midpoint_v1" or not 0 < self.render_near < self.render_far:
            raise ValueError("invalid render profile/bounds")
        if self.render_samples < 1 or self.query_chunk < 1:
            raise ValueError("samples and chunk must be positive")
        if self.innovation_semantics not in ("legacy_image", "disabled", "shared_hidden_v1"):
            raise ValueError("unsupported innovation semantics")
        if self.readout_version == "shared_field_v1" and self.innovation_semantics != "disabled":
            raise ValueError("spatial innovations require the empirical consistency gate")
        if len(self.background_rgb) != 3 or any(not 0 <= c <= 1 for c in self.background_rgb):
            raise ValueError("background_rgb must be three values in [0,1]")
        assert self.grid % self.stratum == 0, "stratum must divide the grid"
        assert self.rgb_from in ("feature", "hidden"), "rgb_from: feature | hidden"
        if self.query_grid:
            up = self.image_size // self.query_grid
            assert self.readout_version == "legacy" and self.rgb_from == "hidden", "query_grid: legacy, rgb_from=hidden"
            assert self.query_grid % self.grid == 0 and up * self.query_grid == self.image_size and up & (up - 1) == 0 \
                and 2 <= up <= 16, "query_grid: a multiple of grid, image_size / query_grid a power of 2 in [2, 16]"
        assert self.grid % self.coarse_grid == 0, "coarse_grid must divide the grid"
        assert self.dim % self.heads == 0
        assert len(self.stem_channels) == 4 and self.image_size // 16 == self.grid, "four stride-2 convs: patch 16"
        assert self.nested_L[-1] == self.n_slots and self.nested_L[0] == 0
        assert self.query_init in ("cond", "uncond")
        assert self.tau_d_mode in ("median", "absolute")
        assert self.phase in ("codec", "prior")
        assert len(self.corr_s) >= self.refine_passes
        assert self.fov_min < self.fov_init < self.fov_max


def tiny_config(**kw) -> GrainPriorConfig:
    """CPU config for unit tests: stub DINO, 64^2 images (4 x 4 cells, 4 grains, 2 x 2 coarse slots)."""
    base = dict(image_size=64, dino_repo=None, stub_dino_dim=32, stub_dino_blocks=2, dim=32, heads=4, layers=2,
                stem_channels=(8, 8, 16, 32), pe_freq=3, mlp_hidden=32, d_f=6, d_coarse=8, grain_blocks=1,
                coarse_grid=2, n_motion=4, n_slots=6, d_e=4, nested_L=(0, 2, 4, 6), upd_blocks=1, res_layers=1, read_layers=2,
                prior_layers=1, rgb_channels=(16, 8, 8, 8), w_lpips=0.0, dino_layers=(1, 2))
    base.update(kw)
    return GrainPriorConfig(**base)


def config_from_dict(d: dict, **overrides) -> GrainPriorConfig:
    fields = {f.name for f in dataclasses.fields(GrainPriorConfig)}
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in {**d, **overrides}.items() if k in fields}
    return GrainPriorConfig(**d)
