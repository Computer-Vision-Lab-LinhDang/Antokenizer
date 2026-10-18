"""Data contracts shared by the v6 encoder, trunk, write rules and decoder.

Shapes: B batch, V cameras (<= cfg.max_cameras), N patches per camera, K slots, A anchors,
W cfg.latent_width, H/Wd image height/width, G cfg.background_tokens.
Frames: "world" is the sequence frame, "rig" the ego frame at time t (cameras are rigidly
mounted, ``cam_to_rig`` is metadata), "object" the canonical frame of a slot.
Quaternions are unit (w, x, y, z), object/rig -> world.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

FREE, TENTATIVE, ACTIVE = 0, 1, 2


def _tensors(obj):
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


@dataclass(frozen=True)
class Observation:
    images: Tensor           # [B,V,T,3,H,Wd] float in [0,1]; T = cfg.tubelet, last frame is time t
    tube_dt: Tensor          # [B] seconds between the tube frames (0 for a still image)
    intrinsics: Tensor       # [B,V,3,3] at H x Wd
    cam_to_rig: Tensor       # [B,V,4,4]
    camera_key: Tensor       # [B,V] int64 rig channel id, routes a camera to its state row
    camera_valid: Tensor     # [B,V] bool
    depth: Tensor            # [B,V,H,Wd] metres for the last frame, 0 where missing
    depth_valid: Tensor      # [B,V,H,Wd] bool
    rig_pose: Tensor         # [B,4,4] rig -> world from metadata (identity when unknown)
    rig_pose_valid: Tensor   # [B] bool
    has_metric_depth: Tensor # [B] bool, source provides measured metric depth
    t: Tensor                # [B] float64 seconds

    def tensor_dict(self):
        return _tensors(self)


@dataclass(frozen=True)
class EncoderOutput:
    tokens: Tensor           # [B,V,N,E] content features, E = cfg.encoder_width
    valid: Tensor            # [B,V,N] bool
    depth_hyp: Tensor        # [B,V,N,D,5] per hypothesis: du, dv (px offset in patch), log z, log sigma, logit
    ray_cam: Tensor          # [B,V,N,3] unit ray through each patch centre, camera frame
    camera_motion: Tensor    # [B,V,12] se(3) motion between tube frames (6) and its log-std (6), camera frame
    measured_depth: Tensor   # [B,V,N,2] phase-2 tag, not computed by the encoder: median log z, log(1+count)
    measured_valid: Tensor   # [B,V,N] bool

    def tensor_dict(self):
        return _tensors(self)


# Rig prior at t0 (t, r, v, w): the world is the rig frame at the first step, so pose is near-exact, but the
# rig may already be moving (~10 m/s, ~0.5 rad/s std).
RIG_PRIOR_VAR = torch.tensor([1e-4] * 6 + [100.] * 3 + [.25] * 3)

@dataclass(frozen=True)
class V6State:
    """Everything kept across steps. Derived caches are recomputed, never stored here."""
    # C: appearance and shape (written only from trusted evidence)
    c_obj: Tensor            # [B,K,6,W] f32
    anchor_canon: Tensor     # [B,K,A,3] f32 object frame, metres
    anchor_desc: Tensor      # [B,K,A,16] f16
    atlas: Tensor            # [B,K,64] uint8 seen-surface bits
    evidence: Tensor         # [B,K,A] f32 accumulated Lambda
    c_evt: Tensor            # [B,K,W] f32 motion law / event code, written only through innovation
    evt_meta: Tensor         # [B,K,2] f32 event confidence, event age (s)
    h: Tensor                # [B,K,W] f32 non-motion change, colour only
    # D: explicit kinematics with covariance (Kalman)
    position: Tensor         # [B,K,3] world
    velocity: Tensor         # [B,K,3] world
    orientation: Tensor      # [B,K,4] object -> world
    angular_velocity: Tensor # [B,K,3] world
    offset: Tensor           # [B,K,A,3] anchor deformation, object frame
    offset_velocity: Tensor  # [B,K,A,3]
    cov_pv: Tensor           # [B,K,6,6] (position, velocity)
    cov_rot: Tensor          # [B,K,6,6] (rotation error, angular velocity)
    cov_offset: Tensor       # [B,K,A,3,2,2] per anchor axis (offset, offset velocity)
    cross_rig: Tensor        # [B,K,6,12] cov between rig pose error (t3, r3) and slot (p, v, r, w)
    # Slot lifecycle
    status: Tensor           # [B,K] int8 FREE/TENTATIVE/ACTIVE
    uid: Tensor              # [B,K] int64
    birth_t: Tensor          # [B,K] f64
    last_seen: Tensor        # [B,K] f64
    exist_logit: Tensor      # [B,K] f32
    seen_streak: Tensor      # [B,K] int16
    miss_streak: Tensor      # [B,K] int16
    cams_seen: Tensor        # [B,K] uint8 bitmask over camera rows
    # Scene
    rig_position: Tensor     # [B,3]
    rig_orientation: Tensor  # [B,4]
    rig_velocity: Tensor     # [B,3] world
    rig_angular_velocity: Tensor  # [B,3] world
    rig_cov: Tensor          # [B,12,12] (t, r, v, w)
    camera: Tensor           # [B,Vmax,32] packed: cam_to_rig 3x4 (12), fx fy cx cy (4), H Wd (2), zeros
    camera_key: Tensor       # [B,Vmax] int64, -1 when the row is unused
    camera_valid: Tensor     # [B,Vmax] bool
    background: Tensor       # [B,G,W] f16
    background_evidence: Tensor  # [B,G] f32
    global_token: Tensor     # [B,W] f32
    metric_scale: Tensor     # [B] f32 depth scale for sources without metric depth
    metric_valid: Tensor     # [B] bool
    t: Tensor                # [B] f64
    config_digest: Tensor    # [B,32] uint8

    @classmethod
    def initial(cls, cfg, batch_size, *, device='cpu', time=0.):
        b, k, a, w = batch_size, cfg.slots, cfg.anchors, cfg.latent_width
        if b < 1:
            raise ValueError('batch_size must be positive')

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=device, dtype=dtype)

        def eye(n, *lead):
            return torch.eye(n, device=device).expand(*lead, n, n).clone()

        unit_q = zeros(b, k, 4)
        unit_q[..., 0] = 1
        rig_q = zeros(b, 4)
        rig_q[:, 0] = 1
        camera = zeros(b, cfg.max_cameras, cfg.camera_rows)
        camera[..., :12] = torch.eye(4, device=device)[:3].reshape(12)
        t = torch.as_tensor(time, device=device, dtype=torch.float64).expand(b).clone()
        digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).digest()
        return cls(
            c_obj=zeros(b, k, cfg.c_obj_tokens, w), anchor_canon=zeros(b, k, a, 3),
            anchor_desc=zeros(b, k, a, cfg.descriptor_dim, dtype=torch.float16),
            atlas=zeros(b, k, 64, dtype=torch.uint8), evidence=zeros(b, k, a),
            c_evt=zeros(b, k, w), evt_meta=zeros(b, k, 2), h=zeros(b, k, w),
            position=zeros(b, k, 3), velocity=zeros(b, k, 3), orientation=unit_q,
            angular_velocity=zeros(b, k, 3), offset=zeros(b, k, a, 3), offset_velocity=zeros(b, k, a, 3),
            cov_pv=eye(6, b, k), cov_rot=eye(6, b, k), cov_offset=eye(2, b, k, a, 3),
            cross_rig=zeros(b, k, 6, 12),
            status=zeros(b, k, dtype=torch.int8), uid=zeros(b, k, dtype=torch.int64),
            birth_t=t[:, None].expand(b, k).clone(), last_seen=t[:, None].expand(b, k).clone(),
            exist_logit=zeros(b, k), seen_streak=zeros(b, k, dtype=torch.int16),
            miss_streak=zeros(b, k, dtype=torch.int16), cams_seen=zeros(b, k, dtype=torch.uint8),
            rig_position=zeros(b, 3), rig_orientation=rig_q, rig_velocity=zeros(b, 3),
            rig_angular_velocity=zeros(b, 3), rig_cov=torch.diag_embed(RIG_PRIOR_VAR.to(device).expand(b, 12)),
            camera=camera, camera_key=torch.full((b, cfg.max_cameras), -1, device=device, dtype=torch.int64),
            camera_valid=zeros(b, cfg.max_cameras, dtype=torch.bool),
            background=zeros(b, cfg.background_tokens, w, dtype=torch.float16),
            background_evidence=zeros(b, cfg.background_tokens), global_token=zeros(b, w),
            metric_scale=torch.ones(b, device=device), metric_valid=zeros(b, dtype=torch.bool), t=t,
            config_digest=torch.tensor(list(digest), device=device, dtype=torch.uint8).expand(b, 32).clone())

    @property
    def allocated(self):
        return self.status > FREE

    def tensor_dict(self):
        return _tensors(self)

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensor_dict().values())

    def clone(self):
        return replace(self, **{k: v.clone() for k, v in self.tensor_dict().items()})

    def detach(self):
        return replace(self, **{k: v.detach() for k, v in self.tensor_dict().items()})

    def to(self, device):
        return replace(self, **{k: v.to(device) for k, v in self.tensor_dict().items()})

    def select(self, indices):
        return replace(self, **{k: v.index_select(0, indices) for k, v in self.tensor_dict().items()})

    def validate(self, cfg):
        reference = self.initial(cfg, self.c_obj.shape[0], device=self.c_obj.device)
        for key, value in self.tensor_dict().items():
            expected = getattr(reference, key)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f'invalid v6 state field {key}: {tuple(value.shape)} {value.dtype}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'nonfinite v6 state field {key}')
        if not torch.equal(self.config_digest, reference.config_digest):
            raise ValueError('state configuration fingerprint mismatch')
        if ((self.status < FREE) | (self.status > ACTIVE)).any():
            raise ValueError('invalid slot status')


@dataclass(frozen=True)
class TrunkOutput:
    """What the trunk hands to the write rules. Measurements never depend on the prior's values."""
    ownership: Tensor        # [B,Ne,K+1] token -> slot (last column: trash), Ne = V*N
    ownership_by_layer: tuple  # ownership at cfg.ownership_layers, each [B,Ne,K+1]
    anchor_meas: Tensor      # [B,K,A,3] measured anchor positions, rig frame
    anchor_meas_valid: Tensor  # [B,K,A] bool
    anchor_log_r: Tensor     # [B,K,A,3] measurement log-variance (floored)
    visibility: Tensor       # [B,K,A] s_a in [0,1]
    depth_conf: Tensor       # [B,K,A] rho_a in [0,1]
    c_obj_prop: Tensor       # [B,K,6,W]
    desc_prop: Tensor        # [B,K,A,16]
    h_prop: Tensor           # [B,K,W]
    anchor_affinity: Tensor  # [B,K,6,A] w_ia, token i <-> anchor a
    rig_meas: Tensor         # [B,6] rig pose correction vs drifted prior (t3, r3), from evidence only
    rig_log_r: Tensor        # [B,6]
    background_prop: Tensor  # [B,G,W]
    background_vis: Tensor   # [B,G]
    spawn_logit: Tensor      # [B,K] existence logit for free slots
    spawn_canon: Tensor      # [B,K,A,3] canonical anchors proposed for a new slot
    camera_rows: Tensor      # [B,V] int64 state camera row used by each input camera


@dataclass(frozen=True)
class RayBatch:
    origin: Tensor           # [B,R,3] world
    direction: Tensor        # [B,R,3] world, unit
    near_far: Tensor         # [B,R,2]


@dataclass(frozen=True)
class RenderOutput:
    rgb: Tensor              # [B,R,3]
    depth: Tensor            # [B,R] expected depth along the ray
    opacity: Tensor          # [B,R] accumulated object opacity (1 - empty mass)
    surface_slot: Tensor     # [B,R] int64 slot of the first surface, -1 for background
    surface_canon: Tensor    # [B,R,3] surface point in that slot's canonical frame
    status: Tensor           # [B,R,3] probabilities: seen, inferred, unknown
    weights: Tensor          # [B,R,K+1] per-slot (+background) contribution, for losses
