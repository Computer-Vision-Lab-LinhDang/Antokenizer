"""V6 model: one filter step = dynamics prior -> encoder -> trunk -> write -> lifecycle."""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .contracts import Observation, RayBatch, V6State
from .decoder import V6Decoder
from .dynamics import V6Dynamics
from .encoder import V6Encoder
from .geometry import quat_to_matrix
from .lifecycle import V6Lifecycle
from .splats import SplatRenderer
from .trunk import V6Trunk
from .understanding import V6Understanding
from .write import V6Writer

# Weights of modules that no longer exist: older checkpoints load without them (the slot-only dense head, replaced
# by rendered dense channels).
RETIRED = ('understanding.dense_slot.', 'understanding.dense_pos.', 'understanding.dense_head.',
           'understanding.dense_background')


def drop_retired(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items() if not k.startswith(RETIRED)}


# Flags a module may expose to checkpoint its own blocks.
_CKPT_FLAGS = ('grad_checkpointing', 'activation_checkpointing', 'checkpoint_layers', 'gradient_checkpointing',
               'use_checkpoint', 'checkpoint')


def _checkpointed(forward):
    def run(*args, **kwargs):
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)
    run._v6_wrapped = forward
    return run


def _set_module_checkpointing(module: nn.Module, enabled: bool) -> str:
    for flag in _CKPT_FLAGS:
        if isinstance(getattr(module, flag, None), bool):
            setattr(module, flag, enabled)
            return f'flag:{flag}'
    blocks = [m for name, m in module.named_children()
              if isinstance(m, nn.ModuleList) and name in ('blocks', 'layers')]
    targets = [b for group in blocks for b in group] or [module]
    for target in targets:
        wrapped = target.__dict__.get('forward')
        if enabled and wrapped is None:
            target.forward = _checkpointed(target.forward)
        elif not enabled and wrapped is not None:
            del target.forward
    return 'blocks' if blocks else 'module'


class V6Model(nn.Module):
    def __init__(self, cfg, num_classes: int = 30, text_dim: int = 1152):
        super().__init__()
        self.cfg = cfg
        self.encoder = V6Encoder(cfg)
        self.trunk = V6Trunk(cfg)
        self.dynamics = V6Dynamics(cfg)
        self.writer = V6Writer(cfg)
        self.lifecycle = V6Lifecycle(cfg)
        self.decoder = V6Decoder(cfg)
        self.understanding = V6Understanding(cfg, num_classes=num_classes, text_dim=text_dim)
        self.splats = SplatRenderer(cfg) if cfg.gaussians else None

    @property
    def device(self):
        return next(self.parameters()).device

    def set_activation_checkpointing(self, enabled: bool = True) -> dict:
        """Checkpoint encoder/trunk blocks: module flag if exposed, else wrap blocks here."""
        return {name: _set_module_checkpointing(getattr(self, name), enabled) for name in ('encoder', 'trunk')}

    def initial_state(self, batch_size: int, device=None, time=0.) -> V6State:
        return V6State.initial(self.cfg, batch_size, device=device or self.device, time=time)

    @staticmethod
    def _dt(state: V6State, t) -> torch.Tensor:
        t = torch.as_tensor(t, device=state.t.device, dtype=torch.float64)
        return t.expand_as(state.t) - state.t  # f64 seconds; dynamics does the math in fp32

    def step(self, state: V6State, obs: Observation, boxes: dict | None = None):
        """Observed step. Returns the posterior at obs.t and aux {enc, meas, prior}.
        boxes (training only): world GT boxes {center, size, quat, valid} for the trunk's box teacher."""
        dt = self._dt(state, obs.t)
        prior = self.dynamics(state, dt)
        prior = replace(prior, t=obs.t.to(prior.t.dtype))  # exact clock, no f32 rounding
        enc = self.encoder(obs)
        meas = self.trunk(prior, enc, obs, boxes=boxes)
        post = self.writer(prior, meas, obs)
        post = self.lifecycle(post, meas, obs)
        post = self.writer.write_scene(post, meas, obs, enc)
        if self.splats is not None:
            post = self.splats.write(post, obs, enc)
        return post, {'enc': enc, 'meas': meas, 'prior': prior}

    def predict(self, state: V6State, dt) -> V6State:
        """Unobserved step: dynamics only (C bit-identical, D drifts)."""
        dt = torch.as_tensor(dt, device=state.t.device, dtype=torch.float64).expand_as(state.t)
        out = self.dynamics(state, dt)
        return replace(out, t=state.t + dt)

    def render(self, state: V6State, rays: RayBatch, chunk: int = 4096):
        return self.decoder(state, rays, chunk=chunk)

    def render_pixels(self, state: V6State, intrinsics, cam_to_rig, pixels, size):
        """Pixels [B,V,R,2] (u, v) of cameras of size (H, W) -> (RenderOutput over the V*R rays, rays, z_scale):
        the field decoder on the rays, or the state's Gaussians with cfg.gaussians."""
        rays, z_scale = self.camera_rays(state, intrinsics, cam_to_rig, pixels, return_z_scale=True)
        if self.splats is None:
            return self.render(state, rays), rays, z_scale
        out = self.splats(state, intrinsics, cam_to_rig, pixels, size, rays, z_scale,
                          lambda dirs: self.decoder.sky(state, dirs))
        return out, rays, z_scale

    def camera_rays(self, state: V6State, intrinsics, cam_to_rig=None, pixels=None, *,
                    near: float = .1, far: float = 120., return_z_scale: bool = False):
        """World rays for pixels [B,V,R,2] (u, v in px) using the state's rig pose; flattened over V.

        Also accepts camera_rays(state, obs, pixels). With return_z_scale, also returns [B,V*R]
        cos(angle to the optical axis), so camera z-depth = ray depth * z_scale.
        """
        if isinstance(intrinsics, Observation):
            intrinsics, cam_to_rig, pixels = intrinsics.intrinsics, intrinsics.cam_to_rig, cam_to_rig
        with torch.autocast(device_type=pixels.device.type, enabled=False):
            k = intrinsics.float()
            fx, fy = k[..., 0, 0].clamp_min(1e-6), k[..., 1, 1].clamp_min(1e-6)
            cx, cy, skew = k[..., 0, 2], k[..., 1, 2], k[..., 0, 1]
            u, v = pixels.float().unbind(-1)
            y = (v - cy[..., None]) / fy[..., None]
            x = (u - cx[..., None] - skew[..., None] * y) / fx[..., None]
            d_cam = torch.stack((x, y, torch.ones_like(x)), -1)
            d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True)
            z_scale = d_cam[..., 2]
            c2r = cam_to_rig.float()
            r_rig = quat_to_matrix(state.rig_orientation.float())          # [B,3,3]
            rot = r_rig[:, None] @ c2r[..., :3, :3]                           # [B,V,3,3]
            origin = (r_rig[:, None] @ c2r[..., :3, 3:])[..., 0] + state.rig_position.float()[:, None]
            direction = d_cam @ rot.transpose(-1, -2)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            b, v_, r = pixels.shape[:3]
            origin = origin[:, :, None].expand(b, v_, r, 3)
            near_far = direction.new_tensor([near, far]).expand(b, v_ * r, 2)
            rays = RayBatch(origin=origin.reshape(b, v_ * r, 3), direction=direction.reshape(b, v_ * r, 3),
                            near_far=near_far.contiguous())
        return (rays, z_scale.reshape(b, v_ * r)) if return_z_scale else rays
