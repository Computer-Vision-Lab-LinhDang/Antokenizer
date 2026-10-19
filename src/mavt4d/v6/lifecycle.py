"""Slot lifecycle: spawn into free slots, TENTATIVE -> ACTIVE, retire on expected-visible misses.

Merge is not implemented yet (chot: merge only when both slots are seen and compete >= 2 steps).
"""
from dataclasses import replace

import torch
from torch import nn

from .contracts import ACTIVE, FREE, TENTATIVE
from .geometry import quat_mul, quat_to_matrix
from .write import anchor_measurements, camera_bits, metric_scale, weighted_kabsch


class V6Lifecycle(nn.Module):
    def __init__(self, cfg, max_position_std=50., spawn_anchors=3, min_depth=.5,
                 spawn_velocity_var=25., spawn_angular_var=1., spawn_offset_var=.01):
        super().__init__()
        self.cfg = cfg
        self.max_position_std, self.spawn_anchors, self.min_depth = max_position_std, spawn_anchors, min_depth
        self.spawn_velocity_var, self.spawn_angular_var, self.spawn_offset_var = spawn_velocity_var, spawn_angular_var, spawn_offset_var

    def expected_visible(self, state, obs):
        """[B,K] slot centre projects in front, inside the image of a valid camera and is not behind the depth map."""
        with torch.autocast(state.position.device.type, enabled=False):
            r_rig = quat_to_matrix(state.rig_orientation.float())
            x_rig = (state.position.float() - state.rig_position.float()[:, None]) @ r_rig     # [B,K,3]
            c2r = obs.cam_to_rig.float()
            x_cam = (x_rig[:, None] - c2r[:, :, None, :3, 3]) @ c2r[..., :3, :3]               # [B,V,K,3]
            z = x_cam[..., 2]
            kk = obs.intrinsics.float()
            zs = z.clamp_min(1e-6)
            u = kk[..., None, 0, 0] * x_cam[..., 0] / zs + kk[..., None, 0, 2]
            v = kk[..., None, 1, 1] * x_cam[..., 1] / zs + kk[..., None, 1, 2]
            h, w = obs.depth.shape[-2:]
            inside = (z > self.min_depth) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & obs.camera_valid[..., None]
            iu, iv = u.nan_to_num().long().clamp(0, w - 1), v.nan_to_num().long().clamp(0, h - 1)
            flat = (iv * w + iu).flatten(2)
            depth = obs.depth.float().flatten(2).gather(-1, flat)
            dvalid = obs.depth_valid.flatten(2).gather(-1, flat)
            occluded = dvalid & (depth < z - torch.clamp(.1 * z, min=1.))
            return (inside & ~occluded).any(1)

    def forward(self, state, meas, obs):
        with torch.autocast(state.position.device.type, enabled=False):
            return self._step(state, meas, obs)

    def _step(self, state, meas, obs):
        cfg = self.cfg
        st = state
        b, k, a = st.evidence.shape
        status = st.status
        alloc = status > FREE
        seen = (meas.anchor_meas_valid & alloc[..., None]).any(-1)
        # A miss counts only when the slot should have been visible (occluded / out of frame do not count).
        missed = alloc & ~seen & self.expected_visible(st, obs)
        miss = torch.where(missed, (st.miss_streak + 1).clamp_max(32767), st.miss_streak)
        pos_std = st.cov_pv[..., :3, :3].diagonal(dim1=-2, dim2=-1).amax(-1).clamp_min(0).sqrt()
        tent, act = status == TENTATIVE, status == ACTIVE
        promote = tent & seen & (st.seen_streak >= cfg.tentative_steps)
        # A tentative slot dies on its first miss; out of view or occluded it waits like an active one.
        drop = (tent & missed) | (act & (miss >= cfg.miss_steps)) | (alloc & (pos_std > self.max_position_std))
        status = torch.where(promote, torch.full_like(status, ACTIVE), status)
        status = torch.where(drop, torch.full_like(status, FREE), status)
        miss = torch.where(drop, torch.zeros_like(miss), miss)
        streak = torch.where(drop, torch.zeros_like(st.seen_streak), st.seen_streak)

        # Spawn: slots free at entry, existence > spawn_threshold (or routed to a box by the training teacher)
        # and enough measured anchors.
        valid = meas.anchor_meas_valid
        want = torch.sigmoid(meas.spawn_logit.float()) > cfg.spawn_threshold
        if meas.spawn_forced is not None:
            want = want | meas.spawn_forced
        spawn = (st.status == FREE) & want & (valid.sum(-1) >= self.spawn_anchors)
        st = replace(st, status=status, miss_streak=miss, seen_streak=streak)
        if not bool(spawn.any()):
            return st
        return self._spawn(st, meas, obs, spawn)

    def _spawn(self, st, meas, obs, spawn):
        cfg = self.cfg
        b, k, a = st.evidence.shape
        dev = st.position.device
        valid = meas.anchor_meas_valid
        z, var_a = anchor_measurements(st, meas, obs, cfg.log_r_floor)
        w_a = valid.float() / var_a.mean(-1)
        z = torch.where(valid[..., None], z, 0.)
        canon = meas.spawn_canon.float() * metric_scale(st, obs)[:, None, None, None]  # same depth units as z
        _, quat_m, t_m, _, _, rot_cov = weighted_kabsch(canon, z, w_a)
        r_rig = quat_to_matrix(st.rig_orientation.float())[:, None]
        position = st.rig_position.float()[:, None] + (r_rig @ t_m[..., None]).squeeze(-1)
        orientation = quat_mul(st.rig_orientation.float()[:, None].expand(b, k, 4), quat_m)
        # Initial covariance from the measurement noise (cross with the rig starts at 0).
        wn = w_a / w_a.sum(-1, keepdim=True).clamp_min(1e-12)
        pos_var = (wn.square()[..., None] * var_a).sum(-2).mean(-1) + st.rig_cov[:, None, :3, :3].diagonal(dim1=-2, dim2=-1).mean(-1)
        rot_var = (rot_cov.diagonal(dim1=-2, dim2=-1).mean(-1) + st.rig_cov[:, None, 3:6, 3:6].diagonal(dim1=-2, dim2=-1).mean(-1))
        ones3 = torch.ones(b, k, 3, device=dev)
        cov_pv = torch.diag_embed(torch.cat((pos_var[..., None] * ones3, self.spawn_velocity_var * ones3), -1))
        cov_rot = torch.diag_embed(torch.cat((rot_var.clamp(1e-4, 10.)[..., None] * ones3, self.spawn_angular_var * ones3), -1))
        cov_off = torch.diag_embed(torch.tensor([self.spawn_offset_var, self.spawn_offset_var], device=dev)).expand(b, k, a, 3, 2, 2)
        uid = st.uid.amax(-1, keepdim=True) + spawn.long().cumsum(-1)
        t_now = obs.t.to(st.birth_t.dtype)[:, None].expand(b, k)

        def put(new, old):
            return torch.where(spawn.view(b, k, *(1,) * (old.dim() - 2)), new.to(old.dtype), old)

        zeros = torch.zeros_like
        return replace(
            st, c_obj=put(meas.c_obj_prop.float(), st.c_obj), anchor_canon=put(canon, st.anchor_canon),
            anchor_desc=put(meas.desc_prop, st.anchor_desc), atlas=put(zeros(st.atlas), st.atlas),
            evidence=put(zeros(st.evidence), st.evidence), c_evt=put(zeros(st.c_evt), st.c_evt),
            evt_meta=put(zeros(st.evt_meta), st.evt_meta), h=put(meas.h_prop.float(), st.h),
            position=put(position, st.position), velocity=put(zeros(st.velocity), st.velocity),
            orientation=put(orientation, st.orientation), angular_velocity=put(zeros(st.angular_velocity), st.angular_velocity),
            offset=put(zeros(st.offset), st.offset), offset_velocity=put(zeros(st.offset_velocity), st.offset_velocity),
            cov_pv=put(cov_pv, st.cov_pv), cov_rot=put(cov_rot, st.cov_rot), cov_offset=put(cov_off, st.cov_offset),
            cross_rig=put(zeros(st.cross_rig), st.cross_rig),
            status=put(torch.full_like(st.status, TENTATIVE), st.status), uid=put(uid, st.uid),
            birth_t=put(t_now, st.birth_t), last_seen=put(t_now, st.last_seen),
            exist_logit=put(meas.spawn_logit.float(), st.exist_logit),
            seen_streak=put(torch.ones_like(st.seen_streak), st.seen_streak),
            miss_streak=put(zeros(st.miss_streak), st.miss_streak),
            cams_seen=put(camera_bits(meas, obs), st.cams_seen))
