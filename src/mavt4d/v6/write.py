"""One-pass write rules (R1-R3, R6). Every write reads the prior and the measurement, never another write.

D: joint EKF (Joseph form, fp64 core) per slot over [rig pose (6), slot (p, v, theta, w)] with the stored
cross block; measurements are the weighted anchor centroid (rig frame) and, with >= 3 anchors, the Procrustes
rotation. The rig itself is updated only from meas.rig_meas and the metadata pose (obs.rig_pose), never from
slot measurements (no K-fold double counting); its cross terms come from the per-slot joint posterior.
C_obj: gain lambda/Lambda with a hard evidence mask (with soft slots: lambda scaled by ownership mass instead;
no innovation gate). C_evt: phi(nu~) - phi(0). h: mean
visibility gain. Unseen slots keep C, Lambda, h, desc and atlas bit-identical. Scene map: same gain rule per
cell (scene.py), written by write_scene after the lifecycle from token points no allocated slot holds.
"""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics import exp_quat, sym
from .geometry import quat_conj, quat_mul, quat_normalize, quat_to_matrix, skew
from .scene import SceneMeasure, scene_write

ATLAS_EXTENT = 4.0  # m; 8^3 cells over [-extent, extent)^3, cell = x + 8y + 64z, LSB first (decoder layout)


def log_quat(q):
    """Unit quaternion -> rotation vector (shortest arc); series near identity keeps gradients finite."""
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    w, v = q[..., :1], q[..., 1:]
    s2 = v.square().sum(-1, keepdim=True)
    small = s2 < 1e-12
    s = s2.clamp_min(1e-12).sqrt()
    wc = w.clamp_min(1e-6)
    k = torch.where(small, 2 / wc * (1 - s2 / (3 * wc.square())), 2 * torch.atan2(s, w) / s)
    return v * k


def matrix_to_quat(m):
    """Rotation matrix -> unit quaternion (Shepperd)."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    a21, a02, a10 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    s01, s02, s12 = m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1]
    cands = torch.stack((torch.stack((1 + tr, a21, a02, a10), -1),
                         torch.stack((a21, 1 + m00 - m11 - m22, s01, s02), -1),
                         torch.stack((a02, s01, 1 - m00 + m11 - m22, s12), -1),
                         torch.stack((a10, s02, s12, 1 - m00 - m11 + m22), -1)), -2)
    pick = torch.stack((tr, m00, m11, m22), -1).argmax(-1)
    return quat_normalize(cands.gather(-2, pick[..., None, None].expand(*pick.shape, 1, 4)).squeeze(-2))


def weighted_kabsch(model, meas, weight, damping=.1):
    """Fit meas ~ R model + t (object -> sensor) over anchors with weights 1/sigma^2 (0 = unused).

    Rotation: SVD without gradient, then one differentiable Gauss-Newton step (implicit-function gradient).
    Returns rot [...,3,3], quat [...,4], t [...,3], model/meas centroids, rotation covariance [...,3,3]
    (left perturbation in the sensor frame; damping caps it at 1/damping rad^2).
    """
    wn = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
    xbar = (wn[..., None] * model).sum(-2)
    zbar = (wn[..., None] * meas).sum(-2)
    x0, z0 = model - xbar[..., None, :], meas - zbar[..., None, :]
    with torch.no_grad():
        u, _, vh = torch.linalg.svd(torch.einsum('...a,...ai,...aj->...ij', wn, z0, x0))
        sign = torch.det(u @ vh).sign()
        fix = torch.stack((torch.ones_like(sign), torch.ones_like(sign), torch.where(sign == 0, 1., sign)), -1)
        r0 = u @ torch.diag_embed(fix) @ vh
        q0 = matrix_to_quat(r0)
    y = x0 @ r0.transpose(-1, -2)
    eye = torch.eye(3, device=model.device, dtype=model.dtype)
    info = (weight * y.square().sum(-1)).sum(-1)[..., None, None] * eye - torch.einsum('...a,...ai,...aj->...ij', weight, y, y)
    reg = info + damping * eye
    delta = torch.linalg.solve(reg, (weight[..., None] * torch.linalg.cross(y, z0, dim=-1)).sum(-2)[..., None])
    quat = quat_normalize(quat_mul(exp_quat(delta.squeeze(-1)), q0))
    rot = quat_to_matrix(quat)
    t = zbar - (rot @ xbar[..., None]).squeeze(-1)
    return rot, quat, t, xbar, zbar, sym(torch.linalg.inv(reg))


def joseph_update(p, h, r, nu):
    """Kalman update in fp64. p [...,n,n], h [...,m,n], r [...,m,m], nu [...,m] -> dx, P+, whitened nu."""
    p, h, r, nu = p.double(), h.double(), r.double(), nu.double()
    ht = h.transpose(-1, -2)
    s = sym(h @ p @ ht + r)
    chol, _ = torch.linalg.cholesky_ex(s)
    gain = torch.cholesky_solve(h @ p, chol).transpose(-1, -2)   # P H^T S^-1 (P symmetric)
    dx = (gain @ nu[..., None]).squeeze(-1)
    ikh = torch.eye(p.shape[-1], dtype=p.dtype, device=p.device) - gain @ h
    p_new = sym(ikh @ p @ ikh.transpose(-1, -2) + gain @ r @ gain.transpose(-1, -2))
    nu_w = torch.linalg.solve_triangular(chol, nu[..., None], upper=False).squeeze(-1)
    return dx, p_new, nu_w


def psd_cross(a, d, c, margin=.999):
    """Shrink cross block c so [[a, c], [c^T, d]] stays PSD (blocks are stored separately)."""
    eps = 1e-12
    la, _ = torch.linalg.cholesky_ex(a + eps * torch.eye(a.shape[-1], dtype=a.dtype, device=a.device))
    ld, _ = torch.linalg.cholesky_ex(d + eps * torch.eye(d.shape[-1], dtype=d.dtype, device=d.device))
    m = torch.linalg.solve_triangular(la, c, upper=False)
    m = torch.linalg.solve_triangular(ld, m.transpose(-1, -2), upper=False)
    scale = (margin ** 2 / m.square().sum((-1, -2)).clamp_min(1e-60)).clamp_max(1).sqrt()  # no sqrt at 0
    return c * scale[..., None, None]


def camera_bits(meas, obs, min_mass=.5):
    """[B,K] uint8 bits of the state camera rows whose tokens give the slot >= min_mass ownership."""
    b, v = obs.camera_valid.shape
    k = meas.ownership.shape[-1] - 1
    mass = meas.ownership[..., :k].float().reshape(b, v, -1, k).sum(2)
    rows = meas.camera_rows.clamp(0, 7)
    bits = torch.where((mass >= min_mass) & (obs.camera_valid & (meas.camera_rows >= 0))[..., None],
                       torch.bitwise_left_shift(torch.ones_like(rows), rows)[..., None], 0)
    out = torch.zeros(b, k, dtype=torch.int64, device=bits.device)
    for i in range(v):
        out = out | bits[:, i]
    return out.to(torch.uint8)


def atlas_bytes(anchor_canon, mark):
    """Pack atlas cells of anchors with mark [B,K,A] into [B,K,64] uint8 (cell of the canonical anchor)."""
    p = anchor_canon.float()
    inside = ((p >= -ATLAS_EXTENT) & (p < ATLAS_EXTENT)).all(-1) & mark
    xyz = ((p + ATLAS_EXTENT) * (4 / ATLAS_EXTENT)).floor().long().clamp(0, 7)
    cell = xyz[..., 0] + 8 * xyz[..., 1] + 64 * xyz[..., 2]
    hit = torch.zeros(*cell.shape[:-1], 512, dtype=torch.int32, device=cell.device).scatter_add(-1, cell, inside.int())
    weights = torch.bitwise_left_shift(torch.ones(8, dtype=torch.int32, device=cell.device),
                                       torch.arange(8, device=cell.device, dtype=torch.int32))
    return ((hit > 0).int().view(*cell.shape[:-1], 64, 8) * weights).sum(-1).to(torch.uint8)


def metric_scale(state, obs):
    """[B] factor to metres: 1 for metric-depth sources, else state.metric_scale (the trunk does not apply it)."""
    return torch.where(obs.has_metric_depth, torch.ones_like(state.metric_scale), state.metric_scale).float()


def anchor_measurements(state, meas, obs, floor):
    """Measured anchors [B,K,A,3] and their variances, rig frame, metres."""
    sc = metric_scale(state, obs)[:, None, None, None]
    var = meas.anchor_log_r.float().clamp(floor, 10.).exp() * sc.square()
    return meas.anchor_meas.float() * sc, var


def rig_measurements(prior, meas, obs, floor):
    """Rig pose rows: trunk correction (z_hat = 0) and metadata pose. -> nu [B,12], var [B,12], mask [B,12]."""
    b = prior.t.shape[0]
    sc = torch.cat((metric_scale(prior, obs)[:, None].expand(b, 3), torch.ones(b, 3, device=prior.t.device)), -1)
    q_obs = matrix_to_quat(obs.rig_pose[:, :3, :3].float())
    meta = torch.cat((obs.rig_pose[:, :3, 3].float() - prior.rig_position,
                      log_quat(quat_mul(q_obs, quat_conj(prior.rig_orientation)))), -1)
    nu = torch.cat((meas.rig_meas.float() * sc, meta), -1)
    var = torch.cat((meas.rig_log_r.float().clamp(floor, 10.).exp() * sc.square(),
                     torch.full((b, 6), math.exp(floor), device=nu.device)), -1)
    mask = torch.cat((torch.isfinite(meas.rig_meas).all(-1, keepdim=True).expand(b, 6),
                      obs.rig_pose_valid[:, None].expand(b, 6)), -1)
    return torch.where(mask, nu, 0), var, mask


class V6Writer(nn.Module):
    def __init__(self, cfg, hidden=None):
        super().__init__()
        self.cfg = cfg
        w = cfg.latent_width
        self.evt_norm = nn.LayerNorm(w)
        self.phi = nn.Sequential(nn.Linear(6 + w + 1, hidden or 2 * w), nn.SiLU(), nn.Linear(hidden or 2 * w, w))
        self.scene = SceneMeasure(cfg) if cfg.scene_samples else None

    def event_delta(self, c_evt, nu_w, log_dt):
        """R2: dC_evt = phi(nu~, C_evt, log dt) - phi(0, C_evt, log dt); exactly 0 where nu~ == 0."""
        ctx = torch.cat((self.evt_norm(c_evt), log_dt[..., None]), -1)
        delta = self.phi(torch.cat((nu_w, ctx), -1)) - self.phi(torch.cat((torch.zeros_like(nu_w), ctx), -1))
        return torch.where((nu_w != 0).any(-1, keepdim=True), delta, torch.zeros_like(delta))

    def forward(self, prior, meas, obs):
        with torch.autocast(prior.position.device.type, enabled=False):
            return self._write(prior, meas, obs)

    def write_scene(self, state, meas, obs, enc):
        """Scene map write, run after the lifecycle: it reads which slots are allocated (and their boxes) to keep
        object points out of the static map. The map itself goes from the prior to here untouched."""
        if self.scene is None or meas.points is None:
            return state
        return replace(state, **scene_write(self.cfg, state, *self.scene(state, meas, obs, enc)))

    def _write(self, prior, meas, obs):
        cfg, st = self.cfg, prior
        b, k, a = st.evidence.shape
        dev = st.position.device
        floor = cfg.log_r_floor
        alloc = st.allocated
        valid = meas.anchor_meas_valid & alloc[..., None]
        n_valid = valid.sum(-1)
        seen, full = n_valid >= 1, n_valid >= 3

        # ---- D: measurement from evidence only (anchor_meas), prediction from the prior mean.
        z, var_a = anchor_measurements(st, meas, obs, floor)                          # rig frame, metres
        z = torch.where(valid[..., None], z, 0.)
        w_a = valid.float() / var_a.mean(-1)
        model = st.anchor_canon.float() + st.offset.float()
        rot_m, quat_m, t_m, xbar, zbar, rot_cov = weighted_kabsch(model, z, w_a)
        wn = w_a / w_a.sum(-1, keepdim=True).clamp_min(1e-12)
        var_c = (wn.square()[..., None] * var_a).sum(-2)                              # centroid variance, rig frame

        r_rig = quat_to_matrix(st.rig_orientation.float())[:, None]                   # [B,1,3,3]
        r_rig_t = r_rig.transpose(-1, -2)
        r_obj = quat_to_matrix(st.orientation.float())
        m_x = (r_obj @ xbar[..., None]).squeeze(-1)
        y = st.position.float() + m_x - st.rig_position.float()[:, None]
        nu_c = zbar - (r_rig_t @ y[..., None]).squeeze(-1)
        q_rig = st.rig_orientation.float()[:, None].expand(b, k, 4)
        nu_r = log_quat(quat_mul(quat_mul(q_rig, quat_m), quat_conj(st.orientation.float())))
        r_rot = r_rig @ rot_cov @ r_rig_t + math.exp(floor) * torch.eye(3, device=dev)

        nu_rig, var_rig, mask_rig = rig_measurements(st, meas, obs, floor)
        eye3 = torch.eye(3, device=dev, dtype=torch.float64).expand(b, k, 3, 3)
        h = torch.zeros(b, k, 18, 18, device=dev, dtype=torch.float64)
        h[..., :6, :6] = torch.eye(6, dtype=torch.float64, device=dev) * mask_rig[:, None, :6, None]
        h[..., 6:12, :6] = torch.eye(6, dtype=torch.float64, device=dev) * mask_rig[:, None, 6:, None]
        mc, mr = seen[..., None, None], full[..., None, None]
        h[..., 12:15, 0:3] = torch.where(mc, -r_rig_t, 0.).double()
        h[..., 12:15, 3:6] = torch.where(mc, r_rig_t @ skew(y), 0.).double()
        h[..., 12:15, 6:9] = torch.where(mc, r_rig_t.expand(b, k, 3, 3), 0.).double()
        h[..., 12:15, 12:15] = torch.where(mc, -r_rig_t @ skew(m_x), 0.).double()
        h[..., 15:18, 3:6] = torch.where(mr, -eye3, 0.)
        h[..., 15:18, 12:15] = torch.where(mr, eye3, 0.)
        nu = torch.cat((nu_rig[:, None].expand(b, k, 12), torch.where(seen[..., None], nu_c, 0.),
                        torch.where(full[..., None], nu_r, 0.)), -1)
        diag = torch.cat((torch.where(mask_rig, var_rig, 1.)[:, None].expand(b, k, 12),
                          torch.where(seen[..., None], var_c, 1.), torch.zeros(b, k, 3, device=dev)), -1)
        r = torch.diag_embed(diag).double()
        r[..., 15:, 15:] = torch.where(mr, r_rot, torch.eye(3, device=dev)).double()

        p_rig = st.rig_cov.double()
        p_slot = torch.zeros(b, k, 12, 12, device=dev, dtype=torch.float64)
        p_slot[..., :6, :6] = st.cov_pv.double()
        p_slot[..., 6:, 6:] = st.cov_rot.double()
        # Free slots: neutral numbers so nothing nonfinite reaches shared gradients.
        p_slot = torch.where(alloc[..., None, None], p_slot, torch.eye(12, dtype=torch.float64, device=dev))
        p_rr = p_rig[:, None, :6, :6].expand(b, k, 6, 6)
        cross = psd_cross(p_rr, p_slot, torch.where(alloc[..., None, None], st.cross_rig.double(), 0.))
        p = torch.cat((torch.cat((p_rr, cross), -1), torch.cat((cross.transpose(-1, -2), p_slot), -1)), -2)
        dx, p_post, nu_w = joseph_update(p, h, r, nu)
        dx, p_post = dx.float(), p_post.float()

        # Rig: own measurements only, full 12-state (t, r, v, w).
        h_rig = torch.zeros(b, 12, 12, device=dev, dtype=torch.float64)
        h_rig[:, :6, :6] = torch.eye(6, dtype=torch.float64, device=dev) * mask_rig[:, :6, None]
        h_rig[:, 6:, :6] = torch.eye(6, dtype=torch.float64, device=dev) * mask_rig[:, 6:, None]
        dx_rig, rig_cov, _ = joseph_update(p_rig, h_rig, torch.diag_embed(torch.where(mask_rig, var_rig, 1.)), nu_rig)
        dx_rig = dx_rig.float()

        def keep(mask, new, old):
            return torch.where(mask.view(*mask.shape, *(1,) * (old.dim() - mask.dim())), new.to(old.dtype), old)

        orientation = quat_normalize(quat_mul(exp_quat(dx[..., 12:15]), st.orientation.float()))
        d_fields = dict(
            position=keep(alloc, st.position + dx[..., 6:9], st.position),
            velocity=keep(alloc, st.velocity + dx[..., 9:12], st.velocity),
            orientation=keep(alloc, orientation, st.orientation),
            angular_velocity=keep(alloc, st.angular_velocity + dx[..., 15:18], st.angular_velocity),
            cov_pv=keep(alloc, p_post[..., 6:12, 6:12], st.cov_pv),
            cov_rot=keep(alloc, p_post[..., 12:18, 12:18], st.cov_rot),
            cross_rig=keep(alloc, p_post[..., :6, 6:18], st.cross_rig),
            rig_position=st.rig_position + dx_rig[:, :3],
            rig_orientation=quat_normalize(quat_mul(exp_quat(dx_rig[:, 3:6]), st.rig_orientation)),
            rig_velocity=st.rig_velocity + dx_rig[:, 6:9],
            rig_angular_velocity=st.rig_angular_velocity + dx_rig[:, 9:12],
            rig_cov=rig_cov.float())

        # Offsets: per-axis 2x2 Kalman on Procrustes residuals (object frame); gain 0 for unseen anchors.
        off_meas = ((z - t_m[..., None, :]) @ rot_m) - st.anchor_canon.float()     # R^T (z - t) - canon
        off_var = (var_a @ rot_m.square()).clamp_min(math.exp(floor))
        upd = valid & full[..., None]
        po = st.cov_offset.float()
        s_off = po[..., 0, 0] + off_var
        g0, g1 = po[..., 0, 0] / s_off, po[..., 1, 0] / s_off
        nu_o = off_meas - st.offset.float()
        gain = torch.stack((g0, g1), -1)[..., None]                                   # [B,K,A,3,2,1]
        ikh = torch.eye(2, device=dev) - gain @ torch.tensor([[1., 0.]], device=dev)
        po_new = sym(ikh @ po @ ikh.transpose(-1, -2) + gain @ gain.transpose(-1, -2) * off_var[..., None, None])
        d_fields.update(
            offset=keep(upd, st.offset + g0 * nu_o, st.offset),
            offset_velocity=keep(upd, st.offset_velocity + g1 * nu_o, st.offset_velocity),
            cov_offset=keep(upd, po_new, st.cov_offset))

        # ---- C_obj (R1): lambda = s * rho with a hard mask, gain lambda / Lambda only where Lambda > 0.
        s_vis, rho = meas.visibility.float(), meas.depth_conf.float()
        trusted = (valid & (s_vis >= cfg.visibility_min) & (rho >= cfg.depth_conf_min)
                   & (n_valid >= cfg.min_anchors_for_shape)[..., None])
        if cfg.soft_slots and meas.anchor_mass is not None:
            # no visibility / confidence / anchor-count floors: weak, lightly owned evidence writes weakly
            trusted = valid
            lam = torch.where(trusted, s_vis * rho * meas.anchor_mass.float().clamp_max(1), 0.)
        else:
            lam = torch.where(trusted, s_vis * rho, 0.)
        lam = torch.where(obs.has_metric_depth[:, None, None], lam, lam.clamp_max(cfg.rgb_only_lambda_cap))
        wrote = lam > 0
        evidence = torch.where(wrote, (st.evidence + lam).clamp_max(cfg.lambda_max), st.evidence)
        gain_a = torch.where(wrote, lam / torch.where(wrote, evidence, 1.), 0.)
        desc = st.anchor_desc.float()
        desc = desc + gain_a[..., None] * (meas.desc_prop.float() - desc)
        aff = meas.anchor_affinity.float().clamp_min(0)
        aff = aff / aff.sum(-1, keepdim=True).clamp_min(1e-12)
        g_tok = (aff * gain_a[:, :, None]).sum(-1).clamp(0, 1)                       # [B,K,6]
        c_obj = st.c_obj + g_tok[..., None] * (meas.c_obj_prop.float() - st.c_obj)

        # ---- C_evt (R2): only through the whitened slot innovation (rows 12:18, given the rig rows).
        nu_slot = nu_w[..., 12:18].float()
        dt_seen = (obs.t[:, None] - st.last_seen).float().clamp_min(0)
        log_dt = dt_seen.clamp_min(1e-3).log()
        c_evt = st.c_evt + self.event_delta(st.c_evt.float(), nu_slot, log_dt)
        chi2 = torch.stack((nu_slot[..., :3].square().sum(-1), nu_slot[..., 3:].square().sum(-1)), -1)
        exceed = (chi2 > cfg.event_chi2_99).any(-1)
        conf = torch.where(seen, .5 * st.evt_meta[..., 0] + .5 * exceed.float(), st.evt_meta[..., 0])
        age = torch.where(seen, torch.where(exceed, 0., st.evt_meta[..., 1] + dt_seen), st.evt_meta[..., 1])

        # ---- h: colour-only change, gain = mean visibility.
        g_h = torch.where(alloc, (s_vis * valid).sum(-1) / a, 0.)
        h_new = st.h + g_h[..., None] * (meas.h_prop.float() - st.h)

        # ---- Background tokens: same Lambda-gain rule.
        bvis = meas.background_vis.float()
        lam_b = torch.where(bvis >= cfg.visibility_min, bvis, 0.)
        lam_b = torch.where(obs.has_metric_depth[:, None], lam_b, lam_b.clamp_max(cfg.rgb_only_lambda_cap))
        wrote_b = lam_b > 0
        ev_b = torch.where(wrote_b, (st.background_evidence + lam_b).clamp_max(cfg.lambda_max), st.background_evidence)
        gain_b = torch.where(wrote_b, lam_b / torch.where(wrote_b, ev_b, 1.), 0.)
        bg = st.background.float()
        bg = bg + gain_b[..., None] * (meas.background_prop.float() - bg)

        # ---- Bookkeeping.
        t_now = obs.t.to(st.last_seen.dtype)[:, None].expand(b, k)
        c_fields = dict(
            c_obj=keep(g_tok > 0, c_obj, st.c_obj),
            anchor_desc=keep(wrote, desc, st.anchor_desc),
            evidence=evidence,
            atlas=st.atlas | atlas_bytes(st.anchor_canon, wrote),
            c_evt=keep(alloc & (nu_slot != 0).any(-1), c_evt, st.c_evt),
            evt_meta=torch.stack((conf, age), -1),
            h=keep(g_h > 0, h_new, st.h),
            background=keep(wrote_b, bg, st.background),
            background_evidence=ev_b,
            last_seen=torch.where(seen, t_now, st.last_seen),
            seen_streak=torch.where(seen, (st.seen_streak + 1).clamp_max(32767), torch.zeros_like(st.seen_streak)),
            miss_streak=torch.where(seen, torch.zeros_like(st.miss_streak), st.miss_streak),
            cams_seen=torch.where(seen, st.cams_seen | camera_bits(meas, obs), st.cams_seen))
        return replace(st, **d_fields, **c_fields, **self._cameras(st, meas, obs))

    def _cameras(self, st, meas, obs):
        """Register calibration and rig key of the observed cameras in their routed state rows (trunk routes by key)."""
        b, v = obs.camera_valid.shape
        rows = meas.camera_rows.clamp(0, st.camera.shape[1] - 1)
        hw = torch.tensor(obs.depth.shape[-2:], dtype=torch.float32, device=rows.device).expand(b, v, 2)
        kk = obs.intrinsics.float()
        packed = torch.cat((obs.cam_to_rig.float()[..., :3, :4].flatten(-2),
                            torch.stack((kk[..., 0, 0], kk[..., 1, 1], kk[..., 0, 2], kk[..., 1, 2]), -1), hw), -1)
        packed = F.pad(packed, (0, st.camera.shape[-1] - packed.shape[-1]))
        camera, key, cvalid = st.camera.clone(), st.camera_key.clone(), st.camera_valid.clone()
        bi = torch.arange(b, device=rows.device)
        for i in range(v):
            sel = obs.camera_valid[:, i] & (meas.camera_rows[:, i] >= 0)
            camera[bi[sel], rows[sel, i]] = packed[sel, i]
            key[bi[sel], rows[sel, i]] = obs.camera_key[sel, i]
            cvalid[bi[sel], rows[sel, i]] = True
        return dict(camera=camera, camera_key=key, camera_valid=cvalid)
