"""D drift over dt (R3, R4, R6): closed-form kinematics, exact Van Loan covariance.

Law parameters come from C_evt, a slot<->slot relation over explicit D and the action. Nothing here
reads c_obj, h, anchor_canon or anchor_desc; C fields are returned as the same tensors.
Error states: position/velocity in world; rotation error is a world-frame left perturbation
(R_true = Exp(dtheta) R); the rig uses the same convention, ordered (t, r, v, w).
"""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import TENTATIVE
from .geometry import quat_mul, quat_normalize, quat_to_matrix, skew

# name, size, initial value (softplus fields) or None (raw, zero init)
LAW_FIELDS = (('kappa', 1, .05), ('omega', 3, None), ('accel', 3, None), ('kappa_w', 1, .05),
              ('spring', 3, .1), ('damping', 3, .5), ('q_pv', 6, (1e-3,) * 3 + (.5,) * 3),
              ('q_rot', 6, (1e-3,) * 3 + (.05,) * 3), ('q_off', 3, .01))
NOISE = ('q_pv', 'q_rot', 'q_off')


def _inv_softplus(x):
    return math.log(math.expm1(x))


def sym(x):
    return .5 * (x + x.transpose(-1, -2))


def exp_quat(v):
    """Rotation vector -> unit quaternion; series near zero keeps the gradient finite."""
    a2 = v.square().sum(-1, keepdim=True)
    small = a2 < 1e-10
    a = a2.clamp_min(1e-10).sqrt()
    k = torch.where(small, .5 - a2 / 48, torch.sin(a / 2) / a)
    c = torch.where(small, 1 - a2 / 8, torch.cos(a / 2))
    return torch.cat((c, v * k), -1)


def van_loan(f, qc, dt):
    """Phi = e^{dt F}, Q = int_0^dt e^{sF} Qc e^{sF}^T ds, exact (so Q(a+b) composes)."""
    n = f.shape[-1]
    m = torch.cat((torch.cat((-f, qc), -1), torch.cat((torch.zeros_like(f), f.transpose(-1, -2)), -1)), -2)
    e = torch.linalg.matrix_exp(m * dt[..., None, None])
    phi = e[..., n:, n:].transpose(-1, -2)
    return phi, sym(phi @ e[..., :n, n:])


def _block2(a, b, c, d):
    return torch.cat((torch.cat((a, b), -1), torch.cat((c, d), -1)), -2)


class V6Dynamics(nn.Module):
    def __init__(self, cfg, action_dim=0, heads=4):
        super().__init__()
        self.cfg, self.action_dim = cfg, action_dim
        w = cfg.latent_width
        self.heads = heads if w % heads == 0 else 1
        self.evt_norm = nn.LayerNorm(w)
        self.inp = nn.Linear(31 + w, w)
        self.rel_norm = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3 * w)
        self.rel_out = nn.Linear(w, w)
        self.action_proj = nn.Linear(action_dim, w) if action_dim else None
        self.unknown_action = nn.Parameter(torch.zeros(w))
        # Unknown action: larger process noise (learned factor >= 1).
        self.unknown_q = nn.Parameter(torch.tensor(_inv_softplus(1.)))
        self.mlp = nn.Sequential(nn.Linear(3 * w, w), nn.SiLU())
        self.head = nn.Linear(w, sum(n for _, n, _ in LAW_FIELDS))
        nn.init.zeros_(self.head.weight)
        bias = []
        for _, n, init in LAW_FIELDS:
            vals = (0.,) * n if init is None else (init,) * n if isinstance(init, float) else init
            bias += [0. if init is None else _inv_softplus(x) for x in vals]
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor(bias))

    @staticmethod
    def features(state):
        """Explicit per-slot inputs: pose, twist, covariance summary, existence (no shape)."""
        rel = (state.position - state.rig_position[:, None]) / 10
        rot = quat_to_matrix(state.orientation)[..., :2].flatten(-2)
        cov = torch.cat((state.cov_pv.diagonal(dim1=-2, dim2=-1), state.cov_rot.diagonal(dim1=-2, dim2=-1),
                         state.cov_offset.diagonal(dim1=-2, dim2=-1).mean((-3, -2))), -1)
        exist = torch.stack((state.allocated.float(), (state.status == TENTATIVE).float()), -1)
        return torch.cat((rel, rot, state.velocity / 5, state.angular_velocity,
                          cov.clamp_min(1e-12).log() / 10, exist), -1)

    def law_params(self, state, action=None):
        """theta = g(C_evt, r, action) -> dict of [B,K,n] fp32 law parameters."""
        with torch.autocast(state.position.device.type, enabled=False):
            b, k = state.status.shape
            evt = self.evt_norm(state.c_evt.float())
            x = self.inp(torch.cat((self.features(state).float(), evt), -1))
            w = x.shape[-1]
            q, kk, v = self.qkv(self.rel_norm(x)).view(b, k, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
            mask = state.allocated[:, None, None, :] | torch.eye(k, dtype=torch.bool, device=x.device)
            r = F.scaled_dot_product_attention(q, kk, v, attn_mask=mask)
            r = self.rel_out(r.transpose(1, 2).reshape(b, k, w))
            if action is None:
                act, inflate = self.unknown_action.expand(b, k, w), 1 + F.softplus(self.unknown_q)
            elif self.action_proj is None:
                raise ValueError('action-free dynamics does not accept actions')
            else:
                act, inflate = self.action_proj(action.float())[:, None].expand(b, k, w), 1.
            raw = self.head(self.mlp(torch.cat((evt, r, act), -1))).split([n for _, n, _ in LAW_FIELDS], -1)
            law = {}
            for (name, _, init), val in zip(LAW_FIELDS, raw):
                if name in NOISE:
                    law[name] = self.cfg.q_min + F.softplus(val) * inflate
                else:
                    law[name] = val if init is None else F.softplus(val)
            return law

    def propagate(self, state, dt, law):
        """Apply frozen law parameters over dt (float or [B] seconds). Free slots untouched."""
        dev = state.position.device
        with torch.autocast(dev.type, enabled=False):
            b, k = state.status.shape
            delta = torch.as_tensor(dt, device=dev, dtype=torch.float64)
            delta = delta.expand(b) if delta.ndim == 0 else delta
            if delta.shape != (b,) or not torch.isfinite(delta).all() or (delta < 0).any():
                raise ValueError('dt must be finite, nonnegative, scalar or [B]')
            d = delta.float()
            dk = d[:, None].expand(b, k)
            law = {n: v.float() for n, v in law.items()}
            eye3 = torch.eye(3, device=dev).expand(b, k, 3, 3)
            z3 = torch.zeros_like(eye3)
            # Translation: dp = v, dv = A v + b, A = -kappa I + [Omega]x. One augmented exp gives Phi, Gamma1, Gamma2.
            a = -law['kappa'][..., None] * eye3 + skew(law['omega'])
            f_pv = _block2(z3, eye3, z3, a)
            aug = torch.cat((torch.cat((f_pv, torch.cat((z3, eye3), -2)), -1),
                             torch.zeros(b, k, 3, 9, device=dev)), -2)
            e = torch.linalg.matrix_exp(aug * dk[..., None, None])
            x = torch.cat((state.position, state.velocity), -1).float()
            x = (e[..., :6, :6] @ x[..., None] + e[..., :6, 6:] @ law['accel'][..., None]).squeeze(-1)
            phi_pv, q_pv = van_loan(f_pv, torch.diag_embed(law['q_pv']), dk)
            cov_pv = sym(phi_pv @ state.cov_pv.float() @ phi_pv.transpose(-1, -2) + q_pv)
            # Rotation: w decays at kappa_w; integrated angle w (1 - e^{-kw dt}) / kw, world frame (left multiply).
            kw = law['kappa_w']
            xk = kw * dk[..., None]
            small = xk < 1e-4
            gain = torch.where(small, dk[..., None] * (1 - xk / 2 + xk.square() / 6),
                               -torch.expm1(-xk) / torch.where(small, torch.ones_like(kw), kw))
            w0 = state.angular_velocity.float()
            orientation = quat_normalize(quat_mul(exp_quat(w0 * gain), state.orientation.float()))
            angular_velocity = w0 * torch.exp(-xk)
            f_rot = _block2(z3, eye3, z3, -kw[..., None] * eye3)
            phi_rot, q_rot = van_loan(f_rot, torch.diag_embed(law['q_rot']), dk)
            cov_rot = sym(phi_rot @ state.cov_rot.float() @ phi_rot.transpose(-1, -2) + q_rot)
            # Anchor offsets, per axis: [d, d'] <- expm(dt [[0, 1], [-k, -c]]) [d, d'].
            one, zero = torch.ones_like(law['spring']), torch.zeros_like(law['spring'])
            f_off = torch.stack((torch.stack((zero, one), -1), torch.stack((-law['spring'], -law['damping']), -1)), -2)
            qc_off = torch.diag_embed(torch.stack((torch.full_like(one, self.cfg.q_min), law['q_off']), -1))
            phi_off, q_off = van_loan(f_off, qc_off, dk[..., None].expand(b, k, 3))
            y = torch.stack((state.offset, state.offset_velocity), -1).float()
            y = (phi_off[:, :, None] @ y[..., None]).squeeze(-1)
            cov_off = sym(phi_off[:, :, None] @ state.cov_offset.float() @ phi_off[:, :, None].transpose(-1, -2)
                          + q_off[:, :, None])
            # Cross with the rig pose: rig pose-pose transition is I (rig vel-slot cross not stored).
            z6 = torch.zeros_like(phi_pv)
            phi_slot = _block2(phi_pv, z6, z6, phi_rot)
            cross = state.cross_rig.float() @ phi_slot.transpose(-1, -2)
            # Rig: constant velocity, rig_cov grows.
            rig_position = state.rig_position + state.rig_velocity * d[:, None]
            rig_orientation = quat_normalize(quat_mul(exp_quat(state.rig_angular_velocity * d[:, None]),
                                                      state.rig_orientation))
            eye6, z66 = torch.eye(6, device=dev).expand(b, 6, 6), torch.zeros(b, 6, 6, device=dev)
            phi_rig, q_rig = van_loan(_block2(z66, eye6, z66, z66), self.cfg.q_min * torch.eye(12, device=dev).expand(b, 12, 12), d)
            rig_cov = sym(phi_rig @ state.rig_cov.float() @ phi_rig.transpose(-1, -2) + q_rig)

            alloc = state.allocated

            def keep(new, old):
                return torch.where(alloc.view(b, k, *(1,) * (old.dim() - 2)), new.to(old.dtype), old)

            return replace(
                state, position=keep(x[..., :3], state.position), velocity=keep(x[..., 3:], state.velocity),
                orientation=keep(orientation, state.orientation),
                angular_velocity=keep(angular_velocity, state.angular_velocity),
                offset=keep(y[..., 0], state.offset), offset_velocity=keep(y[..., 1], state.offset_velocity),
                cov_pv=keep(cov_pv, state.cov_pv), cov_rot=keep(cov_rot, state.cov_rot),
                cov_offset=keep(cov_off, state.cov_offset), cross_rig=keep(cross, state.cross_rig),
                rig_position=rig_position, rig_orientation=rig_orientation, rig_cov=rig_cov,
                t=state.t + delta.to(state.t.dtype))

    def forward(self, state, dt, action=None):
        """Same function for rollout and skipped steps."""
        return self.propagate(state, dt, self.law_params(state, action))
