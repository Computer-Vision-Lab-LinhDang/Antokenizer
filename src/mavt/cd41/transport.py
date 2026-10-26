"""Cameras, se(3) and the sparse geometric transport W_i of C-D 4.1 (section 7 of the design).

Conventions: camera x right, y down, z forward (OpenCV). The continuous image origin is the outer top-left
corner, so the pixel with integer index (u, v) has its centre at (u + 0.5, v + 0.5). Intrinsics are kept
normalised, kappa = (fx / W, fy / H, cx / W, cy / H) with the principal point in edge coordinates; the
transport works in patch units (normalised coordinates x 16 on the 16 x 16 grid), so cell (row a, col b)
has its centre at (b + 0.5, a + 0.5).

Transport of anchor cell j into observation i (rho is the log inverse z-depth, not the ray distance):

    P_j    = exp(-rho_j) K(kappa_0)^-1 [x_j, y_j, 1]
    P_ij   = R_i (P_j + sum_m beta_jm M_i[m]) + t_i,      (R_i, t_i) = exp_se(3)(xi_i)
    u_ij   = pi(K(kappa_i) P_ij)

Row u' of W_i (target cell) holds a softmax over at most `knn` anchor cells j that are valid for u': P_ij
finite, z_ij > z_eps, u_ij inside the crop and |u' - u_ij| < r_cut, with the logits of section 7

    l_u'j = -|u' - u_ij|^2 / (2 sigma_u^2) - z_ij / tau_z

(`smooth_cutoff` adds log max(eps, 1 - d^2 / r_cut^2), off by default). A row without a valid cell is exactly
zero (support 0); every other row sums to 1. W_i is kept sparse, (idx [B, N, k], w [B, N, k], support [B, N]),
and applied by gather in O(k N D); `SparseTransportOp.dense()` exists for tests and offline analysis only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..cd32 import geometry as geo


# ---------------------------------------------------------------------------------- intrinsics
def kappa_from_K(K_pix: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Pixel intrinsics of the model input (edge convention, after crop / resize) [..., 3, 3] -> kappa [..., 4]."""
    K = K_pix.float()
    return torch.stack([K[..., 0, 0] / width, K[..., 1, 1] / height, K[..., 0, 2] / width, K[..., 1, 2] / height], -1)


def kappa_from_index_K(K_pix: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Same, for metadata whose principal point counts the first pixel centre as 0: c_edge = c_index + 0.5."""
    K = K_pix.float().clone()
    K[..., 0, 2] = K[..., 0, 2] + 0.5
    K[..., 1, 2] = K[..., 1, 2] + 0.5
    return kappa_from_K(K, width, height)


def kappa_from_fov(fov_deg: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
    """Horizontal FOV (degrees) [...] and principal point offset from the centre [..., 2] (normalised) ->
    kappa [..., 4] of a square image with square pixels."""
    f = 0.5 / torch.tan(torch.deg2rad(fov_deg.float()) / 2)
    return torch.stack([f, f, 0.5 + pp[..., 0].float(), 0.5 + pp[..., 1].float()], -1)


def K_cells(kappa: torch.Tensor, grid: int) -> torch.Tensor:
    """kappa [..., 4] -> K [..., 3, 3] in patch units (normalised coordinates x grid)."""
    K = torch.zeros(*kappa.shape[:-1], 3, 3, device=kappa.device, dtype=torch.float32)
    K[..., 0, 0], K[..., 1, 1] = kappa[..., 0] * grid, kappa[..., 1] * grid
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = kappa[..., 2] * grid, kappa[..., 3] * grid, 1.0
    return K


# ---------------------------------------------------------------------------------- SO(3), se(3)
def hat(w: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> skew matrix [..., 3, 3] with hat(w) v = w x v."""
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack([O, -wz, wy, wz, O, -wx, -wy, wx, O], -1).reshape(*w.shape[:-1], 3, 3)


def _so3_coeffs(th2: torch.Tensor) -> tuple:
    """theta^2 -> (sin t / t, (1 - cos t) / t^2, (t - sin t) / t^3); Taylor series below theta = 0.1, where the
    closed forms lose float32 precision (the closed forms are evaluated at a safe argument there, so neither
    branch of the where produces NaN in the backward pass)."""
    small = th2 < 1e-2
    t2 = torch.where(small, torch.ones_like(th2), th2)
    t = t2.sqrt()
    A = torch.where(small, 1 - th2 / 6 + th2 * th2 / 120, torch.sin(t) / t)
    Bc = torch.where(small, 0.5 - th2 / 24 + th2 * th2 / 720, (1 - torch.cos(t)) / t2)
    C = torch.where(small, 1 / 6 - th2 / 120 + th2 * th2 / 5040, (t - torch.sin(t)) / (t2 * t))
    return A, Bc, C


def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation [..., 3, 3] -> unit quaternion (w, x, y, z) [..., 4] with w >= 0 (Shepperd: the largest of the
    four squared components is taken from the diagonal, so the division is stable at any angle)."""
    m = R.float()
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    sq = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    r = (sq.clamp_min(1e-12).sqrt() / 2)                                             # |w|, |x|, |y|, |z|
    d = 4 * r
    cand = torch.stack([
        torch.stack([r[..., 0], (m21 - m12) / d[..., 0], (m02 - m20) / d[..., 0], (m10 - m01) / d[..., 0]], -1),
        torch.stack([(m21 - m12) / d[..., 1], r[..., 1], (m01 + m10) / d[..., 1], (m02 + m20) / d[..., 1]], -1),
        torch.stack([(m02 - m20) / d[..., 2], (m01 + m10) / d[..., 2], r[..., 2], (m12 + m21) / d[..., 2]], -1),
        torch.stack([(m10 - m01) / d[..., 3], (m02 + m20) / d[..., 3], (m12 + m21) / d[..., 3], r[..., 3]], -1),
    ], -2)                                                                            # [..., 4 candidates, 4]
    k = sq.argmax(-1)
    q = torch.gather(cand, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(q[..., :1] < 0, -q, q)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation [..., 3, 3] -> axis-angle [..., 3] with angle in [0, pi]; so3_exp(so3_log(R)) = R."""
    q = rot_to_quat(R)
    w, v = q[..., :1], q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(s, w)
    scale = torch.where(s > 1e-8, ang / s.clamp_min(1e-12), 2 / w.clamp_min(1e-12))
    return v * scale


def se3_exp(xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """xi [..., 6] = (omega, v) in se(3) -> (R [..., 3, 3], t [..., 3]) with R = exp(hat omega) and
    t = V(omega) v, V = I + (1 - cos t) / t^2 hat + (t - sin t) / t^3 hat^2. float32; exp(0) = identity."""
    with geo.fp32(xi.device):
        xi = xi.float()
        w, v = xi[..., :3], xi[..., 3:]
        th2 = (w * w).sum(-1)[..., None, None]
        A, Bc, C = _so3_coeffs(th2)
        Wm = hat(w)
        W2 = Wm @ Wm
        I = torch.eye(3, device=xi.device).expand_as(Wm)
        R = I + A * Wm + Bc * W2
        V = I + Bc * Wm + C * W2
        return R, (V @ v.unsqueeze(-1)).squeeze(-1)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Rigid transform [..., 4, 4] (or 3 x 4) -> xi [..., 6] = (omega, v), the inverse of se3_exp:
    v = V(omega)^-1 t, V^-1 = I - hat / 2 + (1 - A / (2 B)) / t^2 hat^2."""
    with geo.fp32(T.device):
        w = so3_log(T[..., :3, :3])
        t = T[..., :3, 3].float()
        th2 = (w * w).sum(-1)[..., None, None]
        A, Bc, _ = _so3_coeffs(th2)
        small = th2 < 1e-2
        D = torch.where(small, 1 / 12 + th2 / 720 + th2 * th2 / 30240,
                        (1 - A / (2 * Bc)) / torch.where(small, torch.ones_like(th2), th2))
        Wm = hat(w)
        Vinv = torch.eye(3, device=T.device).expand_as(Wm) - 0.5 * Wm + D * (Wm @ Wm)
        return torch.cat([w, (Vinv @ t.unsqueeze(-1)).squeeze(-1)], -1)


def relative_pose(c2w: torch.Tensor) -> torch.Tensor:
    """Camera-to-world [B, V, 4, 4] -> T_i = E_i E_0^-1 [B, V, 4, 4] (anchor camera -> camera i), E = w2c."""
    with geo.fp32(c2w.device):
        c2w = c2w.float()
        return torch.linalg.inv(c2w) @ c2w[:, :1]


# ---------------------------------------------------------------------------------- sparse operator
class _GatherWeighted(torch.autograd.Function):
    """out[b, m] = sum_k w[b, m, k] V[b, idx[b, m, k]]: V [B, N, D], idx [B, M, k], w [B, M, k] -> [B, M, D].
    The forward and backward loop over k, so no [B, M, k, D] tensor is formed or saved (O(k M D) work)."""

    @staticmethod
    def forward(ctx, V, idx, w):
        dt = torch.promote_types(V.dtype, torch.float32)
        B, M, k = idx.shape
        D = V.shape[-1]
        Vc, wc = V.to(dt), w.to(dt)
        out = torch.zeros(B, M, D, device=V.device, dtype=dt)
        for j in range(k):
            out += wc[..., j, None] * torch.gather(Vc, 1, idx[..., j, None].expand(B, M, D))
        ctx.save_for_backward(V, idx, w)
        return out

    @staticmethod
    def backward(ctx, g):
        V, idx, w = ctx.saved_tensors
        B, M, k = idx.shape
        D = V.shape[-1]
        dt = g.dtype
        gV = gw = None
        if ctx.needs_input_grad[0]:
            gV = torch.zeros(B, V.shape[1], D, device=V.device, dtype=dt)
            for j in range(k):
                gV.scatter_add_(1, idx[..., j, None].expand(B, M, D), w[..., j, None].to(dt) * g)
            gV = gV.to(V.dtype)
        if ctx.needs_input_grad[2]:
            Vc = V.to(dt)
            gw = torch.stack([(g * torch.gather(Vc, 1, idx[..., j, None].expand(B, M, D))).sum(-1)
                              for j in range(k)], -1).to(w.dtype)
        return gV, None, gw


@dataclass
class SparseTransportOp:
    """W_i as at most k entries per target row. idx [B', N, k] anchor cells, w [B', N, k] weights (0 where not
    valid; empty rows all 0), support [B', N] bool; p [B', N, 2] projected anchor cells (patch units), z
    [B', N], ok [B', N] (in front, finite, inside the crop) are diagnostics, None for the identity."""
    idx: torch.Tensor
    w: torch.Tensor
    support: torch.Tensor
    p: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    ok: Optional[torch.Tensor] = None

    @property
    def batch(self) -> int:
        return self.idx.shape[0]

    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """W_i V. V [B', N, D] (one source per row of the operator) or [B, N, D] shared by the B' / B targets of
        each sample (B' = B T, sample-major) -> [B', N, D] float32 (float64 for float64 V)."""
        Bp, N, k = self.idx.shape
        Bv = V.shape[0]
        if Bp % Bv:
            raise ValueError(f"operator batch {Bp} is not a multiple of the source batch {Bv}")
        t = Bp // Bv
        idx, w = self.idx.reshape(Bv, t * N, k), self.w.reshape(Bv, t * N, k)
        return _GatherWeighted.apply(V, idx, w).reshape(Bp, N, V.shape[-1])

    def dense(self) -> torch.Tensor:
        """[B', N, N] rows = target cells, columns = anchor cells (tests, offline analysis)."""
        Bp, N, _ = self.idx.shape
        return torch.zeros(Bp, N, N, device=self.w.device, dtype=self.w.dtype).scatter(-1, self.idx, self.w)


def identity_op(B: int, N: int, device) -> SparseTransportOp:
    idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, 1).contiguous()
    return SparseTransportOp(idx=idx, w=torch.ones(B, N, 1, device=device),
                             support=torch.ones(B, N, dtype=torch.bool, device=device))


# ---------------------------------------------------------------------------------- transport
def blend_motion(beta: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """beta [B, N, n_motion] (rows sum to 1), M [B, n_motion, 3] -> per-cell displacement [B, N, 3]."""
    return torch.einsum("bnm,bmk->bnk", beta.float(), M.float())


def transport(rho: torch.Tensor, kappa0: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
              disp, kappa_i: torch.Tensor, grid: int, knn: int, r_cut: float, sigma_u: float,
              tau_z: float, z_eps: float = 1e-2, smooth_cutoff: bool = False) -> SparseTransportOp:
    """rho [B, N], kappa0 / kappa_i [B, 4], R [B, 3, 3], t [B, 3], disp [B, N, 3] or None -> the sparse W_i.
    float32; gradients reach rho, the pose, the motion and the intrinsics through the kept logits (the
    neighbour selection itself is not differentiable)."""
    dev = rho.device
    with geo.fp32(dev):
        uv = geo.cell_centres(grid, dev)                                             # [N, 2]
        P = geo.unproject(rho.float(), K_cells(kappa0, grid), uv)
        if disp is not None:
            P = P + disp.float()
        P1 = torch.einsum("bij,bnj->bni", R.float(), P) + t.float().unsqueeze(1)
        z = P1[..., 2]
        Ki = K_cells(kappa_i, grid)
        finite = torch.isfinite(P1).all(-1)
        front = finite & (z > z_eps)
        zc = torch.where(front, z, torch.ones_like(z))
        x = Ki[:, 0, 0:1] * P1[..., 0] / zc + Ki[:, 0, 2:3]
        y = Ki[:, 1, 1:2] * P1[..., 1] / zc + Ki[:, 1, 2:3]
        p = torch.stack([x, y], -1)
        inside = torch.isfinite(p).all(-1) & (p >= 0).all(-1) & (p <= grid).all(-1)
        ok = front & inside
        p = torch.where(ok.unsqueeze(-1), p, torch.zeros_like(p))                    # no NaN into the graph
        z = torch.where(ok, z, torch.ones_like(z))
        d2 = (uv.view(1, -1, 1, 2) - p.unsqueeze(1)).pow(2).sum(-1)                  # [B, Nq, N] (search only)
        B, N = rho.shape
        k = min(knn, N)
        with torch.no_grad():
            far = d2.masked_fill(~ok.unsqueeze(1), math.inf)
            vals, idx = far.topk(k, dim=-1, largest=False)
            keep = vals < r_cut * r_cut                                              # inf never passes
        d2k = torch.gather(d2, -1, idx)
        zk = torch.gather(z.unsqueeze(1).expand(B, N, N), -1, idx)
        logit = -d2k / (2 * sigma_u * sigma_u) - zk / tau_z
        if smooth_cutoff:
            logit = logit + torch.log((1 - d2k / (r_cut * r_cut)).clamp_min(1e-6))
        any_keep = keep.any(-1, keepdim=True)
        logit = logit.masked_fill(~keep, -math.inf)
        logit = torch.where(any_keep, logit, torch.zeros_like(logit))                # empty rows: 0 before the max
        logit = logit - logit.max(-1, keepdim=True).values.detach()
        e = torch.exp(logit) * keep
        w = e / e.sum(-1, keepdim=True).clamp_min(1e-30) * any_keep                 # empty row: exactly 0
    return SparseTransportOp(idx=idx, w=w, support=any_keep.squeeze(-1), p=p, z=z, ok=ok)


def build_sparse_transport(rho, kappa_0, beta, xi_i, M_i, kappa_i, grid: int, r_cut: float, sigma_u: float,
                           tau_z: float, knn: int = 9, z_eps: float = 1e-2, smooth_cutoff: bool = False,
                           geometry: bool = True) -> tuple[SparseTransportOp, torch.Tensor]:
    """Module 1 of the spec: rho [B', N], kappa_0 [B', 4], beta [B', N, n_motion], xi_i [B', 6], M_i
    [B', n_motion, 3] (None: no motion), kappa_i [B', 4] -> (W_i, per-cell displacement [B', N, 3]).
    geometry False: the identity (ablation)."""
    B, N = rho.shape
    if not geometry:
        return identity_op(B, N, rho.device), torch.zeros(B, N, 3, device=rho.device)
    R, t = se3_exp(xi_i)
    disp = blend_motion(beta, M_i) if M_i is not None else torch.zeros(B, N, 3, device=rho.device)
    op = transport(rho, kappa_0, R, t, disp, kappa_i, grid, knn, r_cut, sigma_u, tau_z, z_eps, smooth_cutoff)
    return op, disp
