"""Cameras and the sparse geometric transport W_i of C-D 4.1.

Conventions: camera x right, y down, z forward (OpenCV). The continuous image origin is the outer top-left
corner, so the pixel with integer index (u, v) has its centre at (u + 0.5, v + 0.5). Intrinsics are kept
normalised, kappa = (fx / W, fy / H, cx / W, cy / H) with the principal point in edge coordinates; the
transport works in patch units (normalised coordinates x 16 on the 16 x 16 grid), so cell (row a, col b)
has its centre at (b + 0.5, a + 0.5).

Transport of anchor cell j into observation i (rho is the log inverse z-depth, not the ray distance):

    P_j    = exp(-rho_j) K(kappa_0)^-1 [x_j, y_j, 1]
    P_ij   = R_i (P_j + sum_m beta_jm M_i[m]) + t_i,      u_ij = pi(K(kappa_i) P_ij)

Row u' of W_i (target cell) holds a softmax over at most `knn` anchor cells j whose projection is in front
of the camera, finite, inside the crop and closer than r_cut to u', with logits
-|u' - u_ij|^2 / (2 sigma_u^2) - z_ij / tau_z. A row with no such cell is exactly zero (support 0); every
other row sums to 1. The pose (R_i, t_i) is exp of xi_i = (axis-angle, translation).
"""
from __future__ import annotations

import math

import torch

from ..cd32 import geometry as geo

NEG = -1.0e4        # logit of a neighbour that is not kept (finite, so no inf - inf in the softmax)


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


# ---------------------------------------------------------------------------------- SO(3) log
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


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Rigid transform [..., 4, 4] (or 3 x 4) -> xi [..., 6] = (axis-angle, translation), the inverse of
    cd32.geometry.pose_matrix."""
    return torch.cat([so3_log(T[..., :3, :3]), T[..., :3, 3].float()], -1)


def relative_pose(c2w: torch.Tensor) -> torch.Tensor:
    """Camera-to-world [B, V, 4, 4] -> T_i = E_i E_0^-1 [B, V, 4, 4] (anchor camera -> camera i), E = w2c."""
    with geo.fp32(c2w.device):
        c2w = c2w.float()
        return torch.linalg.inv(c2w) @ c2w[:, :1]


# ---------------------------------------------------------------------------------- transport
def blend_motion(beta: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """beta [B, N, n_motion] (rows sum to 1), M [B, n_motion, 3] -> per-cell displacement [B, N, 3]."""
    return torch.einsum("bnm,bmk->bnk", beta.float(), M.float())


def identity_transport(B: int, N: int, device) -> dict:
    eye = torch.eye(N, device=device).expand(B, N, N)
    return dict(W=eye, support=torch.ones(B, N, dtype=torch.bool, device=device), p=None, z=None, ok=None)


def transport(rho: torch.Tensor, kappa0: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
              disp, kappa_i: torch.Tensor, grid: int, knn: int, r_cut: float, sigma_u: float,
              tau_z: float, z_eps: float = 1e-2) -> dict:
    """rho [B, N], kappa0 / kappa_i [B, 4], R [B, 3, 3], t [B, 3], disp [B, N, 3] or None ->
    dict(W [B, N, N] rows = target cells, columns = anchor cells; support [B, N] bool; p [B, N, 2] projected
    anchor cells (patch units); z [B, N]; ok [B, N] readable). float32; gradients reach rho, the pose,
    the motion and the intrinsics through the kept logits (neighbour selection itself is not differentiable)."""
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
        d2 = (uv.view(1, -1, 1, 2) - p.unsqueeze(1)).pow(2).sum(-1)                  # [B, Nq, N]
        B, N = rho.shape
        k = min(knn, N)
        with torch.no_grad():
            far = d2.masked_fill(~ok.unsqueeze(1), math.inf)
            vals, idx = far.topk(k, dim=-1, largest=False)
            keep = vals < r_cut * r_cut                                              # inf never passes
        d2k = torch.gather(d2, -1, idx)
        zk = torch.gather(z.unsqueeze(1).expand(B, N, N), -1, idx)
        logit = torch.where(keep, -d2k / (2 * sigma_u * sigma_u) - zk / tau_z, torch.full_like(d2k, NEG))
        any_keep = keep.any(-1, keepdim=True)
        logit = logit - logit.max(-1, keepdim=True).values.detach()
        w = torch.exp(logit) * keep
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
        w = torch.where(any_keep, w, torch.zeros_like(w))                            # empty row: exactly 0
        W = torch.zeros(B, N, N, device=dev, dtype=torch.float32).scatter(-1, idx, w)
    return dict(W=W, support=any_keep.squeeze(-1), p=p, z=z, ok=ok)
