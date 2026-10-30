"""Camera convention, se(3) and grain transport of Grain-Prior C-D v1 (sections 2.2, 8.1).

Image coordinates are normalised with pixel centres, u = ((x + 0.5) / W, (y + 0.5) / H) in [0, 1]^2; the cell
(row a, col b) of the G x G grid has its centre at ((b + 0.5) / G, (a + 0.5) / G). Intrinsics are normalised,
kappa = (fx / W, fy / H, cx_edge / W, cy_edge / H), K(kappa) maps camera rays to normalised coordinates.
Camera axes: x right, y down, z forward. The canonical frame is the anchor camera; T_i = E_i E_0^-1 maps it to
camera i (E = world-to-camera), T_i = exp(xi_i) with the twist ordered translation first, then rotation:
xi = (v, omega). rho = log(1 / d) with d the camera z-depth (not the ray length).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..cd32.geometry import fp32
from ..cd41 import transport as _t

kappa_from_K = _t.kappa_from_K
relative_pose = _t.relative_pose


# ---------------------------------------------------------------------------------- positional encodings
def fourier(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """[..., c] -> [..., 2 c n_freq]: sin / cos of 2^k pi x, k = 0 .. n_freq - 1 (fixed, not learned)."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=torch.float32)) * math.pi
    a = x.float().unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def pe_dim(c: int, n_freq: int) -> int:
    return 2 * c * n_freq


def cell_centres(grid: int, device=None) -> torch.Tensor:
    """[G*G, 2] normalised cell centres (x, y), row-major."""
    r = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
    y, x = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2)


def bilinear_weights(u: torch.Tensor, side: int) -> torch.Tensor:
    """Bilinear interpolation weights [B, M, side^2] (fp32) of positions u [B, M, 2] on a side x side grid with
    cell centres at (j + 0.5) / side and border padding: the same taps and weights as F.grid_sample(mode=bilinear,
    padding_mode=border, align_corners=False). Differentiable in u through the fractional parts."""
    B, M = u.shape[:2]
    x = (u[..., 0].float() * side - 0.5).clamp(0, side - 1)
    y = (u[..., 1].float() * side - 0.5).clamp(0, side - 1)
    x0, y0 = x.floor(), y.floor()
    fx, fy = x - x0, y - y0
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0 + 1).clamp(max=side - 1), (y0 + 1).clamp(max=side - 1)
    idx = torch.stack([y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1], -1)     # [B, M, 4]
    w = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], -1)
    return u.new_zeros(B, M, side * side, dtype=torch.float32).scatter_add(-1, idx, w)


def read_coarse(tokens: torch.Tensor, u: torch.Tensor, side: int) -> torch.Tensor:
    """Bilinear read of a side x side token grid [B, side^2, C] at normalised positions u [B, M, 2] -> [B, M, C]
    (token centres at (j + 0.5) / side; outside the grid the border value). Computed as bilinear_weights(u) @ tokens
    in fp32: identical to grid_sample, with a bmm backward instead of grid_sampler_2d_backward (atomic adds into a
    tiny map: the slowest GPU op of a training step on ROCm)."""
    with torch.autocast(u.device.type, enabled=False):
        out = torch.bmm(bilinear_weights(u, side), tokens.float())
    return out.to(tokens.dtype)


# ---------------------------------------------------------------------------------- intrinsics
def focal_bounds(fov_min: float, fov_max: float) -> tuple[float, float]:
    """(log f_lo, log f_hi) of normalised focal length from the FOV bounds (degrees)."""
    f = lambda a: 0.5 / math.tan(math.radians(a) / 2)
    return math.log(f(fov_max)), math.log(f(fov_min))


def kappa_logit_init(cfg) -> float:
    lo, hi = focal_bounds(cfg.fov_min, cfg.fov_max)
    p = (math.log(0.5 / math.tan(math.radians(cfg.fov_init) / 2)) - lo) / (hi - lo)
    return math.log(p / (1 - p))


def kappa_from_logits(zeta: torch.Tensor, cfg) -> torch.Tensor:
    """Camera logits eta [..., 4] (log-focal x, y; principal point x, y) -> kappa [..., 4] with the focal
    lengths inside the registered FOV bounds and the principal point within pp_range of the centre."""
    lo, hi = focal_bounds(cfg.fov_min, cfg.fov_max)
    z = zeta.float()
    f = torch.exp(lo + (hi - lo) * torch.sigmoid(z[..., :2]))
    pp = 0.5 + cfg.pp_range * torch.tanh(z[..., 2:])
    return torch.cat([f, pp], -1)


def K_matrix(kappa: torch.Tensor) -> torch.Tensor:
    K = torch.zeros(*kappa.shape[:-1], 3, 3, device=kappa.device, dtype=torch.float32)
    K[..., 0, 0], K[..., 1, 1] = kappa[..., 0], kappa[..., 1]
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = kappa[..., 2], kappa[..., 3], 1.0
    return K


def unproject(d: torch.Tensor, kappa: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """z-depth d [B, M], kappa [B, 4], u [B, M, 2] (or [M, 2]) -> canonical points [B, M, 3]."""
    k = kappa.float().unsqueeze(1)
    u = u.float().expand(d.shape[0], -1, -1) if u.dim() == 2 else u.float()
    x = (u[..., 0] - k[..., 2]) / k[..., 0]
    y = (u[..., 1] - k[..., 3]) / k[..., 1]
    return torch.stack([x, y, torch.ones_like(x)], -1) * d.float().unsqueeze(-1)


def project(P: torch.Tensor, kappa: torch.Tensor, z_eps: float) -> tuple:
    """Camera points [B, M, 3], kappa [B, 4] -> (u [B, M, 2], z [B, M], valid [B, M]). Points behind the camera
    or not finite are invalid; their u is computed at a safe depth (finite, no gradient trouble)."""
    k = kappa.float().unsqueeze(1)
    z = P[..., 2]
    valid = torch.isfinite(P).all(-1) & (z > z_eps)
    zs = torch.where(valid, z, torch.ones_like(z))
    u = torch.stack([k[..., 0] * P[..., 0] / zs + k[..., 2], k[..., 1] * P[..., 1] / zs + k[..., 3]], -1)
    u = torch.where(valid.unsqueeze(-1), u, torch.zeros_like(u))
    return u, z, valid


# ---------------------------------------------------------------------------------- se(3), xi = (v, omega)
def se3_exp(xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """xi [..., 6] = (v, omega) -> (R [..., 3, 3], t [..., 3]); exp(0) = identity; float32."""
    return _t.se3_exp(torch.cat([xi[..., 3:], xi[..., :3]], -1))


def se3_log(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(R, t) -> xi = (v, omega), the inverse of se3_exp (rotation angle in [0, pi])."""
    T = torch.cat([R.float(), t.float().unsqueeze(-1)], -1)
    w = _t.se3_log(T)
    return torch.cat([w[..., 3:], w[..., :3]], -1)


def compose_left(dxi: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """T <- exp(dxi) T (left composition, target-camera convention)."""
    dR, dt = se3_exp(dxi)
    return dR @ R, (dR @ t.unsqueeze(-1)).squeeze(-1) + dt


def transport_grains(P: torch.Tensor, beta: torch.Tensor, M: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                     kappa: torch.Tensor, z_eps: float) -> dict:
    """Anchor grains P [B, G, 3], blend weights beta [B, G, n_m], controls M [B, n_m, 3], target camera (R, t),
    kappa -> dict(P_can [B, G, 3] at observation i, P_cam, u [B, G, 2], z, valid)."""
    with fp32(P.device):
        P_can = P.float() + beta.float() @ M.float()
        P_cam = (R.float().unsqueeze(1) @ P_can.unsqueeze(-1)).squeeze(-1) + t.float().unsqueeze(1)
        u, z, valid = project(P_cam, kappa, z_eps)
    return dict(P_can=P_can, P_cam=P_cam, u=u, z=z, valid=valid)


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))
