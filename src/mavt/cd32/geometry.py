"""Geometry of the C-D decoder: unproject the keyframe cells, move them, project them, find neighbours.

Units: positions on the image plane are in cells (the 16 x 16 DINO grid), u in [0, G], cell centres at
(col + 0.5, row + 0.5). Intrinsics are in the same units (pixel K / patch). Camera axes are OpenCV
(x right, y down, z forward). rho is the log inverse depth, so z = exp(-rho).

Forward projection of cell u of keyframe k into frame t (bản 3.2):
    P_k(u) = exp(-rho_k(u)) K^-1 [u, 1]
    P'(u)  = R_t (P_k(u) + m_t(u)) + tau_t          (3D flow added before the pose)
    p(u)   = pi(P'(u)),  z(u) = P'_z(u)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

BIG = 1e3          # distance (cells) given to cells that cannot be read (behind the camera)


def fp32(device) -> torch.autocast:
    """Context that turns autocast off: geometry (matmuls of poses, projections, attention logits over
    distances) must stay in float32, a bf16 position has ~0.06 cell resolution on a 16-cell grid."""
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


# ---------------------------------------------------------------------------------- cameras
def cell_centres(grid: int, device=None) -> torch.Tensor:
    """[G*G, 2] cell centres (x = col + 0.5, y = row + 0.5) in row-major order."""
    r = torch.arange(grid, device=device, dtype=torch.float32) + 0.5
    y, x = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2)


def fov_to_focal(fov_deg: torch.Tensor, grid: int) -> torch.Tensor:
    """Horizontal field of view (degrees) -> focal length in cells."""
    return (grid / 2) / torch.tan(torch.deg2rad(fov_deg) / 2)


def intrinsics_from_focal(f: torch.Tensor, grid: int) -> torch.Tensor:
    """f [...] (cells) -> K [..., 3, 3] with the principal point at the image centre."""
    K = torch.zeros(*f.shape, 3, 3, device=f.device, dtype=f.dtype)
    K[..., 0, 0], K[..., 1, 1] = f, f
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = grid / 2, grid / 2, 1.0
    return K


def intrinsics_to_cells(K_pix: torch.Tensor, patch: int) -> torch.Tensor:
    """Pixel intrinsics of the model input -> intrinsics in cell units."""
    K = K_pix.float().clone()
    K[..., :2, :] = K[..., :2, :] / patch
    return K


def unproject(rho: torch.Tensor, K: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """rho [B, N], K [B, 3, 3] (cells), uv [N, 2] -> P [B, N, 3] in the keyframe camera."""
    z = torch.exp(-rho)
    x = (uv[:, 0] - K[:, 0, 2:3]) / K[:, 0, 0:1]
    y = (uv[:, 1] - K[:, 1, 2:3]) / K[:, 1, 1:2]
    return torch.stack([x * z, y * z, z], -1)


def project(P: torch.Tensor, K: torch.Tensor, z_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """P [B, N, 3], K [B, 3, 3] -> (p [B, N, 2] cells, z [B, N], valid [B, N] bool: z > z_eps)."""
    z = P[..., 2]
    valid = z > z_eps
    zc = torch.where(valid, z, torch.full_like(z, z_eps))
    x = K[:, 0, 0:1] * P[..., 0] / zc + K[:, 0, 2:3]
    y = K[:, 1, 1:2] * P[..., 1] / zc + K[:, 1, 2:3]
    return torch.stack([x, y], -1), z, valid


# ---------------------------------------------------------------------------------- SO(3) / SE(3)
def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Axis-angle [..., 3] -> rotation [..., 3, 3] (Rodrigues; Taylor terms near 0, so exp(0) = I exactly)."""
    th2 = (w * w).sum(-1, keepdim=True).unsqueeze(-1)                       # [..., 1, 1]
    th = th2.clamp_min(1e-12).sqrt()
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    W = torch.stack([O, -wz, wy, wz, O, -wx, -wy, wx, O], -1).reshape(*w.shape[:-1], 3, 3)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(W)
    return I + a * W + b * (W @ W)


def pose_matrix(xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """xi [..., 6] = (axis-angle, translation) -> (R [..., 3, 3], tau [..., 3]), in float32."""
    with fp32(xi.device):
        xi = xi.float()
        return so3_exp(xi[..., :3]), xi[..., 3:]


def transform(P: torch.Tensor, R: torch.Tensor, tau: torch.Tensor, scale: Optional[torch.Tensor] = None):
    """P [B, N, 3] -> R P + tau (then times exp(scale) [B] when given)."""
    out = torch.einsum("bij,bnj->bni", R, P) + tau.unsqueeze(1)
    if scale is not None:
        out = out * torch.exp(scale).view(-1, 1, 1)
    return out


def rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    c = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


# ---------------------------------------------------------------------------------- normals / cull
def cell_normals(P: torch.Tensor, grid: int) -> torch.Tensor:
    """Normals [B, N, 3] of the keyframe surface from central differences of P [B, N, 3], oriented to
    face the camera (n . P < 0)."""
    B = P.shape[0]
    g = P.reshape(B, grid, grid, 3).permute(0, 3, 1, 2)                         # [B, 3, G, G]
    g = F.pad(g, (1, 1, 1, 1), mode="replicate")
    dx = (g[..., 1:-1, 2:] - g[..., 1:-1, :-2]).permute(0, 2, 3, 1).reshape(B, -1, 3)
    dy = (g[..., 2:, 1:-1] - g[..., :-2, 1:-1]).permute(0, 2, 3, 1).reshape(B, -1, 3)
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    s = torch.sign((n * P).sum(-1, keepdim=True)).detach()
    return -torch.where(s == 0, torch.ones_like(s), s) * n


def backface_bias(n: torch.Tensor, R: torch.Tensor, P1: torch.Tensor, kappa: float) -> torch.Tensor:
    """log sigmoid(-kappa cos) - log sigmoid(kappa): ~0 for cells facing the target camera, ~ -kappa for
    cells seen from behind. n [B, N, 3] keyframe normals, R [B, 3, 3], P1 [B, N, 3] moved points."""
    n1 = torch.einsum("bij,bnj->bni", R, n)
    cos = (n1 * P1).sum(-1) / P1.norm(dim=-1).clamp_min(1e-6)
    return F.logsigmoid(-kappa * cos) - F.logsigmoid(torch.tensor(kappa, device=cos.device))


# ---------------------------------------------------------------------------------- neighbours
def knn_read(q: torch.Tensor, p: torch.Tensor, valid: torch.Tensor, k: int) -> tuple:
    """Query positions q [Nq, 2], projected cells p [B, N, 2] (valid [B, N]) ->
    (idx [B, Nq, k] neighbour cells, d2 [B, Nq, k] squared distances, R2 [B, Nq] squared cut radius =
    squared distance to the (k+1)-th neighbour, pj [B, Nq, k, 2] neighbour positions). Distances keep
    their gradient w.r.t. p; indices do not. Unreadable cells sit at distance BIG."""
    with torch.no_grad():
        d2_all = (q.unsqueeze(0).unsqueeze(2) - p.detach().unsqueeze(1)).pow(2).sum(-1)       # [B, Nq, N]
        d2_all = d2_all.masked_fill(~valid.unsqueeze(1), BIG * BIG)
        idx = d2_all.topk(k + 1, dim=-1, largest=False).indices                              # [B, Nq, k+1]
    B, Nq, _ = idx.shape
    pj = torch.gather(p.unsqueeze(1).expand(B, Nq, -1, 2), 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))
    vj = torch.gather(valid.unsqueeze(1).expand(B, Nq, -1), 2, idx)
    d2 = (q.view(1, Nq, 1, 2) - pj).pow(2).sum(-1)
    d2 = torch.where(vj, d2, torch.full_like(d2, BIG * BIG))
    return idx[..., :k], d2[..., :k], d2[..., k].clamp_min(1e-6), pj[..., :k, :]


def cut_window(d2: torch.Tensor, R2: torch.Tensor, eps: float) -> torch.Tensor:
    """w_cut = log max(eps, 1 - d^2 / R^2): 0 at the query, log eps at the cut radius (continuous when a
    cell leaves the neighbour set, since it swaps with the (k+1)-th one exactly at d = R)."""
    return torch.log((1 - d2 / R2.unsqueeze(-1)).clamp_min(eps))


def soft_zbuffer(logz: torch.Tensor, d2: torch.Tensor, w_cut: torch.Tensor, beta: float) -> torch.Tensor:
    """Δlog z_j = log z_j - softmin_i(log z_i + beta d_i^2), all [B, Nq, k].

    The softmin is a weighted mean: pi = softmax(-(log z + beta d^2) + w_cut), reference = sum_i pi_i log z_i,
    i.e. the depth of the soft-selected front surface near the query. The cut window inside the softmax
    makes a cell entering / leaving the neighbour set change it continuously, and unlike the log-sum-exp
    form the reference does not drift by log(number of equal neighbours)."""
    pi = torch.softmax(-(logz + beta * d2) + w_cut, dim=-1)
    return logz - (pi * logz).sum(-1, keepdim=True)


# ---------------------------------------------------------------------------------- flow
def convex_upsample(M: torch.Tensor, w_logits: torch.Tensor, grid: int) -> torch.Tensor:
    """Coarse 3D flow M [B, g, g, 3] and per-cell logits w [B, N, 9] -> m [B, N, 3] (RAFT-style convex
    combination of the 3 x 3 coarse neighbours of each fine cell; zero padding outside)."""
    B, g = M.shape[0], M.shape[1]
    s = grid // g
    x = M.permute(0, 3, 1, 2)                                                       # [B, 3, g, g]
    nb = F.unfold(x, 3, padding=1).view(B, 3, 9, g, g)                              # [B, 3, 9, g, g]
    nb = nb.repeat_interleave(s, -2).repeat_interleave(s, -1)                       # [B, 3, 9, G, G]
    nb = nb.reshape(B, 3, 9, grid * grid).permute(0, 3, 2, 1)                       # [B, N, 9, 3]
    w = torch.softmax(w_logits.float(), -1).to(M.dtype).unsqueeze(-1)
    return (w * nb).sum(2)


def fourier(x: torch.Tensor, n_freq: int, scale: float = 1.0) -> torch.Tensor:
    """[..., c] -> [..., c * (1 + 2 n_freq)]: x plus sin / cos of x * scale * 2^i * pi / 2."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=torch.float32)) * (math.pi / 2) * scale
    a = x.float().unsqueeze(-1) * f
    return torch.cat([x.float(), a.sin().flatten(-2), a.cos().flatten(-2)], -1).to(x.dtype)


def fourier_dim(c: int, n_freq: int) -> int:
    return c * (1 + 2 * n_freq)


def median_normalise(rho: torch.Tensor) -> torch.Tensor:
    """rho [B, N] -> rho - median_N(rho), i.e. median depth 1 per keyframe (scale of monocular video)."""
    return rho - rho.median(dim=-1, keepdim=True).values


def view_order(c2w: torch.Tensor) -> torch.Tensor:
    """c2w [B, V, 4, 4] -> permutation [B, V]: view 0 first, the others by increasing angle between their
    optical axis and view 0's (bản 3.2 encode order for posed views)."""
    z = c2w[..., :3, 2]
    cos = (z * z[:, :1]).sum(-1) / (z.norm(dim=-1) * z[:, :1].norm(dim=-1)).clamp_min(1e-8)
    cos[:, 0] = 2.0
    return cos.argsort(dim=1, descending=True)
