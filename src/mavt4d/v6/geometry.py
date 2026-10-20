"""Small rotation helpers (unit quaternions w, x, y, z; rotation vectors)."""
import torch
from torch import Tensor


def quat_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


def quat_conj(q: Tensor) -> Tensor:
    return q * q.new_tensor([1., -1., -1., -1.])


def quat_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotvec_to_quat(v: Tensor) -> Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    half = angle / 2
    # sin(half)/angle with a series near zero
    k = torch.where(angle < 1e-6, 0.5 - angle.square() / 48, torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat((torch.cos(half), v * k), -1)


def quat_to_rotvec(q: Tensor) -> Tensor:
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    s = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., :1])
    k = torch.where(s < 1e-6, 2 / q[..., :1].clamp_min(1e-12), angle / s.clamp_min(1e-12))
    return q[..., 1:] * k


def rotate(q: Tensor, p: Tensor) -> Tensor:
    """Rotate points p [..., N, 3] by quaternions q [..., 4]."""
    return p @ quat_to_matrix(q).transpose(-1, -2)


def skew(v: Tensor) -> Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)
