"""3D box geometry.

Convention (nuScenes): ``Boxes3D.size`` is (w, l, h) and a box's local x axis is its heading (yaw about
world z), so the half-extents along local (x, y, z) are (l/2, w/2, h/2).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from mavt4d.data.sample import Boxes3D


def rot_z(yaw: Tensor) -> Tensor:
    """Rotation matrices (..., 3, 3) about z."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*yaw.shape, 3, 3)


def half_extents(size: Tensor) -> Tensor:
    """(..., 3) half-extents along local (x, y, z) from size (w, l, h)."""
    return torch.stack([size[..., 1], size[..., 0], size[..., 2]], -1) / 2


def world_to_box(p: Tensor, center: Tensor, yaw: Tensor) -> Tensor:
    """World points (N, 3) -> one box's local frame."""
    return (p - center) @ rot_z(yaw)


def box_to_world(q: Tensor, center: Tensor, yaw: Tensor) -> Tensor:
    """One box's local points (N, 3) -> world."""
    return q @ rot_z(yaw).T + center


def points_in_boxes(p: Tensor, boxes: Boxes3D, margin: float = 0.01) -> Tensor:
    """(N,) int64 instance id of the box containing each world point, -1 if none."""
    out = torch.full((p.shape[0],), -1, dtype=torch.int64, device=p.device)
    for i in range(boxes.center.shape[0]):
        q = world_to_box(p, boxes.center[i], boxes.yaw[i])
        inside = (q.abs() <= half_extents(boxes.size[i]) + margin).all(-1)
        out[inside] = boxes.instance_id[i]
    return out


def move_with_boxes(p: Tensor, inst: Tensor, src: Boxes3D, dst: Boxes3D) -> Tensor:
    """Carry world points (N, 3) of box ``inst`` (-1 = static) from box state ``src`` to ``dst``.
    Points of an instance missing from either state stay put."""
    out = p.clone()
    for i in range(src.center.shape[0]):
        j = (dst.instance_id == src.instance_id[i]).nonzero().flatten()
        sel = inst == src.instance_id[i]
        if len(j) and sel.any():
            q = world_to_box(p[sel], src.center[i], src.yaw[i])
            out[sel] = box_to_world(q, dst.center[j[0]], dst.yaw[j[0]])
    return out


def interpolate_boxes(boxes: list[Boxes3D], times: Tensor, tau: float) -> Boxes3D:
    """Box state at time ``tau`` from per-frame states at ``times`` (T,): centres linear, yaw along the
    short arc, clamped outside [times[0], times[-1]]. Instances are those of the earlier bracketing frame."""
    i0 = int((times <= tau).sum().clamp(min=1)) - 1
    i1 = min(i0 + 1, len(boxes) - 1)
    a, b = boxes[i0], boxes[i1]
    span = float(times[i1] - times[i0])
    alpha = 0.0 if span <= 0 else min(max((tau - float(times[i0])) / span, 0.0), 1.0)
    center, yaw = a.center.clone(), a.yaw.clone()
    for i in range(a.center.shape[0]):
        j = (b.instance_id == a.instance_id[i]).nonzero().flatten()
        if len(j):
            center[i] = a.center[i] + alpha * (b.center[j[0]] - a.center[i])
            dyaw = torch.remainder(b.yaw[j[0]] - a.yaw[i] + math.pi, 2 * math.pi) - math.pi
            yaw[i] = a.yaw[i] + alpha * dyaw
    return Boxes3D(center=center, size=a.size, yaw=yaw, instance_id=a.instance_id)
