"""Bounded evidence features and conservative, supervision-only visibility labels.

An atlas bit is approximate spatial coverage, never proof of exact surface history.
No predicted atlas or confidence may create a ground-truth label or mask a target.
"""

import torch
from torch import Tensor

OBSERVED, INFERRED, UNKNOWN, MISSING = 0, 1, 2, -1


def _cell_indices(points: Tensor, extent: float) -> tuple[Tensor, Tensor]:
    if extent <= 0:
        raise ValueError("atlas extent must be positive")
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("canonical points must be [B,K,N,3]")
    p = points.detach()
    valid = torch.isfinite(p).all(-1) & (p >= -extent).all(-1) & (p < extent).all(-1)
    xyz = ((torch.nan_to_num(p) + extent) * (4 / extent)).floor().long().clamp(0, 7)
    return xyz[..., 0] + 8 * xyz[..., 1] + 64 * xyz[..., 2], valid


def _check_bits(bits: Tensor, points: Tensor) -> None:
    if bits.shape != (*points.shape[:2], 64) or bits.dtype != torch.uint8:
        raise ValueError("packed atlas must be uint8 [B,K,64]")
    if bits.device != points.device:
        raise ValueError("atlas and points must share a device")


@torch.no_grad()
def atlas_update(bits: Tensor, canonical_points: Tensor, valid: Tensor, extent: float = 4.0) -> Tensor:
    """OR evidence into a fixed 8^3 grid; repeated writes cannot lose other bits."""
    _check_bits(bits, canonical_points)
    index, inside = _cell_indices(canonical_points, extent)
    if valid.shape != inside.shape or valid.dtype != torch.bool:
        raise ValueError("support validity must be boolean [B,K,N]")
    # Scatter counts per cell, then pack. scatter-replace on bytes would lose
    # simultaneous writes to different bits of the same byte.
    cells = torch.zeros(*bits.shape[:2], 512, device=bits.device, dtype=torch.int32)
    cells.scatter_add_(-1, index, (valid & inside).to(torch.int32))
    weights = (1 << torch.arange(8, device=bits.device, dtype=torch.int64))
    packed = ((cells > 0).reshape(*bits.shape[:2], 64, 8) * weights).sum(-1).to(torch.uint8)
    return bits | packed


@torch.no_grad()
def atlas_lookup(bits: Tensor, canonical_points: Tensor, extent: float = 4.0) -> Tensor:
    _check_bits(bits, canonical_points)
    index, inside = _cell_indices(canonical_points, extent)
    value = bits.gather(-1, index // 8).to(torch.int64)
    return (((value >> (index % 8)) & 1).bool() & inside).to(torch.float32)


@torch.no_grad()
def classify_visibility(visible: Tensor, checked: Tensor, current_index: int | None = None) -> Tensor:
    """Classify trusted same-surface history along the last (time) dimension.

    checked=False means geometry, correspondence, or mask evidence is missing;
    it does NOT mean invisible. Include only actual, causally allowed observed
    steps. At rollout current_index=None: future target cameras are not inputs.
    A valid earlier observation suffices for INFERRED even with incomplete
    past history, but current non-visibility must be known if an observation
    is present. UNKNOWN requires all eligible comparisons to be conclusive.
    """
    if visible.shape != checked.shape or visible.ndim < 1:
        raise ValueError("visible and checked must share [...,time] shape")
    if visible.dtype != torch.bool or checked.dtype != torch.bool:
        raise ValueError("visibility and completeness must be boolean")
    if (visible & ~checked).any():
        raise ValueError("positive visibility must have trusted evidence")
    if current_index is not None and not 0 <= current_index < visible.shape[-1]:
        raise ValueError("current_index outside visibility history")
    labels = torch.full(visible.shape[:-1], MISSING, device=visible.device, dtype=torch.long)
    labels[checked.all(-1)] = UNKNOWN
    inferred = visible.any(-1)
    if current_index is not None:
        inferred = inferred & checked[..., current_index] & ~visible[..., current_index]
    labels[inferred] = INFERRED
    if current_index is not None:
        labels[visible[..., current_index]] = OBSERVED
    return labels


@torch.no_grad()
def transport_rigid_points(points_now: Tensor, object_to_world_now: Tensor, object_to_world_history: Tensor) -> Tensor:
    """Transport material points [B,Q,3] to history [B,T,Q,3] using trusted poses.

    Caller must split points by persistent object identity. This is not valid
    for deforming surfaces: those require trusted nonrigid correspondence.
    """
    if points_now.ndim != 3 or points_now.shape[-1] != 3:
        raise ValueError("points must be [B,Q,3]")
    b = points_now.shape[0]
    if object_to_world_now.shape != (b, 4, 4) or object_to_world_history.ndim != 4 or object_to_world_history.shape[0] != b or object_to_world_history.shape[-2:] != (4, 4):
        raise ValueError("poses must be [B,4,4] and [B,T,4,4]")
    if not all(torch.isfinite(t).all() for t in (points_now, object_to_world_now, object_to_world_history)):
        raise ValueError("transport requires finite trusted geometry")
    local = torch.einsum('bij,bqj->bqi', object_to_world_now[:, :3, :3].transpose(-1, -2), points_now - object_to_world_now[:, None, :3, 3])
    return torch.einsum('btij,bqj->btqi', object_to_world_history[..., :3, :3], local) + object_to_world_history[:, :, None, :3, 3]


@torch.no_grad()
def visibility_from_depth(points_history: Tensor, world_to_camera: Tensor, intrinsics: Tensor,
                          depth: Tensor, depth_valid: Tensor, pixel_valid: Tensor,
                          correspondence_valid: Tensor, tolerance: float = 0.05) -> tuple[Tensor, Tensor]:
    """Supervision-only visibility with camera-z and trusted transported surfaces.

    Returns [B,T,Q] booleans (visible, checked). Outside-frustum and deliberately
    masked pixels are known negatives. Missing sparse depth inside an allowed
    view is uncheckable. A point in front of measured geometry is inconsistent,
    not a proven negative. Never pass model-predicted transforms as GT inputs.
    """
    if points_history.ndim != 4 or points_history.shape[-1] != 3:
        raise ValueError("history points must be [B,T,Q,3]")
    b, t, q, _ = points_history.shape
    if world_to_camera.shape != (b, t, 4, 4) or intrinsics.shape != (b, t, 3, 3):
        raise ValueError("camera shapes must match history")
    if depth.ndim != 4 or depth.shape[:2] != (b, t) or depth_valid.shape != depth.shape or pixel_valid.shape != depth.shape:
        raise ValueError("depth and pixel masks must be [B,T,H,W]")
    if depth_valid.dtype != torch.bool or pixel_valid.dtype != torch.bool or correspondence_valid.shape != (b, t, q) or correspondence_valid.dtype != torch.bool:
        raise ValueError("validity tensors must be boolean with matching shapes")
    if not 0 <= tolerance < 1:
        raise ValueError("relative tolerance must be in [0,1)")
    h, w = depth.shape[-2:]
    camera = torch.einsum('btij,btqj->btqi', world_to_camera[..., :3, :3], points_history) + world_to_camera[..., None, :3, 3]
    projected = torch.einsum('btij,btqj->btqi', intrinsics, camera)
    z = camera[..., 2]
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-12)
    finite = torch.isfinite(camera).all(-1) & torch.isfinite(projected).all(-1)
    trusted = correspondence_valid & finite
    inside = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    safe_uv = torch.nan_to_num(uv).floor().long()
    index = safe_uv[..., 1].clamp(0, h - 1) * w + safe_uv[..., 0].clamp(0, w - 1)
    observed = pixel_valid.flatten(-2).gather(-1, index)
    reference = depth.flatten(-2).gather(-1, index)
    measured = depth_valid.flatten(-2).gather(-1, index) & torch.isfinite(reference) & (reference > 0)
    agree = (z - reference).abs() <= tolerance * z
    occluded = z > reference + tolerance * z
    visible = trusted & inside & observed & measured & agree
    checked = trusted & (~inside | ~observed | (measured & (agree | occluded)))
    return visible, checked
