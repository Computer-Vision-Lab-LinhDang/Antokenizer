from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def _split_head_dim(head_dim: int, num_axes: int) -> Tuple[int, ...]:
    base = head_dim // num_axes
    remainder = head_dim % num_axes
    splits = []
    for idx in range(num_axes):
        dim = base + (1 if idx < remainder else 0)
        # Ensure even number for rotary pairing.
        if dim % 2 != 0:
            dim = max(2, dim - 1)
        splits.append(dim)
    if sum(splits) != head_dim:
        # Adjust the last axis to absorb any drift.
        drift = head_dim - sum(splits)
        splits[-1] += drift
    return tuple(splits)


def _apply_rotary(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    rot_indices = tensor.shape[-1]
    tensor_even = tensor[..., ::2]
    tensor_odd = tensor[..., 1::2]
    rotated = torch.stack((-tensor_odd, tensor_even), dim=-1).reshape(
        tensor.shape
    )
    return tensor * cos[..., :rot_indices] + rotated * sin[..., :rot_indices]


def _rope_factors(
    positions: torch.Tensor,
    dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_dim = dim // 2
    if half_dim == 0:
        raise ValueError("Rotary head dimension must be >= 2")
    freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
    inv_freq = base ** (-2 * freq_seq / dim)
    angles = positions.unsqueeze(-1) * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
    return cos, sin


def _ensure_layout(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, bool]:
    if x.dim() != 4:
        raise ValueError("Expected tensor shape (B, seq, heads, head_dim) or (B, heads, seq, head_dim)")
    if x.shape[1] > x.shape[2]:
        # assume (B, seq, heads, dim)
        return x, False
    # Convert from (B, heads, seq, dim) to (B, seq, heads, dim)
    return x.permute(0, 2, 1, 3), True


def apply_rope_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
    axis_dims: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply multi-axis rotary position embedding to query/key tensors.

    Args:
        q, k: Tensors shaped either (B, seq, heads, head_dim) or (B, heads, seq, head_dim).
        positions: Integer/float tensor of shape (B, seq, 4) containing (t, x, y, z).
        base: Rotary base constant.
        axis_dims: Optional per-axis head dim splits; must sum to head_dim.

    Returns:
        Tuple of tensors with the same layout as the inputs but with RoPE applied.
    """

    q_flat, q_transposed = _ensure_layout(q)
    k_flat, k_transposed = _ensure_layout(k)

    if q_flat.shape != k_flat.shape:
        raise ValueError("Query and key must share shape prior to RoPE application")

    b, seq_len, num_heads, head_dim = q_flat.shape
    if positions.shape[:2] != (b, seq_len):
        raise ValueError("Positions must align with (B, seq_len)")

    if axis_dims is None:
        axis_dims = _split_head_dim(head_dim, num_axes=4)
    axis_dims = tuple(axis_dims)
    if sum(axis_dims) != head_dim:
        raise ValueError("Sum of axis_dims must equal head_dim")

    device = q_flat.device
    dtype = q_flat.dtype

    q_slices = []
    k_slices = []
    start = 0
    for axis, dim in enumerate(axis_dims):
        if dim == 0:
            continue
        end = start + dim
        q_slice = q_flat[..., start:end]
        k_slice = k_flat[..., start:end]
        axis_pos = positions[..., axis].to(dtype=dtype)
        cos, sin = _rope_factors(axis_pos, dim=dim, base=base, device=device, dtype=dtype)
        cos = cos.unsqueeze(2)  # (B, seq, 1, dim)
        sin = sin.unsqueeze(2)
        q_rot = _apply_rotary(q_slice, cos, sin)
        k_rot = _apply_rotary(k_slice, cos, sin)
        q_slices.append(q_rot)
        k_slices.append(k_rot)
        start = end

    q_out = torch.cat(q_slices, dim=-1)
    k_out = torch.cat(k_slices, dim=-1)

    if q_transposed:
        q_out = q_out.permute(0, 2, 1, 3).contiguous()
    if k_transposed:
        k_out = k_out.permute(0, 2, 1, 3).contiguous()
    return q_out, k_out
