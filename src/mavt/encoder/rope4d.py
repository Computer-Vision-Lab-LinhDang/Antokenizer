from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RoPE4D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8 for 4D RoPE.")
        self.head_dim = head_dim
        self.group_dim = head_dim // 4
        half_dim = self.group_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_axis(self, x: torch.Tensor, axis_positions: torch.Tensor) -> torch.Tensor:
        freqs = axis_positions.float().unsqueeze(-1) * self.inv_freq
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_groups = q.split(self.group_dim, dim=-1)
        k_groups = k.split(self.group_dim, dim=-1)
        q_out = []
        k_out = []
        for axis, (q_group, k_group) in enumerate(zip(q_groups, k_groups, strict=True)):
            axis_positions = positions[..., axis]
            q_out.append(self._apply_axis(q_group, axis_positions))
            k_out.append(self._apply_axis(k_group, axis_positions))
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)
