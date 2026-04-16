from __future__ import annotations

import torch


MODALITY_TO_ID = {"image": 0, "video": 1, "threed": 2}


def _repeat_batch(positions: torch.Tensor, batch_size: int) -> torch.Tensor:
    return positions.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def make_positions_image(
    batch_size: int,
    height_blocks: int,
    width_blocks: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    rows = torch.arange(height_blocks, device=device, dtype=dtype)
    cols = torch.arange(width_blocks, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    zeros = torch.zeros_like(grid_x)
    positions = torch.stack([zeros, grid_y, grid_x, zeros], dim=-1).reshape(-1, 4)
    return _repeat_batch(positions, batch_size)


def make_positions_video(
    batch_size: int,
    time_blocks: int,
    height_blocks: int,
    width_blocks: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    times = torch.arange(time_blocks, device=device, dtype=dtype)
    rows = torch.arange(height_blocks, device=device, dtype=dtype)
    cols = torch.arange(width_blocks, device=device, dtype=dtype)
    grid_t, grid_y, grid_x = torch.meshgrid(times, rows, cols, indexing="ij")
    zeros = torch.zeros_like(grid_t)
    positions = torch.stack([grid_t, grid_y, grid_x, zeros], dim=-1).reshape(-1, 4)
    return _repeat_batch(positions, batch_size)


def concat_modality_positions(
    positions: list[torch.Tensor],
    modalities: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    merged = torch.cat(positions, dim=1)
    modality_tensors = []
    for pos, modality in zip(positions, modalities, strict=True):
        ids = torch.full(
            (pos.shape[0], pos.shape[1]),
            MODALITY_TO_ID[modality],
            device=pos.device,
            dtype=torch.long,
        )
        modality_tensors.append(ids)
    return merged, torch.cat(modality_tensors, dim=1)
