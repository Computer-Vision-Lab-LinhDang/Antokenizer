"""Bounded, step-causal GQA with explicit BF16 keys and values.

All projections depend on attention inputs, never on this layer's previous
output. The sequence reference gathers a bounded window per query step rather
than constructing a dense quadratic temporal attention matrix. It is a
correctness reference, not an optimized attention kernel.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import HybridConfig


class TemporalGQA(nn.Module):
    """Attention across steps, with bidirectional slots inside each step."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        kv_width = config.kv_heads * config.head_dim
        self.input_norm = nn.LayerNorm(width)
        self.q_projection = nn.Linear(width, width, bias=False)
        self.k_projection = nn.Linear(width, kv_width, bias=False)
        self.v_projection = nn.Linear(width, kv_width, bias=False)
        self.out_projection = nn.Linear(width, width, bias=False)
        self.time_bias = nn.Linear(1, config.q_heads, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward_sequence(
        self, inputs: Tensor, initial_k: Tensor, initial_v: Tensor,
        initial_valid: Tensor, initial_times: Tensor, initial_ids: Tensor,
        observed: Tensor, times: Tensor, step_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return outputs and newly projected K/V, all time-major.

        Inputs are ``[T,B,P,width]``; initial caches ``[B,W,P,Hkv,hd]``.
        ``step_ids`` are accepted logical-step indices (no-op entries use -1).
        Whole-missing entries create zero, invalid K/V and bypass the block.
        K/V are rounded to BF16 before *any* read, including the current step.
        """
        cfg = self.config
        if inputs.ndim != 4 or inputs.shape[2:] != (cfg.summary_tokens, cfg.width):
            raise ValueError("attention inputs: expected [T,B,P,width]")
        length, batch = inputs.shape[:2]
        cache_shape = (batch, cfg.cache_steps, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim)
        if initial_k.shape != cache_shape or initial_v.shape != cache_shape:
            raise ValueError("attention cache shape does not match config")
        if initial_k.dtype != torch.bfloat16 or initial_v.dtype != torch.bfloat16:
            raise ValueError("attention cache K/V must be BF16")
        if initial_valid.shape != cache_shape[:3] or initial_valid.dtype != torch.bool:
            raise ValueError("attention cache validity: expected bool [B,W,P]")
        if initial_times.shape != cache_shape[:2] or initial_times.dtype != torch.float64:
            raise ValueError("attention cache times: expected FP64 [B,W]")
        if initial_ids.shape != cache_shape[:2] or initial_ids.dtype != torch.int64:
            raise ValueError("attention cache ids: expected int64 [B,W]")
        if observed.shape != (length, batch) or observed.dtype != torch.bool:
            raise ValueError("attention observed: expected bool [T,B]")
        if times.shape != observed.shape or times.dtype != torch.float64:
            raise ValueError("attention times: expected FP64 [T,B]")
        if step_ids.shape != observed.shape or step_ids.dtype != torch.int64:
            raise ValueError("attention step ids: expected int64 [T,B]")
        parameter = next(self.parameters())
        tensors = (inputs, initial_k, initial_v, initial_valid, initial_times,
                   initial_ids, observed, times, step_ids)
        if parameter.dtype != torch.float32 or inputs.dtype != torch.float32:
            raise ValueError("hybrid attention reference requires FP32 parameters and inputs")
        if any(value.device != parameter.device for value in tensors):
            raise ValueError("attention inputs, cache and parameters must share device")
        clean = torch.where(observed[..., None, None], inputs, torch.zeros_like(inputs))
        if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(times).all()):
            raise ValueError("attention observed inputs and times must be finite")

        # Only observed rows enter normalization/projection. Besides bypassing
        # missing observations, this prevents hidden NaNs from entering a graph.
        indices = observed.flatten().nonzero(as_tuple=False).flatten()
        flat = clean.flatten(0, 1)
        active = flat.index_select(0, indices)
        normalized = self.input_norm(active)
        q_active = self.q_projection(normalized).reshape(-1, cfg.summary_tokens, cfg.q_heads, cfg.head_dim)
        k_active = self.k_projection(normalized).reshape(-1, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim).to(torch.bfloat16)
        v_active = self.v_projection(normalized).reshape_as(k_active).to(torch.bfloat16)
        q = inputs.new_zeros(length * batch, cfg.summary_tokens, cfg.q_heads, cfg.head_dim).index_copy(0, indices, q_active)
        kv_shape = (length * batch, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim)
        k = torch.zeros(kv_shape, dtype=torch.bfloat16, device=inputs.device).index_copy(0, indices, k_active)
        v = torch.zeros_like(k).index_copy(0, indices, v_active)
        q = q.reshape(length, batch, *q.shape[1:])
        k = k.reshape(length, batch, *k.shape[1:])
        v = v.reshape(length, batch, *v.shape[1:])
        if length == 0:
            return inputs.clone(), k, v

        all_k = torch.cat((initial_k, k.transpose(0, 1)), dim=1)
        all_v = torch.cat((initial_v, v.transpose(0, 1)), dim=1)
        all_times = torch.cat((initial_times, times.transpose(0, 1)), dim=1)
        all_ids = torch.cat((initial_ids, step_ids.transpose(0, 1)), dim=1)
        all_valid = torch.cat((initial_valid, observed.T[..., None].expand(-1, -1, cfg.summary_tokens)), dim=1)
        aggregates = []
        repetitions = cfg.q_heads // cfg.kv_heads
        for t in range(length):
            for stream in range(batch):
                if not bool(observed[t, stream]):
                    continue
                current_id = step_ids[t, stream]
                # IDs, not ring positions or timestamps, define the window.
                eligible = ((all_ids[stream] >= current_id - cfg.cache_steps + 1)
                            & (all_ids[stream] <= current_id)
                            & (all_ids[stream] >= 0))
                selection = eligible.nonzero(as_tuple=False).flatten()
                # Keep a canonical chronological reduction order after wrap.
                selection = selection[torch.argsort(all_ids[stream, selection])]
                selected_valid = all_valid[stream].index_select(0, selection).flatten()
                keys = all_k[stream].index_select(0, selection).flatten(0, 1)[selected_valid].float()
                values = all_v[stream].index_select(0, selection).flatten(0, 1)[selected_valid].float()
                key_times = all_times[stream].index_select(0, selection)[:, None].expand(-1, cfg.summary_tokens).flatten()[selected_valid]
                # Current observed slots always make this nonempty. A strict
                # error is preferable to an all-minus-infinity softmax.
                if keys.shape[0] == 0:
                    raise ValueError("observed attention query has no valid keys")
                keys = keys.repeat_interleave(repetitions, dim=1).permute(1, 0, 2)
                values = values.repeat_interleave(repetitions, dim=1).permute(1, 0, 2)
                queries = q[t, stream].permute(1, 0, 2)
                logits = queries @ keys.transpose(-1, -2) / math.sqrt(cfg.head_dim)
                elapsed = (times[t, stream] - key_times).clamp_min(0)
                bias = self.time_bias(torch.log1p(elapsed).float()[:, None]).T
                aggregate = (torch.softmax(logits + bias[:, None, :], dim=-1) @ values)
                aggregate = aggregate.permute(1, 0, 2).reshape(cfg.summary_tokens, cfg.width)
                aggregates.append(aggregate)
        if not aggregates:
            # Preserve the missing-step bypass, including unused tail gradients.
            return torch.zeros_like(inputs), k, v
        # Gather/reduction remains chronological per stream. The row-local tail
        # has no temporal dependencies and can use one GEMM batch for all active
        # rows, in the same time-major order as the projection indices above.
        output = active + self.out_projection(torch.stack(aggregates))
        output = output + self.ffn(self.ffn_norm(output))
        result = inputs.new_zeros(length * batch, cfg.summary_tokens, cfg.width)
        result = result.index_copy(0, indices, output).reshape_as(inputs)
        return result, k, v
