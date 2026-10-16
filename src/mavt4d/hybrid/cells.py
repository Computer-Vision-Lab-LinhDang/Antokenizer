"""Input-dependent 2x2 C/D transitions with layerwise associative scans."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from mavt4d.reference.cells import affine_prefix_scan

from .config import HybridConfig


class HybridCoupledCell(nn.Module):
    """Persist raw affine C/D values; all nonlinear maps only prepare writes."""

    def __init__(self, config: HybridConfig, feedback: bool = True) -> None:
        super().__init__()
        if type(feedback) is not bool:
            raise ValueError("feedback: expected bool")
        self.config = config
        self.feedback = feedback
        self.write_queries = nn.Parameter(torch.randn(config.state_tokens, config.width) * 0.02)
        self.evidence_norm = nn.LayerNorm(config.width)
        self.write_attention = nn.MultiheadAttention(config.width, config.q_heads, batch_first=True, dropout=0)
        self.time_projection = nn.Linear(1, config.width)
        self.write_head = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU(),
                                        nn.Linear(config.width, config.state_dim * 7))
        with torch.no_grad():
            self.write_head[-1].bias.reshape(config.state_dim, 7)[:, 6] = math.log(0.01 / 0.99)
        self.missing_rotation = nn.Parameter(torch.full((config.state_tokens, config.state_dim), 0.03))
        self.missing_damping = nn.Parameter(0.05 * torch.eye(2).expand(config.state_tokens, config.state_dim, 2, 2).clone())

    def _normalize(self, raw: Tensor) -> Tensor:
        cut = raw * raw.new_tensor([[1., 0.], [1., 1.]])
        scale = torch.maximum(torch.linalg.matrix_norm(raw, ord=2),
                              torch.linalg.matrix_norm(cut, ord=2)).clamp_min(1)
        return (raw if self.feedback else cut) / scale[..., None, None]

    def coefficients(self, evidence: Tensor, observed: Tensor, delta_time: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if evidence.ndim != 3 or evidence.shape[1:] != (cfg.summary_tokens, cfg.width):
            raise ValueError("cell evidence: expected [B,P,width]")
        batch = evidence.shape[0]
        if observed.shape != (batch,) or observed.dtype != torch.bool:
            raise ValueError("cell observed: expected bool [B]")
        if delta_time.shape != (batch,) or delta_time.dtype != torch.float64:
            raise ValueError("cell delta_time: expected FP64 [B]")
        if not bool(torch.isfinite(delta_time).all()) or bool((delta_time < 0).any()):
            raise ValueError("cell delta_time: expected finite nonnegative seconds")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32 or evidence.dtype != torch.float32:
            raise ValueError("hybrid cell reference requires FP32 inputs/parameters")
        if any(t.device != parameter.device for t in (evidence, observed, delta_time)):
            raise ValueError("cell inputs and parameters must share device")
        clean = torch.where(observed[:, None, None], evidence, torch.zeros_like(evidence))
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("cell observed evidence must be finite")
        identity = torch.eye(2, device=evidence.device, dtype=torch.float32)
        omega = self.missing_rotation
        zero = torch.zeros_like(omega)
        skew = torch.stack((zero, -omega, omega, zero), dim=-1).reshape(cfg.state_tokens, cfg.state_dim, 2, 2)
        generator = skew - self.missing_damping @ self.missing_damping.transpose(-1, -2)
        missing_a = self._normalize(torch.matrix_exp(delta_time.float()[:, None, None, None, None] * generator))
        missing_b = evidence.new_zeros(batch, cfg.state_tokens, cfg.state_dim, 2)

        indices = observed.nonzero(as_tuple=False).flatten()
        if len(indices) == 0:
            return missing_a, missing_b
        active = self.evidence_norm(clean.index_select(0, indices))
        queries = self.write_queries[None].expand(len(indices), -1, -1)
        context, _ = self.write_attention(queries, active, active, need_weights=False)
        time_context = self.time_projection(torch.log1p(delta_time[indices]).float()[:, None])
        raw = self.write_head(context + time_context[:, None]).reshape(len(indices), cfg.state_tokens, cfg.state_dim, 7)
        matrix = identity + 0.1 * torch.tanh(raw[..., :4].reshape(*raw.shape[:-1], 2, 2))
        alpha = torch.sigmoid(raw[..., 6:7])
        observed_a = (1 - alpha[..., None]) * identity + alpha[..., None] * self._normalize(matrix)
        observed_b = alpha * torch.tanh(raw[..., 4:6])
        return missing_a.index_copy(0, indices, observed_a), missing_b.index_copy(0, indices, observed_b)

    def scan(self, initial_c: Tensor, initial_d: Tensor, evidence: Tensor,
             observed: Tensor, delta_time: Tensor, method: str = "sequential") -> tuple[Tensor, Tensor]:
        """Scan a completed layer's input sequence, not a network step loop."""
        if method not in ("sequential", "associative"):
            raise ValueError("method: expected sequential or associative")
        length, batch = evidence.shape[:2]
        cfg = self.config
        expected = (batch, cfg.state_tokens, cfg.state_dim)
        if initial_c.shape != expected or initial_d.shape != expected:
            raise ValueError("cell initial C/D shape mismatch")
        if initial_c.dtype != torch.float32 or initial_d.dtype != torch.float32:
            raise ValueError("cell carry must be FP32")
        if length == 0:
            return initial_c[None][:0].clone(), initial_d[None][:0].clone()
        a, b = self.coefficients(evidence.flatten(0, 1), observed.flatten(), delta_time.flatten())
        a = a.reshape(length, batch, *a.shape[1:])
        b = b.reshape(length, batch, *b.shape[1:])
        initial = torch.stack((initial_c, initial_d), dim=-1)
        if method == "associative":
            prefix_a, prefix_b = affine_prefix_scan(a, b)
            values = (prefix_a @ initial[None, ..., None]).squeeze(-1) + prefix_b
        else:
            values_list = []
            previous = initial
            for index in range(length):
                previous = (a[index] @ previous[..., None]).squeeze(-1) + b[index]
                values_list.append(previous)
            values = torch.stack(values_list)
        return values[..., 0], values[..., 1]


class StageMixer(nn.Module):
    """Read the current lower C/D state without altering its carry tensors."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config
        self.state_projection = nn.Linear(config.state_dim, config.width)
        self.bank_embedding = nn.Parameter(torch.randn(2, config.width) * 0.02)
        self.query_norm = nn.LayerNorm(config.width)
        self.state_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(config.width, config.q_heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width))

    def forward(self, inputs: Tensor, c: Tensor, d: Tensor, observed: Tensor) -> Tensor:
        length, batch, slots, width = inputs.shape
        active_indices = observed.flatten().nonzero(as_tuple=False).flatten()
        if len(active_indices) == 0:
            return torch.zeros_like(inputs)
        active = inputs.flatten(0, 1).index_select(0, active_indices)
        paired = torch.stack((c, d), dim=-3).flatten(0, 1).index_select(0, active_indices)
        projected = self.state_projection(paired) + self.bank_embedding[None, :, None, :]
        memory = self.state_norm(projected.flatten(1, 2))
        readout, _ = self.attention(self.query_norm(active), memory, memory, need_weights=False)
        output = active + readout
        output = output + self.ffn(self.ffn_norm(output))
        zeros = inputs.new_zeros(length * batch, slots, width)
        return zeros.index_copy(0, active_indices, output).reshape_as(inputs)
