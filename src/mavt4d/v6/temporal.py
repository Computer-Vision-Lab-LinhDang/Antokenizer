"""Temporal memory written once per step from a set of tokens: an order-free form of the KDA delta rule.

A memory S [..., K, D] keeps a precision Lam [..., K] per key row. A step first forgets, per row,
    S' = alpha * S,  Lam' = alpha * Lam,  alpha = exp(-dt * softplus(rate))  (KDA's channel-wise gate, over dt)
then writes n tokens with keys A [..., n, K], values V [..., n, D] and strengths beta [..., n] in one solve:
    S = S' + (Diag(Lam') + A^T B A)^-1 A^T B (V - A S'),   Lam = Lam' + diag(A^T B A),   B = diag(beta)
the minimiser of |B^1/2 (A S - V)|^2 + |Lam'^1/2 (S - S')|^2, i.e. an information-filter step on S.
- Token order does not matter (token-by-token KDA would depend on it; the tokens of a step are a set).
- beta 0 and dt 0 leave S and Lam bit-identical.
- One-hot keys (a token belongs to one row) give each row sum beta (v - s) / (Lam' + sum beta): the v6 C rule.
- One token with a unit key and Lam' = lam everywhere is the delta rule with step beta / (lam + beta).
Slots use A = ownership (attention decides the row); a scene memory uses learned unit keys. Reads are A S.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

PRECISION_FLOOR = 1e-4   # keeps the solve well posed for rows that were never written


def forget_factor(rate: Tensor, dt: Tensor) -> Tensor:
    """alpha = exp(-dt * softplus(rate)) in (0, 1]; rate [..., K] (per row), dt broadcast against it (seconds)."""
    return torch.exp(-dt.to(rate.dtype) * F.softplus(rate))


def set_write(memory: Tensor, precision: Tensor, keys: Tensor, values: Tensor, beta: Tensor,
              alpha: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """memory [..., K, D], precision [..., K], keys [..., n, K], values [..., n, D], beta [..., n] (>= 0),
    alpha [..., K] or None (no forgetting). Returns the written (memory, precision), float32."""
    with torch.autocast(memory.device.type, enabled=False):
        s, lam = memory.float(), precision.float()
        if alpha is not None:
            s, lam = alpha.float()[..., None] * s, alpha.float() * lam
        a, v, w = keys.float(), values.float(), beta.float().clamp_min(0)
        aw = a * w[..., None]                                                   # B A
        gram = aw.transpose(-1, -2) @ a                                         # A^T B A [..., K, K]
        rhs = aw.transpose(-1, -2) @ (v - a @ s)                                # A^T B (V - A S')
        system = gram + torch.diag_embed(lam.clamp_min(PRECISION_FLOOR))
        step = torch.cholesky_solve(rhs, torch.linalg.cholesky(system))
        return s + step, lam + gram.diagonal(dim1=-2, dim2=-1)


def read(memory: Tensor, keys: Tensor) -> Tensor:
    """keys [..., n, K] -> values [..., n, D]."""
    return keys.to(memory.dtype) @ memory
