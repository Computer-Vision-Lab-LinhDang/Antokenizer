"""Write-once appearance bank: written from the first observed evidence of a stream."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import HybridConfig


class AppearanceWriter(nn.Module):
    """Learned queries attend to evidence [B,P,W] and emit A [B,appearance_tokens,state_dim]."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        if not isinstance(config, HybridConfig) or getattr(config, "appearance_tokens", 0) <= 0:
            raise ValueError("AppearanceWriter requires a config with appearance_tokens > 0")
        self.config = config
        self.queries = nn.Parameter(torch.empty(config.appearance_tokens, config.width))
        nn.init.normal_(self.queries, std=1 / math.sqrt(config.width))
        self.norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(config.width, config.q_heads, batch_first=True, dropout=0.0)
        self.output = nn.Linear(config.width, config.state_dim)

    def forward(self, evidence: Tensor) -> Tensor:
        cfg = self.config
        if (not isinstance(evidence, Tensor) or evidence.dtype != torch.float32 or evidence.ndim != 3
                or evidence.shape[0] <= 0 or evidence.shape[1:] != (cfg.summary_tokens, cfg.width)):
            raise ValueError(f"evidence must be FP32 [B,{cfg.summary_tokens},{cfg.width}]")
        e = self.norm(evidence)
        q = self.queries.unsqueeze(0).expand(e.shape[0], -1, -1)
        attended, _ = self.attention(q, e, e, need_weights=False)
        return self.output(attended)


def write_once(appearance: Tensor, appearance_valid: Tensor, written: Tensor,
               observed: Tensor) -> tuple[Tensor, Tensor]:
    """Rows with observed & ~valid take `written`; every other row keeps its appearance."""
    fresh = observed & ~appearance_valid
    return (torch.where(fresh[:, None, None], written, appearance), appearance_valid | fresh)
