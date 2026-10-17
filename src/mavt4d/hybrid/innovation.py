"""Patch-level innovation (what changed since a patch was last fully observed)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import HybridV3Config
from .observation import _attention


def patch_innovation(patches: Tensor, coverage: Tensor, prev_tokens: Tensor, prev_seen: Tensor,
                     first_observed: Tensor, *, appearance_on: bool) -> tuple[Tensor, Tensor, Tensor]:
    """delta [T,B,V,hp,wp,S] from per-step tokens; mask changes are never change.

    A patch is observed at coverage 1. Seen+observed: x - prev. Unseen+observed:
    x (new evidence), or 0 at the stream's first observed step when appearance is
    on. Otherwise 0. prev/seen advance only on observed patches.
    """
    deltas = []
    for t in range(patches.shape[0]):
        x, obs = patches[t], (coverage[t] >= 1 - 1e-6).unsqueeze(-1)
        seen = prev_seen.unsqueeze(-1)
        new = torch.where(first_observed[t][:, None, None, None, None], torch.zeros_like(x), x) \
            if appearance_on else x
        deltas.append(torch.where(obs & seen, x - prev_tokens, torch.where(obs, new, torch.zeros_like(x))))
        prev_tokens = torch.where(obs, x, prev_tokens)
        prev_seen = prev_seen | obs.squeeze(-1)
    return torch.stack(deltas), prev_tokens, prev_seen


class InnovationPool(nn.Module):
    """Pool delta patches to summary tokens; exactly 0 when nothing changed."""

    def __init__(self, config: HybridV3Config) -> None:
        super().__init__()
        self.config = config
        self.keys = nn.Linear(config.spatial_width, config.width)
        self.values = nn.Linear(config.spatial_width, config.width, bias=False)
        self.queries = nn.Parameter(torch.empty(config.summary_tokens, config.width))
        nn.init.normal_(self.queries, std=1 / math.sqrt(config.width))
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, delta: Tensor, valid: Tensor, evidence: Tensor) -> Tensor:
        n = delta.shape[0]
        flat = delta.reshape(n, -1, delta.shape[-1])
        queries = self.queries.unsqueeze(0).expand(n, -1, -1)
        pooled = self.output(_attention(queries, self.keys(flat), self.values(flat),
                                        self.config.q_heads, valid.reshape(n, -1).float()))
        # tiny keeps sqrt's backward finite at an exactly-zero pool (forward stays 0)
        rms_pool = (pooled.square().mean(-1, keepdim=True) + torch.finfo(pooled.dtype).tiny).sqrt()
        rms_e = evidence.detach().square().mean((1, 2), keepdim=True).sqrt()
        denom = (rms_pool + self.config.innovation_floor * rms_e).clamp_min(torch.finfo(pooled.dtype).tiny)
        return pooled / denom
