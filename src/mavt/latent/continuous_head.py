from __future__ import annotations

import torch
from torch import nn


class ContinuousLatentHead(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(embed_dim, latent_dim * 2)

    def expand_latent_dim(self, new_dim: int) -> None:
        if new_dim <= self.latent_dim:
            return
        new_proj = nn.Linear(self.embed_dim, new_dim * 2)
        nn.init.zeros_(new_proj.weight)
        nn.init.zeros_(new_proj.bias)
        with torch.no_grad():
            new_proj.weight[: self.latent_dim].copy_(self.proj.weight[: self.latent_dim])
            new_proj.bias[: self.latent_dim].copy_(self.proj.bias[: self.latent_dim])
            new_proj.weight[new_dim : new_dim + self.latent_dim].copy_(
                self.proj.weight[self.latent_dim :]
            )
            new_proj.bias[new_dim : new_dim + self.latent_dim].copy_(
                self.proj.bias[self.latent_dim :]
            )
        self.latent_dim = new_dim
        self.proj = new_proj

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stats = self.proj(x)
        mu, logvar = stats.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        sample = mu + std * torch.randn_like(std)
        return {"sample": sample, "mu": mu, "logvar": logvar}
