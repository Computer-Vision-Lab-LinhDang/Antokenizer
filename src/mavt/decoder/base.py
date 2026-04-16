from __future__ import annotations

import torch
from torch import nn

from mavt.encoder.siglip2_backbone import Attention, MLP


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = Attention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        query_positions: torch.Tensor,
        latents: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), query_positions, attn_mask=query_mask)
        cross, _ = self.cross_attn(self.norm2(x), latents, latents, need_weights=False)
        x = x + cross
        x = x + self.mlp(self.norm3(x))
        return x


class DecoderViT(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def expand_latent_dim(self, new_dim: int) -> None:
        if new_dim <= self.latent_dim:
            return
        new_proj = nn.Linear(new_dim, self.embed_dim)
        nn.init.zeros_(new_proj.weight)
        nn.init.zeros_(new_proj.bias)
        with torch.no_grad():
            new_proj.weight[:, : self.latent_dim].copy_(self.latent_proj.weight)
            new_proj.bias.copy_(self.latent_proj.bias)
        self.latent_dim = new_dim
        self.latent_proj = new_proj

    def forward(
        self,
        latents: torch.Tensor,
        query_positions: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent_tokens = self.latent_proj(latents)
        x = self.pos_proj(query_positions.float())
        for block in self.blocks:
            x = block(x, query_positions, latent_tokens, query_mask=query_mask)
        return self.norm(x)
