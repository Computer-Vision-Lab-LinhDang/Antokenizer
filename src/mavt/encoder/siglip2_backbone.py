from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mavt.encoder.rope4d import RoPE4D


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE4D(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.rope(q, k, positions)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
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
        self.attn = Attention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SigLIP2Backbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        if checkpoint_path:
            self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)

    def _forward_block(
        self,
        block: TransformerBlock,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                lambda hidden, coords, mask: block(hidden, coords, mask),
                x,
                positions,
                attn_mask,
                use_reentrant=False,
            )
        return block(x, positions, attn_mask)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = self._forward_block(block, x, positions, attn_mask)
        return self.norm(x)
