from __future__ import annotations

from torch import nn

from mavt.latent.continuous_head import ContinuousLatentHead
from mavt.latent.discrete_fsq import DiscreteFSQHead
from mavt.latent.discrete_vq import MultiCodebookVQ
from mavt.latent.semantic_head import SemanticHead


class LatentRouter(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        latent_dim: int = 32,
        semantic_dim: int = 256,
        discrete_type: str = "fsq",
        fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5),
        vq_num_codebooks: int = 4,
        vq_codebook_size: int = 1024,
    ) -> None:
        super().__init__()
        self.continuous = ContinuousLatentHead(embed_dim, latent_dim=latent_dim)
        self.semantic = SemanticHead(embed_dim, text_embed_dim=semantic_dim)
        if discrete_type == "vq":
            self.discrete = MultiCodebookVQ(
                embed_dim=embed_dim,
                num_codebooks=vq_num_codebooks,
                codebook_size=vq_codebook_size,
            )
        else:
            self.discrete = DiscreteFSQHead(embed_dim=embed_dim, levels=fsq_levels)

    def forward(self, encoder_out: dict, *, stage: int) -> dict:
        tokens = encoder_out["tokens"]
        outputs = {
            "continuous": self.continuous(tokens),
            "semantic": self.semantic(tokens),
        }
        if stage >= 4:
            outputs["discrete"] = self.discrete(tokens)
        return outputs
