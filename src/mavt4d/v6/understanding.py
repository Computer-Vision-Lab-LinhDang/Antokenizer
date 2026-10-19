"""Query-style understanding heads over the state.

Slot embedding reads only C_obj and canonical anchors (no D, no h, no slot index), so it is
invariant to pose and equivariant to slot order. Dense features decode the channels the decoder renders
per ray (cfg.dense_dim, composited with detached weights) into the dense teacher's feature space.
"""
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import V6Config
from .contracts import RenderOutput, V6State
from .decoder import fourier


class _SelfBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(-2, -3) for t in self.qkv(self.norm(x)).chunk(3, -1))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(-2, -3).flatten(-2))
        return x + self.ffn(x)


class V6Understanding(nn.Module):
    def __init__(self, cfg: V6Config, num_classes: int = 30, text_dim: int = 1152):
        super().__init__()
        e, w = cfg.decoder_width, cfg.latent_width
        self.cfg, self.embed_dim = cfg, e
        point_dim = 3 * (1 + 2 * cfg.fourier_bands)
        self.token_in = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, e))
        self.anchor_in = nn.Sequential(nn.Linear(point_dim, e), nn.SiLU(), nn.Linear(e, e))
        self.shape_in = nn.Linear(2 * e, e)
        self.blocks = nn.ModuleList(_SelfBlock(e, max(1, e // 64)) for _ in range(2))
        self.out = nn.Sequential(nn.LayerNorm(e), nn.Linear(e, e))
        self.classifier = nn.Linear(e, num_classes)
        self.text = nn.Linear(e, text_dim)
        f = cfg.dense_dim
        self.dense = (nn.Sequential(nn.Linear(f, 2 * f), nn.SiLU(), nn.Linear(2 * f, cfg.dense_target_dim))
                      if f else None)

    def slot_embedding(self, state: V6State) -> Tensor:
        """[B,K,E] from c_obj tokens + canonical anchors, per slot."""
        tokens = self.token_in(state.c_obj.float())                                    # [B,K,6,E]
        points = self.anchor_in(fourier(state.anchor_canon.float(), self.cfg.fourier_bands))  # [B,K,A,E]
        shape = self.shape_in(torch.cat((points.mean(-2), points.amax(-2)), -1))       # [B,K,E]
        x = torch.cat((tokens, shape[..., None, :]), -2).flatten(0, 1)                # [B*K,7,E]
        for block in self.blocks:
            x = block(x)
        return self.out(x.mean(-2)).unflatten(0, state.c_obj.shape[:2])

    def class_logits(self, emb: Tensor) -> Tensor:
        return self.classifier(emb)

    def text_proj(self, emb: Tensor) -> Tensor:
        return F.normalize(self.text(emb).float(), dim=-1)

    def dense_features(self, render: RenderOutput) -> Tensor:
        """[B,R,D] in the dense teacher's feature space, from the rendered channels (cfg.dense_dim > 0)."""
        if self.dense is None or render.feature is None:
            raise ValueError('dense features are off (cfg.dense_dim = 0)')
        return self.dense(render.feature.float()).float()
