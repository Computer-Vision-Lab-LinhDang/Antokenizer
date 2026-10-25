"""Frozen DINOv3 read at several depths + pixel stem -> initial cell tokens and read-only memory tokens.

x0   = proj([LN(patch - mean_N patch) ; stem(frame)])       initial cell tokens of both streams (as v0)
mem  = Linear_l(LN_l(F_l)) + e_l, for l in cfg.dino_layers  read-only keys / values in every layer
feat_target = LN(patch - mean_N patch) (no affine), the decoder's DINO target (as v0)

Middle layers keep edges and depth cues better than the last one (DPT-style heads read them), and
re-reading the source in every layer keeps content from being lost with depth.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dino_v0.camera import ZeroLinear
from ..dino_v0.encoder import IMAGENET_MEAN, IMAGENET_STD, PixelStem
from .config import CD32Config


class _StubDINO(nn.Module):
    """Tiny random ViT standing in for DINOv3 in tests: hidden states [embeddings, block 1, ...]."""

    def __init__(self, cfg: CD32Config):
        super().__init__()
        d = cfg.stub_dino_dim
        self.patchify = nn.Conv2d(3, d, cfg.patch, stride=cfg.patch)
        self.cls = nn.Parameter(torch.randn(1, 1 + cfg.dino_registers, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, cfg.n_cells, d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([layer() for _ in range(cfg.stub_dino_blocks)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        p = self.patchify(x).flatten(2).transpose(1, 2) + self.pos
        h = torch.cat([self.cls.expand(p.shape[0], -1, -1), p], 1)
        hs = [h]
        for b in self.blocks:
            h = b(h)
            hs.append(h)
        return self.norm(h), hs


class MultiLayerDINO(nn.Module):
    """Frozen DINOv3 (or stub). x in [-1, 1] -> (last [B', N, D] after the final norm, layers: list of
    [B', N, D] at cfg.dino_layers, cls [B', D]). Never trained, always in eval mode."""

    def __init__(self, cfg: CD32Config):
        super().__init__()
        self.registers, self.layers = cfg.dino_registers, tuple(cfg.dino_layers)
        if cfg.dino_repo is None:
            self.model, self._hf, self._dim = _StubDINO(cfg), False, cfg.stub_dino_dim
        else:
            from transformers import AutoModel
            self.model, self._hf = AutoModel.from_pretrained(cfg.dino_repo), True
            self._dim = self.model.config.hidden_size
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.eval()

    @property
    def out_dim(self) -> int:
        return self._dim

    def train(self, mode: bool = True) -> "MultiLayerDINO":
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = ((x + 1) * 0.5 - self.mean) / self.std
        if self._hf:
            out = self.model(pixel_values=x, output_hidden_states=True)
            last, hs = out.last_hidden_state, out.hidden_states
        else:
            last, hs = self.model(x)
        s = 1 + self.registers
        return last[:, s:], [hs[l][:, s:] for l in self.layers], last[:, 0]


class FrameEncoder(nn.Module):
    """frames [B', 3, H, W] in [-1, 1] -> dict(x0 [B', N, dim], mem [B', L, N, dim], feat_target [B', N, D],
    cls [B', D], patch_last [B', N, D] (centred, for the depth probe))."""

    def __init__(self, cfg: CD32Config):
        super().__init__()
        self.dino = MultiLayerDINO(cfg)
        self.stem = PixelStem(cfg)
        d, L = self.dino.out_dim, len(cfg.dino_layers)
        self.sem_norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d + cfg.pix_dim, cfg.dim)
        self.mem_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.mem_proj = nn.ModuleList([nn.Linear(d, cfg.dim) for _ in range(L)])
        self.mem_layer = nn.Parameter(torch.randn(L, cfg.dim) * 0.02)
        self.ray_proj = ZeroLinear(6, cfg.dim)          # Plücker rays of posed views; zero at init

    def forward(self, frames: torch.Tensor, rays: Optional[torch.Tensor] = None) -> dict:
        last, layers, cls = self.dino(frames)
        last = last.float()
        centred = last - last.mean(1, keepdim=True)
        x0 = self.proj(torch.cat([self.sem_norm(centred), self.stem(frames)], -1))
        mem = torch.stack([p(n(f.float())) + e for f, n, p, e in
                           zip(layers, self.mem_norm, self.mem_proj, self.mem_layer)], 1)
        if rays is not None:
            r = self.ray_proj(rays.to(x0.dtype))
            x0 = x0 + r
            mem = mem + r.unsqueeze(1)
        return dict(x0=x0, mem=mem, feat_target=F.layer_norm(centred, centred.shape[-1:]).detach(),
                    cls=cls.float().detach(), patch=centred.detach())
