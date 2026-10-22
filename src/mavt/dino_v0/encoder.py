"""Tier A: frozen DINOv3 + pixel stem -> per-patch tokens X on the 16 x 16 DINO grid.

x = proj([LN(patch - mean_N patch) ; stem(frame)]). Removing the per-image mean drops the
shared DINO "background" direction so tokens carry what differs between patches; the decoder's
feature target is the same centred patch, layer-normed without affine.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DinoV0Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _StubDINO(nn.Module):
    """Tiny random ViT standing in for DINOv3 in tests: returns [CLS, registers, patches]."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        d = cfg.stub_dino_dim
        self.registers = cfg.dino_registers
        self.patchify = nn.Conv2d(3, d, cfg.patch, stride=cfg.patch)
        self.cls = nn.Parameter(torch.randn(1, 1 + cfg.dino_registers, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.Sequential(layer(), layer())
        self.hidden_size = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patchify(x).flatten(2).transpose(1, 2)
        p = p + self.pos[:, : p.shape[1]]
        return self.blocks(torch.cat([self.cls.expand(p.shape[0], -1, -1), p], 1))


class FrozenDINOv3(nn.Module):
    """Frozen DINOv3 (or stub). x in [-1, 1] -> (patch [B', N, out_dim], cls [B', out_dim])."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        self.registers = cfg.dino_registers
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

    def train(self, mode: bool = True) -> "FrozenDINOv3":
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = ((x + 1) * 0.5 - self.mean) / self.std
        h = self.model(pixel_values=x).last_hidden_state if self._hf else self.model(x)
        return h[:, 1 + self.registers:], h[:, 0]


class PixelStem(nn.Module):
    """4 x (conv k4 s2 p1): [B', 3, H, W] -> [B', N, pix_dim] on the DINO patch grid (row-major)."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        assert cfg.patch == 16, "pixel stem downsamples x16"
        c = cfg.pix_dim
        chans = [3, c // 4, c // 2, c, c]
        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1))
            if i < 3:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(2).transpose(1, 2)


class FrameEncoder(nn.Module):
    """frames [B', 3, H, W] in [-1, 1] -> dict(x, feat_target, cls, mean_energy)."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        self.dino = FrozenDINOv3(cfg)
        self.stem = PixelStem(cfg)
        d = self.dino.out_dim
        self.sem_norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d + cfg.pix_dim, cfg.dim)

    def forward(self, frames: torch.Tensor) -> dict:
        patch, cls = self.dino(frames)
        patch, cls = patch.float(), cls.float()
        mean = patch.mean(1, keepdim=True)
        centred = patch - mean
        x = self.proj(torch.cat([self.sem_norm(centred), self.stem(frames)], -1))
        n = patch.shape[1]
        energy = mean.squeeze(1).pow(2).sum(-1) * n / patch.pow(2).sum((1, 2)).clamp_min(1e-12)
        return dict(x=x, feat_target=F.layer_norm(centred, centred.shape[-1:]).detach(),
                    cls=cls.detach(), mean_energy=energy.mean().detach())
