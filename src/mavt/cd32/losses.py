"""Losses of the C-D 3.2 tokenizer that v0 does not have. rgb_l1, feat_loss and kl come from dino_v0."""
from __future__ import annotations

import torch

from ..dino_v0.losses import feat_loss, kl, rgb_l1  # noqa: F401  (re-exported)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """sqrt(x^2 + eps^2) - eps, mean: robust, so occluded or newly visible regions do not bend the flow."""
    d = pred.float() - target.float()
    return ((d * d + eps * eps).sqrt() - eps).mean()


def kl_masked(mu: torch.Tensor, logvar: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, e^logvar) || N(0, 1)) averaged over the kept tokens. mu, logvar [B, L, d], keep [B, L] bool."""
    k = -0.5 * (1 + logvar.float() - mu.float().pow(2) - logvar.float().exp()).mean(-1)
    w = keep.float()
    return (k * w).sum() / w.sum().clamp_min(1.0)


def edge_aware_smooth(rho: torch.Tensor, feat: torch.Tensor, grid: int) -> torch.Tensor:
    """|∂rho| weighted by exp(-|∂F| / mean|∂F|) on the cell grid (SaLon3R-style). rho [B, N], feat [B, N, C]."""
    B = rho.shape[0]
    r = rho.float().view(B, grid, grid)
    f = feat.float().view(B, grid, grid, -1)
    gx_f = (f[:, :, 1:] - f[:, :, :-1]).abs().mean(-1)
    gy_f = (f[:, 1:] - f[:, :-1]).abs().mean(-1)
    wx = torch.exp(-gx_f / gx_f.mean().clamp_min(1e-6))
    wy = torch.exp(-gy_f / gy_f.mean().clamp_min(1e-6))
    return ((r[:, :, 1:] - r[:, :, :-1]).abs() * wx).mean() + ((r[:, 1:] - r[:, :-1]).abs() * wy).mean()


def scale_shift_invariant(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MiDaS-style: both [B, N] aligned by their median and mean absolute deviation, then L1."""
    def norm(x):
        x = x.float()
        med = x.median(-1, keepdim=True).values
        return (x - med) / (x - med).abs().mean(-1, keepdim=True).clamp_min(1e-6)
    return (norm(pred) - norm(target)).abs().mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean PSNR (dB) over images in [-1, 1]: pred / target [B, 3, H, W]."""
    mse = ((pred.float().clamp(-1, 1) - target.float()) / 2).pow(2).flatten(1).mean(-1)
    return (-10 * torch.log10(mse.clamp_min(1e-10))).mean()


def gaussian_sample(mu: torch.Tensor, logvar: torch.Tensor, train: bool) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp() if train else mu
