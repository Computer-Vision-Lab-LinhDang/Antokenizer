"""Losses of MAVT on DINOv3, v0. Plain functions, each returns a scalar tensor."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def feat_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(1 - cosine).mean() + mse over the last dim."""
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)
    return (1 - cos).mean() + F.mse_loss(pred.float(), target.float())


def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, 1)), mean over all elements."""
    mu, logvar = mu.float(), logvar.float()
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def var_cov(C: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg on C [B, N, D] over the B*N tokens: hinge std >= 1 per channel + off-diag cov^2 mean."""
    x = C.float().reshape(-1, C.shape[-1])
    n, d = x.shape
    x = x - x.mean(0)
    std = (x.var(0) + eps).sqrt()
    var = F.relu(1 - std).mean()
    cov = x.T @ x / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_l = off.pow(2).sum() / max(d * (d - 1), 1)
    return var + cov_l


def sem_loss(zs: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(zs.float(), cls.float(), dim=-1)).mean()


def gate_reg(gate: torch.Tensor) -> torch.Tensor:
    return (1 - gate.float()).mean()


class LPIPSLoss(nn.Module):
    """Frozen LPIPS (VGG) on images in [-1, 1]; the `lpips` package is imported lazily."""

    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips
        except ImportError as e:
            raise ImportError("LPIPSLoss needs the `lpips` package: pip install lpips "
                              "(or set w_lpips = 0)") from e
        self.model = lpips.LPIPS(net=net, verbose=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(pred.float(), target.float()).mean()
