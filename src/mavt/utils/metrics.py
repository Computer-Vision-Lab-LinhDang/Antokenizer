from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return -10.0 * torch.log10(mse.clamp_min(1e-8))


def ssim_like(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_x = prediction.mean()
    mu_y = target.mean()
    sigma_x = prediction.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((prediction - mu_x) * (target - mu_y)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
