from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

def _flatten_video(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        batch, channels, frames, height, width = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    return x


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    features = x.view(batch, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * height * width)


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        lpips_weight: float = 0.0,
        gram_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.gram_weight = gram_weight
        self.lpips_model = None
        if lpips_weight > 0:
            try:  # pragma: no cover - depends on optional env setup
                import lpips

                self.lpips_model = lpips.LPIPS(net="vgg")
            except Exception:
                self.lpips_model = None
        if self.lpips_model is not None:
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False

    def forward(self, target: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}
        losses["l1"] = F.l1_loss(reconstruction, target)

        flat_target = _flatten_video(target)
        flat_reconstruction = _flatten_video(reconstruction)
        if self.lpips_model is not None:
            losses["lpips"] = self.lpips_model(flat_reconstruction, flat_target).mean()
        else:
            losses["lpips"] = flat_target.new_zeros(())
        losses["gram"] = F.l1_loss(gram_matrix(flat_reconstruction), gram_matrix(flat_target))
        losses["total"] = (
            self.l1_weight * losses["l1"]
            + self.lpips_weight * losses["lpips"]
            + self.gram_weight * losses["gram"]
        )
        return losses
