from __future__ import annotations

from torch import nn


class StageManager:
    def __init__(self, stage: int = 1, stage2_latent_dim: int = 48) -> None:
        self.stage = stage
        self.stage2_latent_dim = stage2_latent_dim

    def apply(self, model: nn.Module, *, stage: int) -> None:
        self.stage = stage
        if stage >= 2:
            target_dim = self.stage2_latent_dim
            if model.router.continuous.latent_dim < target_dim:
                model.router.continuous.expand_latent_dim(target_dim)
                model.image_decoder.expand_latent_dim(target_dim)
                model.video_decoder.expand_latent_dim(target_dim)

    def trainable_params(self, model: nn.Module):
        return [param for param in model.parameters() if param.requires_grad]
