from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from atoken.core.sparse_tensor import SparseTensor4D
from atoken.losses.recon import ReconstructionLoss
from atoken.model.decoder import ATokenDecoder
from atoken.model.encoder import ATokenEncoder


class ImageReconstructionTask(nn.Module):
    """Stage 1 image reconstruction task wrapper."""

    def __init__(
        self,
        encoder: ATokenEncoder,
        decoder: ATokenDecoder,
        reconstruction_loss: ReconstructionLoss,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        enc_outputs = self.encoder(images, return_sparse=True)
        sparse: SparseTensor4D = enc_outputs["sparse"]  # type: ignore[assignment]
        dec_outputs = self.decoder(sparse)

        reconstruction = dec_outputs["reconstruction"]
        target = images.unsqueeze(2)
        num_patches = sparse.num_tokens

        loss, logs = self.reconstruction_loss(
            reconstruction,
            target,
            num_patches=num_patches,
        )

        outputs = {
            "loss": loss,
            "logs": logs,
            "reconstruction": reconstruction,
            "pooled": enc_outputs["pooled"],
        }
        return outputs
