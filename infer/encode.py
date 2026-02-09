from __future__ import annotations

from typing import Dict, Optional

import torch

from atoken.core.sparse_tensor import SparseTensor4D
from atoken.model.encoder import ATokenEncoder


@torch.no_grad()
def encode_image_batch(
    encoder: ATokenEncoder,
    images: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    return_sparse: bool = False,
) -> Dict[str, torch.Tensor | SparseTensor4D]:
    device = device or next(encoder.parameters()).device
    images = images.to(device)
    outputs = encoder(images, return_sparse=return_sparse)
    return outputs
