from __future__ import annotations

from typing import Dict

import torch

from atoken.core.sparse_tensor import SparseTensor4D
from atoken.model.decoder import ATokenDecoder


@torch.no_grad()
def decode_latents(
    decoder: ATokenDecoder,
    sparse: SparseTensor4D,
) -> Dict[str, torch.Tensor]:
    device = next(decoder.parameters()).device
    sparse = sparse.to(device)
    outputs = decoder(sparse)
    return outputs
