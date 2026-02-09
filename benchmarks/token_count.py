from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from atoken.model.encoder import ATokenEncoder


def _resolve_vit_tokens(model: torch.nn.Module) -> int:
    if hasattr(model, "seq_length"):
        return int(getattr(model, "seq_length"))
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "num_patches"):
        num_patches = int(model.patch_embed.num_patches)
        cls_tokens = 1 if getattr(model, "cls_token", None) is not None else 0
        return num_patches + cls_tokens
    raise AttributeError("Unable to infer ViT token count from the provided model.")


@torch.no_grad()
def benchmark_token_counts(
    atoken_encoder: ATokenEncoder,
    vit_model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]] | DataLoader,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute average token usage per image for ATOKEN encoder and ViT."""

    if device is None:
        device = next(atoken_encoder.parameters()).device

    atoken_encoder.eval()
    vit_model.eval()

    total_atoken = 0.0
    total_vit = 0.0
    total_images = 0

    vit_tokens = float(_resolve_vit_tokens(vit_model))

    for batch in dataloader:
        images = batch["image"].to(device)
        batch_size = images.size(0)
        outputs = atoken_encoder(images, return_sparse=True)
        sparse = outputs["sparse"]
        batch_tokens = sparse.mask.sum(dim=1).float()
        total_atoken += batch_tokens.sum().item()
        total_vit += vit_tokens * batch_size
        total_images += batch_size

    if total_images == 0:
        raise ValueError("Dataloader yielded no images for benchmarking.")

    avg_atoken = total_atoken / total_images
    avg_vit = total_vit / total_images
    return {
        "avg_atoken_tokens": avg_atoken,
        "avg_vit_tokens": avg_vit,
        "token_ratio": avg_atoken / avg_vit,
        "num_images": float(total_images),
    }


__all__ = ["benchmark_token_counts"]
