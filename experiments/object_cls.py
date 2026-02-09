from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

from atoken.model.encoder import ATokenEncoder
from atoken.model.heads import SemanticHead
from atoken.tasks.object_cls import ObjectClassificationTask


@dataclass
class ATokenClassifierConfig:
    in_channels: int = 3
    num_classes: int = 1000
    patch_size: tuple[int, int, int] = (4, 16, 16)
    stride: Optional[tuple[int, int, int]] = None
    d_model: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    label_smoothing: float = 0.0
    device: str = "cpu"


def build_atoken_classifier(cfg: ATokenClassifierConfig) -> ObjectClassificationTask:
    encoder = ATokenEncoder(
        in_channels=cfg.in_channels,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        d_model=cfg.d_model,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
    )
    head = SemanticHead(dim=cfg.d_model, num_classes=cfg.num_classes, dropout=cfg.dropout)
    task = ObjectClassificationTask(
        encoder=encoder,
        head=head,
        label_smoothing=cfg.label_smoothing,
        device=torch.device(cfg.device),
    )
    return task


@dataclass
class ViTBaselineConfig:
    image_size: int = 224
    patch_size: int = 16
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    mlp_dim: int = 3072
    dropout: float = 0.0
    attention_dropout: float = 0.0
    num_classes: int = 1000


def build_vit_baseline(cfg: ViTBaselineConfig) -> VisionTransformer:
    model = VisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        num_classes=cfg.num_classes,
    )
    return model


__all__ = [
    "ATokenClassifierConfig",
    "ViTBaselineConfig",
    "build_atoken_classifier",
    "build_vit_baseline",
]
