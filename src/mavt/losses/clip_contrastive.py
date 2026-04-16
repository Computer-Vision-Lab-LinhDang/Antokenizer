from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None


class FrozenTextTower(nn.Module):
    def __init__(self, embed_dim: int, model_name: str | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        if model_name and AutoModel is not None and AutoTokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            hidden_size = getattr(self.model.config, "hidden_size", embed_dim)
            self.proj = nn.Linear(hidden_size, embed_dim, bias=False)
        else:
            self.embedding = nn.Embedding(257, embed_dim)
            for param in self.embedding.parameters():
                param.requires_grad = False
            self.proj = nn.Identity()

    def _byte_embed(self, captions: Iterable[str], device: torch.device) -> torch.Tensor:
        outputs = []
        for caption in captions:
            byte_values = list(caption.encode("utf-8"))[:256]
            if not byte_values:
                outputs.append(torch.zeros(self.embed_dim, device=device))
                continue
            tokens = torch.tensor(byte_values, device=device, dtype=torch.long)
            outputs.append(self.embedding(tokens).mean(dim=0))
        return torch.stack(outputs, dim=0)

    def forward(self, captions: list[str], device: torch.device) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            return self._byte_embed(captions, device)
        encoded = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.proj(pooled)


class CLIPContrastiveLoss(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        *,
        text_model_name: str | None = None,
        weight: float = 0.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.text_tower = FrozenTextTower(semantic_dim, model_name=text_model_name)

    def forward(self, semantic_features: torch.Tensor, captions: list[str]) -> torch.Tensor:
        if self.weight <= 0:
            return semantic_features.new_zeros(())
        valid = [bool(caption.strip()) for caption in captions]
        if not any(valid):
            return semantic_features.new_zeros(())
        indices = torch.tensor(valid, device=semantic_features.device, dtype=torch.bool)
        semantic = F.normalize(semantic_features[indices], dim=-1)
        text = F.normalize(
            self.text_tower([caption for caption, keep in zip(captions, valid, strict=True) if keep], semantic.device),
            dim=-1,
        )
        logits = semantic @ text.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
        return loss * self.weight
