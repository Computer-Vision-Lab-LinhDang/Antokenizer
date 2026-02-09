from __future__ import annotations

from typing import Iterator, List

import torch


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(
        self,
        parameters: Iterator[torch.nn.Parameter],
        decay: float = 0.9999,
        device: torch.device | None = None,
    ) -> None:
        self.decay = decay
        self.param_refs: List[torch.nn.Parameter] = []
        self.shadow_params: List[torch.Tensor] = []
        for param in parameters:
            if not param.requires_grad:
                continue
            data = param.detach().clone()
            if device is not None:
                data = data.to(device)
            self.param_refs.append(param)
            self.shadow_params.append(data)

    def update(self, parameters: Iterator[torch.nn.Parameter] | None = None) -> None:
        del parameters
        for idx, param in enumerate(self.param_refs):
            shadow = self.shadow_params[idx]
            shadow.data.lerp_(param.detach(), 1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        for param, shadow in zip(self.param_refs, self.shadow_params):
            param.data.copy_(shadow)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": [p.clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow_params = [p.clone() for p in state_dict["shadow_params"]]
