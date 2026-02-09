"""Noise schedules for discrete diffusion."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch


class NoiseSchedule:
    """Beta schedules for discrete diffusion processes.

    Supports linear, cosine, and sigmoid schedules commonly used
    in diffusion models. For discrete diffusion (D3PM), these
    control the transition probabilities at each timestep.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: Literal["linear", "cosine", "sigmoid", "sqrt"] = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        sigmoid_start: float = -3.0,
        sigmoid_end: float = 3.0,
    ) -> None:
        """Initialize noise schedule.

        Args:
            num_timesteps: Total number of diffusion steps T.
            schedule_type: Type of schedule ("linear", "cosine", "sigmoid", "sqrt").
            beta_start: Starting beta for linear schedule.
            beta_end: Ending beta for linear schedule.
            cosine_s: Offset for cosine schedule to prevent singularity.
            sigmoid_start: Start value for sigmoid schedule.
            sigmoid_end: End value for sigmoid schedule.
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            betas = self._linear_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(num_timesteps, cosine_s)
        elif schedule_type == "sigmoid":
            betas = self._sigmoid_schedule(num_timesteps, sigmoid_start, sigmoid_end)
        elif schedule_type == "sqrt":
            betas = self._sqrt_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.register_schedule(betas)

    def register_schedule(self, betas: torch.Tensor) -> None:
        """Register beta schedule and compute derived quantities."""
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([
            torch.ones(1, dtype=betas.dtype),
            self.alpha_bars[:-1]
        ])

        # For discrete diffusion, we need cumulative transition probabilities
        self.log_alpha_bars = torch.log(self.alpha_bars.clamp(min=1e-10))
        self.log_one_minus_alpha_bars = torch.log((1.0 - self.alpha_bars).clamp(min=1e-10))

    @staticmethod
    def _linear_schedule(
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """Linear beta schedule from beta_start to beta_end."""
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

    @staticmethod
    def _cosine_schedule(
        num_timesteps: int,
        s: float = 0.008,
    ) -> torch.Tensor:
        """Cosine schedule as proposed in 'Improved DDPM'.

        Provides a smoother noise schedule that works better for
        high-resolution generation.
        """
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)

        # f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
        f_t = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return torch.clamp(betas, min=0.0001, max=0.9999)

    @staticmethod
    def _sigmoid_schedule(
        num_timesteps: int,
        start: float = -3.0,
        end: float = 3.0,
    ) -> torch.Tensor:
        """Sigmoid beta schedule for smoother transitions."""
        t = torch.linspace(0, num_timesteps, num_timesteps, dtype=torch.float32)
        v_start = torch.sigmoid(torch.tensor(start))
        v_end = torch.sigmoid(torch.tensor(end))

        sigmoid_vals = torch.sigmoid(start + (end - start) * t / num_timesteps)
        alpha_bars = (v_end - sigmoid_vals) / (v_end - v_start)
        alpha_bars = alpha_bars / alpha_bars[0]

        betas = 1 - (alpha_bars[1:] / torch.cat([torch.ones(1), alpha_bars[:-1]]))
        return torch.clamp(betas, min=0.0001, max=0.9999)

    @staticmethod
    def _sqrt_schedule(num_timesteps: int) -> torch.Tensor:
        """Square root schedule - aggressive early, gentle late."""
        t = torch.linspace(0, 1, num_timesteps, dtype=torch.float32)
        betas = 1 - torch.sqrt(1 - t)
        return torch.clamp(betas, min=0.0001, max=0.9999)

    def get_betas(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get beta values."""
        betas = self.betas
        if device is not None:
            betas = betas.to(device)
        return betas

    def get_alphas(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get alpha values (1 - beta)."""
        alphas = self.alphas
        if device is not None:
            alphas = alphas.to(device)
        return alphas

    def get_alpha_bars(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get cumulative alpha products."""
        alpha_bars = self.alpha_bars
        if device is not None:
            alpha_bars = alpha_bars.to(device)
        return alpha_bars

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar for specific timesteps."""
        return self.alpha_bars.to(t.device)[t]

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Get beta for specific timesteps."""
        return self.betas.to(t.device)[t]


__all__ = ["NoiseSchedule"]
