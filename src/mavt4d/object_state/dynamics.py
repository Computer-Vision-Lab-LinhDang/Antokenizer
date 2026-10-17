"""Explicit object-wise selective SSM; D itself is the recurrent carry."""

from dataclasses import replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ObjectStateConfig
from .state import ObjectState


class SelectiveDynamics(nn.Module):
    def __init__(self, config: ObjectStateConfig):
        super().__init__()
        self.config = config
        z, nd = config.width, config.dynamic_tokens
        self.summary = nn.Linear((config.content_tokens + nd) * z, z)
        self.interaction_qkv = nn.Linear(z, 3 * z)
        self.interaction_output = nn.Linear(z, z)
        self.transition = nn.Sequential(
            nn.Linear(2 * z + config.action_dim, z * 2), nn.SiLU(), nn.Linear(z * 2, z)
        )
        self.rates = nn.Linear(z, z * nd)
        self.forcing_u = nn.Linear(z, config.write_rank * z)
        self.forcing_b = nn.Linear(z, config.write_rank * nd)
        nn.init.constant_(self.rates.bias, -2.0)

    def relations(self, state: ObjectState):
        x = self.summary(torch.cat((state.c.flatten(-2), state.d.flatten(-2)), dim=-1))
        q, k, v = self.interaction_qkv(x).chunk(3, -1)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.config.width)
        mask = state.allocated[:, None, :]
        scores = (
            logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1) * mask
        )
        scores = scores / scores.sum(-1, keepdim=True).clamp_min(1e-12)
        relation = self.interaction_output(scores @ v)
        return x, torch.where(
            state.allocated[..., None], relation, torch.zeros_like(relation)
        )

    def forward(
        self, state: ObjectState, dt: Tensor | float, action: Tensor | None = None
    ):
        state.validate(self.config)
        b, k = state.d.shape[:2]
        delta = torch.as_tensor(dt, device=state.d.device, dtype=torch.float64)
        if delta.ndim == 0:
            delta = delta.expand(b)
        if delta.shape != (b,) or not torch.isfinite(delta).all() or (delta < 0).any():
            raise ValueError("dt must be finite nonnegative scalar or [B]")
        if self.config.action_dim:
            if (
                action is None
                or action.shape != (b, self.config.action_dim)
                or action.device != state.d.device
                or not torch.isfinite(action).all()
            ):
                raise ValueError("action must match [B,action_dim] and state device")
        elif action is not None:
            raise ValueError("action-free dynamics does not accept action")
        if not bool((delta > 0).any()):
            return state
        summary, relations = self.relations(state)
        inputs = [summary, relations]
        if action is not None:
            inputs.append(action.float()[:, None].expand(-1, k, -1))
        hidden = self.transition(torch.cat(inputs, dim=-1))
        z, nd, rank = (
            self.config.width,
            self.config.dynamic_tokens,
            self.config.write_rank,
        )
        rho = F.softplus(self.rates(hidden)).reshape(b, k, z, nd)
        u = self.forcing_u(hidden).reshape(b, k, rank, z)
        basis = self.forcing_b(hidden).reshape(b, k, rank, nd)
        forcing = torch.einsum("bkrz,bkrn->bkzn", u, basis) / math.sqrt(rank)
        elapsed = delta.float()[:, None, None, None]
        x = rho * elapsed
        decay = torch.exp(-x)
        # For rho=0, the first-order Taylor limit avoids 0/0 and its gradient.
        gain = torch.where(
            rho > 1e-5,
            -torch.expm1(-x) / rho.clamp_min(1e-5),
            elapsed * (1 - 0.5 * x + x.square() / 6),
        )
        h = decay * state.native_d + gain * forcing
        changed = state.allocated & (delta > 0)[:, None]
        d = torch.where(changed[..., None, None], h.transpose(-1, -2), state.d)
        status = torch.where(
            changed, torch.full_like(state.slot_status, 2), state.slot_status
        )
        return replace(state, d=d, slot_status=status, t=state.t + delta)
