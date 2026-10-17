"""Natural RGB feasibility objective without C/D labels or invented geometry."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from mavt4d.contracts import ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid_pilot.objectives import slice_state


@dataclass(frozen=True)
class TrialConfig:
    architecture: HybridConfig = field(default_factory=HybridConfig)
    schema: str = "natural-epic-hybrid-trial-v1"
    seed: int = 23
    frames: int = 12
    height: int = 128
    width: int = 192
    steps: int = 600
    global_clips: int = 48
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    observed_weight: float = 1.
    missing_weight: float = 1.
    future_weight: float = .5
    history_weight: float = .2
    dev_every: int = 25
    log_every: int = 10
    image_every: int = 100

    def __post_init__(self):
        for name in ("steps", "global_clips", "dev_every", "log_every", "image_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ordered_rgb_target(rgb):
    differences = torch.stack((rgb[:, 7] - rgb[:, 2], rgb[:, 6] - rgb[:, 3]), dim=1)
    pooled = F.adaptive_avg_pool2d(differences.flatten(0, 1), (8, 12)).reshape(len(rgb), -1)
    return pooled / pooled.square().mean(dim=1, keepdim=True).sqrt().clamp_min(.02)


def make_mask(frames: int, height: int, width: int, *, seed: int, family: str | None = None):
    generator = torch.Generator().manual_seed(seed)
    family = family or ("clean", "wide", "tube")[int(torch.randint(3, (), generator=generator))]
    mask = torch.ones(frames, height, width, dtype=torch.bool)
    if family == "clean":
        return mask
    if family == "wide":
        band = max(1, height // 3)
        top = int(torch.randint(height - band + 1, (), generator=generator))
        mask[:, top:top + band] = False
    elif family == "tube":
        band = max(1, width // 3)
        left = int(torch.randint(width - band + 1, (), generator=generator))
        mask[:, :, left:left + band] = False
    else:
        raise ValueError("unknown mask family")
    return mask


class NaturalTrialModule(nn.Module):
    def __init__(self, cfg: TrialConfig):
        super().__init__()
        self.cfg = cfg
        self.model = HybridCDModel(cfg.architecture)
        c_values = cfg.architecture.stage_count * cfg.architecture.state_tokens * cfg.architecture.state_dim
        self.history_head = nn.Linear(c_values, 2 * 3 * 8 * 12)

    def encode(self, rgb, times, masks):
        initial = self.model.initial_state(len(rgb), last_time=times[:, 0])
        steps = []
        for start in range(0, self.cfg.frames, 2):
            observations = []
            for i in range(len(rgb)):
                observed = masks[i:i + 1, start:start + 2].contiguous()
                images = rgb[i:i + 1, start:start + 2]
                observations.append(ObservationStep(
                    images=torch.where(observed.unsqueeze(2), images, 0.).contiguous(),
                    observed=observed, frame_times=times[i:i + 1, start:start + 2].contiguous()))
            steps.append(tuple(observations))
        return self.model.forward_sequence(initial, tuple(steps), method="associative")

    def render(self, state, times, frame):
        # Recording FPS differs; each query retains its own physical target time.
        # At an endpoint every decoder query has relative time exactly zero.
        # Translate only the temporary readout clock; recurrent C/D+KV clocks
        # and the physical times used by transitions remain untouched.
        if torch.equal(state.last_time, times[:, frame]):
            code = replace(state.code(), last_time=torch.zeros_like(state.last_time))
            return self.model.decoder.render(code, height=self.cfg.height,
                                             width=self.cfg.width, time=0.)
        return torch.cat([self.model.decoder.render(slice_state(state, i, i + 1).code(),
                         height=self.cfg.height, width=self.cfg.width, time=float(times[i, frame]))
                          for i in range(len(times))])

    def forward(self, rgb, times, seeds, *, return_panel=False):
        if rgb.shape[1:] != (self.cfg.frames, 3, self.cfg.height, self.cfg.width):
            raise ValueError("natural RGB shape differs from frozen configuration")
        if times.shape != rgb.shape[:2] or times.dtype != torch.float64 or len(seeds) != len(rgb):
            raise ValueError("natural times/seeds mismatch")
        masks = torch.stack([make_mask(self.cfg.frames, self.cfg.height, self.cfg.width, seed=s)
                             for s in seeds]).to(rgb.device)
        states = self.encode(rgb, times, masks)
        observed_sum = rgb.new_zeros(len(rgb))
        missing_sum = rgb.new_zeros(len(rgb))
        observed_count = rgb.new_zeros(len(rgb))
        missing_count = rgb.new_zeros(len(rgb))
        prediction = None
        for index, state in enumerate(states):
            frame = 2 * index + 1
            prediction = self.render(state, times, frame)
            errors = (prediction - rgb[:, frame]).abs()
            available = masks[:, frame, None].expand_as(errors)
            observed_sum = observed_sum + torch.where(available, errors, 0.).flatten(1).sum(1)
            missing_sum = missing_sum + torch.where(~available, errors, 0.).flatten(1).sum(1)
            observed_count = observed_count + available.flatten(1).sum(1)
            missing_count = missing_count + (~available).flatten(1).sum(1)
        observed_loss = (observed_sum / observed_count.clamp_min(1)).mean()
        missing_loss = (missing_sum / missing_count.clamp_min(1)).mean()
        future_states = self.model.rollout(states[2], times[:, [7, 9]].T.contiguous(), method="associative")
        future_predictions = [self.render(state, times, frame) for state, frame in zip(future_states, (7, 9))]
        future_loss = torch.stack([(pred - rgb[:, frame]).abs().mean()
                                   for pred, frame in zip(future_predictions, (7, 9))]).mean()
        history_target = ordered_rgb_target(rgb)
        history_prediction = self.history_head(states[-1].c.flatten(1))
        history_loss = F.mse_loss(history_prediction, history_target)
        loss = (self.cfg.observed_weight * observed_loss + self.cfg.missing_weight * missing_loss
                + self.cfg.future_weight * future_loss + self.cfg.history_weight * history_loss)
        metrics = {"loss": loss.detach(), "observed_l1": observed_loss.detach(),
                   "missing_l1": missing_loss.detach(), "future_l1": future_loss.detach(),
                   "history_mse": history_loss.detach(), "history_zero_mse": history_target.square().mean().detach(),
                   "observed_elements": observed_count.mean().detach(), "missing_elements": missing_count.mean().detach()}
        if return_panel:
            panel = torch.cat((torch.where(masks[:1, -1, None], rgb[:1, -1], 0.), prediction[:1], rgb[:1, -1],
                               future_predictions[-1][:1], rgb[:1, 9]), dim=-1).detach()
            return loss, metrics, panel
        return loss, metrics
