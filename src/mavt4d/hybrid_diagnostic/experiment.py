"""Isolated interventions and read-only measurements; the pilot stays frozen."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from mavt4d.hybrid.decoder import HybridQueryDecoder, _QueryBlock
from mavt4d.hybrid.observation import _attention
from mavt4d.hybrid_pilot.objectives import full_observation_states, render_group
from mavt4d.hybrid_pilot.runner import DistributedContext, save_checkpoint
from mavt4d.pilot.objectives import ordered_history_target


@dataclass(frozen=True)
class DiagnosticConfig:
    version: str = "hybrid-cd-mechanism-v1"
    arm: str = "baseline"
    steps: int = 200
    probe_every: int = 50
    image_every: int = 100
    selection: str = "tiny_clean_history_mse"

    def __post_init__(self):
        if self.version != "hybrid-cd-mechanism-v1" or self.arm not in ("baseline", "write_gate", "d_strength"):
            raise ValueError("unsupported diagnostic version or arm")
        if self.selection != "tiny_clean_history_mse":
            raise ValueError("diagnostic selection is frozen")
        for name in ("steps", "probe_every", "image_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return {**asdict(self), "initial_write_probability": .1 if self.arm == "write_gate" else .01,
                "decoder_d_strength": 1. if self.arm == "d_strength" else .1,
                "source_groups_per_update": 1, "clips_per_update": 8,
                "fresh_initialization": True, "held_out_evaluation": False}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _StrongerDQueryBlock(_QueryBlock):
    """Exactly the reference routing with its scalar multiplier exposed."""
    strength = 1.

    def forward(self, queries, c, key_identity, modulation):
        queries = queries + self.ffn(self.ffn_norm(queries))
        routing = self.strength * torch.tanh(self.d_routing(modulation))
        projected = self.query_projection(self.query_norm(queries + routing.unsqueeze(1)))
        keys = self.content_keys(c) + key_identity.unsqueeze(0)
        values = self.content_values(c)
        content = self.output_projection(_attention(projected, keys, values, self.heads))
        return queries + content, content


class StrongerDDecoder(HybridQueryDecoder):
    """D remains routing/gain only; all RGB content still comes from C."""
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList(_StrongerDQueryBlock(config) for _ in range(config.decoder_blocks))

    def decode_tiles(self, code, query):
        self._validate(code, query)
        tile = self.config.decoder_tile
        grid_index = (query.coordinates[:, :2] - tile / 2) / tile
        if not torch.allclose(grid_index, grid_index.round(), rtol=0, atol=1e-6) or bool((grid_index < 0).any()):
            raise ValueError("tile query must use nonnegative aligned tile centers")
        queries = self._query_features(code, query)
        modulation = self.d_bottleneck(code.d.flatten(start_dim=1))
        identity = (self.stage_keys[:, None] + self.slot_keys[None]).reshape(-1, self.config.width)
        c = code.c.flatten(1, 2)
        for block in self.blocks:
            queries, content = block(queries, c, identity, modulation)
        gain = 1 + torch.tanh(self.d_gain(modulation))
        rgb = self.tile_head(content * gain.unsqueeze(1))
        return rgb.reshape(code.c.shape[0], query.coordinates.shape[0], 3, tile, tile)


def apply_intervention(module, config: DiagnosticConfig):
    """Apply once to canonical fresh weights, before creating the optimizer."""
    if config.arm == "write_gate":
        with torch.no_grad():
            for cell in module.model.cells:
                cell.write_head[-1].bias.reshape(-1, 7)[:, 6] = math.log(.1 / .9)
    elif config.arm == "d_strength":
        reference = module.model.decoder
        # Creating the replacement must not advance the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            replacement = StrongerDDecoder(module.model.config)
        replacement.load_state_dict(reference.state_dict(), strict=True)
        replacement.to(device=next(reference.parameters()).device, dtype=torch.float32)
        module.model.decoder = replacement


def permutations(labels, device):
    if len(labels) != 8 or set(labels) != {(c, d) for c in range(4) for d in range(2)}:
        raise ValueError("diagnostic requires four known cores crossed with two configurations")
    index = {label: i for i, label in enumerate(labels)}
    configuration = torch.tensor([index[(c, 1-d)] for c, d in labels], device=device)
    direction = torch.tensor([index[(c ^ 1, d)] for c, d in labels], device=device)
    return configuration, direction


def _rms(value):
    return float(value.detach().float().square().mean().sqrt())


def _region_l1(value, mask):
    expanded = mask[:, None].expand_as(value)
    return float(torch.where(expanded, value.abs(), 0.).sum() / expanded.sum().clamp_min(1))


def _parameter_group(name):
    if name.startswith("history_head"):
        return "history_head"
    if name.startswith("semantic_head"):
        return "semantic_head"
    if name.startswith("model.observer"):
        return "observer"
    if name.startswith("model.decoder"):
        return "decoder_d" if any(part in name for part in ("d_bottleneck", "d_gain", "d_routing")) else "decoder_content"
    if name.startswith("model.cells"):
        return "cells"
    if name.startswith("model.attentions"):
        return "temporal_attention"
    return "mixer"


def _gradient_metrics(loss, module, state, name):
    named = list(module.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in named] + [state.c, state.d],
                                    allow_unused=True, retain_graph=True)
    squared, counts = {}, {}
    for (key, parameter), grad in zip(named, gradients):
        group = _parameter_group(key)
        counts[group] = counts.get(group, 0) + parameter.numel()
        squared[group] = squared.get(group, 0.) + (float(grad.detach().float().square().sum()) if grad is not None else 0.)
        if key.startswith("model.cells") and ".write_head.2." in key and grad is not None:
            shaped = grad.detach().float().reshape(module.cfg.architecture.state_dim, 7, -1)
            for label, row in (("write_c", 4), ("write_d", 5), ("write_gate", 6)):
                component = f"{key.split('.')[2]}/{label}"
                counts[component] = counts.get(component, 0) + shaped[:, row].numel()
                squared[component] = squared.get(component, 0.) + float(shaped[:, row].square().sum())
    result = {}
    for key, total in squared.items():
        result[f"gradient/{name}/{key}/l2"] = math.sqrt(total)
        result[f"gradient/{name}/{key}/rms"] = math.sqrt(total / counts[key])
    for label, grad in zip(("c_latent", "d_latent"), gradients[-2:]):
        result[f"gradient/{name}/{label}/l2"] = float(grad.norm()) if grad is not None else 0.
    return result


def measure(module, batch, labels, *, gradients=False, images=True):
    """Clean training-source probes; labels only arrange reporting interventions.

    Gradient losses are history MSE and all-endpoint clean reconstruction L1.
    They are diagnostic components, not the complete six-component objective.
    autograd.grad never populates or alters parameter .grad buffers.
    """
    cfg = module.cfg
    device = batch.images.device
    config_swap, direction_swap = permutations(labels, device)
    gates = []
    handles = [cell.write_head[-1].register_forward_hook(
        lambda _, args, output: gates.append(output.detach().reshape(*output.shape[:-1], cfg.architecture.state_dim, 7)[..., 6].sigmoid()))
        for cell in module.model.cells]
    was_training = module.training
    module.eval()
    try:
        with torch.set_grad_enabled(gradients):
            states = full_observation_states(module.model, [batch])
            state = states[-1]
            target = ordered_history_target(batch.images, cfg)
            prediction = module.history_head(state.c.flatten(1))
            history = F.mse_loss(prediction.float(), target)
            renders = [render_group(module.model, s, batch, (i+1)*cfg.architecture.frames_per_step-1)
                       for i, s in enumerate(states)]
            reconstruction = torch.stack([(rgb.float() - batch.images[:, 0, (i+1)*cfg.architecture.frames_per_step-1]).abs().mean()
                                          for i, rgb in enumerate(renders)]).mean()
            zero = target.square().mean()
            result = {"history/mse": float(history.detach()), "history/zero_mse": float(zero),
                      "history/ratio": float(history.detach() / zero.clamp_min(1e-12)),
                      "history/cosine": float(F.cosine_similarity(prediction.detach(), target, dim=1).mean()),
                      "reconstruction/l1": float(reconstruction.detach())}
            if gradients:
                result.update(_gradient_metrics(history, module, state, "history"))
                result.update(_gradient_metrics(reconstruction, module, state, "reconstruction"))
        with torch.no_grad():
            for name, tensor in (("c", state.c), ("d", state.d)):
                result.update({f"state/{name}_rms": _rms(tensor),
                               f"state/{name}_batch_std": float(tensor.std(dim=0, correction=0).mean()),
                               f"state/{name}_configuration_distance": _rms(tensor - tensor[config_swap]),
                               f"state/{name}_direction_distance": _rms(tensor - tensor[direction_swap])})
            for stage, gate in enumerate(gates[:2]):
                result[f"gate/stage_{stage}/mean"] = float(gate.mean())
                result[f"gate/stage_{stage}/min"] = float(gate.min())
                result[f"gate/stage_{stage}/max"] = float(gate.max())
            modulation = module.model.decoder.d_bottleneck(state.d.flatten(1))
            result["decoder/modulation_rms"] = _rms(modulation)
            result["decoder/modulation_configuration_distance"] = _rms(modulation - modulation[config_swap])
            raw_gain = module.model.decoder.d_gain(modulation)
            strength = 1. if isinstance(module.model.decoder, StrongerDDecoder) else .1
            result["decoder/gain_deviation_rms"] = _rms(strength * raw_gain.tanh())
            # Fixed early endpoint avoids using the identical final suffix as
            # the sole test of configuration-dependent image generation.
            frame = 3
            early = states[frame // cfg.architecture.frames_per_step]
            base_rgb = renders[frame // cfg.architecture.frames_per_step].detach()
            target_rgb = batch.images[:, 0, frame]
            moving = (batch.images[:, 0] - batch.images[:, 0, :1]).abs().amax(dim=(1, 2)) > 1e-5
            result["image/moving_pixel_fraction"] = float(moving.float().mean())
            result["image/target_configuration_distance"] = _region_l1(target_rgb-target_rgb[config_swap], moving)
            result["image/target_direction_distance"] = _region_l1(target_rgb-target_rgb[direction_swap], moving)
            result["history/target_direction_distance"] = _rms(target-target[direction_swap])
            altered_images = {}
            for name, altered in (("d_zero", replace(early, d=torch.zeros_like(early.d))),
                                  ("d_swap", replace(early, d=early.d[config_swap].contiguous()))):
                rgb = render_group(module.model, altered, batch, frame)
                altered_images[name] = rgb
                result[f"intervention/{name}/pixel_l1"] = float((rgb-base_rgb).abs().mean())
                result[f"intervention/{name}/moving_pixel_l1"] = _region_l1(rgb-base_rgb, moving)
                result[f"intervention/{name}/moving_target_l1"] = _region_l1(rgb-target_rgb, moving)
                result[f"intervention/{name}/paired_target_gain"] = (
                    _region_l1(base_rgb-target_rgb[config_swap], moving)
                    - _region_l1(rgb-target_rgb[config_swap], moving))
            c_zero = render_group(module.model, replace(early, c=torch.zeros_like(early.c)), batch, frame)
            if torch.count_nonzero(c_zero):
                raise AssertionError("diagnostic decoder violated exact C-zero contract")
            result["intervention/c_zero_exact_zero"] = 1.
            panel = None
            if images:
                mask = batch.observed.clone()
                mask[:, :, :, cfg.H//4:3*cfg.H//4, cfg.W//3:2*cfg.W//3] = False
                masked = full_observation_states(module.model, [batch], [mask], stop_frame=4)[-1]
                masked_rgb = render_group(module.model, masked, batch, frame)
                prefix = states[cfg.future_prefix_frames // cfg.architecture.frames_per_step-1]
                future_frame = cfg.future_target_frames[0]
                future = module.model.rollout(prefix, batch.frame_times[:, 0, future_frame][None], method="associative")[0]
                future_rgb = render_group(module.model, future, batch, future_frame)
                masked_input = torch.where(mask[:, 0, frame, None], target_rgb, 0.)
                rows = [torch.cat([value[i] for value in (masked_input, masked_rgb, target_rgb,
                        future_rgb, batch.images[:, 0, future_frame], altered_images["d_swap"])], dim=-1)
                        for i in (0, int(config_swap[0]))]
                panel = ((torch.cat(rows, dim=-2).clamp(-1, 1)+1)/2).cpu()
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("nonfinite diagnostic measurement")
        return result, panel
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)


def save_best(output, module, optimizer, *, score, best_score, metadata):
    """Keep one atomic best snapshot, selected by the fixed history MSE metric."""
    if not math.isfinite(score):
        raise FloatingPointError("nonfinite checkpoint selection score")
    if score >= best_score:
        return best_score
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(module.parameters()).device
    context = DistributedContext(0, 1, device.index or 0, device)
    candidate = output / f".best-{uuid.uuid4().hex}.pt"
    save_checkpoint(candidate, module, optimizer, {**metadata, "selection_score": score,
                    "selection_metric": "tiny_clean_history_mse"}, context)
    os.replace(candidate, output / "best.pt")
    return score
