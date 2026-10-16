"""Immutable, canonically hashed configuration for the synthetic pilot.

Changing any field creates another experimental configuration. The published
JSON is the preregistered v1 budget; smaller explicit configurations are useful
for tests but must never be reported as the locked pilot.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from mavt4d.contracts import CDStateSpec
from mavt4d.data.event_synthetic import EventSceneConfig


@dataclass(frozen=True, slots=True)
class PilotConfig:
    version: str = "cd-synthetic-pilot-v1"
    dataset_version: str = "event-cd-v1"
    T: int = 12
    H: int = 32
    W: int = 48
    dt: float = 1 / 12
    source_count: int = 128
    seed_start: int = 0
    val_fraction: float = .2
    dev_fraction: float = .1
    dev_salt: str = "mavt-pilot-dev-v1"
    c_tokens: int = 8
    d_tokens: int = 8
    token_dim: int = 16
    embed_dim: int = 32
    patch_size: int = 4
    modulation_dim: int = 8
    semantic_dim: int = 16
    cells: tuple[str, ...] = ("coupled", "two_scan", "gru")
    seeds: tuple[int, ...] = (17, 29, 43)
    overfit_seed: int = 17
    overfit_steps: int = 300
    train_steps: int = 1200
    batch_size: int = 8
    learning_rate: float = .001
    betas: tuple[float, float] = (.9, .95)
    weight_decay: float = 1e-4
    grad_clip: float = 1.
    query_count: int = 256
    temporal_pool: tuple[int, int] = (8, 12)
    temporal_pairs: tuple[tuple[int, int], ...] = ((2, 7), (3, 6))
    temporal_rms_floor: float = .02
    rec_weight: float = 1.
    missing_weight: float = 1.
    future_weight: float = .5
    temporal_weight: float = 1.
    invariance_weight: float = .1
    noncollapse_weight: float = .05
    variance_target: float = .1
    mask_kinds: tuple[str, ...] = ("clean", "wide", "tube", "snapshot", "never_seen")
    mask_weights: tuple[float, ...] = (.2, .2, .2, .2, .2)
    future_prefix_frames: int = 6
    future_target_frames: tuple[int, ...] = (7, 9)
    log_every: int = 50
    checkpoint_every: int = 300
    probe_ridge: float = 1.
    bootstrap: int = 1000
    bootstrap_seed: int = 31415
    eval_reconstruction_groups: int = 4
    eval_mask_seed: int = 2718
    overfit_recon_ratio: float = .75
    overfit_temporal_ratio: float = .8

    def __post_init__(self) -> None:
        if self.version != "cd-synthetic-pilot-v1" or self.dataset_version != "event-cd-v1":
            raise ValueError("unsupported pilot or dataset version")
        # Reuse the fixture contract, without importing a renderer at runtime.
        self.event_config
        for name in ("source_count", "c_tokens", "d_tokens", "token_dim", "embed_dim",
                     "patch_size", "modulation_dim", "semantic_dim", "overfit_steps",
                     "train_steps", "query_count", "log_every", "checkpoint_every", "bootstrap", "eval_reconstruction_groups"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: expected positive integer")
        for name in ("seed_start", "overfit_seed", "bootstrap_seed", "eval_mask_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value >= 2**63:
                raise ValueError(f"{name}: expected integer in [0, 2**63)")
        if self.seed_start + self.source_count > 2**63:
            raise ValueError("source seed range exceeds 63-bit contract")
        if type(self.batch_size) is not int or self.batch_size != 8:
            raise ValueError("batch_size: v1 uses all eight source variants")
        if self.c_tokens != self.d_tokens:
            raise ValueError("C/D token banks must be equal for the coupled reference")
        if self.H % self.patch_size or self.W % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if self.query_count > self.H * self.W:
            raise ValueError("query_count exceeds pixels per frame")
        positive = ("dt", "learning_rate", "grad_clip", "temporal_rms_floor",
                    "variance_target", "probe_ridge", "overfit_recon_ratio", "overfit_temporal_ratio")
        nonnegative = ("weight_decay", "rec_weight", "missing_weight", "future_weight",
                       "temporal_weight", "invariance_weight", "noncollapse_weight")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name}: expected finite scalar")
            invalid = value <= 0 if name in positive else value < 0
            if invalid:
                raise ValueError(f"{name}: scalar outside allowed range")
        for name in ("val_fraction", "dev_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name}: expected fraction in [0, 1)")
        if not isinstance(self.dev_salt, str) or not self.dev_salt:
            raise ValueError("dev_salt must be nonempty text")
        if self.cells != ("coupled", "two_scan", "gru"):
            raise ValueError("cells: locked three-cell comparison and order required")
        if not isinstance(self.seeds, tuple) or not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or not 0 <= s < 2**63 for s in self.seeds):
            raise ValueError("seeds: expected unique nonnegative 63-bit integer tuple")
        if not isinstance(self.betas, tuple) or len(self.betas) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x < 1 for x in self.betas):
            raise ValueError("betas: expected two finite values in [0,1)")
        if not isinstance(self.temporal_pool, tuple) or len(self.temporal_pool) != 2 or any(type(x) is not int or x <= 0 for x in self.temporal_pool):
            raise ValueError("temporal_pool: expected two positive integers")
        if self.temporal_pool[0] > self.H or self.temporal_pool[1] > self.W:
            raise ValueError("temporal_pool cannot exceed image dimensions")
        if not isinstance(self.temporal_pairs, tuple) or not self.temporal_pairs:
            raise ValueError("temporal_pairs: nonempty tuple required")
        for pair in self.temporal_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2 or any(type(x) is not int or not 0 <= x < self.T for x in pair) or pair[0] == pair[1]:
                raise ValueError("temporal_pairs: distinct valid frame indices required")
        if self.mask_kinds != ("clean", "wide", "tube", "snapshot", "never_seen") or self.mask_weights != (.2,) * 5:
            raise ValueError("mask kinds and equal 20% mixture are locked")
        if type(self.future_prefix_frames) is not int or not 0 < self.future_prefix_frames < self.T or self.future_prefix_frames % 2:
            raise ValueError("future_prefix_frames: expected positive complete tubelet prefix")
        if not isinstance(self.future_target_frames, tuple) or not self.future_target_frames or any(type(t) is not int or not self.future_prefix_frames <= t < self.T or t % 2 != 1 for t in self.future_target_frames) or tuple(sorted(set(self.future_target_frames))) != self.future_target_frames:
            raise ValueError("future_target_frames: sorted unique later tubelet endings required")

    @property
    def state_spec(self) -> CDStateSpec:
        return CDStateSpec(self.c_tokens, self.d_tokens, self.token_dim, 2, 1)

    @property
    def event_config(self) -> EventSceneConfig:
        return EventSceneConfig(self.dataset_version, self.T, self.H, self.W, self.dt, self.val_fraction)

    def to_dict(self) -> dict[str, Any]:
        """JSON-native complete configuration, with no omitted implicit defaults."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PilotConfig":
        if not isinstance(raw, Mapping):
            raise TypeError("pilot configuration must be a mapping")
        expected = {f.name for f in fields(cls)}
        if set(raw) != expected:
            raise ValueError(f"configuration keys differ: missing={sorted(expected - set(raw))}, unknown={sorted(set(raw) - expected)}")
        converted = dict(raw)
        for name in ("cells", "seeds", "betas", "temporal_pool", "mask_kinds", "mask_weights", "future_target_frames"):
            if not isinstance(converted[name], (list, tuple)):
                raise ValueError(f"{name}: expected sequence")
            converted[name] = tuple(converted[name])
        if not isinstance(converted["temporal_pairs"], (list, tuple)):
            raise ValueError("temporal_pairs: expected nested sequence")
        converted["temporal_pairs"] = tuple(tuple(pair) for pair in converted["temporal_pairs"])
        return cls(**converted)

    @classmethod
    def load(cls, path: str | Path) -> "PilotConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
