"""Complete immutable training lock, separate from the three-cell v1 pilot.

Performance choices may change microbatch size only while retaining the same
32 source groups (256 clips) per optimizer update. Private runtime settings,
credentials and deadlines are deliberately outside the scientific config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mavt4d.data.event_synthetic import EventSceneConfig
from mavt4d.hybrid.config import HybridConfig
from mavt4d.pilot.config import PilotConfig


@dataclass(frozen=True, slots=True)
class HybridPilotConfig:
    version: str = "hybrid-synthetic-pilot-v2"
    dataset_version: str = "event-cd-v1"
    architecture: HybridConfig = field(default_factory=HybridConfig)
    T: int = 12
    H: int = 128
    W: int = 192
    dt: float = 1 / 12
    source_count: int = 128
    seed_start: int = 0
    val_fraction: float = .2
    dev_fraction: float = .1
    dev_salt: str = "mavt-pilot-dev-v1"
    seed: int = 17
    overfit_seed: int = 17
    overfit_steps: int = 200
    train_steps: int = 1200
    world_size: int = 8
    global_source_groups: int = 32
    microbatch_candidates: tuple[int, ...] = (1, 2, 4)
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (.9, .95)
    weight_decay: float = 1e-4
    grad_clip: float = 1.
    semantic_dim: int = 16
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
    log_every: int = 10
    checkpoint_every: int = 100
    probe_ridge: float = 1.
    bootstrap: int = 1000
    bootstrap_seed: int = 31415
    eval_reconstruction_groups: int = 4
    eval_mask_seed: int = 2718
    overfit_recon_ratio: float = .75
    overfit_temporal_ratio: float = .8

    def __post_init__(self) -> None:
        if self.version != "hybrid-synthetic-pilot-v2":
            raise ValueError("unsupported hybrid pilot version")
        if not isinstance(self.architecture, HybridConfig):
            raise TypeError("architecture must be HybridConfig")
        # Reuse the established validation for the unchanged RGB-only loss,
        # fixture, split and evaluation fields. No v1 experiment is modified.
        shared = {f.name for f in fields(PilotConfig)} & {f.name for f in fields(self)}
        shared.remove("version")
        PilotConfig(**{name: getattr(self, name) for name in shared},
                    patch_size=self.architecture.patch_size,
                    query_count=1, seeds=(self.seed,))
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        for name in ("world_size", "global_source_groups"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.global_source_groups % self.world_size:
            raise ValueError("global_source_groups must be divisible by world_size")
        if not isinstance(self.microbatch_candidates, tuple) or not self.microbatch_candidates:
            raise ValueError("microbatch_candidates must be a nonempty tuple")
        if tuple(sorted(set(self.microbatch_candidates))) != self.microbatch_candidates:
            raise ValueError("microbatch_candidates must be unique and sorted")
        for candidate in self.microbatch_candidates:
            self.gradient_accumulation(candidate)
        if self.H % self.architecture.decoder_tile or self.W % self.architecture.decoder_tile:
            raise ValueError("training image dimensions must be divisible by decoder_tile")

    @property
    def event_config(self) -> EventSceneConfig:
        return EventSceneConfig(self.dataset_version, self.T, self.H, self.W, self.dt, self.val_fraction)

    @property
    def batch_size(self) -> int:
        """Eight crossed core/configuration variants per source group."""
        return 8

    @property
    def seeds(self) -> tuple[int, ...]:
        return (self.seed,)

    @property
    def local_source_groups(self) -> int:
        return self.global_source_groups // self.world_size

    def gradient_accumulation(self, microbatch_groups: int) -> int:
        if type(microbatch_groups) is not int or microbatch_groups <= 0 or self.local_source_groups % microbatch_groups:
            raise ValueError("microbatch must divide the per-rank source-group budget")
        return self.local_source_groups // microbatch_groups

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> HybridPilotConfig:
        if not isinstance(raw, Mapping):
            raise TypeError("hybrid configuration must be a mapping")
        expected = {f.name for f in fields(cls)}
        if set(raw) != expected:
            raise ValueError(f"configuration keys differ: missing={sorted(expected - set(raw))}, unknown={sorted(set(raw) - expected)}")
        values = dict(raw)
        architecture = values["architecture"]
        if not isinstance(architecture, Mapping) or set(architecture) != {f.name for f in fields(HybridConfig)}:
            raise ValueError("architecture keys differ from HybridConfig")
        values["architecture"] = HybridConfig(**architecture)
        for name in ("microbatch_candidates", "betas", "temporal_pool", "mask_kinds", "mask_weights", "future_target_frames"):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"{name}: expected sequence")
            values[name] = tuple(values[name])
        if not isinstance(values["temporal_pairs"], (list, tuple)):
            raise ValueError("temporal_pairs: expected nested sequence")
        values["temporal_pairs"] = tuple(tuple(pair) for pair in values["temporal_pairs"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> HybridPilotConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
