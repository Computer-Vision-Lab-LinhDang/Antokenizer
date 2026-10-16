"""Immutable geometry for the hybrid reference and its serialized state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """The default architecture, with an explicitly smaller correctness fixture.

    The recurrent state always has two stages. Varying a dimension changes the
    configuration fingerprint and therefore invalidates an exact-restart
    snapshot made with the previous configuration. ``tiny`` is for correctness
    tests; it is not an alternative training preset.
    """

    stage_count: int = 2
    state_tokens: int = 32
    state_dim: int = 32
    summary_tokens: int = 16
    width: int = 256
    spatial_width: int = 384
    spatial_heads: int = 6
    spatial_blocks: int = 8
    patch_size: int = 16
    window_size: int = 8
    q_heads: int = 8
    kv_heads: int = 2
    cache_steps: int = 16
    decoder_blocks: int = 4
    decoder_tile: int = 8
    modulation_dim: int = 16
    frames_per_step: int = 2
    reference_views: int = 1
    feedback: bool = True

    def __post_init__(self) -> None:
        for entry in fields(self):
            value = getattr(self, entry.name)
            if entry.name == "feedback":
                if type(value) is not bool:
                    raise ValueError("feedback: expected a boolean")
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{entry.name}: expected a positive integer")
        if self.stage_count != 2:
            raise ValueError("stage_count: the v2 reference requires exactly two stages")
        if self.frames_per_step != 2:
            raise ValueError("frames_per_step: the v2 reference uses complete two-frame tubelets")
        if self.width % self.q_heads:
            raise ValueError("width: must be divisible by q_heads")
        if self.q_heads % self.kv_heads:
            raise ValueError("q_heads: must be divisible by kv_heads")
        if self.spatial_width % self.spatial_heads:
            raise ValueError("spatial_width: must be divisible by spatial_heads")

    @property
    def head_dim(self) -> int:
        return self.width // self.q_heads

    @property
    def ffn(self) -> int:
        return 4 * self.width

    @property
    def spatial_ffn(self) -> int:
        return 4 * self.spatial_width

    def fingerprint(self) -> str:
        """SHA-256 of the complete, canonical configuration, including schema."""
        encoded = json.dumps(
            {"schema": "mavt-hybrid-cd2-kv-v2", "config": asdict(self)},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def tiny(cls, **overrides: int | bool) -> HybridConfig:
        """Small dimensions for CPU correctness tests, with the same algorithms."""
        values = dict(
            state_tokens=2, state_dim=4, summary_tokens=2, width=16,
            spatial_width=24, spatial_heads=4, spatial_blocks=2,
            patch_size=2, window_size=2, q_heads=4, kv_heads=2,
            cache_steps=3, decoder_blocks=2, decoder_tile=2, modulation_dim=4,
        )
        values.update(overrides)
        return cls(**values)
