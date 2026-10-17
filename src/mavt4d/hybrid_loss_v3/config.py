"""Independent immutable direction-B trial budget; no v2 lock is modified."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

# Labelled upper bound (arm U): derived from the version so existing configs
# keep their exact fields and hashes.
UPPER_BOUND_VERSIONS = (
    "hybrid-cd-loss-v3-upper-bound-v1",
    "hybrid-cd-loss-v3-upper-bound-tiny-v1",
)
# Tiny B baseline on the same overfit profile as the tiny arms and arm U.
TINY_B_VERSION = "hybrid-cd-loss-v3-tiny-v1"
# Architecture arms, also derived from the version: version -> (arm, variant).
# The variant is resolved against the base architecture by LossV3Module.
ARCH_VERSIONS = {
    f"hybrid-cd-arch-{name}{size}-v1": (
        arm,
        {"innovation": "I" in arm, "appearance": "A" in arm},
    )
    for name, arm in (
        ("innovation", "I"),
        ("appearance", "A"),
        ("innovation-appearance", "IA"),
    )
    for size in ("", "-tiny")
}
OVERFIT_VERSIONS = (
    "hybrid-cd-loss-v3-upper-bound-tiny-v1",
    TINY_B_VERSION,
    *(version for version in ARCH_VERSIONS if version.endswith("-tiny-v1")),
)


@dataclass(frozen=True, slots=True)
class LossV3TrialConfig:
    version: str = "hybrid-cd-loss-v3-trial-v1"
    seed: int = 17
    steps: int = 1000
    effective_source_groups: int = 8
    microbatch_candidates: tuple[int, ...] = (1, 2, 4, 8)
    dev_every: int = 200
    image_every: int = 200
    log_every: int = 10
    memory_headroom: float = 0.1
    capacity_warmup: int = 1
    capacity_repeats: int = 2
    selection: str = "loss-v3-common-dev-v1"

    def __post_init__(self):
        if (
            self.version
            not in (
                "hybrid-cd-loss-v3-trial-v1",
                "hybrid-cd-loss-v3-vram-trial-v1",
                "hybrid-cd-loss-v3-vram-trial-v2",
                *UPPER_BOUND_VERSIONS,
                TINY_B_VERSION,
                *ARCH_VERSIONS,
            )
            or self.selection != "loss-v3-common-dev-v1"
        ):
            raise ValueError("unsupported loss v3 trial version")
        if self.seed != 17 or self.dev_every != 200:
            raise ValueError("first trial locks seed17 and dev_every200")
        for name in (
            "steps",
            "effective_source_groups",
            "image_every",
            "log_every",
            "capacity_warmup",
            "capacity_repeats",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.05 <= self.memory_headroom <= 0.5:
            raise ValueError("memory_headroom must lie in [.05,.5]")
        if (
            not isinstance(self.microbatch_candidates, tuple)
            or not self.microbatch_candidates
            or tuple(sorted(set(self.microbatch_candidates)))
            != self.microbatch_candidates
        ):
            raise ValueError(
                "microbatch_candidates must be a nonempty increasing tuple"
            )
        if self.capacity_policy == "largest_safe" and (
            self.effective_source_groups != 64
            or self.microbatch_candidates
            != (
                (8, 16, 32, 60, 63)
                if self.allows_partial_microbatch
                else (8, 16, 32, 64)
            )
            or self.capacity_warmup != 1
            or self.capacity_repeats != 1
            or self.memory_headroom != 0.1
        ):
            raise ValueError(
                "VRAM profile freezes batch64, its versioned candidate grid and 10% headroom"
            )
        if self.overfit and (
            self.effective_source_groups != 1 or self.microbatch_candidates != (1,)
        ):
            raise ValueError("tiny profiles repeat exactly one source group")
        for candidate in self.microbatch_candidates:
            self.accumulation(candidate)

    @property
    def arm(self):
        if self.version in UPPER_BOUND_VERSIONS:
            return "U"
        return ARCH_VERSIONS.get(self.version, ("B", None))[0]

    @property
    def architecture_variant(self):
        """{"innovation", "appearance"} flags for arms I/A/IA; None for B and U."""
        variant = ARCH_VERSIONS.get(self.version, (None, None))[1]
        return None if variant is None else dict(variant)

    @property
    def core_head_weight(self):
        return 1.0 if self.arm == "U" else 0.0

    @property
    def overfit(self):
        return self.version in OVERFIT_VERSIONS

    @property
    def capacity_policy(self):
        return (
            "largest_safe"
            if self.version
            in ("hybrid-cd-loss-v3-vram-trial-v1", "hybrid-cd-loss-v3-vram-trial-v2")
            else "fastest_safe"
        )

    @property
    def capacity_order(self):
        if self.capacity_policy == "largest_safe":
            return tuple(reversed(self.microbatch_candidates))
        return self.microbatch_candidates

    @property
    def allows_partial_microbatch(self):
        return self.version == "hybrid-cd-loss-v3-vram-trial-v2"

    def validate_production(self):
        if self.overfit:
            return
        required_groups = 64 if self.capacity_policy == "largest_safe" else 8
        if self.effective_source_groups != required_groups:
            raise ValueError(
                f"production profile freezes {required_groups} effective source groups"
            )

    def accumulation(self, microbatch_groups):
        if (
            type(microbatch_groups) is not int
            or microbatch_groups <= 0
            or microbatch_groups > self.effective_source_groups
            or (
                not self.allows_partial_microbatch
                and self.effective_source_groups % microbatch_groups
            )
        ):
            raise ValueError("microbatch must divide effective source groups")
        return (
            self.effective_source_groups + microbatch_groups - 1
        ) // microbatch_groups

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def sha256(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {f.name for f in fields(cls)}:
            raise ValueError("trial configuration keys differ")
        return cls(
            **{**raw, "microbatch_candidates": tuple(raw["microbatch_candidates"])}
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
