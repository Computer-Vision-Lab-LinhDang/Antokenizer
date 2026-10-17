"""Independent immutable direction-B trial budget; no v2 lock is modified."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path


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
            self.version != "hybrid-cd-loss-v3-trial-v1"
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
        for candidate in self.microbatch_candidates:
            self.accumulation(candidate)

    def accumulation(self, microbatch_groups):
        if (
            type(microbatch_groups) is not int
            or microbatch_groups <= 0
            or self.effective_source_groups % microbatch_groups
        ):
            raise ValueError("microbatch must divide effective source groups")
        return self.effective_source_groups // microbatch_groups

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
