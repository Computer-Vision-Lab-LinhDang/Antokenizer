"""Reuse audited tensor-only loading while retaining independent source groups.

Different sources may have different calibration. A microbatch is a sequence
of TrainingBatch objects, never a single fabricated shared-camera batch.
"""
from mavt4d.pilot.data import PilotDataset, TrainingBatch, file_sha256

from .config import HybridPilotConfig


class HybridPilotDataset(PilotDataset):
    def __init__(self, root, cfg: HybridPilotConfig) -> None:
        if not isinstance(cfg, HybridPilotConfig):
            raise TypeError("cfg must be HybridPilotConfig")
        super().__init__(root, cfg)


__all__ = ["HybridPilotDataset", "TrainingBatch", "file_sha256"]
