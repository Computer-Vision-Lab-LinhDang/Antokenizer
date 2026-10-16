"""Versioned synthetic training for the explicit hybrid C/D + KV model."""

from .config import HybridPilotConfig
from .data import HybridPilotDataset, TrainingBatch
from .objectives import HybridPilotModule

__all__ = ["HybridPilotConfig", "HybridPilotDataset", "TrainingBatch", "HybridPilotModule"]
