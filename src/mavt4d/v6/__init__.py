"""MAVT v6: shared trunk over evidence and state, Kalman D, sparse box-limited field."""
from .config import V6Config
from .contracts import EncoderOutput, Observation, RayBatch, RenderOutput, TrunkOutput, V6State

__all__ = ["V6Config", "V6State", "Observation", "EncoderOutput", "TrunkOutput", "RayBatch", "RenderOutput"]
