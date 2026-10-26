"""MAVT Grain-Prior C-D v1: anchor grains + coarse memory + causal records on a frozen DINOv3 (prototype)."""
from .config import GrainPriorConfig, config_from_dict, tiny_config
from .model import MAVTGrainPrior, load_grain_prior

__all__ = ["GrainPriorConfig", "MAVTGrainPrior", "config_from_dict", "tiny_config", "load_grain_prior"]
