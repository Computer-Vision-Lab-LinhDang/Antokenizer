"""MAVT C-D 4.1: unified tokenizer with geometric transport and Δ-Tucker on frozen DINOv3 (prototype)."""
from .config import CD41Config, tiny_cd41_config
from .model import MAVTCD41, load_cd41

__all__ = ["CD41Config", "MAVTCD41", "tiny_cd41_config", "load_cd41"]
