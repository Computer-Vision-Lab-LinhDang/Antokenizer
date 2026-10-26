"""MAVT C-D tokenizer, bản 3.2: C (content + log inverse depth per cell, focal) at keyframes, D (pose + coarse
3D flow, then variable-length innovation tokens) at the other frames, two-stream backbone on frozen DINOv3."""
from .config import CD32Config, tiny_config
from .model import MAVTCD32
from .vanilla import CDVConfig, tiny_cdv_config
from .vanilla_model import MAVTCDV

__all__ = ["CD32Config", "MAVTCD32", "tiny_config", "CDVConfig", "MAVTCDV", "tiny_cdv_config"]
