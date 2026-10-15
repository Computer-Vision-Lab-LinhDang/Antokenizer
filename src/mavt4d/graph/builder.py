"""Graph assembly configuration (build_graph follows in T12)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class GraphConfig:
    patch: int = 16
    spatial_radius: int = 2
    spatial_cap: int = 24
    track_cap: int = 2
    track_mode: Literal["correspondence", "same_pixel"] = "correspondence"
    proj_hub_cap: int = 32
    proj_patch_cap: int = 8
    zbuffer_tol_m: float = 0.5
    zbuffer_cell_px: int = 4
    nb3d_k: int = 16
    hubs_K: int = 256
    hubs_M: int = 32
