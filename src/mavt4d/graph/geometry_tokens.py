"""Per-token geometry in TokenLayout order (camera tokens, then hubs)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.hubs import HubGroups
from mavt4d.graph.types import TokenLayout


@dataclass
class TokenGeometry:
    patch_rc: Tensor  # (n_cam, 2) int64
    patch_center_px: Tensor  # (n_cam, 2)
    cam_id: Tensor  # (n_cam,) int64
    time: Tensor  # (n_cam,) float32, seconds from t0
    cam_to_world: Tensor | None  # (n_cam, 4, 4)
    intrinsics: Tensor | None  # (n_cam, 3, 3)
    hub_world: Tensor | None  # (n_hub, 3)
    hub_sensor: Tensor | None  # (n_hub, 3)
    hub_time: Tensor | None  # (n_hub,)
    lidar_to_world: Tensor | None  # (n_hub, 4, 4)


def token_geometry(sample: SceneSample, layout: TokenLayout, hubs: HubGroups | None) -> TokenGeometry:
    """Calibration fields are None when the sample lacks them; hub fields are None without hubs."""
    patch = sample.images.shape[-1] // layout.Wp
    v, t, r, c = (x.flatten() for x in torch.meshgrid(*(torch.arange(n) for n in (layout.V, layout.T, layout.Hp,
                                                                                  layout.Wp)), indexing="ij"))
    hub = {}
    if hubs is not None:
        S, K = hubs.centers_world.shape[:2]
        s = torch.arange(S).repeat_interleave(K)
        hub = dict(hub_world=hubs.centers_world.reshape(-1, 3), hub_sensor=hubs.centers_sensor.reshape(-1, 3),
                   hub_time=sample.lidar_times[s].float(), lidar_to_world=sample.lidar_to_world[s])
    return TokenGeometry(
        patch_rc=torch.stack([r, c], -1),
        patch_center_px=torch.stack([(c + 0.5) * patch, (r + 0.5) * patch], -1).float(),
        cam_id=v, time=sample.frame_times[v, t].float(),
        cam_to_world=None if sample.cam_to_world is None else sample.cam_to_world[v, t],
        intrinsics=None if sample.intrinsics is None else sample.intrinsics[v],
        hub_world=hub.get("hub_world"), hub_sensor=hub.get("hub_sensor"), hub_time=hub.get("hub_time"),
        lidar_to_world=hub.get("lidar_to_world"))
