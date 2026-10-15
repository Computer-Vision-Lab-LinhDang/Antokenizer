"""LiDAR hubs: K farthest-point centres per sweep, each with its M nearest input points as members."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.camera import transform_points


@dataclass
class HubGroups:
    centers_world: Tensor  # (S, K, 3)
    centers_sensor: Tensor  # (S, K, 3)
    rel_xyz: Tensor  # (S, K, M, 3): member minus centre, sensor frame
    intensity: Tensor  # (S, K, M, 1)
    member_idx: Tensor  # (S, K, M) int64, indices into input_points(s)


def farthest_point_sample(xyz: Tensor, K: int, start_idx: int = 0) -> Tensor:
    """(K,) int64 indices of a farthest-point sample of ``xyz`` (N, 3), K <= N. Ties go to the lowest index."""
    idx = torch.empty(K, dtype=torch.int64, device=xyz.device)
    dist = torch.full((xyz.shape[0],), float("inf"), dtype=xyz.dtype, device=xyz.device)
    idx[0] = start_idx
    for i in range(1, K):
        dist = torch.minimum(dist, ((xyz - xyz[idx[i - 1]]) ** 2).sum(-1))
        idx[i] = torch.argmax(dist)
    return idx


def knn(query: Tensor, ref: Tensor, k: int) -> Tensor:
    """(Q, k) int64 indices of the k nearest ``ref`` points to each query, nearest first."""
    d = torch.cdist(query, ref, compute_mode="donot_use_mm_for_euclid_dist")
    return d.topk(k, dim=1, largest=False, sorted=True).indices


def build_hubs(sample: SceneSample, K: int = 256, M: int = 32) -> HubGroups | None:
    """Hubs from ``input_points`` only (never held-out points); None without LiDAR.

    A sweep with fewer than K (or M) input points repeats indices cyclically to fill K centres (M members).
    """
    if not sample.lidar_points:
        return None
    out: dict[str, list[Tensor]] = {f: [] for f in ("centers_world", "centers_sensor", "rel_xyz", "intensity",
                                                    "member_idx")}
    for s in range(len(sample.lidar_points)):
        pts = sample.input_points(s)
        n = pts.shape[0]
        if n == 0:
            raise ValueError(f"lidar_points[{s}]: no input points left after hold-out")
        xyz = pts[:, :3]
        centers = farthest_point_sample(xyz, K) if n >= K else torch.arange(K, device=xyz.device) % n
        c = xyz[centers]
        members = knn(c, xyz, min(M, n))
        members = members[:, torch.arange(M, device=xyz.device) % members.shape[1]]
        out["centers_sensor"].append(c)
        out["centers_world"].append(transform_points(sample.lidar_to_world[s], c))
        out["rel_xyz"].append(xyz[members] - c[:, None])
        out["intensity"].append(pts[members, 3:4])
        out["member_idx"].append(members)
    return HubGroups(**{f: torch.stack(v) for f, v in out.items()})


def pca_normals(points: Tensor) -> Tensor:
    """(G, 3) unit normals (least-variance direction, sign arbitrary) of point groups (G, M, 3)."""
    x = points - points.mean(1, keepdim=True)
    return torch.linalg.eigh(x.transpose(1, 2) @ x).eigenvectors[..., 0]
