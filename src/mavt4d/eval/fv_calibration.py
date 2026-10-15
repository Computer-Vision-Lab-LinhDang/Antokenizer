"""Calibration of the M1 false-visible rule on synthetic scenes with exact visibility (E1.1).

Hub members of every view fall in three sets. ``correct``: in front, in the image and visible by
``gt_visible``. ``wrong``: in front, in the image and occluded in GT (the known-wrong edges a rule must catch).
``used``: members behind the builder's kept PROJ_P2H edges; their true false-visible rate is used & not
gt_visible. A rule's error is the share of ``correct`` members it flags; its recall the share of ``wrong``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from mavt4d.data.sample import Boxes3D, SceneSample
from mavt4d.data.synthetic import SyntheticGT, SyntheticSceneConfig, gt_visible, render_sweeps
from mavt4d.eval.graph_checks import DEPTH_BINS_M, FVRule, hub_normals_world, rule_visible, used_members
from mavt4d.geometry.boxes import half_extents, points_in_boxes, rot_z, world_to_box
from mavt4d.geometry.camera import project, transform_points
from mavt4d.geometry.hubs import HubGroups
from mavt4d.graph.types import CorrespondenceGraph

if TYPE_CHECKING:
    from mavt4d.graph.builder import GraphConfig

# nuScenes m3 geometry: 6 cameras of 448 x 256 with fx ~ 360 px at ~0 pitch, a 32-beam LiDAR (-30..10 deg,
# ~1084 steps per turn) at 20 Hz, a driving ego.
M3_LIKE = SyntheticSceneConfig(V=6, T=1, H=256, W=448, fov_deg=2 * math.degrees(math.atan(224 / 360)),
                               cam_yaw_step_deg=60.0, cam_pitch_deg=0.0, lidar_beams=32, lidar_az=1084,
                               lidar_elev_deg=(-30.0, 10.0), ego_speed=8.0, n_boxes=8, n_moving=1)
SWEEP_DT = 0.05  # LIDAR_TOP period
INC_BINS = (0.1, 0.3, 0.6)  # |cos incidence| bins of correct members
GAP_BINS_M = (1.0, 5.0)  # wrong members: how far behind the GT surface at their pixel
BOUNDARY_CELL_PX = 4
LABEL_BINS = {"depth": len(DEPTH_BINS_M) + 1, "inc": len(INC_BINS) + 1, "boundary": 2, "gap": len(GAP_BINS_M) + 1}
SET_LABELS = {"correct": ("depth", "inc", "boundary"), "wrong": ("depth", "gap"), "used": ("depth",),
              "used_true_fv": ("depth",)}


def gt_normals(p: Tensor, boxes: Boxes3D) -> Tensor:
    """(N, 3) exact unit surface normals (sign arbitrary) of scene points: ground z, or the box face whose
    half-extent the point reaches (largest |local coordinate| / half extent)."""
    inst = points_in_boxes(p, boxes, margin=0.05)
    n = torch.zeros_like(p)
    n[:, 2] = 1.0
    for i in range(boxes.center.shape[0]):
        sel = inst == boxes.instance_id[i]
        if sel.any():
            q = world_to_box(p[sel], boxes.center[i], boxes.yaw[i])
            axis = (q.abs() / half_extents(boxes.size[i])).argmax(-1)
            n[sel] = rot_z(boxes.yaw[i])[:, axis].T
    return n


def boundary_cells(instance: Tensor, valid: Tensor, cell: int) -> Tensor:
    """(H / cell, W / cell) bool: the cell holds more than one GT surface label (sky counts as one)."""
    lab = torch.where(valid, instance, torch.full_like(instance, -1))
    H, W = lab.shape
    r = lab.reshape(H // cell, cell, W // cell, cell)
    return r.amin((1, 3)) != r.amax((1, 3))


def reference_clouds(cfg: SyntheticSceneConfig, seed: int, sample: SceneSample, sides: tuple[int, ...],
                     sweep_dt: float = SWEEP_DT, box_margin: float = 0.3,
                     drop: Literal["moving", "all"] = "moving") -> tuple[dict[int, Tensor], dict[int, dict]]:
    """World-frame reference clouds: all points of the keyframe sweep (held-out included) plus, for each n in
    ``sides``, the sweeps at +-1..n * sweep_dt rendered from the same scene, minus their points inside boxes at
    their own time dilated by ``box_margin``. ``drop="moving"`` removes only boxes that move (static boxes stand in
    for the unannotated walls and poles that stay in nuScenes' cloud); ``drop="all"`` mirrors nuScenes, where
    every annotated box is removed. Single-frame (T = 1) samples."""
    key = transform_points(sample.lidar_to_world[0], sample.lidar_points[0][:, :3])
    t0, n_max = float(sample.lidar_times[0]), max(sides)
    offs = [k for k in range(-n_max, n_max + 1) if k]
    rendered = render_sweeps(cfg, seed, torch.tensor([t0 + k * sweep_dt for k in offs], dtype=torch.float64))
    kept = {}
    if rendered:
        first, last = rendered[0][1], rendered[-1][1]
        moving = (first.center - last.center).norm(dim=-1) > 1e-6
    for k, (pts, boxes) in zip(offs, rendered):
        if drop == "moving":
            boxes = Boxes3D(center=boxes.center[moving], size=boxes.size[moving], yaw=boxes.yaw[moving],
                            instance_id=boxes.instance_id[moving])
        inside = points_in_boxes(pts, boxes, box_margin) >= 0
        kept[k] = (pts[~inside], int(inside.sum()))
    refs, info = {}, {}
    for n in sides:
        ks = [k for k in offs if abs(k) <= n]
        refs[n] = torch.cat([key] + [kept[k][0] for k in ks])
        info[n] = {"sweeps": len(ks), "dropped_box": sum(kept[k][1] for k in ks)}
    return refs, info


def _counts(masks: dict[str, Tensor], labels: dict[str, Tensor]) -> dict:
    return {name: {lab: torch.bincount(labels[lab][m], minlength=LABEL_BINS[lab]).tolist()
                   for lab in SET_LABELS[name]} for name, m in masks.items()}


def merge_counts(a: dict, b: dict) -> dict:
    """Element-wise sum of two nested count dicts of the same shape (lists of ints at the leaves)."""
    if isinstance(a, list):
        return [x + y for x, y in zip(a, b, strict=True)]
    return {k: merge_counts(a[k], b[k]) for k in a}


def calibrate_scene(sample: SceneSample, gt: SyntheticGT, hubs: HubGroups, graph: CorrespondenceGraph,
                    cfg: GraphConfig, refs: dict[int, Tensor], rules: dict[str, FVRule]) -> dict:
    """Counts of one scene: ``n[set][label]`` member totals and ``flag[rule][ref][set][label]`` members the rule
    rejects against reference ``refs[ref]`` (sets correct / wrong / used, per label bin of SET_LABELS)."""
    L, (H, W) = graph.layout, sample.images.shape[-2:]
    M = hubs.member_idx.shape[2]
    normals = hub_normals_world(sample, hubs)
    out = None
    for v in range(L.V):
        for t in range(L.T):
            s, uv, z, used = used_members(sample, hubs, graph, cfg, v, t)
            P = transform_points(sample.lidar_to_world[s], sample.input_points(s)[:, :3])[hubs.member_idx[s].flatten()]
            N = normals[s].repeat_interleave(M, 0)
            in_img = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            gtv = gt_visible(uv, z, gt.depth[v, t], gt.valid[v, t])
            col = torch.nan_to_num(uv[:, 0]).long().clamp(0, W - 1)
            row = torch.nan_to_num(uv[:, 1]).long().clamp(0, H - 1)
            cos = (gt_normals(P, sample.boxes[t]) * F.normalize(P - sample.cam_to_world[v, t, :3, 3], dim=-1)).sum(-1)
            labels = {
                "depth": torch.bucketize(z.contiguous(), torch.tensor(DEPTH_BINS_M), right=True),
                "inc": torch.bucketize(cos.abs().contiguous(), torch.tensor(INC_BINS), right=True),
                "boundary": boundary_cells(gt.instance[v, t], gt.valid[v, t], BOUNDARY_CELL_PX)[
                    row // BOUNDARY_CELL_PX, col // BOUNDARY_CELL_PX].long(),
                "gap": torch.bucketize((z - gt.depth[v, t][row, col]).contiguous(), torch.tensor(GAP_BINS_M), right=True)}
            sets = {"correct": in_img & gtv, "wrong": in_img & ~gtv, "used": used, "used_true_fv": used & ~gtv}
            view = {"n": _counts(sets, labels), "flag": {}}
            K, c2w = sample.intrinsics[v], sample.cam_to_world[v, t]
            for k, ref in refs.items():
                ref_uv, ref_z = project(ref, K, c2w)
                for name, rule in rules.items():
                    flagged = in_img & ~rule_visible(uv, z, ref_uv, ref_z, H, W, rule, P, N, ref)
                    view["flag"].setdefault(name, {})[k] = _counts(
                        {sk: sets[sk] & flagged for sk in ("correct", "wrong", "used")}, labels)
            out = view if out is None else merge_counts(out, view)
    return out
