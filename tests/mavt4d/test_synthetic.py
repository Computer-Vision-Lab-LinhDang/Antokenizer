"""Synthetic scenes: exact ground truth agrees with the rendered images, LiDAR and box motion."""
from __future__ import annotations

import pytest
import torch

from mavt4d.data.synthetic import SyntheticSceneConfig, SyntheticSceneDataset, make_synthetic_scene, texture
from mavt4d.geometry.boxes import half_extents, world_to_box
from mavt4d.geometry.camera import project, transform_points

CFG = SyntheticSceneConfig()


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(CFG, seed=0)


def test_texture_shift_changes_appearance_not_geometry(scene):
    s0, gt0 = scene
    s1, gt1 = make_synthetic_scene(SyntheticSceneConfig(texture_shift=50.0), seed=0)
    assert torch.equal(gt0.depth, gt1.depth) and torch.equal(gt0.instance, gt1.instance)
    assert all(torch.equal(a.center, b.center) for a, b in zip(s0.boxes, s1.boxes))
    ground, box = gt0.valid & (gt0.instance == 0), gt0.instance > 0
    assert (s0.images - s1.images).abs().amax(2)[ground].mean() > 0.1
    assert (s0.images - s1.images).abs().amax(2)[box].mean() > 0.1
    sky = ~gt0.valid
    assert torch.equal(s0.images.permute(0, 1, 3, 4, 2)[sky], s1.images.permute(0, 1, 3, 4, 2)[sky])


def test_shapes_and_tier_A(scene):
    s, gt = scene
    s.validate()
    V, T, H, W = CFG.V, CFG.T, CFG.H, CFG.W
    assert s.tier == "A" and s.source == "synthetic"
    assert s.images.shape == (V, T, 3, H, W) and s.images.dtype == torch.float32
    assert s.images.abs().max() <= 1.0
    assert len(s.lidar_points) == len(s.lidar_keys) == T and len(s.boxes) == T
    assert gt.world_xyz.shape == (V, T, H, W, 3) and gt.depth.shape == gt.instance.shape == (V, T, H, W)
    assert gt.instance.dtype == torch.int64 and gt.valid.dtype == torch.bool
    assert gt.valid.float().mean() > 0.5
    ds = SyntheticSceneDataset(2, CFG, base_seed=0)
    assert len(ds) == 2 and torch.equal(ds[0].images, s.images) and torch.equal(ds.ground_truth(0).depth, gt.depth)


def test_deterministic_given_seed(scene):
    s, gt = scene
    s2, gt2 = make_synthetic_scene(CFG, seed=0)
    assert torch.equal(s.images, s2.images) and torch.equal(gt.world_xyz, gt2.world_xyz)
    assert all(torch.equal(a, b) for a, b in zip(s.lidar_points, s2.lidar_points))
    s3, _ = make_synthetic_scene(CFG, seed=1)
    assert not torch.equal(s.images, s3.images)


def _object_coords(p: torch.Tensor, inst: torch.Tensor, boxes) -> torch.Tensor:
    out = p.clone()
    for i, k in enumerate(boxes.instance_id.tolist()):
        sel = inst == k
        out[sel] = world_to_box(p[sel], boxes.center[i], boxes.yaw[i])
    return out


def test_image_equals_texture_at_gt_points(scene):
    s, gt = scene
    box_pixels = 0
    for v in range(CFG.V):
        for t in range(CFG.T):
            ok = gt.valid[v, t]
            inst = gt.instance[v, t][ok]
            expect = texture(_object_coords(gt.world_xyz[v, t][ok], inst, s.boxes[t]), inst)
            got = s.images[v, t].permute(1, 2, 0)[ok]
            assert (got - expect).abs().max() <= 1e-5
            box_pixels += int((inst > 0).sum())
    assert box_pixels > 500  # box texture (object frame) is exercised, not just the ground


def test_lidar_points_lie_on_surfaces(scene):
    s, _ = scene
    for k, pts in enumerate(s.lidar_points):
        p = transform_points(s.lidar_to_world[k], pts[:, :3])
        dist = p[:, 2].abs()  # ground plane z = 0
        b = s.boxes[k]
        for i in range(b.center.shape[0]):
            q = world_to_box(p, b.center[i], b.yaw[i]).abs() - half_extents(b.size[i])
            outside = q.clamp(min=0).norm(dim=-1)
            inside = q.max(-1).values.clamp(max=0)
            dist = torch.minimum(dist, (outside + inside).abs())
        assert len(p) > 1000 and dist.max() < 1e-3


def test_moving_box_velocity(scene):
    s, _ = scene
    disp = torch.stack([s.boxes[t + 1].center - s.boxes[t].center for t in range(CFG.T - 1)])  # (T-1, M, 3)
    speed = disp.norm(dim=-1) / CFG.dt
    moving = speed[0] > 1e-3
    assert int(moving.sum()) == CFG.n_moving
    assert torch.allclose(speed[:, moving], speed[0, moving].expand_as(speed[:, moving]), atol=1e-3)
    heading = torch.stack([torch.cos(s.boxes[0].yaw), torch.sin(s.boxes[0].yaw)], -1)[moving]
    assert torch.allclose(disp[0, moving, :2] / disp[0, moving, :2].norm(dim=-1, keepdim=True), heading, atol=1e-4)
    assert all(torch.equal(b.instance_id, s.boxes[0].instance_id) for b in s.boxes)
    assert all(torch.equal(b.yaw, s.boxes[0].yaw) for b in s.boxes)


def test_adjacent_cameras_overlap(scene):
    s, gt = scene
    t = 0
    p = gt.world_xyz[0, t].reshape(-1, 3)[gt.valid[0, t].flatten()]
    uv, z = project(p, s.intrinsics[1], s.cam_to_world[1, t])
    u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
    inside = (z > 0) & (u >= 0) & (u < CFG.W) & (v >= 0) & (v < CFG.H)
    d1 = gt.depth[1, t][v[inside], u[inside]]
    seen = (gt.valid[1, t][v[inside], u[inside]] & ((z[inside] - d1).abs() < 0.05 * d1 + 0.05)).sum()
    assert seen / (CFG.H * CFG.W) >= 0.10


def test_render_sweeps_reproduce_the_scene_sweeps(scene):
    """Extra sweeps come from the same scanner, ego motion and box motion: at the frame times they are the
    scene's own sweeps (world frame) and the scene's own box states."""
    from mavt4d.data.synthetic import render_sweeps

    sample, _ = scene
    sweeps = render_sweeps(CFG, 0, sample.lidar_times)
    assert len(sweeps) == CFG.T
    for t, (pts, boxes) in enumerate(sweeps):
        want = transform_points(sample.lidar_to_world[t], sample.lidar_points[t][:, :3])
        assert pts.shape == want.shape and torch.allclose(pts, want, atol=1e-4)
        assert torch.allclose(boxes.center, sample.boxes[t].center) and torch.equal(boxes.yaw, sample.boxes[t].yaw)


def test_render_sweeps_between_frames_move_with_ego():
    """Half a frame later the scanner sits ego_speed * dt / 2 further along +x: the ground hit straight ahead
    on the lowest beam moves by that much."""
    from mavt4d.data.synthetic import render_sweeps

    cfg = SyntheticSceneConfig(n_boxes=0, n_moving=0)
    (a, _), (b, _) = render_sweeps(cfg, 0, torch.tensor([0.0, cfg.dt / 2], dtype=torch.float64))
    def ahead(p):
        ground = p[(p[:, 2].abs() < 1e-4) & (p[:, 1].abs() < 1e-3) & (p[:, 0] > 0)]
        return ground[:, 0].min()
    assert float(ahead(b) - ahead(a)) == pytest.approx(cfg.ego_speed * cfg.dt / 2, abs=1e-3)


def test_camera_pitch_is_configurable():
    flat = SyntheticSceneConfig(cam_pitch_deg=0.0, n_boxes=0, n_moving=0, T=1)
    sample, _ = make_synthetic_scene(flat, 0)
    fwd = sample.cam_to_world[0, 0, :3, 2]  # optical axis in world
    assert abs(float(fwd[2])) < 1e-6
