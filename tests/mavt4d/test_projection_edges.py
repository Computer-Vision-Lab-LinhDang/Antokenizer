"""Projection edges (patch <-> hub) and 3D-neighbour edges (hub -> hub)."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.sample import SceneSample
from mavt4d.data.synthetic import SyntheticSceneConfig, gt_visible, make_synthetic_scene
from mavt4d.geometry.camera import invert_se3, pixel_to_patch, transform_points
from mavt4d.geometry.hubs import build_hubs
from mavt4d.graph.builder import GraphConfig
from mavt4d.graph.projection import build_nb3d_edges, build_projection_edges, visible_members
from mavt4d.graph.track import sweep_for_frame
from mavt4d.graph.types import Provenance, TokenLayout

CFG = SyntheticSceneConfig()
GCFG = GraphConfig()
K_HUBS, M_HUBS = 64, 16


def _layout(sample: SceneSample, hubs, patch: int = 16) -> TokenLayout:
    V, T, _, H, W = sample.images.shape
    return TokenLayout(V=V, T=T, Hp=H // patch, Wp=W // patch, S=hubs.centers_world.shape[0],
                       K=hubs.centers_world.shape[1])


@pytest.fixture(scope="module")
def scene():
    sample, gt = make_synthetic_scene(CFG, 0)
    sample = apply_holdout(sample)
    hubs = build_hubs(sample, K=K_HUBS, M=M_HUBS)
    L = _layout(sample, hubs)
    p2h, h2p = build_projection_edges(sample, hubs, L, GCFG)
    return sample, gt, hubs, L, p2h, h2p


def _decode(L: TokenLayout, cam: torch.Tensor, hub: torch.Tensor) -> tuple[torch.Tensor, ...]:
    frame, rc = cam // (L.Hp * L.Wp), cam % (L.Hp * L.Wp)
    return frame // L.T, frame % L.T, rc, (hub - L.n_cam) // L.K, (hub - L.n_cam) % L.K


def _members_in_patch(sample, hubs, L, cfg, cam, hub) -> torch.Tensor:
    """(E,) number of visible members of each edge's hub that project into the edge's patch."""
    v, t, rc, s, k = _decode(L, cam, hub)
    out = torch.zeros(len(cam), dtype=torch.int64)
    M = hubs.member_idx.shape[2]
    for vv, tt in {(int(a), int(b)) for a, b in zip(v, t)}:
        sel = ((v == vv) & (t == tt)).nonzero().flatten()
        uv, _, vis = visible_members(sample, hubs, sweep_for_frame(sample, vv, tt), vv, tt, cfg)
        prc, inside = pixel_to_patch(uv, cfg.patch, L.Hp, L.Wp)
        patch = torch.where(vis & inside, prc[:, 0] * L.Wp + prc[:, 1], -1).reshape(-1, M)  # (K, M)
        out[sel] = (patch[k[sel]] == rc[sel, None]).sum(1)
    return out


def test_every_edge_reprojects(scene):
    sample, _, hubs, L, p2h, _ = scene
    assert len(p2h.src) >= 100
    assert (_members_in_patch(sample, hubs, L, GCFG, p2h.src, p2h.dst) >= 1).all()


def test_false_visible_below_1pct_synthetic(scene):
    sample, gt, hubs, _, _, _ = scene
    visible = false = 0
    for v in range(CFG.V):
        for t in range(CFG.T):
            uv, depth, vis = visible_members(sample, hubs, sweep_for_frame(sample, v, t), v, t, GCFG)
            visible += int(vis.sum())
            false += int((vis & ~gt_visible(uv, depth, gt.depth[v, t], gt.valid[v, t])).sum())
    assert visible >= 1000, visible
    assert false / visible < 0.01, (false, visible)


def test_occluded_hub_has_no_edges_to_occluding_view():
    """Wall A at x = 5 m fills camera 0's view; patch B at x = 10 m sits right behind it."""
    gy, gz = torch.meshgrid(torch.arange(-4, 4.01, 0.1), torch.arange(-2.5, 2.51, 0.1), indexing="ij")
    wall = torch.stack([torch.full_like(gy, 5.0), gy, gz], -1).reshape(-1, 3)
    by, bz = torch.meshgrid(torch.arange(-0.5, 0.51, 0.05), torch.arange(-0.5, 0.51, 0.05), indexing="ij")
    back = torch.stack([torch.full_like(by, 10.0), by, bz], -1).reshape(-1, 3)
    pts = torch.cat([wall, back])
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looks along +x
    sample = SceneSample(scene_id="occl", source="synthetic", images=torch.zeros(1, 1, 3, 64, 112),
                         frame_times=torch.zeros(1, 1, dtype=torch.float64),
                         intrinsics=torch.tensor([[[80.0, 0, 56], [0, 80.0, 32], [0, 0, 1]]]),
                         cam_to_world=c2w[None, None], lidar_points=[torch.cat([pts, torch.ones(len(pts), 1)], 1)],
                         lidar_to_world=torch.eye(4)[None], lidar_times=torch.zeros(1, dtype=torch.float64),
                         lidar_keys=["occl"])
    hubs = build_hubs(sample, K=32, M=8)
    L = _layout(sample, hubs)
    p2h, h2p = build_projection_edges(sample, hubs, L, GCFG)
    on_back = hubs.centers_world[0, :, 0] > 7.5
    assert on_back.any() and (~on_back).any()
    hub_of_edge = (p2h.dst - L.n_cam) % L.K
    assert len(p2h.src) > 0 and not on_back[hub_of_edge].any()
    assert not on_back[(h2p.src - L.n_cam) % L.K].any()


def test_grazing_surface_dropped_by_min_cos_incidence():
    """Camera 1.5 m above a ground plane, looking along +x: ground from 15 to 40 m is seen at
    cos(incidence) < 0.1, so threshold 0.2 removes its edges; a wall facing the camera keeps all of its own."""
    gx, gy = torch.meshgrid(torch.arange(15.0, 40.0, 0.25), torch.arange(-6.0, 6.0, 0.25), indexing="ij")
    ground = torch.stack([gx, gy, torch.full_like(gx, -1.5)], -1).reshape(-1, 3)
    wy, wz = torch.meshgrid(torch.arange(-1.0, 1.0, 0.05), torch.arange(-0.5, 1.0, 0.05), indexing="ij")
    wall = torch.stack([torch.full_like(wy, 12.0), wy, wz], -1).reshape(-1, 3)
    pts = torch.cat([ground, wall])
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looks along +x
    sample = SceneSample(scene_id="graze", source="synthetic", images=torch.zeros(1, 1, 3, 64, 112),
                         frame_times=torch.zeros(1, 1, dtype=torch.float64),
                         intrinsics=torch.tensor([[[80.0, 0, 56], [0, 80.0, 32], [0, 0, 1]]]),
                         cam_to_world=c2w[None, None], lidar_points=[torch.cat([pts, torch.ones(len(pts), 1)], 1)],
                         lidar_to_world=torch.eye(4)[None], lidar_times=torch.zeros(1, dtype=torch.float64),
                         lidar_keys=["graze"])
    hubs = build_hubs(sample, K=48, M=8)
    L = _layout(sample, hubs)
    on_ground = hubs.centers_world[0, :, 2] < -1.0
    pairs = {}
    for thr in (0.0, 0.2):
        p2h, h2p = build_projection_edges(sample, hubs, L, replace(GCFG, min_cos_incidence=thr))
        hub = (p2h.dst - L.n_cam) % L.K
        pairs[thr] = {(int(a), int(b)) for a, b, g in zip(p2h.src, p2h.dst, on_ground[hub]) if not g}
        assert torch.equal(p2h.src, h2p.dst) and (p2h.attr[:, 2] >= thr).all()
        assert on_ground[hub].any() == (thr == 0.0)
    assert pairs[0.0] and pairs[0.0] <= pairs[0.2]  # wall edges survive the filter

def test_caps_respected(scene):
    sample, _, hubs, L, p2h, h2p = scene
    for e, (hub_cap, patch_cap) in [((p2h, h2p), (GCFG.proj_hub_cap, GCFG.proj_patch_cap)),
                                    (build_projection_edges(sample, hubs, L, replace(GCFG, proj_hub_cap=3,
                                                                                     proj_patch_cap=2)), (3, 2))]:
        hub_deg = torch.bincount(e[0].dst, minlength=L.num_nodes)[L.n_cam:]
        patch_deg = torch.bincount(e[1].dst, minlength=L.num_nodes)[:L.n_cam]
        assert hub_deg.max() <= hub_cap and patch_deg.max() <= patch_cap
    assert hub_deg.max() == 3 and patch_deg.max() == 2  # the small caps bind


def test_weight_equals_visible_fraction(scene):
    sample, _, hubs, L, p2h, _ = scene
    count = _members_in_patch(sample, hubs, L, GCFG, p2h.src, p2h.dst)
    assert torch.allclose(p2h.weight, count.float() / M_HUBS)
    assert (p2h.weight > 0).all() and (p2h.weight <= 1).all()


def test_p2h_h2p_are_reverses(scene):
    _, _, _, L, p2h, h2p = scene
    assert torch.equal(p2h.src, h2p.dst) and torch.equal(p2h.dst, h2p.src)
    assert torch.equal(p2h.weight, h2p.weight) and torch.equal(p2h.attr, h2p.attr)
    assert (p2h.src < L.n_cam).all() and (p2h.dst >= L.n_cam).all()
    assert (p2h.provenance == Provenance.CALIB_LIDAR).all()


def test_nb3d_k_nearest_no_self(scene):
    _, _, hubs, L, _, _ = scene
    e = build_nb3d_edges(hubs, L, k=5)
    assert (e.src != e.dst).all() and (e.src >= L.n_cam).all()
    assert (torch.bincount(e.dst, minlength=L.num_nodes)[L.n_cam:] == 5).all()
    s_src, s_dst = (e.src - L.n_cam) // L.K, (e.dst - L.n_cam) // L.K
    assert torch.equal(s_src, s_dst)  # neighbours stay within a sweep
    for s in range(L.S):
        c = hubs.centers_world[s]
        d = torch.cdist(c, c, compute_mode="donot_use_mm_for_euclid_dist").fill_diagonal_(float("inf"))
        kth = d.topk(5, largest=False).values[:, -1]
        sel = s_dst == s
        dk, sk = (e.dst[sel] - L.n_cam) % L.K, (e.src[sel] - L.n_cam) % L.K
        assert (d[dk, sk] <= kth[dk] + 1e-6).all()


def test_attr_hub_position_in_camera_frame(scene):
    sample, _, hubs, L, p2h, _ = scene
    v, t, _, s, k = _decode(L, p2h.src, p2h.dst)
    c2w = sample.cam_to_world[v, t]
    centre = hubs.centers_world[s, k]
    p = transform_points(invert_se3(c2w), centre[:, None])[:, 0]
    z = p[:, 2]
    a = p2h.attr
    assert torch.allclose(a[:, 3], p[:, 0] / z, atol=1e-4) and torch.allclose(a[:, 4], p[:, 1] / z, atol=1e-4)
    assert torch.allclose(a[:, 5], z.log(), atol=1e-5) and torch.allclose(a[:, 0], z.log(), atol=1e-5)
    assert torch.allclose(a[:, 1], sample.intrinsics[v, 0, 0] / z, atol=1e-3)
    assert torch.allclose(a[:, 6:9], F.normalize(centre - c2w[:, :3, 3], dim=-1), atol=1e-5)
    assert (a[:, 2] >= 0).all() and (a[:, 2] <= 1 + 1e-6).all() and (a[:, 9] == 1).all()
