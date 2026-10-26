"""Geometry of the C-D 3.2 decoder: SO(3), projection, neighbours, cut window, soft z-buffer, flow, budget."""
import math

import torch

from mavt.cd32 import geometry as geo
from mavt.cd32.config import CD32Config, tiny_config


def test_budget_default_and_tiny():
    cfg = CD32Config()
    cfg.check()
    assert cfg.L_max == 82
    assert cfg.segment_numbers("image") == 12288
    assert cfg.c_numbers("video") == 6145 and cfg.geom_numbers == 54
    assert cfg.segment_numbers("video") <= 12288 and cfg.segment_numbers("threed") <= 12288
    t = tiny_config()
    t.check()
    assert t.segment_numbers("video") <= t.budget and t.L_max >= 1


def test_so3_exp_identity_and_orthonormal():
    assert torch.equal(geo.so3_exp(torch.zeros(3)), torch.eye(3))
    w = torch.randn(16, 3)
    R = geo.so3_exp(w)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(16, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(16), atol=1e-5)
    a = torch.tensor([0.0, 0.0, math.pi / 2])                    # 90 deg about z: x -> y
    assert torch.allclose(geo.so3_exp(a) @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6)
    assert torch.allclose(geo.rotation_angle_deg(geo.so3_exp(a)), torch.tensor(90.0), atol=1e-3)


def test_identity_projection_returns_cell_centres():
    G = 16
    uv = geo.cell_centres(G)
    rho = torch.randn(2, G * G)
    K = geo.intrinsics_from_focal(geo.fov_to_focal(torch.tensor([50.0, 70.0]), G), G)
    P = geo.unproject(rho, K, uv)
    assert torch.allclose(P[..., 2], torch.exp(-rho))
    p, z, valid = geo.project(geo.transform(P, torch.eye(3).expand(2, 3, 3), torch.zeros(2, 3)), K, 1e-2)
    assert valid.all() and torch.allclose(p, uv.expand(2, -1, -1), atol=1e-4)


def test_translation_moves_near_cells_more():
    """Parallax: a sideways camera move shifts near cells (small z) more than far ones."""
    G = 8
    uv = geo.cell_centres(G)
    rho = torch.zeros(1, G * G)
    rho[0, : G * G // 2] = math.log(4.0)                          # top half 4x closer
    K = geo.intrinsics_from_focal(torch.tensor([8.0]), G)
    P = geo.unproject(rho, K, uv)
    p, _, _ = geo.project(geo.transform(P, torch.eye(3)[None], torch.tensor([[0.1, 0.0, 0.0]])), K, 1e-2)
    shift = (p - uv)[0, :, 0]
    assert torch.allclose(shift[: G * G // 2], torch.full((G * G // 2,), 3.2), atol=1e-4)
    assert torch.allclose(shift[G * G // 2:], torch.full((G * G // 2,), 0.8), atol=1e-4)


def test_knn_cut_radius_is_k_plus_1_distance_and_window_positive():
    G, k = 8, 9
    uv = geo.cell_centres(G)
    p = (uv + 0.3 * torch.randn_like(uv)).unsqueeze(0)
    valid = torch.ones(1, G * G, dtype=torch.bool)
    idx, d2, R2, pj = geo.knn_read(uv, p, valid, k)
    full = (uv[:, None] - p[0][None]).pow(2).sum(-1).sort(-1).values
    assert torch.allclose(d2[0], full[:, :k], atol=1e-5) and torch.allclose(R2[0], full[:, k], atol=1e-5)
    w = geo.cut_window(d2, R2, 1e-4)
    assert (w > math.log(1e-4)).all() or (d2 == R2.unsqueeze(-1)).any()      # every neighbour keeps weight
    assert torch.allclose(pj, torch.gather(p.unsqueeze(1).expand(1, G * G, -1, 2), 2,
                                           idx.unsqueeze(-1).expand(-1, -1, -1, 2)))


def test_cut_window_continuous_at_the_boundary():
    """A neighbour at d -> R gets weight -> eps: swapping it with the (k+1)-th changes nothing abruptly."""
    R2 = torch.tensor([4.0])
    near = geo.cut_window(torch.tensor([[4.0 - 1e-6]]), R2, 1e-4)
    at = geo.cut_window(torch.tensor([[4.0]]), R2, 1e-4)
    assert torch.allclose(near, at, atol=1e-2) and torch.allclose(at, torch.tensor(math.log(1e-4)))


def test_soft_zbuffer_prefers_the_front_cell():
    logz = torch.log(torch.tensor([[[1.0, 4.0, 4.0]]]))
    d2 = torch.zeros(1, 1, 3)
    w = torch.zeros(1, 1, 3)
    dlz = geo.soft_zbuffer(logz, d2, w, beta=1.0)
    assert dlz[0, 0, 0] < dlz[0, 0, 1] and torch.isclose(dlz[0, 0, 1], dlz[0, 0, 2])
    # a cell outside the window (w = log eps) does not move the reference
    w2 = torch.tensor([[[math.log(1e-8), 0.0, 0.0]]])
    assert geo.soft_zbuffer(logz, d2, w2, 1.0)[0, 0, 1].abs() < 1e-6


def test_convex_upsample_constant_and_uniform():
    G, g = 16, 4
    M = torch.zeros(2, g, g, 3)
    M[..., 0] = 1.5
    w = torch.zeros(2, G * G, 9)
    w[..., 4] = 50.0                                               # all weight on the centre coarse cell
    m = geo.convex_upsample(M, w, G)
    assert m.shape == (2, G * G, 3) and torch.allclose(m[..., 0], torch.full((2, G * G), 1.5), atol=1e-4)


def test_normals_face_camera_and_backface_bias():
    G = 8
    uv = geo.cell_centres(G)
    K = geo.intrinsics_from_focal(torch.tensor([8.0]), G)
    P = geo.unproject(torch.zeros(1, G * G), K, uv)               # fronto-parallel plane at z = 1
    n = geo.cell_normals(P, G)
    assert torch.allclose(n[0, :, 2], -torch.ones(G * G), atol=1e-5)
    I = torch.eye(3)[None]
    front = geo.backface_bias(n, I, P, 10.0)
    flip = geo.so3_exp(torch.tensor([[0.0, math.pi, 0.0]]))       # seen from behind
    back = geo.backface_bias(n, flip, geo.transform(P, flip, torch.tensor([[0.0, 0.0, 2.0]])), 10.0)
    assert front.abs().max() < 0.1 and back.max() < -5


def test_view_order_by_angle():
    c2w = torch.eye(4).repeat(1, 4, 1, 1)
    for v, ang in zip(range(1, 4), (90.0, 10.0, 45.0)):
        c2w[0, v, :3, :3] = geo.so3_exp(torch.tensor([0.0, math.radians(ang), 0.0]))
    assert geo.view_order(c2w).tolist() == [[0, 2, 3, 1]]
