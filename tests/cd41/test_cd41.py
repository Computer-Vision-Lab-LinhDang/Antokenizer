"""C-D 4.1 on the tiny config: the tests of section 13 of the design (shape, budget, transport, geometry,
innovation, bottleneck, state, mode, prefix), the ablations, the trainer and the eval script."""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from mavt.cd32 import geometry as geo
from mavt.cd41 import CD41Config, MAVTCD41, load_cd41, tiny_cd41_config
from mavt.cd41 import state as st
from mavt.cd41 import transport as tr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
S = 64                     # tiny image size: 4 x 4 cells


def _model(**kw):
    torch.manual_seed(0)
    return MAVTCD41(tiny_cd41_config(**kw)).eval()


def _orbit(B, V, size=S, step=0.15):
    """Cameras on a circle of radius 2 looking at the origin, `step` radians apart (OpenCV, camera -> world)."""
    c2w = torch.zeros(B, V, 4, 4)
    for v in range(V):
        a = step * v
        o = torch.tensor([2 * math.cos(a), 2 * math.sin(a), 0.3])
        z = -o / o.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, -1.0]))
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        c2w[:, v, :3, 0], c2w[:, v, :3, 1], c2w[:, v, :3, 2], c2w[:, v, :3, 3] = x, y, z, o
        c2w[:, v, 3, 3] = 1
    f = size / 2 / math.tan(0.3456)
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]).expand(B, V, 3, 3).clone()
    return dict(c2w=c2w, K=K, depth=torch.full((B, V, size, size), 2.0))


def _inputs(modality, B=2, T=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    if modality == "image":
        return torch.rand(B, 3, S, S, generator=g) * 2 - 1, None
    x = torch.rand(B, 3, T, S, S, generator=g) * 2 - 1
    if modality == "video":
        return x, None
    cam = _orbit(B, T)
    if modality == "world":
        cam["dyn"] = torch.zeros(B, T, S, S)
        cam["dyn"][:, :, :32, :32] = 1
    return x, cam


def _random_out(m):
    """ZeroLinear heads start at 0; give the innovation decoder, the Tucker coefficients and the level-2
    increments a non-zero map."""
    torch.manual_seed(1)
    with torch.no_grad():
        m.dec_r.out.weight.normal_(0, 0.1)
        if m.a_head is not None:
            m.a_head.weight.normal_(0, 0.1)
        if m.refiner is not None:
            for head in (m.refiner.d_pose, m.refiner.d_kappa, m.refiner.d_motion, m.refiner.d_a):
                if head is not None:
                    head.weight.normal_(0, 0.01)
    return m


def _random_op(B, N, k=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(N, generator=g)[:k] for _ in range(B * N)]).view(B, N, k)
    w = torch.softmax(torch.randn(B, N, k, generator=g), -1)
    return tr.SparseTransportOp(idx=idx, w=w, support=torch.ones(B, N, dtype=torch.bool))


# ---------------------------------------------------------------------------------------------- budget
def test_ledger_of_the_default_config():
    cfg = CD41Config()
    assert cfg.header_slots == 16 and cfg.anchor_numbers == 6148
    assert cfg.target_numbers() == 6 + 48 + 4 + 32 + 80 * 24 == 2010
    assert cfg.segment_numbers("video") == 12194 and cfg.budget - cfg.segment_numbers("video") == 94
    assert cfg.segment_numbers("image") == 6164
    off = CD41Config(tucker=False, n_slots=81)
    assert off.segment_numbers("video") == 12170 and abs(off.segment_numbers() - cfg.segment_numbers()) <= 24
    with pytest.raises(AssertionError):
        CD41Config(n_slots=90).check()                                    # over the 12 288 ceiling


@pytest.mark.parametrize("modality", ["image", "video", "threed", "world"])
def test_step_shapes_and_serialized_payload(modality):
    m = _model()
    m.train()
    x, cam = _inputs(modality)
    out = m(x, modality, cam, return_outputs=True)
    assert torch.isfinite(out["loss"]) and all(torch.isfinite(v) for v in out["parts"].values())
    out["loss"].backward()
    cfg, o = m.cfg, out["outputs"]
    Z, N, d = o["Z"], cfg.n_cells, m.feat_dim
    n = 1 if modality == "image" else 4
    assert o["F_hat"].shape == (2, n, N, d) and o["rgb"].shape[-3:] == (3, S, S)
    assert Z["c"].shape == (2, N, cfg.d_c) and Z["rho"].shape == (2, N) and Z["kappa0"].shape == (2, 4)
    if modality == "image":
        assert Z["xi"].shape == (2, 0, 6) and "r" not in Z and o["Hb"] is None      # no invented targets
    else:
        assert o["Hb"].shape == (2, 3, N, d) and Z["r"].shape == (2, 3, cfg.n_slots, cfg.d_r)
        assert Z["M"].shape == (2, 3, cfg.n_motion, 3) and Z["a"].shape == (2, 3, cfg.rank_a)
    flats = st.serialize(cfg, Z)
    want = cfg.segment_numbers("image" if modality == "image" else "video")
    assert [f.numel() for f in flats] == [want, want] and out["diag"]["numbers"].item() == want


def test_tucker_term_shape_and_identity():
    m = _random_out(_model())
    B, T, N, d = 2, 3, m.cfg.n_cells, m.feat_dim
    U, a = torch.randn(B, N, m.cfg.rank_xy), torch.randn(B * T, m.cfg.rank_a)
    op = _random_op(B * T, N, k=5)
    out = m.tucker_term(op.apply(U), a)                          # (W U) B(a) Q, U shared by the T targets
    assert out.shape == (B * T, N, d)
    Bm = torch.einsum("pkq,bk->bpq", m.core, a)
    UBQ = m.Q(torch.einsum("bjp,bpq->bjq", U.repeat_interleave(T, 0), Bm))
    ref = torch.bmm(op.dense(), UBQ)                             # W (U B Q)
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------------------------- transport
def _flat(B=1, G=4):
    return torch.zeros(B, G * G), torch.tensor([[0.5, 0.5, 0.5, 0.5]]).expand(B, 4)


def _tr(rho, k0, R, t, disp, ki, r_cut=1.5, sigma=0.35, tau=0.1, G=4, smooth=False):
    op = tr.transport(rho, k0, R, t, disp, ki, G, 9, r_cut, sigma, tau, smooth_cutoff=smooth)
    return dict(W=op.dense(), support=op.support, ok=op.ok, op=op)


def test_zero_motion_identity_only_when_the_kernel_selects_one_cell():
    rho, k = _flat()
    R, t = torch.eye(3).expand(1, 3, 3), torch.zeros(1, 3)
    out = _tr(rho, k, R, t, None, k, r_cut=0.5)
    assert torch.equal(out["W"][0], torch.eye(16)) and out["support"].all()
    assert out["op"].idx.shape == (1, 16, 9) and out["op"].w.shape == (1, 16, 9)            # sparse: 9 per row
    blur = _tr(rho, k, R, t, None, k)["W"][0]                    # 9 neighbours within 1.5 cells: some blur
    assert not torch.equal(blur, torch.eye(16))
    assert torch.allclose(blur.sum(-1), torch.ones(16)) and (blur.diagonal() > 0.9).all()


def test_one_cell_translation_and_exact_empty_rows():
    rho, k = _flat()
    f = 0.5 * 4                                                  # fx in cells; z = 1 everywhere
    R, t = torch.eye(3).expand(1, 3, 3), torch.tensor([[1.0 / f, 0.0, 0.0]])      # +1 cell to the right
    out = _tr(rho, k, R, t, None, k, r_cut=0.5)
    W, sup = out["W"][0], out["support"][0]
    for q in range(16):
        r, c = divmod(q, 4)
        if c == 0:                                               # nothing lands in the first column
            assert not sup[q] and torch.equal(W[q], torch.zeros(16))
        else:
            assert sup[q] and W[q, r * 4 + c - 1] == 1 and W[q].sum() == 1
    assert out["ok"][0].view(4, 4)[:, 3].logical_not().all()     # the last column leaves the crop


def _reference_W(rho, k0, R, t, ki, G, knn, r_cut, sigma, tau, z_eps=1e-2, smooth=False):
    """Explicit loop over target cells: the formula of section 7, written out."""
    N = G * G
    W = torch.zeros(N, N, dtype=torch.float64)
    K0, Ki = tr.K_cells(k0, G)[0].double(), tr.K_cells(ki, G)[0].double()
    pts = []
    for j in range(N):
        a, b = divmod(j, G)
        z = math.exp(-float(rho[0, j]))
        P = z * torch.linalg.inv(K0) @ torch.tensor([b + 0.5, a + 0.5, 1.0], dtype=torch.float64)
        P1 = R[0].double() @ P + t[0].double()
        if not (P1[2] > z_eps):
            pts.append(None)
            continue
        u = Ki @ (P1 / P1[2])
        inside = 0 <= u[0] <= G and 0 <= u[1] <= G
        pts.append((u[:2], float(P1[2])) if inside else None)
    for q in range(N):
        a, b = divmod(q, G)
        uq = torch.tensor([b + 0.5, a + 0.5], dtype=torch.float64)
        cand = sorted(((float((uq - p[0]).pow(2).sum()), j, p[1]) for j, p in enumerate(pts) if p is not None))[:knn]
        cand = [c for c in cand if c[0] < r_cut ** 2]
        if not cand:
            continue
        lg = torch.tensor([-d2 / (2 * sigma ** 2) - z / tau + (math.log(max(1e-6, 1 - d2 / r_cut ** 2)) if smooth else 0)
                           for d2, _, z in cand], dtype=torch.float64)
        w = torch.softmax(lg, 0)
        for (d2, j, z), wj in zip(cand, w):
            W[q, j] = wj
    return W


@pytest.mark.parametrize("depth", [1.0, 2000.0])
def test_transport_matches_the_explicit_reference(depth):
    torch.manual_seed(3)
    G = 4
    rho = torch.randn(1, G * G) * 0.3 - math.log(depth)          # far scenes: logits ~ -z / tau_z = -1e4
    k0 = torch.tensor([[0.55, 0.6, 0.48, 0.52]])
    ki = torch.tensor([[0.5, 0.52, 0.5, 0.47]])
    R, t = geo.so3_exp(torch.tensor([[0.05, -0.08, 0.03]])), torch.tensor([[0.12, -0.05, 0.1]]) * depth
    out = _tr(rho, k0, R, t, None, ki, r_cut=1.2, sigma=0.4, tau=0.2)
    ref = _reference_W(rho, k0, R, t, ki, G, 9, 1.2, 0.4, 0.2)
    assert torch.allclose(out["W"][0].double(), ref, atol=1e-5)
    rows = out["support"][0]
    assert rows.sum() >= 8
    assert torch.allclose(out["W"][0][rows].sum(-1), torch.ones(int(rows.sum())))
    assert torch.equal(out["W"][0][~rows], torch.zeros(int((~rows).sum()), 16))
    smooth = _tr(rho, k0, R, t, None, ki, r_cut=1.2, sigma=0.4, tau=0.2, smooth=True)["W"][0]  # flag, off by default
    assert torch.allclose(smooth.double(), _reference_W(rho, k0, R, t, ki, G, 9, 1.2, 0.4, 0.2, smooth=True), atol=1e-5)
    if depth == 1.0:                                             # far scenes: the depth term picks one cell
        assert not torch.allclose(smooth, out["W"][0], atol=1e-4)


def test_sparse_apply_matches_the_dense_product_with_gradients():
    torch.manual_seed(4)
    G, B, T, D = 4, 2, 3, 7
    rho = (torch.randn(B * T, G * G) * 0.3).requires_grad_(True)
    k = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).expand(B * T, 4)
    R, t = geo.so3_exp(torch.randn(B * T, 3) * 0.05), torch.randn(B * T, 3) * 0.05
    op = tr.transport(rho, k, R, t, None, k, G, 9, 1.5, 0.35, 0.1)
    V = torch.randn(B, G * G, D, requires_grad=True)             # one source per sample, shared by its T targets
    out = op.apply(V)
    ref = torch.bmm(op.dense(), V.repeat_interleave(T, 0))
    assert out.shape == (B * T, G * G, D) and torch.allclose(out, ref, atol=1e-6)
    g = torch.randn_like(out)
    gV, grho = torch.autograd.grad((out * g).sum(), (V, rho), retain_graph=True)
    gV2, grho2 = torch.autograd.grad((ref * g).sum(), (V, rho))
    assert torch.allclose(gV, gV2, atol=1e-5) and torch.allclose(grho, grho2, atol=1e-5) and grho.abs().sum() > 0
    Vd, idx = torch.randn(3, 5, 2, dtype=torch.float64, requires_grad=True), torch.randint(0, 5, (3, 6, 4))
    wd = torch.rand(3, 6, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: tr._GatherWeighted.apply(a, idx, b), (Vd, wd))


def test_points_behind_the_camera_or_nan_are_never_read():
    rho, k = _flat()
    rho = rho.clone().requires_grad_(True)
    R, t = torch.eye(3).expand(1, 3, 3), torch.zeros(1, 3)
    disp = torch.zeros(1, 16, 3)
    disp[0, 5, 2] = -3.0                                         # cell 5 pushed behind the camera
    disp[0, 6, 0] = float("nan")                                 # cell 6 not finite
    out = _tr(rho, k, R, t, disp, k)
    W = out["W"][0]
    assert torch.isfinite(W).all() and (W[:, 5] == 0).all() and (W[:, 6] == 0).all()
    assert not out["ok"][0, 5] and not out["ok"][0, 6]
    W.sum().backward()
    assert torch.isfinite(rho.grad).all()


# ---------------------------------------------------------------------------------------------- geometry
def test_z_depth_is_not_ray_distance_and_kappa_conventions():
    G = 4
    K = tr.K_cells(torch.tensor([[0.5, 0.5, 0.5, 0.5]]), G)
    P = geo.unproject(torch.zeros(1, G * G), K, geo.cell_centres(G))
    assert torch.allclose(P[..., 2], torch.ones(1, G * G))                     # z = exp(-rho) = 1
    assert (P.norm(dim=-1) > 1.0 + 1e-6).all()                                  # every ray is longer than z
    Kp = torch.tensor([[100.0, 0, 64.0], [0, 110.0, 60.0], [0, 0, 1]])
    assert torch.allclose(tr.kappa_from_K(Kp, 128, 120), torch.tensor([100 / 128, 110 / 120, 0.5, 0.5]))
    assert torch.allclose(tr.kappa_from_index_K(Kp, 128, 120), torch.tensor([100 / 128, 110 / 120, 64.5 / 128, 60.5 / 120]))
    Kc = Kp.clone()                                                             # a crop moves the principal point
    Kc[0, 2] -= 32
    assert torch.allclose(tr.kappa_from_K(Kc, 64, 120)[2], torch.tensor(0.5))


def test_so3_and_se3_round_trips_including_half_turns():
    torch.manual_seed(0)
    w = torch.randn(200, 3)
    w = w / w.norm(dim=-1, keepdim=True) * torch.rand(200, 1) * math.pi
    w[:5] = torch.tensor([[math.pi, 0, 0], [0, math.pi - 1e-6, 0], [0, 0, 1e-9], [0, 0, 0], [0.05, 0, 0]])
    R = geo.so3_exp(w)
    assert torch.allclose(geo.so3_exp(tr.so3_log(R)), R, atol=1e-5)
    T = torch.eye(4).repeat(200, 1, 1)
    T[:, :3, :3], T[:, :3, 3] = R, torch.randn(200, 3)
    xi = tr.se3_log(T)
    Rp, tp = tr.se3_exp(xi)
    assert torch.allclose(Rp, R, atol=1e-5) and torch.allclose(tp, T[:, :3, 3], atol=1e-4)
    inner = xi[:, :3].norm(dim=-1) < math.pi - 1e-2                             # away from the cut at pi
    assert torch.allclose(tr.se3_log(torch.cat([torch.cat([Rp, tp.unsqueeze(-1)], -1), T[:, 3:]], 1))[inner],
                          xi[inner], atol=1e-4)
    assert not torch.allclose(xi[5:, 3:], T[5:, :3, 3], atol=1e-2)             # se(3): v = V^-1 t, not t
    R0, t0 = tr.se3_exp(torch.zeros(2, 6))
    assert torch.equal(R0, torch.eye(3).expand(2, 3, 3)) and torch.equal(t0, torch.zeros(2, 3))
    x = torch.zeros(1, 6, requires_grad=True)
    sum(v.sum() for v in tr.se3_exp(x)).backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------------------------- innovation
def test_empty_innovation_is_exactly_zero_and_slots_reach_covered_cells():
    m = _random_out(_model())
    B, N, d = 2, m.cfg.n_cells, m.feat_dim
    Hb = torch.randn(B, N, d)
    sup = torch.ones(B, N, dtype=torch.bool)
    meta = torch.randn(B, m.dec_r.meta.net[0].in_features)
    r = torch.randn(B, m.cfg.n_slots, m.cfg.d_r, requires_grad=True)
    out0 = m.dec_r(Hb, sup, r, torch.zeros(B, dtype=torch.long), meta)
    assert torch.equal(out0, torch.zeros_like(out0))
    mixed = m.dec_r(Hb, sup, r, torch.tensor([0, m.cfg.n_slots]), meta)         # one L = 0 in the batch
    assert torch.equal(mixed[0], torch.zeros_like(mixed[0])) and mixed[1].abs().sum() > 0
    out = m.dec_r(Hb, sup, r, torch.full((B,), m.cfg.n_slots, dtype=torch.long), meta)
    out[:, :, :].sum().backward()                                # every cell has support here
    assert r.grad.abs().sum() > 0 and out.abs().sum() > 0
    out_new = m.dec_r(Hb, sup, r.detach() + 1, torch.full((B,), m.cfg.n_slots, dtype=torch.long), meta)
    assert not torch.allclose(out_new, out.detach())             # new content in covered cells is codable
    out_meta = m.dec_r(Hb, sup, r.detach(), torch.full((B,), m.cfg.n_slots, dtype=torch.long), meta + 1)
    assert not torch.allclose(out_meta, out.detach())            # Dec_R reads xi, M, kappa, a and the header


@pytest.mark.parametrize("causal", [True, False])
def test_slot_causal_mask_keeps_every_prefix_closed(causal):
    m = _model(slot_causal=causal)
    B, N, d, j = 2, m.cfg.n_cells, m.feat_dim, 3
    args = (torch.randn(B, N, d), torch.rand(B, 3, S, S) * 2 - 1, torch.randn(B, N, d),
            torch.ones(B, N, dtype=torch.bool), torch.randn(B, m.enc_r.meta.net[0].in_features))
    with torch.no_grad():
        r = m.enc_r(*args)
        m.enc_r.queries[:, j:] += 1.0                            # change the queries after slot j
        r2 = m.enc_r(*args)
    assert torch.equal(r[:, :j], r2[:, :j]) == causal and not torch.allclose(r[:, j:], r2[:, j:])


# ---------------------------------------------------------------------------------------------- bottleneck
@pytest.mark.parametrize("modality", ["image", "video", "threed", "world"])
def test_serialized_state_alone_decodes_the_same_output(modality):
    m = _random_out(_model())
    x, cam = _inputs(modality)
    with torch.no_grad():
        out = m(x, modality, cam, return_outputs=True)["outputs"]
    flats = [f.clone() for f in st.serialize(m.cfg, out["Z"])]
    rgb, Fh = out["rgb"].clone(), out["F_hat"].clone()
    del out, x, cam                                              # raw tensors gone
    fresh = MAVTCD41(m.cfg).eval()
    fresh.load_state_dict(m.state_dict())
    with torch.no_grad():
        dec = fresh.decode(st.deserialize(m.cfg, flats))
    assert torch.allclose(dec["F_hat"], Fh, atol=1e-5)
    assert torch.allclose(dec["rgb"].reshape(rgb.shape), rgb, atol=1e-5)


def test_variable_length_payload_round_trip():
    m = _random_out(_model())
    x, _ = _inputs("video")
    L = torch.tensor([[0, 2, 6], [6, 1, 0]])
    with torch.no_grad():
        Z = m(x, "video", L=L, return_outputs=True)["outputs"]["Z"]
    flats = st.serialize(m.cfg, Z)
    cfg = m.cfg
    assert [f.numel() for f in flats] == [cfg.segment_numbers() - (6 + 4) * cfg.d_r, cfg.segment_numbers() - (5 + 6) * cfg.d_r]
    Z2 = st.deserialize(cfg, flats)
    for k in ("h", "c", "rho", "kappa0", "xi", "M", "kappa", "a", "r", "L"):
        assert torch.equal(Z2[k], Z[k]), k
    with pytest.raises(ValueError):
        st.deserialize(cfg, [torch.cat([flats[0], torch.zeros(1)])])         # trailing scalar: not this state


# ---------------------------------------------------------------------------------------------- state / header
def test_header_and_segments():
    m = _model()
    x, cam = _inputs("video", T=8)
    with torch.no_grad():
        o = m(x, "video", return_outputs=True)["outputs"]
    h = o["Z"]["h"]
    assert h.shape == (4, 16) and o["Z"]["xi"].shape[1] == 3                 # 2 clips x 2 segments, 3 targets
    assert h[0, :2].tolist() == [1, 4] and h[0, 2:6].tolist() == [0, 1, 2, 3] and h[0, 10:13].tolist() == [6, 6, 6]
    assert h[0, 13:].tolist() == [0, 0, 41]
    x, cam = _inputs("threed")
    with torch.no_grad():
        h = m(x, "threed", cam, return_outputs=True)["outputs"]["Z"]["h"]
    assert h[0, 0] == 2 and h[0, 2:6].tolist() == [0, 0, 0, 0] and h[0, 6:10].tolist() == [0, 1, 2, 3]
    assert h[0, 13:15].tolist() == [1, 1]
    with torch.no_grad():
        h = m(_inputs("image")[0], "image", return_outputs=True)["outputs"]["Z"]["h"]
    assert h[0, :2].tolist() == [0, 1] and h[0, 2:15].abs().sum() == 0
    x, cam = _inputs("threed", T=5)
    with pytest.raises(ValueError):
        m(x, "threed", _orbit(2, 5))                                          # at most three targets


# ---------------------------------------------------------------------------------------------- mode
def test_static_views_copy_the_cameras_and_zero_the_motion():
    m = _random_out(_model())
    x, cam = _inputs("threed")
    with torch.no_grad():
        Z = m(x, "threed", cam, return_outputs=True)["outputs"]["Z"]
    assert torch.equal(Z["M"], torch.zeros_like(Z["M"]))
    xi = tr.se3_log(tr.relative_pose(cam["c2w"]))
    assert torch.allclose(Z["xi"], xi[:, 1:]) and torch.allclose(Z["kappa0"], tr.kappa_from_K(cam["K"][:, 0], S, S))
    assert xi[:, 1:, :3].norm(dim=-1).min() > 0.1                             # real rotations between views


def test_view_order_is_not_a_temporal_signal():
    m = _random_out(_model())
    x, cam = _inputs("threed")
    perm = [0, 3, 1, 2]
    with torch.no_grad():
        a = m(x, "threed", cam, return_outputs=True)["outputs"]
        b = m(x[:, :, perm], "threed", {k: v[:, perm] for k, v in cam.items()}, return_outputs=True)["outputs"]
    assert torch.allclose(a["F_hat"][:, perm], b["F_hat"], atol=1e-5)
    assert torch.allclose(a["Z"]["r"][:, [p - 1 for p in perm[1:]]], b["Z"]["r"], atol=1e-5)


def test_world_given_camera_copies_the_pose_and_predicts_motion():
    m = _random_out(_model(world_camera="given"))
    x, cam = _inputs("world")
    with torch.no_grad():
        Z = m(x, "world", cam, return_outputs=True)["outputs"]["Z"]
    assert torch.allclose(Z["xi"], tr.se3_log(tr.relative_pose(cam["c2w"]))[:, 1:])
    assert Z["h"][0, 14] == 1
    m2 = _model()
    out = m2(x, "world", cam)
    assert "cam_gt" in out["parts"] and "cam_rot_err_deg" in out["diag"]


# ---------------------------------------------------------------------------------------------- prefix
def test_level_two_refinement_starts_at_level_one_and_keeps_static_geometry():
    m = _model()
    x, cam = _inputs("video")
    with torch.no_grad():
        o = m(x, "video", return_outputs=True)["outputs"]           # zero increments at init: level 2 = level 1
    assert torch.equal(o["Z"]["xi"], torch.zeros_like(o["Z"]["xi"]))
    m = _random_out(m)
    with torch.no_grad():
        o = m(x, "video", return_outputs=True)["outputs"]
        x3, cam3 = _inputs("threed")
        s = m(x3, "threed", cam3, return_outputs=True)["outputs"]
    assert o["Z"]["xi"].abs().sum() > 0 and o["Z"]["M"].abs().sum() > 0     # video: geometry refined
    xi = tr.se3_log(tr.relative_pose(cam3["c2w"]))[:, 1:]
    assert torch.equal(s["Z"]["xi"], xi) and torch.equal(s["Z"]["M"], torch.zeros_like(s["Z"]["M"]))
    with torch.no_grad():
        m.refiner.d_a.weight.zero_()
        s0 = m(x3, "threed", cam3, return_outputs=True)["outputs"]
    assert not torch.allclose(s0["Z"]["a"], s["Z"]["a"])                    # static views: only a moves


def test_video_targets_read_the_earlier_state_and_static_views_do_not():
    m = _random_out(_model())
    for modality in ("video", "threed"):
        x, cam = _inputs(modality)
        y = x.clone()
        y[:, :, 1] = torch.rand_like(y[:, :, 1]) * 2 - 1                    # new first target
        with torch.no_grad():
            a = m(x, modality, cam, return_outputs=True)["outputs"]["Z"]
            b = m(y, modality, cam, return_outputs=True)["outputs"]["Z"]
        assert not torch.allclose(a["r"][:, 0], b["r"][:, 0])
        later = torch.allclose(a["r"][:, 1:], b["r"][:, 1:]) and torch.allclose(a["a"][:, 1:], b["a"][:, 1:])
        assert later == (modality == "threed")                              # video: S_{i-1} holds F^_1


def test_replacing_a_later_observation_leaves_the_prefix_unchanged():
    m = _random_out(_model())
    x, _ = _inputs("video")
    y = x.clone()
    y[:, :, 3] = torch.rand_like(y[:, :, 3]) * 2 - 1                         # new last target
    with torch.no_grad():
        a = m(x, "video", return_outputs=True)["outputs"]
        b = m(y, "video", return_outputs=True)["outputs"]
    sa, sb = st.serialize(m.cfg, a["Z"]), st.serialize(m.cfg, b["Z"])
    pre = m.cfg.header_slots + m.cfg.anchor_numbers + 2 * m.cfg.target_numbers()
    for u, v in zip(sa, sb):
        assert torch.equal(u[:pre], v[:pre]) and not torch.equal(u[pre:], v[pre:])
    assert torch.equal(a["F_hat"][:, :3], b["F_hat"][:, :3]) and torch.equal(a["rgb"][:, :3], b["rgb"][:, :3])


# ---------------------------------------------------------------------------------------------- ablations
def test_ablations_without_tucker_and_without_geometry():
    m = _model(tucker=False, n_slots=7, budget=400)
    assert m.a_head is None and not hasattr(m, "core")
    x, cam = _inputs("threed")
    out = m(x, "threed", cam, return_outputs=True)
    assert "a" not in out["outputs"]["Z"] or out["outputs"]["Z"]["a"] is None
    assert out["diag"]["numbers"].item() == m.cfg.segment_numbers() == 16 + 100 + 3 * (6 + 12 + 4 + 7 * 4)
    g = _model(geometry=False)
    out = g(_inputs("video")[0], "video", return_outputs=True)
    assert out["outputs"]["support"].all() and out["diag"]["support_frac"].item() == 1.0
    n = _model(nested=True)
    n.train()
    Ls = {int(v) for _ in range(5) for v in n.sample_L((4, 3), "cpu").flatten()}
    assert Ls <= set(n.cfg.nested_L) and len(Ls) > 1


# ---------------------------------------------------------------------------------------------- scripts
def _script(name):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trainer_dry_run_and_eval(tmp_path):
    tr41, ev41 = _script("train_cd41"), _script("eval_cd41")
    run = tmp_path / "run"
    assert tr41.main(["--dry-run", "--mix-modalities", "--out", str(run)]) == 0
    recs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    assert {"image/loss", "video/loss", "threed/loss", "world/loss", "world/loss_cam_gt"} <= set(recs[0])
    assert all(np.isfinite(r["loss"]) and r["grad_norm"] < 1e3 for r in recs)
    model, cfg, step = load_cd41(str(run / "last.pt"))
    assert step == 3 and cfg.tucker
    off = tmp_path / "off"
    assert tr41.main(["--dry-run", "--set", "tucker=false", "--set", "n_slots=7", "--out", str(off)]) == 0
    assert load_cd41(str(off / "last.pt"))[1].tucker is False
    out = tmp_path / "ev"
    assert ev41.main(["--dry-run", "--ckpt", str(run / "last.pt"), "--L-list", "0,2,max", "--out", str(out)]) == 0
    res = json.loads((out / "eval.json").read_text())
    assert res["L_list"] == [0, 2, 6] and res["segment_numbers"] == cfg.segment_numbers()
    for name in ("video", "threed", "world"):
        v = res["clips"][name]
        assert v["payload"] == cfg.segment_numbers() and [r["L"] for r in v["curve"]] == [0, 2, 6]
        assert v["curve"][0]["numbers"] < v["curve"][-1]["numbers"] == v["payload"]
        assert np.isfinite(v["target"]["psnr"]) and np.isfinite(v["copy"]["psnr"]) and v["no_tucker"] is not None
    assert "cam_rot_err_deg" in res["clips"]["world"]["diag"] and "rho_gt_corr" in res["clips"]["threed"]["diag"]
    assert "a = 0 at decode" in (out / "eval.md").read_text()


def test_e1a_relaxed_fit_recovers_a_planted_tucker_variation(tmp_path):
    e1a = _script("e1a_tucker_fit")
    torch.manual_seed(0)
    B, T, N, d, r = 2, 3, 16, 24, 3
    W = torch.softmax(torch.randn(B, T, N, N) * 3, -1)
    U, Bm, Q = torch.randn(B, N, r), torch.randn(B, T, r, r), torch.randn(B, r, d)
    E = torch.einsum("btij,btjd->btid", W, torch.einsum("bnp,btpq,bqd->btnd", U, Bm, Q))
    om = torch.rand(B, T, N) > 0.2
    good = e1a.fit_relaxed(E, W, om, r, 400, 3e-2)
    low = e1a.fit_relaxed(E, W, om, 1, 400, 3e-2)
    assert good["captured"] > 0.97 and low["captured"] < good["captured"]
    assert e1a.main(["--dry-run", "--ranks", "1,2", "--steps", "20", "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "e1a.json").read_text())
    assert [x["rank"] for x in res["fits"]] == [1, 2] and 0 < res["omega_frac"] <= 1
    assert all(0 <= x["captured_warp"] <= 1 for x in res["fits"])
