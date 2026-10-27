"""Grain-Prior C-D v1 on the tiny config: ledger, state schema, public decoding of a closed state, innovation
prefix, causality, static views, known cameras, memory, the arms, the prior phase and the trainer."""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from mavt.grain_prior import GrainPriorConfig, MAVTGrainPrior, load_grain_prior, tiny_config
from mavt.grain_prior import geometry as g
from mavt.grain_prior import schema as sc
from mavt.grain_prior.layers import attend
from mavt.grain_prior.readout import geom_bias

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
S = 64                     # tiny image size: 4 x 4 cells


def _model(**kw):
    torch.manual_seed(0)
    return MAVTGrainPrior(tiny_config(**kw)).eval()


def _wake(m):
    """Zero-initialised heads (update increments, control queries, grain depth) get a small random map so the
    tests see non-trivial records."""
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad and p.dim() >= 1 and torch.count_nonzero(p) == 0:
                p.normal_(0, 0.05)
    return m


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
    gen = torch.Generator().manual_seed(seed)
    if modality == "image":
        return torch.rand(B, 3, S, S, generator=gen) * 2 - 1, None
    x = torch.rand(B, 3, T, S, S, generator=gen) * 2 - 1
    return x, (None if modality == "video" else _orbit(B, T))


def _grads_finite(m):
    return all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------------------------- ledger
def test_ledger_of_the_reference_config():
    cfg = GrainPriorConfig()
    cfg.check()
    assert (cfg.n_grains, cfg.d_grain, cfg.n_coarse, cfg.d_r, cfg.n_memb) == (64, 64, 16, 24, 32)
    assert cfg.anchor_numbers() == 64 * 64 + 16 * 128 + 4 == 6148
    assert cfg.target_numbers(52) == 6 + 48 + 4 + 32 + 52 * 24 == 1338
    assert cfg.state_numbers([]) == 6180
    assert cfg.state_numbers([52, 52, 52]) == 12242 and 12288 - 12242 == 46
    assert cfg.state_numbers([0]) == 6180 + 2048 + 90                # z_b travels with the first record


# ---------------------------------------------------------------------------------------------- forward
@pytest.mark.parametrize("modality", ["image", "video", "threed", "world"])
def test_forward_backward_and_payload(modality):
    m = _model()
    m.train()
    x, cam = _inputs(modality, T=5)
    out = m(x, modality, cam, return_outputs=True)
    assert torch.isfinite(out["loss"]) and all(torch.isfinite(v) for v in out["parts"].values())
    out["loss"].backward()
    assert _grads_finite(m)
    assert all(p.grad is None for n, p in m.named_parameters() if n.startswith("prior."))   # T3 only
    o, cfg = out["outputs"], m.cfg
    n = 1 if modality == "image" else 5
    assert o["F_hat"].shape[:2] == (2, n) and o["rgb"].shape == (2, n, 3, S, S)
    St = o["S"]
    assert len(St["recs"]) == min(n - 1, cfg.queue)
    flats = sc.serialize(cfg, St)
    for b, f in enumerate(flats):
        sc.validate(cfg, f)
        assert f.numel() == cfg.state_numbers([int(r["L"][b]) for r in St["recs"]])
    if modality == "image":
        assert [f.numel() for f in flats] == [cfg.state_numbers([])] * 2


def test_header_fields():
    m = _model()
    x, cam = _inputs("threed", T=4)
    St = m(x, "threed", cam, return_outputs=True)["outputs"]["S"]
    h = St["h"][0]
    assert int(h[sc.H_VERSION]) == 1 and int(h[sc.H_MODE]) == 2 and int(h[sc.H_NT]) == 3
    assert int(h[sc.H_COORD]) == 1 and int(h[sc.H_CAM]) == 1                     # metric, given cameras
    assert h[sc.H_VIEW + 1:sc.H_VIEW + 4].tolist() == [1, 2, 3]
    assert (h[sc.H_RES:] == 0).all()
    x, _ = _inputs("video", T=4)
    h = m(x, "video", None, return_outputs=True)["outputs"]["S"]["h"][0]
    assert int(h[sc.H_MODE]) == 1 and int(h[sc.H_COORD]) == 0 and int(h[sc.H_CAM]) == 0
    assert torch.allclose(h[sc.H_TT:sc.H_TT + 3], torch.tensor([1.0, 2.0, 3.0]) * m.cfg.frame_seconds)


def test_validate_rejects_bad_states():
    m = _model()
    x, _ = _inputs("video", T=3)
    flat = sc.serialize(m.cfg, m(x, "video", None, return_outputs=True)["outputs"]["S"])[0]
    sc.validate(m.cfg, flat)
    with pytest.raises(ValueError):
        sc.validate(m.cfg, flat[:-1])
    bad = flat.clone()
    bad[sc.H_RES] = 1
    with pytest.raises(ValueError):
        sc.validate(m.cfg, bad)
    bad = flat.clone()
    bad[-1] = float("nan")
    with pytest.raises(ValueError):
        sc.validate(m.cfg, bad)


# ---------------------------------------------------------------------------------------------- public decoder
@pytest.mark.parametrize("modality,T", [("video", 4), ("video", 6), ("threed", 4), ("world", 5)])
def test_decode_of_the_deserialized_state_matches_the_encoder(modality, T):
    """decode(S) reads the closed state alone (after a serialize round trip) and reproduces the in-graph
    outputs of the anchor and of every record in the queue, including after folds into z_b."""
    m = _wake(_model())
    x, cam = _inputs(modality, T=T)
    with torch.no_grad():
        o = m(x, modality, cam, return_outputs=True)["outputs"]
        S2 = sc.deserialize(m.cfg, sc.serialize(m.cfg, o["S"]))
        dec = m.decode(S2)
    q = len(S2["recs"])
    assert q == min(T - 1, m.cfg.queue)
    ref = torch.cat([o["F_hat"][:, :1], o["F_hat"][:, T - q:]], 1)
    assert torch.allclose(dec["F"], ref, atol=1e-4), (dec["F"] - ref).abs().max()
    rgb = torch.cat([o["rgb"][:, :1], o["rgb"][:, T - q:]], 1)
    assert torch.allclose(dec["rgb"], rgb, atol=1e-4)


def test_innovation_is_exactly_zero_at_L0_and_prefix_invariant():
    m = _wake(_model())
    cfg = m.cfg
    B, N = 2, cfg.n_cells
    a, pe = torch.randn(B, N, cfg.dim), m.readout.query_pe(B)
    r = torch.cat([torch.rand(B, cfg.n_slots, 2), torch.randn(B, cfg.n_slots, 1),
                   torch.full((B, cfg.n_slots, 1), 0.8), torch.randn(B, cfg.n_slots, cfg.d_e)], -1)
    with torch.no_grad():
        z = m.readout.innovation(a, pe, r, torch.zeros(B, dtype=torch.long))
        assert torch.count_nonzero(z) == 0
        for k in range(1, cfg.n_slots + 1):
            L = torch.full((B,), k)
            r2 = r.clone()
            r2[:, k:] = torch.randn_like(r2[:, k:]) * 5
            r2[:, k:, 3] = 0                                         # zeroed slots past the prefix: finite
            d1, d2 = m.readout.innovation(a, pe, r, L), m.readout.innovation(a, pe, r2, L)
            assert torch.isfinite(d2).all() and torch.allclose(d1, d2, atol=1e-6)


def test_innovation_slots_are_causal():
    """Slot j of the residual encoder does not read the slots after it."""
    m = _wake(_model())
    cfg = m.cfg
    B, N, D = 2, cfg.n_cells, m.stems.teacher_dim
    args = (torch.randn(B, N, D), torch.randn(B, 3, S, S), torch.randn(B, cfg.n_coarse, cfg.d_coarse),
            torch.randn(B, 6), torch.rand(B, 4), torch.randn(B, cfg.n_motion, 3))
    with torch.no_grad():
        r1 = m.innov(*args)
        j = 2
        m.innov.slots[:, j + 1:] += torch.randn_like(m.innov.slots[:, j + 1:])
        r2 = m.innov(*args)
    assert torch.allclose(r1[:, :j + 1], r2[:, :j + 1], atol=1e-6)
    assert not torch.allclose(r1[:, j + 1:], r2[:, j + 1:])


def test_video_coding_is_causal_in_time():
    m = _wake(_model())
    x, _ = _inputs("video", T=5)
    y = x.clone()
    y[:, :, 4] = torch.rand_like(y[:, :, 4])
    with torch.no_grad():
        a = m(x, "video", None, return_outputs=True)["outputs"]["F_hat"]
        b = m(y, "video", None, return_outputs=True)["outputs"]["F_hat"]
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6) and not torch.allclose(a[:, 4], b[:, 4])


def test_static_views_are_coded_independently_of_their_order():
    m = _wake(_model())
    x, cam = _inputs("threed", T=4)
    perm = [0, 2, 3, 1]
    cam_p = {k: v[:, perm] for k, v in cam.items()}
    with torch.no_grad():
        a = m(x, "threed", cam, return_outputs=True)["outputs"]
        b = m(x[:, :, perm], "threed", cam_p, return_outputs=True)["outputs"]
    assert torch.allclose(a["F_hat"][:, perm], b["F_hat"], atol=1e-5)


def test_known_cameras_are_copied_exactly():
    m = _wake(_model())
    x, cam = _inputs("threed", T=4)
    with torch.no_grad():
        St = m(x, "threed", cam, return_outputs=True)["outputs"]["S"]
    kap = g.kappa_from_K(cam["K"].float(), S, S)
    Tr = g.relative_pose(cam["c2w"])
    xi = g.se3_log(Tr[..., :3, :3], Tr[..., :3, 3])
    assert torch.equal(St["kappa0"], kap[:, 0])
    for j, rec in enumerate(St["recs"]):
        assert torch.equal(rec["xi"], xi[:, j + 1]) and torch.equal(rec["kappa"], kap[:, j + 1])
        assert torch.count_nonzero(rec["M"]) == 0                                # static views: M = 0


def test_world_labels_never_reach_the_estimated_records():
    """World clips coded as video (camera estimated) give the same records whatever the given cameras."""
    m = _wake(_model(world_given_p=0.0))
    m.train()
    x, cam = _inputs("world", T=4)
    cam2 = dict(cam, c2w=cam["c2w"].clone())
    cam2["c2w"][:, 1:, :3, 3] += 0.5
    torch.manual_seed(3)
    a = m(x, "world", cam, return_outputs=True)["outputs"]["S"]
    torch.manual_seed(3)
    b = m(x, "world", cam2, return_outputs=True)["outputs"]["S"]
    for ra, rb in zip(a["recs"], b["recs"]):
        assert torch.allclose(ra["xi"], rb["xi"]) and torch.allclose(ra["r"], rb["r"])


# ---------------------------------------------------------------------------------------------- memory
def test_memory_at_c0_leaves_the_grains_unchanged():
    m = _wake(_model())
    cfg = m.cfg
    B = 2
    f, u = torch.randn(B, cfg.n_grains, cfg.d_f), torch.rand(B, cfg.n_grains, 2)
    c0 = torch.randn(B, cfg.n_coarse, cfg.d_coarse)
    with torch.no_grad():
        assert torch.equal(m.appear(f, u, c0.clone(), c0), f)
        assert not torch.allclose(m.appear(f, u, c0 + 1, c0), f)


def test_memory_off_arm_keeps_z_at_c0():
    m = _wake(_model(memory=False))
    x, _ = _inputs("video", T=4)
    with torch.no_grad():
        St = m(x, "video", None, return_outputs=True)["outputs"]["S"]
        assert all(torch.equal(z, St["c0"].float()) for z in m.replay(St))


# ---------------------------------------------------------------------------------------------- attention
def test_rows_without_valid_keys_return_zero_and_finite_grads():
    q = torch.randn(2, 5, 8, requires_grad=True)
    k, v = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    valid = torch.tensor([[True, False, True], [False, False, False]])
    out = attend(q, k, v, 2, valid=valid)
    assert torch.count_nonzero(out[1]) == 0 and torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all()
    uq = g.cell_centres(4).expand(2, -1, -1)
    lnu = geom_bias(uq, torch.rand(2, 3, 2), torch.ones(2, 3), torch.ones(2, 3), valid, 4, 0.1)
    assert torch.isfinite(lnu).all() and torch.count_nonzero(lnu[1]) == 0
    assert torch.allclose(lnu[0][..., [0, 2]].exp().sum(-1), torch.ones(16))


def test_se3_round_trip_with_translation_first():
    xi = torch.randn(64, 6) * 0.5
    R, t = g.se3_exp(xi)
    assert torch.allclose(g.se3_log(R, t), xi, atol=1e-4)
    R, t = g.se3_exp(torch.tensor([[1.0, 2.0, 3.0, 0, 0, 0]]))
    assert torch.allclose(R, torch.eye(3)) and torch.allclose(t, torch.tensor([[1.0, 2.0, 3.0]]))


def test_kappa_bounds():
    cfg = tiny_config()
    k = g.kappa_from_logits(torch.randn(100, 4) * 10, cfg)
    fov = torch.rad2deg(2 * torch.atan(0.5 / k[:, :2]))
    assert (fov >= cfg.fov_min - 1e-3).all() and (fov <= cfg.fov_max + 1e-3).all()
    assert ((k[:, 2:] - 0.5).abs() <= cfg.pp_range + 1e-6).all()
    k0 = g.kappa_from_logits(torch.full((1, 4), 0.0) + torch.tensor([g.kappa_logit_init(cfg)] * 2 + [0, 0]), cfg)
    assert abs(math.degrees(2 * math.atan(0.5 / k0[0, 0].item())) - cfg.fov_init) < 1e-3


# ---------------------------------------------------------------------------------------------- arms, prior
@pytest.mark.parametrize("kw", [dict(geom_bias=False), dict(query_init="uncond"), dict(memory=False),
                                dict(tau_d_mode="absolute")])
def test_arms_run(kw):
    m = _model(**kw)
    m.train()
    x, cam = _inputs("world", T=4)
    out = m(x, "world", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _grads_finite(m)


def test_prior_phase_trains_only_the_prior():
    m = _model(phase="prior", horizons=(1, 2))
    m.train()
    x, _ = _inputs("video", T=5)
    out = m(x, "video", None)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and "psnr_prior_h2" in out["diag"]
    named = dict(m.named_parameters())
    assert any(p.grad is not None for n, p in named.items() if n.startswith("prior."))
    assert all(p.grad is None for n, p in named.items() if not n.startswith("prior."))
    assert _grads_finite(m)


def test_predicted_record_has_the_posterior_layout():
    m = _model()
    x, _ = _inputs("video", T=3)
    with torch.no_grad():
        St = m(x, "video", None, return_outputs=True)["outputs"]["S"]
        rec = m.predict(St, torch.full((2,), m.cfg.frame_seconds))
    cfg = m.cfg
    assert rec["xi"].shape == (2, 6) and rec["M"].shape == (2, cfg.n_motion, 3) and rec["kappa"].shape == (2, 4)
    assert rec["b"].shape == (2, cfg.n_memb) and rec["r"].shape == (2, cfg.n_slots, cfg.d_r)
    assert ((rec["r"][..., 3] >= cfg.sigma_min) & (rec["r"][..., 3] <= cfg.sigma_min + cfg.sigma_span)).all()


# ---------------------------------------------------------------------------------------------- trainer
def _script(name):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trainer_dry_run(tmp_path):
    tr = _script("train_grain_prior")
    run = tmp_path / "run"
    assert tr.main(["--dry-run", "--mix-modalities", "--out", str(run)]) == 0
    recs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    assert {"image/loss", "video/loss", "threed/loss", "world/loss"} <= set(recs[0])
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in recs)
    model, cfg, step = load_grain_prior(str(run / "last.pt"))
    assert step == 3 and cfg.geom_bias and cfg.query_init == "cond"
    arm = tmp_path / "arm"
    assert tr.main(["--dry-run", "--mix-modalities", "--set", "geom_bias=false", "--set", "query_init=uncond",
                    "--out", str(arm)]) == 0
    cfg = load_grain_prior(str(arm / "last.pt"))[1]
    assert cfg.geom_bias is False and cfg.query_init == "uncond"


def test_eval_dry_run(tmp_path):
    out = tmp_path / "ev"
    assert _script("eval_grain_prior").main(["--dry-run", "--L-list", "0,2,max", "--out", str(out)]) == 0
    res = json.loads((out / "eval.json").read_text())
    assert res["L_list"] == [0, 2, 6] and res["image_numbers"] == tiny_config().state_numbers([])
    assert {"video", "threed", "world (given cameras)", "world (estimated cameras)"} <= set(res["clips"])
    v = res["clips"]["video"]
    assert [c["numbers"] for c in v["curve"]] == [tiny_config().state_numbers([L] * 3) for L in (0, 2, 6)]
    assert "cam_rot_err_deg" in res["clips"]["world (estimated cameras)"]["diag"]


# ---------------------------------------------------------------------------------------------- rgb_from
@pytest.mark.parametrize("modality,T", [("image", 1), ("video", 4), ("threed", 4)])
def test_rgb_from_hidden_decodes_the_closed_state_like_the_encoder(modality, T):
    """rgb_from = hidden: the RGB head reads the decoder state (width dim), not the predicted DINO feature, and
    decode(S) still reproduces the in-graph images."""
    m = _wake(_model(rgb_from="hidden"))
    assert m.rgb.proj.in_features == m.cfg.dim
    x, cam = _inputs(modality, T=T)
    with torch.no_grad():
        o = m(x, modality, cam, return_outputs=True)["outputs"]
        dec = m.decode(sc.deserialize(m.cfg, sc.serialize(m.cfg, o["S"])))
    q = len(o["S"]["recs"])
    n = 1 if modality == "image" else T
    rgb = torch.cat([o["rgb"][:, :1], o["rgb"][:, n - q:]], 1) if q else o["rgb"][:, :1]
    assert torch.allclose(dec["rgb"], rgb, atol=1e-4)


def test_rgb_from_hidden_bypasses_the_feature_projection():
    """The image does not pass through the DINO-space projection w_out: changing it leaves the image as it is."""
    m = _wake(_model(rgb_from="hidden"))
    x, _ = _inputs("image")
    with torch.no_grad():
        a = m(x, "image", None, return_outputs=True)["outputs"]["rgb"]
        m.readout.w_out.weight.add_(1.0)
        b = m(x, "image", None, return_outputs=True)["outputs"]["rgb"]
    assert torch.allclose(a, b, atol=1e-6)


def test_rgb_from_rejects_unknown_values():
    with pytest.raises(AssertionError):
        tiny_config(rgb_from="pixels").check()


# ---------------------------------------------------------------------------------------------- stability (gp3)
def _read_layer_out(layer, d_pe, seed=0):
    gen = torch.Generator().manual_seed(seed)
    B, Q, G, d = 2, 8, 6, layer.norm.normalized_shape[0]
    a, pe = torch.randn(B, Q, d, generator=gen), torch.randn(B, Q, d_pe, generator=gen)
    k, v = torch.randn(B, G, d, generator=gen), torch.randn(B, G, d, generator=gen)
    return layer(a, pe, k, v, None, torch.ones(B, G, dtype=torch.bool))


def test_read_qk_norm_makes_the_logits_scale_free():
    """gp3 diverged when |W_q|, |W_k| of the grain read grew (18 -> 175) and saturated the softmax: with
    read_qk_norm the read is invariant to the scale of the q / k projections."""
    from mavt.grain_prior.readout import ReadLayer
    cfg = tiny_config(read_qk_norm=True)
    d_pe = cfg.dim                                  # as in Readout: ReadLayer(cfg, d)
    layer = ReadLayer(cfg, d_pe)
    with torch.no_grad():
        a = _read_layer_out(layer, d_pe)
        for lin in (layer.wq, layer.wpe, layer.wk):
            lin.weight.mul_(50.0)
            if lin.bias is not None:
                lin.bias.mul_(50.0)
        b = _read_layer_out(layer, d_pe)
    assert torch.allclose(a, b, atol=1e-4)
    assert not any("qn" in n or "kn" in n for n, _ in ReadLayer(tiny_config(), cfg.dim).named_parameters())  # old ckpts load


def test_image_fov_fixed_pins_the_anchor_camera_of_images_only():
    """Images carry no focal signal (unproject and project with the same kappa_0): image_fov_fixed pins it to
    fov_init with a centred principal point, whatever the camera head says; video keeps its estimate."""
    m = _wake(_model(image_fov_fixed=True))
    with torch.no_grad():
        m.anchor.kappa0[-1].bias.add_(3.0)
        img = m(_inputs("image")[0], "image", None, return_outputs=True)["outputs"]["S"]["kappa0"]
        vid = m(_inputs("video")[0], "video", None, return_outputs=True)["outputs"]["S"]["kappa0"]
    cfg = m.cfg
    f0 = 0.5 / math.tan(math.radians(cfg.fov_init) / 2)
    assert torch.allclose(img, torch.tensor([f0, f0, 0.5, 0.5]).expand_as(img), atol=1e-5)
    assert not torch.allclose(vid, img, atol=1e-3)


def test_sigma_min_sets_the_grain_radius_floor():
    m = _model(sigma_min=0.15)
    with torch.no_grad():
        m.anchor.w_sigma[-1].bias.fill_(-30.0)
        s = m(_inputs("image")[0], "image", None, return_outputs=True)["outputs"]["S"]["sigma"]
    assert torch.allclose(s, torch.full_like(s, 0.15), atol=1e-5)
