"""2026-10 architecture flags of grain_prior (all default off), on the tiny config: shapes / ledger, flag on vs off,
gradients. One section per item of the spec (1.1 ... 3.5)."""
import dataclasses

import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, schema as sc, tiny_config
from mavt.grain_prior.config import GrainPriorConfig
from mavt.grain_prior.readout import (PointwiseRGB, ReadLayer, proximity, pyr_down, pyr_up, pyramid_target,
                                      relpos_basis)

from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _inputs, _wake

QG = dict(LEGACY_CRSR, query_grid=16)                     # legacy strict reader, RGB from a 16 x 16 read


def _model(**kw):
    torch.manual_seed(0)
    return MAVTGrainPrior(tiny_config(**kw))


def _record_decodes(m):
    """Wrap decode_obs: every returned dict is kept, its F_base / a keep their gradient."""
    seen, orig = [], m.decode_obs

    def wrapped(*a, **k):
        out = orig(*a, **k)
        for key in ("F_base", "a", "a_rgb"):
            if key in out and out[key].requires_grad:
                out[key].retain_grad()
        seen.append(out)
        return out

    m.decode_obs = wrapped
    return seen


def _grads(m, prefix):
    return [p.grad for n, p in m.named_parameters() if n.startswith(prefix)]


def _nonzero(gs):
    return any(gr is not None and gr.abs().sum() > 0 for gr in gs)


# ------------------------------------------------------------------------------------------------ 1.1
@pytest.mark.parametrize("on", [False, True])
def test_innov_stopgrad_base_cuts_the_prediction_branch_where_L_is_positive(on):
    m = _wake(_model(innov_stopgrad_base=on))
    seen = _record_decodes(m)
    x, cam = _inputs("video", B=2, T=2)
    L = torch.tensor([[0], [6]])                                   # sample 0: L = 0, sample 1: L > 0
    o = m(x, "video", cam, L=L, return_outputs=True)["outputs"]
    o["F_hat"][:, 1].sum().backward()
    base = seen[-1]                                                # the final decode of the target
    gF, ga = base["F_base"].grad, base["a"].grad
    assert gF[0].abs().sum() > 0 and ga[0].abs().sum() > 0         # L = 0: F^ = F_base, unchanged
    if on:
        assert torch.count_nonzero(gF[1]) == 0 and torch.count_nonzero(ga[1]) == 0
    else:
        assert gF[1].abs().sum() > 0


def test_innov_stopgrad_base_needs_L_base():
    with pytest.raises(ValueError):
        tiny_config(innov_stopgrad_base=True, w_base=0.0).check()


def test_innov_stopgrad_base_with_query_grid_cuts_the_rgb_read():
    m = _wake(_model(innov_stopgrad_base=True, **QG))
    seen = _record_decodes(m)
    x, cam = _inputs("video", B=2, T=2)
    o = m(x, "video", cam, L=torch.tensor([[0], [6]]), return_outputs=True)["outputs"]
    o["rgb"][:, 1].sum().backward()
    g_rgb = seen[-1]["a_rgb"].grad
    assert g_rgb[0].abs().sum() > 0 and torch.count_nonzero(g_rgb[1]) == 0


# ------------------------------------------------------------------------------------------------ 1.2
def test_w_geom_ramp_follows_the_step():
    m = _model(w_geom=2.0, w_geom_ramp_steps=100)
    for step, w in ((0, 0.0), (50, 1.0), (100, 2.0), (500, 2.0)):
        m.set_step(step)
        assert m.w_geom_eff() == pytest.approx(w)
    assert _model(w_geom=2.0).w_geom_eff() == 2.0                 # ramp off
    x, cam = _inputs("threed", T=2)
    out = m(x, "threed", cam, step=0)
    assert m.global_step == 0 and out["diag"]["w_geom_eff"] == 0


@pytest.mark.parametrize("wd", [0.0, 1.0])
def test_w_geom_dense_zero_stops_the_dense_depth_loss(wd):
    m = _model(w_geom_dense=wd).train()
    x, cam = _inputs("threed", T=2)
    out = m(x, "threed", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert _nonzero(_grads(m, "readout.depth.")) == (wd > 0)       # depth_dense is read only by L_geom


# ------------------------------------------------------------------------------------------------ 1.3
def test_full_budget_of_the_reference_config():
    cfg = GrainPriorConfig()
    d_f = cfg.full_budget(2)
    assert d_f == (12288 - (32 + 64 * 4 + 16 * 128 + 4)) // 64 == 155
    full = cfg.full_budget_config(2)
    assert full.state_numbers([]) <= 12288 < dataclasses.replace(full, d_f=d_f + 1).state_numbers([])
    assert cfg.full_budget(4) > d_f                                 # fewer grains, more features each


@pytest.mark.parametrize("stratum", [1, 2])
def test_full_budget_state_serializes_to_the_ledger(stratum):
    ceiling = 600
    cfg = tiny_config().full_budget_config(stratum, ceiling)
    m = MAVTGrainPrior(cfg).eval()
    x, _ = _inputs("image")
    with torch.no_grad():
        S = m(x, "image", return_outputs=True)["outputs"]["S"]
    flats = sc.serialize(cfg, S)
    assert all(f.numel() == cfg.state_numbers([]) <= ceiling for f in flats)


# ------------------------------------------------------------------------------------------------ 1.4
def test_dino_multi_layer_adds_an_evidence_source():
    m = _model(dino_multi_layer=True).train()
    assert len(m.stream_c.blocks[0].ca) == 2 and m.anchor.n_src == 4
    x, _ = _inputs("image")
    st = m.stems(x)
    assert st["F_mid"].shape == (2, m.cfg.n_cells, m.cfg.dim)
    out = m(x, "image")
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _nonzero(_grads(m, "stems.w_mid."))
    off = _model()
    assert "F_mid" not in off.stems(x) and not any("w_mid" in k for k in off.state_dict())


def test_dino_multi_layer_needs_two_layers():
    with pytest.raises(AssertionError):
        _model(dino_multi_layer=True, dino_layers=(2,))


# ------------------------------------------------------------------------------------------------ 2.1
def _read_inputs(cfg, far, B=2, Q=6, G=6):
    gen = torch.Generator().manual_seed(0)
    uq = torch.rand(B, Q, 2, generator=gen)
    uh = uq[:, :G] + (5.0 if far else 0.0)
    s, valid = torch.full((B, G), 0.5), torch.ones(B, G, dtype=torch.bool)
    prox = proximity(uq, uh, s, valid, cfg.grid)
    bias = (torch.zeros(B, Q, G) + prox.unsqueeze(-1)).unsqueeze(1)
    a, pe = torch.randn(B, Q, cfg.dim, generator=gen), torch.randn(B, Q, cfg.dim, generator=gen)
    kt, vt = torch.randn(B, G, cfg.dim, generator=gen), torch.randn(B, G, cfg.dim, generator=gen)
    ctok = torch.randn(B, cfg.n_coarse, cfg.dim, generator=gen)
    cbias = (-prox).clamp(max=50.0).unsqueeze(-1).expand(-1, -1, cfg.n_coarse)
    return dict(a=a, pe=pe, ktok=kt, vtok=vt, bias=bias, valid=valid, ctok=ctok, cbias=cbias), uq, uh, s, valid


@pytest.mark.parametrize("sink", [True, False])
def test_coarse_kv_takes_the_read_where_no_grain_covers_the_query(sink):
    cfg = tiny_config(read_coarse_kv=True, coarse_sink=sink)
    layer = ReadLayer(cfg, cfg.dim)
    layer.collect_diag = True
    shares = {}
    with torch.no_grad():
        for far in (False, True):
            layer(**_read_inputs(cfg, far)[0])
            d = layer.last_diag
            tot = d["fine_share_q"] + d["coarse_share_q"] + d["sink_share_q"]
            assert torch.allclose(tot, torch.ones_like(tot), atol=1e-5)
            shares[far] = d
    assert shares[False]["fine_share_q"].mean() > 0.9
    assert shares[True]["fine_share_q"].mean() < 0.01
    assert hasattr(layer, "sink") == sink


def test_coarse_kv_on_the_full_pipeline():
    m = _model(read_coarse_kv=True, **LEGACY_CRSR).train()
    x, cam = _inputs("world", T=3)
    out = m(x, "world", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _nonzero(_grads(m, "readout.coarse_kv_tok."))
    assert _nonzero(_grads(m, "readout.layers.0.wk_c.")) and "coarse_share" in out["diag"]
    with pytest.raises(ValueError):
        tiny_config(read_coarse_kv=True, geom_bias=False).check()


# ------------------------------------------------------------------------------------------------ 2.2
def test_relpos_basis_is_tapered_and_bounded():
    uq = torch.rand(2, 5, 2)
    near, far = uq[:, :3].clone(), uq[:, :3] + 5.0
    s, valid = torch.full((2, 3), 0.5), torch.ones(2, 3, dtype=torch.bool)
    phi = relpos_basis(uq, near, s, valid, 4, 8)
    assert phi.shape == (2, 5, 3, 8) and phi.abs().max() <= 1.0
    assert torch.count_nonzero(relpos_basis(uq, far, s, valid, 4, 8)) == 0          # beyond 3 sigma
    assert torch.count_nonzero(relpos_basis(uq, near, s, ~valid, 4, 8)) == 0        # invalid grains


def test_relpos_value_is_the_plain_read_at_init_and_moves_with_W_r():
    cfg_off, cfg_on = tiny_config(**LEGACY_CRSR), tiny_config(read_relpos_value=True, **LEGACY_CRSR)
    torch.manual_seed(0)
    off = ReadLayer(cfg_off, cfg_off.dim)
    torch.manual_seed(0)
    on = ReadLayer(cfg_on, cfg_on.dim)
    on.load_state_dict(off.state_dict(), strict=False)
    x, uq, uh, s, valid = _read_inputs(cfg_on, far=False)
    x = {k: v for k, v in x.items() if k not in ("ctok", "cbias")}
    phi = relpos_basis(uq, uh, s, valid, cfg_on.grid, cfg_on.relpos_rank)
    with torch.no_grad():
        assert torch.allclose(off(**x), on(**x, phi=phi), atol=1e-5)
        on.wv_rel.weight.normal_(0, 0.1)
        assert not torch.allclose(off(**x), on(**x, phi=phi), atol=1e-3)
    on(**x, phi=phi).sum().backward()
    assert on.wv_rel.weight.grad.abs().sum() > 0


def test_relpos_value_on_the_full_pipeline():
    m = _model(read_relpos_value=True, relpos_rank=4, **LEGACY_CRSR).train()
    x, cam = _inputs("threed", T=2)
    out = m(x, "threed", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _nonzero(_grads(m, "readout.layers.0.wv_rel."))
    with pytest.raises(ValueError):
        tiny_config(read_relpos_value=True, relpos_rank=3).check()


# ------------------------------------------------------------------------------------------------ 2.3
def test_pyramid_keeps_a_constant_and_matches_the_low_size():
    x = torch.full((1, 3, 16, 16), 0.3)
    assert pyr_down(x, 2).shape == (1, 3, 4, 4)
    assert torch.allclose(pyramid_target(x, 2), x, atol=1e-5)
    y = torch.rand(1, 3, 16, 16)
    assert pyr_up(pyr_down(y, 2), 2).shape == y.shape


def test_rgb_coarse_start_begins_at_the_blurred_image_and_trains_it_only_by_its_loss():
    m = _model(rgb_coarse_start=True, w_rgb_coarse=0.0, **QG).train()
    x, cam = _inputs("video", T=2)
    out = m(x, "video", cam, return_outputs=True)
    assert "rgb_coarse" in out["parts"] and "psnr_coarse_rgb" in out["diag"]
    out["loss"].backward()
    assert not _nonzero(_grads(m, "rgb.low."))                     # sg[X~0]: L_rec never reaches RGBLow
    m.zero_grad()
    m.cfg.w_rgb_coarse = 1.0
    m(x, "video", cam)["loss"].backward()
    assert _nonzero(_grads(m, "rgb.low."))
    with torch.no_grad():                                          # R zero-init: X^ = clamp(X~0)
        a0 = torch.randn(2, m.cfg.n_cells, m.cfg.dim)
        H = torch.randn(2, m.cfg.query_grid ** 2, m.cfg.dim)
        low = m.rgb.low_image(a0)
        fresh = _model(rgb_coarse_start=True, **QG)
        assert torch.allclose(fresh.rgb(H, low=fresh.rgb.low_image(a0)), fresh.rgb.low_image(a0))
        assert low.shape == (2, 3, 64, 64)


# ------------------------------------------------------------------------------------------------ 3.1
def test_read_diagnostics_are_reported():
    m = _model(read_coarse_kv=True, **LEGACY_CRSR).eval()
    x, cam = _inputs("video", T=2)
    with torch.no_grad():
        d = m(x, "video", cam)["diag"]
    for k in ("eff_support", "fine_share", "coarse_share", "sink_share", "init_cos_all", "err_low_frac",
              "err_mid_frac", "err_high_frac", "fine_share_hf_corr"):
        assert k in d and torch.isfinite(d[k]), k
    assert d["err_low_frac"] + d["err_mid_frac"] + d["err_high_frac"] == pytest.approx(1.0, abs=1e-4)
    assert 1.0 <= d["eff_support"]


def test_read_diagnostics_without_coarse_slots_are_all_fine():
    m = _model().eval()
    with torch.no_grad():
        d = m(_inputs("image")[0], "image")["diag"]
    assert d["fine_share"] == pytest.approx(1.0) and d["coarse_share"] == 0 and d["sink_share"] == 0


# ------------------------------------------------------------------------------------------------ 3.3
def test_pointwise_head_receptive_field_is_at_most_16_px():
    head = PointwiseRGB(dim=64, n_up=2, ratio=2.0)                 # 16 x 16 grid -> 64 x 64 image (4 px cells)
    torch.nn.init.normal_(head.final.weight, 0, 0.1)
    x = torch.randn(1, 64, 16, 16, requires_grad=True)
    y = head(x)
    assert y.shape == (1, 3, 64, 64)
    y[0, :, 33, 30].sum().backward()
    cells = (x.grad[0].abs().sum(0) > 0).nonzero()
    span_px = 4 * (cells.max(0).values - cells.min(0).values + 1)
    assert cells.numel() and span_px.max() <= 16


def test_pointwise_head_and_feature_input():
    m = _wake(_model(rgb_head="pointwise", rgb_in_feat=True, **QG)).train()      # zero-init last layer woken
    assert hasattr(m.rgb, "pw") and not hasattr(m.rgb, "convs")
    x, cam = _inputs("video", T=2)
    out = m(x, "video", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _nonzero(_grads(m, "rgb.in_feat."))
    with pytest.raises(ValueError):
        tiny_config(rgb_head="unet").check()


def test_rgb_in_feat_is_zero_at_init():
    m = _model(rgb_in_feat=True, **QG)
    H, Fh = torch.randn(2, 256, m.cfg.dim), torch.randn(2, m.cfg.n_cells, m.stems.teacher_dim)
    with torch.no_grad():
        assert torch.allclose(m.rgb(H, feat=Fh), m.rgb(H))


# ------------------------------------------------------------------------------------------------ 3.4
def test_split_layer2_reads_rgb_with_its_own_last_layer():
    """Anchor (image) only: on targets X_base enters the residual encoder (e^RGB), so r moves with the RGB read."""
    m = _wake(_model(read_split_layer2=True, **QG)).eval()
    x, _ = _inputs("image")
    with torch.no_grad():
        o1 = m(x, "image", return_outputs=True)["outputs"]
        for p in m.readout.last_x.parameters():
            p.add_(0.1)
        o2 = m(x, "image", return_outputs=True)["outputs"]
    assert torch.allclose(o1["F_hat"], o2["F_hat"]) and not torch.allclose(o1["rgb"], o2["rgb"])
    assert not any("last_x" in k for k in _model(**QG).state_dict())
    with pytest.raises(ValueError):
        tiny_config(read_split_layer2=True).check()


# ------------------------------------------------------------------------------------------------ 3.5
def test_state_noise_only_in_training_and_ledger_unchanged():
    kw = dict(state_noise_bits=4, geom_noise_px=1.0)
    m, ref = _model(**kw), _model()
    ref.load_state_dict(m.state_dict())
    x, _ = _inputs("image")
    with torch.no_grad():
        a = m.eval()(x, "image", return_outputs=True)["outputs"]
        b = ref.eval()(x, "image", return_outputs=True)["outputs"]
        assert torch.allclose(a["rgb"], b["rgb"])                  # eval: no noise
        torch.manual_seed(1)
        c = m.train()(x, "image", return_outputs=True)["outputs"]
        torch.manual_seed(2)
        d = m.train()(x, "image", return_outputs=True)["outputs"]
    assert not torch.allclose(c["S"]["f"], d["S"]["f"]) and not torch.allclose(c["S"]["u"], d["S"]["u"])
    assert (c["S"]["u"] >= 0).all() and (c["S"]["u"] <= 1).all()
    assert [f.numel() for f in sc.serialize(m.cfg, c["S"])] == [m.cfg.state_numbers([])] * 2
    span = (b["S"]["f"].amax((0, 1)) - b["S"]["f"].amin((0, 1))) / 2 ** 4
    assert ((c["S"]["f"] - b["S"]["f"]).abs() <= span / 2 + 1e-5).all()


# ------------------------------------------------------------------------------------------------ defaults
def test_defaults_add_no_parameters():
    """Every 2026-10 flag at its default: no new state_dict key (checkpoints s1-s3 load strictly)."""
    keys = set(_model().state_dict())
    for name in ("w_mid", "norm_mid", "coarse_kv", "wk_c", "wv_rel", "last_x", "pw.", "in_feat", "rgb.low.",
                 "tau_c", "beta_c", "prev_read", "sem_head", "alpha"):
        assert not any(name in k for k in keys), name


# ------------------------------------------------------------------------------------------------ 2.4
def test_entmax_alpha2_is_sparsemax_and_alpha15_is_sparse():
    from mavt.grain_prior.layers import entmax_bisect
    x = torch.tensor([[0.5, 0.2, -1.0, float("-inf")]])
    p2 = entmax_bisect(x, torch.tensor(2.0))
    assert torch.allclose(p2, torch.tensor([[0.65, 0.35, 0.0, 0.0]]), atol=1e-5)
    p15 = entmax_bisect(torch.tensor([[3.0, 2.9, 0.0, -2.0, float("-inf")]]), torch.tensor(1.5))
    assert p15.sum() == pytest.approx(1.0, abs=1e-5) and (p15[0, 2:] == 0).all() and (p15[0, :2] > 0).all()


def test_entmax_gradients_match_finite_differences():
    from mavt.grain_prior.layers import entmax_bisect
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor([1.3, 1.5, 1.8], dtype=torch.float64).view(1, 3, 1).requires_grad_()
    w = torch.randn(2, 3, 5, dtype=torch.float64)
    f = lambda x_, a_: (entmax_bisect(x_, a_, n_iter=80) * w).sum()
    assert torch.autograd.gradcheck(f, (x, alpha), eps=1e-6, atol=1e-5)


def _rl(cfg, seed=0):
    torch.manual_seed(seed)
    return ReadLayer(cfg, cfg.dim)


def test_read_normalizer_entmax_near_alpha1_is_the_softmax_read():
    cfg = tiny_config(**LEGACY_CRSR)
    soft = _rl(cfg)
    ent = _rl(tiny_config(read_normalizer="entmax15", entmax_alpha_init=1.0, alpha_anneal_steps=100, **LEGACY_CRSR))
    ent.load_state_dict(soft.state_dict(), strict=False)
    x = {k: v for k, v in _read_inputs(cfg, far=False)[0].items() if k not in ("ctok", "cbias")}
    with torch.no_grad():
        ent.step = 0
        assert torch.allclose(soft(**x), ent(**x), atol=1e-5)
        ent.step = 100                                              # alpha = 1.5: sparse, a different read
        assert not torch.allclose(soft(**x), ent(**x), atol=1e-4)


def test_read_normalizer_entmax_learns_alpha_per_head():
    cfg = tiny_config(read_normalizer="entmax15", entmax_alpha_init=1.5, **LEGACY_CRSR)
    layer = _rl(cfg)
    assert layer.alpha().shape == (1, cfg.heads, 1, 1) and torch.allclose(layer.alpha(), torch.tensor(1.5))
    x = {k: v for k, v in _read_inputs(cfg, far=False)[0].items() if k not in ("ctok", "cbias")}
    layer(**x).sum().backward()
    assert layer.alpha_raw.grad is not None and torch.isfinite(layer.alpha_raw.grad).all()


def test_read_normalizer_topk_reads_only_the_k_nearest_grains():
    cfg1 = tiny_config(read_normalizer="topk", read_topk=1, **LEGACY_CRSR)
    top, soft = _rl(cfg1), _rl(tiny_config(**LEGACY_CRSR))
    soft.load_state_dict(top.state_dict())
    gen = torch.Generator().manual_seed(3)
    B, Q, G, d = 2, 5, 6, cfg1.dim
    bias = torch.randn(B, 1, Q, G, generator=gen)
    valid = torch.ones(B, G, dtype=torch.bool)
    a, pe = torch.randn(B, Q, d, generator=gen), torch.randn(B, Q, d, generator=gen)
    kt, vt = torch.randn(B, G, d, generator=gen), torch.randn(B, G, d, generator=gen)
    with torch.no_grad():
        out = top(a, pe, kt, vt, bias, valid)
        best = bias[:, 0].argmax(-1)                                 # [B, Q]
        for q in range(Q):
            only = torch.nn.functional.one_hot(best[:, q], G).bool()
            ref = soft(a[:, q:q + 1], pe[:, q:q + 1], kt, vt, bias[:, :, q:q + 1], only)
            assert torch.allclose(out[:, q:q + 1], ref, atol=1e-5)
        topG = _rl(tiny_config(read_normalizer="topk", read_topk=G, **LEGACY_CRSR))
        topG.load_state_dict(top.state_dict())
        assert torch.allclose(topG(a, pe, kt, vt, bias, valid), soft(a, pe, kt, vt, bias, valid), atol=1e-5)


@pytest.mark.parametrize("kw", [dict(read_normalizer="entmax15"), dict(read_normalizer="topk", read_topk=2),
                                dict(read_normalizer="topk", read_topk=2, read_coarse_kv=True),
                                dict(read_normalizer="entmax15", read_coarse_kv=True, read_relpos_value=True)])
def test_read_normalizer_on_the_full_pipeline(kw):
    m = _model(**kw, **LEGACY_CRSR).train()
    x, cam = _inputs("world", T=3)
    out = m(x, "world", cam, step=10)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert all(l.step == 10 for l in m.readout.layers)


def test_read_normalizer_config_checks():
    for bad in (dict(read_normalizer="sparsemax"), dict(read_normalizer="topk", read_topk=0),
                dict(read_normalizer="topk", read_topk=2, geom_bias=False), dict(entmax_alpha_init=2.5)):
        with pytest.raises(ValueError):
            tiny_config(**bad).check()


# ------------------------------------------------------------------------------------------------ 1.5
def test_prev_read_centres_follow_the_grain_motion():
    from mavt.grain_prior.readout import PrevRead
    cfg = tiny_config(read_prev_frame=True)
    pr = PrevRead(cfg, 16)
    uq = torch.rand(1, 5, 2) * 0.5 + 0.25
    u_prev = uq.clone()                                             # one grain under each query
    shift = torch.tensor([0.1, -0.05])
    valid, s = torch.ones(1, 5, dtype=torch.bool), torch.full((1, 5), 0.5)
    c = pr.centres(uq, u_prev + shift, u_prev, valid, valid, s)
    assert torch.allclose(c, uq - shift, atol=1e-4)
    c0 = pr.centres(uq, u_prev + shift, u_prev, valid, torch.zeros_like(valid), s)   # no previous projection
    assert torch.allclose(c0, uq)


def test_prev_read_reads_only_the_detached_previous_feature():
    from mavt.grain_prior.readout import PrevRead
    cfg = tiny_config(read_prev_frame=True)
    pr = PrevRead(cfg, 16)
    torch.nn.init.normal_(pr.mix.weight, 0, 0.1)
    F_prev = torch.randn(2, cfg.n_cells, 16, requires_grad=True)
    uq, u = torch.rand(2, 7, 2), torch.rand(2, 4, 2)
    valid, s = torch.ones(2, 4, dtype=torch.bool), torch.full((2, 4), 0.5)
    out = pr(F_prev, uq, u, u, valid, valid, s)
    assert out.shape == (2, 7, cfg.dim)
    out.sum().backward()
    assert F_prev.grad is None and pr.mix.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kw", [dict(), QG])
def test_prev_read_is_zero_at_init(kw):
    on, off = _model(read_prev_frame=True, **kw).eval(), _model(**kw).eval()
    off.load_state_dict({k: v for k, v in on.state_dict().items() if "prev_read" not in k})
    x, cam = _inputs("video", T=3)
    with torch.no_grad():
        a, b = on(x, "video", cam, return_outputs=True)["outputs"], off(x, "video", cam, return_outputs=True)["outputs"]
    assert torch.allclose(a["F_hat"], b["F_hat"], atol=1e-5) and torch.allclose(a["rgb"], b["rgb"], atol=1e-5)


def _woken_prev(**kw):
    m = _wake(_model(read_prev_frame=True, **kw)).eval()
    return m


@pytest.mark.parametrize("modality,moves", [("video", True), ("world", True), ("threed", False)])
def test_prev_read_acts_on_video_and_world_only(modality, moves):
    m = _woken_prev()
    x, cam = _inputs(modality, T=3)
    with torch.no_grad():
        a = m(x, modality, cam, return_outputs=True)["outputs"]
        m.readout.prev_read.mix.weight.zero_()
        b = m(x, modality, cam, return_outputs=True)["outputs"]
    assert torch.allclose(a["F_hat"][:, 0], b["F_hat"][:, 0])           # the anchor has no previous frame
    assert (not torch.allclose(a["F_hat"][:, 1:], b["F_hat"][:, 1:])) == moves


@pytest.mark.parametrize("kw", [dict(), QG])
def test_prev_read_decode_of_the_closed_state_matches_without_fold(kw):
    m = _woken_prev(**kw)
    T = m.cfg.queue + 1                                              # no record folded into z_b
    x, cam = _inputs("video", T=T)
    with torch.no_grad():
        o = m(x, "video", cam, return_outputs=True)["outputs"]
        dec = m.decode(sc.deserialize(m.cfg, sc.serialize(m.cfg, o["S"])))
    assert torch.allclose(dec["F"], o["F_hat"], atol=1e-4) and torch.allclose(dec["rgb"], o["rgb"], atol=1e-4)


def test_prev_read_trains_with_L_pred():
    m = _wake(_model(read_prev_frame=True, w_pred_frame=0.5, innov_stopgrad_base=True)).train()
    x, cam = _inputs("video", T=3)
    out = m(x, "video", cam)
    out["loss"].backward()
    assert "pred_feat" in out["parts"] and torch.isfinite(out["loss"])
    assert _nonzero(_grads(m, "readout.prev_read."))


# ------------------------------------------------------------------------------------------------ 2.5
def test_stub_semantic_teacher_is_fixed_and_shaped():
    from mavt.grain_prior.losses import StubSemanticTeacher
    t = StubSemanticTeacher(24)
    x = torch.rand(3, 3, 64, 64) * 2 - 1
    a, b = t(x), StubSemanticTeacher(24)(x)
    assert a.shape == (3, 24) and torch.equal(a, b) and not torch.allclose(a[0], a[1])


def test_semantic_head_trains_c0_and_leaves_the_state_alone():
    m = _model(sem_target="stub", w_sem=1.0, sem_dim=24).train()
    x, cam = _inputs("video", T=2)
    out = m(x, "video", cam, return_outputs=True)
    assert "sem" in out["parts"] and 0 <= out["parts"]["sem"] <= 2
    out["loss"].backward()
    assert _nonzero(_grads(m, "sem_head.")) and _nonzero(_grads(m, "anchor.w_c."))
    S = out["outputs"]["S"]
    assert [f.numel() for f in sc.serialize(m.cfg, S)] == [m.cfg.state_numbers([int(S["recs"][0]["L"][b])])
                                                           for b in range(2)]
    assert not any("teacher" in k for k in m.state_dict())
    with torch.no_grad():
        assert "sem" not in m.eval()(x, "video", cam)["parts"]              # train only


def test_semantic_teacher_plugs_in_and_external_ones_need_it():
    m = _model(sem_target="siglip2", w_sem=1.0, sem_dim=16).train()
    x, _ = _inputs("image")
    with pytest.raises(NotImplementedError):
        m(x, "image")
    m.set_sem_teacher(lambda frames: torch.ones(frames.shape[0], 16))
    out = m(x, "image")
    assert torch.isfinite(out["loss"]) and "sem" in out["parts"]


def test_semantic_config_checks():
    for bad in (dict(sem_target="clip"), dict(w_sem=1.0), dict(sem_target="stub", w_sem=-1.0)):
        with pytest.raises(ValueError):
            tiny_config(**bad).check()


# ------------------------------------------------------------------------------------------------ review fixes
@pytest.mark.parametrize("kw", [dict(rgb_from="hidden"), dict(LEGACY_CRSR, query_grid=8)])
def test_rgb_in_feat_reads_the_dino_width(kw):
    """in_feat maps D_t (not the head input width) to dim: stub DINO width 48 != dim 32."""
    m = _wake(_model(rgb_in_feat=True, stub_dino_dim=48, **kw)).train()
    x, cam = _inputs("video", T=2)
    out = m(x, "video", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _nonzero(_grads(m, "rgb.in_feat."))


def test_coarse_kv_keeps_the_depth_weights_normalised():
    """read_coarse_kv: the proximity shift of the sink / coarse logits stays out of nu, so depth_grain is a convex
    combination of the grain depths."""
    m = _wake(_model(read_coarse_kv=True, **LEGACY_CRSR)).eval()
    x, cam = _inputs("threed", T=2)
    seen = _record_decodes(m)
    with torch.no_grad():
        m(x, "threed", cam)
    o = seen[-1]
    z, v = o["tr"]["z"].float(), o["tr"]["valid"]
    lo = z.masked_fill(~v, float("inf")).amin(-1, keepdim=True)
    hi = z.masked_fill(~v, -float("inf")).amax(-1, keepdim=True)
    assert ((o["depth_grain"] >= lo - 1e-4) & (o["depth_grain"] <= hi + 1e-4)).all()


def test_read_diagnostics_ignore_rows_the_read_zeroes():
    cfg = tiny_config(**LEGACY_CRSR)                               # strict reader with a sink
    cfg_ns = tiny_config(readout_version="legacy", read_strict=True)
    for c in (cfg, cfg_ns):
        layer = ReadLayer(c, c.dim)
        layer.collect_diag = True
        x = {k: v for k, v in _read_inputs(c, far=False)[0].items() if k not in ("ctok", "cbias")}
        x["valid"] = torch.zeros_like(x["valid"])                  # no valid grain at all
        with torch.no_grad():
            layer(**x)
        d = layer.last_diag
        assert torch.count_nonzero(d["fine_share_q"]) == 0 and torch.count_nonzero(d["grain_mass"]) == 0


def test_read_diagnostics_only_on_the_feature_grid():
    m = _model(**QG).eval()
    seen = _record_decodes(m)
    with torch.no_grad():
        m(_inputs("image")[0], "image")
    assert seen[-1]["diag"] is not None and seen[-1]["diag"]["fine_share_q"].shape[-1] == m.cfg.n_cells


def test_pointwise_head_bounds_the_receptive_field():
    with pytest.raises(ValueError):
        tiny_config(rgb_head="pointwise", dim=256).check()          # no query_grid: 16 x upsampling, RF > 16 px
    tiny_config(rgb_head="pointwise", **QG).check()                 # 4 x upsampling
