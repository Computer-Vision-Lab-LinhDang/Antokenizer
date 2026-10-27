"""CRSR reader (shared_ray_v1 + coarse residual init, learned zero-value sink, projected grain radius) and the
log-parametrised grain radius."""
import math

import pytest
import torch

from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.pilot import configure_pilot, pilot_step
from mavt.grain_prior.query import Camera
from mavt.grain_prior.readout import ReadLayer, projected_radius, proximity

CRSR = dict(readout_version="shared_ray_v1", innovation_semantics="disabled", read_coarse_init=True,
            read_sink=True, read_radius_scale=True)


@pytest.mark.parametrize("flag", ["read_coarse_init", "read_sink", "read_radius_scale"])
def test_crsr_flags_need_the_shared_ray_reader(flag):
    with pytest.raises(ValueError):
        tiny_config(**{flag: True}).check()
    tiny_config(readout_version="shared_ray_v1", innovation_semantics="disabled", **{flag: True}).check()


def _layer_inputs(cfg, far):
    gen = torch.Generator().manual_seed(0)
    B, Q, G, d = 2, 6, 6, cfg.dim
    uq = torch.rand(B, Q, 2, generator=gen)
    uh = uq[:, :G] + (5.0 if far else 0.0)                      # grains on the queries, or 5 images away
    s = torch.full((B, G), 0.5)
    valid = torch.ones(B, G, dtype=torch.bool)
    bias = (torch.zeros(B, Q, G) + proximity(uq, uh, s, valid, cfg.grid).unsqueeze(-1)).unsqueeze(1)
    a, pe = torch.randn(B, Q, d, generator=gen), torch.randn(B, Q, d, generator=gen)
    kt, vt = torch.randn(B, G, d, generator=gen), torch.randn(B, G, d, generator=gen)
    return a, pe, kt, vt, bias, valid


def test_sink_takes_the_read_when_no_grain_is_near():
    cfg = tiny_config(**CRSR)
    layer = ReadLayer(cfg, cfg.dim)
    layer.track_sink = True
    with torch.no_grad():
        layer(*_layer_inputs(cfg, far=False))
        near = layer.last_sink_share
        out_far = layer(*_layer_inputs(cfg, far=True))
        far = layer.last_sink_share
        a, pe, kt, vt, bias, valid = _layer_inputs(cfg, far=True)
        empty = layer(a, pe, kt, vt, bias, torch.zeros_like(valid))          # no valid grain at all
    assert near < 0.1 and far > 0.999
    assert torch.allclose(out_far, empty, atol=1e-4)                        # sink value is zero


def test_sink_reader_keeps_its_logits_bounded():
    cfg = tiny_config(**CRSR)
    layer = ReadLayer(cfg, cfg.dim)
    x = _layer_inputs(cfg, far=False)
    with torch.no_grad():
        a = layer(*x)
        for lin in (layer.wq, layer.wpe, layer.wk):
            lin.weight.mul_(100.0)
            lin.bias.mul_(100.0) if lin.bias is not None else None
        b = layer(*x)
    assert torch.allclose(a, b, atol=1e-4)


def test_projected_radius_follows_depth_and_focal():
    s, d0 = torch.full((1, 3), 0.6), torch.tensor([[1.0, 2.0, 4.0]])
    k0 = torch.tensor([[1.0, 1.0, 0.5, 0.5]])
    assert torch.allclose(projected_radius(s, d0, d0, k0, k0), s)
    assert torch.allclose(projected_radius(s, d0, 2 * d0, k0, k0), s / 2)
    assert torch.allclose(projected_radius(s, d0, d0, k0, k0 * torch.tensor([2.0, 2.0, 1.0, 1.0])), 2 * s)


def test_coarse_init_reads_the_coarse_grid_without_the_memory_attention():
    m = MAVTGrainPrior(tiny_config(**CRSR))
    names = [n for n, _ in m.query_decoder.ray.named_parameters()]
    assert not any(n.startswith(("init_ca.", "mem_")) for n in names) and any(n.startswith("coarse_in.") for n in names)
    ro = m.query_decoder.ray
    B = 2
    pe = ro.query_pe(B)
    c0 = torch.randn(B, m.cfg.n_coarse, m.cfg.d_coarse)
    pg = torch.zeros(B, 18)
    with torch.no_grad():
        a = ro.init(pe, c0, c0, pg)
        b = ro.init(pe, c0 + 1.0, c0, pg)
    assert not torch.allclose(a, b)


def test_log_sigma_keeps_its_gradient_below_the_old_floor():
    """sigmoid with a floor: d sigma / d raw = span * s(1 - s) -> 0; log: d log sigma / d raw = 1."""
    for kind, r in (("sigmoid", -6.0), ("log", -2.0)):        # log at -2: sigma ~ 0.13, below the old floor
        raw = torch.tensor([r], requires_grad=True)
        cfg = tiny_config(sigma_param=kind)
        from mavt.grain_prior.anchor import sigma_from_raw
        s = sigma_from_raw(raw, cfg)
        (g,) = torch.autograd.grad(s.log().sum(), raw)
        if kind == "sigmoid":
            assert g.item() < 0.01 and s.item() >= cfg.sigma_min
        else:
            assert g.item() == pytest.approx(1.0) and s.item() < cfg.sigma_min


def test_pilot_step_trains_the_crsr_reader():
    m = MAVTGrainPrior(tiny_config(**CRSR, sigma_param="log", render_samples=4))
    configure_pilot(m, frozen_encoder=False)
    cam = Camera(torch.tensor([[1.0, 1.0, 0.5, 0.5]]), torch.eye(3)[None], torch.zeros(1, 3))
    x = torch.rand(1, 3, 64, 64) * 2 - 1
    r = pilot_step(m, x, cam.kappa, x, cam, torch.ones(1, 64, 64), rays=16, seed=0)
    r["loss"].backward()
    assert math.isfinite(r["mse"].item()) and math.isfinite(r["sigma_mean"].item())
    sink = [p.grad for n, p in m.named_parameters() if n.endswith(".sink")]
    assert sink and all(g is not None and torch.isfinite(g).all() for g in sink)
    assert m.query_decoder.ray.coarse_in.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("version", ["legacy", "shared_ray_v1", "crsr", "shared_field_v1"])
def test_pilot_step_runs_under_bf16_autocast(version):
    """The trainer wraps pilot_step in bf16 autocast; geometry checks and accumulations must stay in fp32."""
    kw = dict(CRSR, sigma_param="log") if version == "crsr" else dict(readout_version=version, innovation_semantics="disabled")
    m = MAVTGrainPrior(tiny_config(render_samples=4, **kw))
    configure_pilot(m, frozen_encoder=False)
    cam = Camera(torch.tensor([[1.0, 1.0, 0.5, 0.5]]), torch.eye(3)[None], torch.zeros(1, 3))
    x = torch.rand(1, 3, 64, 64) * 2 - 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        r = pilot_step(m, x, cam.kappa, x, cam, torch.ones(1, 64, 64), rays=16, seed=0)
    r["loss"].backward()
    assert torch.isfinite(r["loss"]) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


LEGACY_CRSR = dict(readout_version="legacy", read_strict=True, read_coarse_init=True, read_sink=True,
                   read_radius_scale=True, sigma_param="log", rgb_from="hidden")


def test_read_strict_only_on_the_legacy_readout():
    with pytest.raises(ValueError):
        tiny_config(readout_version="shared_ray_v1", innovation_semantics="disabled", read_strict=True).check()
    tiny_config(**LEGACY_CRSR).check()


@pytest.mark.parametrize("modality", ["image", "video", "threed", "world"])
def test_crsr_flags_on_the_full_legacy_pipeline(modality):
    """The multi-modal trainer path (forward, innovation, conv RGB head) with the CRSR reader."""
    from .test_grain_prior import _inputs
    m = MAVTGrainPrior(tiny_config(**LEGACY_CRSR)).train()
    assert all(l.strict and hasattr(l, "sink") for l in m.readout.layers)
    x, cam = _inputs(modality, T=4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x, modality, cam, return_outputs=True)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.readout.layers[0].sink.grad is not None and m.readout.coarse_in.weight.grad is not None


@pytest.mark.parametrize("modality,T", [("image", 1), ("threed", 4)])
def test_crsr_legacy_decode_of_the_closed_state_matches(modality, T):
    from mavt.grain_prior import schema as sc
    from .test_grain_prior import _inputs, _wake
    m = _wake(MAVTGrainPrior(tiny_config(**LEGACY_CRSR)).eval())
    x, cam = _inputs(modality, T=T)
    with torch.no_grad():
        o = m(x, modality, cam, return_outputs=True)["outputs"]
        dec = m.decode(sc.deserialize(m.cfg, sc.serialize(m.cfg, o["S"])))
    q = len(o["S"]["recs"])
    n = 1 if modality == "image" else T
    rgb = torch.cat([o["rgb"][:, :1], o["rgb"][:, n - q:]], 1) if q else o["rgb"][:, :1]
    assert torch.allclose(dec["rgb"], rgb, atol=1e-4)


def test_shared_ray_checkpoint_moves_to_the_legacy_pipeline(tmp_path):
    """Encoder and the ray reader carry over (query_decoder.ray.* -> readout.*); the conv RGB head starts fresh;
    the result loads with the full trainer's strict --init."""
    import dataclasses
    from mavt.grain_prior.pilot import shared_ray_to_legacy
    src = MAVTGrainPrior(tiny_config(**CRSR, sigma_param="log", rgb_from="hidden"))
    p, q = tmp_path / "shared.pt", tmp_path / "legacy.pt"
    torch.save(dict(model=src.state_dict(), cfg=dataclasses.asdict(src.cfg), step=7), p)
    receipt = shared_ray_to_legacy(p, q)
    ck = torch.load(q, weights_only=False)
    dst = MAVTGrainPrior(tiny_config(**{k: v for k, v in ck["cfg"].items() if k in LEGACY_CRSR}))
    assert dst.cfg.readout_version == "legacy" and dst.cfg.read_strict and dst.cfg.read_sink
    missing, unexpected = dst.load_state_dict(ck["model"], strict=False)
    assert not unexpected and not [k for k in missing if not k.startswith("stems.dino.")]
    for k, v in src.query_decoder.ray.state_dict().items():
        assert torch.equal(dst.readout.state_dict()[k], v)
    for k, v in src.anchor.state_dict().items():
        assert torch.equal(dst.anchor.state_dict()[k], v)
    assert receipt["step"] == 7 and receipt["fresh"] and all(k.startswith("rgb.") for k in receipt["fresh"])


def test_query_grid_needs_the_hidden_legacy_rgb():
    for bad in (dict(query_grid=8, rgb_from="feature"), dict(query_grid=6), dict(query_grid=2)):
        with pytest.raises(AssertionError):
            tiny_config(**{**LEGACY_CRSR, **bad}).check()
    tiny_config(**LEGACY_CRSR, query_grid=16).check()


@pytest.mark.parametrize("modality", ["image", "threed", "world"])
def test_query_grid_reads_rgb_on_the_finer_grid(modality):
    """RGB from a 16 x 16 read (4 px cells) of the 64 px tiny image, cell-grid losses unchanged."""
    from .test_grain_prior import _inputs, _wake
    m = _wake(MAVTGrainPrior(tiny_config(**LEGACY_CRSR, query_grid=16))).train()
    assert len(m.rgb.convs) == 2 and m.uq_rgb.shape == (256, 2)
    x, cam = _inputs(modality, T=3)
    out = m(x, modality, cam, return_outputs=True)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert out["outputs"]["rgb"].shape[-2:] == (64, 64) and out["outputs"]["F_hat"].shape[-2] == m.cfg.grid ** 2
    with torch.no_grad():
        dec = m.eval().decode(m(x, modality, cam, return_outputs=True)["outputs"]["S"])
    assert dec["rgb"].shape[-2:] == (64, 64) and torch.isfinite(dec["rgb"]).all()


def test_projected_radius_clip_bounds_the_scale():
    sigma, d = torch.full((1, 3), 0.5), torch.ones(1, 3)
    z = torch.tensor([[1e-3, 1.0, 1e3]])
    k = torch.tensor([[1.0, 1.0, 0.5, 0.5]])
    assert projected_radius(sigma, d, z, k, k)[0, 0] > 100                       # unbounded: covers the image
    r = projected_radius(sigma, d, z, k, k, clip=8.0)
    assert torch.allclose(r, torch.tensor([[4.0, 0.5, 0.0625]]))
