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
