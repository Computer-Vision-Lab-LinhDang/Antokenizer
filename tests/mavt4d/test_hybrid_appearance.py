"""Write-once appearance bank and its decoder readout."""

import pytest
import torch

from mavt4d.contracts import DecoderQuery
from mavt4d.hybrid.appearance import AppearanceWriter, write_once
from mavt4d.hybrid.config import HybridConfig, HybridV3Config
from mavt4d.hybrid.decoder import HybridQueryDecoder
from mavt4d.hybrid.state import CDACode, CDCode


def cfg3(**kw):
    base = HybridConfig.tiny()
    return HybridV3Config.tiny(appearance_tokens=base.state_tokens, **kw)


def codes(cfg, *, zero_c=True, zero_a=False, batch=2):
    shape = (batch, cfg.stage_count, cfg.state_tokens, cfg.state_dim)
    g = torch.Generator().manual_seed(7)
    c = torch.zeros(shape) if zero_c else torch.randn(shape, generator=g)
    a = torch.zeros(batch, cfg.appearance_tokens, cfg.state_dim) if zero_a else \
        torch.randn(batch, cfg.appearance_tokens, cfg.state_dim, generator=g)
    return CDACode(c, torch.randn(shape, generator=g), torch.full((batch,), 2., dtype=torch.float64),
                   a, config=cfg)


def query(points=((2., 2.), (6., 6.))):
    return DecoderQuery(torch.tensor([[x, y, 0.] for x, y in points]),
                        torch.zeros(len(points), dtype=torch.int64),
                        torch.full((len(points),), 2., dtype=torch.float64))


def test_write_once_keeps_first_observation():
    old, new = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    valid = torch.zeros(2, dtype=torch.bool)
    obs = torch.ones(2, dtype=torch.bool)
    first, valid = write_once(torch.zeros(2, 3, 4), valid, old, obs)
    assert valid.all() and torch.equal(first, old)
    second, valid = write_once(first, valid, new, obs)
    assert torch.equal(second, old) and valid.all()


def test_unobserved_step_does_not_write():
    base = torch.randn(2, 3, 4)
    out, valid = write_once(base, torch.zeros(2, dtype=torch.bool), torch.randn(2, 3, 4),
                            torch.tensor([True, False]))
    assert valid.tolist() == [True, False]
    assert torch.equal(out[1], base[1]) and not torch.equal(out[0], base[0])


def test_write_once_does_not_mutate_inputs():
    a, w = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    v, o = torch.tensor([False, True]), torch.tensor([True, True])
    a0, w0, v0, o0 = a.clone(), w.clone(), v.clone(), o.clone()
    write_once(a, v, w, o)
    assert all(torch.equal(x, y) for x, y in ((a, a0), (w, w0), (v, v0), (o, o0)))


def test_write_once_gradient_reaches_only_new_rows():
    w = torch.randn(2, 3, 4, requires_grad=True)
    out, _ = write_once(torch.randn(2, 3, 4), torch.tensor([False, True]), w, torch.tensor([True, True]))
    out.sum().backward()
    assert torch.all(w.grad[0] == 1) and torch.all(w.grad[1] == 0)


def test_writer_shape_and_validation():
    cfg = cfg3()
    writer = AppearanceWriter(cfg)
    out = writer(torch.randn(2, cfg.summary_tokens, cfg.width))
    assert out.shape == (2, cfg.appearance_tokens, cfg.state_dim) and out.dtype == torch.float32
    with pytest.raises(ValueError):
        writer(torch.randn(2, cfg.summary_tokens + 1, cfg.width))
    with pytest.raises(ValueError):
        AppearanceWriter(HybridV3Config.tiny(innovation=True))


def test_decoder_zero_codes_render_zero():
    cfg = cfg3()
    dec = HybridQueryDecoder(cfg)
    rgb = dec(codes(cfg, zero_a=True), query())
    assert torch.equal(rgb, torch.zeros_like(rgb))


def test_decoder_uses_appearance():
    cfg = cfg3()
    dec, writer = HybridQueryDecoder(cfg), AppearanceWriter(cfg)
    a = writer(torch.randn(2, cfg.summary_tokens, cfg.width))
    code = CDACode(*(getattr(codes(cfg), n) for n in ("c", "d", "last_time")), a, config=cfg)
    rgb = dec(code, query())
    assert rgb.abs().sum() > 0
    rgb.sum().backward()
    grads = [p.grad for p in writer.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_appearance_keys_are_used():
    cfg = cfg3()
    dec = HybridQueryDecoder(cfg)
    code = codes(cfg)
    first = dec(code, query())
    with torch.no_grad():
        dec.appearance_keys.add_(torch.randn_like(dec.appearance_keys))
    assert not torch.allclose(first, dec(code, query()))


def test_v3_decoder_requires_cda_code():
    cfg = cfg3()
    plain = codes(cfg)
    with pytest.raises(TypeError):
        HybridQueryDecoder(cfg)(CDCode(plain.c, plain.d, plain.last_time, config=cfg), query())


def test_v2_decoder_unchanged():
    cfg = HybridConfig.tiny()
    dec = HybridQueryDecoder(cfg)
    assert not any("appearance" in k for k in dec.state_dict())
    shape = (2, cfg.stage_count, cfg.state_tokens, cfg.state_dim)
    code = CDCode(torch.randn(shape), torch.randn(shape), torch.full((2,), 2., dtype=torch.float64), config=cfg)
    assert dec(code, query()).shape == (2, 2, 3)


def test_innovation_only_decoder_matches_v2_layout():
    v3 = HybridQueryDecoder(HybridV3Config.tiny(innovation=True))
    assert not hasattr(v3, "appearance_keys")
    assert list(v3.state_dict()) == list(HybridQueryDecoder(HybridConfig.tiny()).state_dict())
