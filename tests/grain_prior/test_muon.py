"""Muon + AdamW hybrid (src/mavt/muon.py), the grain_prior parameter groups, and the trainer features of the final
run (WSD, median clip, per-group grad norms, EMA, parameter-group table)."""
import json
import math

import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.muon import MuonAdamW, newton_schulz, wsd_factor, MedianClip

from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _script

FINAL = dict(LEGACY_CRSR, query_grid=16, read_coarse_kv=True, rgb_coarse_start=True, rgb_head="pointwise",
             read_split_layer2=True, innov_stopgrad_base=True, read_prev_frame=True, read_relpos_value=True,
             relpos_rank=4, dino_multi_layer=True)


def test_newton_schulz_orthogonalises():
    torch.manual_seed(0)
    for shape in ((32, 64), (64, 32), (48, 48)):
        s = torch.linalg.svdvals(newton_schulz(torch.randn(*shape), steps=5).float())
        assert 0.5 < s.min() and s.max() < 1.3                    # NS5 (quintic) leaves singular values near 1


def _groups(m):
    return {g["name"]: g for g in m.optim_groups()}


def test_grain_prior_groups_cover_every_parameter_once():
    m = MAVTGrainPrior(tiny_config(**FINAL))
    rows = m.optim_groups()
    names = [r["name"] for r in rows]
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert sorted(names) == sorted(trainable)
    g = _groups(m)
    kind = lambda n: g[n]["kind"]
    assert kind("stream_c.blocks.0.sa.qkv.weight") == "muon" and g["stream_c.blocks.0.sa.qkv.weight"]["splits"] == 3
    assert kind("stream_c.blocks.0.ca.0.kv.weight") == "muon" and g["stream_c.blocks.0.ca.0.kv.weight"]["splits"] == 2
    assert g["readout.layers.0.wv_rel.weight"]["splits"] == 4 and kind("readout.layers.0.wv_rel.weight") == "muon"
    for n in ("readout.layers.0.wq.weight", "readout.last_x.wk_c.weight", "rgb.pw.blocks.0.fc1.weight",
              "innov.blocks.0.mlp.0.weight", "corr.wq.weight", "readout.r_ca.kv.weight"):
        assert kind(n) == "muon", n
    for n in ("stems.w_f.weight", "stems.pixel.net.0.weight", "readout.w_out.weight", "anchor.w_rho.0.weight",
              "stream_d.heads.xi.0.weight", "rgb.pw.stages.0.1.weight", "rgb.pw.final.weight", "bank.grain.weight",
              "readout.ktok.weight", "innov.head.weight", "readout.prev_read.proj.weight"):
        assert kind(n) == "io", n
    for n in ("readout.layers.0.norm.weight", "stream_c.blocks.0.sa.qkv.bias"):
        assert kind(n) == "scalar" and g[n]["lr_scale"] == 1.0, n
    for n in ("readout.layers.0.temperature", "readout.layers.0.sink", "readout.layers.0.tau_c",
              "readout.layers.0.beta_c"):
        assert kind(n) == "attn_scalar" and g[n]["lr_scale"] == pytest.approx(0.1), n


def test_muon_orthogonalises_each_block_of_a_fused_weight():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(96, 32))
    opt = MuonAdamW([dict(params=[w], kind="muon", splits=3, lr=1.0, weight_decay=0.0)], momentum=0.0, nesterov=False)
    g = torch.randn(96, 32)
    w.grad = g.clone()
    opt.step()
    ref = torch.cat([newton_schulz(b) * 0.2 * math.sqrt(32) for b in g.split(32, 0)], 0)
    assert torch.allclose(-w.detach(), ref, atol=1e-4)
    whole = newton_schulz(g) * 0.2 * math.sqrt(96)
    assert not torch.allclose(-w.detach(), whole, atol=1e-2)


def test_adamw_groups_match_torch_adamw():
    torch.manual_seed(0)
    a, b = torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(8, 4))
    b.data.copy_(a.data)
    mine = MuonAdamW([dict(params=[a], kind="io", lr=1e-2, weight_decay=0.05)])
    ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
    for _ in range(3):
        g = torch.randn(8, 4)
        a.grad, b.grad = g.clone(), g.clone()
        mine.step()
        ref.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_muon_state_dict_round_trip_and_a_loss_goes_down():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 32), torch.nn.GELU(),
                              torch.nn.Linear(32, 1))
    mk = lambda: MuonAdamW([dict(params=[net[2].weight], kind="muon", splits=1, lr=2e-2, weight_decay=0.0),
                            dict(params=[net[0].weight, net[4].weight], kind="io", lr=2e-2, weight_decay=0.0),
                            dict(params=[net[0].bias, net[2].bias, net[4].bias], kind="scalar", lr=2e-2,
                                 weight_decay=0.0)])
    opt = mk()
    x = torch.randn(256, 16)
    y = (x[:, :1] * x[:, 1:2]).tanh()
    first = None
    for _ in range(60):
        loss = (net(x) - y).pow(2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
    sd = opt.state_dict()
    opt2 = mk()
    opt2.load_state_dict(sd)
    assert torch.equal(opt2.state[net[2].weight]["momentum_buffer"], opt.state[net[2].weight]["momentum_buffer"])


def test_wsd_factor():
    kw = dict(warmup=10, steps=100, decay_steps=20, min_ratio=0.0)
    assert wsd_factor(0, **kw) == pytest.approx(0.1)
    assert wsd_factor(50, **kw) == 1.0 and wsd_factor(79, **kw) == 1.0
    assert 0 < wsd_factor(90, **kw) < 1 and wsd_factor(100, **kw) == pytest.approx(0.0)
    assert wsd_factor(55, warmup=10, steps=1000, decay_steps=20, min_ratio=0.0, decay_start=50) < 1.0  # branch


def test_median_clip():
    c = MedianClip(mult=2.0, window=10, fallback=1.0)
    assert c.threshold() == 1.0                                       # before the window fills: fixed clip
    for v in [1.0] * 10:
        c.update(v)
    assert c.threshold() == pytest.approx(2.0)
    assert c.update(5.0) is True and c.update(1.5) is False            # clipped or not; history keeps all
    sd = c.state_dict()
    c2 = MedianClip(2.0, 10, 1.0)
    c2.load_state_dict(sd)
    assert c2.threshold() == c.threshold()


def test_trainer_dry_run_with_the_final_optimizer(tmp_path):
    tr = _script("train_grain_prior")
    out = tmp_path / "run"
    args = ["--dry-run", "--mix-modalities", "--out", str(out), "--optimizer", "muon", "--schedule", "wsd",
            "--decay-steps", "1", "--ema", "0.999", "--clip-mode", "median", "--clip-window", "2"]
    sets = [x for k, v in FINAL.items() for x in ("--set", f"{k}={str(v).lower() if isinstance(v, bool) else v}")]
    assert tr.main(args + sets) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert all(math.isfinite(r["loss"]) for r in recs)
    for k in ("gn_muon", "gn_io", "gn_scalar", "clip_frac", "clip_threshold"):
        assert k in recs[-1], k
    table = (out / "param_groups.txt").read_text()
    assert "muon" in table and "attn_scalar" in table and "splits=3" in table
    assert (out / "last_ema.pt").exists()
    ck = torch.load(out / "last.pt", weights_only=False)
    assert "ema" in ck and "clip" in ck
    assert tr.main(args + sets + ["--steps", "4"]) == 0                # resume restores optimizer, EMA and clip
