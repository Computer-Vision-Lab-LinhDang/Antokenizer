"""3.13 per-grain feature update in the target record (grain_df_k): each target record carries a code
dfc [B, G, k]; the decoder reads the grains with f + dfc @ W at that observation. W starts at zero, so a model
with the channel decodes exactly as without it until W moves. Tiny config, CPU; default off."""
import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior import schema as sc

from .test_grain_prior import _inputs, _wake

K = 3


def _state(m, B=2, T=3, seed=0):
    x, cam = _inputs("video", B=B, T=T, seed=seed)
    m.eval()
    with torch.no_grad():
        return m(x, "video", cam, return_outputs=True)["outputs"]["S"]


def test_off_by_default_and_validated():
    cfg = tiny_config()
    assert cfg.grain_df_k == 0
    S = _state(MAVTGrainPrior(cfg))
    assert all("dfc" not in r for r in S["recs"])
    for bad in (-1, cfg.d_f + 1):
        with pytest.raises(ValueError):
            tiny_config(grain_df_k=bad).check()


def test_records_carry_the_code_and_the_ledger_counts_it():
    cfg = tiny_config(grain_df_k=K)
    S = _state(_wake(MAVTGrainPrior(cfg)))
    for r in S["recs"]:
        assert r["dfc"].shape == (2, cfg.n_grains, K)
    Ls = [int(r["L"][0]) for r in S["recs"]]
    assert cfg.target_numbers(Ls[0]) == tiny_config().target_numbers(Ls[0]) + cfg.n_grains * K
    flats = sc.serialize(cfg, S)
    assert all(f.numel() == cfg.state_numbers(Ls) for f in flats)
    S2 = sc.deserialize(cfg, flats)
    for r, r2 in zip(S["recs"], S2["recs"]):
        assert torch.allclose(r["dfc"], r2["dfc"])


def test_zero_map_at_init_leaves_the_decode_unchanged():
    torch.manual_seed(0)
    m = MAVTGrainPrior(tiny_config(grain_df_k=K))
    assert torch.count_nonzero(m.gdelta.W.weight) == 0
    S = _state(m)
    noisy = dict(S, recs=[dict(r, dfc=torch.randn_like(r["dfc"])) for r in S["recs"]])
    with torch.no_grad():
        assert torch.equal(m.decode(S)["rgb"], m.decode(noisy)["rgb"])


def test_decode_reads_the_code_on_targets_only():
    m = _wake(MAVTGrainPrior(tiny_config(grain_df_k=K)))
    S = _state(m)
    zero = dict(S, recs=[dict(r, dfc=torch.zeros_like(r["dfc"])) for r in S["recs"]])
    with torch.no_grad():
        a, b = m.decode(S)["rgb"], m.decode(zero)["rgb"]
    assert torch.equal(a[:, 0], b[:, 0])                                  # the anchor has no record
    assert not torch.allclose(a[:, 1:], b[:, 1:])


def test_the_channel_trains():
    torch.manual_seed(0)
    m = _wake(MAVTGrainPrior(tiny_config(grain_df_k=K))).train()
    x, cam = _inputs("video", B=2, T=3)
    out = m(x, "video", cam)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for name in ("gdelta.W.weight", "gdelta.out.weight"):
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_warm_start_lists_the_channel_as_new():
    m = MAVTGrainPrior(tiny_config(grain_df_k=K))
    keys = [k for k in m.state_dict() if k.startswith("gdelta.")]
    assert keys and set(keys) <= set(m.new_param_prefixes())


def test_strip_target_posterior_drops_the_code():
    from mavt.grain_prior.eval_baselines import strip_target_posterior
    cfg = tiny_config(grain_df_k=K)
    S = _state(_wake(MAVTGrainPrior(cfg)))
    for r in strip_target_posterior(cfg, S)["recs"]:
        assert torch.count_nonzero(r["dfc"]) == 0
    assert any(torch.count_nonzero(r["dfc"]) > 0 for r in S["recs"])          # S untouched


def test_the_channel_needs_the_legacy_readout():
    with pytest.raises(ValueError):
        tiny_config(grain_df_k=K, readout_version="shared_ray_v1").check()


def test_mechanism_has_a_grain_update_stage():
    from mavt.grain_prior.evaluate_mechanism import _targets
    m = _wake(MAVTGrainPrior(tiny_config(grain_df_k=K)))
    S = _state(m)
    der, zs = m.derived(S), m.replay(S)
    with torch.no_grad():
        a = _targets(m, S, der, zs, "C+xi+M")
        b = _targets(m, S, der, zs, "C+xi+M+df")
    assert not torch.allclose(a[0]["F_hat"], b[0]["F_hat"])
