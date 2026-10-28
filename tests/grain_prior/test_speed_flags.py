"""Speed switches with no effect on the math: read_act_ckpt=false (2.2 explicit reads without activation
checkpointing) and set_diag(False) (3.1 read diagnostics only on the steps the trainer logs)."""
import pytest
import torch

from mavt.grain_prior import readout as ro, tiny_config
from mavt.grain_prior.readout import ReadLayer

from .test_arch_flags import QG, _model, _read_inputs
from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _inputs, _script, _wake

READ_3_1 = ("eff_support", "fine_share", "coarse_share", "sink_share", "init_cos_all", "err_low_frac")


def test_read_act_ckpt_off_skips_checkpointing_and_keeps_output_and_gradients(monkeypatch):
    cfgs = {c: tiny_config(read_relpos_value=True, read_act_ckpt=c, **LEGACY_CRSR) for c in (True, False)}
    torch.manual_seed(0)
    layers = {c: ReadLayer(cfgs[c], cfgs[c].dim) for c in (True, False)}
    layers[False].load_state_dict(layers[True].state_dict())
    with torch.no_grad():
        for l in layers.values():
            w = l.wv_rel.weight
            w.copy_(torch.linspace(-0.1, 0.1, w.numel()).view_as(w))
    x, uq, uh, s, valid = _read_inputs(cfgs[True], far=False)
    x = {k: v for k, v in x.items() if k not in ("ctok", "cbias")}
    rel = (uq, uh, s, valid, cfgs[True].grid)
    calls, real = {}, ro.checkpoint
    out = {}
    for c, l in layers.items():
        calls[c] = 0
        def spy(*a, _c=c, **k):
            calls[_c] += 1
            return real(*a, **k)
        monkeypatch.setattr(ro, "checkpoint", spy)
        out[c] = l(**x, relpos=rel)
        out[c].square().sum().backward()
    assert calls[True] > 0 and calls[False] == 0
    assert torch.allclose(out[True], out[False], atol=1e-6)
    for (n, a), (_, b) in zip(layers[True].named_parameters(), layers[False].named_parameters()):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-6), n


def test_read_act_ckpt_default_is_on():
    assert tiny_config().read_act_ckpt is True


@pytest.mark.parametrize("kind", ["image", "video"])
def test_set_diag_off_skips_the_3_1_read_diagnostics_only(kind):
    m = _wake(_model(read_coarse_kv=True, **LEGACY_CRSR)).eval()
    x, cam = _inputs(kind, T=2) if kind == "video" else _inputs(kind)
    with torch.no_grad():
        on = m(x, kind, cam) if kind == "video" else m(x, kind)
        m.set_diag(False)
        off = m(x, kind, cam) if kind == "video" else m(x, kind)
        m.set_diag(True)
        again = m(x, kind, cam) if kind == "video" else m(x, kind)
    assert all(k in on["diag"] and k in again["diag"] for k in READ_3_1)
    assert not any(k in off["diag"] for k in READ_3_1)
    assert "psnr" in off["diag"]                                   # the cheap diagnostics stay
    assert torch.equal(on["loss"], off["loss"])


def test_set_diag_off_turns_off_the_read_layer_collection():
    m = _wake(_model(**QG)).eval()
    m.set_diag(False)
    with torch.no_grad():
        m(_inputs("image")[0], "image")
    assert not any(isinstance(mod, ReadLayer) and mod.collect_diag for mod in m.modules())


def test_log_due():
    tr = _script("train_dino_v0")
    assert [s for s in range(1, 60) if tr.log_due(s, 25, 55)] == [25, 50, 55]


def test_trainer_logs_read_diagnostics_on_log_steps(tmp_path):
    import json
    tr = _script("train_grain_prior")
    assert tr.main(["--dry-run", "--out", str(tmp_path), "--set", "read_coarse_kv=true"]) == 0
    recs = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert recs and all(any(k.endswith("coarse_share") for k in r) for r in recs)
