from dataclasses import replace
import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.objectives import slice_state
from mavt4d.natural_trial.model import NaturalTrialModule, TrialConfig


def test_scaled_budget_validation_and_fingerprint():
    cfg = TrialConfig()
    changed = replace(cfg, global_clips=768, steps=10000, dev_every=50, image_every=25)
    assert changed.fingerprint() != cfg.fingerprint()
    with pytest.raises(ValueError):
        replace(cfg, global_clips=0)


def test_capacity_selection_uses_stability_margin_and_useful_throughput():
    from mavt4d.natural_trial.calibration import choose_candidate
    rows=[dict(batch_per_gpu=8,status="stable",peak_fraction=.1,clips_per_second=100.),
          dict(batch_per_gpu=32,status="stable",peak_fraction=.3,clips_per_second=180.),
          dict(batch_per_gpu=128,status="stable",peak_fraction=.7,clips_per_second=175.),
          dict(batch_per_gpu=256,status="stable",peak_fraction=.9,clips_per_second=190.),
          dict(batch_per_gpu=512,status="failed_oom_or_nonfinite")]
    assert choose_candidate(rows)==128
    assert choose_candidate([]) is None


def test_local_allocation_failure_cleans_before_collective(monkeypatch):
    from mavt4d.natural_trial import calibration
    events=[]
    def failed():
        events.append("allocate")
        raise torch.OutOfMemoryError("simulated")
    def cleanup(): events.append("cleanup")
    def collective(flag, **kwargs):
        events.append("collective")
        assert events[-2]=="cleanup" and flag.item()==1
    monkeypatch.setattr(calibration.dist,"all_reduce",collective)
    assert calibration._local_phase(failed,cleanup,torch.zeros((),dtype=torch.int32)) is False
    assert events==["allocate","cleanup","collective","cleanup"]


def test_batched_endpoint_decoder_matches_individual_different_absolute_times_and_gradients(monkeypatch):
    cfg = replace(TrialConfig(), architecture=HybridConfig.tiny(), height=16, width=24)
    module = NaturalTrialModule(cfg)
    state = module.model.initial_state(3, last_time=torch.tensor([.3, 123456789.3, 9.75], dtype=torch.float64))
    state = replace(state, c=torch.randn_like(state.c, requires_grad=True),
                    d=torch.randn_like(state.d, requires_grad=True))
    times = state.last_time[:, None].expand(3, 12).clone()
    expected = torch.cat([module.model.decoder.render(slice_state(state, i, i+1).code(),
        height=16, width=24, time=float(times[i, 11])) for i in range(3)])
    expected_grad = torch.autograd.grad(expected.square().mean(), (state.c,state.d), retain_graph=True)
    calls=[]
    original=module.model.decoder.render
    def traced(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.model.decoder, "render", traced)
    actual=module.render(state,times,11)
    assert len(calls)==1
    actual_grad=torch.autograd.grad(actual.square().mean(), (state.c,state.d))
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    for a,b in zip(actual_grad,expected_grad):
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-4)
    assert torch.equal(state.last_time,times[:,11])
