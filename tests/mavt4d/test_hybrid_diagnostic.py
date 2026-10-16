"""Mechanism diagnostics are isolated from the frozen pilot."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.objectives import HybridPilotModule
from mavt4d.hybrid.decoder import tile_grid_query
from mavt4d.hybrid_diagnostic.experiment import (
    DiagnosticConfig, apply_intervention, measure, save_best,
)
from test_hybrid_pilot_objectives import tiny_batch, tiny_config


def test_arms_are_frozen_and_change_only_declared_mechanism():
    base = DiagnosticConfig()
    assert base.steps == 200 and base.arm == "baseline"
    with pytest.raises(ValueError):
        DiagnosticConfig(arm="combined")
    torch.manual_seed(8)
    baseline = HybridPilotModule(tiny_config())
    original = {k: v.clone() for k, v in baseline.state_dict().items()}
    apply_intervention(baseline, base)
    assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in original.items())
    apply_intervention(baseline, replace(base, arm="write_gate"))
    changed = {k for k, v in original.items() if not torch.equal(v, baseline.state_dict()[k])}
    assert changed == {f"model.cells.{i}.write_head.2.bias" for i in range(2)}
    for cell in baseline.model.cells:
        assert torch.allclose(cell.write_head[-1].bias.reshape(-1, 7)[:, 6],
                              torch.full((cell.config.state_dim,), math.log(.1 / .9)))


def test_d_strength_preserves_weights_and_exact_zero_c_contract():
    torch.manual_seed(9)
    module = HybridPilotModule(tiny_config())
    original = {k: v.clone() for k, v in module.state_dict().items()}
    code = module.model.initial_state(8).code()
    code = replace(code, c=torch.randn_like(code.c), d=torch.randn_like(code.d))
    query = tile_grid_query(16, 24, tile_size=module.cfg.architecture.decoder_tile, time=0.)
    before = module.model.decoder.decode_tiles(code, query)
    apply_intervention(module, DiagnosticConfig(arm="d_strength"))
    after = module.model.decoder.decode_tiles(code, query)
    assert all(torch.equal(v, module.state_dict()[k]) for k, v in original.items())
    assert not torch.equal(before, after)
    zero = module.model.decoder.decode_tiles(replace(code, c=torch.zeros_like(code.c)), query)
    assert torch.count_nonzero(zero) == 0


def test_probe_gradient_diagnostics_do_not_change_parameters_or_populate_grads():
    torch.manual_seed(7)
    module = HybridPilotModule(tiny_config())
    before = {k: v.clone() for k, v in module.state_dict().items()}
    # Eight fixture variants: four cores crossed with two configurations.
    labels = [(core, config) for core in range(4) for config in range(2)]
    result, images = measure(module, tiny_batch(), labels, gradients=True)
    assert result["history/zero_mse"] > 0
    assert result["gradient/history/history_head/l2"] > 0
    assert result["gradient/reconstruction/decoder_d/l2"] > 0
    assert result["gradient/history/decoder_d/l2"] == 0
    assert result["state/c_rms"] > 0 and result["state/d_rms"] > 0
    assert "intervention/d_swap/moving_pixel_l1" in result
    assert images.ndim == 3 and images.shape[0] == 3
    assert all(math.isfinite(v) for v in result.values())
    assert all(p.grad is None for p in module.parameters())
    assert all(torch.equal(v, module.state_dict()[k]) for k, v in before.items())


def test_best_only_checkpoint_preserves_best_and_atomic_single_filename(tmp_path):
    module = HybridPilotModule(tiny_config())
    optimizer = torch.optim.AdamW(module.parameters())
    best = save_best(tmp_path, module, optimizer, score=.9, best_score=float("inf"),
                     metadata={"step": 3, "arm": "baseline"})
    assert best == .9 and [p.name for p in tmp_path.iterdir()] == ["best.pt"]
    old = (tmp_path / "best.pt").read_bytes()
    assert save_best(tmp_path, module, optimizer, score=.95, best_score=best,
                     metadata={"step": 4, "arm": "baseline"}) == .9
    assert (tmp_path / "best.pt").read_bytes() == old
    save_best(tmp_path, module, optimizer, score=.8, best_score=best,
              metadata={"step": 5, "arm": "baseline"})
    payload = torch.load(tmp_path / "best.pt", weights_only=True)
    assert payload["metadata"]["step"] == 5 and payload["metadata"]["selection_score"] == .8
    assert {p.name for p in tmp_path.iterdir()} == {"best.pt"}


def test_standalone_runner_saves_only_best_and_keeps_audits_separate(tmp_path):
    from mavt4d.hybrid_diagnostic.runner import run_diagnostic
    cfg = replace(tiny_config(), source_count=5, dev_fraction=0.)
    exporter_path = Path(__file__).resolve().parents[2] / "scripts/mavt4d/build_event_suite.py"
    spec = importlib.util.spec_from_file_location("_diagnostic_export", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    data = tmp_path / "data"
    exporter.build_event_suite(data, cfg.event_config, range(5))
    now = datetime.now(timezone.utc)
    output = tmp_path / "run"
    result = run_diagnostic(cfg, DiagnosticConfig(steps=2, probe_every=1, image_every=2),
        data, output, optimization_at=now+timedelta(hours=1), work_at=now+timedelta(hours=2),
        git_revision="a"*40, device="cpu", spatial_bf16=False, allow_cpu=True)
    assert result["status"] == "complete" and result["step"] == 2
    assert result["held_out_evaluated"] is False
    assert {p.name for p in output.glob("*.pt")} == {"best.pt"}
    assert (output / "images/diagnostic-000002.jpg").is_file()
    assert (output / "progress.jsonl").is_file()
    assert result["best_history_mse"] <= result["initial"]["history/mse"]
    payload = torch.load(output / "best.pt", weights_only=True)
    assert payload["metadata"]["next_step"] == result["best_step"]+1
    assert len(payload["rank_rng"]) == 1
