from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.natural_trial.model import NaturalTrialModule, TrialConfig, ordered_rgb_target, make_mask
from mavt4d.natural_trial.runner import save_best_checkpoint
from mavt4d.natural_trial.runner import run_trial
from mavt4d.natural_trial.prepare import sha256, split_for_source
from mavt4d.hybrid_pilot.runner import DistributedContext, load_checkpoint


def test_order_target_distinguishes_reversal_with_identical_endpoints():
    video = torch.zeros(2, 12, 3, 8, 8)
    video[0, 2] = .8
    video[0, 7] = -.8
    video[1, 2] = -.8
    video[1, 7] = .8
    target = ordered_rgb_target(video)
    assert torch.equal(video[0, -2:], video[1, -2:])
    assert torch.dot(target[0], target[1]) < 0


def test_mask_is_seeded_anisotropic_and_hidden_rgb_has_no_effect():
    cfg = TrialConfig(architecture=HybridConfig.tiny(), height=8, width=8)
    model = NaturalTrialModule(cfg).eval()
    rgb = torch.randn(1, 12, 3, 8, 8).clamp(-1, 1)
    times = torch.arange(12, dtype=torch.float64)[None] / 12
    mask = make_mask(12, 8, 8, seed=11, family="tube")
    assert torch.equal(mask, make_mask(12, 8, 8, seed=11, family="tube"))
    changed = rgb.clone()
    changed[~mask[None, :, None].expand_as(changed)] = 1e6
    with torch.no_grad():
        a = model.encode(rgb, times, mask[None])[-1]
        b = model.encode(changed, times, mask[None])[-1]
    assert torch.equal(a.c, b.c)
    assert torch.equal(a.d, b.d)


def test_trial_loss_and_backward_are_finite_without_labels_or_calibration():
    cfg = TrialConfig(architecture=HybridConfig.tiny(), height=8, width=8)
    model = NaturalTrialModule(cfg)
    rgb = torch.randn(2, 12, 3, 8, 8).clamp(-1, 1)
    times = torch.arange(12, dtype=torch.float64).repeat(2, 1) / 12
    times[1] *= 1.1
    loss, metrics = model(rgb, times, [21, 22])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert model.history_head.weight.grad is not None
    assert model.model.decoder.tile_head.weight.grad is not None


def test_mixed_mask_loss_and_gradient_do_not_depend_on_microbatch_packing():
    cfg = TrialConfig(architecture=HybridConfig.tiny(), height=8, width=8)
    module = NaturalTrialModule(cfg)
    seeds = [next(s for s in range(100) if bool(make_mask(12, 8, 8, seed=s).all()) == clean)
             for clean in (True, False)]
    rgb = torch.randn(2, 12, 3, 8, 8).clamp(-1, 1)
    times = torch.arange(12, dtype=torch.float64).repeat(2, 1) / 12
    combined, _ = module(rgb, times, seeds)
    combined.backward()
    gradient = module.model.decoder.tile_head.weight.grad.clone()
    module.zero_grad(set_to_none=True)
    individual = sum(module(rgb[i:i + 1], times[i:i + 1], seeds[i:i + 1])[0] for i in range(2)) / 2
    individual.backward()
    torch.testing.assert_close(combined, individual, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradient, module.model.decoder.tile_head.weight.grad, atol=1e-6, rtol=2e-5)


def test_best_checkpoint_atomically_replaces_single_snapshot_and_restores(tmp_path):
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(module.parameters())
    metadata = {"step": 2, "next_step": 3, "world_size": 1, "dev_loss": .8}
    save_best_checkpoint(tmp_path / "best.pt", module, optimizer, metadata, context)
    with torch.no_grad():
        module.weight.add_(1)
    metadata = {**metadata, "step": 5, "next_step": 6, "dev_loss": .5}
    save_best_checkpoint(tmp_path / "best.pt", module, optimizer, metadata, context)
    other = torch.nn.Linear(2, 1)
    other_optimizer = torch.optim.AdamW(other.parameters())
    restored = load_checkpoint(tmp_path / "best.pt", other, other_optimizer, context, expected={"world_size": 1})
    assert restored["step"] == 5
    assert torch.equal(other.weight, module.weight)
    assert [p.name for p in tmp_path.iterdir()] == ["best.pt"]


def test_cpu_trial_fresh_resume_and_best_only_checkpoint(tmp_path):
    cfg = TrialConfig(architecture=HybridConfig.tiny(), height=8, width=8,
                      global_clips=2, steps=3, dev_every=1, log_every=1)
    dataset = tmp_path / "data"
    dataset.mkdir()
    rows = []
    for split in ("train", "dev"):
        source = next(f"P01/P01_{i:03d}" for i in range(1, 200)
                      if split_for_source(f"P01/P01_{i:03d}") == split)
        path = dataset / f"{split}.npz"
        rgb = np.full((12, 3, 8, 8), 180, dtype=np.uint8)
        rgb[2] = 10
        rgb[7] = 230
        np.savez(path, images=rgb, frame_times=np.arange(12, dtype=np.float64) / 12)
        rows.append({"record_id": split, "source": source, "split": split,
                     "input": path.name, "sha256": sha256(path)})
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (dataset / "summary.json").write_text(json.dumps({"schema": "natural-epic-trial-v1",
        "frames": 12, "height": 8, "width": 8, "manifest_sha256": sha256(dataset / "manifest.jsonl"),
        "counts": {"train": 1, "dev": 1}}))
    now = datetime.now(timezone.utc)
    kwargs = dict(optimization_at=now + timedelta(minutes=5), work_at=now + timedelta(minutes=6),
                  git_revision="1" * 40, microbatch=1, cfg=cfg, allow_cpu=True)
    output = tmp_path / "run"
    partial = run_trial(dataset, output, max_steps=1, **kwargs)
    assert partial["status"] == "stopped"
    assert [p.name for p in output.glob("*.pt")] == ["best.pt"]
    complete = run_trial(dataset, output, resume=output / "best.pt", **kwargs)
    assert complete["status"] == "complete" and complete["step"] == 3
    assert [p.name for p in output.glob("*.pt")] == ["best.pt"]
    logs = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert all(a["step"] < b["step"] for a, b in zip(logs, logs[1:]))
    assert (output / "images" / "dev-step-000000.png").is_file()
