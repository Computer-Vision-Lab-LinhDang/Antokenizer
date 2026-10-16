"""Separate long-run budget/selection logic never changes the locked pilot."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
import torch

from mavt4d.hybrid_diagnostic.longrun import LongRunConfig, extend_budget, save_distributed_best
from mavt4d.hybrid_pilot.runner import DistributedContext


def test_budget_extensions_depend_on_time_only_and_respect_cap():
    cfg = LongRunConfig(arm="write_gate")
    assert cfg.target_steps == 10000 and cfg.max_steps == 20000
    assert extend_budget(cfg, 10000, remaining_seconds=500, seconds_per_update=.2) == 10000
    assert extend_budget(cfg, 10000, remaining_seconds=1200, seconds_per_update=.2) == 15000
    assert extend_budget(cfg, 15000, remaining_seconds=1200, seconds_per_update=.2) == 20000
    assert extend_budget(cfg, 20000, remaining_seconds=1200, seconds_per_update=.2) == 20000
    with pytest.raises(ValueError):
        LongRunConfig(arm="unreviewed")


def _save_worker(rank, rendezvous, output):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(100 + rank)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        context = DistributedContext(rank, 2, rank, torch.device("cpu"))
        best = save_distributed_best(Path(output), model, optimizer, score=.9,
            best_score=float("inf"), metadata={"step": 3, "next_step": 4}, context=context)
        assert best == .9
        assert save_distributed_best(Path(output), model, optimizer, score=1.,
            best_score=best, metadata={"step": 4, "next_step": 5}, context=context) == best
    finally:
        dist.destroy_process_group()


def test_two_rank_best_checkpoint_retains_both_rngs_and_one_file(tmp_path):
    import torch.multiprocessing as mp
    output = tmp_path / "output"
    output.mkdir()
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    mp.spawn(_save_worker, args=(rendezvous, str(output)), nprocs=2, join=True)
    assert {p.name for p in output.iterdir()} == {"best.pt"}
    payload = torch.load(output / "best.pt", weights_only=True)
    assert len(payload["rank_rng"]) == 2
    assert not torch.equal(payload["rank_rng"][0]["torch"], payload["rank_rng"][1]["torch"])
    assert payload["metadata"]["step"] == 3
    assert payload["metadata"]["selection_metric"] == "fixed_dev_weighted_loss"


def _training_worker(rank, rendezvous, dataset, output, raw_config):
    import os
    import torch.distributed as dist
    from mavt4d.hybrid_pilot.config import HybridPilotConfig
    from mavt4d.hybrid_diagnostic.longrun import run_long
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    now = datetime.now(timezone.utc)
    result = run_long(HybridPilotConfig.from_dict(raw_config),
        LongRunConfig(arm="write_gate", target_steps=2, max_steps=2, dev_every=1, image_every=2),
        dataset, output, optimization_at=now+timedelta(hours=1), work_at=now+timedelta(hours=2),
        git_revision="b"*40, spatial_bf16=False, allow_cpu=True)
    assert result["status"] == "complete" and result["step"] == 2


def test_actual_two_rank_training_uses_full_dev_and_best_only(tmp_path):
    import importlib.util
    from dataclasses import replace
    import torch.multiprocessing as mp
    from test_hybrid_pilot_objectives import tiny_config
    cfg = replace(tiny_config(), source_count=12, dev_fraction=.5)
    exporter_path = Path(__file__).resolve().parents[2] / "scripts/mavt4d/build_event_suite.py"
    spec = importlib.util.spec_from_file_location("_longrun_export", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    data = tmp_path / "data"
    exporter.build_event_suite(data, cfg.event_config, range(12))
    output = tmp_path / "run"
    mp.spawn(_training_worker, args=("file://"+str(tmp_path / "training-rendezvous"),
        str(data), str(output), cfg.to_dict()), nprocs=2, join=True)
    result = json.loads((output / "result.json").read_text())
    assert result["step"] == 2 and result["last_dev_step"] == 2
    assert result["held_out_evaluated"] is False
    assert {p.name for p in output.glob("*.pt")} == {"best.pt"}
    lock = json.loads((output / "run_lock.json").read_text())
    assert len(lock["dev_source_groups"]) > 2
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert len(checkpoint["rank_rng"]) == 2
    assert checkpoint["metadata"]["source_cursor"] == result["best_step"]*2
