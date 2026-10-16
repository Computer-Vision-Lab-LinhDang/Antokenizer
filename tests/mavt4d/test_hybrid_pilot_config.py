"""The hybrid training lock is explicit and separate from the v1 experiment."""
from dataclasses import replace
from pathlib import Path

import pytest

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.config import HybridPilotConfig


def test_lock_roundtrip_architecture_and_batch_budget(tmp_path):
    cfg = HybridPilotConfig()
    assert cfg.architecture == HybridConfig()
    assert (cfg.T, cfg.H, cfg.W) == (12, 128, 192)
    assert (cfg.overfit_steps, cfg.train_steps, cfg.seed) == (200, 1200, 17)
    assert cfg.global_source_groups * cfg.batch_size == 256
    assert [cfg.gradient_accumulation(m) for m in (1, 2, 4)] == [4, 2, 1]
    path = tmp_path / "lock.json"
    path.write_text(cfg.canonical_json())
    assert HybridPilotConfig.load(path) == cfg
    assert replace(cfg, learning_rate=.0002).config_sha256 != cfg.config_sha256
    published = Path(__file__).resolve().parents[2] / "configs/mavt4d/hybrid_synthetic_pilot_v2.json"
    assert HybridPilotConfig.load(published) == cfg


@pytest.mark.parametrize("change", [{"query_count": 256}, {"unused_option": 1}])
def test_unknown_or_obsolete_knobs_are_rejected(change):
    with pytest.raises(ValueError, match="keys"):
        HybridPilotConfig.from_dict({**HybridPilotConfig().to_dict(), **change})


@pytest.mark.parametrize("change", [{"world_size": 3}, {"global_source_groups": 31},
                                    {"learning_rate": float("nan")}, {"seed": True}])
def test_invalid_training_budgets_fail_closed(change):
    with pytest.raises(ValueError):
        replace(HybridPilotConfig(), **change)
