"""Hybrid sources retain the existing audited, observation-only data boundary."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.data import HybridPilotDataset


def test_loader_validates_new_render_and_separates_groups(tmp_path):
    cfg = replace(HybridPilotConfig(), architecture=HybridConfig.tiny(),
                  H=16, W=24, source_count=5, dev_fraction=0.)
    exporter_path = Path(__file__).resolve().parents[2] / "scripts/mavt4d/build_event_suite.py"
    spec = importlib.util.spec_from_file_location("_hybrid_fixture_export", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    root = tmp_path / "new-render"
    exporter.build_event_suite(root, cfg.event_config, range(5))
    dataset = HybridPilotDataset(root, cfg)
    assert len(dataset.train_groups) == 4 and len(dataset.val_groups) == 1
    assert not set(dataset.train_groups) & set(dataset.val_groups)
    batch = dataset.load_group(dataset.overfit_group)
    assert batch.images.shape == (8, 1, 12, 3, 16, 24)
    assert set(batch.__slots__) == {"images", "observed", "frame_times", "intrinsics", "cam_to_world"}
    original = batch.images.clone()
    batch.images.zero_()
    assert torch.equal(dataset.load_group(dataset.overfit_group).images, original)
    with pytest.raises(ValueError, match="config"):
        HybridPilotDataset(root, replace(cfg, H=32))
