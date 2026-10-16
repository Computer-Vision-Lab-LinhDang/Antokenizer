"""Locked observation-only pilot: reproducibility, data isolation and objectives."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mavt4d.pilot.config import PilotConfig
from mavt4d.pilot.data import PilotDataset, TrainingBatch
from mavt4d.pilot.objectives import (
    PilotHeads, full_observation_states, make_training_mask, masked_mean,
    ordered_history_target, training_loss,
)
from mavt4d.reference.model import ReferenceCDModel

ROOT = Path(__file__).resolve().parents[2]


def tiny_config(**kwargs):
    return dataclasses.replace(PilotConfig(), H=16, W=24, c_tokens=2, d_tokens=2,
                               token_dim=4, embed_dim=8, modulation_dim=2,
                               temporal_pool=(2, 3), semantic_dim=4, query_count=12,
                               **kwargs)


def tiny_batch(batch_size=2):
    generator = torch.Generator().manual_seed(88)
    return TrainingBatch(
        images=torch.rand(batch_size, 1, 12, 3, 16, 24, generator=generator) * 2 - 1,
        observed=torch.ones(batch_size, 1, 12, 16, 24, dtype=torch.bool),
        frame_times=torch.arange(12, dtype=torch.float64)[None, None].repeat(batch_size, 1, 1) / 12,
        intrinsics=torch.tensor([[[12., 0., 12.], [0., 12., 8.], [0., 0., 1.]]]).repeat(batch_size, 1, 1, 1),
        cam_to_world=torch.eye(4).repeat(batch_size, 1, 12, 1, 1),
    )


def test_locked_defaults_and_hash_round_trip(tmp_path):
    cfg = PilotConfig()
    assert (cfg.source_count, cfg.T, cfg.H, cfg.W) == (128, 12, 32, 48)
    assert cfg.cells == ("coupled", "two_scan", "gru")
    assert cfg.seeds == (17, 29, 43)
    assert (cfg.eval_reconstruction_groups, cfg.eval_mask_seed) == (4, 2718)
    assert cfg.state_spec.num_state_values == 256
    assert cfg.config_sha256 == hashlib.sha256(cfg.canonical_json().encode()).hexdigest()
    path = tmp_path / "config.json"
    path.write_text(cfg.canonical_json())
    assert PilotConfig.load(path) == cfg
    assert PilotConfig.from_dict(cfg.to_dict()) == cfg
    assert dataclasses.replace(cfg, learning_rate=.002).config_sha256 != cfg.config_sha256
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train_steps = 1


@pytest.mark.parametrize("change", [{"unknown": 1}, {"train_steps": True},
    {"learning_rate": float("nan")}, {"cells": ["gru", "gru"]},
    {"mask_weights": [1, 0, 0, 0, 0]}, {"future_target_frames": [5, 7]},
    {"temporal_pairs": [[2, 99], [3, 6]]}, {"batch_size": 4},
    {"T": 9}, {"dev_fraction": 1.}, {"seed_start": -1}])
def test_config_rejects_invalid_or_unfrozen_contract(change):
    raw = PilotConfig().to_dict()
    raw.update(change)
    with pytest.raises((TypeError, ValueError)):
        PilotConfig.from_dict(raw)


def test_published_config_matches_defaults():
    assert PilotConfig.load(ROOT / "configs/mavt4d/synthetic_pilot_v1.json") == PilotConfig()


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("_pilot_fixture_export", ROOT / "scripts/mavt4d/build_event_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path_factory.mktemp("pilot-core") / "suite"
    cfg = dataclasses.replace(PilotConfig(), source_count=16)
    module.build_event_suite(root, cfg.event_config, range(16))
    return root, cfg


def test_dataset_group_partition_and_metadata_are_outside_batch(dataset_root):
    root, cfg = dataset_root
    dataset = PilotDataset(root, cfg)
    train, dev, val = map(set, (dataset.train_groups, dataset.dev_groups, dataset.val_groups))
    assert not train & dev and not train & val and not dev & val
    assert len(train | dev | val) == 16
    assert dataset.overfit_group == dataset.train_groups[0]
    assert train and dev and val
    other = PilotDataset(root, cfg)
    assert other.train_groups == dataset.train_groups
    for group in train | dev | val:
        rows = dataset.records_for_group(group)
        assert len(rows) == 8
        assert len({r["record_id"] for r in rows}) == 8
        assert {r["split"] for r in rows} == ({"val"} if group in val else {"train"})
        if group not in val:
            expected_dev = int(hashlib.sha256((cfg.dev_salt + group).encode()).hexdigest(), 16) / 2**256 < cfg.dev_fraction
            assert (group in dev) == expected_dev
    batch = dataset.load_group(dataset.overfit_group)
    assert {f.name for f in dataclasses.fields(batch)} == {
        "images", "observed", "frame_times", "intrinsics", "cam_to_world"}
    assert batch.images.shape == (8, 1, 12, 3, 32, 48)
    assert all(isinstance(getattr(batch, f.name), torch.Tensor) for f in dataclasses.fields(batch))
    for name in ("intrinsics", "cam_to_world", "frame_times"):
        value = getattr(batch, name)
        assert torch.equal(value, value[:1].expand_as(value))


def test_training_dataset_never_loads_evaluation_tensors(dataset_root, monkeypatch):
    root, cfg = dataset_root
    original_load = torch.load
    seen = []
    def guarded(path, *args, **kwargs):
        seen.append(str(path))
        assert "targets/" not in str(path)
        assert kwargs.get("weights_only") is True
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", guarded)
    dataset = PilotDataset(root, cfg)
    dataset.load_group(dataset.overfit_group)
    assert len(seen) == 8
    rows = dataset.records_for_group(dataset.overfit_group)
    with pytest.raises(TypeError):
        rows[0]["core"] = "leak"


def test_manifest_integrity_detects_modified_input(dataset_root, tmp_path):
    import shutil
    root, cfg = dataset_root
    copy = tmp_path / "corrupt"
    shutil.copytree(root, copy)
    dataset = PilotDataset(copy, cfg)
    row = dataset.records_for_group(dataset.overfit_group)[0]
    with (copy / row["input_path"]).open("ab") as stream:
        stream.write(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        dataset.load_group(dataset.overfit_group)


def test_dataset_rejects_render_configuration_mismatch(dataset_root):
    root, cfg = dataset_root
    with pytest.raises(ValueError, match="config"):
        PilotDataset(root, dataclasses.replace(cfg, H=16))


def test_batch_rejects_mismatched_calibration():
    batch = tiny_batch()
    bad = batch.cam_to_world.clone()
    bad[1, 0, :, 0, 3] += 1
    with pytest.raises(ValueError, match="shared"):
        dataclasses.replace(batch, cam_to_world=bad)


def test_mask_is_shared_across_variants_and_rgb_independent():
    batch, cfg = tiny_batch(8), tiny_config()
    changed = dataclasses.replace(batch, images=-batch.images)
    for seed in range(30):
        a = make_training_mask(batch, cfg, torch.Generator().manual_seed(seed))
        b = make_training_mask(changed, cfg, torch.Generator().manual_seed(seed))
        assert torch.equal(a, b)
        assert torch.equal(a, a[:1].expand_as(a))
        assert a.shape == batch.observed.shape and a.dtype == torch.bool
    clean = make_training_mask(batch, cfg, torch.Generator().manual_seed(5), phase="overfit")
    assert clean.all()


def test_ordered_target_changes_sign_when_pair_order_reverses():
    cfg, batch = tiny_config(), tiny_batch()
    target = ordered_history_target(batch.images, cfg)
    reversed_cfg = dataclasses.replace(cfg, temporal_pairs=tuple((b, a) for a, b in cfg.temporal_pairs))
    assert torch.allclose(ordered_history_target(batch.images, reversed_cfg), -target)
    assert target.shape == (2, 2 * 3 * 2 * 3)
    assert torch.isfinite(target).all()
    assert torch.equal(ordered_history_target(torch.zeros_like(batch.images), cfg), torch.zeros_like(target))


def test_masked_mean_normalizes_channels_and_empty_is_differentiable():
    values = torch.tensor([[[1., 3., 5.], [10., 10., 10.]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    assert masked_mean(values, mask) == 3
    empty = masked_mean(values, torch.zeros_like(mask))
    assert empty == 0
    empty.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_prefix_states_ignore_later_frames_and_hidden_pixels():
    cfg, batch = tiny_config(), tiny_batch()
    torch.manual_seed(10)
    model = ReferenceCDModel(cfg.state_spec, embed_dim=cfg.embed_dim, patch_size=cfg.patch_size,
                             modulation_dim=cfg.modulation_dim)
    availability = batch.observed.clone()
    availability[:, :, :6, :5, :5] = False
    mutated = batch.images.clone()
    mutated[:, :, 6:] = 700
    mutated[:, :, :6, :, :5, :5] = 900
    changed = dataclasses.replace(batch, images=mutated)
    first = full_observation_states(model, batch, availability=availability, stop_frame=6)
    second = full_observation_states(model, changed, availability=availability, stop_frame=6)
    assert len(first) == 3
    for a, b in zip(first, second):
        assert torch.equal(a.c, b.c) and torch.equal(a.d, b.d)


@pytest.mark.parametrize("cell_kind", ["coupled", "two_scan", "gru"])
def test_one_loss_is_finite_differentiable_and_label_free(cell_kind):
    cfg, batch = tiny_config(), tiny_batch()
    torch.manual_seed(10)
    model = ReferenceCDModel(cfg.state_spec, cell_kind, embed_dim=cfg.embed_dim,
                             patch_size=cfg.patch_size, modulation_dim=cfg.modulation_dim)
    heads = PilotHeads(cfg.state_spec, cfg)
    loss, metrics = training_loss(model, heads, batch, cfg, torch.Generator().manual_seed(16))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"rec", "missing", "future", "temporal", "invariance", "noncollapse", "zero_temporal_mse", "temporal_cosine"} <= metrics.keys()
    assert all(torch.isfinite(value) and value.ndim == 0 for value in metrics.values())
    loss.backward()
    for module in (model.observer, model.cell, model.decoder, heads.temporal, heads.semantic):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())


def test_temporal_target_uses_locked_first_minus_second_sign():
    cfg = tiny_config()
    images = torch.zeros(1, 1, 12, 3, 16, 24)
    images[:, :, 2] = 1.
    images[:, :, 7] = .25
    target = ordered_history_target(images, cfg)
    one_pair = 3 * 2 * 3
    # x2-x7 is positive .75, x3-x6 is zero. Per-clip RMS gives sqrt(2).
    expected = torch.cat((torch.full((1, one_pair), 2 ** .5), torch.zeros(1, one_pair)), dim=1)
    assert torch.allclose(target, expected)


def test_config_rejects_unequal_banks_required_by_coupled_reference():
    with pytest.raises(ValueError, match="equal"):
        dataclasses.replace(PilotConfig(), d_tokens=4)
