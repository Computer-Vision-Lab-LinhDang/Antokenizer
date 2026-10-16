"""The fixture exporter must preserve auditability without leaking targets into inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
INPUT_KEYS = {"images", "observed", "frame_times", "intrinsics", "cam_to_world"}


def _builder():
    spec = importlib.util.spec_from_file_location(
        "_build_event_suite_test", ROOT / "scripts" / "mavt4d" / "build_event_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    builder = _builder()
    root = tmp_path_factory.mktemp("event-export") / "event-cd-v1"
    cfg = builder.EventSceneConfig(T=12, H=32, W=48)
    summary = builder.build_event_suite(root, cfg, (7, 17))
    return root, summary


def test_export_crossing_grouping_and_checksums(exported):
    root, summary = exported
    rows = _rows(root / "manifest_v1.jsonl")
    targets = _rows(root / "eval_targets_v1.jsonl")
    assert len(rows) == len(targets) == 16
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert {row["record_id"] for row in rows} == {row["record_id"] for row in targets}
    by_seed = defaultdict(list)
    for target in targets:
        by_seed[target["source_seed"]].append(target)
    for variants in by_seed.values():
        assert {(row["core"], row["configuration"]) for row in variants} == {
            (core, configuration)
            for core in ("left_then_right", "right_then_left", "clockwise", "counterclockwise")
            for configuration in ("compact", "wide")
        }
        assert len({row["group_key"] for row in variants}) == 1
        assert len({row["split"] for row in variants}) == 1
    samples = Counter(row["split"] for row in rows)
    sources = Counter(variants[0]["split"] for variants in by_seed.values())
    assert summary["counts"]["samples"] == {split: samples[split] for split in ("train", "val")}
    assert summary["counts"]["sources"] == {split: sources[split] for split in ("train", "val")}
    assert summary["source_seeds"] == [7, 17]
    assert summary["version"] == "event-cd-v1"
    assert summary["sha256"]["manifest"] == _sha256(root / "manifest_v1.jsonl")
    assert summary["sha256"]["evaluation"] == _sha256(root / "eval_targets_v1.jsonl")
    assert len(summary["sha256"]["inputs"]) == len(rows)
    for row in rows:
        assert summary["sha256"]["inputs"][row["input_path"]] == _sha256(root / row["input_path"])
        assert "core" not in row and "configuration" not in row
        assert not any(label in row["input_path"] for label in ("left", "right", "clockwise", "compact", "wide"))
    assert json.loads((root / "summary.json").read_text()) == summary


def test_model_tensor_files_have_only_observation_whitelist(exported):
    root, _ = exported
    for row in _rows(root / "manifest_v1.jsonl"):
        payload = torch.load(root / row["input_path"], weights_only=True)
        assert set(payload) == INPUT_KEYS
        assert all(isinstance(value, torch.Tensor) for value in payload.values())
        assert payload["images"].shape == (1, 12, 3, 32, 48)
        assert payload["observed"].shape == (1, 12, 32, 48)
        assert payload["observed"].dtype == torch.bool and payload["observed"].all()
        assert payload["frame_times"].shape == (1, 12)
        assert payload["intrinsics"].shape == (1, 3, 3)
        assert payload["cam_to_world"].shape == (1, 12, 4, 4)


def test_evaluation_geometry_is_stored_separately(exported):
    root, summary = exported
    for row in _rows(root / "eval_targets_v1.jsonl"):
        assert row["target_path"].startswith("targets/")
        target = torch.load(root / row["target_path"], weights_only=True)
        assert set(target) == {"positions", "boxes", "gt"}
        assert target["positions"].shape == (12, 3)
        assert len(target["boxes"]) == 12
        assert set(target["gt"]) == {"world_xyz", "depth", "instance", "valid"}
        visible = target["gt"]["instance"] == row["target_instance_id"]
        assert (visible.flatten(2).sum(-1) >= 4).all()
        assert summary["sha256"]["targets"][row["target_path"]] == _sha256(root / row["target_path"])


def test_export_contains_observed_audits_and_contact_sheet(exported):
    from PIL import Image

    root, summary = exported
    assert summary["audits"] == {
        "identical_final_two": True,
        "different_histories": True,
        "reversal_frame_multiset": True,
        "target_visible": True,
    }
    with Image.open(root / "contact_sheet.png") as preview:
        assert preview.format == "PNG"
        assert preview.width >= 12 * 48 and preview.height >= 4 * 32
    assert (root / "README.md").is_file()


def test_export_is_deterministic_across_output_directories(tmp_path):
    builder = _builder()
    cfg = builder.EventSceneConfig(T=12, H=32, W=48)
    first, second = tmp_path / "first", tmp_path / "second"
    a = builder.build_event_suite(first, cfg, (17, 7))
    b = builder.build_event_suite(second, cfg, (7, 17))
    assert a == b
    assert (first / "manifest_v1.jsonl").read_bytes() == (second / "manifest_v1.jsonl").read_bytes()
    assert (first / "eval_targets_v1.jsonl").read_bytes() == (second / "eval_targets_v1.jsonl").read_bytes()
    assert (first / "contact_sheet.png").read_bytes() == (second / "contact_sheet.png").read_bytes()


@pytest.mark.parametrize("change", [{"H": 48}, {"W": 64}, {"T": 14}, {"dt": 0.2}])
def test_render_configuration_changes_record_identity_without_resplitting(tmp_path, change):
    builder = _builder()
    fields = {"T": 12, "H": 32, "W": 48}
    original, changed = tmp_path / "original", tmp_path / "changed"
    a = builder.build_event_suite(original, builder.EventSceneConfig(**fields), (17,))
    b = builder.build_event_suite(changed, builder.EventSceneConfig(**{**fields, **change}), (17,))
    before = _rows(original / "manifest_v1.jsonl")
    after = _rows(changed / "manifest_v1.jsonl")
    assert a["sha256"]["manifest"] != b["sha256"]["manifest"]
    assert {row["record_id"] for row in before}.isdisjoint(row["record_id"] for row in after)
    assert {(row["group_key"], row["split"]) for row in before} == {
        (row["group_key"], row["split"]) for row in after
    }


def test_existing_output_is_never_modified(tmp_path):
    builder = _builder()
    cfg = builder.EventSceneConfig(T=12, H=32, W=48)
    out = tmp_path / "already-exists"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("do not touch")
    with pytest.raises(FileExistsError):
        builder.build_event_suite(out, cfg, (17,))
    assert sentinel.read_text() == "do not touch"
    assert set(out.iterdir()) == {sentinel}


@pytest.mark.parametrize("seeds", [(), (7, 7), (True,), (1.5,), ("7",)])
def test_invalid_seed_list_fails_before_creating_output(tmp_path, seeds):
    builder = _builder()
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        builder.build_event_suite(out, builder.EventSceneConfig(T=12, H=32, W=48), seeds)
    assert not out.exists()


def test_fixture_audit_failure_aborts_export(tmp_path, monkeypatch):
    builder = _builder()
    real_make = builder.make_event_fixture

    def broken_fixture(cfg, source_seed, core, configuration):
        fixture = real_make(cfg, source_seed, core, configuration)
        if core == "clockwise":
            fixture.scene.images[:, -1] = 0.5
        return fixture

    monkeypatch.setattr(builder, "make_event_fixture", broken_fixture)
    out = tmp_path / "broken"
    with pytest.raises(ValueError, match="final.two|suffix"):
        builder.build_event_suite(out, builder.EventSceneConfig(T=12, H=32, W=48), (17,))
    assert not (out / "summary.json").exists()


def test_cli_rejects_invalid_source_count_without_writing(tmp_path):
    builder = _builder()
    out = tmp_path / "invalid"
    with pytest.raises((SystemExit, ValueError)):
        builder.main(["--output", str(out), "--source-count", "0"])
    assert not out.exists()
