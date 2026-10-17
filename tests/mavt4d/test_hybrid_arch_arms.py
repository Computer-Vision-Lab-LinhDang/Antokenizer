"""Architecture arms I/A/IA: version-selected variants, locks and dev memory controls."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

from mavt4d.hybrid.config import is_v3
from mavt4d.hybrid_loss_v3.config import LossV3TrialConfig
from mavt4d.hybrid_loss_v3.objectives import LossV3Module
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.objectives import full_observation_states

from test_hybrid_loss_v3_objectives import batch, config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/mavt4d"
VARIANTS = {
    "innovation": ("I", {"innovation": True, "appearance": False}),
    "appearance": ("A", {"innovation": False, "appearance": True}),
    "innovation_appearance": ("IA", {"innovation": True, "appearance": True}),
}
V3_ONLY = re.compile(
    r"(innovation_pool\.|appearance_writer\.|cells\.\d+\.innovation_|decoder\.appearance_keys$)"
)
NEW_DEV_KEYS = ("lastwrite", "a_core4", "dc_feedback_ratio", "E_R_aonly")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_arch_{name}", ROOT / f"scripts/mavt4d/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_base():
    from test_hybrid_pilot_objectives import tiny_config

    return replace(tiny_config(), source_count=12, dev_fraction=0.5, world_size=1,
                   global_source_groups=1, microbatch_candidates=(1,))


@pytest.fixture(scope="module")
def event_suite(tmp_path_factory):
    base = _tiny_base()
    data = tmp_path_factory.mktemp("arch-arms") / "data"
    _load_script("build_event_suite").build_event_suite(data, base.event_config, range(12))
    return base, data


def _deadlines():
    return SimpleNamespace(requested=False,
                           work_at=datetime.now(timezone.utc) + timedelta(hours=1))


def test_existing_trial_hashes_unchanged():
    expected = {
        "hybrid_loss_v3_trial": ("3de93fc72fc6adec2f48e08bf697a88bef72416923398adb43ca2b048c586339", "B"),
        "hybrid_loss_v3_vram_trial": ("b40a4a9e4333232df64127c4eeda349da5784fc22865ea3d373d5f937ebf4eb0", "B"),
        "hybrid_loss_v3_vram_trial_v2": ("2b955fb68df3a71f495a4b90b7a3c822d9d2fd122181e5eb7fb7fe1f9e1a1ce4", "B"),
        "hybrid_loss_v3_upper_bound": ("00fb2c2e546af3baf69e353c3ff02dd0923232489fc121ed0c56dc4716a25b21", "U"),
        "hybrid_loss_v3_upper_bound_tiny": ("f2674efc43d3b3b2a7f5b1330c9d5b91c423c4a3f3cb2b7d126544ca7102abe5", "U"),
    }
    for name, (digest, arm) in expected.items():
        cfg = LossV3TrialConfig.load(CONFIGS / f"{name}.json")
        assert (cfg.sha256, cfg.arm, cfg.architecture_variant) == (digest, arm, None)
    base = HybridPilotConfig.load(CONFIGS / "hybrid_synthetic_pilot_v2.json")
    assert base.config_sha256 == "ea21c9341be02f9a0f579f40935ebb3e1d339d188ea13128de725d18a92cc066"


def test_arm_versions_select_variants():
    trial = LossV3TrialConfig.load(CONFIGS / "hybrid_loss_v3_trial.json").to_dict()
    tiny_source = LossV3TrialConfig.load(CONFIGS / "hybrid_loss_v3_upper_bound_tiny.json").to_dict()
    for name, (arm, variant) in VARIANTS.items():
        main = LossV3TrialConfig.load(CONFIGS / f"hybrid_cd_arch_{name}.json")
        tiny = LossV3TrialConfig.load(CONFIGS / f"hybrid_cd_arch_{name}_tiny.json")
        slug = name.replace("_", "-")
        assert main.version == f"hybrid-cd-arch-{slug}-v1"
        assert tiny.version == f"hybrid-cd-arch-{slug}-tiny-v1"
        assert {**main.to_dict(), "version": None} == {**trial, "version": None, "log_every": 1}
        assert {**tiny.to_dict(), "version": None} == {**tiny_source, "version": None}
        for cfg in (main, tiny):
            assert (cfg.arm, cfg.architecture_variant, cfg.core_head_weight) == (arm, variant, 0.0)
            assert cfg.capacity_policy == "fastest_safe"
            cfg.validate_production()
        assert tiny.overfit and not main.overfit
        with pytest.raises(ValueError):
            replace(tiny, effective_source_groups=8)
    with pytest.raises(ValueError, match="unsupported"):
        LossV3TrialConfig(version="hybrid-cd-arch-bogus-v1")


def test_arm_shared_weights_match_b_at_init():
    torch.manual_seed(17)
    plain = LossV3Module(config())
    after_b = torch.get_rng_state()
    reference = plain.model.state_dict()
    arch = config().architecture
    for _, variant in VARIANTS.values():
        torch.manual_seed(17)
        module = LossV3Module(config(), architecture_variant=variant)
        assert torch.equal(torch.get_rng_state(), after_b)
        resolved = module.model.config
        assert is_v3(resolved) and resolved.innovation == variant["innovation"]
        assert resolved.appearance_tokens == (arch.state_tokens if variant["appearance"] else 0)
        assert resolved.innovation_grid == (16 // arch.patch_size, 24 // arch.patch_size)
        weights = module.model.state_dict()
        assert set(reference) < set(weights)
        assert all(torch.equal(reference[key], weights[key]) for key in reference)
        assert all(V3_ONLY.match(key) for key in set(weights) - set(reference))
        # The temporary v2 model is not kept: the module holds only the v3 model.
        assert set(module.state_dict()) == {f"model.{key}" for key in weights}


def test_full_observation_states_defaults_unchanged():
    torch.manual_seed(5)
    model = LossV3Module(config()).model.eval()
    group = batch()
    calls = []
    original = model.forward_sequence

    def record(state, steps, **kwargs):
        calls.append((state, kwargs))
        return original(state, steps, **kwargs)

    model.forward_sequence = record
    with torch.no_grad():
        default = full_observation_states(model, [group])
        initial = model.initial_state(8, last_time=group.frame_times[:, :, 0].min(dim=1).values)
        explicit = full_observation_states(model, [group], initial=initial, diagnostics=None)
        diagnostics = {}
        observed = full_observation_states(model, [group], diagnostics=diagnostics)
    assert calls[0][1] == {"method": "associative", "diagnostics": None}
    assert torch.equal(calls[0][0].c, initial.c) and torch.equal(calls[0][0].last_time, initial.last_time)
    assert calls[1][0] is initial and calls[2][1]["diagnostics"] is diagnostics
    assert set(diagnostics) == {"stage1", "stage2"}
    for first, *others in zip(default, explicit, observed):
        for other in others:
            assert all(torch.equal(getattr(first, name), getattr(other, name))
                       for name in first.STATE_FIELDS)


def test_last_write_only_zeroes_cd_before_last_step():
    from mavt4d.hybrid_loss_v3.evaluation import last_write_only_states

    torch.manual_seed(5)
    model = LossV3Module(config(), architecture_variant=VARIANTS["innovation_appearance"][1]).model.eval()
    group = batch()
    calls = []
    original = model.forward_sequence

    def record(state, steps, **kwargs):
        result = original(state, steps, **kwargs)
        calls.append((state, len(steps), result))
        return result

    model.forward_sequence = record
    with torch.no_grad():
        result = last_write_only_states(model, group)
        normal = full_observation_states(model, [group])[-1]
    (_, prefix_steps, prefix), (resumed, last_steps, last) = calls[:2]
    assert (prefix_steps, last_steps) == (5, 1)
    assert not resumed.c.any() and not resumed.d.any()
    before = prefix[-1]
    for name in ("k", "v", "token_valid", "last_time", "step_count", "slot_times",
                 "prev_tokens", "prev_seen", "appearance", "appearance_valid"):
        assert torch.equal(getattr(resumed, name), getattr(before, name)), name
    assert before.c.any() and before.appearance.any()
    assert result is last[-1] and result.c.any()
    assert not torch.allclose(result.c, normal.c)


def test_arm_resume_phase_check(tmp_path):
    from mavt4d.hybrid_loss_v3 import runner
    from mavt4d.hybrid_pilot.runner import DistributedContext, save_checkpoint

    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    source = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(source.parameters())
    lock = {key: f"value-{key}" for key in (
        "schema", "arm", "base_config", "base_config_sha256", "objective",
        "dataset_sha256", "manifest_sha256", "partition_sha256", "world_size",
        "spatial_bf16", "recurrent_decoder_dtype", "held_out_evaluated")}
    lock["arm"] = "I"
    common = {"selection_metric": "loss-v3-common-dev-v1", "step": 3, "next_step": 4}
    checkpoints = {
        "i": {**lock, **common, "phase": "loss-v3-I"},
        "b": {**lock, **common, "arm": "B", "phase": "loss-v3-B"},
        "i_with_b_phase": {**lock, **common, "phase": "loss-v3-B"},
    }
    for name, metadata in checkpoints.items():
        save_checkpoint(tmp_path / f"{name}.pt", source, optimizer, metadata, context)

    def start(name):
        target = torch.nn.Linear(2, 1)
        return runner.initialize_from_checkpoint(
            tmp_path / f"{name}.pt", target, torch.optim.AdamW(target.parameters()), context, lock)

    assert start("i")["parent_step"] == 3
    with pytest.raises(ValueError, match="checkpoint arm mismatch"):
        start("b")
    with pytest.raises(ValueError, match="checkpoint phase mismatch"):
        start("i_with_b_phase")


def test_b_lock_unchanged_and_variant_lock_for_new_arms(tmp_path, monkeypatch):
    from mavt4d.hybrid_loss_v3 import runner

    root = tmp_path / "data"
    root.mkdir()
    for filename in ("summary.json", "manifest_v1.jsonl"):
        (root / filename).write_text("{}\n")
    initialized = []

    class Dataset:
        train_groups, dev_groups, val_groups = ("a", "b"), ("c",), ("d",)

        def __init__(self, *args):
            self.root = root

        def load_group(self, name, device):
            return 1.0

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.cfg = SimpleNamespace(grad_clip=1e6)

        def forward(self, batches, seeds, labels, phase):
            loss = (self.weight - 1) ** 2
            return loss, {"objective": loss.detach()}

    class Logger:
        run, run_id = None, "id"

        def __init__(self, output, **kwargs):
            self.output = output

        def log(self, *args, **kwargs):
            pass

        def checkpoint(self, *args, **kwargs):
            pass

        def finish(self, **kwargs):
            pass

    def initialize(*args, **kwargs):
        initialized.append(kwargs)
        return Module()

    monkeypatch.setattr(runner, "HybridPilotDataset", Dataset)
    monkeypatch.setattr(runner, "labels_for_dataset", lambda _: {g: [] for g in "abcd"})
    monkeypatch.setattr(runner, "initialize", initialize)
    monkeypatch.setattr(runner, "RunLogger", Logger)
    monkeypatch.setattr(runner, "development", lambda *a, **k: {"complete": True, "S_dev": 0.2})
    monkeypatch.setattr(runner, "aligned_panel", lambda *a: ({}, torch.zeros(3, 2, 2)))
    base = _tiny_base()
    one = dict(steps=1, effective_source_groups=1, microbatch_candidates=(1,))
    trials = {"B": LossV3TrialConfig(**one)}
    for name, (arm, _) in VARIANTS.items():
        trials[arm] = replace(LossV3TrialConfig.load(CONFIGS / f"hybrid_cd_arch_{name}.json"), **one)
    locks = {}
    for arm, cfg in trials.items():
        now = datetime.now(timezone.utc)
        runner.run_trial(base, cfg, root, tmp_path / arm, optimization_at=now + timedelta(hours=1),
                         work_at=now + timedelta(hours=2), git_revision="d" * 40, device="cpu",
                         allow_cpu=True, spatial_bf16=False)
        locks[arm] = json.loads((tmp_path / arm / "run_lock.json").read_text())
        last = torch.load(tmp_path / arm / "last.pt", weights_only=True)
        assert last["metadata"]["phase"] == f"loss-v3-{arm}"
    assert [kwargs["architecture_variant"] for kwargs in initialized] == [
        None, *(variant for _, variant in VARIANTS.values())]
    assert set(locks["B"]["objective"]) == {
        "weights", "endpoints", "pixel_threshold", "witness_pixels", "witness_frames",
        "c_pair_scale", "c_std_floor", "order_margin"}
    volatile = ("arm", "objective", "trial_config", "trial_config_sha256", "optimization_at", "work_at")
    for name, (arm, variant) in VARIANTS.items():
        lock = locks[arm]
        assert lock["arm"] == arm
        assert lock["objective"] == {**locks["B"]["objective"], "architecture_variant": {
            "innovation": variant["innovation"],
            "appearance_tokens": base.architecture.state_tokens if variant["appearance"] else 0}}
        assert {k: v for k, v in lock.items() if k not in volatile} == {
            k: v for k, v in locks["B"].items() if k not in volatile}
    # The capacity probe builds the same arm as the trial.
    now = datetime.now(timezone.utc)
    runner.capacity_probe(base, trials["IA"], root, tmp_path / "capacity",
                          optimization_at=now + timedelta(hours=1), work_at=now + timedelta(hours=2),
                          device="cpu", spatial_bf16=False, allow_cpu=True)
    assert initialized[-1]["architecture_variant"] == VARIANTS["innovation_appearance"][1]
    # Exact resume (last.pt): an I run continues; a B run cannot become an I run.
    two = replace(trials["I"], steps=2)

    def resume(source, name):
        now = datetime.now(timezone.utc)
        return runner.run_trial(base, two, root, tmp_path / name, optimization_at=now + timedelta(hours=1),
                                work_at=now + timedelta(hours=2), git_revision="d" * 40, device="cpu",
                                allow_cpu=True, spatial_bf16=False, resume=tmp_path / source)

    assert resume("I", "I-resumed")["step"] == 2
    resumed = json.loads((tmp_path / "I-resumed" / "run_lock.json").read_text())
    assert resumed["objective"] == locks["I"]["objective"]
    with pytest.raises(ValueError, match="checkpoint arm mismatch"):
        resume("B", "B-into-I")


def test_b_and_u_dev_keys_unchanged(event_suite):
    from mavt4d.hybrid_loss_v3.evaluation import development, labels_for_dataset
    from mavt4d.hybrid_pilot.data import HybridPilotDataset

    torch.set_num_threads(1)
    base, data = event_suite
    dataset = HybridPilotDataset(data, base)
    labels = labels_for_dataset(dataset)
    for weight in (0.0, 1.0):
        torch.manual_seed(17)
        scores = development(LossV3Module(base, core_head_weight=weight), dataset, labels,
                             _deadlines(), torch.device("cpu"))
        assert scores["complete"] and "c_core4" in scores and "E_R" in scores
        assert not [key for key in scores if any(part in key for part in NEW_DEV_KEYS)]


def test_v3_dev_reports_controls(event_suite, tmp_path, monkeypatch):
    from mavt4d.hybrid_loss_v3 import runner

    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    base, data = event_suite
    cfg = replace(LossV3TrialConfig.load(CONFIGS / "hybrid_cd_arch_innovation_appearance_tiny.json"),
                  steps=2)
    now = datetime.now(timezone.utc)
    output = tmp_path / "ia"
    result = runner.run_trial(base, cfg, data, output, optimization_at=now + timedelta(hours=1),
                              work_at=now + timedelta(hours=2), git_revision="e" * 40,
                              device="cpu", allow_cpu=True, spatial_bf16=False)
    assert result["step"] == 2 and result["last_dev_step"] == 2
    lock = json.loads((output / "run_lock.json").read_text())
    assert lock["arm"] == "IA"
    assert lock["objective"]["architecture_variant"] == {
        "innovation": True, "appearance_tokens": base.architecture.state_tokens}
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["metadata"]["phase"] == "loss-v3-IA"
    assert any(key.startswith("model.innovation_pool.") for key in checkpoint["module"])
    assert any(key.startswith("model.appearance_writer.") for key in checkpoint["module"])
    dev = json.loads((output / "dev-latest.json").read_text())
    assert dev["step"] == 2 and dev["metrics"]["complete"]
    metrics = dev["metrics"]
    for name in ("c_core4_lastwrite", "a_core4"):
        for suffix in ("", "_train", "_ci_low", "_ci_high"):
            assert 0 <= metrics[f"{name}{suffix}"] <= 1
    for name in ("dc_feedback_ratio_stage1", "dc_feedback_ratio_stage2", "E_R_aonly"):
        assert math.isfinite(metrics[name]) and metrics[name] >= 0
    # Core-4 only: no direction/configuration probes for the control features.
    assert {key for key in metrics if "lastwrite" in key or key.startswith("a_")} == {
        f"{name}{suffix}" for name in ("c_core4_lastwrite", "a_core4")
        for suffix in ("", "_train", "_ci_low", "_ci_high")}
    journal = [json.loads(row) for row in (output / "progress.jsonl").read_text().splitlines()]
    for key in ("dev/c_core4_lastwrite", "dev/a_core4", "dev/dc_feedback_ratio_stage1",
                "dev/dc_feedback_ratio_stage2", "dev/E_R_aonly"):
        assert sum(key in row for row in journal) == 2, key


def test_probe_script_runs_on_tiny_config(tmp_path):
    torch.set_num_threads(1)
    output = tmp_path / "probe.json"
    _load_script("probe_variant_contrast").main([
        "--base-config", str(CONFIGS / "hybrid_synthetic_pilot_v2.json"),
        "--sources", "1", "--seed", "17", "--tiny", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["seed"] == 17 and len(report["sources"]) == 1
    from mavt4d.data.event_synthetic import make_event_fixture

    full = HybridPilotConfig.load(CONFIGS / "hybrid_synthetic_pilot_v2.json")
    source = make_event_fixture(full.event_config, report["sources"][0], "clockwise", "compact")
    assert source.evaluation.split != "val"  # never a held-out source
    assert (report["H"], report["W"]) == (16, 24)
    arms = report["arms"]
    assert list(arms) == ["B", "I", "A", "IA"]
    for arm, values in arms.items():
        assert len(values["c1"]) == len(values["c2"]) == 6
        assert all(math.isfinite(x) and x >= 0 for x in (*values["c1"], *values["c2"], values["d_final"]))
        assert ("a_final" in values) == ("A" in arm)
        assert ("innovation_rms" in values) == ("I" in arm)
        assert len(values["per_source"]) == 1
    assert len(arms["IA"]["innovation_rms"]) == 6
    # Shared weights: the stateless observer gives identical evidence in every arm.
    assert len({values["evidence_final"] for values in arms.values()}) == 1
