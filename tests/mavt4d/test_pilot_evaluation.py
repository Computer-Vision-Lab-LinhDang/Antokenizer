"""Frozen linear probes must measure order without learning from held-out sources."""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from mavt4d.contracts import CDStateSpec
from mavt4d.data.event_synthetic import EVENT_CONFIGURATIONS, EVENT_CORES
from mavt4d.pilot.evaluation import (
    FeatureRecord, FeatureTable, evaluate_probe_suite, extract_features,
    fit_ridge_probe, source_bootstrap, summarize_runs, reconstruction_mask,
    rgb_error_summary, evaluate_reconstruction,
)
from mavt4d.reference.model import ReferenceCDModel


def config():
    return SimpleNamespace(probe_ridge=1.0, bootstrap=100, bootstrap_seed=31415)


def table(prefix="train", groups=3, *, order_information=True):
    records, c, d, static = [], [], [], []
    for source in range(groups):
        for core in range(4):
            for configuration in range(2):
                records.append(FeatureRecord(f"{prefix}-{source}-{core}-{configuration}",
                                             f"{prefix}-{source}", prefix, core, configuration))
                c.append(torch.eye(4)[core] if order_information else torch.eye(4)[core // 2])
                d.append(torch.eye(2)[configuration])
                static.append(torch.tensor([float(configuration), float(core // 2)]))
    return FeatureTable(tuple(records), {
        "c": torch.stack(c), "d": torch.stack(d),
        "last_c": torch.stack(static), "last_d": torch.stack(static),
        "orderless": torch.stack(static),
    })


def test_ridge_uses_train_statistics_mean_gram_and_unpenalized_intercept():
    x = torch.tensor([[1., 7.], [3., 7.], [5., 7.], [7., 7.]])
    y = torch.tensor([0, 0, 1, 1])
    probe = fit_ridge_probe(x, y, classes=2, ridge=1.0)
    mean = x.double().mean(0)
    scale = x.double().std(0, correction=0).clamp_min(1e-4)
    z = (x.double() - mean) / scale
    target = torch.eye(2, dtype=torch.float64)[y]
    weights = torch.linalg.solve(z.T @ z / 4 + torch.eye(2, dtype=torch.float64),
                                 z.T @ (target - target.mean(0)) / 4)
    torch.testing.assert_close(probe.mean, mean)
    torch.testing.assert_close(probe.scale, scale)
    torch.testing.assert_close(probe.weights, weights)
    torch.testing.assert_close(probe.intercept, target.mean(0))
    assert probe.weights.dtype == torch.float64
    before = probe.mean.clone(), probe.scale.clone()
    probe.predict(torch.tensor([[1000., -1000.]]))
    torch.testing.assert_close(probe.mean, before[0], rtol=0, atol=0)
    torch.testing.assert_close(probe.scale, before[1], rtol=0, atol=0)


@pytest.mark.parametrize("ridge", [0., -1., float("nan")])
def test_ridge_rejects_invalid_regularization(ridge):
    with pytest.raises(ValueError, match="ridge"):
        fit_ridge_probe(torch.eye(2), torch.tensor([0, 1]), classes=2, ridge=ridge)


def test_oracle_features_solve_core_and_configuration_with_correct_denominators():
    result = evaluate_probe_suite(table(), table("val", 2), config())
    assert result["metrics"]["c_direction"] == 1.0
    assert result["metrics"]["c_core"] == 1.0
    assert result["metrics"]["d_configuration"] == 1.0
    assert result["metrics"]["c_configuration"] == .5
    assert result["metrics"]["d_direction"] == .5
    assert result["metrics"]["d_core"] == .25
    assert len(result["predictions"]["c_core"]) == 16
    assert set(result["group_scores"]["c_direction"]) == {"val-0", "val-1"}
    assert all(score == 1 for score in result["group_scores"]["d_configuration"].values())
    for core in range(4):
        assert result["metrics"][f"d_configuration_core_{core}"] == 1
    json.dumps(result, allow_nan=False)


def test_orderless_and_suffix_controls_are_chance_for_each_reversed_pair():
    result = evaluate_probe_suite(table(order_information=False),
                                  table("val", order_information=False), config())
    assert result["metrics"]["c_core"] == .5  # Family is visible without order.
    for name in ("c_direction", "last_c_direction", "orderless_direction"):
        assert result["metrics"][name] == .5
    for family in ("lateral", "loop"):
        assert result["metrics"][f"c_direction_{family}"] == .5


def test_per_core_configuration_scores_use_the_same_global_configuration_probe():
    def conflicting(prefix):
        data = table(prefix)
        features = dict(data.features)
        features["d"] = torch.tensor([[float(row.configuration ^ (row.core >= 2))]
                                       for row in data.records])
        return FeatureTable(data.records, features)
    result = evaluate_probe_suite(conflicting("train"), conflicting("val"), config())
    assert result["metrics"]["d_configuration"] == .5
    # Refitting a different probe within each core would conceal the conflict.
    for core in range(4):
        assert result["metrics"][f"d_configuration_core_{core}"] == .5


def test_probe_suite_rejects_shared_sources_and_incomplete_crossings():
    with pytest.raises(ValueError, match="overlap|disjoint"):
        evaluate_probe_suite(table(), table(), config())
    partial = table("val")
    partial = FeatureTable(partial.records[:-1], {k: v[:-1] for k, v in partial.features.items()})
    with pytest.raises(ValueError, match="crossing|eight|8"):
        evaluate_probe_suite(table(), partial, config())


def test_feature_table_rejects_duplicate_ids_and_nonfinite_features():
    source = table()
    with pytest.raises(ValueError, match="duplicate"):
        FeatureTable((source.records[0], source.records[0]),
                     {name: value[:2] for name, value in source.features.items()})
    corrupt = dict(source.features)
    corrupt["c"] = corrupt["c"].clone()
    corrupt["c"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FeatureTable(source.records, corrupt)


def test_bootstrap_resamples_source_units_and_is_deterministic():
    scores = {"a": 0., "b": 1.}
    actual = source_bootstrap(scores, samples=100, seed=4)
    generator = torch.Generator().manual_seed(4)
    indices = torch.randint(2, (100, 2), generator=generator)
    expected = torch.tensor([0., 1.], dtype=torch.float64)[indices].mean(1)
    assert actual == source_bootstrap(scores, samples=100, seed=4)
    assert actual["mean"] == .5
    assert actual["n_groups"] == 2
    assert actual["low"] == float(torch.quantile(expected, .025))
    assert actual["high"] == float(torch.quantile(expected, .975))


def test_run_summary_pairs_the_same_groups_and_reports_sample_seed_sd():
    rows = []
    for cell, offset in (("coupled", .25), ("gru", 0.)):
        for seed, score in ((17, .25), (29, .5), (43, .75)):
            rows.append({"cell": cell, "seed": seed, "result": {
                "metrics": {"c_direction": score + offset},
                "group_scores": {"c_direction": {"a": score + offset,
                                                   "b": score + offset}},
            }})
    summary = summarize_runs(rows, config())
    assert summary["cells"]["gru"]["c_direction"]["mean"] == .5
    assert summary["cells"]["gru"]["c_direction"]["seed_sd"] == .25
    comparison = summary["paired"]["coupled-minus-gru"]["c_direction"]
    assert comparison["mean"] == comparison["low"] == comparison["high"] == .25
    assert comparison["n_groups"] == 2


def test_summary_rejects_missing_seed_or_source_pair_instead_of_silent_intersection():
    result = {"metrics": {"c_direction": .5},
              "group_scores": {"c_direction": {"a": .5}}}
    rows = [{"cell": "coupled", "seed": 17, "result": result},
            {"cell": "gru", "seed": 29, "result": result}]
    with pytest.raises(ValueError, match="seed"):
        summarize_runs(rows, config())


@pytest.mark.parametrize("family", ["clean", "wide", "tube", "snapshot", "never_seen"])
def test_reconstruction_masks_measure_missing_endpoint_and_share_source_support(family):
    batch = SimpleNamespace(observed=torch.ones(8, 1, 12, 16, 24, dtype=torch.bool))
    mask, endpoint = reconstruction_mask(batch, family, torch.Generator().manual_seed(2718))
    again, again_endpoint = reconstruction_mask(batch, family, torch.Generator().manual_seed(2718))
    assert torch.equal(mask, again) and endpoint == again_endpoint
    assert torch.equal(mask, mask[:1].expand_as(mask))
    assert endpoint == (7 if family == "snapshot" else 11)
    if family == "clean":
        assert mask.all()
    else:
        assert (~mask[:, :, endpoint]).any()
    if family == "snapshot":
        assert mask[:, :, :4].all() and not mask[:, :, 4:8].any()
    if family == "never_seen":
        assert torch.equal(mask[:, :, :1].expand_as(mask), mask)


def test_rgb_error_summary_uses_range_two_unclipped_predictions_and_rgb_counts():
    prediction = torch.tensor([[[2., 2., 2.], [0., 0., 0.]]])
    target = torch.ones_like(prediction)
    mask = torch.tensor([[True, False]])
    result = rgb_error_summary(prediction, target, mask)
    for region in ("observed", "missing"):
        assert result[region]["rgb_count"] == 3
        assert result[region]["l1"] == result[region]["mse"] == 1
        assert result[region]["psnr"] == pytest.approx(6.020599913279624)
    assert result["all"]["rgb_count"] == 6
    clean = rgb_error_summary(target, target, torch.ones_like(mask))
    assert clean["missing"]["rgb_count"] == 0
    assert clean["missing"]["psnr"] is None
    assert clean["observed"]["perfect"] is True
    json.dumps(clean, allow_nan=False)


def test_reconstruction_aggregates_sse_before_psnr_and_uses_fixed_first_sources():
    class ZeroDecoderModel(torch.nn.Module):
        spec = SimpleNamespace(frames_per_step=2)

        def initial_state(self, batch_size, last_time):
            return SimpleNamespace(batch_size=batch_size)

        def forward_sequence(self, state, steps, method):
            assert method == "sequential"
            for observations in steps:
                for item in observations:
                    hidden = ~item.observed[:, :, None].expand_as(item.images)
                    assert torch.count_nonzero(item.images[hidden]) == 0
            return (SimpleNamespace(batch_size=state.batch_size, last_time=steps[-1][0].end_time),)

        def decode(self, state, query):
            assert torch.equal(query.times, state.last_time.expand_as(query.times))
            return torch.zeros(state.batch_size, len(query.times), 3)

    seen = []

    def load(group, device):
        seen.append(group)
        intensity = {"a": .5, "b": 1., "c": -1.}[group]
        return SimpleNamespace(
            images=torch.full((2, 1, 12, 3, 4, 6), intensity),
            observed=torch.ones(2, 1, 12, 4, 6, dtype=torch.bool),
            frame_times=torch.arange(12, dtype=torch.float64)[None, None].repeat(2, 1, 1),
            intrinsics=torch.eye(3, dtype=torch.float64)[None, None].repeat(2, 1, 1, 1),
            cam_to_world=torch.eye(4, dtype=torch.float64)[None, None, None].repeat(2, 1, 12, 1, 1),
        )

    cfg = SimpleNamespace(eval_reconstruction_groups=2, eval_mask_seed=2718)
    dataset = SimpleNamespace(load_group=load, records_for_group=lambda _: ({"split": "val"},) * 2)
    model = ZeroDecoderModel().train()
    result = evaluate_reconstruction(model, dataset, ("c", "b", "a"), cfg, "cpu")
    assert model.training
    assert result["source_groups"] == ["a", "b"]
    assert set(seen) == {"a", "b"}
    assert result["families"]["snapshot"]["endpoint_frame"] == 7
    for values in result["families"].values():
        assert values["all"]["mse"] == .625
        assert values["all"]["l1"] == .75
        assert values["all"]["rgb_count"] == 2 * 2 * 4 * 6 * 3
        assert values["all"]["psnr"] == pytest.approx(10 * math.log10(4 / .625))
    assert result["families"]["snapshot"]["observed"]["rgb_count"] == 0
    assert result["families"]["clean"]["missing"]["rgb_count"] == 0


def test_extract_features_uses_exact_single_frame_orderless_and_reset_suffix(tmp_path):
    generator = torch.Generator().manual_seed(37)
    path = torch.rand(4, 3, 4, 6, generator=generator) * 2 - 1
    end = path[:1].repeat(2, 1, 1, 1)
    forward = torch.cat((path, end))
    reverse = torch.cat((path.flip(0), end))
    images = torch.stack((forward, reverse))[:, None]
    batch = SimpleNamespace(
        images=images, observed=torch.ones(2, 1, 6, 4, 6, dtype=torch.bool),
        frame_times=torch.arange(6, dtype=torch.float64)[None, None].repeat(2, 1, 1) / 12,
        intrinsics=torch.eye(3, dtype=torch.float64)[None, None].repeat(2, 1, 1, 1),
        cam_to_world=torch.eye(4, dtype=torch.float64)[None, None, None].repeat(2, 1, 6, 1, 1),
    )
    records = [dict(record_id=f"clip-{i}", group_key="source", split="val") for i in range(2)]
    labels = [{**row, "core": EVENT_CORES[i], "configuration": EVENT_CONFIGURATIONS[0]}
              for i, row in enumerate(records)]
    (tmp_path / "eval_targets_v1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    digest = hashlib.sha256((tmp_path / "eval_targets_v1.jsonl").read_bytes()).hexdigest()
    (tmp_path / "summary.json").write_text(json.dumps({"sha256": {"evaluation": digest}}))
    dataset = SimpleNamespace(root=tmp_path, records_for_group=lambda _: tuple(records),
                              load_group=lambda _, device: batch)
    torch.manual_seed(1)
    model = ReferenceCDModel(CDStateSpec(c_tokens=2, d_tokens=2, token_dim=4),
                              embed_dim=8, patch_size=2, modulation_dim=2)
    model.train()
    assert torch.equal(batch.images[0, :, -2:], batch.images[1, :, -2:])
    features = extract_features(model, dataset, ("source",), config(), "cpu")
    assert model.training  # Evaluation does not permanently change training mode.
    for name in ("orderless", "last_c", "last_d"):
        difference = (features.features[name][0] - features.features[name][1]).abs().max().item()
        assert torch.equal(features.features[name][0], features.features[name][1]), (name, difference)
    assert features.features["orderless"].shape == (2, 32)
    assert features.features["c"].shape == (2, 8)
    assert not features.features["c"].requires_grad
    assert [record.core for record in features.records] == [0, 1]
    assert not torch.equal(features.features["c"][0], features.features["c"][1])
    (tmp_path / "eval_targets_v1.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum|sha256"):
        extract_features(model, dataset, ("source",), config(), "cpu")
