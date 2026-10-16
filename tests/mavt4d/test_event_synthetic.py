"""Event fixtures must require history, not a core-dependent colour or last image."""
from __future__ import annotations

import hashlib
from dataclasses import fields
from itertools import product

import pytest
import torch

from mavt.data.splits import assign
from mavt4d.contracts import ObservationStep
from mavt4d.data.event_synthetic import (
    EVENT_CONFIGURATIONS,
    EVENT_CORES,
    EventSceneConfig,
    EventSyntheticDataset,
    make_event_fixture,
)


CFG = EventSceneConfig(T=12, H=32, W=48)


@pytest.fixture(scope="module")
def crossed():
    return {
        (core, configuration): make_event_fixture(CFG, 17, core, configuration)
        for core, configuration in product(EVENT_CORES, EVENT_CONFIGURATIONS)
    }


def _frame_multiset(images):
    return sorted(hashlib.sha256(frame.numpy().tobytes()).hexdigest() for frame in images[0])


def _equal_observations(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        for name in ("images", "observed", "frame_times", "intrinsics", "cam_to_world"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_crossing_is_complete_and_grouped_before_rendering():
    assert set(EVENT_CORES) == {
        "left_then_right", "right_then_left", "clockwise", "counterclockwise",
    }
    assert EVENT_CONFIGURATIONS == ("compact", "wide")
    seeds = tuple(range(20))
    train = EventSyntheticDataset(CFG, seeds, split="train")
    val = EventSyntheticDataset(CFG, seeds, split="val")
    keys = train.keys + val.keys
    assert len(keys) == len(seeds) * 8
    assert len(train) and len(val)
    for seed in seeds:
        group = [key for key in keys if key.source_seed == seed]
        assert {(key.core, key.configuration) for key in group} == set(product(EVENT_CORES, EVENT_CONFIGURATIONS))
        assert len({key.group_key for key in group}) == len({key.split for key in group}) == 1
        assert group[0].split == assign(group[0].group_key, val_frac=0.2, test_frac=0.0)
    assert not {key.group_key for key in train.keys} & {key.group_key for key in val.keys}


def test_group_split_is_stable_under_expansion_and_order():
    def mapping(seeds):
        return {
            (key.source_seed, key.core, key.configuration): (key.group_key, key.split)
            for split in ("train", "val")
            for key in EventSyntheticDataset(CFG, seeds, split).keys
        }

    original = mapping((1, 4, 7, 10))
    expanded = mapping((10, 99, 7, 4, 1, 150))
    assert all(expanded[key] == value for key, value in original.items())


@pytest.mark.parametrize("configuration", EVENT_CONFIGURATIONS)
def test_same_last_tubelet_and_input_metadata_across_all_cores(crossed, configuration):
    reference = crossed[EVENT_CORES[0], configuration]
    for core in EVENT_CORES[1:]:
        fixture = crossed[core, configuration]
        assert torch.equal(reference.scene.images[:, -2:], fixture.scene.images[:, -2:])
        assert torch.equal(reference.evaluation.positions[-2:], fixture.evaluation.positions[-2:])
        assert not torch.equal(reference.scene.images[:, :-2], fixture.scene.images[:, :-2])
        for name in ("frame_times", "intrinsics", "cam_to_world"):
            assert torch.equal(getattr(reference.scene, name), getattr(fixture.scene, name))
        _equal_observations(reference.observation_prefix(CFG.T)[-1:], fixture.observation_prefix(CFG.T)[-1:])


@pytest.mark.parametrize("configuration", EVENT_CONFIGURATIONS)
@pytest.mark.parametrize("forward,reverse", [
    ("left_then_right", "right_then_left"), ("clockwise", "counterclockwise"),
])
def test_reversed_history_has_exact_same_frame_multiset(crossed, configuration, forward, reverse):
    a, b = crossed[forward, configuration], crossed[reverse, configuration]
    assert torch.equal(a.evaluation.positions[:-2].flip(0), b.evaluation.positions[:-2])
    assert torch.equal(a.scene.images[:, :-2].flip(1), b.scene.images[:, :-2])
    assert _frame_multiset(a.scene.images) == _frame_multiset(b.scene.images)
    assert a.evaluation.core != b.evaluation.core


def test_event_label_is_readable_from_motion_and_not_from_appearance(crossed):
    for configuration in EVENT_CONFIGURATIONS:
        for core in ("left_then_right", "right_then_left"):
            positions = crossed[core, configuration].evaluation.positions
            initial_side = positions[1, 1] - positions[0, 1]
            # The fixed camera looks along +x: world +y appears to image left.
            assert initial_side > 0 if core == "left_then_right" else initial_side < 0
            assert torch.allclose(positions[:, 0], positions[0, 0].expand(CFG.T))
        for core in ("clockwise", "counterclockwise"):
            positions = crossed[core, configuration].evaluation.positions[:-2, :2]
            area = (positions[:-1, 0] * positions[1:, 1] - positions[1:, 0] * positions[:-1, 1]).sum()
            # Rotation labels use an overhead world-coordinate view (+z looking down).
            assert area < 0 if core == "clockwise" else area > 0


def test_configuration_changes_geometry_without_changing_core(crossed):
    for core in EVENT_CORES:
        a, b = crossed[core, "compact"], crossed[core, "wide"]
        assert a.evaluation.core == b.evaluation.core
        assert not torch.equal(a.evaluation.positions[-1], b.evaluation.positions[-1])
        assert not torch.equal(a.scene.images[:, -1], b.scene.images[:, -1])
        assert a.evaluation.group_key == b.evaluation.group_key


def test_geometry_is_exact_and_target_remains_visible(crossed):
    for fixture in crossed.values():
        scene, target = fixture.scene, fixture.evaluation
        scene.validate()
        assert scene.images.shape == (1, CFG.T, 3, CFG.H, CFG.W)
        assert scene.images.dtype == torch.float32
        assert torch.isfinite(scene.images).all() and scene.images.abs().max() <= 1
        assert scene.frame_times.dtype == torch.float64
        assert torch.isfinite(scene.frame_times).all()
        assert (scene.frame_times[:, 1:] > scene.frame_times[:, :-1]).all()
        assert scene.tier == "B" and scene.lidar_points is None
        assert scene.boxes is None and scene.patch_flow is None
        assert len(target.boxes) == CFG.T
        assert target.target_instance_id == 1
        for t, boxes in enumerate(target.boxes):
            assert boxes.instance_id.tolist() == [1]
            assert torch.equal(boxes.center[0], target.positions[t])
        visible = target.gt.instance == target.target_instance_id
        assert (visible.flatten(2).sum(-1) >= 4).all()
        assert torch.isfinite(target.gt.depth).all()
        assert (target.gt.depth[visible] > 0).all()
        assert torch.equal(target.gt.depth[:, -2], target.gt.depth[:, -1])


def test_repeatability_ignores_global_rng_and_source_changes_nuisance(crossed):
    fixture = crossed["left_then_right", "compact"]
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(9813)
        torch.rand(31)
        again = make_event_fixture(CFG, 17, "left_then_right", "compact")
    finally:
        torch.random.set_rng_state(state)
    assert torch.equal(fixture.scene.images, again.scene.images)
    assert torch.equal(fixture.evaluation.gt.depth, again.evaluation.gt.depth)
    other = make_event_fixture(CFG, 18, "left_then_right", "compact")
    assert torch.equal(fixture.evaluation.positions, other.evaluation.positions)
    assert not torch.equal(fixture.scene.images, other.scene.images)
    assert not torch.equal(fixture.scene.images[:, -1], other.scene.images[:, -1])


@pytest.mark.parametrize("source_seed", (0, 4))
def test_high_source_seed_bits_change_rendered_nuisance(source_seed):
    """Seeds that collide in torch's CPU RNG must not create split leakage."""
    low = make_event_fixture(CFG, source_seed, "left_then_right", "compact")
    high = make_event_fixture(CFG, source_seed + 2**32, "left_then_right", "compact")
    assert low.evaluation.group_key != high.evaluation.group_key
    if source_seed == 4:
        assert low.evaluation.split == "val" and high.evaluation.split == "train"
    assert torch.equal(low.evaluation.positions, high.evaluation.positions)
    assert not torch.equal(low.scene.images, high.scene.images)
    assert not torch.equal(low.scene.images[:, -1], high.scene.images[:, -1])
    assert not torch.equal(low.scene.cam_to_world, high.scene.cam_to_world)


def test_safe_prefix_hides_targets_ids_future_and_unavailable_pixels(crossed):
    fixture = crossed["clockwise", "compact"]
    mask = torch.ones(1, CFG.T, CFG.H, CFG.W, dtype=torch.bool)
    mask[:, :, :CFG.H // 2] = False
    prefix = fixture.observation_prefix(5, availability=mask)
    assert [step.images.shape[1] for step in prefix] == [2, 2, 1]
    assert all(isinstance(step, ObservationStep) for step in prefix)
    allowed = {"images", "observed", "frame_times", "intrinsics", "cam_to_world"}
    for step in prefix:
        assert {field.name for field in fields(step)} == allowed
        assert (step.images.masked_select(~step.observed[:, :, None].expand_as(step.images)) == 0).all()
        assert step.frame_times.max() <= fixture.scene.frame_times[0, 4]
        for value in (step.images, step.observed, step.frame_times, step.cam_to_world):
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    before = tuple(step.images.clone() for step in prefix)
    future_before = fixture.scene.images[:, 5:].clone()
    hidden_before = fixture.scene.images[:, :5, :, :CFG.H // 2].clone()
    try:
        fixture.scene.images[:, 5:] = 0.75
        fixture.scene.images[:, :5, :, :CFG.H // 2] = -0.75
        after = fixture.observation_prefix(5, availability=mask)
        _equal_observations(prefix, after)
        assert all(torch.equal(a, step.images) for a, step in zip(before, prefix))
    finally:
        fixture.scene.images[:, 5:] = future_before
        fixture.scene.images[:, :5, :, :CFG.H // 2] = hidden_before


def test_dataset_safe_inputs_match_fixture():
    dataset = EventSyntheticDataset(CFG, tuple(range(20)), "val")
    fixture = dataset[0]
    assert fixture.evaluation.split == "val"
    _equal_observations(dataset.model_inputs(0, 4), fixture.observation_prefix(4))


@pytest.mark.parametrize("kwargs", [
    {"T": 6}, {"T": 9}, {"H": 0}, {"W": 0}, {"dt": 0},
    {"dt": float("nan")}, {"val_fraction": 1.0}, {"version": "unknown"},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EventSceneConfig(**kwargs)


def test_invalid_requests_are_rejected(crossed):
    with pytest.raises(ValueError):
        make_event_fixture(CFG, 17, "appearance_red", "compact")
    with pytest.raises(ValueError):
        make_event_fixture(CFG, 17, "clockwise", "unknown")
    with pytest.raises(ValueError):
        EventSyntheticDataset(CFG, (0, 1), "test")
    with pytest.raises(ValueError):
        EventSyntheticDataset(CFG, (0, 0), "train")
    fixture = crossed["clockwise", "compact"]
    for stop in (0, CFG.T + 1):
        with pytest.raises(ValueError):
            fixture.observation_prefix(stop)
    with pytest.raises(ValueError):
        fixture.observation_prefix(2, frames_per_step=0)
    with pytest.raises(ValueError):
        fixture.observation_prefix(2, availability=torch.ones(1, 1, CFG.H, CFG.W))
