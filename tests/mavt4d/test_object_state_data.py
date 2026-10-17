"""Metric factorial scenes preserve intervention and observation boundaries."""
import pytest
import torch
from torch.utils.data import DataLoader

from mavt4d.object_state.data import (
    FactorialSceneConfig, FactorialSceneDataset, CachedFactorialDataset, flatten_source_batch,
    observation_step, paired_variant_indices, sample_ray_targets, patch_instance_targets,
)


@pytest.fixture(scope="module")
def group():
    cfg = FactorialSceneConfig(frames=6, prefix_frames=4, views=2,
                               height=24, width=32, max_objects=3)
    return FactorialSceneDataset(2, cfg, split="train", base_seed=17)[0]


def test_factorial_group_is_deterministic_and_source_split_disjoint(group):
    cfg = FactorialSceneConfig(frames=6, prefix_frames=4, views=2,
                               height=24, width=32, max_objects=3)
    same = FactorialSceneDataset(2, cfg, split="train", base_seed=17)[0]
    other = FactorialSceneDataset(2, cfg, split="dev", base_seed=17)[0]
    assert torch.equal(group["targets"]["rgb"], same["targets"]["rgb"])
    assert not torch.equal(group["metadata"]["source_uid"], other["metadata"]["source_uid"])
    assert group["targets"]["rgb"].shape == (8, 6, 2, 3, 24, 32)
    assert group["observations"]["rgb"].shape == (8, 4, 1, 3, 24, 32)
    assert group["targets"]["rgb"].min() >= 0
    assert group["targets"]["rgb"].max() <= 1


def test_paired_factors_have_known_counterfactuals(group):
    t = group["targets"]
    # Variant bits: shape, appearance, motion; every pair keeps canonical IDs.
    assert torch.equal(t["depth"][0], t["depth"][2])
    assert torch.equal(t["object_poses"][0], t["object_poses"][2])
    assert not torch.equal(t["rgb"][0], t["rgb"][2])
    assert torch.equal(t["object_poses"][0], t["object_poses"][4])
    assert not torch.equal(t["object_sizes"][0], t["object_sizes"][4])
    # Opposite velocities pass through exactly the same pose at cutoff.
    assert torch.equal(t["object_poses"][0, 3], t["object_poses"][1, 3])
    assert torch.equal(t["rgb"][0, 3], t["rgb"][1, 3])
    assert not torch.equal(t["rgb"][0, 0], t["rgb"][1, 0])
    assert not torch.equal(t["rgb"][0, 5], t["rgb"][1, 5])
    assert torch.allclose(t["object_velocity"][0], -t["object_velocity"][1])


def test_observations_have_no_future_or_oracle_targets(group):
    assert set(group["observations"]) == {"rgb", "valid", "intrinsics", "camera_c2w", "times", "geometry_valid"}
    batch = flatten_source_batch(torch.utils.data.default_collate([group]))
    before = batch["observations"]["rgb"].clone()
    batch["targets"]["rgb"].fill_(float("nan"))
    assert torch.equal(before, batch["observations"]["rgb"])
    with pytest.raises(ValueError, match="prefix"):
        observation_step(batch, 4)


def test_ray_targets_are_calibrated_z_to_range(group):
    batch = flatten_source_batch(torch.utils.data.default_collate([group]))
    rays = sample_ray_targets(batch, 3, 64, generator=torch.Generator().manual_seed(2))
    assert rays["rgb"].shape == (8, 64, 3)
    assert torch.allclose(rays["query"].directions.norm(dim=-1), torch.ones(8, 64), atol=1e-6)
    assert (rays["range"][rays["depth_valid"]] >= rays["depth"][rays["depth_valid"]]).all()
    # The z-range distinction matters away from the optical centre.
    assert (rays["range"] - rays["depth"])[rays["depth_valid"]].max() > .01
    assert torch.equal(rays["query"].time, batch["targets"]["times"][:, 3])
    p = rays["query"].origins + rays["query"].directions * rays["range"][..., None]
    ground = rays["depth_valid"] & (rays["instance"] == 0)
    assert ground.any()
    assert p[..., 2][ground].abs().max() < 1e-5


def test_ray_geometry_stays_float32_inside_autocast(group):
    batch = flatten_source_batch(torch.utils.data.default_collate([group]))
    reference = sample_ray_targets(batch, 3, 24, generator=torch.Generator().manual_seed(9))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = sample_ray_targets(batch, 3, 24, generator=torch.Generator().manual_seed(9))
    assert actual["query"].directions.dtype == torch.float32
    assert torch.allclose(reference["query"].directions, actual["query"].directions, atol=1e-6)
    assert torch.allclose(reference["range"], actual["range"], atol=1e-6)


def test_pair_indices_preserve_source_and_target_factors(group):
    batch = flatten_source_batch(torch.utils.data.default_collate([group]))
    pairs = paired_variant_indices(batch)
    meta = batch["metadata"]
    a, b, target = pairs["content"], pairs["dynamic"], pairs["target"]
    assert torch.equal(meta["source_uid"][a], meta["source_uid"][b])
    for field in ("shape_factor", "appearance_factor"):
        assert torch.equal(meta[field][a], meta[field][target])
    assert torch.equal(meta["motion_factor"][b], meta["motion_factor"][target])


def test_config_rejects_empty_prefix_or_no_forecast():
    with pytest.raises(ValueError):
        FactorialSceneConfig(prefix_frames=0)
    with pytest.raises(ValueError):
        FactorialSceneConfig(frames=6, prefix_frames=6)


def test_patch_supervision_uses_observed_endpoint_and_never_enters_observation(group):
    batch = flatten_source_batch(torch.utils.data.default_collate([group]))
    labels, valid = patch_instance_targets(batch, 3, patch_size=16)
    assert labels.shape == valid.shape == (8, 4)
    obs = observation_step(batch, 3)
    assert obs.images.shape == (8, 1, 2, 3, 24, 32)
    assert obs.depth is None
    expected = batch["targets"]["instance"][:, 3, 0][:, [8, 23]][:, :, [8, 24]].flatten(1)
    assert torch.equal(labels, expected)
    with pytest.raises(ValueError, match="prefix"):
        patch_instance_targets(batch, 5, patch_size=16)


def test_cache_publishes_once_and_reloads_without_generator(tmp_path, monkeypatch):
    import mavt4d.object_state.data as module
    cfg = FactorialSceneConfig(frames=4, prefix_frames=2, height=8, width=12, max_objects=1)
    dataset = CachedFactorialDataset(tmp_path, 2, cfg, split="dev")
    first = dataset[0]
    path = dataset.cache_path(0)
    before = path.stat().st_mtime_ns
    monkeypatch.setattr(module, "_render_source", lambda *args: pytest.fail("cache should not render"))
    second = dataset[0]
    assert path.stat().st_mtime_ns == before
    assert torch.equal(first["targets"]["rgb"], second["targets"]["rgb"])
    second["targets"]["rgb"].fill_(float("nan"))
    assert torch.isfinite(second["observations"]["rgb"]).all()
    assert not list(path.parent.glob("*.tmp"))


def test_cache_config_hash_and_source_split_are_distinct(tmp_path):
    a = CachedFactorialDataset(tmp_path, 1, FactorialSceneConfig(height=8, width=12))
    b = CachedFactorialDataset(tmp_path, 1, FactorialSceneConfig(height=16, width=12))
    c = CachedFactorialDataset(tmp_path, 1, FactorialSceneConfig(height=8, width=12), split="dev")
    assert len({a.cache_path(0), b.cache_path(0), c.cache_path(0)}) == 3


def test_cache_rejects_corruption_without_overwrite(tmp_path):
    cfg = FactorialSceneConfig(frames=4, prefix_frames=2, height=8, width=12, max_objects=1)
    dataset = CachedFactorialDataset(tmp_path, 1, cfg)
    dataset[0]
    path = dataset.cache_path(0)
    payload = torch.load(path, weights_only=True)
    payload["sample"]["targets"]["rgb"][0, 0, 0, 0, 0, 0] += .1
    torch.save(payload, path)
    before = path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="digest"):
        dataset[0]
    assert path.stat().st_mtime_ns == before
