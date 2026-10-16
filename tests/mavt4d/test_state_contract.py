"""The exported C/D state is the complete persistent sequence memory.

These are interface tests only. Recurrent restart/scan equivalence belongs to
the later cell implementation, which this contract does not claim to verify.
"""

import io
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery, ObservationStep


def _spec(**overrides):
    return CDStateSpec(**{"c_tokens": 2, "d_tokens": 3, "token_dim": 4, **overrides})


def _state(dtype=torch.float32, requires_grad=False):
    return CDState(
        torch.arange(16, dtype=dtype).reshape(2, 2, 4).requires_grad_(requires_grad),
        torch.arange(24, dtype=dtype).reshape(2, 3, 4).requires_grad_(requires_grad),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        spec=_spec(),
    )


def _observation(**overrides):
    values = {
        "images": torch.arange(72, dtype=torch.float32).reshape(2, 2, 3, 2, 3),
        "observed": torch.ones(2, 2, 2, 3, dtype=torch.bool),
        "frame_times": torch.tensor([[0.0, 0.5], [0.1, 0.6]], dtype=torch.float64),
    }
    values.update(overrides)
    return ObservationStep(**values)


def _query(**overrides):
    values = {
        "coordinates": torch.tensor([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]]),
        "view_ids": torch.tensor([0, 1], dtype=torch.int64),
        "times": torch.tensor([0.5, 0.6], dtype=torch.float64),
    }
    values.update(overrides)
    return DecoderQuery(**values)


def test_fixed_state_budget_includes_declared_timestamp_metadata():
    assert CDStateSpec().num_state_values == 4096
    state = _state()
    assert state.num_state_values == 40  # Both samples, with no hidden cache.
    assert state.state_nbytes == 40 * 4
    assert state.metadata_nbytes == 2 * 8
    assert state.total_nbytes == 176
    assert _state(torch.float16).total_nbytes == 40 * 2 + 2 * 8
    assert _spec(reference_views=2).num_state_values == 20  # Explicit fixed budget.


def test_retained_memory_counts_oversized_tensor_views_and_shared_storage_once():
    backing = torch.arange(128, dtype=torch.float32)
    state = CDState(
        backing[:16].reshape(2, 2, 4),
        backing[16:40].reshape(2, 3, 4),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        spec=_spec(),
    )
    assert state.total_nbytes == 176  # Logical state and timestamp payload.
    assert state.retained_storage_nbytes == 128 * 4 + 2 * 8
    assert state.detach().retained_storage_nbytes == state.retained_storage_nbytes
    snapshot = state.snapshot()
    assert snapshot.retained_storage_nbytes == snapshot.total_nbytes == 176
    restored = CDState.from_payload(state.to_payload())
    assert restored.retained_storage_nbytes == restored.total_nbytes == 176


@pytest.mark.parametrize("field", ["c_tokens", "d_tokens", "token_dim", "frames_per_step", "reference_views"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_spec_requires_positive_integer_sizes(field, invalid):
    with pytest.raises((TypeError, ValueError)):
        _spec(**{field: invalid})


def test_contracts_are_slotted_and_frozen():
    for value in (_spec(), _state(), _observation(), _query()):
        assert not hasattr(value, "__dict__")
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            value.hidden_cache = torch.zeros(1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_complete_state_roundtrip_with_weights_only_loading_preserves_dtype(dtype):
    state = _state(dtype)
    payload = state.to_payload()
    assert set(payload) == {"schema_version", "spec", "c", "d", "last_time"}
    assert payload["schema_version"] == 1
    assert set(payload["spec"]) == {
        "c_tokens", "d_tokens", "token_dim", "frames_per_step", "reference_views"
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    restored = CDState.from_payload(torch.load(buffer, weights_only=True))
    assert restored.spec == state.spec
    for name in ("c", "d", "last_time"):
        actual, expected = getattr(restored, name), getattr(state, name)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    assert restored.total_nbytes == state.total_nbytes


@pytest.mark.parametrize("change", ["extra", "missing", "version", "bool_version", "extra_spec", "missing_spec"])
def test_state_restore_rejects_undeclared_memory_or_ambiguous_schema(change):
    payload = _state().to_payload()
    if change == "extra":
        payload["conv_cache"] = torch.zeros(2)
    elif change == "missing":
        del payload["last_time"]
    elif change == "version":
        payload["schema_version"] = 2
    elif change == "bool_version":
        payload["schema_version"] = True
    elif change == "extra_spec":
        payload["spec"]["cache_tokens"] = 4
    else:
        del payload["spec"]["reference_views"]
    with pytest.raises((TypeError, ValueError)):
        CDState.from_payload(payload)


def test_snapshot_and_export_are_detached_copies_not_mutable_aliases():
    state = _state(requires_grad=True)
    snapshot = state.snapshot()
    payload = state.to_payload()
    with torch.no_grad():
        state.c.add_(10)
        state.d.mul_(0)
        state.last_time.add_(100)
    for name in ("c", "d", "last_time"):
        assert torch.equal(getattr(snapshot, name), payload[name])
        assert not getattr(snapshot, name).requires_grad
        assert not payload[name].requires_grad
        assert getattr(snapshot, name).data_ptr() != getattr(state, name).data_ptr()


def test_detach_keeps_all_state_values_without_resetting_memory():
    state = _state(requires_grad=True)
    detached = state.detach()
    assert detached.spec == state.spec
    for name in ("c", "d", "last_time"):
        assert torch.equal(getattr(detached, name), getattr(state, name))
        assert not getattr(detached, name).requires_grad


@pytest.mark.parametrize("change", ["c_shape", "d_shape", "batch", "empty", "dtype", "integer", "time_shape", "time_dtype", "nan", "inf_time", "not_tensor"])
def test_state_rejects_invalid_tensors(change):
    state = _state()
    args = {"c": state.c, "d": state.d, "last_time": state.last_time, "spec": state.spec}
    if change == "c_shape":
        args["c"] = torch.zeros(2, 3, 4)
    elif change == "d_shape":
        args["d"] = torch.zeros(2, 3, 5)
    elif change == "batch":
        args["d"] = torch.zeros(1, 3, 4)
    elif change == "empty":
        args.update(c=torch.zeros(0, 2, 4), d=torch.zeros(0, 3, 4), last_time=torch.zeros(0, dtype=torch.float64))
    elif change == "dtype":
        args["d"] = state.d.double()
    elif change == "integer":
        args.update(c=state.c.long(), d=state.d.long())
    elif change == "time_shape":
        args["last_time"] = state.last_time[:, None]
    elif change == "time_dtype":
        args["last_time"] = state.last_time.float()
    elif change == "nan":
        args["c"][0, 0, 0] = float("nan")
    elif change == "inf_time":
        args["last_time"][0] = float("inf")
    else:
        args["c"] = state.c.tolist()
    with pytest.raises((TypeError, ValueError)):
        CDState(**args)


def test_validation_and_export_detect_later_tensor_mutation():
    state = _state()
    state.c[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        state.validate()
    with pytest.raises(ValueError):
        state.to_payload()


def test_masking_happens_before_patchify_and_is_invariant_to_hidden_values():
    observation = _observation()
    observed = observation.observed.clone()
    observed[..., 0, 1] = False
    hidden = ~observed.unsqueeze(2).expand_as(observation.images)
    ordinary = _observation(observed=observed)
    corrupted_images = observation.images.clone()
    corrupted_images[hidden] = float("nan")
    corrupted = _observation(images=corrupted_images, observed=observed)
    masked = ordinary.masked_images()
    assert torch.equal(masked, corrupted.masked_images())
    assert torch.isfinite(masked).all()
    assert torch.count_nonzero(masked[hidden]) == 0
    assert torch.equal(masked[~hidden], observation.images[~hidden])
    assert torch.isnan(corrupted.images[hidden]).all()  # Input not mutated.


def test_all_missing_observation_remains_valid_and_preserves_time():
    observation = _observation(
        images=torch.full((2, 2, 3, 2, 3), float("nan")),
        observed=torch.zeros(2, 2, 2, 3, dtype=torch.bool),
    )
    assert torch.count_nonzero(observation.masked_images()) == 0
    assert observation.end_time.item() == 0.6
    assert observation.end_time.dtype == torch.float64


@pytest.mark.parametrize("change", ["image_rank", "channels", "image_integer", "mask_shape", "mask_dtype", "time_shape", "time_dtype", "nonchronological", "equal_times", "nan_time", "nan_visible", "empty_view"])
def test_observation_rejects_invalid_visibility_and_temporal_inputs(change):
    observation = _observation()
    args = {"images": observation.images, "observed": observation.observed, "frame_times": observation.frame_times}
    if change == "image_rank":
        args["images"] = observation.images[0]
    elif change == "channels":
        args["images"] = observation.images[:, :, :2]
    elif change == "image_integer":
        args["images"] = observation.images.long()
    elif change == "mask_shape":
        args["observed"] = observation.observed[:, :1]
    elif change == "mask_dtype":
        args["observed"] = observation.observed.float()
    elif change == "time_shape":
        args["frame_times"] = observation.frame_times[0]
    elif change == "time_dtype":
        args["frame_times"] = observation.frame_times.float()
    elif change == "nonchronological":
        args["frame_times"] = observation.frame_times.flip(1)
    elif change == "equal_times":
        args["frame_times"] = torch.zeros(2, 2, dtype=torch.float64)
    elif change == "nan_time":
        args["frame_times"][0, 0] = float("nan")
    elif change == "nan_visible":
        args["images"][0, 0, 0, 0, 0] = float("nan")
    else:
        args = {key: value[:0] for key, value in args.items()}
    with pytest.raises((TypeError, ValueError)):
        ObservationStep(**args)


def test_single_frame_and_calibrated_multiview_observations_are_supported():
    observation = _observation()
    single = ObservationStep(observation.images[:, :1], observation.observed[:, :1], observation.frame_times[:, :1])
    assert single.end_time.item() == 0.1
    calibrated = _observation(
        intrinsics=torch.eye(3).repeat(2, 1, 1),
        cam_to_world=torch.eye(4).repeat(2, 2, 1, 1),
    )
    assert calibrated.cam_to_world.shape == (2, 2, 4, 4)


def test_observation_spec_validation_keeps_reference_views_even_when_missing():
    spec = _spec(reference_views=2, frames_per_step=2)
    _observation().validate_against(spec)
    _observation(observed=torch.zeros(2, 2, 2, 3, dtype=torch.bool)).validate_against(spec)
    with pytest.raises(ValueError):
        _observation().validate_against(_spec(reference_views=1))
    with pytest.raises(ValueError):
        _observation().validate_against(_spec(reference_views=2, frames_per_step=1))
    observation = _observation()
    ObservationStep(
        observation.images[:, :1], observation.observed[:, :1], observation.frame_times[:, :1]
    ).validate_against(spec)


@pytest.mark.parametrize("change", ["unpaired", "shape", "nonfinite", "bad_homogeneous_row", "singular_intrinsics", "not_rigid"])
def test_observation_rejects_invalid_camera_metadata(change):
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    cam_to_world = torch.eye(4).repeat(2, 2, 1, 1)
    if change == "unpaired":
        cam_to_world = None
    elif change == "shape":
        intrinsics = intrinsics[:1]
    elif change == "nonfinite":
        intrinsics[0, 0, 0] = float("nan")
    elif change == "bad_homogeneous_row":
        cam_to_world[0, 0, 3, 0] = 1
    elif change == "singular_intrinsics":
        intrinsics[0, 0, 0] = 0
    else:
        cam_to_world[0, 0, 0, 0] = 2
    with pytest.raises((TypeError, ValueError)):
        _observation(intrinsics=intrinsics, cam_to_world=cam_to_world)


def test_student_interfaces_have_no_oracle_or_feature_fields():
    assert {field.name for field in fields(ObservationStep)} == {
        "images", "observed", "frame_times", "intrinsics", "cam_to_world"
    }
    assert {field.name for field in fields(DecoderQuery)} == {
        "coordinates", "view_ids", "times", "intrinsics", "cam_to_world", "coordinate_space"
    }
    with pytest.raises(TypeError):
        _query(features=torch.ones(2, 3))
    with pytest.raises(TypeError):
        _observation(event_id=1)


def test_contracts_reject_mixed_devices_before_reading_tensor_values():
    state = _state()
    with pytest.raises(ValueError, match="device"):
        CDState(state.c, state.d, state.last_time.to("meta"), spec=state.spec)
    with pytest.raises(ValueError, match="device"):
        _observation(observed=torch.ones(2, 2, 2, 3, dtype=torch.bool, device="meta"))
    with pytest.raises(ValueError, match="device"):
        _query(view_ids=torch.tensor([0, 1], dtype=torch.int64, device="meta"))


def test_decoder_query_uses_explicit_calibration_view_table():
    query = _query(
        intrinsics=torch.eye(3).repeat(2, 1, 1),
        cam_to_world=torch.eye(4).repeat(2, 1, 1),
    )
    query.validate()
    assert query.cam_to_world.shape == (2, 4, 4)


@pytest.mark.parametrize("factory", [_observation, _query])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_camera_metadata_accepts_non_axis_aligned_full_precision_rotation(factory, dtype):
    intrinsics = torch.eye(3, dtype=dtype).repeat(2, 1, 1)
    poses = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    poses[:, :3, :3] = torch.tensor(
        [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
    )
    if factory is _observation:
        poses = poses[:, None].repeat(1, 2, 1, 1)
    value = factory(intrinsics=intrinsics, cam_to_world=poses)
    value.validate()
    assert value.intrinsics.dtype == dtype
    assert value.cam_to_world.dtype == dtype


@pytest.mark.parametrize("factory", [_observation, _query])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("field", ["intrinsics", "cam_to_world"])
def test_camera_metadata_explicitly_rejects_half_precision(factory, dtype, field):
    args = {"intrinsics": torch.eye(3).repeat(2, 1, 1)}
    shape = (2, 2, 1, 1) if factory is _observation else (2, 1, 1)
    args["cam_to_world"] = torch.eye(4).repeat(*shape)
    args[field] = args[field].to(dtype)
    with pytest.raises(ValueError, match=f"{field}.*float32.*float64"):
        factory(**args)


def test_half_precision_images_can_use_full_precision_calibration():
    value = _observation(
        images=torch.zeros(2, 2, 3, 2, 3, dtype=torch.float16),
        intrinsics=torch.eye(3).repeat(2, 1, 1),
        cam_to_world=torch.eye(4).repeat(2, 2, 1, 1),
    )
    assert value.images.dtype == torch.float16
    assert value.cam_to_world.dtype == torch.float32


def test_query_coordinate_space_is_explicit_and_defaults_to_image_pixels():
    query = _query(coordinates=torch.tensor([[0.5, 0.5, 0.0], [12.5, 5.5, 0.0]]))
    assert query.coordinate_space == "image_pixel"
    world = _query(
        coordinates=torch.tensor([[1.0, -2.0, 5.0], [-3.0, 0.0, 0.5]]),
        coordinate_space="world_xyz",
    )
    assert world.coordinate_space == "world_xyz"
    world.validate()


@pytest.mark.parametrize("space", ["camera_xyz", "normalized", "", None, 3])
def test_query_rejects_undeclared_coordinate_space(space):
    with pytest.raises(ValueError, match="coordinate_space"):
        _query(coordinate_space=space)


@pytest.mark.parametrize("space", ["image_pixel", "triplane_uv"])
def test_two_dimensional_query_spaces_require_zero_z(space):
    with pytest.raises(ValueError, match="coordinates.*z"):
        _query(
            coordinates=torch.tensor([[0.5, 0.5, 1.0], [0.5, 0.5, 0.0]]),
            coordinate_space=space,
        )


def test_triplane_queries_use_normalized_uv_and_explicit_plane_ids():
    query = _query(
        coordinates=torch.tensor([[-1.0, 1.0, 0.0], [0.0, 0.5, 0.0], [1.0, -1.0, 0.0]]),
        view_ids=torch.tensor([0, 1, 2]),
        times=torch.zeros(3, dtype=torch.float64),
        coordinate_space="triplane_uv",
    )
    query.validate()
    assert query.coordinate_space == "triplane_uv"


@pytest.mark.parametrize("change", ["uv_range", "plane_id", "camera"])
def test_triplane_query_rejects_ambiguous_geometry(change):
    args = {"coordinate_space": "triplane_uv"}
    if change == "uv_range":
        args["coordinates"] = torch.tensor([[1.01, 0.5, 0.0], [0.5, 0.5, 0.0]])
    elif change == "plane_id":
        args["view_ids"] = torch.tensor([0, 3])
    else:
        args.update(intrinsics=torch.eye(3).repeat(2, 1, 1), cam_to_world=torch.eye(4).repeat(2, 1, 1))
    with pytest.raises(ValueError):
        _query(**args)


@pytest.mark.parametrize("change", ["coordinate_shape", "coordinate_integer", "nonfinite", "view_shape", "view_dtype", "negative_view", "time_shape", "time_dtype", "nan_time", "camera_range", "unpaired_camera"])
def test_decoder_query_rejects_invalid_payload(change):
    query = _query()
    args = {"coordinates": query.coordinates, "view_ids": query.view_ids, "times": query.times}
    if change == "coordinate_shape":
        args["coordinates"] = query.coordinates[:, :2]
    elif change == "coordinate_integer":
        args["coordinates"] = query.coordinates.long()
    elif change == "nonfinite":
        args["coordinates"][0, 0] = float("inf")
    elif change == "view_shape":
        args["view_ids"] = query.view_ids[:, None]
    elif change == "view_dtype":
        args["view_ids"] = query.view_ids.float()
    elif change == "negative_view":
        args["view_ids"][0] = -1
    elif change == "time_shape":
        args["times"] = query.times[:1]
    elif change == "time_dtype":
        args["times"] = query.times.float()
    elif change == "nan_time":
        args["times"][0] = float("nan")
    elif change == "camera_range":
        args.update(intrinsics=torch.eye(3)[None], cam_to_world=torch.eye(4)[None])
    else:
        args["intrinsics"] = torch.eye(3).repeat(2, 1, 1)
    with pytest.raises((TypeError, ValueError)):
        DecoderQuery(**args)
