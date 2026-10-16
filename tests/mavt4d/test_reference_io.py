"""Reference I/O checks: observation-only evidence and C-valued readout."""

import inspect

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery, ObservationStep
from mavt4d.reference.decoder import CTemporalHead, StructuredQueryDecoder
from mavt4d.reference.observation import ObservationEncoder, _local_neighbor_attention


def _spec(views=1):
    return CDStateSpec(c_tokens=3, d_tokens=2, token_dim=8, reference_views=views)


def _observation(views=1, frames=2, height=8, width=12, calibration=True):
    generator = torch.Generator().manual_seed(91)
    images = torch.rand(views, frames, 3, height, width, generator=generator)
    observed = torch.ones(views, frames, height, width, dtype=torch.bool)
    times = torch.arange(frames, dtype=torch.float64).expand(views, -1).clone() / 10 + 1
    if not calibration:
        return ObservationStep(images, observed, times)
    intrinsics = torch.tensor([[8., 0., width / 2], [0., 8., height / 2], [0., 0., 1.]])
    intrinsics = intrinsics.expand(views, -1, -1).clone()
    poses = torch.eye(4).expand(views, frames, -1, -1).clone()
    poses[:, :, 0, 3] = torch.arange(views).unsqueeze(-1)
    return ObservationStep(images, observed, times, intrinsics, poses)


def _state(spec=None, batch=2, requires_grad=False):
    spec = spec or _spec()
    generator = torch.Generator().manual_seed(18)
    c = torch.randn(batch, spec.c_tokens, spec.token_dim, generator=generator,
                    requires_grad=requires_grad)
    d = torch.randn(batch, spec.d_tokens, spec.token_dim, generator=generator,
                    requires_grad=requires_grad)
    return CDState(c, d, torch.full((batch,), 2., dtype=torch.float64), spec=spec)


def _query(count=4, space="image_pixel"):
    coordinates = torch.linspace(-.8, .8, count * 3).reshape(count, 3)
    if space != "world_xyz":
        coordinates[:, 2] = 0
    views = torch.arange(count, dtype=torch.int64) % 3 if space == "triplane_uv" else torch.zeros(count, dtype=torch.int64)
    return DecoderQuery(coordinates, views, torch.linspace(1.5, 2., count, dtype=torch.float64),
                        coordinate_space=space)


def _slice_query(query, index):
    return DecoderQuery(query.coordinates[index], query.view_ids[index], query.times[index],
                        query.intrinsics, query.cam_to_world, query.coordinate_space)


@pytest.mark.parametrize("frames", [1, 2])
def test_observation_evidence_shape_and_single_shared_projection(frames):
    encoder = ObservationEncoder(_spec(), embed_dim=16, patch_size=4)
    evidence, has_observation = encoder(_observation(frames=frames))
    assert evidence.shape == (5, 16)
    assert has_observation.shape == () and has_observation.dtype == torch.bool
    assert bool(has_observation)
    convolutions = [module for module in encoder.modules() if isinstance(module, torch.nn.Conv3d)]
    assert len(convolutions) == 1
    assert convolutions[0].kernel_size == (2, 4, 4)
    assert torch.isfinite(evidence).all()


def test_hidden_pixels_do_not_change_evidence_and_have_zero_gradient():
    torch.manual_seed(1)
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    original = _observation()
    mask = original.observed.clone()
    mask[:, :, :, :4] = False
    images = original.images.clone().requires_grad_()
    first = ObservationStep(images, mask, original.frame_times)
    poisoned = torch.where(mask.unsqueeze(2), images.detach(), torch.full_like(images, float("nan")))
    second = ObservationStep(poisoned, mask, original.frame_times)
    actual, _ = encoder(first)
    expected, _ = encoder(second)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.count_nonzero(images.grad[~mask.unsqueeze(2).expand_as(images)]) == 0
    assert torch.count_nonzero(images.grad[mask.unsqueeze(2).expand_as(images)]) > 0


def test_fully_missing_observation_is_exact_zero_despite_metadata_and_biases():
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    sample = _observation()
    missing = ObservationStep(torch.full_like(sample.images, float("nan")),
                              torch.zeros_like(sample.observed), sample.frame_times,
                              sample.intrinsics, sample.cam_to_world)
    evidence, has_observation = encoder(missing)
    assert not bool(has_observation)
    assert torch.isfinite(evidence).all()
    assert torch.count_nonzero(evidence) == 0


def test_coverage_distinguishes_black_observation_from_missing_pixels():
    torch.manual_seed(2)
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    sample = _observation(calibration=False)
    black = torch.zeros_like(sample.images)
    partial_mask = sample.observed.clone()
    partial_mask[:, :, :, :4] = False
    full, _ = encoder(ObservationStep(black, sample.observed, sample.frame_times))
    partial, _ = encoder(ObservationStep(black, partial_mask, sample.frame_times))
    assert not torch.allclose(full, partial)


def test_camera_view_permutation_preserves_evidence_with_metadata():
    torch.manual_seed(3)
    encoder = ObservationEncoder(_spec(3), embed_dim=16)
    sample = _observation(views=3)
    permutation = torch.tensor([2, 0, 1])
    shuffled = ObservationStep(sample.images[permutation], sample.observed[permutation],
                               sample.frame_times[permutation], sample.intrinsics[permutation],
                               sample.cam_to_world[permutation])
    first, _ = encoder(sample)
    second, _ = encoder(shuffled)
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("change", ["timestamp", "geometry", "coverage"])
def test_observation_metadata_changes_evidence(change):
    torch.manual_seed(4)
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    sample = _observation()
    times, pose, mask = sample.frame_times.clone(), sample.cam_to_world.clone(), sample.observed.clone()
    if change == "timestamp":
        times[:, 0] -= .5
    elif change == "geometry":
        pose[:, :, 0, 3] += 2
    else:
        mask[:, :, :4] = False
    modified = ObservationStep(sample.images, mask, times, sample.intrinsics, pose)
    assert not torch.allclose(encoder(sample)[0], encoder(modified)[0])


def test_triplane_uses_shared_projection_and_meaningful_plane_identity():
    torch.manual_seed(5)
    encoder = ObservationEncoder(_spec(3), embed_dim=16)
    sample = _observation(views=3, frames=1, calibration=False)
    calls = []
    projection = next(module for module in encoder.modules() if isinstance(module, torch.nn.Conv3d))
    handle = projection.register_forward_hook(lambda *args: calls.append(True))
    camera, _ = encoder(sample, modality="camera")
    triplane, _ = encoder(sample, modality="triplane")
    swapped = ObservationStep(sample.images.flip(0), sample.observed, sample.frame_times)
    reordered, _ = encoder(swapped, modality="triplane")
    handle.remove()
    assert len(calls) == 3
    assert triplane.shape == camera.shape
    assert not torch.allclose(camera, triplane)
    assert not torch.allclose(triplane, reordered)


@pytest.mark.parametrize("height,width", [(9, 12), (8, 13)])
def test_patchify_rejects_silent_spatial_truncation(height, width):
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    with pytest.raises(ValueError, match="divisible"):
        encoder(_observation(height=height, width=width))


def test_triplane_rejects_camera_metadata_and_temporal_tubelets():
    encoder = ObservationEncoder(_spec(3), embed_dim=16)
    with pytest.raises(ValueError, match="triplane"):
        encoder(_observation(3, frames=1), modality="triplane")
    with pytest.raises(ValueError, match="triplane"):
        encoder(_observation(3, frames=2, calibration=False), modality="triplane")


def test_local_neighbor_mix_is_spatial_and_ignores_missing_neighbors():
    patches = torch.zeros(1, 5, 6, 2)
    coverage = torch.ones(1, 5, 6)
    patches[0, 2, 2] = 9
    query_key = torch.zeros_like(patches)
    baseline = _local_neighbor_attention(query_key, query_key, patches, coverage, radius=1)
    assert baseline[0, 2, 2, 0] > 0
    assert baseline[0, 0, 5, 0] == 0
    coverage[0, 2, 2] = 0
    assert torch.count_nonzero(_local_neighbor_attention(query_key, query_key, patches, coverage, radius=1)) == 0


def test_encoder_does_not_materialize_patch_by_patch_attention():
    class Shapes(TorchDispatchMode):
        def __init__(self):
            self.shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            values = result if isinstance(result, (tuple, list)) else (result,)
            self.shapes.extend(tuple(value.shape) for value in values if isinstance(value, torch.Tensor))
            return result

    encoder = ObservationEncoder(_spec(2), embed_dim=16)
    sample = _observation(2, height=16, width=24)
    patch_count = 2 * 4 * 6
    mode = Shapes()
    with mode:
        encoder(sample)
    forbidden_sizes = (patch_count, patch_count // 2)
    assert not any(len(shape) >= 2 and shape[-2:] == (size, size)
                   for shape in mode.shapes for size in forbidden_sizes)


@pytest.mark.parametrize("space", ["image_pixel", "world_xyz", "triplane_uv"])
def test_decoder_supports_declared_coordinate_spaces(space):
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16, modulation_dim=4)
    output = decoder(_state(), _query(space=space))
    assert output.shape == (2, 4, 3)
    assert torch.isfinite(output).all()


def test_decoder_reads_c_content_and_nonzero_d_routing_gradients():
    torch.manual_seed(6)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16, modulation_dim=4)
    state = _state(requires_grad=True)
    decoder(state, _query()).square().sum().backward()
    assert torch.isfinite(state.c.grad).all() and torch.count_nonzero(state.c.grad) > 0
    assert torch.isfinite(state.d.grad).all() and torch.count_nonzero(state.d.grad) > 0


def test_zero_c_removes_every_d_to_rgb_content_path():
    torch.manual_seed(7)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16, modulation_dim=4)
    state = _state()
    zero_c = torch.zeros_like(state.c)
    first = CDState(zero_c, state.d, state.last_time, spec=state.spec)
    second = CDState(zero_c, state.d * 100 - 17, state.last_time, spec=state.spec)
    torch.testing.assert_close(decoder(first, _query()), decoder(second, _query()), rtol=0, atol=0)


def test_decoder_query_batches_are_independent_and_permutation_equivariant():
    torch.manual_seed(8)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16, modulation_dim=4)
    state, query = _state(), _query(7)
    whole = decoder(state, query)
    split = torch.cat([decoder(state, _slice_query(query, slice(0, 3))),
                       decoder(state, _slice_query(query, slice(3, 7)))], dim=1)
    torch.testing.assert_close(whole, split, atol=1e-6, rtol=1e-5)
    permutation = torch.tensor([3, 0, 6, 1, 5, 4, 2])
    reordered = decoder(state, _slice_query(query, permutation))
    torch.testing.assert_close(reordered, whole[:, permutation], atol=1e-6, rtol=1e-5)


def test_decoder_rejects_future_query_for_any_stream():
    state = _state()
    state = CDState(state.c, state.d, torch.tensor([2., 1.9], dtype=torch.float64), spec=state.spec)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16)
    with pytest.raises(ValueError, match="rollout"):
        decoder(state, _query())


def test_camera_ids_have_no_invented_geometry_without_calibration():
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16)
    state, query = _state(), _query()
    other_ids = DecoderQuery(query.coordinates, query.view_ids + 97, query.times)
    torch.testing.assert_close(decoder(state, query), decoder(state, other_ids), rtol=0, atol=0)


def test_decoder_camera_table_permutation_preserves_geometry():
    torch.manual_seed(9)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16)
    state, query = _state(), _query()
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 3
    ids = torch.tensor([0, 1, 0, 1])
    calibrated = DecoderQuery(query.coordinates, ids, query.times, intrinsics, poses)
    reordered = DecoderQuery(query.coordinates, 1 - ids, query.times, intrinsics.flip(0), poses.flip(0))
    actual = decoder(state, calibrated)
    torch.testing.assert_close(actual, decoder(state, reordered), rtol=0, atol=0)
    shifted = poses.clone()
    shifted[:, 0, 3] += 5
    changed = DecoderQuery(query.coordinates, ids, query.times, intrinsics, shifted)
    assert not torch.allclose(actual, decoder(state, changed))


def test_modules_have_no_sequence_buffers_and_forward_does_not_mutate_weights():
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16)
    for module in (encoder, decoder):
        assert list(module.named_buffers()) == []
    before = {name: value.clone() for name, value in encoder.state_dict().items()}
    encoder(_observation())
    assert all(torch.equal(before[name], value) for name, value in encoder.state_dict().items())
    before = {name: value.clone() for name, value in decoder.state_dict().items()}
    decoder(_state(), _query())
    assert all(torch.equal(before[name], value) for name, value in decoder.state_dict().items())


def test_temporal_head_accepts_only_c_and_has_no_d_parameter():
    head = CTemporalHead(_spec(), output_dim=6)
    state = _state(requires_grad=True)
    output = head(state.c)
    assert output.shape == (2, 6)
    output.square().sum().backward()
    assert state.c.grad is not None
    assert state.d.grad is None
    assert tuple(inspect.signature(head.forward).parameters) == ("c",)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_io_supports_float32_and_float64_with_double_timestamps(dtype):
    encoder = ObservationEncoder(_spec(), embed_dim=16).to(dtype=dtype)
    decoder = StructuredQueryDecoder(_spec(), hidden_dim=16).to(dtype=dtype)
    sample = _observation()
    sample = ObservationStep(sample.images.to(dtype), sample.observed, sample.frame_times,
                             sample.intrinsics, sample.cam_to_world)
    assert encoder(sample)[0].dtype == dtype
    state = _state()
    state = CDState(state.c.to(dtype), state.d.to(dtype), state.last_time, spec=state.spec)
    assert decoder(state, _query())[0].dtype == dtype


def test_mismatched_reference_dtypes_fail_with_clear_error():
    encoder = ObservationEncoder(_spec(), embed_dim=16)
    sample = _observation()
    wrong = ObservationStep(sample.images.double(), sample.observed, sample.frame_times)
    with pytest.raises(ValueError, match="dtype"):
        encoder(wrong)
    state = _state()
    wrong_state = CDState(state.c.double(), state.d.double(), state.last_time, spec=state.spec)
    with pytest.raises(ValueError, match="dtype"):
        StructuredQueryDecoder(_spec(), hidden_dim=16)(wrong_state, _query())
