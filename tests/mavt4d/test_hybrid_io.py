"""Stateless visual input and C/D-only tile-readout gates for hybrid v2."""

from dataclasses import replace

import pytest
import torch

from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.decoder import HybridQueryDecoder, tile_grid_query, tiles_to_image
from mavt4d.hybrid.observation import HybridObservationEncoder, SpatialBlock
from mavt4d.hybrid.state import CDCode


def config(**kwargs):
    return HybridConfig.tiny(**kwargs)


def observation(*, height=5, width=7, views=1, frames=2):
    rng = torch.Generator().manual_seed(105)
    return ObservationStep(
        torch.rand(views, frames, 3, height, width, generator=rng),
        torch.ones(views, frames, height, width, dtype=torch.bool),
        torch.arange(1, frames + 1, dtype=torch.float64)[None].expand(views, -1).clone(),
    )


def code(cfg=None, *, zero_c=False):
    cfg = config() if cfg is None else cfg
    shape = (2, cfg.stage_count, cfg.state_tokens, cfg.state_dim)
    rng = torch.Generator().manual_seed(106)
    c = torch.zeros(shape) if zero_c else torch.randn(shape, generator=rng)
    return CDCode(c, torch.randn(shape, generator=rng), torch.full((2,), 2., dtype=torch.float64), config=cfg)


def query(points, *, time=2.):
    return DecoderQuery(torch.tensor([[x, y, 0.] for x, y in points]),
                        torch.zeros(len(points), dtype=torch.int64),
                        torch.full((len(points),), time, dtype=torch.float64))


def test_encoder_masks_hidden_nan_before_patchify_and_blocks_hidden_gradients():
    torch.manual_seed(100)
    encoder = HybridObservationEncoder(config())
    obs = observation()
    mask = obs.observed.clone()
    mask[..., :3, :3] = False
    clean = replace(obs, observed=mask)
    hidden = ~mask.unsqueeze(2).expand_as(obs.images)
    corrupt = obs.images.clone()
    corrupt[hidden] = float("nan")
    corrupt.requires_grad_()
    expected, observed = encoder(clean)
    actual, actual_observed = encoder(replace(clean, images=corrupt))
    assert bool(observed) and bool(actual_observed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(corrupt.grad).all()
    assert torch.count_nonzero(corrupt.grad[hidden]) == 0
    assert corrupt.grad[~hidden].abs().sum() > 0


def test_encoder_missing_skips_patchify_and_rejects_observed_nan():
    encoder = HybridObservationEncoder(config())
    obs = observation()
    missing = replace(obs, images=torch.full_like(obs.images, float("nan")),
                      observed=torch.zeros_like(obs.observed))
    calls = []
    hook = encoder.patchify.register_forward_hook(lambda *args: calls.append(True))
    evidence, observed = encoder(missing)
    hook.remove()
    assert not bool(observed) and not calls
    assert evidence.shape == (encoder.config.summary_tokens, encoder.config.width)
    assert torch.count_nonzero(evidence) == 0
    with pytest.raises(ValueError, match="finite"):
        replace(obs, images=torch.full_like(obs.images, float("nan")))


@pytest.mark.parametrize("height,width", [(1, 1), (3, 7), (8, 8)])
def test_encoder_patch_and_window_padding_are_finite_without_retained_memory(height, width):
    cfg = config(spatial_blocks=4)
    encoder = HybridObservationEncoder(cfg)
    obs = observation(height=height, width=width)
    result, observed = encoder(obs)
    assert result.shape == (cfg.summary_tokens, cfg.width) and bool(observed)
    assert torch.isfinite(result).all()
    assert not list(encoder.named_buffers())
    encoder(observation(height=9, width=3))
    torch.testing.assert_close(encoder(obs)[0], result, rtol=0, atol=0)


def test_shared_patchify_image_video_triplane_and_plane_identity():
    cfg = config(reference_views=3)
    encoder = HybridObservationEncoder(cfg)
    weight_id = id(encoder.patchify.weight)
    image = observation(views=3, frames=1)
    camera = encoder(image, modality="camera")[0]
    plane = encoder(image, modality="triplane")[0]
    video = encoder(observation(views=3), modality="camera")[0]
    assert camera.shape == plane.shape == video.shape
    assert id(encoder.patchify.weight) == weight_id
    assert not torch.allclose(camera, plane)
    with pytest.raises(ValueError, match="triplane"):
        encoder(observation(views=3), modality="triplane")


@pytest.mark.parametrize("calibrated", [False, True])
def test_camera_indices_distinguish_swapped_observations_with_and_without_geometry(calibrated):
    torch.manual_seed(107)
    cfg = config(reference_views=2, spatial_blocks=4)
    encoder = HybridObservationEncoder(cfg)
    obs = observation(views=2)
    if calibrated:
        obs = replace(obs, intrinsics=torch.eye(3).repeat(2, 1, 1),
                      cam_to_world=torch.eye(4).repeat(2, 2, 1, 1))
    swapped = replace(obs, images=obs.images.flip(0), observed=obs.observed.flip(0),
                      frame_times=obs.frame_times.flip(0))
    # Row zero is the fixed identity queried by DecoderQuery.view_ids==0.
    # Swapping its RGB with row one must not erase this target distinction.
    assert not torch.allclose(encoder(obs)[0], encoder(swapped)[0], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("views", [1, 3])
def test_camera_index_metadata_is_fixed_per_row_and_receives_gradients(views):
    torch.manual_seed(108)
    cfg = config(reference_views=views)
    encoder = HybridObservationEncoder(cfg)
    assert encoder.camera_view_embedding.weight.shape == (views, cfg.spatial_width)
    obs = observation(views=views)
    # Identical RGB, times, and geometry isolate only the camera-row metadata.
    obs = replace(obs, images=obs.images[:1].expand_as(obs.images).clone())
    patches = []
    hook = encoder.blocks[0].register_forward_pre_hook(lambda _module, inputs: patches.append(inputs[0]))
    evidence, _ = encoder(obs)
    hook.remove()
    for row in range(views):
        actual = patches[0][row] - patches[0][0]
        expected = encoder.camera_view_embedding.weight[row] - encoder.camera_view_embedding.weight[0]
        torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-5, atol=1e-6)
    evidence.square().sum().backward()
    gradient = encoder.camera_view_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool((gradient.abs().sum(dim=-1) > 0).all())


def test_geometry_remains_separate_from_camera_index_metadata():
    torch.manual_seed(112)
    encoder = HybridObservationEncoder(config(reference_views=2))
    obs = observation(views=2)
    calibrated = replace(obs, intrinsics=torch.eye(3).repeat(2, 1, 1),
                         cam_to_world=torch.eye(4).repeat(2, 2, 1, 1))
    assert not torch.allclose(encoder(obs)[0], encoder(calibrated)[0])


@pytest.mark.parametrize("shift", [0, 1])
def test_window_attention_cannot_cross_views_or_wrap_opposite_edge(shift):
    torch.manual_seed(109)
    block = SpatialBlock(width=8, heads=2, window_size=2, shift=shift, global_attention=False)
    tokens = torch.randn(2, 3, 5, 8)
    coverage = torch.ones(2, 3, 5)
    changed = tokens.clone()
    changed[1] = torch.randn_like(changed[1]) * 10
    changed[0, -1, -1] = torch.randn(8) * 10
    baseline = block(tokens, coverage)
    perturbed = block(changed, coverage)
    torch.testing.assert_close(baseline[0, 0, 0], perturbed[0, 0, 0], rtol=0, atol=0)
    unavailable = block(tokens, torch.zeros_like(coverage))
    assert torch.isfinite(unavailable).all()
    assert torch.count_nonzero(unavailable) == 0


def test_decoder_zero_content_has_exact_zero_rgb_even_with_nonzero_biases():
    decoder = HybridQueryDecoder(config())
    for name, parameter in decoder.named_parameters():
        if name.endswith("bias"):
            with torch.no_grad():
                parameter.fill_(.7)
    initial = code(zero_c=True)
    points = query([(0.5, .5), (3.5, 2.5), (4.5, 6.5)])
    assert torch.count_nonzero(decoder(initial, points)) == 0
    altered = replace(initial, d=initial.d * 100)
    assert torch.count_nonzero(decoder(altered, points)) == 0
    assert torch.count_nonzero(decoder.decode_tiles(initial, tile_grid_query(5, 7, tile_size=2, time=2.))) == 0


def test_decoder_no_query_coupling_sparse_full_chunk_and_arbitrary_pixel_parity():
    torch.manual_seed(110)
    decoder = HybridQueryDecoder(config())
    initial = code()
    grid = tile_grid_query(5, 7, tile_size=2, time=2.)
    tiles = decoder.decode_tiles(initial, grid)
    parts = []
    for index in range(grid.coordinates.shape[0]):
        single = replace(grid, coordinates=grid.coordinates[index:index + 1],
                         view_ids=grid.view_ids[index:index + 1], times=grid.times[index:index + 1])
        parts.append(decoder.decode_tiles(initial, single))
    torch.testing.assert_close(tiles, torch.cat(parts, dim=1), rtol=2e-5, atol=2e-6)
    image = tiles_to_image(tiles, height=5, width=7)
    torch.testing.assert_close(image, decoder.render(initial, height=5, width=7, time=2., chunk_size=3),
                               rtol=2e-5, atol=2e-6)
    points = [(6.5, 4.5), (.5, .5), (3.5, 2.5), (6.5, 4.5)]
    pixels = decoder(initial, query(points))
    selected = torch.stack([image[:, :, int(y), int(x)] for x, y in points], dim=1)
    torch.testing.assert_close(pixels, selected, rtol=2e-5, atol=2e-6)
    reordered = decoder(initial, query(points[::-1]))
    torch.testing.assert_close(reordered, pixels.flip(1), rtol=2e-5, atol=2e-6)


def test_tile_channel_order_row_major_and_boundary_crop():
    # Four 2x2 tiles, unique value encodes tile/channel/dy/dx independently.
    tiles = torch.arange(4 * 3 * 2 * 2).reshape(1, 4, 3, 2, 2).float()
    image = tiles_to_image(tiles, height=3, width=3)
    assert image.shape == (1, 3, 3, 3)
    for y in range(3):
        for x in range(3):
            tile = (y // 2) * 2 + x // 2
            torch.testing.assert_close(image[0, :, y, x], tiles[0, tile, :, y % 2, x % 2])
    grid = tile_grid_query(3, 3, tile_size=2, time=2.)
    torch.testing.assert_close(grid.coordinates, torch.tensor([[1., 1., 0.], [3., 1., 0.],
                                                             [1., 3., 0.], [3., 3., 0.]]))


def test_decoder_rejects_future_world_and_misaligned_tiles_and_no_hidden_buffers():
    decoder = HybridQueryDecoder(config())
    initial = code()
    with pytest.raises(ValueError, match="future|rollout"):
        decoder(initial, query([(.5, .5)], time=2.1))
    with pytest.raises(ValueError, match="image_pixel"):
        decoder(initial, replace(query([(.5, .5)]), coordinate_space="world_xyz"))
    with pytest.raises(ValueError, match="align|center"):
        decoder.decode_tiles(initial, query([(.5, .5)]))
    assert not list(decoder.named_buffers())
    points = query([(.5, .5)])
    expected = decoder(initial, points)
    decoder(replace(initial, c=initial.c + 1), query([(4.5, 2.5)]))
    torch.testing.assert_close(decoder(initial, points), expected, rtol=0, atol=0)


def test_decoder_c_and_d_receive_finite_nonzero_gradients_without_kv_input():
    torch.manual_seed(111)
    decoder = HybridQueryDecoder(config())
    initial = code()
    initial = replace(initial, c=initial.c.clone().requires_grad_(), d=initial.d.clone().requires_grad_())
    decoder(initial, query([(.5, .5), (3.5, 2.5)])).square().sum().backward()
    for tensor in (initial.c, initial.d):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
