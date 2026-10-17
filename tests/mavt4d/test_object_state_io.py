"""Observation/readout correctness, independent of optimization results."""

from dataclasses import replace

import pytest
import torch

from mavt4d.object_state.config import ObjectStateConfig
from mavt4d.object_state.state import ObjectState
from mavt4d.object_state.observation import ObservationStep, SparseObservationEncoder
from mavt4d.object_state.decoder import (
    CanonicalReadout,
    RayQuery,
    make_ray_query,
    volume_render,
)
from mavt4d.object_state.model import ObjectStateModel


def cfg():
    return ObjectStateConfig(
        slots=2,
        content_tokens=2,
        dynamic_tokens=2,
        width=16,
        background_tokens=2,
        evidence_width=24,
        spatial_blocks=4,
        spatial_heads=4,
        window_size=2,
        relay_tokens=2,
        resampler_heads=4,
        candidates=8,
        patch_size=4,
        samples_per_ray=8,
    )


def observation():
    images = torch.rand(2, 2, 2, 3, 8, 8)
    valid = torch.ones(2, 2, 2, 1, 8, 8, dtype=torch.bool)
    k = (
        torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
        .expand(2, 2, 3, 3)
        .clone()
    )
    poses = torch.eye(4).expand(2, 2, 4, 4).clone()
    poses[:, 1, 0, 3] = 1.0
    return ObservationStep(
        images=images,
        valid=valid,
        time=torch.tensor([1.0, 2.0], dtype=torch.float64),
        intrinsics=k,
        camera_c2w=poses,
        depth=torch.full((2, 2, 1, 8, 8), 2.0),
        depth_valid=torch.ones(2, 2, 1, 8, 8, dtype=torch.bool),
    )


def populated_state(config):
    state = ObjectState.initial(config, 2, time=1.0)
    return replace(
        state,
        c=torch.randn_like(state.c),
        d=torch.randn_like(state.d),
        c_bg=torch.randn_like(state.c_bg),
        slot_status=torch.ones_like(state.slot_status),
        p_exist=torch.ones_like(state.p_exist),
    )


def query(count=7):
    directions = torch.randn(2, count, 3)
    directions[..., 2] += 3.0
    directions = torch.nn.functional.normalize(directions, dim=-1)
    return RayQuery(
        origins=torch.zeros(2, count, 3),
        directions=directions,
        near=0.2,
        far=3.0,
        time=torch.ones(2, dtype=torch.float64),
    )


def test_encoder_keeps_endpoint_geometry_and_fixed_evidence_width():
    config = cfg()
    encoded = SparseObservationEncoder(config)(observation())
    encoded.validate(config)
    assert encoded.tokens.shape == (2, 8, 24)
    assert (
        encoded.valid.all()
        and encoded.geometry_valid.all()
        and encoded.point_valid.all()
    )
    torch.testing.assert_close(encoded.ray_origins[:, 4:, 0], torch.ones(2, 4))
    torch.testing.assert_close(encoded.points[..., 2], torch.full((2, 8), 2.0))
    torch.testing.assert_close(encoded.ray_directions.norm(dim=-1), torch.ones(2, 8))


def test_masked_pixels_cannot_change_evidence_or_introduce_nan():
    torch.manual_seed(4)
    encoder = SparseObservationEncoder(cfg()).eval()
    obs = observation()
    obs.valid[:, :, 0] = False
    clean = encoder(obs)
    poisoned = obs.images.clone()
    poisoned[:, :, 0] = float("nan")
    result = encoder(replace(obs, images=poisoned))
    torch.testing.assert_close(result.tokens, clean.tokens, rtol=0, atol=0)
    absent = encoder(
        replace(
            obs,
            images=torch.full_like(obs.images, float("nan")),
            valid=torch.zeros_like(obs.valid),
        )
    )
    assert not absent.valid.any()
    assert torch.isfinite(absent.tokens).all() and (absent.tokens == 0).all()


def test_uncalibrated_rgb_marks_geometry_missing_and_encoder_has_no_cache():
    encoder = SparseObservationEncoder(cfg()).eval()
    obs = replace(
        observation(), intrinsics=None, camera_c2w=None, depth=None, depth_valid=None
    )
    first = encoder(obs)
    encoder(replace(obs, images=obs.images * 0.1))
    second = encoder(obs)
    assert first.valid.all() and not first.geometry_valid.any()
    assert first.points is None and first.point_valid is None
    torch.testing.assert_close(first.tokens, second.tokens, rtol=0, atol=0)


def test_encoder_rejects_future_tubelet_and_bad_calibration():
    obs = observation()
    with pytest.raises(ValueError, match="tubelet|frames"):
        SparseObservationEncoder(cfg())(
            replace(
                obs,
                images=obs.images.repeat(1, 1, 2, 1, 1, 1),
                valid=obs.valid.repeat(1, 1, 2, 1, 1, 1),
            )
        )
    bad = obs.camera_c2w.clone()
    bad[0, 0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="rotation|camera"):
        SparseObservationEncoder(cfg())(replace(obs, camera_c2w=bad))


def test_controls_are_identity_at_zero_and_valid_rotations():
    config = cfg()
    readout = CanonicalReadout(config)
    t = readout.controls(torch.zeros(2, 2, 2, 16))
    torch.testing.assert_close(t, torch.eye(4).expand_as(t), rtol=0, atol=1e-6)
    t = readout.controls(torch.randn(2, 2, 2, 16))
    r = t[..., :3, :3]
    torch.testing.assert_close(
        r.transpose(-1, -2) @ r, torch.eye(3).expand_as(r), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        torch.linalg.det(r), torch.ones(2, 2), atol=1e-5, rtol=1e-5
    )


def test_volume_render_preserves_background_and_reports_missing_hits():
    sigma = torch.full((1, 1, 4), torch.log(torch.tensor(2.0)).item())
    color = torch.ones(1, 1, 4, 3)
    distances = torch.tensor([[[0.125, 0.375, 0.625, 0.875]]])
    deltas = torch.full_like(sigma, 0.25)
    background = torch.zeros(1, 1, 3)
    out = volume_render(sigma, color, distances, deltas, background)
    torch.testing.assert_close(out["opacity"], torch.tensor([[0.5]]))
    torch.testing.assert_close(out["rgb"], torch.full((1, 1, 3), 0.5))
    assert out["hit"].all() and 0 < out["range"].item() < 1
    empty = volume_render(
        torch.zeros_like(sigma), color, distances, deltas, background + 0.4
    )
    torch.testing.assert_close(empty["rgb"], background + 0.4)
    assert not empty["hit"].any() and (empty["range"] == 0).all()
    assert torch.isfinite(empty["range"]).all()


def test_readout_chunking_matches_and_gradients_reach_content_and_controls():
    torch.manual_seed(12)
    config = cfg()
    readout = CanonicalReadout(config)
    state = populated_state(config)
    state = replace(state, c=state.c.requires_grad_(), d=state.d.requires_grad_())
    q = query()
    one = readout(state, q, ray_chunk_size=2)
    whole = readout(state, q, ray_chunk_size=16)
    for key in ("rgb", "range", "opacity", "points"):
        torch.testing.assert_close(one[key], whole[key], atol=1e-6, rtol=1e-5)
    assert one["rgb"].shape == (2, 7, 3)
    (one["rgb"].sum() + one["range"].sum()).backward()
    for value in (state.c, state.d):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_empty_slots_do_not_render_and_invalid_rays_are_safe():
    config = cfg()
    readout = CanonicalReadout(config).eval()
    state = ObjectState.initial(config, 2, time=1.0)
    q = query()
    out = readout(state, q)
    changed = replace(
        state, c=torch.randn_like(state.c) * 10, d=torch.randn_like(state.d) * 5
    )
    torch.testing.assert_close(out["rgb"], readout(changed, q)["rgb"])
    invalid = replace(
        q,
        origins=torch.full_like(q.origins, float("nan")),
        valid=torch.zeros(2, 7, dtype=torch.bool),
    )
    result = readout(state, invalid)
    assert torch.isfinite(result["rgb"]).all() and not result["hit"].any()
    assert (result["rgb"] == 0).all()


def test_query_camera_rays_and_timestamp_contract():
    config = cfg()
    k = torch.tensor([[[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]])
    pose = torch.eye(4)[None]
    q = make_ray_query(
        k,
        pose,
        torch.tensor([[[4.0, 4.0]]]),
        time=torch.tensor([1.0], dtype=torch.float64),
        near=0.1,
        far=4.0,
    )
    torch.testing.assert_close(q.directions, torch.tensor([[[0.0, 0.0, 1.0]]]))
    state = ObjectState.initial(config, 1, time=0.0)
    with pytest.raises(ValueError, match="time|timestamp"):
        CanonicalReadout(config)(state, q)


def test_endpoint_points_use_pixel_centers_and_float32_geometry_under_autocast():
    obs = observation()
    encoder = SparseObservationEncoder(cfg()).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        evidence = encoder(obs)
    # Array index (2, 2) is the center (2.5, 2.5), camera-z is 2 m.
    expected = torch.tensor([-0.375, -0.375, 2.0])
    torch.testing.assert_close(evidence.points[:, 0], expected.expand(2, -1))
    assert evidence.points.dtype == evidence.ray_directions.dtype == torch.float32


def test_bfloat16_full_filter_read_and_backward_preserve_state_and_metric_precision():
    torch.manual_seed(17)
    config = cfg()
    model = ObjectStateModel(config)
    obs = observation()
    initial = model.initial_state(2, time=0.0)
    q = replace(query(), time=obs.time)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        state = model(initial, obs)
        rendered = model.read(state, q)
        loss = rendered["rgb"].square().mean() + rendered["range"].mean()
    loss.backward()
    state.validate(config)
    assert rendered["controls"].dtype == rendered["range"].dtype == torch.float32
    for name in ("observer.patchify.weight", "decoder.control_head.0.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert (
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )


def test_activation_checkpoint_matches_outputs_and_gradients():
    torch.manual_seed(31)
    direct = CanonicalReadout(replace(cfg(), decoder_checkpoint=False))
    checkpointed = CanonicalReadout(replace(cfg(), decoder_checkpoint=True))
    checkpointed.load_state_dict(direct.state_dict())
    base = populated_state(cfg())
    first = replace(
        base, c=base.c.clone().requires_grad_(), d=base.d.clone().requires_grad_()
    )
    second = replace(
        base, c=base.c.clone().requires_grad_(), d=base.d.clone().requires_grad_()
    )
    q = query()
    outputs = [
        module(state, q, ray_chunk_size=3)
        for module, state in ((direct, first), (checkpointed, second))
    ]
    for name in ("rgb", "range", "opacity"):
        torch.testing.assert_close(outputs[0][name], outputs[1][name])
    for output in outputs:
        (output["rgb"].sum() + output["range"].sum()).backward()
    torch.testing.assert_close(first.c.grad, second.c.grad)
    torch.testing.assert_close(first.d.grad, second.d.grad)
    for left, right in zip(direct.parameters(), checkpointed.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad)
