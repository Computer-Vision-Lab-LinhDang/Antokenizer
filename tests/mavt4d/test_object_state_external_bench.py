"""External interfaces permit one image and planned cameras, never target evidence."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mavt4d.object_state.external_bench import (
    BenchmarkInput,
    prepare_input,
    render_first_frame,
    camera_sequence,
    assumed_wrbench_input,
    read_memo_cameras,
)


def example():
    cameras = torch.eye(4).repeat(3, 1, 1)
    cameras[:, 0, 3] = torch.tensor([10.0, 11.0, 12.0])
    return BenchmarkInput(
        torch.rand(3, 8, 12),
        torch.tensor([[12.0, 0.0, 6.0], [0.0, 12.0, 4.0], [0.0, 0.0, 1.0]]).repeat(
            3, 1, 1
        ),
        cameras,
        torch.tensor([0.0, 0.2, 0.4], dtype=torch.float64),
        benchmark="MemoBench",
        case_id="fixture",
        camera_provenance="synthetic_engine",
        calibrated=True,
        metric_geometry_valid=False,
        camera_units="unverified_native",
    )


def test_prepare_uses_only_first_camera_gauge_and_scales_intrinsics():
    raw = example()
    actual = prepare_input(raw, output_hw=(4, 6))
    assert actual.first_frame.shape == (3, 4, 6)
    assert torch.allclose(actual.camera_c2w[0], torch.eye(4))
    assert torch.equal(actual.camera_c2w[:, 0, 3], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(actual.intrinsics[:, :2], raw.intrinsics[:, :2] / 2)
    future_changed = raw.camera_c2w.clone()
    future_changed[1:, 0, 3] += 100
    other = prepare_input(replace(raw, camera_c2w=future_changed), output_hw=(4, 6))
    assert torch.equal(actual.camera_c2w[0], other.camera_c2w[0])
    assert torch.equal(actual.first_frame, other.first_frame)


def test_resizing_saturated_input_preserves_valid_rgb_range():
    raw = replace(example(), first_frame=torch.ones(3, 1080, 1920))
    resized = prepare_input(raw, output_hw=(128, 192))
    assert resized.first_frame.min() >= 0
    assert resized.first_frame.max() <= 1


def test_rejects_future_rgb_noncausal_time_and_metric_assumption():
    with pytest.raises(ValueError, match="first_frame"):
        prepare_input(replace(example(), first_frame=torch.rand(3, 3, 8, 12)))
    with pytest.raises(ValueError, match="times"):
        prepare_input(replace(example(), times=torch.tensor([0.0, 1.0, 0.5])))
    assumed = assumed_wrbench_input(
        torch.rand(3, 8, 12),
        torch.eye(4).repeat(3, 1, 1),
        torch.arange(3).double(),
        case_id="test",
        fov_degrees=65.0,
    )
    assert not assumed.calibrated and not assumed.metric_geometry_valid
    with pytest.raises(ValueError, match="metric"):
        prepare_input(replace(assumed, metric_geometry_valid=True))


def test_camera_resampling_retains_duration_and_readers_never_need_target_images(
    tmp_path,
):
    import numpy as np

    poses = np.tile(np.eye(4), (10, 1, 1))
    poses[:, 0, 3] = np.arange(10)
    intrinsic = np.tile([12.0, 13.0, 6.0, 4.0], (10, 1))
    np.save(tmp_path / "poses.npy", poses)
    np.save(tmp_path / "intrinsics.npy", intrinsic)
    k, c2w = read_memo_cameras(tmp_path, real=False)
    sampled_k, sampled_c2w, times, indices = camera_sequence(
        k, c2w, output_frames=4, source_fps=30.0
    )
    assert indices.tolist() == [0, 3, 6, 9]
    assert times[-1].item() == pytest.approx(0.3)
    assert sampled_k.shape == (4, 3, 3)
    assert sampled_c2w[:, 0, 3].tolist() == [0.0, 3.0, 6.0, 9.0]


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.observations, self.predictions, self.read_times = [], [], []

    def initial_state(self, batch_size, device, time):
        return SimpleNamespace(t=time, c=torch.ones(1), d=torch.zeros(1))

    def forward(self, state, observation):
        self.observations.append(observation)
        assert observation.depth is None and observation.depth_valid is None
        return state

    def predict(self, state, dt):
        self.predictions.append(dt.clone())
        return SimpleNamespace(t=state.t + dt, c=state.c, d=state.d + dt)

    def read(self, state, query):
        self.read_times.append(query.time.clone())
        assert torch.allclose(
            query.directions.norm(dim=-1),
            torch.ones_like(query.valid, dtype=torch.float32),
        )
        return {"rgb": torch.full_like(query.directions, 0.25)}


def test_render_corrects_once_then_predicts_and_streams_chunked_rays():
    model = RecordingModel()
    prepared = prepare_input(example(), output_hw=(4, 6))
    result = render_first_frame(model, prepared, near=0.1, far=30.0, ray_chunk_size=7)
    assert result["rgb"].shape == (3, 3, 4, 6)
    assert torch.equal(result["rgb"], torch.full_like(result["rgb"], 0.25))
    assert len(model.observations) == 1
    observation = model.observations[0]
    assert observation.images.shape == (1, 1, 1, 3, 4, 6)
    assert observation.geometry_valid.item()
    assert len(model.predictions) == 2
    assert all(x.item() == pytest.approx(0.2) for x in model.predictions)
    assert result["provenance"]["observed_frames"] == 1
    assert result["provenance"]["future_camera_conditioned_readout"] is True
    assert result["provenance"]["event_conditioning"] is False
    assert model.training  # adapter restores caller's mode


def test_uncalibrated_render_requires_explicit_assumed_ray_opt_in():
    raw = assumed_wrbench_input(
        torch.rand(3, 8, 12),
        torch.eye(4).repeat(3, 1, 1),
        torch.arange(3).double(),
        case_id="assumed",
        fov_degrees=65.0,
    )
    prepared = prepare_input(raw)
    with pytest.raises(ValueError, match="assumed"):
        render_first_frame(RecordingModel(), prepared, near=0.1, far=30.0)
    model = RecordingModel()
    result = render_first_frame(
        model, prepared, near=0.1, far=30.0, use_assumed_rays=True
    )
    assert model.observations[0].geometry_valid.item()
    assert result["provenance"]["ray_geometry_assumed"] is True
    assert result["provenance"]["calibrated"] is False
    assert result["provenance"]["metric_geometry_valid"] is False


def test_assumed_ray_opt_in_writes_actual_model_content():
    from mavt4d.object_state.config import ObjectStateConfig
    from mavt4d.object_state.model import ObjectStateModel

    config = ObjectStateConfig(
        slots=2,
        content_tokens=2,
        dynamic_tokens=2,
        width=16,
        background_tokens=2,
        evidence_width=16,
        spatial_blocks=4,
        spatial_heads=4,
        window_size=2,
        relay_tokens=2,
        resampler_heads=4,
        candidates=4,
        patch_size=4,
        samples_per_ray=4,
        query_chunk_size=32,
    )
    model = ObjectStateModel(config)
    captured = []
    handle = model.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    raw = assumed_wrbench_input(
        torch.rand(3, 4, 4),
        torch.eye(4)[None],
        torch.zeros(1).double(),
        case_id="content",
        fov_degrees=65.0,
    )
    result = render_first_frame(
        model, prepare_input(raw), near=0.1, far=30.0, use_assumed_rays=True
    )
    handle.remove()
    state = captured[0]
    assert state.c.abs().sum() + state.c_bg.abs().sum() > 0
    assert result["provenance"]["assumed_ray_content_write"] is True
