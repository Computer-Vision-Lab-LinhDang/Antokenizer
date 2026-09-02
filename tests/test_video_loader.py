"""ManifestVideoDataset v2: streaming PyAV decode, aspect-preserving crop, contiguous stride sampling."""
from __future__ import annotations
import json
import numpy as np
import pytest
import torch


def _write_video(path, n_frames=40, w=96, h=32, fps=24):
    """Left third red, middle green, right third blue; frame index encoded in green channel of row 0."""
    import av
    with av.open(str(path), mode="w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "0", "preset": "ultrafast"}  # lossless-ish so colours survive
        for i in range(n_frames):
            img = np.zeros((h, w, 3), np.uint8)
            img[:, : w // 3, 0] = 255
            img[:, w // 3: 2 * w // 3, 1] = 255
            img[:, 2 * w // 3:, 2] = 255
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture()
def video_manifest(tmp_path):
    vp = tmp_path / "rgb.mp4"
    _write_video(vp)
    m = tmp_path / "videos.jsonl"
    m.write_text(json.dumps({"path": str(vp), "caption": "rgb bars"}) + "\n")
    return m


def test_video_clip_shape_and_range(video_manifest):
    from mavt.data.datasets import ManifestVideoDataset
    ds = ManifestVideoDataset(str(video_manifest), n_frames=8, resolution=16)
    s = ds[0]
    assert s["data"].shape == (3, 8, 16, 16)
    assert s["modality"] == "video" and s["caption"] == "rgb bars" and s["id"] == "rgb"
    assert -1.0 <= s["data"].min() and s["data"].max() <= 1.0


def test_video_crop_preserves_aspect_ratio(video_manifest):
    """A 96x32 frame resized to 16 short side becomes 48x16; the centre 16x16 crop is the GREEN bar.
    The old behaviour (anisotropic squash to 16x16) would keep red and blue thirds."""
    from mavt.data.datasets import ManifestVideoDataset
    ds = ManifestVideoDataset(str(video_manifest), n_frames=4, resolution=16)
    clip = ds[0]["data"]                           # (3, T, 16, 16) in [-1, 1]
    rgb = (clip + 1) / 2
    assert rgb[1].mean() > 0.9, "centre crop should be almost entirely green"
    assert rgb[0].mean() < 0.1 and rgb[2].mean() < 0.1, "red/blue thirds must be cropped away, not squashed in"


def test_video_frames_are_contiguous_with_stride(video_manifest):
    """Frames must be a contiguous window (stride `frame_stride`), not spread over the whole clip."""
    from mavt.data.datasets import ManifestVideoDataset
    ds = ManifestVideoDataset(str(video_manifest), n_frames=8, resolution=16, frame_stride=2)
    idxs = ds._sample_indices(n_total=40, rng=torch.Generator().manual_seed(0))
    assert len(idxs) == 8
    assert all(b - a == 2 for a, b in zip(idxs, idxs[1:])), idxs
    assert idxs[0] >= 0 and idxs[-1] < 40


def test_video_short_clip_falls_back_to_smaller_stride_then_pads():
    from mavt.data.datasets import ManifestVideoDataset
    ds = ManifestVideoDataset.__new__(ManifestVideoDataset)
    ds.n_frames, ds.frame_stride = 8, 4
    g = torch.Generator().manual_seed(0)
    idxs = ds._sample_indices(n_total=20, rng=g)     # 8 frames @ stride 4 needs 29 frames -> stride 2 fits (15)
    assert len(idxs) == 8 and all(b - a == 2 for a, b in zip(idxs, idxs[1:]))
    idxs = ds._sample_indices(n_total=5, rng=g)      # shorter than n_frames -> all frames, padded later
    assert idxs == [0, 1, 2, 3, 4]


def test_video_loader_does_not_materialise_whole_clip(video_manifest, monkeypatch):
    """Streaming decode: torchvision.io.read_video must not be used (it decodes every frame at full res)."""
    import torchvision.io as tvio
    from mavt.data.datasets import ManifestVideoDataset

    def boom(*a, **k):
        raise AssertionError("read_video (full-clip decode) must not be called")
    monkeypatch.setattr(tvio, "read_video", boom, raising=False)
    ds = ManifestVideoDataset(str(video_manifest), n_frames=4, resolution=16)
    assert ds[0]["data"].shape == (3, 4, 16, 16)


def test_video_decode_is_single_threaded(video_manifest):
    """libav frame threading deadlocked inside forked DataLoader workers (2026-09-02)."""
    import av
    from mavt.data.datasets import ManifestVideoDataset
    ds = ManifestVideoDataset(str(video_manifest), n_frames=4, resolution=16)
    path = ds.records[0]["path"]
    with av.open(path) as c:
        st = c.streams.video[0]
        ds._configure_stream(st)
        assert st.thread_count == 1 and st.thread_type != "AUTO"
    assert ds[0]["data"].shape == (3, 4, 16, 16)         # decode still works


def test_loader_timeout_hparam_reaches_dataloader(tmp_path):
    from mavt.data.datamodule import MAVTDataModule
    dm = MAVTDataModule(active_modalities=["image"], num_workers=2, loader_timeout=123, batch_size=2)
    extras = dm._loader_extras()
    assert extras.get("timeout") == 123
    dm0 = MAVTDataModule(active_modalities=["image"], num_workers=0, loader_timeout=123, batch_size=2)
    assert "timeout" not in dm0._loader_extras()
