"""Trainer dry-run (CPU, tiny config, synthetic data) and the v0 data loaders on tiny synthetic files."""
import importlib.util
import io
import json
import os
import random
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _trainer():
    spec = importlib.util.spec_from_file_location("train_dino_v0", os.path.join(ROOT, "scripts", "train_dino_v0.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_writes_ckpt_and_log(tmp_path):
    tr = _trainer()
    out = tmp_path / "run"
    assert tr.main(["--dry-run", "--out", str(out)]) == 0
    assert (out / "last.pt").exists() and (out / "log.jsonl").exists()
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [1, 2, 3]
    assert [r["modality"] for r in recs] == ["image", "video", "threed"]
    assert all(np.isfinite(r["loss"]) and "lr" in r and "it_s" in r for r in recs)
    assert "loss_fut" in recs[1] and "loss_rgb" in recs[0] and "loss_rgb" in recs[2]
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3 and {"model", "optim", "sched", "cfg"} <= set(ck)
    assert not any(k.startswith("encoder.dino.model.") for k in ck["model"])   # frozen DINO not saved
    # auto-resume picks up last.pt and continues to a larger step count
    assert tr.main(["--dry-run", "--out", str(out)]) == 0        # already at 3 steps: no-op
    assert len((out / "log.jsonl").read_text().splitlines()) == 3


def test_yaml_config_parses(tmp_path):
    tr = _trainer()
    args, mcfg = tr.parse_args(["--config", os.path.join(ROOT, "configs", "dino_v0", "v0_alpha.yaml"),
                                "--out", str(tmp_path), "--lr", "1e-4"])
    cfg = tr.make_config(mcfg, dry=False)
    assert cfg.image_size == 256 and cfg.keyframe_every == 4 and cfg.train_queries_per_frame == 64
    assert cfg.dino_repo.endswith("dinov3-vitl16")
    assert args.lr == 1e-4 and args.bf16 and args.objaverse and args.video_tail == 1000
    assert tr.parse_weights(args.modality_weights) == dict(image=0.4, video=0.4, threed=0.2)


# ---------------------------------------------------------------------------- data
def _jpeg(w, h, seed):
    from PIL import Image
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_parquet(root, n_files=2, rows=10, row_group=4):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    d = root / "data"
    d.mkdir(parents=True)
    k = 0
    for f in range(n_files):
        imgs, labels = [], []
        for _ in range(rows):
            imgs.append({"bytes": _jpeg(80 + 7 * k % 30, 60 + 3 * k % 20, k), "path": None})
            labels.append(k % 5)
            k += 1
        t = pa.table({"image": pa.array(imgs, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                      "label": pa.array(labels, type=pa.int64())})
        pq.write_table(t, d / f"train-{f:05d}-of-{n_files:05d}.parquet", row_group_size=row_group)
    return k


def test_imagenet_parquet(tmp_path):
    pytest.importorskip("PIL")
    n = _write_parquet(tmp_path)
    from mavt.dino_v0.data import ImageNetParquet, ParquetGroupBatchSampler, collate
    ds = ImageNetParquet(str(tmp_path), "train", size=32)
    assert len(ds) == n == 20 and len(ds.groups) == 6
    s = ds[13]
    assert s["data"].shape == (3, 32, 32) and s["modality"] == "image" and s["label"] == 13 % 5
    assert -1 <= s["data"].min() and s["data"].max() <= 1
    ev = ImageNetParquet(str(tmp_path), "train", size=32, train=False)
    assert torch.equal(ev[3]["data"], ev[3]["data"])
    # batch sampler: rank shards are disjoint, every batch stays within one worker's row groups
    seen = []
    for rank in range(2):
        bs = ParquetGroupBatchSampler(ds, batch_size=2, num_workers=2, rank=rank, world=2, seed=0, window=2)
        batches = list(bs)
        assert batches and all(len(b) == 2 for b in batches)
        seen.append({i for b in batches for i in b})
    assert not (seen[0] & seen[1])
    b = collate([ds[i] for i in (0, 1, 2)])
    assert b["data"].shape == (3, 3, 32, 32) and b["label"].tolist() == [0, 1, 2]


def _write_video(path, n=30, w=96, h=64, fps=24):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.codec_context.gop_size = 8
        for i in range(n):
            arr = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)       # frame index encoded in brightness
            for p in st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def test_openvid_clips(tmp_path, caplog):
    pytest.importorskip("av")
    vd = tmp_path / "videos"
    vd.mkdir()
    _write_video(vd / "a.mp4")
    _write_video(tmp_path / "b.mp4", n=5)                               # shorter than the span: padded
    (vd / "broken.mp4").write_bytes(b"not a video")
    man = tmp_path / "manifest.jsonl"
    rows = [dict(path="videos/a.mp4", n_frames=30, fps=24, width=96, height=64, caption="x"),
            dict(path=str(tmp_path / "b.mp4"), n_frames=5, fps=24, width=96, height=64, caption="y"),
            dict(path="videos/broken.mp4", n_frames=30, fps=24, width=96, height=64, caption="z")]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n{\"path\": \"partial")   # growing file
    from mavt.dino_v0.data import OpenVidClips, decode_clip_av
    ds = OpenVidClips(str(man), n_frames=4, frame_stride=2, size=32, train=True)
    assert len(ds) == 3
    for i in range(3):
        s = ds[i]
        assert s["data"].shape == (3, 4, 32, 32) and s["modality"] == "video"
        assert -1 <= s["data"].min() and s["data"].max() <= 1
    # frames are `stride` apart: brightness steps by 2 * 8 levels between kept frames
    clip = decode_clip_av(str(vd / "a.mp4"), 4, 2, 32, random.Random(1), total_hint=30, fps_hint=24)
    v = clip.reshape(4, -1).mean(1)
    assert clip.shape[0] == 4 and clip.shape[1] == 32
    assert np.allclose(np.diff(v), 16, atol=4), v


def test_mixed_loaders(tmp_path):
    pytest.importorskip("av")
    _write_parquet(tmp_path / "in")
    _write_video(tmp_path / "a.mp4")
    man = tmp_path / "m.jsonl"
    man.write_text(json.dumps(dict(path="a.mp4", n_frames=30, fps=24)) + "\n")
    from mavt.dino_v0.data import infinite, mixed_loaders
    _write_objaverse(tmp_path / "obj", n_obj=4)
    lds = mixed_loaders(str(tmp_path / "in"), str(man), str(tmp_path / "obj"), batch_image=4, batch_video=1,
                        batch_3d=2, size=32, n_frames=4, n_views=3, video_tail=0, objaverse_holdout=1,
                        num_workers=0)
    it = infinite(*lds["image"])
    for _ in range(7):                                                  # crosses an epoch boundary
        b = next(it)
        assert b["data"].shape == (4, 3, 32, 32) and b["modality"] == "image"
    vb = next(infinite(*lds["video"]))
    assert vb["data"].shape == (1, 3, 4, 32, 32)
    tb = next(infinite(*lds["threed"]))
    assert tb["data"].shape == (2, 3, 3, 32, 32) and tb["c2w"].shape == (2, 3, 4, 4) and tb["K"].shape == (2, 3, 3, 3)
    assert tb["modality"] == "threed" and isinstance(tb["label"], list)


def test_openvid_tail_holdout(tmp_path):
    from mavt.dino_v0.data import OpenVidClips
    man = tmp_path / "m.jsonl"
    man.write_text("".join(json.dumps(dict(path=f"v{i}.mp4")) + "\n" for i in range(10)))
    tr = OpenVidClips(str(man), tail=3, decoder=lambda *a, **k: None)
    ho = OpenVidClips(str(man), tail=3, train=False, decoder=lambda *a, **k: None)
    assert len(tr) == 7 and len(ho) == 3 and ho.items[0]["path"].endswith("v7.mp4")
    assert len(OpenVidClips(str(tmp_path), tail=0, decoder=lambda *a, **k: None)) == 10   # dir -> *.jsonl


# ---------------------------------------------------------------------------- objaverse
def _write_objaverse(root, n_obj=5, n_views=6, res=64):
    from PIL import Image
    import math
    (root / "renders").mkdir(parents=True)
    rows = []
    for k in range(n_obj):
        uid = f"uid{k:03d}"
        blank = k == 1
        rows.append(dict(uid=uid, category=f"cat{k % 2}", n_views=n_views, blank=blank, alpha_mean=0.2))
        d = root / "renders" / uid
        d.mkdir()
        for v in range(n_views):
            a = np.zeros((res, res, 4), np.uint8)
            a[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = (200, 30, 30, 255)   # opaque red square
            Image.fromarray(a, "RGBA").save(d / f"{v:05d}.png")
            Image.fromarray(np.zeros((res, res), np.uint16)).save(d / f"{v:05d}_depth.png")
            ang = 2 * math.pi * v / n_views
            o = [2 * math.cos(ang), 2 * math.sin(ang), 0.0]
            z = [-o[0] / 2, -o[1] / 2, 0.0]
            x = [-z[1], z[0], 0.0]
            cam = dict(max_depth=5.0, origin=o, x_fov=0.6911, y_fov=0.6911, x=x, y=[0.0, 0.0, -1.0], z=z)
            (d / f"{v:05d}.json").write_text(json.dumps(cam))
    (root / "manifest.lvis_full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_objaverse_views(tmp_path):
    pytest.importorskip("PIL")
    import math
    _write_objaverse(tmp_path, n_obj=5)
    from mavt.dino_v0.data import ObjaverseViews
    tr = ObjaverseViews(str(tmp_path), n_views=4, size=32, train=True, holdout=2)
    ho = ObjaverseViews(str(tmp_path), n_views=4, size=32, train=False, holdout=2)
    assert [r["uid"] for r in tr.items] == ["uid000", "uid002"]           # blank uid001 skipped
    assert [r["uid"] for r in ho.items] == ["uid003", "uid004"]
    assert ObjaverseViews.heldout_uids(str(tmp_path), 2) == ["uid003", "uid004"]
    s = tr[0]
    assert s["data"].shape == (3, 4, 32, 32) and s["modality"] == "threed" and s["label"] == "cat0"
    assert s["c2w"].shape == (4, 4, 4) and s["K"].shape == (4, 3, 3)
    # distinct views; c2w columns = x, y, z, origin; rotation orthonormal
    origins = s["c2w"][:, :3, 3]
    assert len({tuple(o.round(decimals=4).tolist()) for o in origins}) == 4
    R = s["c2w"][:, :3, :3]
    assert torch.allclose(R.transpose(-1, -2) @ R, torch.eye(3).expand(4, 3, 3), atol=1e-3)
    f = 32 / 2 / math.tan(0.6911 / 2)                                     # K scaled from render res to size
    assert torch.allclose(s["K"][0], torch.tensor([[f, 0, 16.0], [0, f, 16.0], [0, 0, 1.0]]), atol=1e-3)
    # eval background is white, train background is one gray level per object
    e = ho[0]["data"]
    assert torch.allclose(e[:, :, 0, 0], torch.ones(3, 4))
    corners = s["data"][:, :, 0, 0]
    assert torch.allclose(corners, corners[0, 0].expand_as(corners))
    assert torch.equal(ho[1]["data"], ho[1]["data"])                       # eval is deterministic
