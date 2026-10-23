"""v1 data: random frame stride (with the short-video fallback), SSv2 mixed into the video stream,
collate keeping the stride, trainer flags. Synthetic decoders only: no real video is decoded."""
import collections
import importlib.util
import json
import os
import random

import numpy as np
import pytest
import torch

from mavt.dino_v0 import data as D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class _FakeDecoder:
    """Stands in for decode_clip_av: records (file, stride, total_hint); frames of files under a
    directory named ssv2 are white, others black."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, n_frames, stride, size, rng, total_hint=None, fps_hint=None):
        self.calls.append((os.path.basename(path), stride, total_hint))
        level = 255 if os.path.basename(os.path.dirname(path)) == "ssv2" else 0
        return np.full((n_frames, size, size + 2, 3), level, dtype=np.uint8)


def _manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_parse_strides():
    assert D.parse_strides(2) == (2,)
    assert D.parse_strides("4") == (4,)
    assert D.parse_strides("2,4,8") == (2, 4, 8)
    assert D.parse_strides("2, 4 8") == (2, 4, 8)
    assert D.parse_strides([2, 8]) == (2, 8)
    assert D.parse_strides(np.int64(3)) == (3,)
    for bad in ("", "0", [], "2,-1"):
        with pytest.raises(ValueError):
            D.parse_strides(bad)


def test_pick_stride_uniform_and_short_fallback():
    rng = random.Random(0)
    n = 6000
    long = collections.Counter(D.pick_stride((2, 4, 8), 8, 300, rng) for _ in range(n))
    assert set(long) == {2, 4, 8}
    assert all(abs(long[s] / n - 1 / 3) < 0.03 for s in (2, 4, 8)), long
    unknown = collections.Counter(D.pick_stride((2, 4, 8), 8, 0, rng) for _ in range(n))   # 0 = length unknown
    assert set(unknown) == {2, 4, 8}
    # SSv2-like 45 frames: span(8) = 57 does not fit -> those draws fall back to 4 (the largest that fits)
    short = collections.Counter(D.pick_stride((2, 4, 8), 8, 45, rng) for _ in range(n))
    assert set(short) == {2, 4}
    assert abs(short[4] / n - 2 / 3) < 0.03 and abs(short[2] / n - 1 / 3) < 0.03, short
    assert D.clip_span(8, 4) == 29 and D.clip_span(8, 8) == 57
    # shorter than every span: the smallest stride (the decoder pads, as with a fixed stride)
    assert {D.pick_stride((2, 4, 8), 8, 10, rng) for _ in range(50)} == {2}
    assert {D.pick_stride((4,), 8, 10, rng) for _ in range(20)} == {4}


def test_openvid_random_stride_train_fixed_eval(tmp_path):
    man = _manifest(tmp_path / "m.jsonl", [dict(path="long.mp4", n_frames=300, fps=24),
                                           dict(path="short.mp4", n_frames=40, fps=24)])
    dec = _FakeDecoder()
    ds = D.OpenVidClips(man, n_frames=8, frame_stride="2,4,8", size=16, train=True, decoder=dec)
    ds._rng = random.Random(0)
    assert ds.strides == (2, 4, 8) and ds.stride == 2
    by_item = collections.defaultdict(collections.Counter)
    for k in range(600):
        s = ds[k % 2]
        assert s["data"].shape == (3, 8, 16, 16) and s["modality"] == "video"
        assert s["stride"] == dec.calls[-1][1]                          # the sample reports the decoded stride
        by_item[k % 2][s["stride"]] += 1
    assert set(by_item[0]) == {2, 4, 8} and min(by_item[0].values()) > 70     # long video: ~uniform
    assert set(by_item[1]) == {2, 4}                                    # 40 frames: span(8) = 57 never fits
    assert by_item[1][4] > by_item[1][2]                                # stride-8 draws fall back to 4
    # eval: one fixed stride (first of the list unless eval_stride is given)
    ev = D.OpenVidClips(man, n_frames=8, frame_stride="2,4,8", size=16, train=False, decoder=dec)
    assert {ev[i % 2]["stride"] for i in range(10)} == {2}
    ev8 = D.OpenVidClips(man, n_frames=8, frame_stride=[2, 4, 8], size=16, train=False, decoder=dec,
                         eval_stride=8)
    assert {ev8[i % 2]["stride"] for i in range(10)} == {8}
    # a single int keeps the v0 behaviour: that stride, train or eval
    one = D.OpenVidClips(man, n_frames=8, frame_stride=4, size=16, train=True, decoder=dec)
    assert {one[i % 2]["stride"] for i in range(10)} == {4}


def test_stride_probe_without_manifest_frame_count(tmp_path):
    man = _manifest(tmp_path / "ssv2.jsonl", [dict(path="ssv2/a.webm"), dict(path="ssv2/b.webm", fps=12)])
    probed = collections.Counter()

    def probe(path, fps_hint=None):
        probed[os.path.basename(path)] += 1
        return 45                                                       # SSv2-like length from the headers

    dec = _FakeDecoder()
    ds = D.OpenVidClips(man, n_frames=8, frame_stride="2,4,8", size=8, train=True, decoder=dec, probe=probe)
    ds._rng = random.Random(1)
    strides = {ds[i % 2]["stride"] for i in range(200)}
    assert strides == {2, 4}
    assert probed == {"a.webm": 1, "b.webm": 1}                         # probed once per item, then cached
    assert {c[2] for c in dec.calls} == {45}                            # the probed count reaches the decoder
    # a failed probe (0 = unknown) draws from every stride and leaves the decoder to pad
    ds0 = D.OpenVidClips(man, n_frames=8, frame_stride="2,4,8", size=8, train=True, decoder=dec,
                         probe=lambda p, fps_hint=None: 0)
    ds0._rng = random.Random(2)
    assert {ds0[i % 2]["stride"] for i in range(200)} == {2, 4, 8}


def test_probe_frames_reads_headers(tmp_path):
    av = pytest.importorskip("av")
    p = tmp_path / "v.mp4"
    with av.open(str(p), "w") as c:
        st = c.add_stream("mpeg4", rate=24)
        st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
        for i in range(30):
            for pk in st.encode(av.VideoFrame.from_ndarray(np.full((32, 32, 3), i, np.uint8), format="rgb24")):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)
    assert abs(D.probe_frames(str(p)) - 30) <= 1
    (tmp_path / "broken.webm").write_bytes(b"not a video")
    assert D.probe_frames(str(tmp_path / "broken.webm")) == 0
    assert D.probe_frames(str(tmp_path / "missing.webm")) == 0


def test_collate_keeps_stride():
    batch = [dict(data=torch.zeros(3, 4, 8, 8), modality="video", stride=s) for s in (2, 8, 4)]
    out = D.collate(batch)
    assert out["data"].shape == (3, 3, 4, 8, 8) and out["modality"] == "video"
    assert torch.is_tensor(out["stride"]) and out["stride"].dtype == torch.long
    assert out["stride"].tolist() == [2, 8, 4]
    assert "stride" not in D.collate([dict(data=torch.zeros(3, 8, 8), modality="image", label=1)])


def test_mixture_sampler_weights_and_rank_shards():
    s = D.MixtureSampler([1000, 200], [0.7, 0.3], seed=0, num_samples=6000)
    idx = list(s)
    assert len(idx) == len(s) == 6000 and all(0 <= i < 1200 for i in idx)
    frac = sum(i >= 1000 for i in idx) / len(idx)
    assert abs(frac - 0.3) < 0.02, frac
    # the small source is cycled (reshuffled each pass); every item of it is used
    assert len({i for i in idx if i >= 1000}) == 200
    # ranks: disjoint shards of each source on each pass, together covering the source
    r0, r1 = (D.MixtureSampler([100, 50], [0.5, 0.5], num_replicas=2, rank=r, seed=3) for r in (0, 1))
    for j in (0, 1):
        for p in (0, 1):
            a, b = set(r0.shard(j, p)), set(r1.shard(j, p))
            assert not (a & b) and len(a | b) == (100, 50)[j]
    assert r0.shard(0, 0) != r0.shard(0, 1)                             # new pass, new order
    # within one pass of each source the ranks never read the same item
    q0, q1 = (D.MixtureSampler([100, 100], [0.5, 0.5], num_replicas=2, rank=r, seed=3, num_samples=40)
              for r in (0, 1))
    assert len(q0) == 40 and not (set(q0) & set(q1))
    e0 = list(q0)
    q0.set_epoch(1)
    assert list(q0) != e0
    assert len(D.MixtureSampler([10, 20], [1, 1], num_replicas=3)) == 10
    with pytest.raises(ValueError):
        D.MixtureSampler([10], [0.5, 0.5])


def test_mixed_loaders_mixes_ssv2(tmp_path, monkeypatch):
    dec = _FakeDecoder()
    monkeypatch.setattr(D, "_pick_decoder", lambda: dec)
    ov = _manifest(tmp_path / "openvid.jsonl", [dict(path=f"ov{i}.mp4", n_frames=300, fps=24) for i in range(40)])
    ss = _manifest(tmp_path / "ssv2.jsonl", [dict(path=f"ssv2/{i}.webm", n_frames=45, fps=12, label="Pushing [x]",
                                                  template="Pushing something") for i in range(10)])
    lds = D.mixed_loaders(openvid=ov, ssv2=ss, ssv2_weight=0.3, batch_video=4, size=8, n_frames=8,
                          frame_stride="2,4,8", video_tail=0, num_workers=0, seed=0)
    loader, smp = lds["video"]
    assert isinstance(smp, D.MixtureSampler) and len(loader.dataset) == 50
    it = D.infinite(loader, smp)
    n_ssv2, n, strides = 0, 0, {True: collections.Counter(), False: collections.Counter()}
    for _ in range(150):
        b = next(it)
        assert b["data"].shape == (4, 3, 8, 8, 8) and b["modality"] == "video" and "label" not in b
        is_ssv2 = b["data"].flatten(1).mean(1) > 0                      # white frames come from SSv2
        for f, s in zip(is_ssv2.tolist(), b["stride"].tolist()):
            strides[f][s] += 1
        n_ssv2 += int(is_ssv2.sum())
        n += len(is_ssv2)
    assert abs(n_ssv2 / n - 0.3) < 0.06, n_ssv2 / n
    assert set(strides[True]) == {2, 4}                                 # 45-frame SSv2: never stride 8
    assert set(strides[False]) == {2, 4, 8}
    # weights 0 / 1 and SSv2 alone collapse to a single source with the plain DistributedSampler
    for kw, n_items in ((dict(openvid=ov, ssv2=ss, ssv2_weight=0.0), 40), (dict(openvid=ov, ssv2=ss, ssv2_weight=1.0), 10),
                        (dict(ssv2=ss), 10), (dict(openvid=ov), 40)):
        ld, sp = D.mixed_loaders(**kw, batch_video=2, size=8, n_frames=4, video_tail=0, num_workers=0)["video"]
        assert isinstance(sp, torch.utils.data.DistributedSampler) and len(ld.dataset) == n_items


class _RandDS(torch.utils.data.Dataset):
    def __init__(self):
        self._rng = random.Random(0)

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.tensor(self._rng.random())


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_worker_init_reseeds_mixture_children():
    """Forked workers copy the parent's _rng: _worker_init must reseed the datasets of a ConcatDataset."""
    ds = torch.utils.data.ConcatDataset([_RandDS(), _RandDS()])
    ld = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, worker_init_fn=D._worker_init)
    b0, b1 = list(ld)[:2]                                               # worker 0, worker 1
    assert not torch.equal(b0, b1)


def _trainer():
    spec = importlib.util.spec_from_file_location("train_dino_v0", os.path.join(ROOT, "scripts", "train_dino_v0.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trainer_flags_reach_mixed_loaders(tmp_path, monkeypatch):
    tr = _trainer()
    args = tr.build_parser().parse_args(["--out", str(tmp_path), "--openvid", "ov.jsonl", "--frame-stride", "2,4,8",
                                         "--ssv2-train", "ssv2.jsonl", "--ssv2-weight", "0.25"])
    seen = {}

    def fake_loaders(*a, **kw):
        seen.update(kw, positional=a)
        return {}

    monkeypatch.setattr(D, "mixed_loaders", fake_loaders)
    assert tr.build_data(args, tr.tiny_config(), 0, 1, 0) == {}
    assert D.parse_strides(seen["frame_stride"]) == (2, 4, 8)
    assert seen["ssv2"] == "ssv2.jsonl" and seen["ssv2_weight"] == 0.25 and seen["positional"][1] == "ov.jsonl"
    d = tr.build_parser().parse_args(["--out", str(tmp_path)])
    assert D.parse_strides(d.frame_stride) == (2,) and d.ssv2_train is None and d.ssv2_weight == 0.3
