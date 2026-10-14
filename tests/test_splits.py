"""mavt.data.splits + MAVTDataModule's group_hash split: no group (source video / image id / 3D uid)
straddles train and val, and a sample's side is a pure function of (salt, group), so shuffling or
growing the manifest never moves it."""
from __future__ import annotations

import json
import random
import string
import warnings

import pytest

from mavt.data import splits as sp

_YT_CHARS = string.ascii_letters + string.digits + "_-"


def _clips(n_src: int = 400, per: int = 5, seed: int = 0) -> list:
    """Panda-style clip paths: ``per`` scene clips cut from each of ``n_src`` YouTube videos."""
    rng = random.Random(seed)
    ids = ["".join(rng.choice(_YT_CHARS) for _ in range(11)) for _ in range(n_src)]
    return [f"/v/part0/{yt}_{k}_0to{100 + k}.mp4" for yt in ids for k in range(per)]


def test_group_keys():
    assert (sp.group_key("video", "/v/ECfF08Vn_5w_14_0to274.mp4")
            == sp.group_key("video", "/x/celebv_ECfF08Vn_5w_3.mp4") == "yt:ECfF08Vn_5w")
    assert sp.group_key("video", "/v/mixkit-unknown-layout.mp4") == "clip:mixkit-unknown-layout"
    assert sp.group_key("image", "/oi/train/0a1b2c3d4e5f6a7b.jpg") == "img:0a1b2c3d4e5f6a7b"
    assert sp.group_key("threed", "renders/abc123/") == "obj:abc123"
    with pytest.raises(ValueError):
        sp.group_key("audio", "x.wav")


def test_no_group_straddles_splits():
    paths = _clips()
    keys = [sp.group_key("video", p) for p in paths]
    idx = sp.split_indices(keys, 0.1, 0.1)
    side = {}
    for split, members in idx.items():
        for i in members:
            assert side.setdefault(keys[i], split) == split
    assert sorted(i for members in idx.values() for i in members) == list(range(len(paths)))
    assert all(idx[s] for s in ("train", "val", "test"))


def test_assignment_survives_shuffle_and_growth():
    paths = _clips(seed=1)
    before = {p: sp.assign(sp.group_key("video", p), 0.05) for p in paths}
    grown = paths + _clips(seed=2)
    random.Random(3).shuffle(grown)
    idx = sp.split_indices([sp.group_key("video", p) for p in grown], 0.05)
    after = {grown[i]: split for split, members in idx.items() for i in members}
    assert all(after[p] == split for p, split in before.items())


def test_raising_val_only_moves_train_to_val():
    keys = [f"img:{i:016x}" for i in range(5000)]
    small, big = sp.split_indices(keys, 0.01)["val"], sp.split_indices(keys, 0.05)["val"]
    assert set(small) < set(big)
    assert sp.split_indices(keys, 0.05, 0.05)["val"] == big          # adding a test split keeps val


def test_fraction_is_of_groups_and_salted():
    keys = [f"obj:{i}" for i in range(20000)]
    val = sp.split_indices(keys, 0.02)["val"]
    assert 0.017 * len(keys) < len(val) < 0.023 * len(keys)
    assert sp.split_indices(keys, 0.02, salt="other-salt")["val"] != val
    one_group = sp.split_indices(["yt:same"] * 50, 0.5)
    assert sorted(len(v) for v in one_group.values()) == [0, 0, 50]  # a group never splits


def test_fraction_for_and_bad_fractions():
    assert sp.fraction_for("video", None, sp.DEFAULT_VAL) == 0.02
    assert sp.fraction_for("video", 0.1, sp.DEFAULT_VAL) == 0.1
    assert sp.fraction_for("threed", {"image": 0.2}, sp.DEFAULT_VAL) == 0.01
    with pytest.raises(ValueError):
        sp.assign("k", 0.6, 0.5)


def _video_dm(manifest, **kw):
    from mavt.data.datamodule import MAVTDataModule
    return MAVTDataModule(active_modalities=["video"], video_manifest=str(manifest), batch_size=2,
                          num_workers=0, **kw)


def test_datamodule_group_hash_split(tmp_path):
    paths = _clips(n_src=200, per=4)
    m = tmp_path / "ov.jsonl"
    m.write_text("".join(json.dumps({"path": p}) + "\n" for p in paths))
    dm = _video_dm(m, val_split={"video": 0.1})
    dm.setup("fit")
    train, val = list(dm._train_ds.indices), list(dm._val_ds.indices)
    group = lambda i: sp.group_key("video", paths[i])  # noqa: E731
    assert val and not ({group(i) for i in train} & {group(i) for i in val})
    assert sorted(train + val) == list(range(len(paths)))
    again = _video_dm(m, val_split={"video": 0.1})
    again.setup("fit")
    assert list(again._val_ds.indices) == val


def test_datamodule_empty_val_raises(tmp_path):
    m = tmp_path / "ov.jsonl"
    m.write_text(json.dumps({"path": "/v/ECfF08Vn_5w_14_0to274.mp4"}) + "\n")
    with pytest.raises(ValueError, match="raise val_split"):
        _video_dm(m, val_split=0.01).setup("fit")


def test_datamodule_synthetic_falls_back_quietly():
    from mavt.data.datamodule import MAVTDataModule
    dm = MAVTDataModule(active_modalities=["image"], batch_size=2, num_workers=0, synthetic_n=40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dm.setup("fit")
    assert not [w for w in caught if "exposes no sample ids" in str(w.message)]
    assert len(dm._val_ds) >= 1 and len(dm._train_ds) + len(dm._val_ds) == 40
