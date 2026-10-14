"""check_split.py: grouping keys (source video / image content / 3D asset) and leak counts on a tiny synthetic manifest."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_split", ROOT / "scripts" / "data" / "check_split.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)  # type: ignore[union-attr]

CLIPS = [
    "ECfF08Vn_5w_14_0to274.mp4", "ECfF08Vn_5w_15_0to100.mp4",   # ytid with "_" -> one source, 2 scenes
    "celebv_ECfF08Vn_5w_3.mp4",                                  # same YouTube video via CelebV-HQ
    "celebv_ECfF08Vn_5w_59_0.mp4",                               # CelebV with two trailing numbers
    "magictime_vGOR2_vlq9M_0001.mp4", "magictime_vGOR2_vlq9M_0022_0003.mp4",  # MagicTime, YouTube source
    "magictime_7136539298299530527_0001.mp4",                    # MagicTime, TikTok source
    "---_iRTHryQ_13_0to241.mp4",                                 # ytid starting with "-"
    "pixabay_Spring_Blossoms_39162_002.mp4", "pixabay_flowers_videos_39162_005.mp4",  # one stock video, 2 categories
    "pexels_landscape_landscape_39162_000.mp4",                  # same number, other site: separate source, flagged
    "mixkit-unknown-layout.mp4",                                 # unrecognised -> counted, never silently grouped
]


def _write_jsonl(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_video_source_keys():
    keys = [cs.video_source(c)[0] for c in CLIPS]
    assert keys == ["yt:ECfF08Vn_5w"] * 4 + ["yt:vGOR2_vlq9M"] * 2 + [
        "tt:7136539298299530527", "yt:---_iRTHryQ", "pixabay:39162", "pixabay:39162", "pexels:39162", None]
    assert cs.video_source("magictime_✨How to Grow.mp4")[0] is None


def test_video_keys_notes(tmp_path):
    m = tmp_path / "ov.jsonl"
    _write_jsonl(m, [{"path": f"/v/part0/{c}"} for c in CLIPS])
    keys, notes = cs.video_keys(str(m))
    assert len(set(keys)) == 7
    assert notes["unparsed_names"] == 1 and notes["unparsed_examples"] == ["mixkit-unknown-layout.mp4"]
    assert notes["yt_ids_in_both_celebv_and_panda"] == 1
    assert notes["stock_ids_under_two_sites"] == 1


def test_leak_table_exact_counts():
    keys = ["a", "a", "a", "b", "c", "c", "d"]
    assign = ["train", "val", "val", "val", "train", "test", "test"]
    t = cs.leak_table(keys, assign)
    assert t["val"] == {"items": 3, "groups": 2, "leaked_items": 2, "leaked_groups": 1}
    assert t["test"] == {"items": 2, "groups": 2, "leaked_items": 1, "leaked_groups": 1}


def test_simulate_split_matches_datamodule_sizes():
    a = cs.simulate_split(1000, 0.002, 0.0, seed=0)
    assert a.count("val") == 2 and a.count("train") == 998 and "test" not in a
    b = cs.simulate_split(1000, 0.05, 0.1, seed=0)
    assert (b.count("train"), b.count("val"), b.count("test")) == (850, 50, 100)
    assert a == cs.simulate_split(1000, 0.002, 0.0, seed=0) != cs.simulate_split(1000, 0.002, 0.0, seed=1)
    assert cs.simulate_split(10, 0.002, 0.0, seed=0).count("val") == 1          # max(1, int(n * val))


def test_image_content_groups_two_stage(tmp_path):
    head = bytes(range(256)) * 512                      # 128 KiB: identical first 64 KiB for all below
    files = {"a": head + b"x", "b": head + b"x",       # exact duplicates
             "c": head + b"y",                          # same size and head as a/b, differs at the end
             "d": b"tiny"}
    recs = []
    for name, data in files.items():
        p = tmp_path / f"{name}.jpg"; p.write_bytes(data)
        recs.append({"path": str(p), "bytes": len(data)})
    recs.append({"path": str(tmp_path / "a.jpg"), "bytes": len(files["a"])})   # same path listed twice
    m = tmp_path / "oi.jsonl"; _write_jsonl(m, recs)
    keys, notes = cs.image_keys(str(m), workers=2)
    ka, kb, kc, kd, ka2 = keys
    assert ka == kb == ka2 and len({ka, kc, kd}) == 3
    assert notes == {"duplicate_paths": 1, "duplicate_ids": 1}


def test_threed_duplicate_asset_under_two_uids(tmp_path):
    renders = tmp_path / "renders"
    for uid, colour in [("u1", b"red"), ("u2", b"red"), ("u3", b"blue")]:
        d = renders / uid; d.mkdir(parents=True)
        for p in cs.PLANES:
            (d / f"{p}.png").write_bytes(colour + p.encode())
    (renders / "u3" / "front.glb").write_bytes(b"leftover")
    (renders / "u4").mkdir(); (renders / "u4" / "oxoy.png").write_bytes(b"red")     # incomplete -> not a sample
    keys, notes = cs.threed_keys(str(renders), workers=2)
    assert len(keys) == 3 and keys[0] == keys[1] != keys[2]
    assert notes == {"dirs_with_files_besides_planes": 1}


def test_main_prints_table_and_json(tmp_path, capsys):
    m = tmp_path / "ov.jsonl"
    _write_jsonl(m, [{"path": f"/v/{c}"} for c in CLIPS])
    out = tmp_path / "s.json"
    assert cs.main(["video", str(m), "--val-split", "0.25", "--seeds", "0", "1", "--json", str(out)]) == 0
    assert "| val |" in capsys.readouterr().out
    s = json.loads(out.read_text())
    assert s["items"] == 12 and s["groups"] == 7 and s["items_in_multi_member_groups"] == 8
    assert s["held_out"]["val"]["items"] == [3, 3.0, 3]


def test_group_hash_split_has_no_video_leak(tmp_path, capsys):
    m = tmp_path / "ov.jsonl"
    _write_jsonl(m, [{"path": f"/v/{c}"} for c in CLIPS])
    out = tmp_path / "h.json"
    assert cs.main(["video", str(m), "--split-mode", "group_hash", "--val-split", "0.5", "--json", str(out)]) == 0
    assert "group_hash split" in capsys.readouterr().out
    s = json.loads(out.read_text())
    assert s["seeds"] == [] and s["salt"] == "mavt-split-v1"
    val = s["held_out"]["val"]
    assert val["items"][0] > 0 and val["leaked_items"] == [0, 0.0, 0] and val["leaked_groups"] == [0, 0.0, 0]
