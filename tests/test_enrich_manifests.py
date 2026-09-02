"""enrich_manifests.py: scan meta (+ OpenVid csv) → training manifests, broken/short clips excluded."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("enrich_manifests", ROOT / "scripts" / "data" / "enrich_manifests.py")
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)  # type: ignore[union-attr]


def test_enrich_videos_joins_csv_and_drops_short_clips(tmp_path):
    csv_path = tmp_path / "ov.csv"
    csv_path.write_text(
        "video,caption,aesthetic score,motion score,temporal consistency score,camera motion,frame,fps,seconds\n"
        "a.mp4,cap a,5.1,3.2,0.99,static,100,24,4.1\n"
        "b.mp4,cap b,4.9,,0.98,pan_left,10,24,0.4\n"
        "zzz.mp4,unused,1,1,1,static,1,1,1\n")
    meta = [
        {"path": "/v/a.mp4", "caption": "cap a", "width": 512, "height": 512, "fps": 24.0, "frames": 100,
         "duration": 4.1, "codec": "h264", "bytes": 1000, "extra_scan_field": 1},
        {"path": "/v/b.mp4", "caption": "cap b", "width": 512, "height": 512, "fps": 24.0, "frames": 10,
         "duration": 0.4, "codec": "h264", "bytes": 100},
    ]
    csv_meta = em.load_openvid_csv(str(csv_path), {"a.mp4", "b.mp4"})
    assert "zzz.mp4" not in csv_meta and csv_meta["b.mp4"] == {"aesthetic": 4.9, "temporal_consistency": 0.98, "camera_motion": "pan_left"}
    out = em.enrich_videos(meta, csv_meta, min_frames=16)
    assert [r["path"] for r in out] == ["/v/a.mp4"], "b.mp4 (10 frames) must be dropped"
    a = out[0]
    assert a["motion"] == 3.2 and a["aesthetic"] == 5.1 and a["camera_motion"] == "static"
    assert "extra_scan_field" not in a and a["frames"] == 100 and a["caption"] == "cap a"


def test_enrich_images_keeps_only_known_keys_sorted():
    meta = [{"path": "/i/b.jpg", "width": 3, "height": 2, "mode": "L", "bytes": 9, "junk": 0},
            {"path": "/i/a.jpg", "width": 4, "height": 4, "mode": "RGB", "bytes": 8}]
    out = em.enrich_images(meta)
    assert [r["path"] for r in out] == ["/i/a.jpg", "/i/b.jpg"]
    assert out[1] == {"path": "/i/b.jpg", "width": 3, "height": 2, "mode": "L", "bytes": 9}


def test_cli_round_trip(tmp_path):
    import subprocess
    meta = tmp_path / "m.meta.jsonl"
    meta.write_text(json.dumps({"path": "/i/a.jpg", "width": 4, "height": 4, "mode": "RGB", "bytes": 8}) + "\n")
    out = tmp_path / "images_v2.jsonl"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/data/enrich_manifests.py"), "image", str(meta), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["path"] == "/i/a.jpg"
