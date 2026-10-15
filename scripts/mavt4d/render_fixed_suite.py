#!/usr/bin/env python3
"""Contact sheets for the E1.3 fixed suite: a visual and geometric check of the mined clips.

    PYTHONPATH=src python scripts/mavt4d/render_fixed_suite.py --dataroot data/datasets/nuscenes \
        --suite data/eval/fixed_suite_v2.jsonl --out data/eval/fixed_suite_v2_check

Picks ``--n`` clips deterministically (round robin over categories, bucket order within one), and for
each the camera where the primary key instance's centre projects in front of it and inside the image at
the reference frame: the event frame (short_event), the last keyframe (pairs), else the middle one. Draws the K keyframes in a grid with projected 3D boxes (key instances bold orange,
others thin grey): global -> ego (ego_pose of that camera sample_data) -> camera (calibrated_sensor)
-> intrinsics, via mavt4d.geometry.camera.project.

Checks written to check.json and index.html (self-contained, no external resources):
  * key centres in image at the reference frame (the spec's check; partly circular, since that is how
    the camera was chosen) and over all K frames;
  * devkit agreement: the same corner projection against nuscenes-devkit get_sample_data on v1.0-mini
    (an independent implementation of the transform chain).
Reads only the jsonl and the <suite stem>.cache.json written by build_fixed_suite.py.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mavt4d.eval import fixed_suite as fs
from mavt4d.geometry.camera import project

EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
KEY_COLOR, OTHER_COLOR = (255, 90, 0), (175, 175, 175)
MIN_DEPTH = 0.1  # m: an edge with an endpoint closer than this (or behind) is not drawn
COLS = 4


def pick_clips(rows: list[dict], n: int) -> list[dict]:
    by = {c: sorted((r for r in rows if r["category"] == c), key=lambda r: fs.bucket(f"render:{r['id']}"))
          for c in fs.CATEGORIES}
    out, i = [], 0
    while len(out) < n and any(len(v) > i for v in by.values()):
        out += [v[i] for v in by.values() if len(v) > i][: n - len(out)]
        i += 1
    return out


def camera(cache: dict, sd_tok: str) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """cam_to_world (4, 4), intrinsics (3, 3), both float64, and image (W, H) of one camera sample_data."""
    sd = cache["sample_data"][sd_tok]
    pose, cs = cache["ego_pose"][sd["ego_pose"]], cache["calibrated_sensor"][sd["cs"]]
    c2w = fs.pose_matrix(pose["translation"], pose["rotation"]) @ fs.pose_matrix(cs["translation"], cs["rotation"])
    return torch.from_numpy(c2w), torch.tensor(cs["camera_intrinsic"], dtype=torch.float64), sd["width"], sd["height"]


def centres_in_image(cache: dict, sd_tok: str, anns: list[dict]) -> tuple[int, float]:
    """(number of ``anns`` centres in front of the camera and inside the image, mean |u - W/2| / W of those)."""
    if not anns:
        return 0, 0.0
    c2w, K, W, H = camera(cache, sd_tok)
    uv, d = project(torch.tensor([a["xyz"] for a in anns], dtype=torch.float64), K, c2w)
    ok = (d > MIN_DEPTH) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    off = float((uv[ok, 0] - W / 2).abs().mean() / W) if ok.any() else 0.0
    return int(ok.sum()), off


def reference_frame(row: dict, by_id: dict[str, dict]) -> int | None:
    """Frame where the tested thing is visible: the primary instance's event frame (short_event), the last
    keyframe (pairs: the final position), None otherwise. A missing_modality row uses its base clip's."""
    src = by_id.get(row["criteria"].get("base_id"), row) if row["category"] == fs.MISSING else row
    if src["category"] == fs.EVENT:
        return src["criteria"]["instances"][0]["frame"]
    if src["category"] == fs.PAIR:
        return len(row["sample_tokens"]) - 1
    return None


def choose_camera(row: dict, cache: dict, ref: int | None = None) -> tuple[str, int, list[list[dict]]]:
    """(camera, reference frame, key-instance annotations per frame). Without ``ref`` the reference frame
    is the middle keyframe, or the nearest one where the primary key instance (the first, nearest) is
    annotated. The camera must see the primary centre first, then as many other key centres as possible;
    masked-out cameras are skipped."""
    key = set(row["instance_tokens"])
    primary = row["instance_tokens"][0] if row["instance_tokens"] else None
    k = len(row["sample_tokens"])
    key_at = [[a for a in cache["annotations"][s] if a["instance"] in key] for s in row["sample_tokens"]]
    if ref is None:
        ref = min((t for t in range(k) if any(a["instance"] == primary for a in key_at[t])),
                  key=lambda t: (abs(t - k // 2), t), default=k // 2)
    cams = [c for c in row["cameras"] if row["mask"] is None or row["mask"]["cameras"][c][ref]]

    def score(c: str):
        sd_tok = row["camera_sd_tokens"][c][ref]
        p, _ = centres_in_image(cache, sd_tok, [a for a in key_at[ref] if a["instance"] == primary])
        n, off = centres_in_image(cache, sd_tok, key_at[ref])
        return p, n, -off, -fs.CAMERAS.index(c)

    return max(cams, key=score), ref, key_at


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _caption(row: dict) -> str:
    crit = dict(row["criteria"])
    for long_key in ("instances", "moving"):
        if isinstance(crit.get(long_key), list) and len(crit[long_key]) > 2:
            crit[long_key] = crit[long_key][:2] + [f"... {len(crit[long_key]) - 2} more"]
    return json.dumps(crit, separators=(",", ":"))


def render_clip(row: dict, cache: dict, dataroot: Path, width: int, path: Path, ref: int | None = None) -> dict:
    cam, ref, key_at = choose_camera(row, cache, ref)
    key = set(row["instance_tokens"])
    n_ref, _ = centres_in_image(cache, row["camera_sd_tokens"][cam][ref], key_at[ref])
    primary = row["instance_tokens"][0] if row["instance_tokens"] else None
    p_ref, _ = centres_in_image(cache, row["camera_sd_tokens"][cam][ref],
                                [a for a in key_at[ref] if a["instance"] == primary])
    tiles, n_all, n_key_all = [], 0, 0
    font = _font(14)
    for t, (sample, sd_tok) in enumerate(zip(row["sample_tokens"], row["camera_sd_tokens"][cam])):
        c2w, K, W, H = camera(cache, sd_tok)
        img = Image.open(dataroot / cache["sample_data"][sd_tok]["filename"]).convert("RGB")
        s = width / W
        img = img.resize((width, round(H * s)), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        for a in sorted(cache["annotations"][sample], key=lambda a: a["instance"] in key):  # key boxes on top
            uv, d = project(torch.from_numpy(fs.box_corners(a["xyz"], a["size"], a["rotation"])), K, c2w)
            uv, d, bold = (uv * s).tolist(), d.tolist(), a["instance"] in key
            for i, j in EDGES:
                if d[i] > MIN_DEPTH and d[j] > MIN_DEPTH:
                    draw.line([tuple(uv[i]), tuple(uv[j])], fill=KEY_COLOR if bold else OTHER_COLOR, width=3 if bold else 1)
        n, _ = centres_in_image(cache, sd_tok, key_at[t])
        n_all, n_key_all = n_all + n, n_key_all + len(key_at[t])
        dt = (row["timestamps_us"][t] - row["timestamps_us"][0]) / 1e6
        draw.text((5, 4), f"t{t}  +{dt:.1f}s  key in image {n}/{len(key_at[t])}", fill=(255, 255, 0), font=font)
        tiles.append(img)
    th = tiles[0].height
    lines = [f"{row['category']}  |  {row['id']}  |  camera {cam}",
             f"key instances {len(key)}  |  primary in image at frame {ref}: {p_ref}  |  "
             f"key centres in image at frame {ref}: {n_ref}/{len(key_at[ref])}  |  "
             f"all {len(tiles)} frames: {n_all}/{n_key_all}"
             + (f"  |  mask variant: {row['criteria']['variant']}" if row["mask"] else "")]
    lines += textwrap.wrap(_caption(row), width=COLS * width // 8)[:4]
    head = 8 + 18 * len(lines)
    sheet = Image.new("RGB", (COLS * width, head + math.ceil(len(tiles) / COLS) * th), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, line in enumerate(lines):
        d.text((6, 4 + 18 * i), line, fill=(235, 235, 235), font=font)
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % COLS) * width, head + (i // COLS) * th))
    sheet.save(path, quality=88)
    return {"id": row["id"], "category": row["category"], "camera": cam, "ref_frame": ref, "file": path.name,
            "primary_in_image_ref": p_ref, "key_in_image_ref": n_ref, "key_at_ref": len(key_at[ref]),
            "key_in_image_all": n_all, "key_all": n_key_all, "caption": lines}


def devkit_check(dataroot: Path, n_samples: int = 4) -> dict:
    """Max |pixel difference| between our corner projection and nuscenes-devkit's on v1.0-mini keyframes."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points

    nusc = NuScenes("v1.0-mini", dataroot=str(dataroot), verbose=False)
    diffs = []
    for sample in nusc.sample[:: max(1, len(nusc.sample) // n_samples)][:n_samples]:
        for cam in fs.CAMERAS:
            sd_tok = sample["data"][cam]
            _, boxes, Kd = nusc.get_sample_data(sd_tok, box_vis_level=BoxVisibility.ALL)
            sd = nusc.get("sample_data", sd_tok)
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            c2w = torch.from_numpy(fs.pose_matrix(pose["translation"], pose["rotation"])
                                   @ fs.pose_matrix(cs["translation"], cs["rotation"]))
            K = torch.tensor(np.asarray(Kd), dtype=torch.float64)
            for box in boxes:
                ann = nusc.get("sample_annotation", box.token)
                uv, _ = project(torch.from_numpy(fs.box_corners(ann["translation"], ann["size"], ann["rotation"])), K, c2w)
                ref = view_points(box.corners(), np.asarray(Kd), normalize=True)[:2].T
                diffs.append(float(np.abs(uv.numpy() - ref).max()))
    return {"version": "v1.0-mini", "samples": n_samples, "cameras": len(fs.CAMERAS), "boxes": len(diffs),
            "max_abs_px": max(diffs, default=None), "median_abs_px": float(np.median(diffs)) if diffs else None}


def write_index(out: Path, clips: list[dict], summary: dict) -> None:
    e = html.escape
    rows = "\n".join(
        f'<figure><img src="{e(c["file"])}" alt="{e(c["id"])}"><figcaption>'
        + "<br>".join(e(x) for x in c["caption"]) + "</figcaption></figure>" for c in clips)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Fixed suite v1 check</title>
<style>body{{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
pre{{background:#fff;border:1px solid #ddd;padding:8px;overflow:auto}}
figure{{margin:0 0 24px}}img{{max-width:100%;height:auto;border:1px solid #ccc}}
figcaption{{font-family:ui-monospace,monospace;font-size:12px;color:#444}}</style></head><body>
<h1>E1.3 fixed suite v1: contact sheets</h1>
<p>Orange boxes: key instances of the clip. Grey: other annotated boxes. Projection: global &rarr; ego (ego_pose of
the camera sample_data) &rarr; camera (calibrated_sensor) &rarr; intrinsics. The camera was chosen to maximise key
centres in image at the reference frame, so that number is partly circular; the all-frames number and the
devkit comparison are the independent checks.</p>
<pre>{e(json.dumps(summary, indent=1))}</pre>
{rows}
</body></html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Contact sheets for the E1.3 fixed suite.")
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--suite", default="data/eval/fixed_suite_v2.jsonl")
    ap.add_argument("--cache", default=None, help="default: <suite stem>.cache.json")
    ap.add_argument("--out", default="data/eval/fixed_suite_v2_check")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--no-devkit-check", action="store_true")
    a = ap.parse_args(argv)
    suite, dataroot, out = Path(a.suite), Path(a.dataroot), Path(a.out)
    rows = [json.loads(line) for line in suite.read_text(encoding="utf-8").splitlines()]
    with open(a.cache or suite.with_suffix(".cache.json"), encoding="utf-8") as f:
        cache = json.load(f)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    clips = [render_clip(r, cache, dataroot, a.width, out / f"{i:02d}_{r['category']}.jpg",
                         reference_frame(r, by_id)) for i, r in enumerate(pick_clips(rows, a.n))]
    tot = lambda k: sum(c[k] for c in clips)  # noqa: E731
    summary = {
        "clips": len(clips), "per_category": {c: sum(x["category"] == c for x in clips) for c in fs.CATEGORIES},
        "primary_in_image_at_ref_frame": f"{tot('primary_in_image_ref')}/{len(clips)}",
        "key_in_image_at_ref_frame": f"{tot('key_in_image_ref')}/{tot('key_at_ref')}",
        "key_in_image_ref_frac": round(tot("key_in_image_ref") / max(1, tot("key_at_ref")), 4),
        "key_in_image_all_frames": f"{tot('key_in_image_all')}/{tot('key_all')}",
        "key_in_image_all_frac": round(tot("key_in_image_all") / max(1, tot("key_all")), 4),
        "devkit_check": None if a.no_devkit_check else devkit_check(dataroot),
    }
    (out / "check.json").write_text(json.dumps({"summary": summary, "clips": clips}, indent=1), encoding="utf-8")
    write_index(out, clips, summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
