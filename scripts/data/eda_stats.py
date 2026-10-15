#!/usr/bin/env python3
"""EDA of the MAVT stage-1 datasets -> one machine-readable stats JSON plus thumbnail montages.

    cd ~/edtech/antoken && nice -n 10 env TMPDIR=data/.tmp PYTHONPATH=../antoken-eda/src \
        env/bin/python ../antoken-eda/scripts/data/eda_stats.py --root . --git-commit <sha>

Light metadata + small-sample work (headnode, no Slurm). Every dataset path is resolved against
--root (the repo checkout that holds data/), never against cwd. Each section writes
<out>/sections/<name>.json as soon as it finishes (a failing section records its traceback and the
others still run); the final eda_stats.json merges every section file present, so a rerun with
--sections <name> refreshes one section. nuScenes big tables are streamed (mavt4d.data.jsonstream)
and folded into counters/arrays on the fly; nothing per-annotation is kept as objects.

Histograms are {edges, counts, below, above}: counts follow numpy.histogram (last bin closed),
below/above count values outside [edges[0], edges[-1]], so sum(counts)+below+above == n.
Categoricals are {labels, counts} sorted by count.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import resource
import sys
import time
import traceback
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from mavt4d.data.jsonstream import iter_json_array  # noqa: E402  (mavt4d/__init__ is import-free)

# mavt/__init__ imports the model (torch, transformers): load the stdlib-only splits module directly.
_spec = importlib.util.spec_from_file_location("mavt_splits", SRC / "mavt" / "data" / "splits.py")
splits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(splits)  # type: ignore[union-attr]

OI_MANIFEST = "data/manifests/openimages_v2.jsonl"
OV_MANIFEST = "data/manifests/openvid_v2.jsonl"
SPLIT_DIR = "data/manifests/split_v1"
NUSC = "data/datasets/nuscenes"
SUITE = "data/eval/fixed_suite_v2"
IMAGENETTE = "data/eval/imagenette2-320"

# thresholds written verbatim into the JSON
SQUARE_TOL = 0.05            # |w/h - 1| <= tol counts as square
GRAY_CHROMA = 6.0            # near-grayscale: mean per-pixel (max(RGB) - min(RGB)) < this, 0..255
STATIC_MAD = 1.0             # near-static window: mean |frame diff| < this grey level, 0..255
CUT_RATIO, CUT_FLOOR = 5.0, 10.0  # hard cut: diff > 5 x clip median (spec); "_floor" variant also needs diff > 10
STATIONARY_MPS = 0.2
MOVED_M = 3.0
WINDOW_FRAMES, WINDOW_STRIDE = 16, 2  # training window: 16 frames at stride 2 = 32 source frames
PIX_SHORT = 256
VIDEO_SOURCES = ("celebv", "panda", "magictime")  # stratification strata (unparsed names map by prefix)
KNOWN_PREFIXES = ("celebv", "magictime", "pixabay", "pexels", "mixkit")

STOPWORDS = frozenset("""a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your yours yourself yourselves while upon onto within
without toward towards there's it's""".split())
PERSON_WORDS = frozenset("""person persons people man men woman women boy boys girl girls child children kid kids
baby babies lady ladies guy guys gentleman individual individuals he she his her him himself herself face faces facial
""".split())
FACE_WORDS = frozenset({"face", "faces", "facial"})
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_NIGHT = re.compile(r"\bnight\b", re.I)
_RAIN = re.compile(r"\brain(?:y|ing)?\b", re.I)
IMAGENETTE_NAMES = {"n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player",
                    "n03000684": "chain saw", "n03028079": "church", "n03394916": "French horn",
                    "n03417042": "garbage truck", "n03425413": "gas pump", "n03445777": "golf ball",
                    "n03888257": "parachute"}


def log(msg: str) -> None:
    print(f"[eda {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- generic helpers (unit-tested)
def _r(x: float) -> float:
    return float(f"{float(x):.6g}")


def num_stats(values, edges=None, bins: int = 30) -> dict:
    """n, missing (None/NaN), mean, median, p5, p95, min, max and a histogram. Default edges span the
    0.5-99.5 percentiles in ``bins`` equal bins; out-of-range values go to below/above."""
    raw = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64).ravel()
    a = raw[np.isfinite(raw)]
    out = {"n": int(a.size), "missing": int(raw.size - a.size)}
    if a.size == 0:
        return out
    p5, med, p95 = np.percentile(a, [5, 50, 95])
    out.update(mean=_r(a.mean()), median=_r(med), p5=_r(p5), p95=_r(p95), min=_r(a.min()), max=_r(a.max()))
    if edges is None:
        lo, hi = np.percentile(a, [0.5, 99.5])
        if hi <= lo:
            lo, hi = lo - 0.5, lo + 0.5
        edges = np.linspace(lo, hi, bins + 1)
    out["hist"] = hist(a, edges)
    return out


def hist(a, edges) -> dict:
    a, e = np.asarray(a, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    counts, _ = np.histogram(a, e)
    return {"edges": [_r(x) for x in e], "counts": counts.tolist(),
            "below": int((a < e[0]).sum()), "above": int((a > e[-1]).sum())}


def cat(counter, top: int | None = None) -> dict:
    items = sorted(Counter(counter).items(), key=lambda kv: (-kv[1], str(kv[0])))
    out = {"n": int(sum(c for _, c in items))}
    if top is not None and len(items) > top:
        out["other"] = int(sum(c for _, c in items[top:]))
        out["distinct"] = len(items)
        items = items[:top]
    out.update(labels=[str(k) for k, _ in items], counts=[int(c) for _, c in items])
    return out


def share(mask) -> float:
    m = np.asarray(mask, dtype=bool)
    return _r(m.mean()) if m.size else float("nan")


def source_dataset(path: str) -> str:
    """OpenVid origin dataset of a clip: video_source's tag; names it cannot parse fall back to a
    known filename prefix (Panda-70M clips have none: their name starts with the YouTube id)."""
    tag = splits.video_source(path)[1]
    if tag != "unparsed":
        return tag
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = stem.split("_", 1)[0]
    return prefix if prefix in KNOWN_PREFIXES else "unparsed"


def caption_tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def mentions(tokens, words) -> bool:
    return not words.isdisjoint(tokens)


def weather_flags(description: str) -> dict:
    return {"night": bool(_NIGHT.search(description or "")), "rain": bool(_RAIN.search(description or ""))}


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) on an (H, W, 3) array in 0..255."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))


def laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def hard_cuts(diffs, ratio: float = CUT_RATIO, floor: float = 0.0) -> int:
    """Number of frame differences above ``ratio`` x the clip median (and above ``floor``)."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 3:
        return 0
    return int(((d > ratio * np.median(d)) & (d > floor)).sum())


def flatten_criteria(obj, prefix: str = "", cat_keys=("category", "from", "to", "variant", "base_category")):
    """(key, value) leaves of a fixed-suite criteria dict: numbers (bools excluded) and the
    categorical strings in ``cat_keys``; lists flatten under ``key[]``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_criteria(v, f"{prefix}.{k}" if prefix else k, cat_keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten_criteria(v, f"{prefix}[]", cat_keys)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, obj
    elif isinstance(obj, str) and prefix.rsplit(".", 1)[-1] in cat_keys:
        yield prefix, obj


def wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------- io helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 23):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux: KiB


def square_tile(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BILINEAR)


def montage(rows: list[list[Image.Image]], tile: tuple[int, int], labels: list[str] | None = None) -> Image.Image:
    cols = max(len(r) for r in rows)
    pad = 18 if labels else 0
    W, H = tile
    out = Image.new("RGB", (cols * W, len(rows) * (H + pad)), (20, 20, 20))
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(rows):
        if labels:
            draw.text((4, i * (H + pad) + 3), labels[i], fill=(240, 240, 240))
        for j, im in enumerate(row):
            out.paste(im.convert("RGB").resize((W, H)), (j * W, i * (H + pad) + pad))
    return out


def save_jpeg(im: Image.Image, path: Path, max_bytes: int = 350_000) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        for q in (88, 80, 70, 60, 50, 40):
            im.save(path, "JPEG", quality=q, optimize=True)
            if path.stat().st_size <= max_bytes:
                return path.stat().st_size
        im = im.resize((im.width * 3 // 4, im.height * 3 // 4), Image.BILINEAR)


def load_short_side(path: Path, short: int = PIX_SHORT) -> Image.Image:
    """RGB image resized so the short side is ``short``; JPEG draft mode decodes at a reduced scale first."""
    with Image.open(path) as im:
        w, h = im.size
        s = short / min(w, h)
        im.draft("RGB", (max(1, math.ceil(w * s)), max(1, math.ceil(h * s))))
        im = im.convert("RGB")
        w, h = im.size
        s = short / min(w, h)
        return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


# ---------------------------------------------------------------- Open Images
def image_pixel_stats(path: Path) -> dict:
    im = load_short_side(path)
    x = np.asarray(im, dtype=np.float32)
    lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    sat = np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    return {"mean_r": float(x[..., 0].mean()), "mean_g": float(x[..., 1].mean()), "mean_b": float(x[..., 2].mean()),
            "lum_mean": float(lum.mean()), "lum_std": float(lum.std()), "sat_mean": float(sat.mean()),
            "colourfulness": colourfulness(x), "chroma_mean": float((x.max(-1) - x.min(-1)).mean()),
            "lap_var": laplacian_var(lum)}


def section_openimages(a) -> dict:
    rows = read_jsonl(a.root / OI_MANIFEST)
    w = np.array([r["width"] for r in rows], dtype=np.float64)
    h = np.array([r["height"] for r in rows], dtype=np.float64)
    b = np.array([r["bytes"] for r in rows], dtype=np.float64)
    short, long_ = np.minimum(w, h), np.maximum(w, h)
    aspect = w / h
    crop = short / long_  # 256^2 centre crop of the short-side-256 resize keeps short/long of it
    out = {
        "n": len(rows), "total_bytes": int(b.sum()),
        "width": num_stats(w), "height": num_stats(h), "short_side": num_stats(short),
        "long_side": num_stats(long_), "long_side_top": cat(Counter(long_.astype(int).tolist()), top=10),
        "aspect_w_over_h": num_stats(aspect, edges=np.linspace(0.25, 3.0, 23)),
        "orientation_share": {"portrait": share(aspect < 1 - SQUARE_TOL), "square": share(abs(aspect - 1) <= SQUARE_TOL),
                              "landscape": share(aspect > 1 + SQUARE_TOL), "square_tol": SQUARE_TOL},
        "megapixels": num_stats(w * h / 1e6), "bytes": num_stats(b),
        "bytes_per_pixel": num_stats(b / (w * h)),
        "mode": cat(Counter(r["mode"] for r in rows)),
        "centre_crop_256_area_share": num_stats(crop, edges=np.linspace(0.2, 1.0, 17)),
        "share_short_side_lt_256": share(short < 256), "share_short_side_lt_512": share(short < 512),
    }
    rng = random.Random(f"{a.seed}:openimages")
    idx = sorted(rng.sample(range(len(rows)), min(a.n_images, len(rows))))
    paths = [a.root / rows[i]["path"] for i in idx]
    t0 = time.monotonic()

    def safe(p):
        try:
            return image_pixel_stats(p)
        except Exception as e:  # noqa: BLE001 - one bad file must not kill the sample
            return {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(a.workers) as ex:
        px = list(ex.map(safe, paths))
    ok = [p for p in px if "error" not in p]
    log(f"openimages: pixel stats on {len(ok)}/{len(px)} images in {time.monotonic() - t0:.0f} s")
    col = {k: [p[k] for p in ok] for k in (ok[0] if ok else {})}
    out["pixel_sample"] = {
        "n_requested": len(px), "n_ok": len(ok), "errors": [p["error"] for p in px if "error" in p][:10],
        "seed": f"{a.seed}:openimages", "resize": f"short side {PIX_SHORT}, bilinear (JPEG draft decode)",
        **{k: num_stats(v, edges=np.linspace(0, 255, 18)) for k, v in col.items() if k.startswith("mean_")},
        "lum_mean": num_stats(col.get("lum_mean", []), edges=np.linspace(0, 255, 18)),
        "lum_std": num_stats(col.get("lum_std", []), edges=np.linspace(0, 120, 25)),
        "sat_mean": num_stats(col.get("sat_mean", []), edges=np.linspace(0, 1, 21)),
        "colourfulness": num_stats(col.get("colourfulness", []), edges=np.linspace(0, 150, 31)),
        "chroma_mean": num_stats(col.get("chroma_mean", []), edges=np.linspace(0, 150, 31)),
        "laplacian_var": num_stats(col.get("lap_var", []),
                                   edges=[0, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]),
        "near_grayscale_share": share(np.array(col.get("chroma_mean", [])) < GRAY_CHROMA),
        "near_grayscale_rule": f"mean per-pixel max(RGB)-min(RGB) < {GRAY_CHROMA} (0..255)",
    }
    thumbs = [square_tile(load_short_side(p, 192), 192) for p in paths[:24]]
    size = save_jpeg(montage([thumbs[i:i + 6] for i in range(0, len(thumbs), 6)], (192, 192)),
                     a.out / "thumbs" / "openimages_24.jpg")
    out["thumbs"] = {"file": "thumbs/openimages_24.jpg", "bytes": size, "grid": "6x4 centre crops",
                     "paths": [rows[i]["path"] for i in idx[:24]]}
    return out


# ---------------------------------------------------------------- OpenVid
def clip_motion(path: Path, n: int = WINDOW_FRAMES, stride: int = WINDOW_STRIDE, short: int = PIX_SHORT) -> dict:
    """First n frames at ``stride`` from the clip start, resized to short side ``short``: mean |diff| of
    consecutive sampled frames (luminance, 0..255) and the first frame as a PIL image."""
    import av  # lazy: tests of the helpers must not need PyAV
    frames = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "NONE"  # as mavt.data.datasets: frame threads deadlock in forked workers; keeps CPU low
        st.thread_count = 1
        cw, ch = st.codec_context.width, st.codec_context.height
        s = short / min(cw, ch)
        nw, nh = max(2, round(cw * s / 2) * 2), max(2, round(ch * s / 2) * 2)
        for i, f in enumerate(c.decode(st)):
            if i >= n * stride:
                break
            if i % stride == 0:
                frames.append(f.reformat(width=nw, height=nh, format="rgb24").to_ndarray())
    if not frames:
        raise ValueError("no frames decoded")
    gray = [0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2] for x in frames]
    diffs = [float(np.abs(gray[i + 1] - gray[i]).mean()) for i in range(len(gray) - 1)]
    return {"n_frames": len(frames), "diffs": diffs, "first": Image.fromarray(frames[0])}


def _corr(x, y) -> dict:
    from scipy import stats as ss  # lazy
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return {"n": int(x.size)}
    return {"n": int(x.size), "pearson": _r(ss.pearsonr(x, y)[0]), "spearman": _r(ss.spearmanr(x, y)[0])}


def section_openvid(a) -> dict:
    rows = read_jsonl(a.root / OV_MANIFEST)
    src = [source_dataset(r["path"]) for r in rows]
    gkey = [splits.group_key("video", r["path"]) for r in rows]
    gsize = Counter(gkey)
    out: dict = {"n": len(rows), "total_bytes": int(sum(r["bytes"] for r in rows)),
                 "source_dataset": cat(Counter(src)),
                 "source_note": "tag of mavt.data.splits.video_source; panda = Panda-70M names <ytid>_<scene>_<s>to<e> "
                                "(no prefix); unparsed magictime_* names fall back to their prefix",
                 "video_source_tag_raw": cat(Counter(splits.video_source(r["path"])[1] for r in rows)),
                 "openvid_part": cat(Counter(Path(r["path"]).parent.name for r in rows))}
    by_src = defaultdict(list)
    for i, s in enumerate(src):
        by_src[s].append(i)
    sizes = list(gsize.values())
    bucketed = Counter(str(n) if n < 10 else "10+" for n in sizes)
    out["clips_per_source_video"] = {
        "groups": len(gsize), "max": max(sizes), "stats": num_stats(sizes, edges=[1, 2, 3, 4, 5, 6, 8, 10, 20, 40, 80]),
        "group_size_hist": {k: bucketed.get(k, 0) for k in [*map(str, range(1, 10)), "10+"]},
        "clips_in_multi_clip_groups": int(sum(n for n in sizes if n > 1)),
        "per_source": {},
    }
    for s, ix in sorted(by_src.items()):
        keys = {gkey[i] for i in ix}  # a group can span sources (4 yt ids in both celebv and panda): its full size
        out["clips_per_source_video"]["per_source"][s] = {
            "clips": len(ix), "groups": len(keys), "max_clips_per_group": max(gsize[k] for k in keys),
            "clips_per_group": num_stats([gsize[k] for k in keys], edges=[1, 2, 3, 4, 5, 6, 8, 10, 20, 40, 80])}
    split_of = [splits.assign(k, splits.DEFAULT_VAL["video"]) for k in gkey]
    out["split_group_hash"] = {"val_fraction_of_groups": splits.DEFAULT_VAL["video"], "salt": splits.SPLIT_SALT,
                               "per_source": {s: dict(Counter(split_of[i] for i in ix)) for s, ix in sorted(by_src.items())},
                               "total": dict(Counter(split_of))}

    def field(name):
        return [r.get(name) for r in rows]

    fps, frames, dur = np.array(field("fps"), float), np.array(field("frames"), float), np.array(field("duration"), float)
    out.update(
        fps=num_stats(fps, edges=[0, 10, 15, 20, 24, 25, 26, 30, 31, 50, 61]), fps_top=cat(Counter(field("fps")), top=12),
        frames=num_stats(frames), duration_s=num_stats(dur),
        resolution=cat(Counter(f"{r['width']}x{r['height']}" for r in rows), top=15),
        short_side=num_stats([min(r["width"], r["height"]) for r in rows]),
        codec=cat(Counter(field("codec"))), bytes=num_stats(field("bytes")),
        bitrate_kbps=num_stats([r["bytes"] * 8 / 1000 / r["duration"] if r.get("duration") else None for r in rows]),
        camera_motion=cat(Counter(str(r.get("camera_motion")) for r in rows)),
        camera_motion_per_source={s: cat(Counter(str(rows[i].get("camera_motion")) for i in ix)) for s, ix in sorted(by_src.items())},
    )
    for name in ("aesthetic", "motion", "temporal_consistency"):
        vals = np.array([np.nan if v is None else v for v in field(name)], float)
        overall = num_stats(vals)
        edges = overall.get("hist", {}).get("edges")
        out[name] = {"overall": overall, "per_source": {s: num_stats(vals[ix], edges=edges) for s, ix in sorted(by_src.items())}}
    per_source_simple = {}
    for s, ix in sorted(by_src.items()):
        per_source_simple[s] = {"frames": num_stats(frames[ix]), "duration_s": num_stats(dur[ix]), "fps": num_stats(fps[ix]),
                                "resolution": cat(Counter(f"{rows[i]['width']}x{rows[i]['height']}" for i in ix), top=5)}
    out["per_source_basic"] = per_source_simple

    tf, df, wc, person, face = Counter(), Counter(), [], [], []
    for r in rows:
        toks = caption_tokens(r.get("caption", ""))
        wc.append(len((r.get("caption") or "").split()))
        content = [t for t in toks if t not in STOPWORDS]
        tf.update(content)
        df.update(set(content))
        person.append(mentions(toks, PERSON_WORDS))
        face.append(mentions(toks, FACE_WORDS))
    top = tf.most_common(50)
    out["caption"] = {
        "words": num_stats(wc), "missing_or_empty": int(sum(1 for r in rows if not (r.get("caption") or "").strip())),
        "top50_words": {"labels": [w for w, _ in top], "counts": [c for _, c in top],
                        "doc_share": [_r(df[w] / len(rows)) for w, _ in top]},
        "person_mention_share": share(person), "face_mention_share": share(face),
        "person_mention_share_per_source": {s: share(np.array(person)[ix]) for s, ix in sorted(by_src.items())},
        "face_mention_share_per_source": {s: share(np.array(face)[ix]) for s, ix in sorted(by_src.items())},
        "person_words": sorted(PERSON_WORDS), "tokeniser": _TOKEN.pattern + " on lowercased text, stopwords removed",
    }
    span = WINDOW_FRAMES * WINDOW_STRIDE
    covered = np.minimum(1.0, span / frames)
    out["window_coverage"] = {
        "window": f"{WINDOW_FRAMES} frames at stride {WINDOW_STRIDE} = {span} source frames",
        "covered_fraction": num_stats(covered, edges=np.linspace(0, 1, 21)),
        "seconds_covered": num_stats(np.minimum(frames, span) / fps),
        "share_clips_shorter_than_window": share(frames < span),
        "per_source_covered_fraction_mean": {s: _r(covered[ix].mean()) for s, ix in sorted(by_src.items())},
    }
    out["cross_checks"] = _openvid_checks(a, rows, gsize, out)
    out["pixel_motion_sample"] = _openvid_sample(a, rows, src, by_src)
    return out


def _openvid_checks(a, rows, gsize, out) -> dict:
    chk = read_json(a.root / SPLIT_DIR / "check_openvid_group_hash.json")
    summ = read_json(a.root / SPLIT_DIR / "summary.json")["manifests"]["openvid_v2.jsonl"]
    mine = out["clips_per_source_video"]
    return {
        "items": {"ok": chk["items"] == len(rows), "mine": len(rows), "split_check": chk["items"]},
        "groups": {"ok": chk["groups"] == len(gsize), "mine": len(gsize), "split_check": chk["groups"]},
        "largest_group": {"ok": chk["largest_group"] == mine["max"], "mine": mine["max"], "split_check": chk["largest_group"]},
        "group_size_hist": {"ok": chk["group_size_hist"] == mine["group_size_hist"]},
        "split_counts": {"ok": out["split_group_hash"]["total"].get("val") == summ["val"]["items"]
                         and out["split_group_hash"]["total"].get("train") == summ["train"]["items"],
                         "mine": out["split_group_hash"]["total"], "summary": {k: summ[k]["items"] for k in ("train", "val")}},
    }


def _openvid_sample(a, rows, src, by_src) -> dict:
    rng = random.Random(f"{a.seed}:openvid")
    strata = [s for s in VIDEO_SOURCES if s in by_src]
    quota = a.n_clips // max(1, len(strata))
    picked = {s: rng.sample(by_src[s], min(quota, len(by_src[s]))) for s in strata}
    t0, res, errors, firsts = time.monotonic(), [], [], defaultdict(list)
    jobs = [(s, i) for s in strata for i in picked[s]]

    def safe(job):
        try:
            return clip_motion(a.root / rows[job[1]]["path"])
        except Exception as e:  # noqa: BLE001 - one bad clip must not kill the sample
            return f"{rows[job[1]]['path']}: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(a.workers) as ex:  # threads overlap cold NFS reads; decode is ~0.1 s CPU per clip
        decoded = list(ex.map(safe, jobs))
    for (s, i), m in zip(jobs, decoded):
        if isinstance(m, str):
            errors.append(m)
            continue
        if len(firsts[s]) < 6:
            firsts[s].append((rows[i]["path"], m["first"]))
        d = m["diffs"]
        res.append({"source": s, "motion": rows[i].get("motion"), "n_frames": m["n_frames"],
                    "energy": float(np.mean(d)) if d else float("nan"),
                    "cuts": hard_cuts(d), "cuts_floor": hard_cuts(d, floor=CUT_FLOOR)})
    del decoded
    log(f"openvid: decoded {len(res)} clips ({len(errors)} errors) in {time.monotonic() - t0:.0f} s")

    def summarise(rs):
        e = np.array([r["energy"] for r in rs], float)
        return {"n": len(rs), "motion_energy": num_stats(e, edges=[0, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100]),
                "corr_energy_vs_motion_score": _corr(e, [np.nan if r["motion"] is None else r["motion"] for r in rs]),
                "near_static_share": share(e < STATIC_MAD),
                "hard_cut_share": share([r["cuts"] > 0 for r in rs]),
                "hard_cut_share_floor": share([r["cuts_floor"] > 0 for r in rs]),
                "short_window_share": share([r["n_frames"] < WINDOW_FRAMES for r in rs])}

    out = {"n_requested": sum(map(len, picked.values())), "n_ok": len(res), "errors": errors[:10], "n_errors": len(errors),
           "seed": f"{a.seed}:openvid", "strata": {s: len(v) for s, v in picked.items()},
           "note": "equal allocation per source: the pooled numbers are not representative of the manifest mix",
           "method": f"PyAV thread_count=1; first {WINDOW_FRAMES} frames at stride {WINDOW_STRIDE} from the clip start, "
                     f"resized to short side {PIX_SHORT}; energy = mean over consecutive sampled frames of mean |dY| (0..255)",
           "thresholds": {"near_static_energy_lt": STATIC_MAD, "cut_ratio_x_median": CUT_RATIO, "cut_floor": CUT_FLOOR},
           "pooled": summarise(res), "per_source": {s: summarise([r for r in res if r["source"] == s]) for s in strata}}
    thumb_rows = [[square_tile(im, 192) for _, im in firsts[s]] for s in strata if firsts[s]]
    labels = [f"{s} (first frames)" for s in strata if firsts[s]]
    if thumb_rows:
        out["thumbs"] = {"file": "thumbs/openvid_first_frames.jpg", "rows": labels,
                         "bytes": save_jpeg(montage(thumb_rows, (192, 192), labels), a.out / "thumbs" / "openvid_first_frames.jpg"),
                         "paths": {s: [p for p, _ in firsts[s]] for s in strata}}
    return out


# ---------------------------------------------------------------- nuScenes
def _nusc_small(root: Path, version: str, names) -> dict:
    return {n: read_json(root / version / f"{n}.json") for n in names}


def section_nuscenes(a) -> dict:
    from nuscenes.utils.splits import create_splits_scenes  # lazy: pulls matplotlib/cv2
    root = a.root / NUSC
    t = _nusc_small(root, "v1.0-trainval", ("scene", "sample", "log", "instance", "category", "attribute",
                                            "visibility", "sensor", "calibrated_sensor"))
    official = create_splits_scenes()
    name_split = {n: s for s in ("train", "val") for n in official[s]}
    loc = {r["token"]: r["location"] for r in t["log"]}
    scenes = {r["token"]: {**r, "split": name_split.get(r["name"]), "location": loc[r["log_token"]],
                           **weather_flags(r["description"])} for r in t["scene"]}
    sample_scene = {r["token"]: r["scene_token"] for r in t["sample"]}
    sample_split = {s: scenes[sc]["split"] for s, sc in sample_scene.items()}
    samples = {r["token"]: r for r in t["sample"]}
    out: dict = {"version": "v1.0-trainval", "table_rows": {k: len(v) for k, v in t.items()},
                 "split_source": "nuscenes.utils.splits.create_splits_scenes()", "per_split": {}}

    for sp in ("train", "val"):
        sc = [s for s in scenes.values() if s["split"] == sp]
        out["per_split"][sp] = {
            "scenes": len(sc), "samples": int(sum(s["nbr_samples"] for s in sc)),
            "samples_per_scene": num_stats([s["nbr_samples"] for s in sc], edges=np.arange(30, 46, 1)),
            "locations": cat(Counter(s["location"] for s in sc)),
            "night_share": share([s["night"] for s in sc]), "rain_share": share([s["rain"] for s in sc]),
            "night_scenes": int(sum(s["night"] for s in sc)), "rain_scenes": int(sum(s["rain"] for s in sc))}
    out["weather_rule"] = {"night": _NIGHT.pattern, "rain": _RAIN.pattern, "field": "scene.description"}
    out["scenes_not_in_train_or_val"] = sum(1 for s in scenes.values() if s["split"] is None)

    ordered = sorted(scenes.values(), key=lambda s: s["name"])
    picks = {
        "boston_day": next(s for s in ordered if s["location"] == "boston-seaport" and not s["night"] and not s["rain"]),
        "singapore_day": next(s for s in ordered if s["location"].startswith("singapore") and not s["night"] and not s["rain"]),
        "night": next(s for s in ordered if s["night"] and not s["rain"]),
        "rain": next(s for s in ordered if s["rain"] and not s["night"]),
    }
    thumb_samples = {}
    for key, s in picks.items():
        chain, tok = [], s["first_sample_token"]
        while tok:
            chain.append(tok)
            tok = samples[tok]["next"]
        for j in np.linspace(0, len(chain) - 1, 4).round().astype(int):
            thumb_samples[chain[j]] = key

    channel = {r["token"]: r["channel"] for r in t["sensor"]}
    cs_channel = {r["token"]: channel[r["sensor_token"]] for r in t["calibrated_sensor"]}
    t0 = time.monotonic()
    lidar_kf, key_n, nonkey_n, expected, cam_file = {}, Counter(), Counter(), defaultdict(set), {}
    for r in iter_json_array(root / "v1.0-trainval" / "sample_data.json"):
        ch = cs_channel[r["calibrated_sensor_token"]]
        if not r["is_key_frame"]:
            nonkey_n[ch] += 1
            continue
        key_n[ch] += 1
        if ch in ("CAM_FRONT", "LIDAR_TOP"):
            expected[ch].add(os.path.basename(r["filename"]))
        if ch == "LIDAR_TOP":
            lidar_kf[r["sample_token"]] = (r["ego_pose_token"], r["timestamp"])
        elif ch == "CAM_FRONT" and r["sample_token"] in thumb_samples:
            cam_file[r["sample_token"]] = r["filename"]
    log(f"nuscenes: sample_data streamed ({time.monotonic() - t0:.0f} s), rss {peak_rss_mb()} MB")
    need = {v[0] for v in lidar_kf.values()}
    pose = {}
    for r in iter_json_array(root / "v1.0-trainval" / "ego_pose.json"):
        if r["token"] in need:
            pose[r["token"]] = (r["translation"][0], r["translation"][1], quat_yaw(r["rotation"]))
    if len(pose) != len(need):
        raise ValueError(f"ego_pose: {len(need) - len(pose)} referenced LIDAR_TOP keyframe poses missing")
    ego = {s: (*pose[p], ts) for s, (p, ts) in lidar_kf.items()}  # sample -> (x, y, yaw, t_us)
    log(f"nuscenes: ego_pose streamed ({time.monotonic() - t0:.0f} s), rss {peak_rss_mb()} MB")
    out["ego_motion"] = _ego_motion(scenes, samples, ego)
    out["annotations"] = _annotations(a, root, t, scenes, sample_split, ego)
    log(f"nuscenes: sample_annotation streamed ({time.monotonic() - t0:.0f} s), rss {peak_rss_mb()} MB")
    out["on_disk"] = _on_disk(root, key_n, nonkey_n, expected)
    out["keyframes_per_channel"] = dict(key_n)
    out["nonkey_sample_data_per_channel"] = dict(nonkey_n)
    out["mini"] = _mini_intervals(root)
    out["thumbs"] = _nusc_thumbs(a, root, picks, thumb_samples, cam_file)
    return out


def _ego_motion(scenes, samples, ego) -> dict:
    res = {}
    for sp in ("train", "val"):
        speed, yaw_rate, dts = [], [], []
        for s in scenes.values():
            if s["split"] != sp:
                continue
            tok = s["first_sample_token"]
            while tok and samples[tok]["next"]:
                nxt = samples[tok]["next"]
                (x0, y0, h0, t0), (x1, y1, h1, t1) = ego[tok], ego[nxt]
                dt = (t1 - t0) / 1e6
                if dt > 0:
                    speed.append(math.hypot(x1 - x0, y1 - y0) / dt)
                    yaw_rate.append(wrap_deg(math.degrees(h1 - h0)) / dt)
                    dts.append(dt)
                tok = nxt
        sp_arr = np.array(speed)
        res[sp] = {"pairs": len(speed), "speed_mps": num_stats(sp_arr, edges=np.arange(0, 22, 1)),
                   "yaw_rate_deg_s": num_stats(yaw_rate, edges=np.arange(-40, 42, 4)),
                   "abs_yaw_rate_deg_s": num_stats(np.abs(yaw_rate), edges=[0, 1, 2, 5, 10, 20, 40, 90]),
                   "keyframe_dt_s": num_stats(dts, edges=np.arange(0.3, 0.75, 0.05)),
                   "stationary_share": share(sp_arr < STATIONARY_MPS)}
    res["definition"] = (f"consecutive keyframes of one scene (sample.next), LIDAR_TOP keyframe ego_pose; speed = xy "
                         f"displacement / dt (mean over the ~0.5 s gap); stationary = speed < {STATIONARY_MPS} m/s")
    return res


def _annotations(a, root, t, scenes, sample_split, ego) -> dict:
    cat_name = {r["token"]: r["name"] for r in t["category"]}
    inst = t["instance"]
    inst_idx = {r["token"]: i for i, r in enumerate(inst)}
    inst_cat = [cat_name[r["category_token"]] for r in inst]
    first_tok = {r["first_annotation_token"]: i for i, r in enumerate(inst)}
    last_tok = {r["last_annotation_token"]: i for i, r in enumerate(inst)}
    attr = {r["token"]: r["name"] for r in t["attribute"]}
    vis = {r["token"]: r["level"] for r in t["visibility"]}
    n = len(inst)
    n_ann = [0] * n
    inst_split: list = [None] * n
    mnx, mxx, mny, mxy = [math.inf] * n, [-math.inf] * n, [math.inf] * n, [-math.inf] * n
    first_xy, last_xy = [None] * n, [None] * n
    ann_n, inst_n, attr_n, vis_n = (defaultdict(Counter) for _ in range(4))
    lidar = {sp: array("f") for sp in ("train", "val")}
    dist = {sp: array("f") for sp in ("train", "val")}
    size_sum, size_n = defaultdict(lambda: [0.0, 0.0, 0.0]), Counter()
    per_sample = Counter()
    for r in iter_json_array(root / "v1.0-trainval" / "sample_annotation.json"):
        st = r["sample_token"]
        sp = sample_split[st]
        i = inst_idx[r["instance_token"]]
        c = inst_cat[i]
        x, y = r["translation"][0], r["translation"][1]
        ann_n[sp][c] += 1
        n_ann[i] += 1
        if not r["prev"]:
            inst_n[sp][c] += 1
            inst_split[i] = sp
        for tok in r["attribute_tokens"] or ["<none>"]:
            attr_n[sp][attr.get(tok, tok)] += 1
        vis_n[sp][vis.get(r["visibility_token"], f"<{r['visibility_token'] or 'empty'}>")] += 1
        lidar[sp].append(r["num_lidar_pts"])
        e = ego[st]
        dist[sp].append(math.hypot(x - e[0], y - e[1]))
        w, l_, h = r["size"]
        acc = size_sum[c]
        acc[0] += w
        acc[1] += l_
        acc[2] += h
        size_n[c] += 1
        per_sample[st] += 1
        mnx[i], mxx[i], mny[i], mxy[i] = min(mnx[i], x), max(mxx[i], x), min(mny[i], y), max(mxy[i], y)
        if r["token"] in first_tok:
            first_xy[i] = (x, y)
        if r["token"] in last_tok:
            last_xy[i] = (x, y)

    cats = sorted(set(inst_cat))
    tr_inst_total = sum(inst_n["train"].values())
    out = {"categories": {
        "labels": cats,
        "instances": {sp: [inst_n[sp][c] for c in cats] for sp in ("train", "val")},
        "annotations": {sp: [ann_n[sp][c] for c in cats] for sp in ("train", "val")},
        "train_instance_share": [_r(inst_n["train"][c] / tr_inst_total) for c in cats],
        "rare_lt_0.5pct_train_instances": [c for c in cats if inst_n["train"][c] / tr_inst_total < 0.005]}}
    out["attributes"] = {sp: cat(attr_n[sp]) for sp in ("train", "val")}
    out["visibility"] = {sp: cat(vis_n[sp]) for sp in ("train", "val")}
    lid_edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]
    out["num_lidar_pts"] = {sp: {**num_stats(lidar[sp], edges=lid_edges), "zero_share": share(np.frombuffer(lidar[sp], np.float32) == 0),
                                 "bins": "edges are log-spaced; bin [0,1) is exactly 0 points"} for sp in ("train", "val")}
    out["radial_distance_m"] = {sp: num_stats(dist[sp], edges=np.arange(0, 105, 5)) for sp in ("train", "val")}
    out["radial_distance_definition"] = "xy distance from the LIDAR_TOP keyframe ego position of the annotation's sample"
    out["share_within_50m"] = {sp: share(np.frombuffer(dist[sp], np.float32) <= 50) for sp in ("train", "val")}
    out["mean_box_size_wlh_m"] = {c: {"n": size_n[c], "w": _r(size_sum[c][0] / size_n[c]), "l": _r(size_sum[c][1] / size_n[c]),
                                      "h": _r(size_sum[c][2] / size_n[c])} for c in cats}
    out["annotations_per_sample"] = {sp: num_stats([per_sample.get(s, 0) for s, x in sample_split.items() if x == sp],
                                                   edges=[0, 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 600])
                                     for sp in ("train", "val")}
    tl = np.array(n_ann)
    ext = np.hypot(np.array(mxx) - np.array(mnx), np.array(mxy) - np.array(mny))
    fl = np.array([math.hypot(l[0] - f[0], l[1] - f[1]) if f and l else np.nan for f, l in zip(first_xy, last_xy)])
    isp = np.array([s or "none" for s in inst_split])
    ic = np.array(inst_cat)
    out["tracks"] = {sp: {"instances": int((isp == sp).sum()),
                          "track_length_keyframes": num_stats(tl[isp == sp], edges=[1, 2, 3, 5, 10, 15, 20, 30, 41]),
                          "moved_gt3m_share_first_to_last": share(fl[isp == sp] > MOVED_M),
                          "moved_gt3m_share_xy_extent": share(ext[isp == sp] > MOVED_M)} for sp in ("train", "val")}
    out["tracks"]["moved_gt3m_share_first_to_last_per_category"] = {c: share(fl[ic == c] > MOVED_M) for c in cats}
    out["tracks"]["definition"] = ("track length = annotations (keyframes) per instance; moved = xy distance first->last "
                                   "annotation > 3 m (also: diagonal of the xy bounding box of the track > 3 m)")
    out["cross_checks"] = {"nbr_annotations": {"ok": all(n_ann[i] == r["nbr_annotations"] for i, r in enumerate(inst))},
                           "instances_with_split": {"ok": int((isp == "none").sum()) == 0, "none": int((isp == "none").sum())}}
    suite_sum = a.root / f"{SUITE}.summary.json"
    if suite_sum.is_file():
        s = read_json(suite_sum)
        mine = {c: inst_n["train"][c] for c in cats if inst_n["train"][c]}
        out["cross_checks"]["train_instances_vs_fixed_suite"] = {"ok": mine == s["train_instances_by_category"]}
        out["cross_checks"]["rare_classes_vs_fixed_suite"] = {
            "ok": sorted(out["categories"]["rare_lt_0.5pct_train_instances"]) == sorted(s["rare_classes"])}
    return out


def _count_dir(d: Path) -> list[str]:
    try:
        return os.listdir(d)
    except FileNotFoundError:
        return []


def _on_disk(root: Path, key_n, nonkey_n, expected) -> dict:
    out = {"samples": {}, "sweeps": {}}
    for ch in sorted(key_n):
        names = _count_dir(root / "samples" / ch)
        rec = {"files": len(names), "keyframes_in_trainval": key_n[ch]}
        if ch in expected:
            have = set(names)
            rec["expected_present"] = len(expected[ch] & have)
            extra = sorted(have - expected[ch])
            rec["extra_files"], rec["extra_examples"] = len(extra), extra[:5]
        out["samples"][ch] = rec
    for ch in ("CAM_FRONT", "LIDAR_TOP"):
        out["sweeps"][ch] = {"files": len(_count_dir(root / "sweeps" / ch)), "nonkey_in_trainval": nonkey_n[ch]}
    out["note"] = "file counts from os.listdir (no du/stat on NFS); sweeps compare with v1.0-mini non-keyframes in 'mini'"
    return out


def _mini_intervals(root: Path) -> dict:
    t = _nusc_small(root, "v1.0-mini", ("scene", "sample", "sample_data", "sensor", "calibrated_sensor"))
    channel = {r["token"]: r["channel"] for r in t["sensor"]}
    cs_channel = {r["token"]: channel[r["sensor_token"]] for r in t["calibrated_sensor"]}
    sample_scene = {r["token"]: r["scene_token"] for r in t["sample"]}
    ts = defaultdict(list)
    nonkey = Counter()
    for r in t["sample_data"]:
        ch = cs_channel[r["calibrated_sensor_token"]]
        if ch in ("CAM_FRONT", "LIDAR_TOP"):
            ts[ch, sample_scene[r["sample_token"]]].append(r["timestamp"])
            nonkey[ch] += not r["is_key_frame"]
    out = {"scenes": len(t["scene"]), "samples": len(t["sample"]), "nonkey_sample_data": dict(nonkey)}
    for ch, edges, bands in (("CAM_FRONT", np.arange(0, 0.305, 0.005), (0, 0.07, 0.095, 0.15, math.inf)),
                             ("LIDAR_TOP", np.arange(0, 0.2025, 0.0025), (0, 0.045, 0.055, 0.075, math.inf))):
        dts = []
        for (c, _), v in ts.items():
            if c == ch:
                dts += (np.diff(sorted(v)) / 1e6).tolist()  # within one scene: no cross-scene gaps
        d = np.array(dts)
        out[ch] = {"intervals_s": num_stats(d, edges=edges),
                   "band_share": {f"[{lo},{hi})": share((d >= lo) & (d < hi)) for lo, hi in zip(bands[:-1], bands[1:])},
                   "drops_ge_0.15s": int((d >= 0.15).sum()), "nominal_hz": 12 if ch == "CAM_FRONT" else 20,
                   "mean_hz": _r(1 / d.mean()) if d.size else None}
    return out


def _nusc_thumbs(a, root: Path, picks, thumb_samples, cam_file) -> dict:
    rows, labels, files = [], [], {}
    for key, s in picks.items():
        toks = [tok for tok, k in thumb_samples.items() if k == key]
        rows.append([load_short_side(root / cam_file[tok], 180) for tok in toks])
        labels.append(f"{key}: {s['name']} {s['location']} | {s['description'][:70]}")
        files[key] = {"scene": s["name"], "location": s["location"], "description": s["description"],
                      "files": [cam_file[tok] for tok in toks]}
    size = save_jpeg(montage(rows, (320, 180), labels), a.out / "thumbs" / "nuscenes_cam_front.jpg")
    return {"file": "thumbs/nuscenes_cam_front.jpg", "bytes": size, "grid": "4 scenes x 4 evenly spaced CAM_FRONT keyframes",
            "scenes": files, "rule": "first scene by name matching each condition (weather_rule regexes)"}


# ---------------------------------------------------------------- E1.3 fixed suite
def section_fixed_suite(a) -> dict:
    rows = read_jsonl(a.root / f"{SUITE}.jsonl")
    summ = read_json(a.root / f"{SUITE}.summary.json")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out = {"rows": len(rows), "scenes_total": len({r["scene_name"] for r in rows}),
           "requires_train_exclusion": int(sum(bool(r.get("requires_train_exclusion")) for r in rows)),
           "per_category": {}, "rare_classes": summ.get("rare_classes"),
           "quota": summ.get("per_category_quota"), "shortfall": summ.get("shortfall")}
    for c, rs in by_cat.items():
        nums, cats = defaultdict(list), defaultdict(Counter)
        for r in rs:
            for k, v in flatten_criteria(r.get("criteria") or {}):
                if isinstance(v, str):
                    cats[k][v] += 1
                else:
                    nums[k].append(v)
        out["per_category"][c] = {"rows": len(rs), "scenes": len({r["scene_name"] for r in rs}),
                                  "criteria": {k: num_stats(v, bins=10) for k, v in sorted(nums.items())},
                                  "criteria_categorical": {k: cat(v) for k, v in sorted(cats.items())}}
    rare_rows = [i["category"] for r in by_cat.get("rare_category", []) for i in r["criteria"].get("instances", [])]
    out["rare_category_instances_by_class"] = cat(Counter(rare_rows))
    out["cross_checks"] = {"counts": {"ok": {c: len(v) for c, v in by_cat.items()} == {c: n for c, n in summ["counts"].items() if n},
                                      "summary": summ["counts"]},
                           "rows": {"ok": len(rows) == summ["rows"]}}
    return out


# ---------------------------------------------------------------- eval sets
CIFAR_GLOBS = ("data/eval/*[cC][iI][fF][aA][rR]*", "data/datasets/*[cC][iI][fF][aA][rR]*", "data/*[cC][iI][fF][aA][rR]*",
               "~/.cache/huggingface/datasets/*[cC][iI][fF][aA][rR]*", "~/.cache/huggingface/hub/*[cC][iI][fF][aA][rR]*",
               "~/.cache/torch/*[cC][iI][fF][aA][rR]*", "~/.cache/torch/*/*[cC][iI][fF][aA][rR]*",
               "~/edtech/*/data/eval/*[cC][iI][fF][aA][rR]*", "~/edtech/*/data/*[cC][iI][fF][aA][rR]*",
               "~/edtech/datasets/*[cC][iI][fF][aA][rR]*", "~/edtech/*[cC][iI][fF][aA][rR]*")


def section_eval_sets(a) -> dict:
    out = {"imagenette2_320": {}}
    for sp in ("train", "val"):
        d = a.root / IMAGENETTE / sp
        counts = {}
        for cls in sorted(os.listdir(d)) if d.is_dir() else []:
            counts[cls] = sum(1 for e in os.scandir(d / cls) if e.is_file())
        out["imagenette2_320"][sp] = {"labels": [f"{k} {IMAGENETTE_NAMES.get(k, '?')}" for k in counts],
                                      "counts": list(counts.values()), "n": sum(counts.values())}
    hits = []
    for g in CIFAR_GLOBS:
        pat = os.path.expanduser(g) if g.startswith("~") else str(a.root / g)
        hits += glob.glob(pat)
    out["cifar100"] = {"present": any("100" in h for h in hits), "hits": sorted(set(hits)), "searched": list(CIFAR_GLOBS),
                       "note": "glob search only (no download); data/* patterns are relative to --root"}
    return out


# ---------------------------------------------------------------- overview + driver
def overview(a, res) -> dict:
    summ = read_json(a.root / SPLIT_DIR / "summary.json")["manifests"]
    oi, ov, ns, fx, ev = (res.get(k, {}) for k in ("openimages", "openvid", "nuscenes", "fixed_suite", "eval_sets"))
    nsd = ns.get("on_disk", {})
    return {
        "openimages_v2": {"items": oi.get("n"), "total_bytes": oi.get("total_bytes"), "modality": "image",
                          "labels": "none in the manifest (path, width, height, mode, bytes); reconstruction only",
                          "split": {k: summ["openimages_v2.jsonl"][k] for k in ("train", "val")}},
        "openvid_v2": {"items": ov.get("n"), "total_bytes": ov.get("total_bytes"), "modality": "video",
                       "labels": "caption + aesthetic, motion, temporal_consistency scores + camera_motion tag",
                       "split": {k: summ["openvid_v2.jsonl"][k] for k in ("train", "val")}},
        "nuscenes_v1.0-trainval": {"modality": "multi-view video (6 cams) + LiDAR + radar, 2 Hz keyframes",
                                   "labels": "3D boxes, tracks, attributes, visibility, maps",
                                   "split": {sp: {k: v for k, v in ns.get("per_split", {}).get(sp, {}).items()
                                                  if k in ("scenes", "samples")} for sp in ("train", "val")},
                                   "keyframes_on_disk": {ch: r.get("files") for ch, r in nsd.get("samples", {}).items()},
                                   "sweeps_on_disk": nsd.get("sweeps"), "total_bytes": None,
                                   "total_bytes_note": "no manifest with sizes and no du on NFS by rule"},
        "nuscenes_v1.0-mini": {k: ns.get("mini", {}).get(k) for k in ("scenes", "samples", "nonkey_sample_data")},
        "fixed_suite_v2": {"rows": fx.get("rows"), "scenes": fx.get("scenes_total"), "source": "nuScenes val keyframes"},
        "imagenette2_320": {sp: ev.get("imagenette2_320", {}).get(sp, {}).get("n") for sp in ("train", "val")},
        "cifar100_present": ev.get("cifar100", {}).get("present"),
        "split_note": "group-hash split (mavt.data.splits, salt mavt-split-v1) from data/manifests/split_v1/summary.json",
    }


SECTIONS = {"openimages": section_openimages, "openvid": section_openvid, "nuscenes": section_nuscenes,
            "fixed_suite": section_fixed_suite, "eval_sets": section_eval_sets}


def _collect_checks(obj, path=""):
    if isinstance(obj, dict):
        if "ok" in obj and isinstance(obj["ok"], bool) and not obj["ok"]:
            yield path
        for k, v in obj.items():
            yield from _collect_checks(v, f"{path}.{k}" if path else k)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repo checkout holding data/")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/data/eda/eda_v1")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--n-images", type=int, default=3000)
    ap.add_argument("--n-clips", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--workers", type=int, default=4, help="threads for image decode")
    ap.add_argument("--git-commit", default="unknown")
    a = ap.parse_args(argv)
    a.root = a.root.resolve()
    a.out = (a.out or a.root / "data" / "eda" / "eda_v1").resolve()
    (a.out / "sections").mkdir(parents=True, exist_ok=True)
    t_all = time.monotonic()
    for name in a.sections.split(","):
        t0 = time.monotonic()
        log(f"section {name} ...")
        try:
            res = SECTIONS[name](a)
        except Exception:  # noqa: BLE001 - record and keep going
            res = {"error": traceback.format_exc()}
            log(f"section {name} FAILED:\n{res['error']}")
        res["_runtime_s"], res["_peak_rss_mb"] = round(time.monotonic() - t0, 1), peak_rss_mb()
        res["_git_commit"], res["_generated_at"] = a.git_commit, datetime.now(timezone.utc).isoformat(timespec="seconds")
        (a.out / "sections" / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
        log(f"section {name} done in {res['_runtime_s']} s, peak rss {res['_peak_rss_mb']} MB")

    res = {p.stem: read_json(p) for p in sorted((a.out / "sections").glob("*.json"))}
    summ = read_json(a.root / SPLIT_DIR / "summary.json")
    inputs = {f: sha256(a.root / f) for f in (OI_MANIFEST, OV_MANIFEST, f"{SUITE}.jsonl", f"{SUITE}.summary.json",
                                             f"{SPLIT_DIR}/summary.json")}
    checks = {f"manifest_sha256_matches_split_summary.{m}": {"ok": inputs[f"data/manifests/{m}"] == summ["manifests"][m]["sha256"]}
              for m in ("openimages_v2.jsonl", "openvid_v2.jsonl")}
    import av
    import PIL
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": a.git_commit,
        "host": platform.node(), "python": platform.python_version(),
        "versions": {"numpy": np.__version__, "pillow": PIL.__version__, "av": av.__version__},
        "root": str(a.root), "input_sha256": inputs, "manifest_checks": checks,
        "samples": {"openimages_pixel": a.n_images, "openvid_clips": a.n_clips, "seed": a.seed,
                    "seed_rule": "random.Random(f'{seed}:<section>')"},
        "thresholds": {"square_tol": SQUARE_TOL, "near_gray_chroma_lt": GRAY_CHROMA, "near_static_energy_lt": STATIC_MAD,
                       "cut_ratio": CUT_RATIO, "cut_floor": CUT_FLOOR, "stationary_mps": STATIONARY_MPS, "moved_m": MOVED_M,
                       "window": [WINDOW_FRAMES, WINDOW_STRIDE], "pixel_short_side": PIX_SHORT},
        "section_runtime_s": {k: v.get("_runtime_s") for k, v in res.items()},
        "section_git_commit": {k: v.get("_git_commit") for k, v in res.items()},
        "errors": {k: v["error"] for k, v in res.items() if "error" in v},
        "peak_rss_mb": max([peak_rss_mb()] + [v.get("_peak_rss_mb", 0) for v in res.values()]),
        "runtime_s_this_invocation": round(time.monotonic() - t_all, 1),
        "overview": overview(a, res), **res,
    }
    stats["failed_cross_checks"] = sorted(_collect_checks({**res, "manifest_checks": checks}))
    (a.out / "eda_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    log(f"wrote {a.out / 'eda_stats.json'}; failed cross-checks: {stats['failed_cross_checks'] or 'none'}; "
        f"errors: {list(stats['errors']) or 'none'}")


if __name__ == "__main__":
    main()
