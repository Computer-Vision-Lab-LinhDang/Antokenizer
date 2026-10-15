#!/usr/bin/env python3
"""Mine the MAVT E1.3 fixed test suite v1 (roadmap §1.4) from nuScenes val annotations.

    PYTHONPATH=src python scripts/mavt4d/build_fixed_suite.py --dataroot data/datasets/nuscenes \
        --out data/eval/fixed_suite_v1.jsonl --clip-len 8 --stride 4 --per-category 40 --git-commit <sha>

Light data prep (run it on a login node, not through Slurm). The three big tables are streamed
(mavt4d.data.jsonstream) keeping only val keyframe rows: sample_data (LIDAR_TOP + 6 cameras), the
ego_pose rows they reference, and val sample_annotation rows (train rows only feed the per-category
instance counts that define rare classes). The small tables are loaded whole. Mining rules live in
mavt4d.eval.fixed_suite.

Writes <out> (one row per clip, token references only), <out stem>.summary.json and
<out stem>.cache.json: the sample_data, ego_pose, calibrated_sensor and annotation rows of the selected
clips, which is all render_fixed_suite.py needs (so it never re-streams the big tables).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from mavt4d.data.jsonstream import iter_json_array
from mavt4d.eval import fixed_suite as fs

SMALL = ("scene", "sample", "sensor", "calibrated_sensor", "instance", "category", "attribute", "visibility")
BIG = ("sample_data", "ego_pose", "sample_annotation")


def _log(msg: str) -> None:
    print(f"[build_fixed_suite {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_nuscenes(dataroot, version: str, scene_names, train_names=(), log=_log) -> dict:
    """Keyframe records (fs.Scene) of ``scene_names`` plus the raw rows they reference, and the number of
    distinct instances per category in ``train_names``."""
    root = Path(dataroot) / version
    t = {n: _read(root / f"{n}.json") for n in SMALL}
    by_name = {s["name"]: s for s in t["scene"]}
    unknown = sorted(set(scene_names) - by_name.keys())
    if unknown:
        raise ValueError(f"{len(unknown)} scenes not in {version}/scene.json, e.g. {unknown[:5]}")
    keep_scenes = {by_name[n]["token"] for n in scene_names}
    train_scenes = {by_name[n]["token"] for n in train_names if n in by_name}
    samples = {s["token"]: s for s in t["sample"]}
    keep_samples = {k for k, s in samples.items() if s["scene_token"] in keep_scenes}
    channel = {s["token"]: s["channel"] for s in t["sensor"]}
    cs = {c["token"]: c for c in t["calibrated_sensor"]}
    cs_channel = {k: channel[c["sensor_token"]] for k, c in cs.items()}
    wanted = set(fs.CAMERAS) | {fs.LIDAR}

    t0, n, sd = time.monotonic(), 0, {}
    for r in iter_json_array(root / "sample_data.json"):
        n += 1
        if r["is_key_frame"] and r["sample_token"] in keep_samples:
            ch = cs_channel[r["calibrated_sensor_token"]]
            if ch in wanted:
                sd[r["token"]] = {"sample": r["sample_token"], "channel": ch, "ego_pose": r["ego_pose_token"],
                                  "cs": r["calibrated_sensor_token"], "timestamp": r["timestamp"],
                                  "filename": r["filename"], "width": r["width"], "height": r["height"]}
    log(f"sample_data: kept {len(sd)} of {n} rows ({time.monotonic() - t0:.0f} s)")

    t0, n, need, ego = time.monotonic(), 0, {v["ego_pose"] for v in sd.values()}, {}
    for r in iter_json_array(root / "ego_pose.json"):
        n += 1
        if r["token"] in need:
            ego[r["token"]] = {"translation": r["translation"], "rotation": r["rotation"]}
    if len(ego) != len(need):
        raise ValueError(f"ego_pose: {len(need) - len(ego)} referenced poses missing")
    log(f"ego_pose: kept {len(ego)} of {n} rows ({time.monotonic() - t0:.0f} s)")

    cat_name = {c["token"]: c["name"] for c in t["category"]}
    inst_cat = {i["token"]: cat_name[i["category_token"]] for i in t["instance"]}
    attr = {a["token"]: a["name"] for a in t["attribute"]}
    t0, n, anns, train_inst = time.monotonic(), 0, defaultdict(list), defaultdict(set)
    for r in iter_json_array(root / "sample_annotation.json"):
        n += 1
        st, inst = r["sample_token"], r["instance_token"]
        if st in keep_samples:
            anns[st].append(fs.Ann(inst, inst_cat[inst], tuple(r["translation"]), tuple(r["size"]),
                                   tuple(r["rotation"]), int(r["visibility_token"] or 0),
                                   tuple(attr[a] for a in r["attribute_tokens"]), r["num_lidar_pts"]))
        elif samples[st]["scene_token"] in train_scenes:
            train_inst[inst_cat[inst]].add(inst)
    log(f"sample_annotation: kept {sum(map(len, anns.values()))} of {n} rows ({time.monotonic() - t0:.0f} s)")

    by_sample: dict[str, dict[str, str]] = defaultdict(dict)
    for tok, v in sd.items():
        by_sample[v["sample"]][v["channel"]] = tok
    scenes = []
    for name in scene_names:
        sc, kfs, tok = by_name[name], [], by_name[name]["first_sample_token"]
        while tok:
            ch = by_sample[tok]
            if wanted - ch.keys():
                raise ValueError(f"{name}: sample {tok} has no keyframe for {sorted(wanted - ch.keys())}")
            pose = ego[sd[ch[fs.LIDAR]]["ego_pose"]]
            kfs.append(fs.Keyframe(tok, samples[tok]["timestamp"], tuple(pose["translation"]), tuple(pose["rotation"]),
                                   ch[fs.LIDAR], {c: ch[c] for c in fs.CAMERAS}, tuple(anns.get(tok, ()))))
            tok = samples[tok]["next"]
        if len(kfs) != sc["nbr_samples"]:
            raise ValueError(f"{name}: walked {len(kfs)} samples, scene.json says {sc['nbr_samples']}")
        scenes.append(fs.Scene(name, sc["token"], tuple(kfs)))
    return {"scenes": scenes, "sd": sd, "ego": ego, "cs": cs, "anns": anns,
            "train_counts": {c: len(v) for c, v in sorted(train_inst.items())}}


def assert_files_exist(dataroot, filenames) -> None:
    missing = [f for f in filenames if not (Path(dataroot) / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(filenames)} referenced files missing under {dataroot}, "
                                f"e.g. {missing[:3]}")


def _sd_tokens(row: dict) -> list[str]:
    return row["lidar_sd_tokens"] + [t for cam in row["cameras"] for t in row["camera_sd_tokens"][cam]]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 23):
            h.update(chunk)
    return h.hexdigest()


def _devkit_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("nuscenes-devkit")
    except PackageNotFoundError:
        return "not installed"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(dataroot, version: str, out, clip_len: int, stride: int, per_category: int, val_names, train_names,
          git_commit: str, log=_log) -> dict:
    t_start = time.monotonic()
    val_names, train_names = sorted(set(val_names)), sorted(set(train_names))
    if set(val_names) & set(train_names):
        raise ValueError(f"val and train share scenes: {sorted(set(val_names) & set(train_names))[:5]}")
    data = load_nuscenes(dataroot, version, val_names, train_names, log)
    rare = fs.rare_classes(data["train_counts"])
    rows, stats = fs.mine(data["scenes"], clip_len, stride, per_category, rare, version)
    used = {r["scene_name"] for r in rows}
    if not used <= set(val_names) or used & set(train_names):
        raise ValueError("selected scenes must be official val and disjoint from train")
    sd = data["sd"]
    tokens = sorted({t for r in rows for t in _sd_tokens(r)})
    files = sorted({sd[t]["filename"] for t in tokens})
    assert_files_exist(dataroot, files)
    t_mined = time.monotonic() - t_start

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
    cache_path, summary_path = out.with_suffix(".cache.json"), out.with_suffix(".summary.json")
    samples = sorted({s for r in rows for s in r["sample_tokens"]})
    cache = {
        "suite": fs.SUITE, "version": version,
        "sample_data": {t: sd[t] for t in tokens},
        "ego_pose": {sd[t]["ego_pose"]: data["ego"][sd[t]["ego_pose"]] for t in tokens},
        "calibrated_sensor": {sd[t]["cs"]: {k: data["cs"][sd[t]["cs"]][k] for k in
                                            ("translation", "rotation", "camera_intrinsic")} for t in tokens},
        "annotations": {s: [dataclasses.asdict(a) for a in data["anns"].get(s, ())] for s in samples},
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))

    counts = {c: sum(r["category"] == c for r in rows) for c in fs.CATEGORIES}
    summary = {
        "suite": fs.SUITE, "version": version, "split": "val", "fps": 2, "keyframe_only": True,
        "clip_len": clip_len, "stride": stride, "per_category_quota": per_category, "per_scene_cap": fs.PER_SCENE,
        "counts": counts, "shortfall": {c: max(0, per_category - n) for c, n in counts.items()},
        "rows": len(rows), "pairs": counts[fs.PAIR] // 2,
        "quota_notes": {fs.PAIR: "quota counts rows; 2 rows (one per window) per pair",
                        fs.MISSING: f"{len(fs.MISSING_VARIANTS)} variants per base clip, whole clips only, "
                                    f"<= quota; bases = first {fs.THRESHOLDS[fs.MISSING]['bases_per_category']} "
                                    f"selected clips of each other category"},
        "windows": stats["windows"], "candidates_before_selection": stats["candidates"],
        "pair_instance_candidates": stats["pair_instance_candidates"],
        "thresholds": fs.THRESHOLDS, "rare_classes": rare,
        "train_instances_total": sum(data["train_counts"].values()),
        "train_instances_by_category": data["train_counts"],
        "val_scene_count": len(val_names), "train_scene_count": len(train_names), "val_scenes_used": len(used),
        "referenced_files": len(files), "git_commit": git_commit, "devkit_version": _devkit_version(),
        "python": platform.python_version(),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),  # Linux: KiB
        "load_and_mine_s": round(t_mined, 1),
    }
    summary["sha256"] = {"output": _sha256(out), "cache": _sha256(cache_path),
                         "inputs": {f"{version}/{n}.json": _sha256(Path(dataroot) / version / f"{n}.json")
                                    for n in SMALL + BIG}}
    summary["runtime_s"] = round(time.monotonic() - t_start, 1)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log(f"wrote {len(rows)} rows to {out}")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Mine the E1.3 fixed test suite v1 from nuScenes val annotations.")
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--out", default="data/eval/fixed_suite_v1.jsonl")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--git-commit", default=None, help="commit of this code (default: git rev-parse HEAD)")
    a = ap.parse_args(argv)
    from nuscenes.utils import splits  # official split: 150 val / 700 train scenes

    s = build(a.dataroot, a.version, a.out, a.clip_len, a.stride, a.per_category, splits.val, splits.train,
              a.git_commit or _git_commit())
    print(json.dumps({k: s[k] for k in ("counts", "shortfall", "rare_classes", "peak_rss_mb", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main()
