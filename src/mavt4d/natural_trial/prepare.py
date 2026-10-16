"""Bounded EPIC RGB archive export; this module never imports Torch."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile


def recording_key(path: str) -> str:
    parts = PurePosixPath(path).parts
    for i, part in enumerate(parts[:-1]):
        if re.fullmatch(r"P\d+", part) and re.fullmatch(r"P\d+_\d+", parts[i + 1]):
            return "/".join(parts[i:i + 2])
    raise ValueError("unrecognized EPIC recording path")


def split_for_source(source: str) -> str:
    # Identical to mavt.data.splits.bucket; avoid mavt.__init__ importing Torch
    # in the CPU-only archive preparation process.
    bucket = int(hashlib.sha1(("mavt-split-v1\0epic:" + source).encode()).hexdigest()[:15], 16) / 16**15
    if bucket < .02:
        return "val"
    score = int(hashlib.sha256(("mavt-natural-trial-v1:" + source).encode()).hexdigest()[:8], 16)
    return "dev" if score / 2**32 < .2 else "train"


def select_frame_indices(indices: list[int], *, count: int = 12) -> list[int]:
    values = sorted(set(indices))
    if count < 2 or len(values) < count:
        raise ValueError("insufficient distinct frames")
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(metadata: Path, archives: Path, output: Path, *, sources: int = 16,
            clips_per_source: int = 8, frames: int = 12, height: int = 128,
            width: int = 192, seconds: float = 2.) -> dict:
    import numpy as np
    from PIL import Image
    if sources < 2 or clips_per_source < 1 or frames < 2 or min(height, width) < 1 or seconds <= 0:
        raise ValueError("invalid bounded export dimensions")
    if output.exists():
        raise FileExistsError("natural dataset exports are immutable; choose a new version")
    intervals = []
    for archive in sorted(archives.rglob("*.tar.gz")):
        match = re.search(r"_(\d+)_(\d+)\.tar\.gz$", archive.name)
        if match:
            intervals.append((int(match[1]), int(match[2]), archive))
    grouped = defaultdict(list)
    excluded_fps = 0
    for row in csv.DictReader(metadata.open(encoding="utf-8")):
        try:
            fps = float(row["FPS"])
            if not math.isfinite(fps) or not 0 < fps <= 240:
                raise ValueError
        except ValueError:
            excluded_fps += 1
            continue
        index = int(row["Index"])
        matches = [p for start, end, p in intervals if start <= index <= end]
        if len(matches) != 1:
            continue
        source = recording_key(row["Video Path"])
        if split_for_source(source) == "val":
            continue
        relative = row["Video Path"].removeprefix("epic_kitchen/").rstrip("/")
        grouped[source].append({"index": index, "source": source, "clip": relative,
                                "fps": fps, "archive": matches[0]})
    selected = sorted(grouped, key=lambda s: hashlib.sha256(("selection-v1:" + s).encode()).hexdigest())[:sources]
    if {split_for_source(s) for s in selected} != {"train", "dev"}:
        raise ValueError("selected sources must include train and dev; increase source budget")
    by_archive = defaultdict(dict)
    for source in selected:
        for item in sorted(grouped[source], key=lambda r: r["index"])[:clips_per_source]:
            by_archive[item["archive"]][item["clip"]] = item
    output.mkdir(parents=True)
    (output / "clips").mkdir()
    rows, failures, archive_records = [], [], []
    for archive, wanted in sorted(by_archive.items(), key=lambda p: p[0].name):
        buffers = defaultdict(dict)
        with tarfile.open(archive, "r|*") as stream:
            for member in stream:
                if not member.isfile() or member.size > 20_000_000:
                    continue
                member_path = PurePosixPath(member.name)
                clip = str(member_path.parent)
                if clip not in wanted or not re.fullmatch(r"\d+\.png", member_path.name):
                    continue
                frame = int(member_path.stem)
                if frame / wanted[clip]["fps"] > seconds:
                    continue
                handle = stream.extractfile(member)
                if handle is None:
                    continue
                with Image.open(io.BytesIO(handle.read())) as image:
                    rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                    buffers[clip][frame] = np.asarray(rgb, dtype=np.uint8).transpose(2, 0, 1).copy()
                # Safe early termination requires EVERY possible integer frame
                # in the bounded time interval for EVERY selected clip. It
                # makes no assumption about archive member ordering. Gaps
                # force a full stream read rather than silently changing data.
                if all(len(buffers[key]) == math.floor(item["fps"] * seconds) + 1
                       for key, item in wanted.items()):
                    break
        for clip, item in wanted.items():
            try:
                indices = select_frame_indices(list(buffers[clip]), count=frames)
            except ValueError:
                failures.append({"clip": clip, "reason": "fewer_than_requested_frames"})
                continue
            record_id = hashlib.sha256(("epic-natural-v1:" + clip).encode()).hexdigest()
            path = output / "clips" / f"{record_id}.npz"
            times = np.asarray(indices, dtype=np.float64) / item["fps"]
            times -= times[0]
            np.savez_compressed(path, images=np.stack([buffers[clip][i] for i in indices]), frame_times=times)
            rows.append({"record_id": record_id, "source": item["source"],
                         "split": split_for_source(item["source"]), "clip": clip,
                         "input": str(path.relative_to(output)), "sha256": sha256(path),
                         "frame_indices": indices, "fps": item["fps"],
                         "fps_provenance": "OmniWorld EPIC metadata FPS", "archive": archive.name})
        archive_records.append({"name": archive.name, "size": archive.stat().st_size,
                                "full_archive_checksum": "not_computed"})
        print(json.dumps({"archive": archive.name, "exported": len(rows), "skipped": len(failures)}), flush=True)
    rows.sort(key=lambda r: r["record_id"])
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "dev")}
    if min(counts.values()) < 1:
        raise ValueError("export lacks a usable train or dev partition")
    summary = {"schema": "natural-epic-trial-v1", "frames": frames, "height": height,
               "width": width, "maximum_source_seconds": seconds,
               "source_split": "canonical assign(epic:recording,.02,salt=mavt-split-v1); exclude val; train subset dev=sha256(mavt-natural-trial-v1:recording)<0.2",
               "manifest_sha256": sha256(manifest), "metadata_sha256": sha256(metadata),
               "counts": counts, "sources": {s: sorted({r["source"] for r in rows if r["split"] == s}) for s in counts},
               "excluded_metadata_invalid_fps": excluded_fps, "skipped_clips": failures,
               "archives": archive_records, "calibration": "unavailable; omitted, not invented",
               "spatial_transform": "RGB resize to HxW, aspect ratio changed; no geometry claim"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
