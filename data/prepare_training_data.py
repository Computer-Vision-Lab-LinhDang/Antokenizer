#!/usr/bin/env python3
"""Prepare dataset directory structure for MAVT progressive training.

Creates the expected directory layout:
    data/ready/
        images/          -> Stage 1, 2, 3 (symlinks Open Images V7 train)
        videos/          -> Stage 2, 3 (placeholder + metadata index)
        3d_objects/      -> Stage 3 (placeholder + metadata index)
        captions/
            images.json  -> image_id -> caption mapping
            videos.json  -> video_id -> caption mapping
            3d.json      -> obj_id -> caption mapping
        manifests/
            stage_readiness.json

Usage:
    python data/prepare_training_data.py [--dry-run]
"""
import argparse
import csv
import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent  # data/
DATASETS = BASE / "datasets"
READY = BASE / "ready"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def fast_count(directory: Path, pattern: str = "*.jpg") -> int:
    """Count files quickly using shell command (follows symlinks)."""
    try:
        result = subprocess.run(
            ["find", "-L", str(directory), "-maxdepth", "1", "-name", pattern, "-type", "f"],
            capture_output=True, text=True, timeout=60,
        )
        return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    except Exception:
        return 0


def fast_line_count(filepath: Path) -> int:
    """Count lines in a file without reading it all into memory."""
    try:
        result = subprocess.run(
            ["wc", "-l", str(filepath)],
            capture_output=True, text=True, timeout=30,
        )
        return int(result.stdout.strip().split()[0])
    except Exception:
        return 0


# ─── Open Images V7 ──────────────────────────────────────────────────────────

def prepare_images(dry_run: bool = False):
    """Symlink Open Images V7 images into ready/images/."""
    src_train = DATASETS / "open_images_v7" / "images" / "train"
    src_val = DATASETS / "open_images_v7" / "images" / "validation"
    dst = READY / "images"

    if not src_train.exists():
        log.error("Open Images V7 train images not found at %s", src_train)
        return False

    n_train = fast_count(src_train, "*.jpg")
    n_val = fast_count(src_val, "*.jpg") if src_val.exists() else 0
    log.info("Open Images V7: %d train + %d val images", n_train, n_val)

    if dry_run:
        log.info("[DRY RUN] Would symlink %s -> %s", dst, src_train)
        return True

    ensure_dir(READY)

    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src_train)
    log.info("Symlinked %s -> %s", dst, src_train)

    dst_val = READY / "images_val"
    if dst_val.exists() or dst_val.is_symlink():
        dst_val.unlink()
    if src_val.exists():
        dst_val.symlink_to(src_val)
        log.info("Symlinked %s -> %s", dst_val, src_val)

    return True


def build_image_captions(dry_run: bool = False):
    """Create placeholder caption JSON for Open Images V7."""
    caption_out = READY / "captions" / "images.json"

    if dry_run:
        log.info("[DRY RUN] Would create placeholder image captions at %s", caption_out)
        return True

    ensure_dir(caption_out.parent)
    with open(caption_out, "w") as f:
        json.dump({}, f)
    log.info("Created placeholder image captions (OI-V7 has no text captions)")
    log.warning(
        "Open Images V7 has NO text captions. "
        "For caption-conditioned training, integrate DFN-2B or LAION data."
    )
    return True


# ─── Video data ───────────────────────────────────────────────────────────────

def prepare_videos(dry_run: bool = False):
    """Create video metadata index from WebVid-10M and Panda-70M CSVs.

    Does NOT read entire CSVs — only counts lines and reads headers + samples.
    """
    dst = READY / "videos"
    index_out = READY / "manifests" / "video_index.json"
    caption_out = READY / "captions" / "videos.json"

    # ── WebVid-10M: count and sample ──
    webvid_dir = DATASETS / "webvid_10m" / "metadata" / "data" / "train" / "partitions"
    webvid_count = 0
    webvid_samples = []
    if webvid_dir.exists():
        csv_files = sorted(webvid_dir.glob("*.csv"))
        for csv_file in csv_files:
            webvid_count += max(0, fast_line_count(csv_file) - 1)  # -1 for header
        # Read a few samples from first file
        if csv_files:
            with open(csv_files[0], "r") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 5:
                        break
                    webvid_samples.append({
                        "video_id": row.get("videoid", ""),
                        "url": row.get("contentUrl", ""),
                        "caption": row.get("name", ""),
                    })
        log.info("WebVid-10M: ~%d entries across %d CSV files", webvid_count, len(csv_files))
    else:
        log.warning("WebVid-10M metadata not found at %s", webvid_dir)

    # ── Panda-70M: count and sample ──
    panda_csv = DATASETS / "panda_70m" / "csv" / "panda70m_training_2m.csv"
    panda_count = 0
    panda_samples = []
    if panda_csv.exists():
        panda_count = max(0, fast_line_count(panda_csv) - 1)
        with open(panda_csv, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                panda_samples.append({
                    "video_id": row.get("videoID", ""),
                    "url": row.get("url", ""),
                    "caption": row.get("caption", ""),
                })
        log.info("Panda-70M: ~%d entries", panda_count)
    else:
        log.warning("Panda-70M CSV not found at %s", panda_csv)

    if dry_run:
        log.info("[DRY RUN] Would create video index (%d WebVid + %d Panda)", webvid_count, panda_count)
        return True

    ensure_dir(dst)
    ensure_dir(index_out.parent)

    index = {
        "status": "METADATA_ONLY",
        "note": "Video files NOT downloaded. Download actual videos before Stage 2.",
        "webvid_10m": {"count": webvid_count, "sample": webvid_samples},
        "panda_70m": {"count": panda_count, "sample": panda_samples},
        "csv_paths": {
            "webvid_dir": str(webvid_dir) if webvid_dir.exists() else None,
            "panda_csv": str(panda_csv) if panda_csv.exists() else None,
        },
    }
    with open(index_out, "w") as f:
        json.dump(index, f, indent=2)
    log.info("Saved video index to %s", index_out)

    # Build captions from samples only (full extraction too slow for prep)
    ensure_dir(caption_out.parent)
    captions = {}
    for s in webvid_samples + panda_samples:
        if s.get("video_id") and s.get("caption"):
            captions[s["video_id"]] = s["caption"]
    with open(caption_out, "w") as f:
        json.dump(captions, f)
    log.info("Saved %d sample video captions (full extraction deferred)", len(captions))

    return True


# ─── 3D data ──────────────────────────────────────────────────────────────────

def prepare_3d(dry_run: bool = False):
    """Create 3D index from Cap3D captions and Objaverse metadata."""
    dst = READY / "3d_objects"
    index_out = READY / "manifests" / "3d_index.json"
    caption_out = READY / "captions" / "3d.json"

    # ── Cap3D captions ──
    cap3d_csv = DATASETS / "cap3d" / "Cap3D_automated_Objaverse_full.csv"
    cap3d_count = 0
    cap3d_samples = []
    if cap3d_csv.exists():
        cap3d_count = fast_line_count(cap3d_csv)
        with open(cap3d_csv, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= 10:
                    break
                if len(row) >= 2:
                    cap3d_samples.append({"obj_id": row[0].strip(), "caption": row[1].strip()})
        log.info("Cap3D: ~%d entries", cap3d_count)
    else:
        log.warning("Cap3D CSV not found at %s", cap3d_csv)

    # ── Objaverse UIDs ──
    obj_uids_file = DATASETS / "objaverse_xl" / "metadata" / "objaverse_uids.json"
    obj_uid_count = 0
    if obj_uids_file.exists():
        # Just get file size to estimate count (avoid loading 28MB JSON)
        fsize = obj_uids_file.stat().st_size
        obj_uid_count = fsize // 50  # rough estimate: ~50 bytes per UID entry
        log.info("Objaverse UIDs: ~%d entries (estimated from %d MB file)",
                 obj_uid_count, fsize // (1024*1024))
    else:
        log.warning("Objaverse UIDs not found at %s", obj_uids_file)

    if dry_run:
        log.info("[DRY RUN] Would create 3D index (%d Cap3D + ~%d Objaverse)",
                 cap3d_count, obj_uid_count)
        return True

    ensure_dir(dst)
    ensure_dir(index_out.parent)

    index = {
        "status": "METADATA_ONLY",
        "note": "Pre-rendered views NOT available. Need to render from 3D meshes.",
        "cap3d": {"count": cap3d_count, "sample": cap3d_samples},
        "objaverse": {"count_estimate": obj_uid_count},
        "csv_paths": {
            "cap3d_csv": str(cap3d_csv) if cap3d_csv.exists() else None,
            "objaverse_uids": str(obj_uids_file) if obj_uids_file.exists() else None,
        },
    }
    with open(index_out, "w") as f:
        json.dump(index, f, indent=2)
    log.info("Saved 3D index to %s", index_out)

    # Save sample captions
    ensure_dir(caption_out.parent)
    captions = {s["obj_id"]: s["caption"] for s in cap3d_samples}
    with open(caption_out, "w") as f:
        json.dump(captions, f)
    log.info("Saved %d sample 3D captions (full extraction deferred)", len(captions))

    return True


# ─── Stage readiness report ──────────────────────────────────────────────────

def build_stage_manifests(dry_run: bool = False):
    """Create stage readiness report."""
    manifest_dir = READY / "manifests"

    if dry_run:
        log.info("[DRY RUN] Would create stage manifests in %s", manifest_dir)
        return {}

    ensure_dir(manifest_dir)

    # Count available data
    img_dir = READY / "images"
    n_images = fast_count(img_dir, "*.jpg") if img_dir.exists() else 0

    vid_dir = READY / "videos"
    n_videos = fast_count(vid_dir, "*.mp4") if vid_dir.exists() else 0

    obj_dir = READY / "3d_objects"
    n_3d = 0
    if obj_dir.exists():
        try:
            result = subprocess.run(
                ["find", str(obj_dir), "-name", "view_0.png", "-type", "f"],
                capture_output=True, text=True, timeout=10,
            )
            n_3d = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        except Exception:
            pass

    report = {
        "stage1_image_foundation": {
            "status": "READY" if n_images > 0 else "NOT READY",
            "modalities": ["image"],
            "images_available": n_images,
            "images_source": "Open Images V7 train",
            "data_path": str(img_dir),
            "missing": [] if n_images > 0 else ["No images found"],
            "notes": [
                "Open Images V7 has NO text captions - class labels only.",
                "Image-only reconstruction training is fully supported.",
                "For caption-conditioned training, integrate DFN-2B / LAION.",
            ],
        },
        "stage2_video_dynamics": {
            "status": "PARTIAL" if n_images > 0 else "NOT READY",
            "modalities": ["image", "video"],
            "images_available": n_images,
            "videos_available": n_videos,
            "video_metadata": "WebVid-10M + Panda-70M CSVs available",
            "data_paths": {"image": str(img_dir), "video": str(vid_dir)},
            "missing": (
                ["Video files NOT downloaded (WebVid-10M, Panda-70M)"]
                if n_videos == 0 else []
            ),
        },
        "stage3_3d_geometry": {
            "status": "PARTIAL" if n_images > 0 else "NOT READY",
            "modalities": ["image", "video", "3d"],
            "images_available": n_images,
            "videos_available": n_videos,
            "3d_objects_available": n_3d,
            "3d_metadata": "Cap3D captions + Objaverse UIDs available",
            "data_paths": {"image": str(img_dir), "video": str(vid_dir), "3d": str(obj_dir)},
            "missing": [
                m for m in [
                    "Video files NOT downloaded" if n_videos == 0 else None,
                    "3D pre-rendered views NOT available" if n_3d == 0 else None,
                ] if m
            ],
        },
    }

    report_path = manifest_dir / "stage_readiness.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved stage readiness report to %s", report_path)

    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare MAVT training data")
    parser.add_argument("--dry-run", action="store_true", help="Only check, don't modify")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MAVT Data Preparation")
    log.info("=" * 70)
    log.info("Datasets: %s", DATASETS)
    log.info("Output:   %s", READY)
    log.info("=" * 70)

    log.info("\n--- Step 1: Preparing images (Open Images V7) ---")
    img_ok = prepare_images(dry_run=args.dry_run)

    log.info("\n--- Step 2: Building image captions ---")
    build_image_captions(dry_run=args.dry_run)

    log.info("\n--- Step 3: Preparing video metadata ---")
    prepare_videos(dry_run=args.dry_run)

    log.info("\n--- Step 4: Preparing 3D metadata ---")
    prepare_3d(dry_run=args.dry_run)

    log.info("\n--- Step 5: Building stage readiness report ---")
    report = build_stage_manifests(dry_run=args.dry_run)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("STAGE READINESS SUMMARY")
    log.info("=" * 70)

    if isinstance(report, dict):
        for stage, info in report.items():
            status = info.get("status", "UNKNOWN")
            icon = {"READY": "[OK]", "PARTIAL": "[!!]", "NOT READY": "[XX]"}.get(status, "[??]")
            log.info("  %s %s: %s", icon, stage, status)
            for m in info.get("missing", []):
                log.info("       -> %s", m)

    log.info("")
    log.info("TO START TRAINING:")
    log.info("  Stage 1 (Image only): READY")
    log.info("    python -m train.example_stage_training \\")
    log.info("      --data-root %s --start-stage 0", READY)
    log.info("")
    log.info("  Stage 2: Need video files first. Download with:")
    log.info("    See video_index.json for source URLs")
    log.info("")
    log.info("  Stage 3: Need 3D rendered views. Render with:")
    log.info("    See 3d_index.json for Objaverse UIDs + Cap3D captions")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
