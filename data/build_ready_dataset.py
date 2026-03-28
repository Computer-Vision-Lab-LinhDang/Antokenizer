#!/usr/bin/env python3
"""Build unified ready/ dataset with image, video, and 3D object data.

Creates the directory structure expected by MAVT training pipeline:

    datasets/datasets/ready/
    ├── images/                    -> symlink to Open Images V7 train
    ├── images_val/                -> symlink to Open Images V7 validation
    ├── videos/
    │   ├── webvid/                -> WebVid-10M downloaded videos
    │   └── panda70m/              -> Panda-70M downloaded videos
    ├── 3d_objects/
    │   └── renders/<obj_id>/      -> view_0..7.png + cameras.json
    ├── captions/
    │   ├── images.json            -> image_id -> caption
    │   ├── videos.json            -> video_id -> caption
    │   └── 3d.json                -> obj_id -> caption
    └── manifests/
        ├── stage_readiness.json
        └── dataset_stats.json

Usage:
    # Full setup (images + video metadata + 3D captions)
    python data/build_ready_dataset.py

    # Mini dataset with 100 samples per modality (for testing)
    python data/build_ready_dataset.py --sample 100

    # Include video download (WebVid, small batch)
    python data/build_ready_dataset.py --download-videos --video-limit 10000

    # Dry run to see what would happen
    python data/build_ready_dataset.py --dry-run
"""
import argparse
import ast
import csv
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent  # Antoken/
DATASETS = BASE / "datasets" / "datasets"
READY = DATASETS / "ready"

# Source datasets
OI_DIR = DATASETS / "open_images_v7"
WEBVID_DIR = DATASETS / "webvid_10m"
PANDA_DIR = DATASETS / "panda_70m"
CAP3D_DIR = DATASETS / "cap3d"
OBJAVERSE_DIR = DATASETS / "objaverse_xl"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def count_files(directory: Path, pattern: str = "*.jpg") -> int:
    try:
        result = subprocess.run(
            ["find", "-L", str(directory), "-maxdepth", "1", "-name", pattern, "-type", "f"],
            capture_output=True, text=True, timeout=60,
        )
        return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    except Exception:
        return 0


# ─── 1. Images ───────────────────────────────────────────────────────────────

def setup_images(dry_run: bool = False):
    """Symlink Open Images V7 into ready/images/."""
    src_train = OI_DIR / "images" / "train"
    src_val = OI_DIR / "images" / "validation"
    dst_train = READY / "images"
    dst_val = READY / "images_val"

    if not src_train.exists():
        log.error("Open Images V7 train not found: %s", src_train)
        return 0

    n_train = count_files(src_train, "*.jpg")
    n_val = count_files(src_val, "*.jpg") if src_val.exists() else 0
    log.info("Open Images V7: %d train + %d val images", n_train, n_val)

    if dry_run:
        log.info("[DRY RUN] Would symlink %s -> %s", dst_train, src_train)
        return n_train

    ensure_dir(READY)

    # Train symlink
    if dst_train.exists() or dst_train.is_symlink():
        dst_train.unlink()
    dst_train.symlink_to(src_train)
    log.info("Symlinked %s -> %s", dst_train, src_train)

    # Val symlink
    if src_val.exists():
        if dst_val.exists() or dst_val.is_symlink():
            dst_val.unlink()
        dst_val.symlink_to(src_val)
        log.info("Symlinked %s -> %s", dst_val, src_val)

    return n_train


def build_image_captions(dry_run: bool = False):
    """Build image caption JSON (Open Images V7 has class labels, not captions)."""
    caption_out = READY / "captions" / "images.json"

    if dry_run:
        log.info("[DRY RUN] Would create image captions at %s", caption_out)
        return

    ensure_dir(caption_out.parent)

    # Try to load Open Images class descriptions
    class_desc_file = OI_DIR / "metadata" / "class-descriptions-boxable.csv"
    captions = {}
    if class_desc_file.exists():
        with open(class_desc_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    captions[row[0].strip()] = row[1].strip()
        log.info("Loaded %d Open Images class descriptions", len(captions))

    with open(caption_out, 'w') as f:
        json.dump(captions, f)
    log.info("Saved image captions to %s", caption_out)


# ─── 2. Videos ───────────────────────────────────────────────────────────────

def setup_videos(download: bool = False, limit: int = 0, dry_run: bool = False):
    """Set up video directory with WebVid and Panda-70M data."""
    vid_dir = READY / "videos"
    webvid_dst = vid_dir / "webvid"
    panda_dst = vid_dir / "panda70m"

    if dry_run:
        log.info("[DRY RUN] Would set up video dirs at %s", vid_dir)
        return 0

    ensure_dir(webvid_dst)
    ensure_dir(panda_dst)

    n_videos = 0

    # Check for already downloaded WebVid videos
    webvid_data = WEBVID_DIR / "data" / "train"
    if webvid_data.exists():
        existing = list(webvid_data.rglob("*.mp4"))
        if existing:
            # Symlink the whole data dir
            link = webvid_dst / "data"
            if not link.exists():
                link.symlink_to(webvid_data)
            n_videos += len(existing)
            log.info("WebVid: linked %d existing videos", len(existing))

    # Check for already downloaded Panda-70M videos
    panda_vids = PANDA_DIR / "videos"
    if panda_vids.exists():
        existing = list(panda_vids.rglob("*.mp4"))
        if existing:
            link = panda_dst / "data"
            if not link.exists():
                link.symlink_to(panda_vids)
            n_videos += len(existing)
            log.info("Panda-70M: linked %d existing videos", len(existing))

    # Also check Panda-70M video2dataset output
    panda_v2d = Path("/home/sagemaker-user/ws/Panda-70M/dataset_dataloading/training_2m")
    if panda_v2d.exists():
        existing = list(panda_v2d.rglob("*.mp4"))
        if existing:
            link = panda_dst / "training_2m"
            if not link.exists():
                link.symlink_to(panda_v2d)
            n_videos += len(existing)
            log.info("Panda-70M (v2d): linked %d existing videos", len(existing))

    # Download videos if requested
    if download and limit > 0:
        log.info("Downloading videos (limit=%d)...", limit)
        n_downloaded = download_webvid_videos(webvid_dst, limit)
        n_videos += n_downloaded

    if n_videos == 0:
        log.warning("No video files found. Videos need to be downloaded separately.")
        log.info("  WebVid: python data/build_ready_dataset.py --download-videos --video-limit 10000")
        log.info("  Panda-70M: use video2dataset (see data/extract_3d_data.py)")

    return n_videos


def download_webvid_videos(dest_dir: Path, limit: int = 10000):
    """Download WebVid-10M videos from direct MP4 URLs."""
    import concurrent.futures
    import requests

    csv_dir = WEBVID_DIR / "metadata" / "data" / "train" / "partitions"
    if not csv_dir.exists():
        log.warning("WebVid CSV dir not found: %s", csv_dir)
        return 0

    # Collect URLs from CSV files
    urls = []
    csv_files = sorted(csv_dir.glob("*.csv"))
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row.get("videoid", "")
                url = row.get("contentUrl", "")
                caption = row.get("name", "")
                if vid_id and url and url.endswith(".mp4"):
                    urls.append((vid_id, url, caption))
                if len(urls) >= limit:
                    break
        if len(urls) >= limit:
            break

    log.info("Collected %d WebVid URLs to download", len(urls))

    data_dir = dest_dir / "data"
    ensure_dir(data_dir)
    captions = {}

    def _download_one(item):
        vid_id, url, caption = item
        dest = data_dir / f"{vid_id}.mp4"
        if dest.exists():
            return vid_id, caption, True
        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return vid_id, caption, True
        except Exception:
            return vid_id, caption, False

    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_download_one, item) for item in urls]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            vid_id, caption, ok = fut.result()
            if ok:
                success += 1
                captions[vid_id] = caption
            if (i + 1) % 500 == 0:
                log.info("  Progress: %d/%d downloaded (%d success)", i + 1, len(urls), success)

    log.info("Downloaded %d/%d WebVid videos", success, len(urls))

    # Save captions
    caption_file = dest_dir / "captions_webvid.json"
    with open(caption_file, 'w') as f:
        json.dump(captions, f)

    return success


def build_video_captions(dry_run: bool = False):
    """Build video caption JSON from WebVid and Panda-70M CSVs."""
    caption_out = READY / "captions" / "videos.json"

    if dry_run:
        log.info("[DRY RUN] Would create video captions at %s", caption_out)
        return

    ensure_dir(caption_out.parent)
    captions = {}

    # WebVid-10M captions
    webvid_csv_dir = WEBVID_DIR / "metadata" / "data" / "train" / "partitions"
    if webvid_csv_dir.exists():
        count = 0
        for csv_file in sorted(webvid_csv_dir.glob("*.csv")):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid_id = row.get("videoid", "").strip()
                    caption = row.get("name", "").strip()
                    if vid_id and caption:
                        captions[vid_id] = caption
                        count += 1
        log.info("Loaded %d WebVid captions", count)

    # Panda-70M captions (2M subset)
    panda_csv = PANDA_DIR / "csv" / "panda70m_training_2m.csv"
    if panda_csv.exists():
        count = 0
        with open(panda_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row.get("videoID", "").strip()
                caption_raw = row.get("caption", "").strip()
                if vid_id and caption_raw:
                    # Panda captions are lists, take first one
                    try:
                        cap_list = ast.literal_eval(caption_raw)
                        if isinstance(cap_list, list) and cap_list:
                            captions[vid_id] = cap_list[0]
                        else:
                            captions[vid_id] = caption_raw
                    except (ValueError, SyntaxError):
                        captions[vid_id] = caption_raw
                    count += 1
        log.info("Loaded %d Panda-70M captions", count)

    log.info("Total video captions: %d", len(captions))
    with open(caption_out, 'w') as f:
        json.dump(captions, f)
    log.info("Saved video captions to %s", caption_out)


# ─── 3. 3D Objects ──────────────────────────────────────────────────────────

def setup_3d(dry_run: bool = False):
    """Check 3D objects status (extraction handled by extract_3d_data.py)."""
    render_dir = READY / "3d_objects" / "renders"

    if render_dir.exists():
        n_objects = sum(1 for d in render_dir.iterdir() if d.is_dir())
    else:
        n_objects = 0

    log.info("3D objects: %d extracted in %s", n_objects, render_dir)

    if n_objects == 0:
        log.warning("No 3D objects found. Run extract_3d_data.py first:")
        log.info("  python data/extract_3d_data.py --download-renders --build-captions --num-zips 53")

    return n_objects


def build_3d_captions(dry_run: bool = False):
    """Build 3D caption JSON from Cap3D CSVs."""
    caption_out = READY / "captions" / "3d.json"

    if caption_out.exists():
        with open(caption_out, 'r') as f:
            existing = json.load(f)
        log.info("3D captions already exist: %d entries", len(existing))
        return

    if dry_run:
        log.info("[DRY RUN] Would create 3D captions at %s", caption_out)
        return

    ensure_dir(caption_out.parent)
    captions = {}

    csv_files = [
        (CAP3D_DIR / "Cap3D_automated_Objaverse_full.csv", "Objaverse"),
        (CAP3D_DIR / "Cap3D_automated_ABO.csv", "ABO"),
        (CAP3D_DIR / "Cap3D_automated_ShapeNet.csv", "ShapeNet"),
    ]

    for csv_path, name in csv_files:
        if not csv_path.exists():
            continue
        count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    captions[row[0].strip()] = row[1].strip().strip('"')
                    count += 1
        log.info("Loaded %d captions from %s (%s)", count, csv_path.name, name)

    with open(caption_out, 'w') as f:
        json.dump(captions, f)
    log.info("Saved %d 3D captions to %s", len(captions), caption_out)


# ─── 4. Manifests ───────────────────────────────────────────────────────────

def build_manifests(n_images: int, n_videos: int, n_3d: int, dry_run: bool = False):
    """Build stage readiness and dataset stats manifests."""
    manifest_dir = READY / "manifests"

    if dry_run:
        log.info("[DRY RUN] Would create manifests at %s", manifest_dir)
        return

    ensure_dir(manifest_dir)

    # Dataset stats
    stats = {
        "images": {
            "count": n_images,
            "source": "Open Images V7",
            "path": str(READY / "images"),
            "caption_file": str(READY / "captions" / "images.json"),
        },
        "videos": {
            "count": n_videos,
            "sources": ["WebVid-10M", "Panda-70M"],
            "path": str(READY / "videos"),
            "caption_file": str(READY / "captions" / "videos.json"),
        },
        "3d_objects": {
            "count": n_3d,
            "source": "Cap3D (Objaverse)",
            "path": str(READY / "3d_objects" / "renders"),
            "caption_file": str(READY / "captions" / "3d.json"),
        },
    }

    with open(manifest_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Stage readiness
    readiness = {
        "stage1_image_foundation": {
            "status": "READY" if n_images > 0 else "NOT READY",
            "modalities": ["image"],
            "data": {"images": n_images},
        },
        "stage2_video_dynamics": {
            "status": "READY" if (n_images > 0 and n_videos > 0) else "PARTIAL" if n_images > 0 else "NOT READY",
            "modalities": ["image", "video"],
            "data": {"images": n_images, "videos": n_videos},
        },
        "stage3_3d_geometry": {
            "status": "READY" if (n_images > 0 and n_videos > 0 and n_3d > 0) else "PARTIAL",
            "modalities": ["image", "video", "3d"],
            "data": {"images": n_images, "videos": n_videos, "3d_objects": n_3d},
        },
    }

    with open(manifest_dir / "stage_readiness.json", 'w') as f:
        json.dump(readiness, f, indent=2)

    log.info("Saved manifests to %s", manifest_dir)


# ─── 5. Sample mode ─────────────────────────────────────────────────────────

def build_sample_dataset(n_samples: int, download_videos: bool = True):
    """Build a mini dataset with exactly n_samples per modality.

    Output: datasets/datasets/ready_sample_{n}/
        ├── images/          n symlinked .jpg files
        ├── videos/          n downloaded .mp4 files
        ├── 3d_objects/renders/  n object dirs (symlinked)
        ├── captions/
        │   ├── images.json
        │   ├── videos.json
        │   └── 3d.json
        └── manifests/
    """
    sample_dir = DATASETS / f"ready_sample_{n_samples}"
    if sample_dir.exists():
        log.info("Removing existing sample dir: %s", sample_dir)
        shutil.rmtree(sample_dir)

    log.info("=" * 70)
    log.info("Building sample dataset: %d per modality", n_samples)
    log.info("Output: %s", sample_dir)
    log.info("=" * 70)

    # ── Images: symlink n random images from Open Images V7 ──
    log.info("\n--- Images: selecting %d samples ---", n_samples)
    img_src = OI_DIR / "images" / "train"
    img_dst = sample_dir / "images"
    ensure_dir(img_dst)
    img_captions = {}

    if img_src.exists():
        all_imgs = list(img_src.glob("*.jpg"))
        selected = random.sample(all_imgs, min(n_samples, len(all_imgs)))
        for img in selected:
            (img_dst / img.name).symlink_to(img)
            img_captions[img.stem] = f"An image of {img.stem}"
        log.info("  Linked %d images", len(selected))
    else:
        log.error("  Open Images V7 not found at %s", img_src)
        selected = []

    ensure_dir(sample_dir / "captions")
    with open(sample_dir / "captions" / "images.json", 'w') as f:
        json.dump(img_captions, f)

    n_images = len(selected)

    # ── Videos: download n from WebVid (direct MP4 URLs, fast) ──
    log.info("\n--- Videos: downloading %d samples ---", n_samples)
    vid_dst = sample_dir / "videos"
    ensure_dir(vid_dst)
    vid_captions = {}

    webvid_csv_dir = WEBVID_DIR / "metadata" / "data" / "train" / "partitions"
    if download_videos and webvid_csv_dir.exists():
        import concurrent.futures
        import requests

        # Collect URLs
        urls = []
        for csv_file in sorted(webvid_csv_dir.glob("*.csv")):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid_id = row.get("videoid", "")
                    url = row.get("contentUrl", "")
                    caption = row.get("name", "")
                    if vid_id and url and url.endswith(".mp4"):
                        urls.append((vid_id, url, caption))
                    if len(urls) >= n_samples * 3:  # buffer for failures
                        break
            if len(urls) >= n_samples * 3:
                break

        random.shuffle(urls)
        log.info("  Collected %d candidate URLs, downloading...", len(urls))

        def _dl(item):
            vid_id, url, caption = item
            dest = vid_dst / f"{vid_id}.mp4"
            if dest.exists():
                return vid_id, caption, True
            try:
                resp = requests.get(url, timeout=15, stream=True)
                resp.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return vid_id, caption, True
            except Exception:
                if dest.exists():
                    dest.unlink()
                return vid_id, caption, False

        success = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for vid_id, caption, ok in pool.map(_dl, urls):
                if ok:
                    vid_captions[vid_id] = caption
                    success += 1
                if success >= n_samples:
                    break

        log.info("  Downloaded %d videos", success)
    else:
        # Try linking existing videos
        existing_vids = list((READY / "videos").rglob("*.mp4")) if (READY / "videos").exists() else []
        selected_vids = random.sample(existing_vids, min(n_samples, len(existing_vids)))
        for v in selected_vids:
            (vid_dst / v.name).symlink_to(v)
            vid_captions[v.stem] = f"A video of {v.stem}"
        success = len(selected_vids)
        log.info("  Linked %d existing videos", success)

    with open(sample_dir / "captions" / "videos.json", 'w') as f:
        json.dump(vid_captions, f, indent=2)

    n_videos = success

    # ── 3D Objects: symlink n from already extracted renders ──
    log.info("\n--- 3D Objects: selecting %d samples ---", n_samples)
    obj_src = READY / "3d_objects" / "renders"
    obj_dst = sample_dir / "3d_objects" / "renders"
    ensure_dir(obj_dst)
    obj_captions = {}

    # Load full caption file
    full_3d_captions = {}
    cap3d_json = READY / "captions" / "3d.json"
    if cap3d_json.exists():
        with open(cap3d_json, 'r') as f:
            full_3d_captions = json.load(f)

    if obj_src.exists():
        all_objs = [d for d in obj_src.iterdir() if d.is_dir() and (d / "view_0.png").exists()]
        selected_objs = random.sample(all_objs, min(n_samples, len(all_objs)))
        for obj_dir in selected_objs:
            (obj_dst / obj_dir.name).symlink_to(obj_dir)
            obj_captions[obj_dir.name] = full_3d_captions.get(
                obj_dir.name, f"A 3D model of {obj_dir.name}"
            )
        log.info("  Linked %d 3D objects", len(selected_objs))
    else:
        log.error("  No extracted 3D objects found at %s", obj_src)
        log.info("  Run: python data/extract_3d_data.py --download-renders --num-zips 1 --build-captions")
        selected_objs = []

    with open(sample_dir / "captions" / "3d.json", 'w') as f:
        json.dump(obj_captions, f, indent=2)

    n_3d = len(selected_objs)

    # ── Manifests ──
    manifest_dir = sample_dir / "manifests"
    ensure_dir(manifest_dir)

    stats = {
        "mode": f"sample_{n_samples}",
        "images": {"count": n_images, "path": str(img_dst)},
        "videos": {"count": n_videos, "path": str(vid_dst)},
        "3d_objects": {"count": n_3d, "path": str(obj_dst)},
    }
    with open(manifest_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("SAMPLE DATASET READY")
    log.info("=" * 70)
    log.info("  %s/", sample_dir.relative_to(BASE))
    log.info("  ├── images/           %d images", n_images)
    log.info("  ├── videos/           %d videos", n_videos)
    log.info("  ├── 3d_objects/       %d objects", n_3d)
    log.info("  └── captions/         per-modality JSON")
    log.info("")

    total_size = sum(
        f.stat().st_size for f in sample_dir.rglob("*") if f.is_file() and not f.is_symlink()
    )
    log.info("  Disk usage (non-symlink): %.1f MB", total_size / (1024 * 1024))
    log.info("")
    log.info("To use in training:")
    rel = sample_dir.relative_to(BASE)
    log.info("  img_ds = ImageDataset(['%s/images'])", rel)
    log.info("  vid_ds = VideoDataset(['%s/videos'])", rel)
    log.info("  obj_ds = Object3DDataset('%s/3d_objects/renders',", rel)
    log.info("              caption_file='%s/captions/3d.json')", rel)
    log.info("=" * 70)

    return n_images, n_videos, n_3d


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build unified ready/ dataset for MAVT training"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", type=int, default=0,
                        help="Build mini dataset with N samples per modality (e.g. --sample 100)")
    parser.add_argument("--download-videos", action="store_true",
                        help="Download WebVid videos (direct MP4 URLs)")
    parser.add_argument("--video-limit", type=int, default=10000,
                        help="Max videos to download (default: 10000)")
    args = parser.parse_args()

    # ── Sample mode ──
    if args.sample > 0:
        build_sample_dataset(args.sample, download_videos=True)
        return

    # ── Full mode ──
    log.info("=" * 70)
    log.info("MAVT Unified Dataset Builder")
    log.info("=" * 70)
    log.info("Output: %s", READY)
    log.info("=" * 70)

    # Step 1: Images
    log.info("\n--- Step 1: Setting up images (Open Images V7) ---")
    n_images = setup_images(dry_run=args.dry_run)
    build_image_captions(dry_run=args.dry_run)

    # Step 2: Videos
    log.info("\n--- Step 2: Setting up videos (WebVid + Panda-70M) ---")
    n_videos = setup_videos(
        download=args.download_videos,
        limit=args.video_limit,
        dry_run=args.dry_run,
    )
    build_video_captions(dry_run=args.dry_run)

    # Step 3: 3D Objects
    log.info("\n--- Step 3: Setting up 3D objects (Cap3D) ---")
    n_3d = setup_3d(dry_run=args.dry_run)
    build_3d_captions(dry_run=args.dry_run)

    # Step 4: Manifests
    log.info("\n--- Step 4: Building manifests ---")
    build_manifests(n_images, n_videos, n_3d, dry_run=args.dry_run)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("DATASET SUMMARY")
    log.info("=" * 70)
    log.info("  datasets/datasets/ready/")
    log.info("  ├── images/           %d images (Open Images V7)", n_images)
    log.info("  ├── videos/           %d videos (WebVid + Panda-70M)", n_videos)
    log.info("  ├── 3d_objects/       %d objects (Cap3D)", n_3d)
    log.info("  ├── captions/         image + video + 3D captions")
    log.info("  └── manifests/        stage readiness reports")
    log.info("")

    icons = {"READY": "[OK]", "PARTIAL": "[!!]", "NOT READY": "[XX]"}
    stages = [
        ("Stage 1 (Image)", "READY" if n_images > 0 else "NOT READY"),
        ("Stage 2 (+ Video)", "READY" if n_videos > 0 else "PARTIAL" if n_images > 0 else "NOT READY"),
        ("Stage 3 (+ 3D)", "READY" if n_3d > 0 else "PARTIAL"),
    ]
    for name, status in stages:
        log.info("  %s %s: %s", icons.get(status, "[??]"), name, status)

    log.info("=" * 70)

    # Next steps
    if n_videos == 0:
        log.info("\nTo add videos:")
        log.info("  # WebVid (direct download, fast):")
        log.info("  python data/build_ready_dataset.py --download-videos --video-limit 50000")
        log.info("")
        log.info("  # Panda-70M (YouTube, needs cookies):")
        log.info("  cd /home/sagemaker-user/ws/Panda-70M/dataset_dataloading")
        log.info("  video2dataset --url_list=... --config=video2dataset/video2dataset/configs/panda70m.yaml")

    if n_3d == 0:
        log.info("\nTo add 3D objects:")
        log.info("  python data/extract_3d_data.py --download-renders --build-captions --num-zips 53")

    log.info("\nTo use in training:")
    log.info("  from train.datasets_modality import ImageDataset, VideoDataset, Object3DDataset")
    log.info("  img_ds = ImageDataset(['datasets/datasets/ready/images'])")
    log.info("  vid_ds = VideoDataset(['datasets/datasets/ready/videos'])")
    log.info("  obj_ds = Object3DDataset('datasets/datasets/ready/3d_objects/renders',")
    log.info("                            caption_file='datasets/datasets/ready/captions/3d.json')")


if __name__ == "__main__":
    main()
