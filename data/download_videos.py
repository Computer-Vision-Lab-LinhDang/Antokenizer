#!/usr/bin/env python3
"""Download video files from WebVid-10M and Panda-70M metadata CSVs.

WebVid-10M CSVs contain direct download URLs like:
    https://ak.picdn.net/shutterstock/videos/21157780/preview/stock-footage-...mp4

Usage:
    # Download first N videos from WebVid-10M
    python data/download_videos.py --source webvid --limit 10000 --workers 16

    # Download from Panda-70M
    python data/download_videos.py --source panda --limit 5000 --workers 16

    # Download both (for Stage 2 training)
    python data/download_videos.py --source all --limit 50000 --workers 32
"""
import argparse
import csv
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
DATASETS = BASE / "datasets"
READY = BASE / "ready"
VIDEO_DIR = READY / "videos"


def download_one(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a single video file using curl."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True  # Already downloaded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")

    try:
        result = subprocess.run(
            [
                "curl", "-sS", "-L",
                "--connect-timeout", "10",
                "--max-time", str(timeout),
                "-o", str(tmp_path),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )

        if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1000:
            tmp_path.rename(output_path)
            return True
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def iter_webvid_entries(limit: int = -1):
    """Iterate over WebVid-10M entries from CSV partitions."""
    meta_dir = DATASETS / "webvid_10m" / "metadata" / "data" / "train" / "partitions"
    if not meta_dir.exists():
        log.error("WebVid-10M metadata not found at %s", meta_dir)
        return

    csv_files = sorted(meta_dir.glob("*.csv"))
    count = 0

    for csv_file in csv_files:
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_id = row.get("videoid", "").strip()
                    url = row.get("contentUrl", "").strip()
                    caption = row.get("name", "").strip()
                    duration = row.get("duration", "").strip()

                    if not video_id or not url:
                        continue

                    yield {
                        "video_id": video_id,
                        "url": url,
                        "caption": caption,
                        "duration": duration,
                        "source": "webvid",
                    }

                    count += 1
                    if limit > 0 and count >= limit:
                        return
        except Exception as e:
            log.warning("Error reading %s: %s", csv_file, e)


def iter_panda_entries(limit: int = -1):
    """Iterate over Panda-70M entries from CSV."""
    # Try 2M subset first (smaller), then 10M
    for csv_name in ["panda70m_training_2m.csv", "panda70m_training_10m.csv"]:
        csv_path = DATASETS / "panda_70m" / "csv" / csv_name
        if csv_path.exists():
            break
    else:
        log.error("No Panda-70M CSV found")
        return

    count = 0
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get("videoID", "").strip()
                url = row.get("url", "").strip()
                caption = row.get("caption", "").strip()
                timestamp = row.get("timestamp", "").strip()

                if not video_id or not url:
                    continue

                # Panda-70M URLs are YouTube links — need yt-dlp
                yield {
                    "video_id": video_id,
                    "url": url,
                    "caption": caption,
                    "timestamp": timestamp,
                    "source": "panda",
                }

                count += 1
                if limit > 0 and count >= limit:
                    return
    except Exception as e:
        log.warning("Error reading %s: %s", csv_path, e)


def download_webvid(limit: int, workers: int, timeout: int):
    """Download WebVid-10M videos (direct MP4 URLs)."""
    out_dir = VIDEO_DIR / "webvid"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = list(iter_webvid_entries(limit))
    log.info("WebVid-10M: downloading %d videos with %d workers", len(entries), workers)

    # Save caption mapping
    captions = {}
    tasks = []
    for entry in entries:
        vid_id = entry["video_id"]
        url = entry["url"]
        captions[vid_id] = entry["caption"]

        # Determine output filename
        ext = Path(urlparse(url).path).suffix or ".mp4"
        out_path = out_dir / f"{vid_id}{ext}"
        tasks.append((url, out_path))

    # Save captions
    caption_file = READY / "captions" / "videos_webvid.json"
    caption_file.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_file, "w") as f:
        import json
        json.dump(captions, f)
    log.info("Saved %d WebVid captions to %s", len(captions), caption_file)

    # Download in parallel
    success = 0
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, url, path, timeout): (url, path)
            for url, path in tasks
        }

        for i, future in enumerate(as_completed(futures)):
            url, path = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                log.info(
                    "Progress: %d/%d (%.0f/s) | OK: %d | Failed: %d",
                    i + 1, len(tasks), rate, success, failed,
                )

    elapsed = time.time() - start
    log.info(
        "WebVid download complete: %d OK, %d failed (%.1f min)",
        success, failed, elapsed / 60,
    )
    return success


def download_panda(limit: int, workers: int, timeout: int):
    """Download Panda-70M videos (YouTube URLs — needs yt-dlp)."""
    out_dir = VIDEO_DIR / "panda"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.error(
            "yt-dlp not found. Panda-70M uses YouTube URLs. Install with:\n"
            "  pip install yt-dlp"
        )
        return 0

    entries = list(iter_panda_entries(limit))
    log.info("Panda-70M: downloading %d videos with yt-dlp", len(entries))

    # Save captions
    captions = {}
    for entry in entries:
        captions[entry["video_id"]] = entry["caption"]

    caption_file = READY / "captions" / "videos_panda.json"
    caption_file.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_file, "w") as f:
        import json
        json.dump(captions, f)

    success = 0
    failed = 0
    start = time.time()

    for i, entry in enumerate(entries):
        vid_id = entry["video_id"]
        url = entry["url"]
        ts = entry.get("timestamp", "")
        out_path = out_dir / f"{vid_id}.mp4"

        if out_path.exists() and out_path.stat().st_size > 1000:
            success += 1
            continue

        try:
            cmd = [
                "yt-dlp",
                "-f", "worst[ext=mp4]",  # smallest quality for training
                "--no-playlist",
                "-o", str(out_path),
                "--socket-timeout", str(timeout),
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
            if result.returncode == 0 and out_path.exists():
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            log.info("Panda progress: %d/%d | OK: %d | Failed: %d", i + 1, len(entries), success, failed)

    elapsed = time.time() - start
    log.info("Panda download complete: %d OK, %d failed (%.1f min)", success, failed, elapsed / 60)
    return success


def merge_video_captions():
    """Merge all video caption files into one."""
    import json
    caption_dir = READY / "captions"
    merged = {}

    for f in caption_dir.glob("videos_*.json"):
        with open(f) as fh:
            data = json.load(fh)
            merged.update(data)

    merged_path = caption_dir / "videos.json"
    with open(merged_path, "w") as fh:
        json.dump(merged, fh)
    log.info("Merged %d video captions into %s", len(merged), merged_path)


def main():
    parser = argparse.ArgumentParser(description="Download videos for MAVT training")
    parser.add_argument("--source", choices=["webvid", "panda", "all"], default="webvid",
                        help="Which dataset to download from")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Max number of videos to download (default: 10000)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel download workers (default: 16)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-video download timeout in seconds (default: 30)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("MAVT Video Downloader")
    log.info("=" * 70)
    log.info("Source: %s", args.source)
    log.info("Limit: %d videos", args.limit)
    log.info("Workers: %d", args.workers)
    log.info("Output: %s", VIDEO_DIR)
    log.info("=" * 70)

    total = 0

    if args.source in ("webvid", "all"):
        total += download_webvid(args.limit, args.workers, args.timeout)

    if args.source in ("panda", "all"):
        panda_limit = args.limit // 5 if args.source == "all" else args.limit
        total += download_panda(panda_limit, args.workers, args.timeout)

    # Merge captions
    merge_video_captions()

    log.info("")
    log.info("=" * 70)
    log.info("Total videos downloaded: %d", total)
    log.info("Videos directory: %s", VIDEO_DIR)

    # Count actual files
    n_mp4 = 0
    for ext in ["*.mp4", "*.webm"]:
        result = subprocess.run(
            ["find", "-L", str(VIDEO_DIR), "-name", ext, "-type", "f"],
            capture_output=True, text=True,
        )
        n_mp4 += len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    log.info("Total video files on disk: %d", n_mp4)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
