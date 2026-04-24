"""
download_objaverse.py
---------------------
Download Objaverse 1.0 LVIS subset with robust retry and resume support.

Fixes from previous version:
  - Batch downloads instead of 1-at-a-time
  - Exponential backoff retry (3 attempts per batch)
  - Reduced worker count to avoid rate limiting
  - File integrity validation after download
  - --retry_failed mode to re-attempt only failed UIDs

Usage:
    # Download with retry (first run or resume)
    python obja.py --output_dir ./dataset/objaverse --resume

    # Retry only previously failed UIDs
    python obja.py --output_dir ./dataset/objaverse --retry_failed

    # Small test
    python obja.py --output_dir ./dataset/objaverse --max_objects 50

    # Filter by category
    python obja.py --output_dir ./dataset/objaverse --categories chair table
"""

import os, json, time, gzip, argparse, traceback
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import objaverse
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Progress tracker
# ─────────────────────────────────────────────────────────────
class ProgressTracker:
    def __init__(self, output_dir: Path):
        self.progress_file = output_dir / "progress.json"
        self.failed_file   = output_dir / "failed.txt"
        self.stats_file    = output_dir / "stats.json"
        self.output_dir    = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total": 0, "downloaded": 0, "failed": 0, "skipped": 0,
                "completed_uids": [], "failed_uids": [],
            }
        self._start_time = time.time()
        self._session_count = 0

    def is_done(self, uid: str) -> bool:
        return uid in self.progress["completed_uids"]

    def mark_success(self, uid: str, file_path: str):
        if uid not in self.progress["completed_uids"]:
            self.progress["completed_uids"].append(uid)
            self.progress["downloaded"] += 1
            self._session_count += 1
        # Remove from failed if previously failed
        if uid in self.progress["failed_uids"]:
            self.progress["failed_uids"].remove(uid)
            self.progress["failed"] = max(0, self.progress["failed"] - 1)
        self._save()

    def mark_failed(self, uid: str, error: str):
        if uid not in self.progress["failed_uids"]:
            self.progress["failed_uids"].append(uid)
            self.progress["failed"] += 1
        with open(self.failed_file, "a") as f:
            f.write(f"{uid}\t{error}\n")
        self._save()

    def mark_skipped(self):
        self.progress["skipped"] += 1

    def set_total(self, total: int):
        self.progress["total"] = total
        self._save()

    def reset_for_retry(self):
        """Reset failed state for retry_failed mode."""
        self.progress["failed"] = 0
        self.progress["failed_uids"] = []
        # Clear failed.txt
        if self.failed_file.exists():
            self.failed_file.rename(
                self.failed_file.with_suffix(f".bak.{int(time.time())}")
            )
        self._save()

    def get_failed_uids(self) -> list:
        return list(self.progress.get("failed_uids", []))

    def finalize(self):
        self.progress["status"] = "completed"
        self.progress["updated_at"] = datetime.now().isoformat()
        self._save()
        self._save_stats()

    def _save(self):
        self.progress["updated_at"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def _save_stats(self):
        elapsed = time.time() - self._start_time
        stats = {
            "total": self.progress["total"],
            "downloaded": self.progress["downloaded"],
            "failed": self.progress["failed"],
            "skipped": self.progress["skipped"],
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_human": str(timedelta(seconds=int(elapsed))),
            "avg_speed_per_min": round(
                self._session_count / max(elapsed / 60, 0.01), 1
            ),
        }
        with open(self.stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    def summary(self) -> str:
        p = self.progress
        elapsed = time.time() - self._start_time
        speed = self._session_count / max(elapsed / 60, 0.01)
        remaining = p["total"] - p["downloaded"] - p["failed"] - p["skipped"]
        eta_min = remaining / max(speed, 0.1)
        return (
            f"\n{'='*55}\n"
            f"  Progress: {p['downloaded']:,} / {p['total']:,}\n"
            f"  Failed  : {p['failed']:,}\n"
            f"  Skipped : {p['skipped']:,}\n"
            f"  Speed   : {speed:.1f} obj/min\n"
            f"  ETA     : {timedelta(seconds=int(eta_min * 60))}\n"
            f"{'='*55}"
        )


# ─────────────────────────────────────────────────────────────
# Download with retry
# ─────────────────────────────────────────────────────────────
def validate_glb(filepath: str) -> bool:
    """Check that a downloaded .glb or .glb.gz is valid."""
    try:
        if not os.path.exists(filepath):
            return False
        size = os.path.getsize(filepath)
        if size < 100:  # Too small to be real
            return False
        # If gzipped, try decompressing header
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rb") as f:
                f.read(100)
        return True
    except Exception:
        return False


def download_batch_with_retry(
    uids: list, save_dir: Path, max_retries: int = 3,
    base_delay: float = 5.0, download_processes: int = 3,
) -> dict:
    """Download a batch of UIDs with exponential backoff retry.

    Returns dict {uid: filepath} for successful downloads.
    """
    remaining = list(uids)
    results = {}

    for attempt in range(max_retries):
        if not remaining:
            break

        try:
            batch_result = objaverse.load_objects(
                uids=remaining,
                download_processes=download_processes,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (3 ** attempt)
                tqdm.write(
                    f"  [Retry {attempt+1}/{max_retries}] Batch error: "
                    f"{str(e)[:60]}. Waiting {delay:.0f}s..."
                )
                time.sleep(delay)
                continue
            else:
                break

        # Check which ones succeeded
        newly_done = []
        for uid in remaining:
            if uid in batch_result:
                fp = batch_result[uid]
                if validate_glb(fp):
                    results[uid] = fp
                    newly_done.append(uid)

        remaining = [u for u in remaining if u not in newly_done]

        if remaining and attempt < max_retries - 1:
            delay = base_delay * (3 ** attempt)
            tqdm.write(
                f"  [Retry {attempt+1}] {len(newly_done)} ok, "
                f"{len(remaining)} remaining. Waiting {delay:.0f}s..."
            )
            time.sleep(delay)

    return results


def download_all(
    uids: list, save_dir: Path, tracker: ProgressTracker,
    num_workers: int = 4, batch_size: int = 50, resume: bool = True,
):
    """Download objects in batches with progress tracking."""
    if resume:
        todo = [uid for uid in uids if not tracker.is_done(uid)]
        skipped = len(uids) - len(todo)
        if skipped > 0:
            print(f"  Resume: skip {skipped:,} already downloaded")
            for _ in range(skipped):
                tracker.mark_skipped()
    else:
        todo = uids

    if not todo:
        print("  All already downloaded!")
        return

    print(f"  To download: {len(todo):,} objects")
    print(f"  Batch size : {batch_size}, Workers: {num_workers}\n")

    # Split into batches
    batches = [todo[i:i+batch_size] for i in range(0, len(todo), batch_size)]

    with tqdm(total=len(todo), desc="Downloading", unit="obj",
              dynamic_ncols=True) as pbar:
        for batch_idx, batch_uids in enumerate(batches):
            results = download_batch_with_retry(
                uids=batch_uids, save_dir=save_dir,
                download_processes=num_workers,
            )

            for uid in batch_uids:
                if uid in results:
                    tracker.mark_success(uid, results[uid])
                else:
                    tracker.mark_failed(uid, "download/validation failed")
                pbar.update(1)

            # Summary every 10 batches
            if (batch_idx + 1) % 10 == 0:
                tqdm.write(tracker.summary())


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download Objaverse 1.0 LVIS subset"
    )
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--max_objects",  type=int, default=None)
    parser.add_argument("--num_workers",  type=int, default=4,
                        help="Download processes (default 4, lower=safer)")
    parser.add_argument("--batch_size",   type=int, default=50)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Retry only previously failed UIDs")
    parser.add_argument("--categories",   nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tracker    = ProgressTracker(output_dir)
    os.environ["OBJAVERSE_SAVE_PATH"] = str(output_dir)

    # ── Load LVIS annotations ────────────────────────────────
    print("\n[1/3] Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    print(f"  Categories: {len(lvis):,}")
    print(f"  Objects   : {sum(len(v) for v in lvis.values()):,}")

    # ── Select UIDs ──────────────────────────────────────────
    print("\n[2/3] Selecting UIDs...")

    if args.retry_failed:
        uids = tracker.get_failed_uids()
        print(f"  Retrying {len(uids):,} previously failed UIDs")
        tracker.reset_for_retry()
    elif args.categories:
        selected = []
        for cat in args.categories:
            if cat in lvis:
                selected.extend(lvis[cat])
                print(f"  {cat:30s}: {len(lvis[cat]):4d} objects")
            else:
                matches = [k for k in lvis if cat.lower() in k.lower()]
                for m in matches:
                    selected.extend(lvis[m])
                    print(f"  {m:30s}: {len(lvis[m]):4d} (matched '{cat}')")
        uids = list(set(selected))
    else:
        uids = list(set(uid for vals in lvis.values() for uid in vals))

    if args.max_objects:
        uids = uids[:args.max_objects]
        print(f"  [DEBUG] Limited to {args.max_objects} objects")

    print(f"  Selected UIDs: {len(uids):,}")

    uid_list_path = output_dir / "selected_uids.json"
    with open(uid_list_path, "w") as f:
        json.dump(uids, f)
    tracker.set_total(len(uids))

    # ── Download ─────────────────────────────────────────────
    print(f"\n[3/3] Downloading {len(uids):,} objects...")
    print(f"  Output  : {output_dir}")
    print(f"  Workers : {args.num_workers}")
    print(f"  Batch   : {args.batch_size}")

    try:
        download_all(
            uids=uids, save_dir=output_dir, tracker=tracker,
            num_workers=args.num_workers, batch_size=args.batch_size,
            resume=args.resume or args.retry_failed,
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Resume with --resume flag")
    finally:
        tracker.finalize()
        print(tracker.summary())


if __name__ == "__main__":
    main()
