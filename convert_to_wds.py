"""
convert_to_wds.py
-----------------
Convert image datasets (HuggingFace Hub) -> WebDataset .tar shards.

Streams data directly from HuggingFace Hub -- no local parquet files needed.

Usage:
    python convert_to_wds.py \
        --dataset_name  ILSVRC/imagenet-1k \
        --output_dir    ./dataset/image10k \
        --split         train \
        --shard_size    1000 \
        --max_samples   10000

    # Resume interrupted conversion
    python convert_to_wds.py \
        --dataset_name  ILSVRC/imagenet-1k \
        --output_dir    ./dataset/image10k \
        --split         train --resume
"""

import os, io, json, argparse
from pathlib import Path

import webdataset as wds
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ── Progress tracking ────────────────────────────────────────────────────────
class ConversionProgress:
    def __init__(self, output_dir: Path, split: str):
        self.progress_file = output_dir / f"{split}_progress.json"
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.data = json.load(f)
        else:
            self.data = {"samples_written": 0, "errors": 0, "status": "running"}

    @property
    def samples_written(self) -> int:
        return self.data["samples_written"]

    def update(self, samples: int, errors: int):
        self.data.update(samples_written=samples, errors=errors, status="running")
        self._save()

    def finalize(self):
        self.data["status"] = "completed"
        self._save()

    def _save(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Core converter ───────────────────────────────────────────────────────────
def convert_hf_to_shards(
    dataset_name: str, output_dir: str, split: str = "train",
    shard_size: int = 1000, image_size: int = 256, jpeg_quality: int = 95,
    max_samples: int = None, image_column: str = "image",
    label_column: str = "label", resume: bool = False, hf_token: str = None,
):
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[WARN] No HF_TOKEN found. Some datasets require authentication.")

    # ── Load dataset ─────────────────────────────────────────────────────────
    # Try non-streaming first (downloads parquets then iterates locally = much faster).
    # Fall back to streaming for very large datasets or if non-streaming fails.
    is_streaming = False
    try:
        print(f"\n[INFO] Loading '{dataset_name}' split='{split}' (non-streaming)...")
        ds = load_dataset(dataset_name, split=split, token=token)
        print(f"[INFO] Loaded {len(ds):,} samples")
    except Exception as e1:
        print(f"[INFO] Non-streaming failed ({e1}), trying streaming...")
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True, token=token)
            is_streaming = True
        except Exception as e2:
            print(f"[ERROR] Failed to load: {e2}")
            raise

    features = ds.features
    if image_column not in features:
        img_cols = [k for k, v in features.items() if "image" in str(v).lower()]
        if img_cols:
            image_column = img_cols[0]
            print(f"[INFO] Auto-detected image column: '{image_column}'")
    print(f"[INFO] Features: {list(features.keys())}")
    print(f"[INFO] Mode: {'streaming' if is_streaming else 'non-streaming (fast)'}")

    # ── Class map ───────────────────────────────────────────────────────────
    class_map = {}
    if label_column in features:
        try:
            lf = features[label_column]
            if hasattr(lf, 'names'):
                class_map = {i: name for i, name in enumerate(lf.names)}
        except Exception:
            pass
    if class_map:
        cm_path = Path(output_dir) / "class_map.json"
        with open(cm_path, "w") as f:
            json.dump(class_map, f, indent=2)
        print(f"[INFO] Saved class_map -> {cm_path} ({len(class_map)} classes)")

    # ── Resume ──────────────────────────────────────────────────────────────
    progress = ConversionProgress(Path(output_dir), split)
    skip_count = progress.samples_written if resume else 0
    if skip_count:
        print(f"[RESUME] Skipping first {skip_count:,} samples")

    # ── Transform ───────────────────────────────────────────────────────────
    resize_tf = transforms.Resize(image_size,
                                  interpolation=transforms.InterpolationMode.BICUBIC)

    # ── Write shards ────────────────────────────────────────────────────────
    shard_pattern = str(output_path / "shard_%05d.tar")
    start_shard = skip_count // shard_size if resume else 0
    stats = {"total": 0, "errors": 0}

    total_est = len(ds) if not is_streaming and hasattr(ds, '__len__') else max_samples
    with wds.ShardWriter(shard_pattern, maxcount=shard_size, compress=False,
                         start_shard=start_shard) as sink:
        pbar = tqdm(desc=f"Converting {split}", unit="img", total=total_est,
                    dynamic_ncols=True)
        for idx, sample in enumerate(ds):
            if idx < skip_count:
                continue
            if max_samples and idx >= max_samples:
                break
            try:
                img = sample.get(image_column)
                label = sample.get(label_column, -1)
                if img is None:
                    stats["errors"] += 1
                    continue

                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if max(img.size) > image_size * 2:
                    img = resize_tf(img)

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)

                class_name = class_map.get(label, str(label)) if isinstance(label, int) else str(label)
                sink.write({
                    "__key__": f"{idx:08d}",
                    "jpg": buf.getvalue(),
                    "cls": str(label).encode(),
                    "txt": class_name.encode(),
                })
                stats["total"] += 1
                pbar.update(1)

                if stats["total"] % 5000 == 0:
                    progress.update(skip_count + stats["total"], stats["errors"])
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 10:
                    tqdm.write(f"[WARN] Error sample {idx}: {e}")
        pbar.close()

    progress.update(skip_count + stats["total"], stats["errors"])
    progress.finalize()

    shards = sorted(output_path.glob("*.tar"))
    print(f"\n{'='*50}")
    print(f"[{split}] Done!  Dataset: {dataset_name}")
    print(f"  Written : {stats['total']:,}  Errors: {stats['errors']}  Shards: {len(shards)}")
    if shards:
        sz = sum(s.stat().st_size for s in shards) / 1e9
        print(f"  Size    : {sz:.2f} GB")

    shard_list = Path(output_dir) / f"{split}_shards.txt"
    with open(shard_list, "w") as f:
        for s in shards:
            f.write(str(s) + "\n")
    print(f"  Shard list: {shard_list}")
    print(f"{'='*50}\n")
    return [str(s) for s in shards]


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Convert HF image datasets -> WDS shards")
    p.add_argument("--dataset_name", required=True, help="HF dataset name")
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--split",        default="train")
    p.add_argument("--shard_size",   type=int, default=1000)
    p.add_argument("--image_size",   type=int, default=256)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--max_samples",  type=int, default=None)
    p.add_argument("--image_column", default="image")
    p.add_argument("--label_column", default="label")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--hf_token",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        convert_hf_to_shards(
            dataset_name=args.dataset_name, output_dir=args.output_dir,
            split=split, shard_size=args.shard_size, image_size=args.image_size,
            jpeg_quality=args.jpeg_quality, max_samples=args.max_samples,
            image_column=args.image_column, label_column=args.label_column,
            resume=args.resume, hf_token=args.hf_token,
        )
