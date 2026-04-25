"""
prepare_dataset.py
------------------
Organize downloaded data into the universal data root layout expected by
the MAVT training pipeline.

Target layout (matching sample100-1 format):
    data_root/
        images/          <- JPEG files from ImageNet WDS shards
        videos/          <- MP4 files from WebVid10M (dataset_10m/)
        3d_objects/
            renders/     <- triplane PNGs from Objaverse .glb files
                <uid>/
                    oxoy.png
                    oxoz.png
                    oyoz.png
        captions/
            images.json  <- {stem: label_name}  from ImageNet
            videos.json  <- {stem: caption}     from WebVid10M .txt/.json
            3d.json      <- {uid: category}     from Objaverse LVIS
        manifests/
            images.txt
            videos.txt
            3d.txt

Usage:
    # Organize all available data
    python prepare_dataset.py \
        --data_root ./data/universal \
        --image_shards_dir ./dataset/image10k/train \
        --video_shards_dir ./dataset/dataset_10m \
        --objaverse_dir ./dataset/objaverse

    # Only images
    python prepare_dataset.py \
        --data_root ./data/universal \
        --image_shards_dir ./dataset/image10k/train

    # Status check
    python prepare_dataset.py --data_root ./data/universal --status
"""

import os, io, json, argparse, tarfile, glob, shutil
from pathlib import Path
from tqdm import tqdm


def prepare_images(shards_dir: str, data_root: Path, max_images: int = None):
    """Extract images from WebDataset .tar shards into images/ directory."""
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions = {}

    shards = sorted(glob.glob(os.path.join(shards_dir, "*.tar")))
    if not shards:
        print(f"  [WARN] No .tar shards found in {shards_dir}")
        return

    count = 0
    for shard_path in tqdm(shards, desc="Extracting images"):
        try:
            with tarfile.open(shard_path, "r") as tar:
                members = tar.getmembers()
                for m in members:
                    if m.name.endswith(".jpg") or m.name.endswith(".jpeg"):
                        key = Path(m.name).stem
                        out_path = images_dir / f"{key}.jpg"
                        if not out_path.exists():
                            f = tar.extractfile(m)
                            if f:
                                out_path.write_bytes(f.read())
                                count += 1
                    elif m.name.endswith(".txt"):
                        key = Path(m.name).stem
                        f = tar.extractfile(m)
                        if f:
                            captions[key] = f.read().decode("utf-8", errors="replace").strip()

                    if max_images and count >= max_images:
                        break
        except Exception as e:
            tqdm.write(f"  [WARN] Shard error {shard_path}: {e}")
        if max_images and count >= max_images:
            break

    # Save captions
    cap_path = data_root / "captions" / "images.json"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cap_path, "w") as f:
        json.dump(captions, f, indent=2)

    # Save manifest
    manifest_dir = data_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(images_dir.glob("*.jpg"))
    with open(manifest_dir / "images.txt", "w") as f:
        for p in img_files:
            f.write(p.name + "\n")

    print(f"  Images: {count:,} extracted -> {images_dir}")
    print(f"  Captions: {len(captions):,} -> {cap_path}")


def prepare_videos(video_shards_dir: str, data_root: Path, max_videos: int = None):
    """Symlink/copy videos from video2dataset shard format into videos/ dir."""
    videos_dir = data_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions = {}

    # video2dataset format: NNNNN/{NNNNNNNN.mp4, NNNNNNNN.txt, NNNNNNNN.json}
    shard_dirs = sorted(
        d for d in Path(video_shards_dir).iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not shard_dirs:
        print(f"  [WARN] No shard directories found in {video_shards_dir}")
        return

    count = 0
    for shard_dir in tqdm(shard_dirs, desc="Linking videos"):
        mp4_files = sorted(shard_dir.glob("*.mp4"))
        for mp4 in mp4_files:
            key = f"{shard_dir.name}_{mp4.stem}"  # e.g. 00000_00000001
            dst = videos_dir / f"{key}.mp4"

            # Symlink instead of copy to save disk
            if not dst.exists():
                try:
                    dst.symlink_to(mp4.resolve())
                    count += 1
                except OSError:
                    shutil.copy2(mp4, dst)
                    count += 1

            # Read caption from .txt or .json
            txt_file = shard_dir / f"{mp4.stem}.txt"
            json_file = shard_dir / f"{mp4.stem}.json"
            if txt_file.exists():
                captions[key] = txt_file.read_text(errors="replace").strip()
            elif json_file.exists():
                try:
                    meta = json.loads(json_file.read_text())
                    captions[key] = meta.get("caption", "")
                except Exception:
                    pass

            if max_videos and count >= max_videos:
                break
        if max_videos and count >= max_videos:
            break

    # Save captions
    cap_path = data_root / "captions" / "videos.json"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cap_path, "w") as f:
        json.dump(captions, f, indent=2)

    # Save manifest
    manifest_dir = data_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    vid_files = sorted(videos_dir.glob("*.mp4"))
    with open(manifest_dir / "videos.txt", "w") as f:
        for p in vid_files:
            f.write(p.name + "\n")

    print(f"  Videos: {count:,} linked -> {videos_dir}")
    print(f"  Captions: {len(captions):,} -> {cap_path}")


def prepare_3d(objaverse_dir: str, data_root: Path, max_objects: int = None):
    """Index downloaded Objaverse .glb files and create 3D metadata.

    Note: Triplane rendering (.glb -> oxoy/oxoz/oyoz PNGs) requires a
    separate rendering step with PyTorch3D or Blender. This function
    creates the metadata and links the raw .glb files.
    """
    threed_dir = data_root / "3d_objects"
    renders_dir = threed_dir / "renders"
    raw_dir = threed_dir / "raw_glb"
    renders_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Find downloaded .glb files
    obja_path = Path(objaverse_dir)
    glb_files = list(obja_path.rglob("*.glb"))
    glb_gz_files = list(obja_path.rglob("*.glb.gz"))
    all_files = glb_files + glb_gz_files

    if not all_files:
        print(f"  [WARN] No .glb files found in {objaverse_dir}")
        print("  Run obja.py first to download objects.")
        return

    # Load LVIS annotations for captions/categories
    captions = {}
    try:
        import objaverse
        lvis = objaverse.load_lvis_annotations()
        uid_to_cat = {}
        for cat, uids in lvis.items():
            for uid in uids:
                uid_to_cat[uid] = cat
    except Exception:
        uid_to_cat = {}

    count = 0
    for glb_path in tqdm(all_files[:max_objects], desc="Indexing 3D"):
        uid = glb_path.stem.replace(".glb", "")
        dst = raw_dir / f"{uid}.glb"
        if not dst.exists():
            try:
                dst.symlink_to(glb_path.resolve())
            except OSError:
                shutil.copy2(glb_path, dst)

        cat = uid_to_cat.get(uid, "unknown")
        captions[uid] = cat
        count += 1

    # Save captions
    cap_path = data_root / "captions" / "3d.json"
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cap_path, "w") as f:
        json.dump(captions, f, indent=2)

    # Save manifest
    manifest_dir = data_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "3d.txt", "w") as f:
        for uid in sorted(captions.keys()):
            f.write(uid + "\n")

    print(f"  3D objects: {count:,} indexed -> {raw_dir}")
    print(f"  Categories: {len(captions):,} -> {cap_path}")
    if count > 0 and not list(renders_dir.iterdir()):
        print("  [NOTE] Triplane renders not yet generated.")
        print("         Run rendering pipeline to create oxoy/oxoz/oyoz PNGs")


def print_status(data_root: Path):
    """Print status of all data in the universal root."""
    print(f"\n{'='*60}")
    print(f"  Data Root: {data_root}")
    print(f"{'='*60}")

    for name, subdir, ext in [
        ("Images", "images", "*.jpg"),
        ("Videos", "videos", "*.mp4"),
        ("3D Raw", "3d_objects/raw_glb", "*.glb"),
        ("3D Renders", "3d_objects/renders", "*"),
    ]:
        d = data_root / subdir
        if d.exists():
            files = list(d.glob(ext))
            dirs = [x for x in d.iterdir() if x.is_dir()] if ext == "*" else []
            n = len(files) or len(dirs)
            print(f"  {name:15s}: {n:,}")
        else:
            print(f"  {name:15s}: (not found)")

    cap_dir = data_root / "captions"
    if cap_dir.exists():
        for f in sorted(cap_dir.glob("*.json")):
            data = json.loads(f.read_text())
            print(f"  Caption {f.stem:8s}: {len(data):,} entries")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Organize data into universal training layout"
    )
    parser.add_argument("--data_root", required=True,
                        help="Output universal data root directory")
    parser.add_argument("--image_shards_dir", default=None,
                        help="Dir with image WDS .tar shards")
    parser.add_argument("--video_shards_dir", default=None,
                        help="Dir with video2dataset shard dirs")
    parser.add_argument("--objaverse_dir", default=None,
                        help="Dir with downloaded Objaverse .glb files")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument("--status", action="store_true",
                        help="Print status and exit")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status(data_root)
        return

    print(f"\nPreparing dataset at: {data_root}\n")

    if args.image_shards_dir:
        print("[1] Images...")
        prepare_images(args.image_shards_dir, data_root, args.max_images)

    if args.video_shards_dir:
        print("\n[2] Videos...")
        prepare_videos(args.video_shards_dir, data_root, args.max_videos)

    if args.objaverse_dir:
        print("\n[3] 3D Objects...")
        prepare_3d(args.objaverse_dir, data_root, args.max_objects)

    print_status(data_root)


if __name__ == "__main__":
    main()
