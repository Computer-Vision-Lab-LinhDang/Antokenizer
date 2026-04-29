"""
create_sample_data.py
---------------------
Create a sample multi-modal data directory for MAVT training
using REAL data from existing datasets:

  - 10 image shards   (symlinked from dataset/image10k/train/)
  - 100 videos + captions (symlinked from dataset/dataset_10m/00000/)
  - 10 3D objects      (rendered from ~/.objaverse GLBs → triplane PNGs)

Output: data/sample_multimodal/

Usage:
    python create_sample_data.py
    python create_sample_data.py --output data/my_sample --n_images 5 --n_videos 50 --n_3d 10
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
# 1. IMAGE SHARDS — symlink .tar files from image10k
# ═══════════════════════════════════════════════════════════════════════════
def setup_images(output: Path, n_shards: int = 10) -> int:
    img_src = BASE / "dataset" / "image10k" / "train"
    shards = sorted(img_src.glob("shard_*.tar"))[:n_shards]
    if not shards:
        print("[WARN] No image shards found in dataset/image10k/train/")
        return 0

    for shard in shards:
        dst = output / shard.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(shard.resolve())

    print(f"[IMG ] Symlinked {len(shards)} image shards → {output}/")
    return len(shards)


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIDEOS — symlink mp4s + extract captions from txt
# ═══════════════════════════════════════════════════════════════════════════
def setup_videos(output: Path, n_videos: int = 100) -> int:
    vid_src = BASE / "dataset" / "dataset_10m" / "00000"
    vid_dst = output / "videos"
    vid_dst.mkdir(parents=True, exist_ok=True)

    if not vid_src.exists():
        print("[WARN] Video source not found: dataset/dataset_10m/00000/")
        return 0

    captions = {}
    mp4s = sorted(vid_src.glob("*.mp4"))[:n_videos]

    for mp4 in mp4s:
        key = mp4.stem
        # Symlink video (fast, saves disk)
        link = vid_dst / mp4.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(mp4.resolve())

        # Read caption
        txt = vid_src / f"{key}.txt"
        caption = ""
        if txt.exists():
            try:
                caption = txt.read_text(errors="replace").strip()
            except Exception:
                pass
        captions[key] = caption

    cap_dir = output / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    with open(cap_dir / "videos.json", "w") as f:
        json.dump(captions, f, indent=2)

    print(f"[VID ] Symlinked {len(mp4s)} videos + captions → {vid_dst}/")
    return len(mp4s)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 3D OBJECTS — render GLBs from ~/.objaverse into triplane PNGs
# ═══════════════════════════════════════════════════════════════════════════

def _parse_glb_vertices(glb_path: str) -> np.ndarray:
    """Extract vertex positions from a GLB (glTF Binary) file.

    Parses the GLB header, reads the JSON chunk to find the POSITION
    accessor, then reads raw float32 vertex data from the binary chunk.
    Returns Nx3 float32 array of vertex positions.
    """
    with open(glb_path, "rb") as f:
        # GLB Header: magic(4) + version(4) + length(4)
        magic = f.read(4)
        if magic != b"glTF":
            raise ValueError(f"Not a GLB file: {glb_path}")
        _version = struct.unpack("<I", f.read(4))[0]
        _total = struct.unpack("<I", f.read(4))[0]

        # Chunk 0: JSON
        json_len = struct.unpack("<I", f.read(4))[0]
        json_type = f.read(4)
        json_data = json.loads(f.read(json_len).decode("utf-8"))

        # Chunk 1: BIN
        bin_len = struct.unpack("<I", f.read(4))[0]
        _bin_type = f.read(4)
        bin_data = f.read(bin_len)

    # Find POSITION accessor from the first mesh primitive
    meshes = json_data.get("meshes", [])
    if not meshes:
        raise ValueError("No meshes in GLB")

    accessors = json_data.get("accessors", [])
    buffer_views = json_data.get("bufferViews", [])

    all_verts = []
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            pos_idx = prim.get("attributes", {}).get("POSITION")
            if pos_idx is None:
                continue
            acc = accessors[pos_idx]
            bv = buffer_views[acc.get("bufferView", 0)]
            offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            count = acc["count"]
            # Read float32 x 3
            raw = bin_data[offset: offset + count * 12]
            verts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)
            all_verts.append(verts)

    if not all_verts:
        raise ValueError("No POSITION data found")
    return np.concatenate(all_verts, axis=0)


def _render_orthographic(vertices: np.ndarray, axis_pair: tuple,
                         resolution: int = 256) -> Image.Image:
    """Render an orthographic projection of vertices onto a 2D image.

    axis_pair: (col_axis, row_axis) indices into the xyz coords.
      oxoy → (0, 1) = X horizontal, Y vertical (front view)
      oxoz → (0, 2) = X horizontal, Z vertical (top view)
      oyoz → (1, 2) = Y horizontal, Z vertical (side view)
    """
    ax, ay = axis_pair
    pts2d = vertices[:, [ax, ay]]

    # Normalize to [margin, resolution-margin]
    margin = 10
    mn = pts2d.min(axis=0)
    mx = pts2d.max(axis=0)
    span = mx - mn
    span[span < 1e-6] = 1.0  # avoid div-by-zero
    pts_norm = (pts2d - mn) / span  # [0, 1]
    pts_px = pts_norm * (resolution - 2 * margin) + margin

    # Flip Y axis (image coords)
    pts_px[:, 1] = resolution - pts_px[:, 1]

    # Create image
    img = Image.new("RGB", (resolution, resolution), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Draw points with depth-based coloring
    other_axis = ({0, 1, 2} - {ax, ay}).pop()
    depths = vertices[:, other_axis]
    d_min, d_max = depths.min(), depths.max()
    d_range = d_max - d_min if (d_max - d_min) > 1e-6 else 1.0
    d_norm = (depths - d_min) / d_range  # 0=near, 1=far

    # Sort by depth (far first → near on top)
    order = np.argsort(-d_norm)
    for i in order:
        x, y = pts_px[i]
        d = d_norm[i]
        # Color: near = bright warm, far = dim cool
        r = int(80 + 175 * (1 - d))
        g = int(120 + 100 * (1 - d * 0.5))
        b = int(180 + 75 * d)
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(r, g, b))

    return img


def setup_3d(output: Path, n_objects: int = 10) -> int:
    """Find real GLB files from ~/.objaverse and render triplane PNGs."""
    objaverse_dir = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"

    if not objaverse_dir.exists():
        print(f"[WARN] Objaverse cache not found: {objaverse_dir}")
        return 0

    # Collect all GLB files
    glb_files = []
    for sub in sorted(objaverse_dir.iterdir()):
        if sub.is_dir():
            for glb in sorted(sub.glob("*.glb")):
                glb_files.append(glb)
                if len(glb_files) >= n_objects:
                    break
        if len(glb_files) >= n_objects:
            break

    if not glb_files:
        print("[WARN] No GLB files found in ~/.objaverse/")
        return 0

    obj_dir = output / "3d_objects" / "renders"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # Plane definitions: name → (horizontal_axis, vertical_axis)
    planes = {
        "oxoy": (0, 1),  # Front: X-right, Y-up
        "oxoz": (0, 2),  # Top:   X-right, Z-up
        "oyoz": (1, 2),  # Side:  Y-right, Z-up
    }

    captions = {}
    rendered = 0

    for glb_path in glb_files:
        obj_id = glb_path.stem
        render_dir = obj_dir / obj_id
        render_dir.mkdir(parents=True, exist_ok=True)

        try:
            verts = _parse_glb_vertices(str(glb_path))
            print(f"  [3D] {obj_id}: {len(verts)} vertices", end="")

            for plane_name, axes in planes.items():
                img = _render_orthographic(verts, axes, resolution=256)
                img.save(render_dir / f"{plane_name}.png")

            captions[obj_id] = f"3D object {obj_id}"
            rendered += 1
            print(" ✓")

        except Exception as e:
            print(f"  [3D] {obj_id}: FAILED ({e})")
            # Clean up partial render
            import shutil
            if render_dir.exists():
                shutil.rmtree(render_dir)
            continue

    # Write captions
    cap_dir = output / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    with open(cap_dir / "3d.json", "w") as f:
        json.dump(captions, f, indent=2)

    print(f"[3D  ] Rendered {rendered}/{len(glb_files)} objects → {obj_dir}/")
    return rendered


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Create sample multi-modal dataset from real data"
    )
    parser.add_argument("--output", default="data/sample_multimodal",
                        help="Output directory (default: data/sample_multimodal)")
    parser.add_argument("--n_images", type=int, default=10,
                        help="Number of image shards to symlink")
    parser.add_argument("--n_videos", type=int, default=100,
                        help="Number of videos to symlink")
    parser.add_argument("--n_3d", type=int, default=10,
                        help="Number of 3D objects to render")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = BASE / output
    output.mkdir(parents=True, exist_ok=True)

    print(f"Creating sample multi-modal dataset at: {output}\n")
    print(f"Sources:")
    print(f"  Images : dataset/image10k/train/ (WDS shards)")
    print(f"  Videos : dataset/dataset_10m/00000/ (mp4 + txt)")
    print(f"  3D     : ~/.objaverse/hf-objaverse-v1/glbs/ (GLB → triplane PNG)")
    print()

    n_img = setup_images(output, args.n_images)
    n_vid = setup_videos(output, args.n_videos)
    n_3d = setup_3d(output, args.n_3d)

    # Ensure captions dir has images.json (WDS shards carry their own captions)
    cap_dir = output / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    if not (cap_dir / "images.json").exists():
        with open(cap_dir / "images.json", "w") as f:
            json.dump({}, f)

    print(f"\n{'='*60}")
    print(f" Sample dataset ready: {output}")
    print(f"   Images : {n_img} shards (~{n_img * 1000} samples)")
    print(f"   Videos : {n_vid} clips")
    print(f"   3D     : {n_3d} objects (triplane renders from real GLBs)")
    print(f"{'='*60}")
    print(f"\n Config usage:")
    print(f"   data:")
    print(f"     universal_data_root: {output.relative_to(BASE)}")
    print(f"     active_modalities: [image, video, threed]")


if __name__ == "__main__":
    main()
