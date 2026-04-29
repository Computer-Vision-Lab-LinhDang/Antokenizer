"""
Render GLB files from Objaverse into tripplane images.
CPU-only: uses pure NumPy z-buffer rasteriser — no OpenGL, no display.

Output structure:
    <output-dir>/<object-id>/oxoy.png   (front: camera along +Z, XY plane)
    <output-dir>/<object-id>/oxoz.png   (top:   camera along +Y, XZ plane)
    <output-dir>/<object-id>/oyoz.png   (side:  camera along +X, YZ plane)

Usage:
    python scripts/glb_to_tripplane_mpl.py \
        --input-dir  dataset/objaverse/glbs \
        --output-dir dataset/tripplane \
        [--img-size 256] \
        [--workers 8] \
        [--skip-existing]
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
import trimesh


# ──────────────────────────────────────────────────────────────────
# Geometry loading
# ──────────────────────────────────────────────────────────────────

def load_mesh(glb_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path, force="scene")
    if isinstance(loaded, trimesh.scene.Scene):
        geometries = list(loaded.geometry.values())
    else:
        geometries = [loaded]

    meshes = [g for g in geometries
              if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]

    if not meshes:
        clouds = [g for g in geometries
                  if isinstance(g, trimesh.points.PointCloud) and len(g.vertices) > 0]
        if clouds:
            pts = np.vstack([c.vertices for c in clouds])
            try:
                hull = trimesh.convex.convex_hull(pts)
                meshes = [hull]
            except Exception:
                # Fall back: tiny cube at centroid
                meshes = [trimesh.creation.box()]

    if not meshes:
        raise ValueError("No renderable geometry")

    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    # Normalise to unit sphere
    v = combined.vertices.copy()
    v -= v.mean(axis=0)
    scale = np.max(np.linalg.norm(v, axis=1))
    if scale > 0:
        v /= scale
    return trimesh.Trimesh(vertices=v, faces=combined.faces, process=False)


# ──────────────────────────────────────────────────────────────────
# Pure-NumPy z-buffer rasteriser
# ──────────────────────────────────────────────────────────────────

def _cross2d(ax, ay, bx, by):
    return ax * by - ay * bx


def rasterise(verts3d: np.ndarray, faces: np.ndarray,
              proj_axes: tuple, depth_axis: int,
              img_size: int, light_dir: np.ndarray) -> np.ndarray:
    """
    Orthographic rasteriser.

    proj_axes  : (col_axis, row_axis) — which 3-D axes map to image x/y
    depth_axis : which 3-D axis is the camera depth (used for z-buffer)
    light_dir  : unit vector toward light source (camera direction)
    Returns    : (img_size, img_size, 3) uint8 RGB image
    """
    H = W = img_size
    px_axis, py_axis = proj_axes

    # Project vertices to image space  [-1.2, 1.2] → [0, W-1]
    scale = 0.9 * (W / 2)
    cx = cy = W / 2

    px = verts3d[:, px_axis] * scale + cx   # col
    py = -verts3d[:, py_axis] * scale + cy  # row (flip Y)
    pz = verts3d[:, depth_axis]             # depth

    # Pre-compute face normals for shading
    v0 = verts3d[faces[:, 0]]
    v1 = verts3d[faces[:, 1]]
    v2 = verts3d[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals /= nlen
    brightness = np.clip(normals @ light_dir, 0.15, 1.0)  # diffuse + ambient

    # Z-buffer init
    zbuf   = np.full((H, W), np.inf,  dtype=np.float32)
    colbuf = np.full((H, W, 3), 230,  dtype=np.uint8)   # background grey

    for fi in range(len(faces)):
        i0, i1, i2 = faces[fi]
        x0, y0, z0 = px[i0], py[i0], pz[i0]
        x1, y1, z1 = px[i1], py[i1], pz[i1]
        x2, y2, z2 = px[i2], py[i2], pz[i2]

        # Bounding box (clipped to image)
        minx = int(max(0,   min(x0, x1, x2)))
        maxx = int(min(W-1, max(x0, x1, x2)))
        miny = int(max(0,   min(y0, y1, y2)))
        maxy = int(min(H-1, max(y0, y1, y2)))
        if minx > maxx or miny > maxy:
            continue

        denom = _cross2d(x1-x0, y1-y0, x2-x0, y2-y0)
        if abs(denom) < 1e-6:
            continue

        # Pixel grid
        xs = np.arange(minx, maxx+1, dtype=np.float32)
        ys = np.arange(miny, maxy+1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        # Barycentric coords
        w1 = _cross2d(x1-x0, y1-y0, gx-x0, gy-y0) / denom
        w2 = _cross2d(x2-x0, y2-y0, gx-x0, gy-y0) / (-denom)
        w0 = 1.0 - w1 - w2

        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue

        depth = w0 * z0 + w1 * z1 + w2 * z2

        col_slice = slice(minx, maxx+1)
        row_slice = slice(miny, maxy+1)

        cur_z  = zbuf[row_slice, col_slice]
        update = mask & (depth < cur_z)
        if not update.any():
            continue

        zbuf[row_slice, col_slice][update] = depth[update]

        c = int(brightness[fi] * 220)
        colbuf[row_slice, col_slice][update] = [c, c, c]

    return colbuf


# ──────────────────────────────────────────────────────────────────
# View definitions
# (proj_axes, depth_axis, light_dir)
# ──────────────────────────────────────────────────────────────────

# Axis indices: X=0, Y=1, Z=2
VIEWS = {
    "oxoy": ((0, 1), 2, np.array([0., 0.,  1.])),  # front: looking -Z
    "oyoz": ((2, 1), 0, np.array([1., 0.,  0.])),  # side:  looking -X
    "oxoz": ((0, 2), 1, np.array([0., 1.,  0.])),  # top:   looking -Y
}


def render_triplane(mesh: trimesh.Trimesh, output_dir: Path, img_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    verts = mesh.vertices
    faces = mesh.faces

    for name, (proj_axes, depth_axis, light_dir) in VIEWS.items():
        img_arr = rasterise(verts, faces, proj_axes, depth_axis, img_size, light_dir)
        Image.fromarray(img_arr).save(output_dir / f"{name}.png")


# ──────────────────────────────────────────────────────────────────
# Worker (runs in child process)
# ──────────────────────────────────────────────────────────────────

def process_one(args):
    glb_path, output_dir, img_size = args
    obj_id  = Path(glb_path).stem
    obj_out = Path(output_dir) / obj_id
    try:
        mesh = load_mesh(glb_path)
        render_triplane(mesh, obj_out, img_size)
        return obj_id, None
    except Exception as exc:  # noqa: BLE001
        return obj_id, str(exc)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render GLBs to tripplane images (CPU z-buffer)")
    parser.add_argument("--input-dir",     default="dataset/objaverse/glbs")
    parser.add_argument("--output-dir",    default="dataset/tripplane")
    parser.add_argument("--img-size",      type=int, default=256)
    parser.add_argument("--workers",       type=int, default=min(os.cpu_count() or 4, 16))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip objects whose output folder already has 3 PNGs")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    glbs = sorted(input_dir.rglob("*.glb"))
    if not glbs:
        print(f"No .glb files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.skip_existing:
        before = len(glbs)
        glbs = [g for g in glbs
                if len(list((output_dir / g.stem).glob("*.png"))) < 3]
        print(f"Skipping {before - len(glbs)} already done — {len(glbs)} remaining")

    total = len(glbs)
    print(f"Processing {total} GLBs → {output_dir}  (workers={args.workers})")

    tasks  = [(str(g), str(output_dir), args.img_size) for g in glbs]
    errors = []
    done   = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            done += 1
            obj_id, err = fut.result()
            if err:
                print(f"[{done}/{total}] ERROR {obj_id}: {err}", file=sys.stderr)
                errors.append((obj_id, err))
            else:
                if done % 500 == 0 or done == total:
                    print(f"[{done}/{total}] ok: {obj_id}")

    if errors:
        err_file = output_dir / "render_errors.txt"
        with open(err_file, "w") as f:
            for oid, emsg in errors:
                f.write(f"{oid}\t{emsg}\n")
        print(f"\n{len(errors)} error(s) logged to {err_file}", file=sys.stderr)
    else:
        print("All done.")


if __name__ == "__main__":
    main()
