#!/usr/bin/env python3
"""Orthographic triplane renders (oxoy / oxoz / oyoz PNGs) from meshes — pure numpy + trimesh.

No OpenGL / EGL needed (the GPU server has no sudo for OSMesa). Surface points are sampled
with colour (vertex colours or texture), normalised to a unit cube, projected orthographically
onto the three coordinate planes with a nearest-wins depth buffer at 2x resolution, lightly
shaded by |n·view|, then box-downsampled. Output layout matches UniversalThreeDDataset:

    <out>/3d_objects/renders/<obj_id>/{oxoy,oxoz,oyoz}.png     (RGB, res x res, white background)

Plane convention (right-handed, y up in GLB):
    oxoy  "front": view along -z, image u=x (right), v=y (up)
    oxoz  "top"  : view along -y, image u=x (right), v=-z (front of object at the bottom)
    oyoz  "side" : view along -x, image u=-z (right), v=y (up)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

PLANES = ("oxoy", "oxoz", "oyoz")
# (u_axis, u_sign, v_axis, v_sign, depth_axis) — depth larger = closer to the camera
_PLANE_DEF = {"oxoy": (0, +1, 1, +1, 2), "oxoz": (0, +1, 2, -1, 1), "oyoz": (2, -1, 1, +1, 0)}
_VIEW_DIR = {"oxoy": np.array([0, 0, 1.0]), "oxoz": np.array([0, 1.0, 0]), "oyoz": np.array([1.0, 0, 0])}


def _geom_to_coloured(geom, T=None):
    """Rebuild a geometry as a plain Trimesh with a valid (n_vertices, 4) vertex-colour array.

    Objaverse GLBs often carry degenerate visuals (a single RGBA instead of per-face colours,
    materials without textures); trimesh's own copy()/to_color() then raises. We never copy
    the visual object — we extract colours defensively and fall back to grey.
    """
    import trimesh
    v = np.asarray(geom.vertices, dtype=np.float64)
    if T is not None:
        v = trimesh.transform_points(v, T)
    f = np.asarray(geom.faces)
    vc = None
    try:
        vis = geom.visual
        if hasattr(vis, "to_color"):
            vis = vis.to_color()
        vc = np.asarray(vis.vertex_colors)
        if vc.ndim != 2 or vc.shape[0] != len(v) or vc.shape[1] < 3:
            vc = None
    except Exception:  # noqa: BLE001
        vc = None
    if vc is None:
        try:                                            # single face colour → broadcast
            fc = np.asarray(geom.visual.face_colors)
            if fc.ndim == 2 and fc.shape[0] == len(f) and fc.shape[1] >= 3:
                vc = np.zeros((len(v), 4), np.uint8); vc[f.reshape(-1)] = np.repeat(fc[:, :4], 3, axis=0)
        except Exception:  # noqa: BLE001
            vc = None
    if vc is None:
        vc = np.tile(np.array([180, 180, 180, 255], np.uint8), (len(v), 1))
    if vc.shape[1] == 3:
        vc = np.concatenate([vc, np.full((len(v), 1), 255, vc.dtype)], 1)
    return trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vc[:, :4].astype(np.uint8), process=False)


def load_mesh(path: str):
    """Load any trimesh-readable file (glb/gltf/obj/ply) as one Trimesh with per-vertex colours."""
    import trimesh
    obj = trimesh.load(path, force="scene")
    geoms = []
    if isinstance(obj, trimesh.Scene):
        for node in obj.graph.nodes_geometry:
            T, gname = obj.graph[node]
            geom = obj.geometry.get(gname)
            if isinstance(geom, trimesh.Trimesh) and geom.faces.shape[0] > 0:
                geoms.append(_geom_to_coloured(geom, T))
    elif isinstance(obj, trimesh.Trimesh) and obj.faces.shape[0] > 0:
        geoms.append(_geom_to_coloured(obj))
    if not geoms:
        raise ValueError("no triangle geometry")
    mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    if mesh.faces.shape[0] == 0:
        raise ValueError("empty mesh")
    return mesh


def sample_coloured_points(mesh, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points (n,3) in a unit cube centred at 0, colours (n,3) uint8, normals (n,3)."""
    import trimesh
    try:
        pts, fidx, cols = trimesh.sample.sample_surface(mesh, n, sample_color=True)
        cols = np.asarray(cols)[:, :3]
    except Exception:  # noqa: BLE001  (no colour info at all)
        pts, fidx = trimesh.sample.sample_surface(mesh, n)
        cols = np.full((len(pts), 3), 180, np.uint8)
    if cols is None or len(cols) != len(pts):
        cols = np.full((len(pts), 3), 180, np.uint8)
    normals = mesh.face_normals[fidx]
    pts = np.asarray(pts, dtype=np.float64)
    lo, hi = pts.min(0), pts.max(0)
    center, extent = (lo + hi) / 2, float((hi - lo).max())
    pts = (pts - center) / max(extent, 1e-8)             # longest side → [-0.5, 0.5]
    return pts, cols.astype(np.uint8), normals


def splat_plane(pts, cols, normals, plane: str, res: int, margin: float = 0.05) -> np.ndarray:
    """Nearest-wins orthographic splat at 2x res, then 2x2 box downsample → (res, res, 3) uint8."""
    ua, us, va, vs, da = _PLANE_DEF[plane]
    R = res * 2
    scale = (1.0 - 2 * margin)
    u = ((pts[:, ua] * us) * scale + 0.5) * (R - 1)
    v = ((-pts[:, va] * vs) * scale + 0.5) * (R - 1)     # image rows grow downwards
    ui = np.clip(np.rint(u).astype(np.int64), 0, R - 1)
    vi = np.clip(np.rint(v).astype(np.int64), 0, R - 1)
    depth = pts[:, da]
    shade = 0.55 + 0.45 * np.abs(normals @ _VIEW_DIR[plane])
    shaded = np.clip(cols.astype(np.float32) * shade[:, None], 0, 255).astype(np.uint8)
    img = np.full((R, R, 3), 255, np.uint8)
    # 5-point cross splat closes sampling holes at 2x res; nearest sample wins per pixel
    for du, dv in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        if True:
            uu = np.clip(ui + du, 0, R - 1); vv = np.clip(vi + dv, 0, R - 1)
            lin = vv * R + uu
            order = np.lexsort((depth, lin))                  # by pixel, then far→near
            lin_s = lin[order]
            last = np.r_[lin_s[1:] != lin_s[:-1], True]       # last (nearest) sample per pixel
            sel = order[last]
            img.reshape(-1, 3)[lin[sel]] = shaded[sel]
    small = img.reshape(res, 2, res, 2, 3).mean(axis=(1, 3))
    return np.rint(small).astype(np.uint8)


def render_object(mesh_path: str, out_dir: str, res: int = 256, n_points: int = 200_000) -> Dict[str, str]:
    from PIL import Image
    mesh = load_mesh(mesh_path)
    pts, cols, normals = sample_coloured_points(mesh, n_points)
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for plane in PLANES:
        arr = splat_plane(pts, cols, normals, plane, res)
        p = os.path.join(out_dir, f"{plane}.png")
        Image.fromarray(arr, "RGB").save(p, compress_level=6)
        written[plane] = p
    return written


def _worker(args):
    mesh_path, out_dir, res, n_points = args
    try:
        render_object(mesh_path, out_dir, res, n_points)
        return os.path.basename(out_dir), ""
    except Exception as exc:  # noqa: BLE001
        return os.path.basename(out_dir), f"{type(exc).__name__}: {str(exc)[:120]}"


def render_many(items, res: int, n_points: int, workers: int):
    """items: list of (mesh_path, out_dir). Returns (n_ok, {obj_id: error})."""
    errs: Dict[str, str] = {}
    ok = 0
    jobs = [(m, o, res, n_points) for m, o in items]
    if workers <= 1:
        results = map(_worker, jobs)
    else:
        from multiprocessing import Pool
        pool = Pool(workers)
        results = pool.imap_unordered(_worker, jobs, chunksize=1)
    for oid, err in results:
        if err:
            errs[oid] = err
        else:
            ok += 1
    if workers > 1:
        pool.close(); pool.join()
    return ok, errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+", help="mesh files (glb/gltf/obj/ply)")
    ap.add_argument("--out", required=True, help="dataset root; renders go to <out>/3d_objects/renders/<stem>/")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--points", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    items = [(m, os.path.join(a.out, "3d_objects", "renders", os.path.splitext(os.path.basename(m))[0])) for m in a.meshes]
    ok, errs = render_many(items, a.res, a.points, a.workers)
    print(json.dumps({"ok": ok, "failed": len(errs), "errors": dict(list(errs.items())[:10])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
