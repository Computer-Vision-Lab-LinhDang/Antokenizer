"""
Render GLB files from objaverse into tripplane images.

Output structure:
    <output-dir>/<object-id>/oxoy.png   (front, XY plane, camera along +Z)
    <output-dir>/<object-id>/oxoz.png   (top,   XZ plane, camera along +Y)
    <output-dir>/<object-id>/oyoz.png   (side,  YZ plane, camera along +X)

Usage:
    python glb_to_tripplane.py --input-dir .objaverse/hf-objaverse-v1/glbs \
                                --output-dir tripplane_images \
                                [--img-size 256]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _normalise_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = vertices - vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(vertices, axis=1))
    if scale > 0:
        vertices /= scale
    return vertices


def load_scene(glb_path: str) -> pyrender.Scene:
    loaded = trimesh.load(glb_path, force="scene")
    geometries = list(loaded.geometry.values()) if isinstance(loaded, trimesh.scene.Scene) else [loaded]
    if not geometries:
        raise ValueError(f"No geometry in {glb_path}")

    meshes = [g for g in geometries if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
    clouds = [g for g in geometries if isinstance(g, trimesh.points.PointCloud) and len(g.vertices) > 0]

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.vertices = _normalise_vertices(combined.vertices)
        # Strip materials to avoid texture-upload errors on headless EGL
        combined.visual = trimesh.visual.ColorVisuals(mesh=combined)
        scene.add(pyrender.Mesh.from_trimesh(combined, smooth=False))
    elif clouds:
        # Merge all point clouds
        all_verts = np.vstack([c.vertices for c in clouds])
        all_colors = (
            np.vstack([c.colors[:, :3] for c in clouds])
            if all(c.colors is not None for c in clouds) and all(c.colors.shape[1] >= 3 for c in clouds)
            else None
        )
        all_verts = _normalise_vertices(all_verts)
        scene.add(pyrender.Mesh.from_points(all_verts, colors=all_colors))
    else:
        raise ValueError(f"No renderable geometry found in {glb_path}")

    return scene


def ortho_camera(extent: float = 1.4) -> pyrender.OrthographicCamera:
    return pyrender.OrthographicCamera(xmag=extent, ymag=extent, znear=0.01, zfar=10.0)


# name -> (camera eye, up vector)  — name matches output filename
VIEWS = {
    "oxoy": (np.array([0.0, 0.0, 3.0]), np.array([0.0, 1.0, 0.0])),   # front: XY plane
    "oyoz": (np.array([3.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),   # side:  YZ plane
    "oxoz": (np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, -1.0])),  # top:   XZ plane
}


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def render_view(scene: pyrender.Scene, eye: np.ndarray, up: np.ndarray,
                img_size: int, renderer: pyrender.OffscreenRenderer) -> np.ndarray:
    camera = ortho_camera()
    pose = look_at(eye, np.zeros(3), up)
    cam_node = scene.add(camera, pose=pose)

    # Key directional light from camera position
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light, pose=pose)

    color, _ = renderer.render(scene)
    scene.remove_node(cam_node)
    scene.remove_node(light_node)
    return color  # HxWx3 uint8


def glb_to_planes(glb_path: str, output_dir: str, img_size: int = 256) -> None:
    """Render a GLB and save oxoy/oxoz/oyoz PNGs into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = load_scene(glb_path)
    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    for name, (eye, up) in VIEWS.items():
        panel = render_view(scene, eye, up, img_size, renderer)
        Image.fromarray(panel).save(out / f"{name}.png")
    renderer.delete()


def iter_glbs(input_dir: str):
    for glb in Path(input_dir).rglob("*.glb"):
        yield glb


def process(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    glbs = list(iter_glbs(input_dir))
    if not glbs:
        print(f"No .glb files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(glbs)} GLB files -> {output_dir}")
    errors = []
    for i, glb_path in enumerate(glbs, 1):
        obj_id = glb_path.stem  # hash filename, no extension
        obj_out = output_dir / obj_id
        if all((obj_out / f"{v}.png").exists() for v in ("oxoy", "oxoz", "oyoz")):
            continue
        try:
            glb_to_planes(str(glb_path), str(obj_out), img_size=args.img_size)
            print(f"[{i}/{len(glbs)}] {obj_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(glbs)}] ERROR {obj_id}: {exc}", file=sys.stderr)
            errors.append((str(glb_path), str(exc)))

    if errors:
        print(f"\n{len(errors)} file(s) failed:", file=sys.stderr)
        for path, err in errors:
            print(f"  {path}: {err}", file=sys.stderr)
    else:
        print("All done.")


def main():
    parser = argparse.ArgumentParser(description="Render GLBs to tripplane images")
    parser.add_argument("--input-dir", default=".objaverse/hf-objaverse-v1/glbs",
                        help="Root directory containing .glb files")
    parser.add_argument("--output-dir", default="tripplane_images",
                        help="Directory to write output PNGs")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Width and height of each view panel (default: 256)")
    process(parser.parse_args())


if __name__ == "__main__":
    main()
