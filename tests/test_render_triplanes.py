"""render_triplanes.py: orthographic triplane PNGs with the dataset's plane convention."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))       # importable by name → picklable for worker processes
import render_triplanes as rt  # noqa: E402
trimesh = pytest.importorskip("trimesh")


def _box_glb(path, extents=(0.2, 1.0, 0.5)):
    box = trimesh.creation.box(extents=extents)
    box.visual.vertex_colors = np.tile(np.array([200, 30, 30, 255], np.uint8), (len(box.vertices), 1))
    box.export(str(path))


def _fg_bbox(img):
    fg = (img < 250).any(-1)
    ys, xs = np.where(fg)
    return fg.mean(), (xs.max() - xs.min() + 1), (ys.max() - ys.min() + 1)


def test_renders_three_planes_with_expected_orientation(tmp_path):
    glb = tmp_path / "box.glb"; _box_glb(glb)
    out = tmp_path / "ds"
    written = rt.render_object(str(glb), str(out / "3d_objects" / "renders" / "box"), res=128, n_points=60_000)
    assert set(written) == {"oxoy", "oxoz", "oyoz"}
    from PIL import Image
    imgs = {k: np.asarray(Image.open(v).convert("RGB")) for k, v in written.items()}
    for k, im in imgs.items():
        assert im.shape == (128, 128, 3)
        cov, w, h = _fg_bbox(im)
        assert 0.05 < cov < 0.95, f"{k}: coverage {cov}"
    # extents (x=0.2, y=1.0, z=0.5): front (x,y) tall; top (x,z) tall; side (z,y) → u=z 0.5, v=y 1.0 → tall
    _, w, h = _fg_bbox(imgs["oxoy"]); assert h > 3 * w, ("oxoy should be tall", w, h)
    _, w, h = _fg_bbox(imgs["oxoz"]); assert h > 1.8 * w, ("oxoz should be tall", w, h)
    _, w, h = _fg_bbox(imgs["oyoz"]); assert h > 1.5 * w, ("oyoz should be tall", w, h)
    # colour survives (red box)
    fg = (imgs["oxoy"] < 250).any(-1)
    assert imgs["oxoy"][fg][:, 0].mean() > imgs["oxoy"][fg][:, 1].mean() + 60


def test_dataset_layout_is_readable_by_UniversalThreeDDataset(tmp_path):
    from mavt.data.datasets import UniversalThreeDDataset
    glb = tmp_path / "obj1.glb"; _box_glb(glb, (1.0, 0.6, 0.3))
    root = tmp_path / "ds"
    rt.render_object(str(glb), str(root / "3d_objects" / "renders" / "obj1"), res=64, n_points=20_000)
    (root / "captions").mkdir(); (root / "captions" / "3d.json").write_text('{"obj1": "a red box"}')
    ds = UniversalThreeDDataset(str(root), resolution=64)
    assert len(ds) == 1
    s = ds[0]
    assert s["data"].shape == (3, 3, 64, 64) and s["modality"] == "threed" and s["caption"] == "a red box" and s["id"] == "obj1"


def test_render_many_reports_failures(tmp_path):
    bad = tmp_path / "bad.glb"; bad.write_bytes(b"not a mesh")
    ok, errs = rt.render_many([(str(bad), str(tmp_path / "ds/3d_objects/renders/bad"))], res=32, n_points=1000, workers=2)
    assert ok == 0 and "bad" in errs
