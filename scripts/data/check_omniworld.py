"""Check a downloaded OmniWorld-Game subset.

verify : every file in the pick list has the Hub's size and LFS sha256 (tree.json from the Hub API).
geom   : on a few scenes, decode depth/pose/flow and check they agree with each other:
         unproject depth_t with K and the world-to-camera pose, move to frame t+k, compare with depth_{t+k};
         compare the flow those poses induce on static pixels with the stored forward flow.

Formats follow the dataset README (revision 4e34897). Its flow reader has a sign slip
(flow_min=-FLOW_MIN gives an empty range); the stored range is [-300, 300] px.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

FLOW_MIN, FLOW_MAX = -300.0, 300.0
NEAR, FAR = 1.0, 1000.0


# ---------------------------------------------------------------- decoding
def decode_depth(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint16 PNG -> depth in the scene's unscaled units, valid mask (README §2.3)."""
    d = raw.astype(np.float64) / 65535.0
    valid = (d >= 0.0015) & (d <= 65500.0 / 65535.0)
    depth = d / (FAR - d * (FAR - NEAR)) / 0.004
    return np.where(valid, depth, np.nan), valid


def decode_flow(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """two uint16 PNGs -> (H, W, 2) forward flow in pixels."""
    to = lambda x: x.astype(np.float64) / 65535.0 * (FLOW_MAX - FLOW_MIN) + FLOW_MIN
    return np.stack([to(u), to(v)], -1)


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def camera_split(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """camera/split_i.json -> K (3,3), world-to-camera (S,4,4) in OpenCV convention, unscaled."""
    K = np.array([[cam["focals"], 0, cam["cx"]], [0, cam["focals"], cam["cy"]], [0, 0, 1]], np.float64) \
        if np.ndim(cam["focals"]) == 0 else None
    if K is None:  # per-frame focal: keep the first, report spread elsewhere
        f = np.asarray(cam["focals"], np.float64)
        K = np.array([[f[0], 0, np.asarray(cam["cx"]).ravel()[0]], [0, f[0], np.asarray(cam["cy"]).ravel()[0]], [0, 0, 1]])
    S = len(cam["quats"])
    w2c = np.tile(np.eye(4), (S, 1, 1))
    w2c[:, :3, :3] = quat_wxyz_to_R(cam["quats"])
    w2c[:, :3, 3] = np.asarray(cam["trans"], np.float64)
    return K, w2c


def reproject(depth_a, K, w2c_a, w2c_b):
    """pixels of frame a with depth -> (u_b, v_b, z_b) seen from camera b; arrays over a's grid."""
    H, W = depth_a.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    p_cam_a = rays * depth_a[..., None]
    T = w2c_b @ np.linalg.inv(w2c_a)
    p_b = p_cam_a @ T[:3, :3].T + T[:3, 3]
    z = p_b[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ub = K[0, 0] * p_b[..., 0] / z + K[0, 2]
        vb = K[1, 1] * p_b[..., 1] / z + K[1, 2]
    return ub, vb, z, u, v


def pair_errors(depth_a, depth_b, K, w2c_a, w2c_b, static_a):
    """abs-rel depth error at a's pixels that land inside b, plus induced flow (static pixels)."""
    H, W = depth_a.shape
    ub, vb, zb, u, v = reproject(depth_a, K, w2c_a, w2c_b)
    ok = static_a & np.isfinite(depth_a) & np.isfinite(zb) & (zb > 0) & (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)
    ib, jb = np.clip(vb.astype(int), 0, H - 1), np.clip(ub.astype(int), 0, W - 1)
    db = depth_b[ib, jb]
    ok &= np.isfinite(db)
    rel = np.abs(db[ok] - zb[ok]) / db[ok]
    vis = rel < 0.05            # occluded pixels (b sees something nearer) are expected outliers
    return {"n": int(ok.sum()), "absrel_median": float(np.median(rel)) if rel.size else None,
            "within_5pct": float(vis.mean()) if rel.size else None}, (ub - u, vb - v), ok


# ---------------------------------------------------------------- verify
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def verify(root: Path, paths: list[str], tree: dict) -> dict:
    out, bad = [], []
    for rel in paths:
        p, t = root / rel, tree[rel]
        if not p.exists():
            bad.append({"path": rel, "why": "missing"}); continue
        size_ok = p.stat().st_size == t["size"]
        want = t.get("lfs", {}).get("oid")
        got = sha256_file(p) if want else git_blob_sha1(p)
        ok = size_ok and got == (want or t["oid"])
        out.append(ok)
        if not ok:
            bad.append({"path": rel, "why": "size" if not size_ok else "hash"})
    return {"files": len(paths), "ok": int(sum(out)), "bad": bad}


# ---------------------------------------------------------------- geom
def read_members(tar_path: Path, want) -> dict[str, bytes]:
    """stream a .tar.gz and keep only members for which want(name) is true."""
    keep = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if m.isfile() and want(m.name):
                keep[m.name] = tf.extractfile(m).read()
    return keep


def png(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)))


def geom_scene(root: Path, uid: str, meta: dict, n_pairs: int, gaps: list[int]) -> dict:
    a = root / "annotations" / "OmniWorld-Game" / uid
    v = root / "videos" / "OmniWorld-Game" / uid
    others = read_members(a / f"{uid}_others.tar.gz",
                          lambda n: n.endswith(("split_info.json", "split_0.json", "fps.txt")) or "/gdino_mask/" in n)
    info = json.loads(next(b for n, b in others.items() if n.endswith("split_info.json")))
    cam = json.loads(next(b for n, b in others.items() if n.endswith("camera/split_0.json")))
    frames = info["split"][0]
    K, w2c = camera_split(cam)
    scale = float(meta["Metric Scale"])
    gmax = max(gaps)
    starts = np.linspace(0, len(frames) - 1 - gmax, n_pairs).astype(int).tolist()
    need = sorted({frames[s] for s in starts} | {frames[s + g] for s in starts for g in gaps})
    stems = {f"{i:06d}" for i in need}
    pick = lambda n: Path(n).stem in stems or Path(n).parent.name in stems
    depth_b, flow_b = {}, {}
    for t in sorted(a.glob(f"{uid}_depth_*.tar.gz")):
        depth_b.update(read_members(t, lambda n: "/depth/" in "/" + n and pick(n)))
    for t in sorted(a.glob(f"{uid}_flow_*.tar.gz")):
        flow_b.update(read_members(t, lambda n: pick(n) and n.endswith(("flow_u_16.png", "flow_v_16.png"))))
    find = lambda d, i, suffix="": next((b for n, b in d.items() if (Path(n).stem == f"{i:06d}" or Path(n).parent.name == f"{i:06d}") and n.endswith(suffix)), None)
    gd = lambda i: find(others, i, ".png")

    res = {"uid": uid, "split0_frames": len(frames), "splits": info["split_num"], "metric_scale": scale,
           "focal_px": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "pairs": {}, "flow": [],
           "depth_valid_share": [], "depth_metric_m": []}
    for s in starts:
        fa = frames[s]
        ra = find(depth_b, fa)
        if ra is None:
            continue
        da, va = decode_depth(png(ra))
        res["depth_valid_share"].append(float(va.mean()))
        res["depth_metric_m"] += (da[va] * scale)[:: max(1, va.sum() // 2000)].tolist()
        g = gd(fa)
        static = ~(png(g) > 127) if g is not None else np.ones_like(va)
        if static.ndim == 3:
            static = static[..., 0]
        for gap in gaps:
            rb = find(depth_b, frames[s + gap])
            if rb is None:
                continue
            db, _ = decode_depth(png(rb))
            for name, sc in (("unscaled", 1.0), ("metric", scale)):
                wa, wb = w2c[s].copy(), w2c[s + gap].copy()
                if sc != 1.0:  # README: scale camera-to-world translation only
                    for w in (wa, wb):
                        c2w = np.linalg.inv(w); c2w[:3, 3] *= sc; w[:] = np.linalg.inv(c2w)
                e, (fu, fv), ok = pair_errors(da * sc, db * sc, K, wa, wb, static)
                res["pairs"].setdefault(f"{name}_gap{gap}", []).append(e)
                if gap == 1 and name == "metric":
                    fl = find(flow_b, fa, "flow_u_16.png"), find(flow_b, fa, "flow_v_16.png")
                    if all(x is not None for x in fl):
                        F = decode_flow(png(fl[0]), png(fl[1]))
                        epe = np.hypot(F[..., 0] - fu, F[..., 1] - fv)[ok]
                        res["flow"].append({"n": int(ok.sum()), "epe_median_px": float(np.median(epe)),
                                            "within_1px": float((epe < 1).mean()), "gt_mag_median_px": float(np.median(np.hypot(F[..., 0], F[..., 1])[ok]))})
    summ = lambda xs, k: float(np.median([x[k] for x in xs if x[k] is not None])) if xs else None
    res["summary"] = {k: {"absrel_median": summ(v, "absrel_median"), "within_5pct": summ(v, "within_5pct"), "pairs": len(v)}
                      for k, v in res["pairs"].items()}
    res["summary"]["flow"] = {"epe_median_px": summ(res["flow"], "epe_median_px"), "within_1px": summ(res["flow"], "within_1px"),
                              "gt_mag_median_px": summ(res["flow"], "gt_mag_median_px"), "pairs": len(res["flow"])}
    dm = np.array(res.pop("depth_metric_m"))
    res["summary"]["depth_metric_m"] = {"median": float(np.median(dm)), "p5": float(np.percentile(dm, 5)), "p95": float(np.percentile(dm, 95))} if dm.size else None
    res["summary"]["depth_valid_share"] = float(np.mean(res["depth_valid_share"])) if res["depth_valid_share"] else None
    del res["pairs"], res["flow"], res["depth_valid_share"]
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--meta-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--paths", default="pick_paths.txt", help="file list (in --meta-dir) to verify")
    ap.add_argument("--tier", default=None, help="key in pick.json holding the scenes to check, e.g. tier1")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--geom-scenes", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args(argv)
    t0 = time.time()
    paths = [p for p in (args.meta_dir / args.paths).read_text().split() if p]
    tree = {t["path"]: t for t in json.load(open(args.meta_dir / "tree.json"))}
    meta = {r["UID"]: r for r in csv.DictReader(open(args.meta_dir / "game_meta.csv"))}
    pick = json.load(open(args.meta_dir / "pick.json"))
    scenes = pick[args.tier] if args.tier else pick["scenes"]
    report = {"revision": pick["revision"], "scenes": scenes}
    if not args.skip_verify:
        report["verify"] = verify(args.root, paths, tree)
    report["scene_meta"] = [{"uid": u, "frames": int(meta[u]["Split Img Num"]), "splits": int(meta[u]["Split Num"]),
                             "fps": float(meta[u]["FPS"]), "metric_scale": float(meta[u]["Metric Scale"]),
                             "test_splits": meta[u]["Test Split Index"]} for u in scenes]
    report["geom"] = [geom_scene(args.root, u, meta[u], args.pairs, [1, 5]) for u in scenes[: args.geom_scenes]]
    report["runtime_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("verify",) if k in report}, indent=1))
    for g in report["geom"]:
        print(g["uid"], json.dumps(g["summary"]))


if __name__ == "__main__":
    main()
