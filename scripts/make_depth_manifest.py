#!/usr/bin/env python3
"""Build depth manifests (src/mavt/cd32/depth_data.py) for the depth probe and the depth eval.

  # OmniWorld-Game as packed by scripts/pack_omniworld_dino.py (dense metric depth, no copy: rows index depth.npy)
  python scripts/make_depth_manifest.py omniworld --root <pack root> --split train --every 8 --out <train.jsonl>

  # Argoverse 2 sensor: LiDAR sweeps projected into the ring cameras (real depth, sparse), written as PNGs
  python scripts/make_depth_manifest.py argoverse2 --root <av2 sensor root> --split val --per-log 2 \\
      --out-dir <dir> [--logs 20] [--workers 16]

  # NYU Depth v2 labeled set (needs h5py and scipy): the official 795 / 654 split, Eigen crop on the test split
  python scripts/make_depth_manifest.py nyu --mat nyu_depth_v2_labeled.mat --splits splits.mat --out-dir <dir>

Argoverse 2: each sampled sweep is moved into the camera with the two ego poses (sweep and image time stamps),
projected with the radial distortion model (k1, k2, k3), z-buffered at half the image resolution, and points
hidden behind a nearer point in a small window are dropped (the LiDAR sits above the camera, so at object
edges it sees background the camera does not). Depth PNGs are uint16, depth = value / 256 m.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ---------------------------------------------------------------------------------- OmniWorld-Game
def omniworld(args) -> int:
    root = os.path.abspath(args.root)
    split_dir = os.path.join(root, args.split)
    rows, scenes = [], sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    for uid in scenes:
        sd = os.path.join(split_dir, uid)
        with open(os.path.join(sd, "scene.json")) as f:
            scene = json.load(f)
        for seg in scene.get("segments", []):
            name = seg["name"] if isinstance(seg, dict) else seg
            seg_dir = os.path.join(sd, name)
            if not os.path.exists(os.path.join(seg_dir, "depth.npy")):
                continue
            frames = np.load(os.path.join(seg_dir, "meta.npz"))["frames"]
            for j in range(0, len(frames), args.every):
                img = os.path.join(sd, "rgb", f"{int(frames[j])}.jpg")
                if os.path.exists(img):
                    rows.append(dict(image=img, depth=os.path.join(seg_dir, "depth.npy"), index=j,
                                     max_depth=args.max_depth, source="omniworld-game", scene=uid))
    _write(args.out, rows)
    print(f"omniworld {args.split}: {len(scenes)} scenes, {len(rows)} rows -> {args.out}")
    return 0


# ---------------------------------------------------------------------------------- Argoverse 2
def _feather(path: str) -> dict:
    import pyarrow.feather as pf
    t = pf.read_table(path)
    return {c: t.column(c).to_numpy() for c in t.column_names}


def _quat(qw, qx, qy, qz) -> np.ndarray:
    w, x, y, z = np.array([qw, qx, qy, qz], np.float64) / np.linalg.norm([qw, qx, qy, qz])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _se3_rows(t: dict, key: str) -> dict:
    """feather table of (key, qw, qx, qy, qz, tx_m, ty_m, tz_m) -> {key: (R, t)}."""
    out = {}
    for i in range(len(t[key])):
        k = t[key][i]
        k = k.decode() if isinstance(k, bytes) else (int(k) if key == "timestamp_ns" else str(k))
        out[k] = (_quat(t["qw"][i], t["qx"][i], t["qy"][i], t["qz"][i]),
                  np.array([t["tx_m"][i], t["ty_m"][i], t["tz_m"][i]], np.float64))
    return out


def _r_limit(k1: float, k2: float, k3: float, r_img: float) -> float:
    """Largest undistorted radius to keep: the radial map r (1 + k1 r^2 + k2 r^4 + k3 r^6) must still be
    increasing and must not pass the image corner by much (beyond it a polynomial can fold points back in)."""
    rs = np.linspace(0, 3, 30001)
    rd = rs * (1 + k1 * rs ** 2 + k2 * rs ** 4 + k3 * rs ** 6)
    inc = np.nonzero(np.diff(rd) <= 0)[0]
    stop = len(rs) - 1 if inc.size == 0 else inc[0]
    past = np.nonzero(rd[:stop + 1] > 1.1 * r_img)[0]
    return float(rs[past[0]] if past.size else rs[stop])


def _min_filter(a: np.ndarray, window: int) -> np.ndarray:
    """Minimum over a window x window neighbourhood (inf outside the map)."""
    r = window // 2
    pad = np.pad(a, r, constant_values=np.inf)
    return np.lib.stride_tricks.sliding_window_view(pad, (window, window)).min(axis=(2, 3))


def project_lidar(pts_cam: np.ndarray, K: dict, scale: float, window: int = 5, tol: float = 0.1) -> np.ndarray:
    """Points in the camera frame [P, 3] (OpenCV axes) -> depth map [h * scale, w * scale] (0 = no point).
    K: fx, fy, cx, cy, k1, k2, k3 (radial distortion), w, h (image size)."""
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    k1, k2, k3, W, H = K["k1"], K["k2"], K["k3"], K["w"], K["h"]
    z = pts_cam[:, 2]
    keep = z > 0.5
    xn, yn, z = pts_cam[keep, 0] / z[keep], pts_cam[keep, 1] / z[keep], z[keep]
    r_img = np.hypot(max(cx, W - cx) / fx, max(cy, H - cy) / fy)
    r2 = xn * xn + yn * yn
    ok = r2 < _r_limit(k1, k2, k3, r_img) ** 2
    xn, yn, z, r2 = xn[ok], yn[ok], z[ok], r2[ok]
    f = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    u, v = (fx * xn * f + cx) * scale, (fy * yn * f + cy) * scale
    w, h = int(round(W * scale)), int(round(H * scale))
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z = ui[ok], vi[ok], z[ok]
    depth = np.full((h, w), np.inf, np.float32)
    order = np.argsort(-z)                                         # far first, so the nearest point wins
    depth[vi[order], ui[order]] = z[order]
    near = _min_filter(depth, window)
    depth[depth > near * (1 + tol) + 0.5] = np.inf                 # hidden behind a nearer point (parallax)
    depth[~np.isfinite(depth)] = 0
    return depth


def _av2_log(job) -> list:
    from PIL import Image
    log_dir, out_dir, cams, per_log, scale, max_dt = job
    log_id = os.path.basename(log_dir)
    ext = _se3_rows(_feather(os.path.join(log_dir, "calibration", "egovehicle_SE3_sensor.feather")), "sensor_name")
    intr = _feather(os.path.join(log_dir, "calibration", "intrinsics.feather"))
    Ks = {}
    for i in range(len(intr["sensor_name"])):
        n = intr["sensor_name"][i]
        n = n.decode() if isinstance(n, bytes) else str(n)
        Ks[n] = dict(fx=float(intr["fx_px"][i]), fy=float(intr["fy_px"][i]), cx=float(intr["cx_px"][i]),
                     cy=float(intr["cy_px"][i]), k1=float(intr["k1"][i]), k2=float(intr["k2"][i]),
                     k3=float(intr["k3"][i]), w=int(intr["width_px"][i]), h=int(intr["height_px"][i]))
    poses = _se3_rows(_feather(os.path.join(log_dir, "city_SE3_egovehicle.feather")), "timestamp_ns")
    lid_dir = os.path.join(log_dir, "sensors", "lidar")
    sweeps = sorted(int(f[:-8]) for f in os.listdir(lid_dir) if f.endswith(".feather"))
    if not sweeps:
        return []
    pick = sorted(set(np.linspace(0, len(sweeps) - 1, per_log).round().astype(int).tolist()))
    rows = []
    cam_ts = {}
    for cam in cams:
        d = os.path.join(log_dir, "sensors", "cameras", cam)
        cam_ts[cam] = np.array(sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".jpg")), np.int64) \
            if os.path.isdir(d) else np.zeros(0, np.int64)
    for si in pick:
        tl = sweeps[si]
        if tl not in poses:
            continue
        pt = _feather(os.path.join(lid_dir, f"{tl}.feather"))
        p_ego = np.stack([pt["x"], pt["y"], pt["z"]], -1).astype(np.float64)
        Rl, tl_t = poses[tl]
        p_city = p_ego @ Rl.T + tl_t
        for cam in cams:
            ts = cam_ts[cam]
            if ts.size == 0 or cam not in Ks or cam not in ext:
                continue
            tc = int(ts[np.argmin(np.abs(ts - tl))])
            if abs(tc - tl) > max_dt * 1e6 or tc not in poses:
                continue
            Rc, tc_t = poses[tc]
            Rs, ts_t = ext[cam]
            p_cam = ((p_city - tc_t) @ Rc - ts_t) @ Rs                # city -> ego(t_cam) -> camera
            depth = project_lidar(p_cam, Ks[cam], scale)
            if (depth > 0).sum() < 100:
                continue
            rel = os.path.join(log_id, cam, f"{tc}.png")
            os.makedirs(os.path.join(out_dir, log_id, cam), exist_ok=True)
            Image.fromarray(np.clip(depth * 256, 0, 65535).astype(np.uint16)).save(os.path.join(out_dir, rel))
            rows.append(dict(image=os.path.join(log_dir, "sensors", "cameras", cam, f"{tc}.jpg"), depth=rel,
                             scale=1 / 256, max_depth=200.0, min_cell_frac=0.02, source="argoverse2",
                             log=log_id, cam=cam, valid=round(float((depth > 0).mean()), 4)))
    return rows


def argoverse2(args) -> int:
    split_dir = os.path.join(os.path.abspath(args.root), args.split)
    logs = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    if args.logs > 0:
        logs = logs[:: max(1, len(logs) // args.logs)][:args.logs]
    out_dir = os.path.join(os.path.abspath(args.out_dir), args.split)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(os.path.join(split_dir, l), out_dir, args.cams.split(","), args.per_log, args.scale, args.max_dt)
            for l in logs]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_av2_log, jobs)):
            rows.extend(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(logs)} logs, {len(rows)} images", flush=True)
    out = os.path.join(out_dir, "manifest.jsonl")
    _write(out, rows)
    dens = np.mean([r["valid"] for r in rows]) if rows else 0.0
    print(f"argoverse2 {args.split}: {len(logs)} logs, {len(rows)} images (mean valid {dens:.3f}) -> {out}")
    return 0


# ---------------------------------------------------------------------------------- NYU Depth v2
def nyu(args) -> int:
    import h5py
    from PIL import Image
    from scipy.io import loadmat
    sp = loadmat(args.splits)
    splits = {"train": sp["trainNdxs"].ravel() - 1, "test": sp["testNdxs"].ravel() - 1}
    out_dir = os.path.abspath(args.out_dir)
    for s in ("rgb", "depth"):
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)
    with h5py.File(args.mat, "r") as f:
        imgs, deps = f["images"], f["depths"]                      # [N, 3, W, H], [N, W, H] (MATLAB order)
        for name, idx in splits.items():
            rows = []
            for i in idx:
                rgb = np.transpose(np.array(imgs[i]), (2, 1, 0))
                d = np.transpose(np.array(deps[i]), (1, 0)).astype(np.float32)
                if name == "test":                                 # Eigen crop: the area with valid Kinect depth
                    m = np.zeros_like(d, bool)
                    m[45:471, 41:601] = True
                    d[~m] = 0
                Image.fromarray(rgb).save(os.path.join(out_dir, "rgb", f"{i:04d}.png"))
                Image.fromarray(np.clip(d * 1000, 0, 65535).astype(np.uint16)).save(
                    os.path.join(out_dir, "depth", f"{i:04d}.png"))
                rows.append(dict(image=f"rgb/{i:04d}.png", depth=f"depth/{i:04d}.png", scale=0.001,
                                 min_depth=1e-3, max_depth=10.0, source="nyuv2"))
            _write(os.path.join(out_dir, f"{name}.jsonl"), rows)
            print(f"nyu {name}: {len(rows)} rows")
    return 0


def _write(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("omniworld")
    o.add_argument("--root", required=True)
    o.add_argument("--split", default="train")
    o.add_argument("--every", type=int, default=8, help="every N-th packed frame of each segment")
    o.add_argument("--max-depth", type=float, default=250.0)
    o.add_argument("--out", required=True)
    a = sub.add_parser("argoverse2")
    a.add_argument("--root", required=True, help="sensor root with train/ val/ test/")
    a.add_argument("--split", default="train")
    a.add_argument("--cams", default="ring_front_center,ring_front_left,ring_front_right")
    a.add_argument("--per-log", type=int, default=6, help="LiDAR sweeps per log, evenly spaced")
    a.add_argument("--logs", type=int, default=0, help="spread-out subset of N logs (0: all)")
    a.add_argument("--scale", type=float, default=0.5, help="depth map resolution / image resolution")
    a.add_argument("--max-dt", type=float, default=30.0, help="ms between sweep and image")
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--out-dir", required=True)
    n = sub.add_parser("nyu")
    n.add_argument("--mat", required=True)
    n.add_argument("--splits", required=True)
    n.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    return {"omniworld": omniworld, "argoverse2": argoverse2, "nyu": nyu}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
