"""E1.1 data check of the nuScenes reader on v1.0-mini (first slice of T27's check_nuscenes.py).

    python scripts/mavt4d/check_nuscenes.py --dataroot $NUSC --version v1.0-mini --out results/e1_1

Writes e1_1_report.md and e1_1_report.json. Graph-edge metrics (T27) are a marked TODO, not computed.
Run under ``/usr/bin/time -v`` if you want the external peak-RSS log; the report records the same kernel
counter through getrusage (self, and joined DataLoader workers).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.nuscenes import CAM6, LIDAR, NuScenesSceneDataset, load_official_split
from mavt4d.geometry.camera import invert_se3, project, transform_points, unproject

SPLITS = ("mini_train", "mini_val")


def _identity(x):
    return x


def _stats(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def _rigid_errors(T: torch.Tensor) -> tuple[float, float, float]:
    """max |R^T R - I|, max |det R - 1|, max |bottom row - [0 0 0 1]| over a stack (..., 4, 4)."""
    T = T.double().reshape(-1, 4, 4)
    R = T[:, :3, :3]
    orth = (R.transpose(1, 2) @ R - torch.eye(3, dtype=T.dtype)).abs().max().item()
    det = (torch.linalg.det(R) - 1).abs().max().item()
    row = (T[:, 3] - torch.tensor([0, 0, 0, 1.0], dtype=T.dtype)).abs().max().item()
    return orth, det, row


def _iterate(ds, idx, workers: int):
    return DataLoader(Subset(ds, idx), batch_size=None, num_workers=workers, collate_fn=_identity)


def _direct_K(tables, sd: str, H: int, W: int) -> np.ndarray:
    """Recompute K' from the raw calibration and the JPEG's actual size (independent of the reader)."""
    with Image.open(tables.dataroot / tables.sample_data[sd]["filename"]) as im:
        w, h = im.size
    s = H / h
    left = (round(w * s) - W) // 2
    K = tables.raw_intrinsics(sd)
    return np.array([[K[0, 0] * s, 0, K[0, 2] * s - left], [0, K[1, 1] * s, K[1, 2] * s], [0, 0, 1]])


class Acc:
    """Running maxima and lists for one (split, mode)."""

    def __init__(self):
        self.lists: dict[str, list] = {}
        self.maxes: dict[str, float] = {}

    def add(self, key: str, v) -> None:
        self.lists.setdefault(key, []).extend(np.atleast_1d(np.asarray(v, dtype=np.float64)).tolist())

    def mx(self, key: str, v: float) -> None:
        self.maxes[key] = max(self.maxes.get(key, 0.0), float(v))


def _geometry(acc: Acc, s, tables, sds_frame0: list[str]) -> None:
    """Shared checks: rigid validity, recentring and intrinsics."""
    for name, T in (("cam_to_world", s.cam_to_world), ("lidar_to_world", s.lidar_to_world)):
        orth, det, row = _rigid_errors(T)
        acc.mx(f"{name}.orth_err", orth), acc.mx(f"{name}.det_err", det), acc.mx(f"{name}.bottom_row_err", row)
    ref_ego = s.lidar_to_world[0].double() @ invert_se3(torch.from_numpy(tables.sensor_to_ego(s.lidar_keys[0])))
    acc.mx("recentre.ref_lidar_ego_minus_I", (ref_ego - torch.eye(4, dtype=torch.float64)).abs().max().item())
    H, W = s.images.shape[-2:]
    for v, sd in enumerate(sds_frame0):
        acc.mx("intrinsics.max_abs_diff", np.abs(s.intrinsics[v].double().numpy() - _direct_K(tables, sd, H, W)).max())


def _split_scenes(ds, version: str, split: str) -> list[str]:
    """Scene tokens of an official split, straight from the devkit list (not from the dataset index)."""
    names = set(load_official_split(version, split))
    return [tok for tok, sc in ds.tables.scene.items() if sc["name"] in names]


def check_m3(ds, scenes: list[str], workers: int) -> dict:
    tb, acc = ds.tables, Acc()
    cams = ds.cameras
    per_cam = {c: {"front": [], "inside": [], "dt": []} for c in cams}
    for i, s in enumerate(_iterate(ds, range(len(ds)), workers)):
        sample = ds.items[i][1]
        dt = (s.frame_times[:, 0] - s.lidar_times[0]).numpy()
        acc.add("max_abs_dt_cam_lidar_s", np.abs(dt).max())
        _geometry(acc, s, tb, [tb.keyframe_data[sample][c] for c in cams])
        H, W = s.images.shape[-2:]
        pw = transform_points(s.lidar_to_world[0], s.lidar_points[0][:, :3])
        any_inside = torch.zeros(pw.shape[0], dtype=torch.bool)
        for v, c in enumerate(cams):
            uv, d = project(pw, s.intrinsics[v], s.cam_to_world[v, 0])
            front = d > 0
            inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            any_inside |= inside
            per_cam[c]["front"].append(front.float().mean().item())
            per_cam[c]["inside"].append(inside.float().mean().item())
            per_cam[c]["dt"].append(float(dt[v]))
            back = unproject(uv[inside], d[inside], s.intrinsics[v], s.cam_to_world[v, 0])
            err = (back - pw[inside]).norm(dim=1)
            if err.numel():
                acc.mx("roundtrip.max_err_m", err.max().item())
                acc.add("roundtrip.median_err_m_per_cam", err.median().item())
        acc.add("frac_points_inside_any_camera", any_inside.float().mean().item())
    out = {"samples": len(ds), "samples_expected_from_scenes": sum(tb.scene[sc]["nbr_samples"] for sc in scenes),
           "max_abs_dt_cam_lidar_s": _stats(acc.lists["max_abs_dt_cam_lidar_s"]),
           "frac_points_inside_any_camera": _stats(acc.lists["frac_points_inside_any_camera"]),
           "roundtrip_median_err_m": _stats(acc.lists.get("roundtrip.median_err_m_per_cam", [])),
           "maxes": acc.maxes,
           "per_camera": {c: {"mean_frac_front": float(np.mean(d["front"])),
                              "mean_frac_inside": float(np.mean(d["inside"])),
                              "median_dt_cam_minus_lidar_s": float(np.median(d["dt"]))} for c, d in per_cam.items()}}
    return out


def check_m2(ds, scenes: list[str], stride: int, frac: float, workers: int) -> dict:
    tb, acc = ds.tables, Acc()
    expected = sum(max(0, len(tb.chain(sc, ds.m2_camera)) - ds.clip_len + 1) for sc in scenes)
    idx = [i for i, (_, start) in enumerate(ds.items) if start % stride == 0]
    gap_windows = dup_windows = pairs = mismatches = 0
    prev = None  # (scene, {key: mask}) of the previous sampled window
    for i, s in zip(idx, _iterate(ds, idx, workers)):
        scene, start = ds.items[i]
        ft = s.frame_times[0].numpy()
        spacing = np.diff(ft)
        acc.add("frame_spacing_s", spacing)
        gap_windows += int(spacing.max() > 1.5 / 12)
        acc.add("nearest_sweep_abs_dt_s", np.abs(s.lidar_times.numpy() - ft))
        dup_windows += int(len(set(s.lidar_keys)) < len(s.lidar_keys))
        _geometry(acc, s, tb, [ds._chain(scene, ds.m2_camera)[start]])
        h = apply_holdout(s, frac)
        masks: dict[str, torch.Tensor] = {}
        for k, m in zip(h.lidar_keys, h.heldout):
            if k in masks:  # same sweep twice in one window
                pairs, mismatches = pairs + 1, mismatches + int(not torch.equal(masks[k], m))
            masks[k] = m
        if prev is not None and prev[0] == scene:
            for k in masks.keys() & prev[1].keys():
                pairs, mismatches = pairs + 1, mismatches + int(not torch.equal(masks[k], prev[1][k]))
        prev = (scene, masks)
    return {"windows": len(ds), "windows_expected_from_chains": expected, "checked_windows": len(idx),
            "check_stride": stride, "frame_spacing_s": _stats(acc.lists["frame_spacing_s"]),
            "frac_windows_with_gap_gt_1.5_frames": gap_windows / max(len(idx), 1),
            "nearest_sweep_abs_dt_s": _stats(acc.lists["nearest_sweep_abs_dt_s"]),
            "frac_windows_with_repeated_sweep": dup_windows / max(len(idx), 1),
            "heldout_shared_key_pairs": pairs, "heldout_mismatches": mismatches, "maxes": acc.maxes}


def devkit_crosscheck(ds, n: int) -> dict:
    """uv from the devkit's own map_pointcloud_to_image chain (pyquaternion + view_points, float64, full
    resolution), mapped through u*s - left, v*s, against the reader + project() in float32."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    tb, worst_uv, worst_d, compared = ds.tables, 0.0, 0.0, 0
    for i in range(min(n, len(ds))):
        s = ds[i]
        sample = ds.items[i][1]
        H, W = s.images.shape[-2:]
        lidar_sd = tb.keyframe_data[sample][LIDAR]
        pw = transform_points(s.lidar_to_world[0], s.lidar_points[0][:, :3])
        pc0 = LidarPointCloud.from_file(str(tb.dataroot / tb.sample_data[lidar_sd]["filename"]))
        for v, cam in enumerate(ds.cameras):
            cam_sd = tb.keyframe_data[sample][cam]
            cs_l, ep_l, ep_c, cs_c = (_rec(tb, sd, kind) for sd, kind in
                                      ((lidar_sd, "cs"), (lidar_sd, "ep"), (cam_sd, "ep"), (cam_sd, "cs")))
            pc = LidarPointCloud(pc0.points.copy())  # devkit map_pointcloud_to_image, steps 1-4
            pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix), pc.translate(np.array(cs_l["translation"]))
            pc.rotate(Quaternion(ep_l["rotation"]).rotation_matrix), pc.translate(np.array(ep_l["translation"]))
            pc.translate(-np.array(ep_c["translation"])), pc.rotate(Quaternion(ep_c["rotation"]).rotation_matrix.T)
            pc.translate(-np.array(cs_c["translation"])), pc.rotate(Quaternion(cs_c["rotation"]).rotation_matrix.T)
            K = np.array(cs_c["camera_intrinsic"])
            uv_dev = view_points(pc.points[:3], K, normalize=True)[:2].T
            rec = tb.sample_data[cam_sd]
            sc = H / rec["height"]
            left = (round(rec["width"] * sc) - W) // 2
            uv_dev = uv_dev * sc - np.array([left, 0.0])
            uv, d = project(pw, s.intrinsics[v], s.cam_to_world[v, 0])
            ok = (pc.points[2] > 1.0) & (d.numpy() > 1.0)
            ok &= (uv_dev[:, 0] >= 0) & (uv_dev[:, 0] < W) & (uv_dev[:, 1] >= 0) & (uv_dev[:, 1] < H)
            worst_uv = max(worst_uv, float(np.abs(uv.double().numpy()[ok] - uv_dev[ok]).max()))
            worst_d = max(worst_d, float(np.abs(d.double().numpy()[ok] - pc.points[2][ok]).max()))
            compared += int(ok.sum())
    return {"samples": min(n, len(ds)), "points_compared": compared, "max_abs_uv_diff_px": worst_uv,
            "max_abs_depth_diff_m": worst_d}


def _rec(tb, sd: str, kind: str) -> dict:
    r = tb.sample_data[sd]
    return tb.calibrated_sensor[r["calibrated_sensor_token"]] if kind == "cs" else tb.ego_pose[r["ego_pose_token"]]


def timing(ds, n: int, workers: int) -> dict:
    t = time.perf_counter()
    count = sum(1 for _ in _iterate(ds, range(min(n, len(ds))), workers))
    wall = time.perf_counter() - t
    return {"n": count, "workers": workers, "wall_s": wall, "s_per_sample": wall / max(count, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers for the full check passes")
    ap.add_argument("--m2-stride", type=int, default=4, help="check m2 windows whose start %% stride == 0")
    ap.add_argument("--timing-n", type=int, default=24)
    ap.add_argument("--devkit-n", type=int, default=8)
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    args = ap.parse_args()
    if args.version != "v1.0-mini":
        raise SystemExit("only v1.0-mini is checked here: the trainval m3 check is pending (full trainval JSON "
                         "needs several GB of RAM; run it as a Slurm CPU job or with a streaming reader)")
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"version": args.version, "dataroot": args.dataroot, "cameras": list(CAM6), "loadavg": os.getloadavg(),
           "m3": {}, "m2": {}, "timing": {}}
    t_all = time.perf_counter()
    with open(Path(args.dataroot) / args.version / "sample.json") as f:
        rep["n_sample_rows"] = len(json.load(f))
    for split in SPLITS:
        ds = NuScenesSceneDataset(args.dataroot, args.version, split, mode="m3")
        rep["m3"][split] = check_m3(ds, _split_scenes(ds, args.version, split), args.workers)
        if split == "mini_train":
            rep["devkit_crosscheck"] = devkit_crosscheck(ds, args.devkit_n)
            rep["timing"]["m3"] = [timing(ds, args.timing_n, w) for w in (0, 2)]
        del ds
        ds = NuScenesSceneDataset(args.dataroot, args.version, split, mode="m2")
        rep["m2"][split] = check_m2(ds, _split_scenes(ds, args.version, split), args.m2_stride,
                                     args.heldout_frac, args.workers)
        if split == "mini_train":
            rep["timing"]["m2"] = [timing(ds, args.timing_n, w) for w in (0, 2)]
        del ds
    rep["wall_s"] = time.perf_counter() - t_all
    rep["peak_rss_mb"] = {"self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          "largest_joined_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}
    (out / "e1_1_report.json").write_text(json.dumps(rep, indent=2))
    (out / "e1_1_report.md").write_text(render(rep))
    print(f"wrote {out / 'e1_1_report.md'}")


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}g}"


def render(r: dict) -> str:
    L = [f"# E1.1 nuScenes data check ({r['version']})", "",
         f"dataroot `{r['dataroot']}`; wall {r['wall_s']:.0f} s; headnode load average at start "
         f"{', '.join(f'{x:.0f}' for x in r['loadavg'])} (timings are inflated by a shared, loaded headnode).", "",
         "## Sample counts", "",
         "| split | m3 samples | m3 expected (sum of scene.nbr_samples) | m2 windows (clip 8, stride 1) "
         "| m2 expected (sum of chain length - 7) |", "|---|---|---|---|---|"]
    for sp in SPLITS:
        L.append(f"| {sp} | {r['m3'][sp]['samples']} | {r['m3'][sp]['samples_expected_from_scenes']} | "
                 f"{r['m2'][sp]['windows']} | {r['m2'][sp]['windows_expected_from_chains']} |")
    L.append(f"\nRows in sample.json: {r['n_sample_rows']}; m3 total over both splits: "
             f"{sum(r['m3'][sp]['samples'] for sp in SPLITS)}.")
    L += ["", "## Timestamps", "", "| split | quantity | n | median | p95 | max |", "|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for name, st in (("m3 max over 6 cams of abs(t_cam - t_lidar) [s]", r["m3"][sp]["max_abs_dt_cam_lidar_s"]),
                         ("m2 frame spacing [s]", r["m2"][sp]["frame_spacing_s"]),
                         ("m2 nearest-sweep abs(dt) [s]", r["m2"][sp]["nearest_sweep_abs_dt_s"])):
            L.append(f"| {sp} | {name} | {st['n']} | {_f(st['median'])} | {_f(st['p95'])} | {_f(st['max'])} |")
    L += [""]
    for sp in SPLITS:
        m2 = r["m2"][sp]
        L.append(f"- {sp} m2 (checked {m2['checked_windows']} windows with start % {m2['check_stride']} == 0, "
                 f"which cover every consecutive frame pair): min spacing {_f(m2['frame_spacing_s']['min'])} s; "
                 f"windows with a gap > 1.5 frames (0.125 s): {m2['frac_windows_with_gap_gt_1.5_frames']:.1%}; "
                 f"windows where two frames share one nearest sweep: {m2['frac_windows_with_repeated_sweep']:.1%}.")
    L += ["", "Per camera (mini_train m3): median t_cam - t_lidar and LiDAR points landing in front of / inside the "
          "448x256 crop.", "", "| camera | median dt [s] | front | inside |", "|---|---|---|---|"]
    for c, d in r["m3"]["mini_train"]["per_camera"].items():
        L.append(f"| {c} | {_f(d['median_dt_cam_minus_lidar_s'])} | {d['mean_frac_front']:.3f} | {d['mean_frac_inside']:.3f} |")
    L += ["", "## Geometry", "", "| split | mode | check | max |", "|---|---|---|---|"]
    for sp in SPLITS:
        for mode in ("m3", "m2"):
            for k, v in sorted(r[mode][sp]["maxes"].items()):
                L.append(f"| {sp} | {mode} | {k} | {v:.3g} |")
    for sp in SPLITS:
        st = r["m3"][sp]["frac_points_inside_any_camera"]
        L.append(f"\n- {sp} m3: fraction of a sweep's points inside at least one camera: median {st['median']:.3f} "
                 f"(min {st['min']:.3f}, max {st['max']:.3f}); project->unproject median error per camera "
                 f"{_f(r['m3'][sp]['roundtrip_median_err_m']['median'])} m (float32, only measures numeric error).")
    dk = r["devkit_crosscheck"]
    L += ["", f"Independent check against the devkit's map_pointcloud_to_image chain (pyquaternion, view_points, "
          f"float64) on {dk['samples']} mini_train keyframes x 6 cameras, {dk['points_compared']} points with depth > 1 m: "
          f"max |uv diff| = {dk['max_abs_uv_diff_px']:.3g} px, max |depth diff| = {dk['max_abs_depth_diff_m']:.3g} m.",
          "", "## Held-out stability", ""]
    for sp in SPLITS:
        m2 = r["m2"][sp]
        L.append(f"- {sp}: {m2['heldout_shared_key_pairs']} (window, window) pairs sharing a lidar_key compared "
                 f"(overlapping checked windows and repeats inside one window): {m2['heldout_mismatches']} mismatches.")
    L += ["", "## Load time and memory", "", "| mode | workers | n | wall [s] | s/sample |", "|---|---|---|---|---|"]
    for mode in ("m3", "m2"):
        for t in r["timing"][mode]:
            L.append(f"| {mode} | {t['workers']} | {t['n']} | {t['wall_s']:.1f} | {t['s_per_sample']:.3f} |")
    rss = r["peak_rss_mb"]
    L += ["", f"Peak RSS (getrusage, same counter as `/usr/bin/time -v`): main process {rss['self']:.0f} MB, largest "
          f"joined DataLoader worker {rss['largest_joined_child']:.0f} MB. The `/usr/bin/time -v` log of this run, "
          "if wrapped, sits beside this report as `time_v.log`.",
          "", "## Pending", "",
          "- **trainval m3 check: pending.** Keyframes for all 850 scenes are on DGX, but loading the full trainval "
          "JSON on the headnode needs several GB of RAM; run it as a Slurm CPU job or after a streaming reader exists.",
          "- **TODO (T27, needs T11/T12 graph code): graph-edge metrics** — projection edges passing re-projection "
          "(must be 100%), false-visible rate against LiDAR merged over ±5 sweeps (< 5%), degree and provenance "
          "table, per-sample prepare time with num_workers=8. Not computed here; no numbers are claimed.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
