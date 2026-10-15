"""E1.1 check of the nuScenes reader and the graph edges on v1.0-mini (T27's check_nuscenes.py).

    python scripts/mavt4d/check_nuscenes.py --dataroot $NUSC --version v1.0-mini --out results/e1_1

Writes e1_1_report.md and e1_1_report.json: data checks, then the graph-edge checks of gate M1 (projection
re-check, false-visible rate against LiDAR merged over +-5 sweeps, degree/provenance, prepare time).
Run under ``/usr/bin/time -v`` if you want the external peak-RSS log; the report records the same kernel
counter through getrusage (self, and joined DataLoader workers).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.nuscenes import CAM6, LIDAR, NuScenesSceneDataset, load_official_split
from mavt4d.eval.graph_checks import CHEB_MAX, DEPTH_BINS_M, false_visible_counts, recheck_projection_edges
from mavt4d.geometry.camera import invert_se3, project, transform_points, unproject
from mavt4d.geometry.hubs import build_hubs
from mavt4d.graph.builder import GraphConfig, build_graph
from mavt4d.graph.types import EdgeType

SPLITS = ("mini_train", "mini_val")
GCFG = GraphConfig()  # default graph configuration, as in training
FV_MAX = 0.05  # M1: false-visible rate against LiDAR merged over +-5 sweeps
REFS = ("train", "pm0", "pm1", "pm5", "pm5_ego3", "pm5_noground")  # false-visible references, see _GraphItems
GROUND_Z_M = 0.3  # pm5_noground: neighbour points below this world z (keyframe ego frame, origin on the ground)


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


def _worker_init(_):
    torch.set_num_threads(1)


def _prepare(sample, frac: float):
    """Training-side preparation of a tier-A sample: hold-out -> hubs -> graph (default GraphConfig)."""
    h = apply_holdout(sample, frac)
    hubs = build_hubs(h, GCFG.hubs_K, GCFG.hubs_M)
    graph, stats = build_graph(h, hubs, GCFG)
    return h, hubs, graph, stats


class _GraphItems(Dataset):
    """m3 item -> prepared graph. ``check=False``: (sample, hubs, graph) plus in-worker times, as a training
    loader would deliver them. ``check=True``: the graph-edge checks of one sample as a small dict.

    False-visible references (world frame, keyframe sweep paired with all 6 cameras):
    train = the graph's own z-buffer reference (input points of the keyframe sweep; must give 0);
    pm0 = all points of the keyframe sweep, held-out included;
    pm5 = pm0 plus the up to +-5 neighbouring LIDAR_TOP sweeps of the scene (``neighbour_points``: ego-body
    points and points inside the boxes interpolated at each sweep's time dropped); the M1 gate row.
    Attribution only (never a gate): pm1 = +-1 sweep; pm5_ego3 = pm5 with the ego exclusion widened to 3 m;
    pm5_noground = pm5 without neighbour points below GROUND_Z_M."""

    def __init__(self, ds, frac: float, n_sweeps: int, check: bool):
        self.ds, self.frac, self.n_sweeps, self.check = ds, frac, n_sweeps, check

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        t0 = time.perf_counter()
        sample = self.ds[i]
        t1 = time.perf_counter()
        h, hubs, graph, stats = _prepare(sample, self.frac)
        times = {"load_s": t1 - t0, "prepare_s": time.perf_counter() - t1}
        if not self.check:
            return h, hubs, graph, times
        full = transform_points(h.lidar_to_world[0], h.lidar_points[0][:, :3])
        nb, info = self.ds.neighbour_points(i, self.n_sweeps)
        nb1, _ = self.ds.neighbour_points(i, 1)
        nb_ego3, _ = self.ds.neighbour_points(i, self.n_sweeps, ego_radius=3.0)
        refs = {"train": [transform_points(h.lidar_to_world[0], h.input_points(0)[:, :3])], "pm0": [full],
                "pm1": [torch.cat([full, nb1])], "pm5": [torch.cat([full, nb])],
                "pm5_ego3": [torch.cat([full, nb_ego3])], "pm5_noground": [torch.cat([full, nb[nb[:, 2] >= GROUND_Z_M]])]}
        cams = self.ds.cameras
        return {**times, "stats": asdict(stats),
                "max_in_degree": {et.name: int(graph.in_degree(et).max()) for et in EdgeType},
                "recheck": recheck_projection_edges(h, hubs, graph, GCFG),
                "fv": {name: {cams[v]: c for (v, _), c in false_visible_counts(h, hubs, graph, ref, GCFG).items()}
                       for name, ref in refs.items()},
                "neighbours": info, "ref_points": {name: len(ref[0]) for name, ref in refs.items()}}


def _graph_loader(ds, frac: float, n_sweeps: int, check: bool, idx, workers: int):
    return DataLoader(Subset(_GraphItems(ds, frac, n_sweeps, check), idx), batch_size=None, num_workers=workers,
                      collate_fn=_identity, worker_init_fn=_worker_init)


def _hist_quantile(hist: list[int], q: float) -> int:
    """Smallest bin index whose cumulative count reaches fraction q of the total."""
    cum = np.cumsum(hist)
    return int(np.searchsorted(cum, q * cum[-1]))


def _degree(items: list[dict], et: str) -> dict:
    """Edges and provenance summed over samples, mean in-degree averaged, max in-degree maxed."""
    prov = [it["stats"]["provenance"].get(et, {}) for it in items]
    return {"edges": sum(it["stats"]["num_edges"].get(et, 0) for it in items),
            "mean_in_degree": float(np.mean([it["stats"]["mean_in_degree"].get(et, 0.0) for it in items])),
            "max_in_degree": max(it["max_in_degree"][et] for it in items),
            "provenance": {k: sum(p.get(k, 0) for p in prov) for k in sorted({k for p in prov for k in p})}}

def check_graph(ds, frac: float, n_sweeps: int, workers: int) -> dict:
    items = list(_graph_loader(ds, frac, n_sweeps, True, range(len(ds)), workers))
    rc = [it["recheck"] for it in items]
    hist = np.sum([r["centre_cheb_hist"] for r in rc], 0).tolist()
    fails = [d for r in rc for d in r["fail_min_dist_px"]]
    recheck = {"edges": sum(r["edges"] for r in rc), "passed": sum(r["passed"] for r in rc),
               "h2p_is_reverse_all": all(r["h2p_is_reverse"] for r in rc),
               "centre_in_patch": sum(r["centre_in_patch"] for r in rc), "centre_cheb_hist": hist,
               "centre_cheb_median": _hist_quantile(hist, 0.5), "centre_cheb_p95": _hist_quantile(hist, 0.95),
               "failed": len(fails), "fail_min_dist_px": _stats(fails) if fails else {"n": 0}}
    fv = {}
    for name in REFS:
        per_cam = {c: {"used": np.sum([it["fv"][name][c]["used"] for it in items], 0).tolist(),
                       "false": np.sum([it["fv"][name][c]["false"] for it in items], 0).tolist()} for c in ds.cameras}
        used, false = (np.sum([d[k] for d in per_cam.values()], 0).tolist() for k in ("used", "false"))
        fv[name] = {"per_camera": per_cam, "used": used, "false": false}
    deg = {et.name: _degree(items, et.name) for et in EdgeType}
    return {"samples": len(items), "recheck": recheck, "false_visible": fv, "degree": deg,
            "neighbour_sweeps": _stats([it["neighbours"]["sweeps"] for it in items]),
            "dropped_ego_points": sum(it["neighbours"]["dropped_ego"] for it in items),
            "dropped_box_points": sum(it["neighbours"]["dropped_box"] for it in items),
            "ref_points": {name: _stats([it["ref_points"][name] for it in items]) for name in REFS},
            "in_worker_s": {k: _stats([it[k] for it in items]) for k in ("load_s", "prepare_s")}}


def graph_timing(ds, frac: float, n: int, workers: int) -> dict:
    """Wall time of a loader that delivers (sample, hubs, graph) as training would, plus in-worker times."""
    t = time.perf_counter()
    times = [it[3] for it in _graph_loader(ds, frac, 0, False, range(min(n, len(ds))), workers)]
    wall = time.perf_counter() - t
    return {"n": len(times), "workers": workers, "wall_s": wall, "s_per_sample": wall / max(len(times), 1),
            **{k: _stats([x[k] for x in times]) for k in ("load_s", "prepare_s")}}


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
    ap.add_argument("--graph-workers", type=int, default=8, help="DataLoader workers for the graph passes")
    ap.add_argument("--graph-timing-n", type=int, default=160)
    ap.add_argument("--fv-sweeps", type=int, default=5, help="neighbour sweeps on each side for the pm5 reference")
    args = ap.parse_args()
    if args.version != "v1.0-mini":
        raise SystemExit("only v1.0-mini is checked here: the trainval m3 check is pending (full trainval JSON "
                         "needs several GB of RAM; run it as a Slurm CPU job or with a streaming reader)")
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"version": args.version, "dataroot": args.dataroot, "cameras": list(CAM6), "loadavg": os.getloadavg(),
           "m3": {}, "m2": {}, "timing": {}, "graph": {}, "graph_config": asdict(GCFG),
           "heldout_frac": args.heldout_frac, "fv_sweeps": args.fv_sweeps, "cpu_count": os.cpu_count()}
    t_all = time.perf_counter()
    with open(Path(args.dataroot) / args.version / "sample.json") as f:
        rep["n_sample_rows"] = len(json.load(f))
    for split in SPLITS:
        ds = NuScenesSceneDataset(args.dataroot, args.version, split, mode="m3")
        rep["m3"][split] = check_m3(ds, _split_scenes(ds, args.version, split), args.workers)
        if split == "mini_train":
            rep["devkit_crosscheck"] = devkit_crosscheck(ds, args.devkit_n)
            rep["timing"]["m3"] = [timing(ds, args.timing_n, w) for w in (0, 2)]
            rep["loadavg_graph_timing"] = os.getloadavg()
            rep["timing"]["graph_m3"] = graph_timing(ds, args.heldout_frac, args.graph_timing_n, args.graph_workers)
        t = time.perf_counter()
        rep["graph"][split] = check_graph(ds, args.heldout_frac, args.fv_sweeps, args.graph_workers)
        rep["graph"][split]["wall_s"] = time.perf_counter() - t
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
          ""]
    L += render_graph(r)
    L += ["## Pending", "",
          "- **trainval m3 check: pending.** Keyframes for all 850 scenes are on DGX, but loading the full trainval "
          "JSON on the headnode needs several GB of RAM; run it as a Slurm CPU job or after a streaming reader exists.",
          "- **M1 `data_wait_frac` < 0.10: pending (needs the T25 trainer).** The prepare time above is a loader-side "
          "number only; the GPU-side wait fraction is not measured here.", ""]
    return "\n".join(L)


def _rate(false, used) -> str:
    return f"{false / used:.2%}" if used else "n/a"


def _verdict(ok: bool) -> str:
    return "**PASS**" if ok else "**FAIL**"


def render_graph(r: dict) -> list[str]:
    g, cfg = r["graph"], r["graph_config"]
    rc = {sp: g[sp]["recheck"] for sp in SPLITS}
    edges, passed = sum(x["edges"] for x in rc.values()), sum(x["passed"] for x in rc.values())
    reverse = all(x["h2p_is_reverse_all"] for x in rc.values())
    fv5 = [sum(sum(g[sp]["false_visible"]["pm5"][k]) for sp in SPLITS) for k in ("false", "used")]
    proj_ok = passed == edges and reverse and edges > 0
    fv_ok = fv5[1] > 0 and fv5[0] / fv5[1] < FV_MAX
    bins = [f"[{a:g}, {b:g})" for a, b in zip((0,) + DEPTH_BINS_M, DEPTH_BINS_M + (float("inf"),))]
    L = ["## Graph edges (T27, gate M1)", "",
         f"Every m3 sample of {' and '.join(SPLITS)}: hold-out (frac {r['heldout_frac']}) -> `build_hubs` "
         f"(K={cfg['hubs_K']}, M={cfg['hubs_M']}) -> `build_graph` with the default `GraphConfig` (patch {cfg['patch']}, "
         f"z-buffer cell {cfg['zbuffer_cell_px']} px, tol {cfg['zbuffer_tol_m']} m, caps hub {cfg['proj_hub_cap']} / "
         f"patch {cfg['proj_patch_cap']}). Graph passes use {r['timing']['graph_m3']['workers']} DataLoader workers.", "",
         "| M1 criterion | measured | threshold | verdict |", "|---|---|---|---|",
         f"| projection edges re-project into their patch | {passed}/{edges} = {_rate(passed, edges)}; "
         f"H2P = reverse of P2H in every sample: {reverse} | 100% | {_verdict(proj_ok)} |",
         f"| false-visible vs LiDAR merged over ±{r['fv_sweeps']} sweeps (`pm5`, graph's cell rule: upper bound) "
         f"| {fv5[0]}/{fv5[1]} = {_rate(*fv5)} | < {FV_MAX:.0%} | {_verdict(fv_ok)} (upper bound, cell rule) |",
         "| data_wait_frac | not measured (needs T25) | < 0.10 | pending |", "",
         "### Projection re-check", "",
         "An edge passes when at least one member of its hub, lifted to world in float64 and projected with "
         "`project` (not the builder's float32 `visible_members` path), lands in the edge's patch in front of the "
         "camera (spec T11, test 1). Visibility is judged by the false-visible check below, not here.", "",
         "| split | P2H edges | passed | failed | hub centre in the edge's patch | centre distance [patches] median / p95 |",
         "|---|---|---|---|---|---|"]
    for sp, x in rc.items():
        L.append(f"| {sp} | {x['edges']} | {x['passed']} | {x['failed']} | {_rate(x['centre_in_patch'], x['edges'])} "
                 f"| {x['centre_cheb_median']} / {x['centre_cheb_p95']} |")
    for sp, x in rc.items():
        if x["failed"]:
            d = x["fail_min_dist_px"]
            L.append(f"\n- {sp}: failed edges' nearest member lies {_f(d['min'])}–{_f(d['max'])} px outside the patch "
                     f"(median {_f(d['median'])}); < 0.01 px would be a float32/float64 border flip.")
    L += ["", "The hub-centre column is information only: the task brief phrased the check as \"the hub centre "
          "re-projects into the connected patch\", but edges are built from members (spec T11), so a hub whose "
          f"{cfg['hubs_M']} members span several patches links to all of them and its centre lies in only one. "
          "The gate uses the spec's member definition. The centre distance is Chebyshev in patch units "
          f"(capped at {CHEB_MAX}).", "",
          "### False-visible rate", "",
          "Members counted: one per hub member slot, visible to the builder's z-buffer and inside the patch of a kept "
          "(post-cap) PROJ_P2H edge, i.e. exactly the members behind the edge weights. A member is false-visible "
          "when `zbuffer_visibility` with the same cell and tol rejects it against a reference cloud:", "",
          "- `train`: the graph's own reference (input points of the keyframe sweep). Must be 0; checks the counting.",
          "- `pm0`: every point of the keyframe sweep, held-out included (effect of hold-out sparsity alone).",
          f"- `pm5` (gate): `pm0` plus up to ±{r['fv_sweeps']} neighbouring LIDAR_TOP sweeps inside the scene, each placed "
          "with its own ego pose; points with |x|, |y| < 1 m in the sensor frame (ego body, as devkit `remove_close`) "
          "and points inside the annotation boxes interpolated at the sweep's time, dilated 0.3 m (moving objects "
          "smear across sweeps), are dropped from the neighbours. Same cell and tol as the graph, so grazing ground "
          "at range can be rejected by a denser cloud of the same surface: this rate is an upper bound. The synthetic "
          "test of T11 (`gt_visible`) excludes grazing on purpose (farthest of the 4 surrounding GT depths); no such "
          "definition is applied to real data here.",
          "- Attribution rows, not gates: `pm1` = ±1 sweep (±50 ms, so pose drift is negligible: a jump from `pm0` "
          "to `pm1` comes from point density alone; growth from `pm1` to `pm5` mixes density with registration over "
          "time and does not separate them); `pm5_ego3` = ego exclusion widened to |x|, |y| < 3 m (a collapse on "
          "CAM_FRONT / CAM_BACK means ghosts of the ego body from neighbour sweeps); `pm5_noground` = neighbour points "
          f"below world z {GROUND_Z_M} m dropped (a collapse in the far bins means grazing ground).", "",
          "| split | reference | used members | false-visible | rate | " + " | ".join(bins) + " |",
          "|---|---|---|---|---|" + "---|" * len(bins)]
    for sp in SPLITS:
        for name in REFS:
            f = g[sp]["false_visible"][name]
            per_bin = " | ".join(_rate(a, b) for a, b in zip(f["false"], f["used"]))
            L.append(f"| {sp} | {name} | {sum(f['used'])} | {sum(f['false'])} | {_rate(sum(f['false']), sum(f['used']))} "
                     f"| {per_bin} |")
    refs = [n for n in REFS if n != "train"]
    L += ["", "Per camera, false-visible rate per reference (both splits):", "",
          "| camera | used | " + " | ".join(refs) + " |", "|---|---|" + "---|" * len(refs)]
    for c in r["cameras"]:
        u = sum(sum(g[sp]["false_visible"]["pm5"]["per_camera"][c]["used"]) for sp in SPLITS)
        f = [sum(sum(g[sp]["false_visible"][n]["per_camera"][c]["false"]) for sp in SPLITS) for n in refs]
        L.append(f"| {c} | {u} | " + " | ".join(_rate(x, u) for x in f) + " |")
    L += [""]
    for sp in SPLITS:
        x = g[sp]
        L.append(f"- {sp}: neighbour sweeps per sample {x['neighbour_sweeps']['min']:.0f}–{x['neighbour_sweeps']['max']:.0f} "
                 f"(median {x['neighbour_sweeps']['median']:.0f}); reference points per sample median: "
                 + ", ".join(f"{n} {x['ref_points'][n]['median']:.0f}" for n in REFS)
                 + f"; dropped from neighbours: {x['dropped_ego_points']} ego-body, {x['dropped_box_points']} in boxes.")
    L += ["", "### Degree and provenance", "",
          "Summed over samples (edges, provenance), mean over samples of the mean in-degree over eligible receivers, "
          "max in-degree over all nodes. m3 has T = 1, so no TRACK edges.", ""]
    for sp in SPLITS:
        L += [f"{sp} ({g[sp]['samples']} samples):", "",
              "| type | edges | mean in-degree | max in-degree | provenance |", "|---|---|---|---|---|"]
        for et, d in g[sp]["degree"].items():
            prov = ", ".join(f"{k}={v}" for k, v in d["provenance"].items()) or "-"
            L.append(f"| {et} | {d['edges']} | {d['mean_in_degree']:.2f} | {d['max_in_degree']} | {prov} |")
        L.append("")
    caps = {"SPATIAL": cfg["spatial_cap"], "PROJ_P2H": cfg["proj_hub_cap"], "PROJ_H2P": cfg["proj_patch_cap"],
            "NB3D": min(cfg["nb3d_k"], cfg["hubs_K"] - 1)}
    worst = {et: max(g[sp]["degree"][et]["max_in_degree"] for sp in SPLITS) for et in caps}
    L.append("Caps: " + "; ".join(f"{et} max {worst[et]} vs cap {c} ({'ok' if worst[et] <= c else 'EXCEEDED'})"
                                  for et, c in caps.items()) + ".")
    t = r["timing"]["graph_m3"]
    L += ["", "### Prepare time (mini_train m3)", "",
          f"Loader delivering (sample, hubs, graph) with {t['workers']} workers (1 torch thread each) over {t['n']} "
          f"samples: wall {t['wall_s']:.1f} s = **{t['s_per_sample']:.3f} s/sample**. Inside a worker: load (6 JPEGs + "
          f"sweep) median {_f(t['load_s']['median'])} s (p95 {_f(t['load_s']['p95'])}), hold-out + hubs + graph median "
          f"{_f(t['prepare_s']['median'])} s (p95 {_f(t['prepare_s']['p95'])}). Headnode load average before this pass: "
          f"{', '.join(f'{x:.0f}' for x in r['loadavg_graph_timing'])} on a shared {r.get('cpu_count', '?')}-core "
          "headnode, so these are "
          "upper bounds for a dedicated node.", ""]
    for sp in SPLITS:
        w = g[sp]["in_worker_s"]
        L.append(f"- {sp} check pass ({g[sp]['wall_s']:.0f} s wall incl. checks): in-worker load median "
                 f"{_f(w['load_s']['median'])} s, prepare median {_f(w['prepare_s']['median'])} s.")
    L.append("")
    return L


if __name__ == "__main__":
    main()
