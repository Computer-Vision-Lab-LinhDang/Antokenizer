"""Calibrate the M1 false-visible rule on synthetic scenes with exact visibility (E1.1).

    python scripts/mavt4d/calibrate_false_visible.py --out results/e1_1

Renders M3_LIKE scenes and runs the nuScenes pipeline on them (hold-out -> hubs -> default graph). A grid of
occlusion rules is scored against references of the keyframe sweep / +-1 / +-5 sweeps: error = share of
GT-correct members flagged, recall = share of GT-occluded (known-wrong) members flagged, plus the rule's rate on
the builder's edges next to their true rate. Decisions are fixed before looking (see MISCALIBRATED and
choose_rule): a rule is chosen on CALIB_SEEDS and every reported number comes from REPORT_SEEDS.
Writes fv_calibration.json and fv_calibration.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from mavt4d.data.holdout import apply_holdout  # noqa: E402
from mavt4d.data.synthetic import make_synthetic_scene  # noqa: E402
from mavt4d.eval.fv_calibration import (GAP_BINS_M, INC_BINS, M3_LIKE, calibrate_scene,  # noqa: E402
                                        merge_counts, reference_clouds)
from mavt4d.eval.graph_checks import DEPTH_BINS_M, FVRule  # noqa: E402
from mavt4d.geometry.hubs import build_hubs  # noqa: E402
from mavt4d.graph.builder import GraphConfig, build_graph  # noqa: E402

GCFG = GraphConfig()
SIDES = (0, 1, 5)
REFS = ("pm0", "pm1", "pm5", "pm1_allbox", "pm5_allbox")  # pmN: static boxes kept as structure; _allbox: mini mirror
CALIB_SEEDS, REPORT_SEEDS = tuple(range(16)), tuple(range(16, 32))
R0 = "c4"  # the M1 rule as first measured: cell 4 px, tol 0.5 m
RULES = {f"c{c}" + ("+rel" if rel else "") + ("+plane" if plane else ""):
         FVRule(cell_px=c, tol_m=0.5, tol_rel=0.05 if rel else 0.0, plane_tau=0.2 if plane else None)
         for c in (4, 2) for rel in (False, True) for plane in (False, True)}
MISCALIBRATED = 0.05  # R0 error on GT-correct members at +-5 at or above this: perfect edges alone would fail M1
TIE = 0.005  # Youden J within this of the best counts as tied; the rule with fewer changes wins
AMENDMENT = (
    "Selection amended on 2026-10-15 after seeing calibration seed 1 and before any report seed. The pre-set "
    "constraint (recall at ±5 at most 5 points below R0's) is degenerate: R0 at ±5 flags most GT-correct members, "
    "so its recall there carries little information and only rules that over-flag as much can meet it. The rule is "
    "now the one with the highest Youden J = recall − error at ±5 on the calibration seeds (ties within 0.5 point "
    "go to fewer changes). Also amended at the same time: neighbour sweeps keep static boxes as structure (the "
    "first version dropped every box, leaving the ±5 sweeps without any occluder), and 16 + 16 seeds instead of "
    "4 + 4 (GT-wrong members are scarce).")


def _rate(flag: dict, n: dict, split: str, label: str = "depth", b: int | None = None) -> float | None:
    f, d = flag[split][label], n[split][label]
    num, den = (sum(f), sum(d)) if b is None else (f[b], d[b])
    return num / den if den else None


def _complexity(name: str) -> int:
    return name.count("+") + (not name.startswith("c4"))


def choose_rule(agg: dict) -> tuple[str | None, dict]:
    """None if R0 is not miscalibrated; else the rule with the highest Youden J = recall - error at +-5, ties
    (within TIE) broken towards fewer changes."""
    n, flag = agg["n"], agg["flag"]
    err = {k: _rate(flag[k]["pm5"], n, "correct") for k in RULES}
    rec = {k: _rate(flag[k]["pm5"], n, "wrong") for k in RULES}
    youden = {k: rec[k] - err[k] for k in RULES}
    why = {"r0_error_pm5": err[R0], "error_pm5": err, "recall_pm5": rec, "youden_pm5": youden}
    if err[R0] < MISCALIBRATED:
        return None, why
    best = max(youden.values())
    return min((k for k in RULES if youden[k] >= best - TIE), key=lambda k: (_complexity(k), -youden[k])), why


def run_seed(seed: int, frac: float) -> tuple[dict, dict]:
    sample, gt = make_synthetic_scene(M3_LIKE, seed)
    h = apply_holdout(sample, frac)
    hubs = build_hubs(h, GCFG.hubs_K, GCFG.hubs_M)
    graph, _ = build_graph(h, hubs, GCFG)
    moving, info = reference_clouds(M3_LIKE, seed, h, SIDES, drop="moving")
    allbox, info_all = reference_clouds(M3_LIKE, seed, h, SIDES[1:], drop="all")
    refs = {**{f"pm{k}": r for k, r in moving.items()}, **{f"pm{k}_allbox": r for k, r in allbox.items()}}
    return (calibrate_scene(h, gt, hubs, graph, GCFG, refs, RULES),
            {"ref_points": {k: len(r) for k, r in refs.items()}, "sweeps": info, "sweeps_allbox": info_all})


def aggregate(seeds: tuple[int, ...], frac: float) -> tuple[dict, list[dict]]:
    agg, infos = None, []
    for seed in seeds:
        t = time.perf_counter()
        counts, info = run_seed(seed, frac)
        infos.append({"seed": seed, "wall_s": time.perf_counter() - t, **info})
        agg = counts if agg is None else merge_counts(agg, counts)
        print(f"seed {seed}: {infos[-1]['wall_s']:.1f} s", flush=True)
    return agg, infos


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    calib, calib_info = aggregate(CALIB_SEEDS, args.heldout_frac)
    chosen, why = choose_rule(calib)
    report, report_info = aggregate(REPORT_SEEDS, args.heldout_frac)
    rep = {"config": asdict(M3_LIKE), "graph_config": asdict(GCFG), "rules": {k: asdict(v) for k, v in RULES.items()},
           "r0": R0, "chosen": chosen, "selection": why, "amendment": AMENDMENT,
           "decision": {"miscalibrated_at": MISCALIBRATED, "tie": TIE},
           "calib_seeds": CALIB_SEEDS, "report_seeds": REPORT_SEEDS, "calib": calib, "report": report,
           "calib_info": calib_info, "report_info": report_info}
    rep = json.loads(json.dumps(rep))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fv_calibration.json").write_text(json.dumps(rep, indent=1))
    (out / "fv_calibration.md").write_text(render(rep))
    print(f"wrote {out / 'fv_calibration.md'}")


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def _bins(edges: tuple[float, ...]) -> list[str]:
    return [f"[{a:g}, {b:g})" for a, b in zip((0,) + edges, edges + (float("inf"),))]


def render(r: dict) -> str:
    rep, n, chosen, r0 = r["report"], r["report"]["n"], r["chosen"], r["r0"]
    cfg, sel = r["config"], r["selection"]
    rules = [r0] + ([chosen] if chosen and chosen != r0 else [])
    true_fv = sum(n["used_true_fv"]["depth"]) / max(sum(n["used"]["depth"]), 1)
    L = ["# Calibration of the M1 false-visible rule on synthetic GT", "",
         f"Synthetic scenes shaped like nuScenes m3: {cfg['V']} cameras {cfg['W']}x{cfg['H']}, fov {cfg['fov_deg']:.1f}° "
         f"(fx ≈ 360 px), pitch {cfg['cam_pitch_deg']:g}°, {cfg['lidar_beams']}-beam LiDAR ({cfg['lidar_elev_deg'][0]:g}..."
         f"{cfg['lidar_elev_deg'][1]:g}°, {cfg['lidar_az']} steps) at 20 Hz, ego {cfg['ego_speed']:g} m/s, "
         f"{cfg['n_boxes']} boxes ({cfg['n_moving']} moving) on a ground plane. Same pipeline as mini: hold-out "
         f"0.3 -> hubs (K={r['graph_config']['hubs_K']}, M={r['graph_config']['hubs_M']}) -> default graph.", "",
         "References: every point of the keyframe sweep (held-out included), plus the sweeps rendered at ±1 or ±5 x "
         "50 ms from the same scene minus their points inside moving boxes at their own time (dilated 0.3 m); static "
         "boxes stay, standing in for the unannotated walls and poles that stay in nuScenes' ±5 cloud. `_allbox` "
         "rows drop every box from the neighbour sweeps, as mini does with annotated objects.", "",
         "Member sets per view: **GT-correct** = in front, in the image, visible by `gt_visible` (true false-visible "
         "rate 0); **GT-wrong** = in front, in the image, occluded in GT (the known-wrong edges a rule must catch); "
         "**builder** = members behind the default builder's kept PROJ_P2H edges, true rate from GT.", "",
         f"Decision fixed before the run: the M1 rule ({r0}) is miscalibrated if it flags ≥ "
         f"{r['decision']['miscalibrated_at']:.0%} of GT-correct members at ±5 on the calibration seeds "
         f"({len(r['calib_seeds'])} seeds). Every table below uses the {len(r['report_seeds'])} report seeds, never "
         "used for any choice.", "", f"> {r['amendment']}", "",
         f"**Verdict:** {r0} flags {_p(sel['r0_error_pm5'])} of GT-correct members at ±5 on the calibration seeds -> "
         + (f"miscalibrated; corrected rule: **{chosen}** = {r['rules'][chosen]}." if chosen else
            "not miscalibrated by the pre-set threshold; no corrected rule is proposed."), "",
         "## Main table (report seeds)", "",
         f"True false-visible rate of the builder's edges (GT): **{_p(true_fv)}** ({sum(n['used_true_fv']['depth'])} of "
         f"{sum(n['used']['depth'])} members).", "",
         "| row | rule | members | keyframe | ±1 | ±5 |", "|---|---|---|---|---|---|"]
    rows = [("(1)", r0, "GT-correct (rate = rule error)", "correct"),
            ("(2)", r0, f"builder edges (true {_p(true_fv)})", "used")]
    if chosen:
        rows += [("(3)", chosen, "GT-correct (rate = rule error)", "correct"),
                 ("(4)", chosen, f"builder edges (true {_p(true_fv)})", "used")]
    for tag, name, label, split in rows:
        L.append(f"| {tag} | {name} | {label} | "
                 + " | ".join(_p(_rate(rep["flag"][name][f"pm{k}"], n, split)) for k in SIDES) + " |")
    L += ["", f"Recall on GT-wrong members ({sum(n['wrong']['depth'])} members; an occluder without a LiDAR return "
          "cannot be found, so recall has a ceiling):", "",
          "| rule | keyframe | ±1 | ±5 | ±1 all boxes dropped | ±5 all boxes dropped |", "|---|---|---|---|---|---|"]
    for name in rules:
        f = rep["flag"][name]
        L.append(f"| {name} | " + " | ".join(_p(_rate(f[k], n, "wrong")) for k in REFS) + " |")
    L += ["", "## Rule grid at ±5 (calibration seeds: the selection)", "",
          "| rule | cell / tol / plane | error on GT-correct | recall on GT-wrong | Youden J |", "|---|---|---|---|---|"]
    for k, spec in r["rules"].items():
        tol = f"{spec['tol_m']:g} m" + (f" + {spec['tol_rel']:.0%}·z" if spec["tol_rel"] else "")
        plane = f"τ {spec['plane_tau']:g} m" if spec["plane_tau"] is not None else "off"
        L.append(f"| {k}{' ←' if k == chosen else ''} | {spec['cell_px']} px / {tol} / {plane} | {_p(sel['error_pm5'][k])} "
                 f"| {_p(sel['recall_pm5'][k])} | {sel['youden_pm5'][k]:+.3f} |")
    L += ["", "## Where the rule errs on GT-correct members (report seeds, ±5)", ""]
    for label, edges, title in (("depth", DEPTH_BINS_M, "member depth [m]"), ("inc", INC_BINS, "|cos incidence| (GT normal)"),
                                ("boundary", None, "4 px cell holds one GT surface / several (or sky)")):
        heads = _bins(edges) if edges else ["one surface", "boundary"]
        L += [f"By {title}:", "", "| rule | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads),
              "| members | " + " | ".join(str(x) for x in n["correct"][label]) + " |"]
        for name in rules:
            L.append(f"| {name} | " + " | ".join(_p(_rate(rep["flag"][name]["pm5"], n, "correct", label, b))
                                               for b in range(len(heads))) + " |")
        L.append("")
    heads = _bins(GAP_BINS_M)
    L += ["Recall on GT-wrong members by how far they sit behind the GT surface [m] (±5):", "",
          "| rule | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads),
          "| members | " + " | ".join(str(x) for x in n["wrong"]["gap"]) + " |"]
    for name in rules:
        L.append(f"| {name} | " + " | ".join(_p(_rate(rep["flag"][name]["pm5"], n, "wrong", "gap", b))
                                           for b in range(len(heads))) + " |")
    L += ["", "Caveats: the synthetic world is a flat ground plane with boxes, much flatter than real streets, so it shows "
          "the mechanism, not mini's magnitude; a plane-aware rule is favoured on a perfect plane. The mini re-measure "
          "is the check on real geometry.", ""]
    return "\n".join(L)

if __name__ == "__main__":
    main()
