# Calibration of the M1 false-visible rule on synthetic GT

Synthetic scenes shaped like nuScenes m3: 6 cameras 448x256, fov 63.8° (fx ≈ 360 px), pitch 0°, 32-beam LiDAR (-30...10°, 1084 steps) at 20 Hz, ego 8 m/s, 8 boxes (1 moving) on a ground plane. Same pipeline as mini: hold-out 0.3 -> hubs (K=256, M=32) -> default graph.

References: every point of the keyframe sweep (held-out included), plus the sweeps rendered at ±1 or ±5 x 50 ms from the same scene minus their points inside moving boxes at their own time (dilated 0.3 m); static boxes stay, standing in for the unannotated walls and poles that stay in nuScenes' ±5 cloud. `_allbox` rows drop every box from the neighbour sweeps, as mini does with annotated objects.

Member sets per view: **GT-correct** = in front, in the image, visible by `gt_visible` (true false-visible rate 0); **GT-wrong** = in front, in the image, occluded in GT (the known-wrong edges a rule must catch); **builder** = members behind the default builder's kept PROJ_P2H edges, true rate from GT.

Decision fixed before the run: the M1 rule (c4) is miscalibrated if it flags ≥ 5% of GT-correct members at ±5 on the calibration seeds (16 seeds). Every table below uses the 16 report seeds, never used for any choice.

> Selection amended on 2026-10-15 after seeing calibration seed 1 and before any report seed. The pre-set constraint (recall at ±5 at most 5 points below R0's) is degenerate: R0 at ±5 flags most GT-correct members, so its recall there carries little information and only rules that over-flag as much can meet it. The rule is now the one with the highest Youden J = recall − error at ±5 on the calibration seeds (ties within 0.5 point go to fewer changes). Also amended at the same time: neighbour sweeps keep static boxes as structure (the first version dropped every box, leaving the ±5 sweeps without any occluder), and 16 + 16 seeds instead of 4 + 4 (GT-wrong members are scarce).

**Verdict:** c4 flags 58.39% of GT-correct members at ±5 on the calibration seeds -> miscalibrated; corrected rule: **c4+rel+plane** = {'cell_px': 4, 'tol_m': 0.5, 'tol_rel': 0.05, 'plane_tau': 0.2}.

## Main table (report seeds)

True false-visible rate of the builder's edges (GT): **1.27%** (1630 of 128428 members).

| row | rule | members | keyframe | ±1 | ±5 |
|---|---|---|---|---|---|
| (1) | c4 | GT-correct (rate = rule error) | 0.29% | 1.26% | 59.18% |
| (2) | c4 | builder edges (true 1.27%) | 0.21% | 1.36% | 59.49% |
| (3) | c4+rel+plane | GT-correct (rate = rule error) | 0.22% | 0.36% | 0.69% |
| (4) | c4+rel+plane | builder edges (true 1.27%) | 0.18% | 0.49% | 1.49% |

Recall on GT-wrong members (2171 members; an occluder without a LiDAR return cannot be found, so recall has a ceiling):

| rule | keyframe | ±1 | ±5 | ±1 all boxes dropped | ±5 all boxes dropped |
|---|---|---|---|---|---|
| c4 | 32.15% | 42.47% | 88.53% | 32.43% | 78.21% |
| c4+rel+plane | 31.87% | 41.96% | 82.17% | 31.87% | 31.87% |

## Rule grid at ±5 (calibration seeds: the selection)

| rule | cell / tol / plane | error on GT-correct | recall on GT-wrong | Youden J |
|---|---|---|---|---|
| c4 | 4 px / 0.5 m / off | 58.39% | 88.97% | +0.306 |
| c4+plane | 4 px / 0.5 m / τ 0.2 m | 1.67% | 64.95% | +0.633 |
| c4+rel | 4 px / 0.5 m + 5%·z / off | 1.89% | 64.95% | +0.631 |
| c4+rel+plane ← | 4 px / 0.5 m + 5%·z / τ 0.2 m | 0.64% | 64.46% | +0.638 |
| c2 | 2 px / 0.5 m / off | 41.83% | 74.51% | +0.327 |
| c2+plane | 2 px / 0.5 m / τ 0.2 m | 0.36% | 43.97% | +0.436 |
| c2+rel | 2 px / 0.5 m + 5%·z / off | 0.18% | 43.59% | +0.434 |
| c2+rel+plane | 2 px / 0.5 m + 5%·z / τ 0.2 m | 0.17% | 43.21% | +0.430 |

## Where the rule errs on GT-correct members (report seeds, ±5)

By member depth [m]:

| rule | [0, 10) | [10, 20) | [20, 40) | [40, inf) |
|---|---|---|---|---|
| members | 15439 | 36607 | 39817 | 38237 |
| c4 | 1.20% | 49.74% | 69.14% | 81.24% |
| c4+rel+plane | 0.04% | 0.89% | 0.72% | 0.71% |

By |cos incidence| (GT normal):

| rule | [0, 0.1) | [0.1, 0.3) | [0.3, 0.6) | [0.6, inf) |
|---|---|---|---|---|
| members | 90828 | 32129 | 1907 | 5236 |
| c4 | 74.04% | 30.04% | 1.31% | 1.07% |
| c4+rel+plane | 0.83% | 0.33% | 0.26% | 0.57% |

By 4 px cell holds one GT surface / several (or sky):

| rule | one surface | boundary |
|---|---|---|
| members | 127943 | 2157 |
| c4 | 59.45% | 42.93% |
| c4+rel+plane | 0.15% | 32.45% |

Recall on GT-wrong members by how far they sit behind the GT surface [m] (±5):

| rule | [0, 1) | [1, 5) | [5, inf) |
|---|---|---|---|
| members | 6 | 77 | 2088 |
| c4 | 100.00% | 59.74% | 89.56% |
| c4+rel+plane | 0.00% | 37.66% | 84.05% |

Caveats: the synthetic world is a flat ground plane with boxes, much flatter than real streets, so it shows the mechanism, not mini's magnitude; a plane-aware rule is favoured on a perfect plane. The mini re-measure is the check on real geometry.
