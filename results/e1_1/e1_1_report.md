# E1.1 nuScenes data check (v1.0-mini)

dataroot `/home/user02/edtech/antoken/data/datasets/nuscenes`; wall 661 s; headnode load average at start 46, 102, 113 (timings are inflated by a shared, loaded headnode).

## Sample counts

| split | m3 samples | m3 expected (sum of scene.nbr_samples) | m2 windows (clip 8, stride 1) | m2 expected (sum of chain length - 7) |
|---|---|---|---|---|
| mini_train | 323 | 323 | 1817 | 1817 |
| mini_val | 81 | 81 | 455 | 455 |

Rows in sample.json: 404; m3 total over both splits: 404.

## Timestamps

| split | quantity | n | median | p95 | max |
|---|---|---|---|---|---|
| mini_train | m3 max over 6 cams of abs(t_cam - t_lidar) [s] | 323 | 0.04313 | 0.04583 | 0.04835 |
| mini_train | m2 frame spacing [s] | 3213 | 0.1 | 0.1 | 0.2 |
| mini_train | m2 nearest-sweep abs(dt) [s] | 3672 | 0.0145 | 0.01566 | 0.039 |
| mini_val | m3 max over 6 cams of abs(t_cam - t_lidar) [s] | 81 | 0.04289 | 0.04412 | 0.04537 |
| mini_val | m2 frame spacing [s] | 805 | 0.1 | 0.1 | 0.2 |
| mini_val | m2 nearest-sweep abs(dt) [s] | 920 | 0.01474 | 0.01591 | 0.03519 |

- mini_train m2 (checked 459 windows with start % 4 == 0, which cover every consecutive frame pair): min spacing 0.04868 s; windows with a gap > 1.5 frames (0.125 s): 8.1%; windows where two frames share one nearest sweep: 0.4%.
- mini_val m2 (checked 115 windows with start % 4 == 0, which cover every consecutive frame pair): min spacing 0.04998 s; windows with a gap > 1.5 frames (0.125 s): 8.7%; windows where two frames share one nearest sweep: 0.0%.

Per camera (mini_train m3): median t_cam - t_lidar and LiDAR points landing in front of / inside the 448x256 crop.

| camera | median dt [s] | front | inside |
|---|---|---|---|
| CAM_FRONT | -0.03551 | 0.341 | 0.085 |
| CAM_FRONT_RIGHT | -0.02761 | 0.355 | 0.091 |
| CAM_BACK_RIGHT | -0.02004 | 0.362 | 0.103 |
| CAM_BACK | -0.01043 | 0.328 | 0.127 |
| CAM_BACK_LEFT | -0.000548 | 0.387 | 0.116 |
| CAM_FRONT_LEFT | -0.04313 | 0.372 | 0.095 |

## Geometry

| split | mode | check | max |
|---|---|---|---|
| mini_train | m3 | cam_to_world.bottom_row_err | 0 |
| mini_train | m3 | cam_to_world.det_err | 9.14e-08 |
| mini_train | m3 | cam_to_world.orth_err | 8.07e-08 |
| mini_train | m3 | intrinsics.max_abs_diff | 1.39e-05 |
| mini_train | m3 | lidar_to_world.bottom_row_err | 0 |
| mini_train | m3 | lidar_to_world.det_err | 4.41e-08 |
| mini_train | m3 | lidar_to_world.orth_err | 4.95e-08 |
| mini_train | m3 | recentre.ref_lidar_ego_minus_I | 8.27e-08 |
| mini_train | m3 | roundtrip.max_err_m | 2.51e-05 |
| mini_train | m2 | cam_to_world.bottom_row_err | 0 |
| mini_train | m2 | cam_to_world.det_err | 8.4e-08 |
| mini_train | m2 | cam_to_world.orth_err | 6.02e-08 |
| mini_train | m2 | intrinsics.max_abs_diff | 1.39e-05 |
| mini_train | m2 | lidar_to_world.bottom_row_err | 0 |
| mini_train | m2 | lidar_to_world.det_err | 8.48e-08 |
| mini_train | m2 | lidar_to_world.orth_err | 5.97e-08 |
| mini_train | m2 | recentre.ref_lidar_ego_minus_I | 8.27e-08 |
| mini_val | m3 | cam_to_world.bottom_row_err | 0 |
| mini_val | m3 | cam_to_world.det_err | 7.65e-08 |
| mini_val | m3 | cam_to_world.orth_err | 8.04e-08 |
| mini_val | m3 | intrinsics.max_abs_diff | 1.39e-05 |
| mini_val | m3 | lidar_to_world.bottom_row_err | 0 |
| mini_val | m3 | lidar_to_world.det_err | 4.41e-08 |
| mini_val | m3 | lidar_to_world.orth_err | 4.95e-08 |
| mini_val | m3 | recentre.ref_lidar_ego_minus_I | 8.27e-08 |
| mini_val | m3 | roundtrip.max_err_m | 2.81e-05 |
| mini_val | m2 | cam_to_world.bottom_row_err | 0 |
| mini_val | m2 | cam_to_world.det_err | 8.15e-08 |
| mini_val | m2 | cam_to_world.orth_err | 6.06e-08 |
| mini_val | m2 | intrinsics.max_abs_diff | 1.39e-05 |
| mini_val | m2 | lidar_to_world.bottom_row_err | 0 |
| mini_val | m2 | lidar_to_world.det_err | 8.55e-08 |
| mini_val | m2 | lidar_to_world.orth_err | 5.96e-08 |
| mini_val | m2 | recentre.ref_lidar_ego_minus_I | 8.27e-08 |

- mini_train m3: fraction of a sweep's points inside at least one camera: median 0.581 (min 0.498, max 0.625); project->unproject median error per camera 9.611e-07 m (float32, only measures numeric error).

- mini_val m3: fraction of a sweep's points inside at least one camera: median 0.589 (min 0.421, max 0.630); project->unproject median error per camera 9.83e-07 m (float32, only measures numeric error).

Independent check against the devkit's map_pointcloud_to_image chain (pyquaternion, view_points, float64) on 8 mini_train keyframes x 6 cameras, 174306 points with depth > 1 m: max |uv diff| = 0.0282 px, max |depth diff| = 0.000168 m.

## Held-out stability

- mini_train: 1806 (window, window) pairs sharing a lidar_key compared (overlapping checked windows and repeats inside one window): 0 mismatches.
- mini_val: 452 (window, window) pairs sharing a lidar_key compared (overlapping checked windows and repeats inside one window): 0 mismatches.

## Load time and memory

| mode | workers | n | wall [s] | s/sample |
|---|---|---|---|---|
| m3 | 0 | 24 | 1.8 | 0.074 |
| m3 | 2 | 24 | 1.3 | 0.056 |
| m2 | 0 | 24 | 6.2 | 0.260 |
| m2 | 2 | 24 | 3.3 | 0.138 |

Peak RSS (getrusage, same counter as `/usr/bin/time -v`): main process 871 MB, largest joined DataLoader worker 664 MB. The `/usr/bin/time -v` log of this run, if wrapped, sits beside this report as `time_v.log`.

## Graph edges (T27, gate M1)

Every m3 sample of mini_train and mini_val: hold-out (frac 0.3) -> `build_hubs` (K=256, M=32, shared by all variants) -> `build_graph` per builder variant. The builders read only the keyframe sweep's input points (held-out excluded): no sweeps, so they run unchanged on trainval keyframes. The ±5 sweeps enter only the metric. Graph passes use 8 DataLoader workers.

False-visible metric, the same for every row (M1 rule): a member behind a kept PROJ_P2H edge is false-visible when `zbuffer_visibility` with cell 4 px and tol 0.5 m rejects it against LiDAR merged over ±5 sweeps (`pm5`, gate) or against the keyframe sweep with held-out points restored (`pm0`). `own rule` re-runs each variant's own z-buffer on its own reference and must be 0 (counting check).

Corrected rule (metric revision 2026-10-15, calibrated on synthetic GT, `fv_calibration.md`): cell 4 px, tol 0.5 m + 5% of depth, reference points within 0.2 m of the member's hub plane ignored. Both rules are reported; the M1 rule's numbers are kept.

### Builder variants

| variant | z-buffer cell / tol | grazing filter | FV ±5 sweeps | FV keyframe sweep | FV ±5, corrected rule | own rule | re-check | P2H = H2P edges | used members | mean in-degree P2H per hub / H2P per patch | prepare median [s] | M1 re-check / FV (M1 rule) / FV (corrected) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 4 px / 0.5 m | off | **30.66%** | 4.28% | **15.83%** | 0 | 100.00% | 974852 | 3018725 | 9.47 / 0.90 | 0.237 | **PASS** / **FAIL** / **FAIL** |
| grazing | 4 px / 0.5 m | cos ≥ 0.2 | **27.64%** | 4.18% | **14.98%** | 0 | 100.00% | 622252 | 1947346 | 6.15 / 0.59 | 0.238 | **PASS** / **FAIL** / **FAIL** |
| reltol | 4 px / 5% of depth | off | **33.81%** | 8.62% | **15.84%** | 0 | 100.00% | 979200 | 3160699 | 9.52 / 0.91 | 0.242 | **PASS** / **FAIL** / **FAIL** |
| cell2 | 2 px / 0.5 m | off | **37.05%** | 13.08% | **19.82%** | 0 | 100.00% | 1010816 | 3319111 | 9.81 / 0.93 | 0.244 | **PASS** / **FAIL** / **FAIL** |
| grazing+reltol | 4 px / 5% of depth | cos ≥ 0.2 | **31.13%** | 8.79% | **14.96%** | 0 | 100.00% | 625497 | 2045492 | 6.18 / 0.59 | 0.231 | **PASS** / **FAIL** / **FAIL** |
| all | 2 px / 5% of depth | cos ≥ 0.2 | **34.51%** | 13.27% | **18.83%** | 0 | 100.00% | 646096 | 2150540 | 6.37 / 0.61 | 0.233 | **PASS** / **FAIL** / **FAIL** |

Settings, each chosen once (no sweep): grazing filter cos ≥ 0.2 (incidence ≤ ~78°; with a 5% tolerance and fx / cell ≈ 90 px a surface stays inside the tolerance up to tan(incidence) ≈ 4.5); relative tolerance 5% of depth replacing the fixed 0.5 m (equal to it at 10 m: stricter nearer, looser farther); cell 2 px against 4 px. Prepare = hold-out + hubs + graph inside a worker, all variants timed in the same item so they share the headnode's load.

Grazing filter on the baseline graph: it would drop 390496 of 974852 P2H edges (40.06%); 6.39% of the dropped and 1.41% of the kept come from near-collinear hubs (λ_mid / λ_max < 0.05), whose PCA normal, and so cos_incidence, is ill-defined.

False-visible (`pm5`, M1 rule) per member-depth bin, both splits:

| variant | [0, 10) | [10, 20) | [20, 40) | [40, inf) |
|---|---|---|---|---|
| baseline | 9.29% (187191) | 25.05% (568040) | 30.34% (1167558) | 37.55% (1095936) |
| grazing | 6.60% (119317) | 19.03% (315249) | 26.53% (796615) | 36.18% (716165) |
| reltol | 9.27% (186966) | 25.48% (571285) | 33.18% (1216472) | 42.33% (1185976) |
| cell2 | 10.47% (189516) | 27.80% (589179) | 36.63% (1280033) | 45.81% (1260383) |
| grazing+reltol | 6.58% (119177) | 19.45% (316885) | 29.51% (830127) | 41.37% (779303) |
| all | 7.73% (120746) | 21.82% (326466) | 33.05% (873738) | 44.94% (829590) |

False-visible (`pm5`, M1 rule) per camera, both splits:

| camera | baseline | grazing | reltol | cell2 | grazing+reltol | all |
|---|---|---|---|---|---|---|
| CAM_FRONT | 32.75% | 29.96% | 35.77% | 38.10% | 33.41% | 35.68% |
| CAM_FRONT_RIGHT | 27.28% | 25.61% | 29.44% | 32.61% | 27.90% | 30.98% |
| CAM_BACK_RIGHT | 25.94% | 23.20% | 28.11% | 31.65% | 25.45% | 28.73% |
| CAM_BACK | 38.11% | 34.39% | 43.09% | 46.24% | 39.98% | 44.21% |
| CAM_BACK_LEFT | 27.45% | 22.71% | 29.07% | 33.70% | 24.24% | 28.14% |
| CAM_FRONT_LEFT | 25.16% | 22.79% | 27.64% | 31.46% | 25.85% | 28.96% |

- mini_train: neighbour sweeps per sample 5–10 (median 10); reference points per sample median: train 24304, pm0 34720, pm1 82606, pm5 273452; dropped from neighbours: 27680260 ego-body, 9924706 in boxes (|x|, |y| < 1 m in the sensor frame; boxes interpolated at each sweep's time, dilated 0.3 m).
- mini_val: neighbour sweeps per sample 5–10 (median 10); reference points per sample median: train 24304, pm0 34720, pm1 81183, pm5 266331; dropped from neighbours: 6870304 ego-body, 2574932 in boxes (|x|, |y| < 1 m in the sensor frame; boxes interpolated at each sweep's time, dilated 0.3 m).

### Default GraphConfig (`baseline`): patch 16, cell 4 px, tol 0.5 m, caps hub 32 / patch 8

| M1 criterion | measured | threshold | verdict |
|---|---|---|---|
| projection edges re-project into their patch | 974852/974852 = 100.00%; H2P = reverse of P2H in every sample: True | 100% | **PASS** |
| false-visible vs LiDAR merged over ±5 sweeps, M1 rule as first measured | 925455/3018725 = 30.66% | < 5% | **FAIL** |
| false-visible vs LiDAR merged over ±5 sweeps, corrected rule (revision 2026-10-15) | 477719/3018725 = 15.83% | < 5% | **FAIL** |
| data_wait_frac | not measured (needs T25) | < 0.10 | pending |

False-visible of the default builder's edges against reference density:

| rule | keyframe sweep | ±1 sweep | ±5 sweeps |
|---|---|---|---|
| M1 rule | 4.28% | 15.60% | 30.66% |
| corrected rule | 2.07% | 7.21% | 15.83% |

Projection re-check: an edge passes when at least one member of its hub, lifted to world in float64 and projected with `project` (not the builder's float32 path), lands in the edge's patch in front of the camera (spec T11, test 1).

| split | P2H edges | passed | failed | hub centre in the edge's patch | centre distance [patches] median / p95 |
|---|---|---|---|---|---|
| mini_train | 776675 | 776675 | 0 | 10.52% | 2 / 5 |
| mini_val | 198177 | 198177 | 0 | 10.62% | 2 / 5 |

The hub-centre column is information only: edges are built from members (spec T11), so a hub whose 32 members span several patches links to all of them and its centre lies in only one. Chebyshev distance in patch units, capped at 64.

Degree and provenance: edges and provenance summed over samples, mean in-degree over eligible receivers averaged over samples, max in-degree over all nodes. m3 has T = 1, so no TRACK edges.

mini_train (323 samples):

| type | edges | mean in-degree | max in-degree | provenance |
|---|---|---|---|---|
| SPATIAL | 18348984 | 21.13 | 24 | GRID=18348984 |
| TRACK | 0 | 0.00 | 0 | - |
| PROJ_P2H | 776675 | 9.39 | 32 | CALIB_LIDAR=776675 |
| PROJ_H2P | 776675 | 0.89 | 8 | CALIB_LIDAR=776675 |
| NB3D | 1323008 | 16.00 | 16 | CALIB_LIDAR=1323008 |

mini_val (81 samples):

| type | edges | mean in-degree | max in-degree | provenance |
|---|---|---|---|---|
| SPATIAL | 4601448 | 21.13 | 24 | GRID=4601448 |
| TRACK | 0 | 0.00 | 0 | - |
| PROJ_P2H | 198177 | 9.56 | 32 | CALIB_LIDAR=198177 |
| PROJ_H2P | 198177 | 0.91 | 8 | CALIB_LIDAR=198177 |
| NB3D | 331776 | 16.00 | 16 | CALIB_LIDAR=331776 |

Caps, max over all variants: SPATIAL max 24 vs cap 24 (ok); PROJ_P2H max 32 vs cap 32 (ok); PROJ_H2P max 8 vs cap 8 (ok); NB3D max 16 vs cap 16 (ok).

### Prepare time (mini_train m3, default GraphConfig)

Loader delivering (sample, hubs, graph) with 8 workers (1 torch thread each) over 160 samples: wall 10.1 s = **0.063 s/sample**. Inside a worker: load (6 JPEGs + sweep) median 0.1861 s (p95 0.263), hold-out + hubs + graph median 0.2124 s (p95 0.3397). Headnode load average before this pass: 32, 93, 110 on a shared 8-core headnode, so these are upper bounds for a dedicated node. Per-variant prepare medians are in the variant table (measured inside the check pass, which also runs the checks).

## Pending

- **trainval m3 check: pending.** Keyframes for all 850 scenes are on DGX, but loading the full trainval JSON on the headnode needs several GB of RAM; run it as a Slurm CPU job or after a streaming reader exists.
- **M1 `data_wait_frac` < 0.10: pending (needs the T25 trainer).** The prepare time above is a loader-side number only; the GPU-side wait fraction is not measured here.
