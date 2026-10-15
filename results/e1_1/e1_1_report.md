# E1.1 nuScenes data check (v1.0-mini)

dataroot `/home/user02/edtech/antoken/data/datasets/nuscenes`; wall 356 s; headnode load average at start 129, 109, 107 (timings are inflated by a shared, loaded headnode).

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
| m3 | 0 | 24 | 5.6 | 0.232 |
| m3 | 2 | 24 | 3.4 | 0.141 |
| m2 | 0 | 24 | 3.7 | 0.152 |
| m2 | 2 | 24 | 2.2 | 0.091 |

Peak RSS (getrusage, same counter as `/usr/bin/time -v`): main process 866 MB, largest joined DataLoader worker 665 MB. The `/usr/bin/time -v` log of this run, if wrapped, sits beside this report as `time_v.log`.

## Graph edges (T27, gate M1)

Every m3 sample of mini_train and mini_val: hold-out (frac 0.3) -> `build_hubs` (K=256, M=32) -> `build_graph` with the default `GraphConfig` (patch 16, z-buffer cell 4 px, tol 0.5 m, caps hub 32 / patch 8). Graph passes use 8 DataLoader workers.

| M1 criterion | measured | threshold | verdict |
|---|---|---|---|
| projection edges re-project into their patch | 974852/974852 = 100.00%; H2P = reverse of P2H in every sample: True | 100% | **PASS** |
| false-visible vs LiDAR merged over ±5 sweeps (`pm5`, graph's cell rule: upper bound) | 925455/3018725 = 30.66% | < 5% | **FAIL** (upper bound, cell rule) |
| data_wait_frac | not measured (needs T25) | < 0.10 | pending |

### Projection re-check

An edge passes when at least one member of its hub, lifted to world in float64 and projected with `project` (not the builder's float32 `visible_members` path), lands in the edge's patch in front of the camera (spec T11, test 1). Visibility is judged by the false-visible check below, not here.

| split | P2H edges | passed | failed | hub centre in the edge's patch | centre distance [patches] median / p95 |
|---|---|---|---|---|---|
| mini_train | 776675 | 776675 | 0 | 10.52% | 2 / 5 |
| mini_val | 198177 | 198177 | 0 | 10.62% | 2 / 5 |

The hub-centre column is information only: the task brief phrased the check as "the hub centre re-projects into the connected patch", but edges are built from members (spec T11), so a hub whose 32 members span several patches links to all of them and its centre lies in only one. The gate uses the spec's member definition. The centre distance is Chebyshev in patch units (capped at 64).

### False-visible rate

Members counted: one per hub member slot, visible to the builder's z-buffer and inside the patch of a kept (post-cap) PROJ_P2H edge, i.e. exactly the members behind the edge weights. A member is false-visible when `zbuffer_visibility` with the same cell and tol rejects it against a reference cloud:

- `train`: the graph's own reference (input points of the keyframe sweep). Must be 0; checks the counting.
- `pm0`: every point of the keyframe sweep, held-out included (effect of hold-out sparsity alone).
- `pm5` (gate): `pm0` plus up to ±5 neighbouring LIDAR_TOP sweeps inside the scene, each placed with its own ego pose; points with |x|, |y| < 1 m in the sensor frame (ego body, as devkit `remove_close`) and points inside the annotation boxes interpolated at the sweep's time, dilated 0.3 m (moving objects smear across sweeps), are dropped from the neighbours. Same cell and tol as the graph, so grazing ground at range can be rejected by a denser cloud of the same surface: this rate is an upper bound. The synthetic test of T11 (`gt_visible`) excludes grazing on purpose (farthest of the 4 surrounding GT depths); no such definition is applied to real data here.
- Attribution rows, not gates: `pm1` = ±1 sweep (±50 ms, so pose drift is negligible: a jump from `pm0` to `pm1` comes from point density alone; growth from `pm1` to `pm5` mixes density with registration over time and does not separate them); `pm5_ego3` = ego exclusion widened to |x|, |y| < 3 m (a collapse on CAM_FRONT / CAM_BACK means ghosts of the ego body from neighbour sweeps); `pm5_noground` = neighbour points below world z 0.3 m dropped (a collapse in the far bins means grazing ground).

| split | reference | used members | false-visible | rate | [0, 10) | [10, 20) | [20, 40) | [40, inf) |
|---|---|---|---|---|---|---|---|---|
| mini_train | train | 2395904 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mini_train | pm0 | 2395904 | 105794 | 4.42% | 0.64% | 1.63% | 4.12% | 6.86% |
| mini_train | pm1 | 2395904 | 382229 | 15.95% | 3.36% | 9.30% | 15.83% | 21.78% |
| mini_train | pm5 | 2395904 | 727237 | 30.35% | 9.00% | 24.21% | 29.49% | 38.27% |
| mini_train | pm5_ego3 | 2395904 | 709119 | 29.60% | 7.75% | 23.25% | 28.91% | 37.51% |
| mini_train | pm5_noground | 2395904 | 621225 | 25.93% | 6.05% | 13.50% | 24.75% | 37.20% |
| mini_val | train | 622821 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| mini_val | pm0 | 622821 | 23327 | 3.75% | 0.23% | 1.19% | 4.00% | 5.32% |
| mini_val | pm1 | 622821 | 88634 | 14.23% | 2.14% | 7.36% | 15.76% | 18.07% |
| mini_val | pm5 | 622821 | 198218 | 31.83% | 10.56% | 28.11% | 33.84% | 34.97% |
| mini_val | pm5_ego3 | 622821 | 195392 | 31.37% | 8.48% | 27.50% | 33.39% | 34.83% |
| mini_val | pm5_noground | 622821 | 163680 | 26.28% | 4.59% | 14.40% | 27.65% | 34.27% |

Per camera, false-visible rate per reference (both splits):

| camera | used | pm0 | pm1 | pm5 | pm5_ego3 | pm5_noground |
|---|---|---|---|---|---|---|
| CAM_FRONT | 655058 | 3.72% | 15.70% | 32.75% | 31.09% | 25.66% |
| CAM_FRONT_RIGHT | 543238 | 3.32% | 13.77% | 27.28% | 27.22% | 22.83% |
| CAM_BACK_RIGHT | 486404 | 3.54% | 13.33% | 25.94% | 25.75% | 22.71% |
| CAM_BACK | 734408 | 6.24% | 20.10% | 38.11% | 36.94% | 32.26% |
| CAM_BACK_LEFT | 252444 | 3.88% | 13.94% | 27.45% | 27.42% | 25.98% |
| CAM_FRONT_LEFT | 347173 | 3.99% | 13.10% | 25.16% | 25.12% | 23.00% |

- mini_train: neighbour sweeps per sample 5–10 (median 10); reference points per sample median: train 24304, pm0 34720, pm1 82606, pm5 273452, pm5_ego3 260577, pm5_noground 134685; dropped from neighbours: 27680260 ego-body, 9924706 in boxes.
- mini_val: neighbour sweeps per sample 5–10 (median 10); reference points per sample median: train 24304, pm0 34720, pm1 81183, pm5 266331, pm5_ego3 260846, pm5_noground 127806; dropped from neighbours: 6870304 ego-body, 2574932 in boxes.

### Degree and provenance

Summed over samples (edges, provenance), mean over samples of the mean in-degree over eligible receivers, max in-degree over all nodes. m3 has T = 1, so no TRACK edges.

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

Caps: SPATIAL max 24 vs cap 24 (ok); PROJ_P2H max 32 vs cap 32 (ok); PROJ_H2P max 8 vs cap 8 (ok); NB3D max 16 vs cap 16 (ok).

### Prepare time (mini_train m3)

Loader delivering (sample, hubs, graph) with 8 workers (1 torch thread each) over 160 samples: wall 17.5 s = **0.109 s/sample**. Inside a worker: load (6 JPEGs + sweep) median 0.3279 s (p95 0.8833), hold-out + hubs + graph median 0.2358 s (p95 0.9241). Headnode load average before this pass: 136, 117, 110 on a shared 8-core headnode, so these are upper bounds for a dedicated node.

- mini_train check pass (142 s wall incl. checks): in-worker load median 0.2448 s, prepare median 0.2172 s.
- mini_val check pass (24 s wall incl. checks): in-worker load median 0.1746 s, prepare median 0.225 s.

## Pending

- **trainval m3 check: pending.** Keyframes for all 850 scenes are on DGX, but loading the full trainval JSON on the headnode needs several GB of RAM; run it as a Slurm CPU job or after a streaming reader exists.
- **M1 `data_wait_frac` < 0.10: pending (needs the T25 trainer).** The prepare time above is a loader-side number only; the GPU-side wait fraction is not measured here.
