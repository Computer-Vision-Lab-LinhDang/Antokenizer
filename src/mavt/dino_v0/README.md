# MAVT on DINOv3 (`mavt.dino_v0`)

The current MAVT line of work (from 22/10/2026). A tokenizer for images, video and posed 3D object views,
built on a **frozen DINOv3 ViT-L/16**. The older pipelines described in the repo-root `README.md`
(SigLIP2 backbone, slots, v5-v7d) are paused; new work goes here.

Status on 23/10/2026:
- **v0** trained on alpha (38,511 steps) and evaluated. Its main flaw: D is barely used.
- **v1 phase 1** (the fixes below) is training on alpha since 12:26 UTC, 16,000 steps, ETA ~20:15 UTC.
- Code lives on branch `claude/project-thread-wco299` (PR #20). Package name stays `dino_v0` for both
  versions: v1 is a set of config flags whose defaults reproduce v0 exactly.

## The idea in one picture

```
frames ─► FrameEncoder: frozen DINOv3 + 256-ch pixel stem ─► 1024-d token per 16x16 patch (16x16 grid at 256²)
       ─► Trunk: 8 layers; a token reads its whole frame + the 3x3 cells around it at frame t-1 (masked axial RoPE on t,u,v,z)
       ─► C_k: 256 cells, built at each keyframe (every K = 4 frames) from X_k ONLY
          D_t: 16 tokens per frame = how frame t deviates from C_k (predict from D_{t-1}, then correct with X_t)
       ─► latents: z_r (Gaussian + KL) of C and D; z_s = attention pool of [C ; D] (understanding vector)
       ─► QueryDecoder: one query per patch reads C at (u,v) + Δ(D_t) in a 5x5 window, reads D by content,
          mixes them with a gate o; outputs rgb + a DINOv3 feature per query
```

Rules that define the C/D split (do not break them):
- C never reads D and never reads frames other than its keyframe. D may read C, X_t and D.
- Images have only C. Video and posed 3D views have C + D (3D views pass camera pose to D as an "action").
- Latent budget is fixed: 12,288 numbers per image and per 4-frame segment
  (v0: 256x40 + 4x16x32; v1: 256x32 + 4x16x64; images 256x48).

## Files

| File | What it holds |
|---|---|
| `config.py` | `DinoV0Config`: every hyper-parameter; v1 flags in the "v1 phase 1" block; `tiny_config()` for CPU tests |
| `encoder.py` | frozen DINOv3 (local HF dir) + pixel stem; DINOv3 patch features are the `feat` targets |
| `rope.py` | masked axial RoPE on (t, u, v, z); cos/sin tables built once per forward |
| `trunk.py` | 8-layer trunk (in-frame + causal t-1 window attention); optional R2 learned relations (`rgat_relations`, off) |
| `cd_state.py` | `CBuilder` (C at keyframes), `DState` / `CDState` (D per frame, forecasts `D_fut[h]`) |
| `latent.py` | z_r heads (KL) and the z_s pool |
| `decoder.py` | `QueryDecoder`: Δ (mean or per-query), dense masked 5x5 window read of C, gate, rgb head (linear or conv), warp-only mode |
| `losses.py` | rgb L1, feat (1-cos + MSE), KL, VICReg var/cov, sem (z_s -> DINOv3 CLS), gate, LPIPS + Gram |
| `model.py` | `MAVTDinoV0`: `encode`, `decode_frame`, `reconstruct`, `forward` (all losses, batched over frames) |
| `data.py` | ImageNet parquet, OpenVid / SSv2 clips (PyAV, random stride), Objaverse posed views, mixed loaders |
| `evaluate.py` | PSNR/SSIM/LPIPS/rFID, C/D split metrics, forecasts vs copy baselines, kNN / linear probes |
| `camera.py` | Plücker rays and relative poses for posed views |

Scripts: `scripts/train_dino_v0.py` (torchrun DDP), `scripts/eval_dino_v0.py`, `scripts/profile_dino_v0.py`.
Configs: `configs/dino_v0/v0_alpha.yaml`, `configs/dino_v0/v1_alpha.yaml`. Tests: `tests/dino_v0/`.

## v0 vs v1 (config flags)

| Flag | v0 (default) | v1 phase 1 | Why |
|---|---|---|---|
| `rgb_head` | `linear` (per patch) | `conv` (full-grid upsampler, ~29M, `conv_channels` 640/384/192/96/64) | v0 images show the patch grid and are blurry |
| `w_lpips`, `w_gram` | 0, 0 | 0.5, 50 | perceptual + texture losses on full frames (`lpips_max_frames` per step) |
| `delta_mode` | `mean` (Δ from the mean of D) | `query` (each query attends to D for its own Δ) | different regions move differently |
| `gate_loss`, `w_gate`, `gate_init_bias` | `mean`, 0.01, 4.0 | `hinge` (floor 0.5), 1.0, 1.0 | v0's `mean(1-o)` pushed the gate to 1, so the decoder ignored D |
| `w_warp` | 0 | 0.5 | decode non-key frames from C at (u,v)+Δ only: Δ must follow the motion |
| `zr_c_video`, `zr_d` | 40, 32 | 32, 64 | more budget for the frames between keyframes |
| `forecast_horizons`, `forecast_residual`, `w_fut_rgb` | (1,), false, 0 | (1, 4), true, 0.5 | forecast the change (starts as copy-last), at t+1 and t+4, with an rgb loss |
| `train_queries_per_frame` | 64 | 256 (all; required by the conv head) | |
| trainer `frame_stride` | 2 | `2,4,8` (drawn per clip) | at stride 2 objects move < 1 patch, so Δ ≈ 0 was the right answer |
| trainer `ssv2_train` | none | SSv2 train split, weight 0.3 | more motion-heavy video; labels never read |
| trainer `mix_modalities` | off (one modality per step) | on (image + video + 3D every step) | one backward; DDP without `find_unused_parameters`, `static_graph` |
| trainer `compile` | off | on | regional `torch.compile`: -15% time, -34% memory on MI355X |

Not in phase 1 (planned, see the v1 doc): a temporal objective for z_s on video, the track edge
C_{k-K} -> C_k, ray-based global reads for novel 3D views, the event memory S (needs long clips, v2).
They wait until dropping D costs at least ~2 dB on OpenVid at stride 4, i.e. until D is really used.

## Commands

```bash
# tests (CPU, ~30 s here; ~20 min on alpha's CPU)
python -m pytest tests/dino_v0 -q

# CPU smoke runs of the trainer (tiny config, 3 steps)
python scripts/train_dino_v0.py --dry-run --out /tmp/dry
python scripts/train_dino_v0.py --dry-run --mix-modalities --out /tmp/dry_mix

# train v1 on alpha (GPUs 0-3 only); the yaml `train:` block holds the run's settings, CLI flags win
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 scripts/train_dino_v0.py \
    --config configs/dino_v0/v1_alpha.yaml --out /scratch/vunguyen13/mavt/runs/muse-spark-mavt-v1
# re-running the same command resumes from <out>/last.pt; SIGTERM / Ctrl-C saves last.pt

# evaluate a checkpoint (one GPU); writes <out>/eval.json + eval.md
PYTHONPATH=src python scripts/eval_dino_v0.py --ckpt <run>/last.pt --device cuda --amp \
    --imagenet-root /scratch/vunguyen13/data/imagenet-1k \
    --openvid-manifest /scratch/vunguyen13/data/dino_v0_data/openvid1m --video-limit 500 --video-frames 16 \
    --frame-stride 4 --stem eval_s4 --out <run>/eval

# profile one GPU (step time, memory, top kernels); --modality image|video|threed|mixed
PYTHONPATH=src python scripts/profile_dino_v0.py --config configs/dino_v0/v1_alpha.yaml --modality mixed \
    --batch-image 128 --batch-video 16 --batch-3d 16 --compile
```

Training logs (`<out>/log.jsonl`, one line per 20 steps) carry unweighted loss parts per modality
(`video/loss_rgb`, ...) and diagnostics: `diag_gate_mean`, `diag_delta_abs_cells_key/_nonkey`, `diag_psnr`,
`diag_psnr_warp`, `diag_psnr_fut_h1/_h4`. Health checks for v1: the gate of video / 3D should stay well
below 1 and Δ on non-key frames should grow with the stride; `w_lpips*loss_lpips` and `w_gram*loss_gram`
should stay within ~0.3-3x of `loss_rgb`.

Eval keys for the C/D split (video): `psnr_full`, `psnr_noD` (no D at all), `psnr_zeroD`, `psnr_eraseC`,
`psnr_warp`, their `gap_*`, `delta_cells_mean/p95`, `gate`, each also `_key` / `_nonkey`; forecasts:
`forecast_h{h}_psnr` vs `_copy_rec` and `_copy_gt`. Objaverse novel-view numbers of v0 came from an
ad-hoc script on alpha; they are not in `eval_dino_v0.py` yet.

## Runs on alpha

alpha: 8x AMD MI355X (288 GB), ROCm torch 2.14; env `/scratch/vunguyen13/mavt/env`. **GPUs 4-7 belong to
other people: never use them.** Data: ImageNet-1k parquet, OpenVid-1M at original resolution
(281,067 clips, last 1,000 held out), Objaverse-LVIS posed views (38,961 objects, last 500 held out),
SSv2 (train manifest `/scratch/vunguyen13/data/dino_v0_data/ssv2/ssv2_train_manifest.jsonl`, val for eval only).

| Run | Dir under `/scratch/vunguyen13/mavt/runs/` | Code | Status |
|---|---|---|---|
| v0 | `muse-spark-mavt` | 5e319ba | done, 38,511 steps; eval in `eval/` |
| v1 phase 1 | `muse-spark-mavt-v1` | a755732 (copy in `/scratch/vunguyen13/mavt/muse-spark-mavt/code-a755732`) | training, tmux `muse-spark-mavt-v1` |

v0 results (step 38,511):
- ImageNet val: PSNR 22.50, SSIM 0.560, LPIPS 0.533, rFID 57.7.
- OpenVid 21.73 dB (key 22.83 / non-key 21.36); SSv2 23.27 dB (key 27.25 / non-key 21.94).
- Objaverse: seen views 24.15, novel views 20.18 (copy-previous-view 17.51).
- Dropping D entirely: OpenVid -0.6 dB, SSv2 -1.4, Objaverse seen -4.6 / novel -2.0. Gate 0.999, |Δ| ~0.06 cells.
- Forecast feature cosine 0.845 vs copy-last 0.952.
- Probes (kNN / linear): ImageNet z_s 82.7 / 83.9 vs DINOv3 CLS 82.8 / 83.7; SSv2 z_s 7.0 / 10.8 vs CLS 7.8 / 11.4.

Speed (1 GPU, same batch, before -> after the v1 speed work, eager / compiled): v0 image step
798 -> 404 / 318 ms, video 1271 -> 645 / 552 ms, 3D 619 -> 444 / 428 ms, memory roughly halved.
v0 code passed 5-D tensors to SDPA and never reached the fused attention kernels.

## Design documents

- Information flow and weaknesses of v0 vs baselines: https://claude.ai/artifact/12NAQUf4tYndWBk7svCiAz
- v1 fixes, options and roadmap (phases and gates): https://claude.ai/artifact/FGxTRWcXvaUasE7gkMbMND
- Philosophy and contribution vs AToken / VideoTok4D: https://claude.ai/artifact/7Hrioen1NA6CnhTRnVZaUK
- Speed analysis: `/mnt/project-files/mavt-review/tang-toc/de-xuat-tang-toc-v0.md` (project files)

## Working rules for people and agents

- Develop on the PR branch; keep v0 behaviour bit-identical when the v1 flags are off (tests check it).
- Every speed change must come with a test comparing old and new outputs and gradients.
- On alpha: GPUs 0-3 only; no package installs and no deletions without an's explicit OK;
  credentials only through environment variables or stdin, never in files or logs.
- On an's Mac, work in the main clone `~/Workspace/AToken/atoken`; do not create new git worktrees there.
