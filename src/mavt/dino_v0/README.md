# MAVT on DINOv3 (`mavt.dino_v0`)

The current MAVT line of work (from 22/10/2026). A tokenizer for images, video and posed 3D object views,
built on a **frozen DINOv3 ViT-L/16**. The older pipelines described in the repo-root `README.md`
(SigLIP2 backbone, slots, v5-v7d) are paused; new work goes here.

Status on 23/10/2026:
- **v0** trained and evaluated. Its main flaw: the decoder barely uses D (the gate sat at ~1).
- **v1 phase 1** (the fixes below) is training.
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
| `config.py` | `DinoV0Config`: every hyper-parameter; v1 flags in the "v1 phase 1" and "v1 phase 2" blocks; `tiny_config()` for CPU tests |
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

## v1 phase 2: speed options (`configs/dino_v0/v1b_alpha.yaml`)

The phase-1 model, unchanged; only how much of it runs per step. A phase-1 step decodes 688 full frames per
GPU at 128 images + 16 clips x 8 frames + 16 objects x 4 views: 320 reconstructions, 224 forecasts (every
(t, h) pair for h = 1 and 4) and 144 warp-only frames, each through the 4 decoder blocks and the conv head.
Tests: `tests/dino_v0/test_speed_v1b.py` (old vs new outputs and gradients; defaults keep phase 1 bit-identical).

| Flag | phase 1 (default) | phase 2 | What it does |
|---|---|---|---|
| `fut_frames_per_horizon` | 0 (every pair) | 2 | forecast loss on 2 random source frames per horizon per step (one draw per step, shared by the batch, so shapes are constant for `torch.compile`) |
| `warp_frames` | 0 (every non-key frame) | 2 | warp-only loss on 2 random non-key frames per step |
| `conv_channels_last` | false | false | conv rgb head and its input in channels_last: the token grid `[B, G, G, D]` already is the NHWC layout of `[B, D, G, G]`, so the convs could run NHWC end to end without the NCHW <-> NHWC transposes MIOpen / cuDNN insert around their NHWC kernels. Measured useless on ROCm + inductor (below); kept for cuDNN |
| `window_read` | `dense` | `dense` | how the decoder reads the 5x5 C window: `dense` = masked SDPA over all 256 cells; `gather` = the 25 window cells gathered per query (the reference implementation, kept for tests) |

Measured on one MI355X (torch 2.14.1+rocm7.2, 128 images + 16 clips + 16 objects per step, regional compile,
one run at a time on an idle GPU, `scripts/profile_dino_v0.py --modality mixed`):

| Configuration | ms / step | peak memory | decoder forward |
|---|---|---|---|
| phase 1 (`v1_alpha.yaml`) | 1731 | 230 GB | 281 ms |
| + `fut_frames_per_horizon: 2`, `warp_frames: 2` (= `v1b_alpha.yaml`) | 1515 | 188 GB | 216 ms |
| + `conv_channels_last: true`, conv head compiled | 1487 | 188 GB | 218 ms |
| + `conv_channels_last: true`, conv head eager, `PYTORCH_MIOPEN_SUGGEST_NHWC=1` | 1834 | 275 GB | 368 ms |

The frame subsets cut the decodes per step from 688 to 480 (320 reconstructions + 96 forecasts + 64
warp-only): -12.5% time and -18% memory, less than the -30% of decodes because DINOv3 (145 ms), C / D (55 ms),
the trunk (30 ms), the attention and the optimizer step do not change. `conv_channels_last` changes nothing on
this stack: under inductor the NCHW <-> NHWC transposes around MIOpen's kernels stay (about 1,240 per step, 7%
of GPU time) whether or not the weights are channels_last, and PyTorch's MIOpen backend only honours
channels_last with `PYTORCH_MIOPEN_SUGGEST_NHWC=1`, which under inductor adds a second transpose kernel (1528
ms). Running the conv head eagerly with that variable does remove the transposes (conv time -23%) but loses
inductor's fused GroupNorm / SiLU kernels: +23% step time, +46% memory. So `v1b_alpha.yaml` keeps it off.

`scripts/bench_window_read.py` measures the C read at training shapes (688 rows, 12 heads, bf16, fwd + bwd):
on an MI355X the dense masked read takes 18.3 ms / 7.1 GB, within 5% of the same SDPA with no mask at all
(17.4 ms), while the gathered read takes 196 ms / 69 GB (25 copies of k / v and their scatter-add backward).
So the window read is not where the time goes and `dense` stays the default; `gather` is a correctness
reference only.

## Commands

```bash
# tests (CPU, ~30 s to ~20 min depending on the machine)
python -m pytest tests/dino_v0 -q

# CPU smoke runs of the trainer (tiny config, 3 steps)
python scripts/train_dino_v0.py --dry-run --out /tmp/dry
python scripts/train_dino_v0.py --dry-run --mix-modalities --out /tmp/dry_mix

# train v1 on 4 GPUs; the yaml `train:` block holds the run's settings (data paths too), CLI flags win
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 scripts/train_dino_v0.py \
    --config configs/dino_v0/v1_alpha.yaml --out <run>
# re-running the same command resumes from <out>/last.pt; SIGTERM / Ctrl-C saves last.pt

# evaluate a checkpoint (one GPU); writes <out>/eval.json + eval.md
PYTHONPATH=src python scripts/eval_dino_v0.py --ckpt <run>/last.pt --device cuda --amp \
    --imagenet-root <imagenet-1k parquet root> \
    --openvid-manifest <openvid manifest or dir> --video-limit 500 --video-frames 16 \
    --frame-stride 4 --stem eval_s4 --out <run>/eval

# profile one GPU (step time, memory, top kernels); --modality image|video|threed|mixed
PYTHONPATH=src python scripts/profile_dino_v0.py --config configs/dino_v0/v1_alpha.yaml --modality mixed \
    --batch-image 128 --batch-video 16 --batch-3d 16 --compile
# the same with the phase-2 speed options: --config configs/dino_v0/v1b_alpha.yaml

# the decoder's C read at training shapes: dense masked SDPA vs gathered window (fwd + bwd, bf16)
PYTHONPATH=src python scripts/bench_window_read.py --rows 688
```

Training logs (`<out>/log.jsonl`, one line per 20 steps) carry unweighted loss parts per modality
(`video/loss_rgb`, ...) and diagnostics: `diag_gate_mean`, `diag_delta_abs_cells_key/_nonkey`, `diag_psnr`,
`diag_psnr_warp`, `diag_psnr_fut_h1/_h4`. Health checks for v1: the gate of video / 3D should stay well
below 1 and Δ on non-key frames should grow with the stride; `w_lpips*loss_lpips` and `w_gram*loss_gram`
should stay within ~0.3-3x of `loss_rgb`.

Eval keys for the C/D split (video): `psnr_full`, `psnr_noD` (no D at all), `psnr_zeroD`, `psnr_eraseC`,
`psnr_warp`, their `gap_*`, `delta_cells_mean/p95`, `gate`, each also `_key` / `_nonkey`; forecasts:
`forecast_h{h}_psnr` vs `_copy_rec` and `_copy_gt`. Objaverse novel-view eval is not in `eval_dino_v0.py` yet.

## Runs, results and design notes

Run directories, data paths on the training server, eval numbers and the design documents are kept
outside this public repo (the team's local handoff notes). Data used: ImageNet-1k, OpenVid-1M at original
resolution (last 1,000 clips held out), Objaverse-LVIS posed views (last 500 objects held out),
Something-Something v2 (train split as unlabeled video; validation only for probes).

The v1 speed work roughly halved step time and memory of the v0 model on AMD MI355X: v0 passed 5-D
tensors to SDPA and never reached the fused attention kernels.

## Related papers

Ids of the tokenizers, DINOv3, RAE, CMD, iVideoGPT, DCVC and Kimi Linear were checked on arXiv on
23/10/2026; the others are long-standing references.

**Closest tokenizers (baselines / positioning)**

| Paper | arXiv | Why it matters here |
|---|---|---|
| AToken: A Unified Tokenizer for Vision | [2509.14476](https://arxiv.org/abs/2509.14476) | one tokenizer for images, video, 3D; our budget (12,288 per image) and main baseline |
| VideoTok4D: A 4D-Aware Video Tokenizer for Compact World Representation | [2609.12874](https://arxiv.org/abs/2609.12874) | static / dynamic query branches + track attention; closest to the C/D split |
| VTok: A Unified Video Tokenizer with Decoupled Spatial-Temporal Latents | [2602.04202](https://arxiv.org/abs/2602.04202) | keyframe + temporal residual latents |
| VidTok: A Versatile and Open-Source Video Tokenizer | [2412.13061](https://arxiv.org/abs/2412.13061) | video tokenizer baseline |
| An Image is Worth 32 Tokens for Reconstruction and Generation (TiTok) | [2406.07550](https://arxiv.org/abs/2406.07550) | compact 1D image tokens |
| HYDRA-X: Native Unified Multimodal Models with Holistic Visual Tokenizers | [2606.13289](https://arxiv.org/abs/2606.13289) | unified visual tokenizer for multimodal models |

**Frozen features and semantic latents**

| Paper | arXiv | Why it matters here |
|---|---|---|
| DINOv3 | [2508.10104](https://arxiv.org/abs/2508.10104) | the frozen encoder (ViT-L/16) and the feat / CLS targets |
| DINOv2: Learning Robust Visual Features without Supervision | [2304.07193](https://arxiv.org/abs/2304.07193) | predecessor; feature-space targets |
| Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models (VA-VAE) | [2501.01423](https://arxiv.org/abs/2501.01423) | aligning the latent to foundation features (planned for z_r) |
| Diffusion Transformers with Representation Autoencoders (RAE) | [2510.11690](https://arxiv.org/abs/2510.11690) | decoding straight from frozen representation encoders |
| High-Resolution Image Synthesis with Latent Diffusion Models (SD-VAE) | [2112.10752](https://arxiv.org/abs/2112.10752) | reconstruction target (~26 dB) and the conv decoder style |

**Content vs motion, latent actions, codecs**

| Paper | arXiv | Why it matters here |
|---|---|---|
| Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition (CMD) | [2403.14148](https://arxiv.org/abs/2403.14148) | content frame + motion latent, like C + D |
| iVideoGPT: Interactive VideoGPTs are Scalable World Models | [2405.15223](https://arxiv.org/abs/2405.15223) | future frames tokenized conditioned on context frames |
| Genie: Generative Interactive Environments | [2402.15391](https://arxiv.org/abs/2402.15391) | latent actions: D as an action-like bottleneck |
| Deep Contextual Video Compression (DCVC) | [2109.15047](https://arxiv.org/abs/2109.15047) | neural codec: motion + residual (the hybrid fallback) |

**Temporal objectives, world models, memory**

| Paper | arXiv | Why it matters here |
|---|---|---|
| Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA) | [2404.08471](https://arxiv.org/abs/2404.08471) | feature prediction; planned temporal objective for video z_s |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | [2506.09985](https://arxiv.org/abs/2506.09985) | same, at scale |
| DINO-WM: World Models on Pre-trained Visual Features | [2411.04983](https://arxiv.org/abs/2411.04983) | forecasting in a frozen DINO space |
| Mastering Diverse Domains through World Models (DreamerV3) | [2301.04104](https://arxiv.org/abs/2301.04104) | stochastic latent state |
| Kimi Linear: An Expressive, Efficient Attention Architecture (KDA) | [2510.26692](https://arxiv.org/abs/2510.26692) | event memory S (v2) |
| Gated Delta Networks: Improving Mamba2 with Delta Rule | [2412.06464](https://arxiv.org/abs/2412.06464) | delta-rule linear attention behind KDA |

**Views and rays (3D)**

| Paper | arXiv | Why it matters here |
|---|---|---|
| LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias | [2410.17242](https://arxiv.org/abs/2410.17242) | global ray-conditioned reads for novel views (planned) |
| Light Field Networks | [2106.02634](https://arxiv.org/abs/2106.02634) | Plücker ray embedding |

**Building blocks, losses, diagnostics**

| Paper | arXiv | Why it matters here |
|---|---|---|
| RoFormer: Enhanced Transformer with Rotary Position Embedding | [2104.09864](https://arxiv.org/abs/2104.09864) | axial RoPE on (t, u, v, z) |
| Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (MMDiT) | [2403.03206](https://arxiv.org/abs/2403.03206) | joint attention with per-stream weights (considered for C / D) |
| VICReg | [2105.04906](https://arxiv.org/abs/2105.04906) | variance / covariance loss on C |
| The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS) | [1801.03924](https://arxiv.org/abs/1801.03924) | perceptual loss and metric |
| A Neural Algorithm of Artistic Style (Gram loss) | [1508.06576](https://arxiv.org/abs/1508.06576) | Gram texture loss |
| RAFT: Recurrent All-Pairs Field Transforms for Optical Flow | [2003.12039](https://arxiv.org/abs/2003.12039) | flow reference to check Δ (eval only) |
| FlashAttention | [2205.14135](https://arxiv.org/abs/2205.14135) | fused attention kernels (SDPA) |

**Data**

| Paper | arXiv | Why it matters here |
|---|---|---|
| OpenVid-1M | [2407.02371](https://arxiv.org/abs/2407.02371) | training video at original resolution |
| The "something something" video database (SSv2) | [1706.04261](https://arxiv.org/abs/1706.04261) | unlabeled training video (train split) + temporal probes (val) |
| Objaverse: A Universe of Annotated 3D Objects | [2212.08051](https://arxiv.org/abs/2212.08051) | posed object views |

Not verified: "Co4DGen" (asymmetric static / dynamic attention routing, cited in VideoTok4D Fig. 3) and
"TivTok" (named in our early notes); no arXiv entry found under these names. ImageNet: Deng et al., CVPR 2009.

## Working rules for people and agents

- Develop on the PR branch; keep v0 behaviour bit-identical when the v1 flags are off (tests check it).
- Every speed change must come with a test comparing old and new outputs and gradients.
- On the shared training server use only the GPUs assigned to this project; no package installs and no
  deletions without the owner's explicit OK; credentials only through environment variables or stdin,
  never in files or logs.
