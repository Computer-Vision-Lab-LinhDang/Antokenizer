# AnToken — Unified Visual Tokenizer Implementation Plan

## Context

Build an AToken-inspired unified visual tokenizer in PyTorch Lightning. Single shared encoder + modality-specific decoders produce continuous latents (for diffusion/flow), discrete tokens (for AR), and semantic features (for understanding) across Image, Video, (and 3D — deferred). Trained with adversarial-free losses (L1 + LPIPS + Gram + CLIP) through a progressive curriculum.

Codebase is currently empty (hello-world `main.py` + empty `pyproject.toml`). Everything is greenfield.

Full spec lives in `implementation.md`. This plan implements the spec with four concrete decisions:

- **3D decoder strategy**: when 3D is later added, use multi-view RGB regression (treat 3D as 8-view "video" with known cameras). Simplest path; no mesh/Gaussian Splats dependency.
- **3D in this plan**: **deferred**. We don't have mesh/point-cloud ground truth; dataset only has 8 pre-rendered views + cameras. Implementing Stage 3 now would force dense-grid / visual-hull hacks that don't match the research goal. Scope this plan to Stages 1, 2, and 4. Stage 3 is a follow-up once mesh data is available, and the architecture is kept extensible.
- **Backbone**: SigLIP2-So400m (D=1152, ~400M). Matches Apple's AToken paper. Config-driven so `-base` swap is trivial.
- **Discrete tokenizer**: both FSQ and multi-codebook VQ, behind a `class_path` registry. FSQ is the default for Stage 4; VQ available for ablation.

---

## Project Structure

```
AnToken/
├── pyproject.toml                              # dependencies
├── configs/
│   ├── base.yaml                               # shared trainer + model defaults
│   ├── model/
│   │   ├── antoken_base.yaml                   # SigLIP2-Base (dev)
│   │   └── antoken_so400m.yaml                 # SigLIP2-So400m (main)
│   ├── data/
│   │   ├── image_only.yaml
│   │   ├── image_video.yaml
│   │   └── all_with_discrete.yaml
│   └── stages/
│       ├── stage1_image.yaml
│       ├── stage2_video.yaml
│       └── stage4_discrete.yaml
├── scripts/
│   ├── train.py                                # LightningCLI entrypoint (fit/validate/test)
│   ├── eval.py                                 # rFID/PSNR/SSIM/LPIPS/CLIP-score
│   └── visualize.py                            # recon grids + latent PCA
├── src/antoken/
│   ├── __init__.py
│   ├── cli.py                                  # LightningCLI subclass
│   ├── patchify/
│   │   ├── image.py                            # ImagePatchifier
│   │   ├── video.py                            # VideoPatchifier (Conv3d + inflation)
│   │   └── coords.py                           # 4D position helpers
│   ├── encoder/
│   │   ├── rope4d.py                           # RoPE4D (head dim split into 4 groups)
│   │   ├── patch_embed.py                      # SpaceTimePatchEmbed (shared Conv3d)
│   │   ├── siglip2_backbone.py                 # ViT reimpl + load SigLIP2-So400m weights
│   │   └── unified_encoder.py                  # Patchifiers → backbone
│   ├── latent/
│   │   ├── continuous_head.py                  # mu/logvar + KL + expand_dim()
│   │   ├── discrete_fsq.py                     # FSQ quantizer
│   │   ├── discrete_vq.py                      # Multi-codebook VQ
│   │   ├── semantic_head.py                    # attn-pool → SigLIP2 text dim
│   │   └── router.py                           # stage-aware head dispatcher
│   ├── decoder/
│   │   ├── base.py                             # DecoderViT (cross-attn latents)
│   │   ├── image.py                            # ImageDecoder → RGB
│   │   └── video.py                            # VideoDecoder → frames (causal)
│   ├── losses/
│   │   ├── recon.py                            # L1 + LPIPS + Gram
│   │   ├── temporal.py                         # RAFT optical-flow consistency
│   │   ├── clip_contrastive.py                 # InfoNCE against SigLIP2 text tower
│   │   ├── kl.py                               # KL regularizer
│   │   └── fsq_vq_aux.py                       # codebook usage + VQ commit
│   ├── data/
│   │   ├── image_dataset.py
│   │   ├── video_dataset.py                    # decord
│   │   ├── transforms.py
│   │   ├── samplers.py                         # weighted modality sampler
│   │   └── datamodule.py                       # UnifiedDataModule (stage-aware)
│   ├── model/
│   │   ├── antoken.py                          # AToken LightningModule
│   │   └── stage_manager.py                    # freeze/unfreeze + latent-dim expansion
│   └── utils/
│       ├── inflate.py                          # 2D→3D Conv weight inflation
│       ├── metrics.py                          # rFID, PSNR, SSIM, LPIPS, CLIP-score
│       ├── vis.py                              # recon grids, latent PCA, codebook hist
│       └── logging.py
└── tests/
    ├── test_rope4d.py                          # t=z=0 equivalence to 2D RoPE
    ├── test_inflate.py                         # Conv3d(T=1) ≡ Conv2d (fp32 eps)
    ├── test_patchify_image.py
    ├── test_patchify_video.py
    ├── test_encoder_shapes.py
    ├── test_decoder_roundtrip.py
    ├── test_datamodule.py
    ├── test_fsq.py                             # STE grad nonzero
    └── test_vq.py                              # codebook usage warm-up
```

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "antoken"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.3",
  "torchvision>=0.18",           # includes RAFT for optical flow
  "lightning>=2.3",
  "jsonargparse[signatures]>=4.27",
  "transformers>=4.45",          # SigLIP2 + text tower
  "einops>=0.8",
  "decord>=0.6",                 # video decode
  "lpips>=0.1.4",
  "torchmetrics[image]>=1.4",    # FID/SSIM/LPIPS/PSNR
  "numpy>=1.26",
  "pillow>=10",
  "wandb>=0.17",
  "tensorboard>=2.16",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
antoken-train = "scripts.train:main"
antoken-eval  = "scripts.eval:main"
```

Notes:
- RAFT ships in `torchvision.models.optical_flow` — no extra dep.
- `lpips` loads VGG; reuse it for the Gram loss so VGG loads once.
- No `gsplat` (3D deferred).

---

## Core Modules — Design Details

### 1. `patchify/`

**`ImagePatchifier`** (wraps the shared `SpaceTimePatchEmbed` with T=1 causal-padded input, so a single Conv3d handles both modalities)
- `forward(x: (B,3,H,W)) → tokens: (B,N,D), pos: (B,N,4)` with `pos=(0,i,j,0)`.

**`VideoPatchifier`**
- `forward(x: (B,3,T,H,W)) → tokens, pos` with `pos=(k,i,j,0)`, `k = floor(t/t_p)`.
- `t_p=2, p=16` default. Produces `(T/2)·(H/16)·(W/16)` tokens.
- Temporal-tiling helper `iter_tiles(x, tile_frames=32, stride=16)` for inference on long videos with decoder-side KV cache.

**`coords.py`**
- `make_positions_image/video`, `concat_modality_positions()` returning `(B,N,4)` + `modality_ids`.

### 2. `encoder/`

**`RoPE4D`** — split per-head dim into 4 equal groups (assert `head_dim % 8 == 0`); rotate each group against one axis of `pos`. For images, `t=z=0` reduces exactly to 2D RoPE (test enforces this).

**`SpaceTimePatchEmbed`** — `nn.Conv3d(3, D, (t_p,p,p), stride=(t_p,p,p))`. `from_2d_weights(W_2d)` inflates per spec: last temporal slice = `W_2d`, others = 0. Shared between image and video paths.

**`SigLIP2Backbone`** — reimplemented ViT blocks (~150 LOC) to inject 4D RoPE cleanly. Loads `google/siglip2-so400m-patch16-384` state dict by key; strips original position embedding. Supports:
- Causal-in-t attention mask (from `pos[:,:,0]` diffs).
- Gradient checkpointing on blocks (mandatory at So400m scale).

**`UnifiedEncoder`** — owns patchifiers + backbone; returns `z, pos, modality_ids, mask`. Builds attention mask (no cross-sample leakage; causal-in-t for video).

### 3. `latent/`

**`ContinuousHead`**
- `Linear(D, 2·L)` → `mu, logvar` → reparameterize.
- `L=32` at stage 1; `expand_latent_dim(48)` at stage 2 via `StageManager` (copies first 32 out-channels, zero-inits the new 16 — mirrors the Conv3d inflation philosophy).

**`DiscreteFSQHead`**
- `Linear(D, 6)` then FSQ with levels `[8,8,8,5,5,5]`. Straight-through estimator. No codebook, no EMA.

**`DiscreteVQHead`**
- Multi-codebook (product quantization): e.g., 4 codebooks × 1024 entries. EMA update + cosine-similarity lookup. Commit + codebook loss exposed via `losses/fsq_vq_aux.py`.

**`SemanticHead`**
- Attention-pool (learned query cross-attending over modality-valid tokens) → `Linear(D, text_embed_dim=1152)`. Output `(B, 1152)`.

**`LatentRouter`**
- Given `encoder_out` + `stage`, returns dict `{continuous, discrete, semantic}`. Stages 1–2: continuous + semantic. Stage 4: add discrete (FSQ by default; VQ swap via config).

### 4. `decoder/`

**`DecoderViT`** — 8-layer, width=D transformer. Takes modality-specific learned queries (positioned in 4D target space) and cross-attends to latents. 4D RoPE on queries for target geometry awareness.

**`ImageDecoder`** — queries = `(0,i,j,0)` grid at `H/p × W/p`; final `Linear(D, p²·3)`; unpatchify to `(B,3,H,W)`.

**`VideoDecoder`** — queries = `(k,i,j,0)` grid; causal-in-t mask matches encoder; final `Linear(D, t_p·p²·3)`; unpatchify to `(B,3,T,H,W)`. KV-cache helper for inference on tiled long videos.

### 5. `losses/`

- `ReconstructionLoss`: `w_l1·L1 + w_lpips·LPIPS + w_gram·Gram`. Gram computed from LPIPS's VGG feature maps (shared module; compute once).
- `OpticalFlowConsistency`: RAFT-small (frozen) predicts flow `frame_t → frame_{t+1}` on originals; warp reconstructed `frame_t` by that flow; penalize `L1(warped_rec_t, rec_{t+1})`. Only active on video batches.
- `CLIPContrastiveLoss`: InfoNCE between `z_sem` (reconstructed semantic) and frozen SigLIP2 text embedding from caption.
- `KLRegLoss`: standard Gaussian KL, weight `~1e-5`.
- `FSQVQAuxLoss`: codebook-usage entropy + VQ commit loss (if VQ); mostly a logging channel for FSQ.

### 6. `data/`

**`ImageDataset`** — scan `images/*.jpg`, join by stem against `captions/images.json`. Transforms: resize short side → 384, random-crop 384×384 for train / center-crop for val, SigLIP2 normalize (`mean=std=[0.5]*3`).

**`VideoDataset`** — `decord.VideoReader`, sample 16 contiguous frames from random start, spatial crop 256×256 (compute budget), join `captions/videos.json`.

**`UnifiedDataModule`** — holds per-modality datasets; train loader uses a **weighted alternating strategy**: each step picks one modality per rank with probability from `modality_weights` config. Simpler than mixed-batch, avoids token-padding across modalities. Each rank independently samples (no cross-rank coordination); DDP balance is statistical.

**Captions**: loaded lazily per item. Missing caption → empty string (CLIP loss skipped for that sample via mask).

**3D path stubbed**: `ThreeDDataset` scaffolded but not wired into DataModule until a follow-up. Keep the interface ready (`threed_objects/` loader reading `cameras.json` + `view_*.png`).

### 7. `model/antoken.py` — LightningModule

```python
class AToken(pl.LightningModule):
    def __init__(self, encoder_cfg, latent_cfg, decoder_cfg, loss_cfg,
                 stage: int, optimizer_cfg, scheduler_cfg): ...

    def training_step(self, batch, batch_idx):
        modality = batch['modality']
        enc_out = self.encoder(batch)
        latents = self.router(enc_out, stage=self.hparams.stage)
        rec = self.decoders[modality](latents, batch)
        losses = self.loss_fn(batch, rec, latents, enc_out,
                              modality=modality, stage=self.hparams.stage)
        self.log_dict({f'train/{k}': v for k, v in losses.items()}, sync_dist=True)
        return losses['total']

    def configure_optimizers(self):
        params = self.stage_manager.trainable_params(self)
        opt = torch.optim.AdamW(params, **self.hparams.optimizer_cfg)
        sched = get_cosine_schedule_with_warmup(opt, **self.hparams.scheduler_cfg)
        return [opt], [{'scheduler': sched, 'interval': 'step'}]

    def on_fit_start(self):
        self.stage_manager.apply(self, stage=self.hparams.stage)
```

### 8. `stage_manager.py`

Encapsulates all stage-switching side effects:

| Stage | Data | Trainable | Frozen | Latent dim | Notes |
|------:|------|-----------|--------|-----------:|-------|
| 1 | image | patch embed, backbone, continuous, semantic, image decoder | temporal Conv3d channel (via grad mask) | 32 | SigLIP2 weights loaded; inflation zero-init means stage 1 ≡ standard ViT |
| 2 | image + video | + unfreeze temporal Conv3d channel, add video decoder | — | **32→48 (inflated)** | KV-cache temporal tiling for long videos |
| 4 | all + discrete | discrete head + heads-above-it | encoder optionally frozen (config flag) | 48 | FSQ or VQ; track codebook usage |

`expand_latent_dim(48)` runs once at stage 2 `on_fit_start`: it swaps `Linear(D, 2·32)` for `Linear(D, 2·48)`, copies old weight into the first 32 out-channels, zero-inits the rest. Decoder's input projection gets the symmetric expansion.

Stage transitions happen via `--ckpt_path prev_stage/last.ckpt` (Lightning's `fit` resume), with `strict=False` to accept new modules.

### 9. Configs

Use LightningCLI's native `--config` stacking (no Hydra).

**`configs/base.yaml`** (abridged):
```yaml
seed_everything: 42
trainer:
  accelerator: gpu
  devices: -1
  strategy: ddp_find_unused_parameters_true
  precision: bf16-mixed
  max_steps: 300000
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  log_every_n_steps: 50
  val_check_interval: 5000
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args: {dirpath: runs/stage1, save_top_k: 3, monitor: val/rfid, mode: min, save_last: true}
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args: {project: antoken}
model:
  class_path: antoken.model.AToken
  init_args:
    stage: 1
    encoder_cfg:
      backbone: google/siglip2-so400m-patch16-384
      embed_dim: 1152
      patch_size: 16
      temporal_patch: 2
      rope4d: true
      grad_checkpoint: true
    latent_cfg:
      continuous_dim: 32
      discrete: null
      semantic_dim: 1152
    decoder_cfg:
      depth: 8
      width: 1152
    loss_cfg:
      w_l1: 1.0
      w_lpips: 1.0
      w_gram: 0.1
      w_clip: 0.1
      w_kl: 1.0e-5
      w_flow: 0.0
    optimizer_cfg: {lr: 1.0e-4, betas: [0.9, 0.95], weight_decay: 0.05}
    scheduler_cfg: {num_warmup_steps: 2000, num_training_steps: 300000}
data:
  class_path: antoken.data.UnifiedDataModule
  init_args:
    root: /path/to/data
    modality_weights: {image: 1.0, video: 0.0}
    batch_size: 32
    num_workers: 8
```

**`configs/stages/stage2_video.yaml`**:
```yaml
model:
  init_args:
    stage: 2
    latent_cfg: {continuous_dim: 48}
    loss_cfg: {w_flow: 0.1}
trainer:
  max_steps: 200000
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args: {dirpath: runs/stage2, save_top_k: 3, monitor: val/rfid, mode: min, save_last: true}
data:
  init_args:
    modality_weights: {image: 0.4, video: 0.6}
    batch_size: 16
ckpt_path: runs/stage1/last.ckpt
```

**`configs/stages/stage4_discrete.yaml`**:
```yaml
model:
  init_args:
    stage: 4
    latent_cfg:
      discrete:
        class_path: antoken.latent.DiscreteFSQHead
        init_args: {levels: [8, 8, 8, 5, 5, 5]}
trainer:
  max_steps: 100000
data:
  init_args:
    modality_weights: {image: 0.5, video: 0.5}
ckpt_path: runs/stage2/last.ckpt
```

Entrypoint:
```bash
python scripts/train.py fit --config configs/base.yaml --config configs/stages/stage2_video.yaml
```

### 10. `scripts/`

**`train.py`** — ~15 lines: construct `LightningCLI(AToken, UnifiedDataModule, subclass_mode_model=False, save_config_kwargs={'overwrite': True})`.

**`eval.py`** — load checkpoint, iterate val dataloader per modality, compute & print metrics table, dump recon grids to `--out-dir`.

**`visualize.py`** — same but focus on visuals: reconstruction grids, 4-frame video filmstrips, latent-PCA scatter colored by modality, codebook histogram for stage 4.

---

## Implementation Order (for the engineer)

Each step ends with a verifiable check so we can loop until green (per `CLAUDE.md` §4).

1. **Scaffolding + deps**
   - Populate `pyproject.toml`, create package skeleton, add `pytest` config.
   - Verify: `pip install -e .` succeeds; `pytest tests/ -q` collects zero tests without error.

2. **RoPE4D + inflation + patch embed (with tests)** — these are load-bearing correctness foundations.
   - Verify: `test_rope4d.py` passes (t=z=0 ≡ standard 2D RoPE); `test_inflate.py` passes (Conv3d(T=1, causal-padded) output ≡ Conv2d output within fp32 eps); `test_patchify_image.py` and `test_patchify_video.py` pass shape + position assertions.

3. **SigLIP2 backbone load**
   - Reimplement ViT blocks with 4D RoPE. Load `google/siglip2-so400m-patch16-384` weights by key.
   - Verify: encode a batch of 16 ImageNet-val images, compute cosine similarity against HF reference SigLIP2 image embeddings — should match to ≥0.98 cosine (proves weight load + attention rewire didn't break the pretrained backbone).

4. **ImageDecoder + continuous head + recon loss**
   - Verify: overfit 100 images for 2K steps; train L1 → < 0.02; reconstructions visually recognizable.

5. **DataModule (image path only) + Stage 1 training end-to-end**
   - Verify: launch `python scripts/train.py fit --config configs/base.yaml` on 1 GPU; loss decreases; W&B logs populate; checkpoint saved; `val/rfid` computed.

6. **VideoPatchifier + VideoDecoder + Stage 2 resume**
   - Verify: `stage_manager.expand_latent_dim(48)` preserves old outputs numerically on an image batch (first 32 dims identical, last 16 near zero); resume training and train loss doesn't spike >2× before recovering.

7. **Optical flow loss + full Stage 2 run**
   - Verify: video reconstructions temporally coherent (watch rendered GIF); flow loss decreasing.

8. **Discrete heads (FSQ + VQ) + Stage 4**
   - Verify: FSQ straight-through gives nonzero gradients; codebook usage entropy stable > 80% of max; rFID gap vs. continuous within ~20%.

9. **Evaluation + visualization scripts**
   - Verify: `scripts/eval.py` produces metrics table with all entries populated; `scripts/visualize.py` produces recon grids + latent PCA + codebook histogram.

10. **DDP + multi-node sanity run**
    - Verify: `--devices 8 --num_nodes 1` matches single-GPU loss curve within noise over 1K steps.

---

## Key Files to Create

- `src/antoken/encoder/rope4d.py` — correctness foundation; any bug here silently degrades everything.
- `src/antoken/encoder/patch_embed.py` + `src/antoken/utils/inflate.py` — stage-1/2 weight-sharing hinges on this inflation being identity for T=1.
- `src/antoken/encoder/siglip2_backbone.py` — reimplemented ViT with 4D RoPE + state-dict load. Largest single file (~300 LOC).
- `src/antoken/model/antoken.py` + `src/antoken/model/stage_manager.py` — the LightningModule and all stage-switching logic.
- `src/antoken/data/datamodule.py` — weighted-alternating modality sampling.
- `configs/base.yaml` + `configs/stages/stage{1,2,4}_*.yaml` — LightningCLI config stack.
- `scripts/train.py` + `scripts/eval.py` + `scripts/visualize.py`.

---

## Risks & Mitigations

1. **4D RoPE correctness bug** (silent, catastrophic). → Dedicated numerical test vs. 2D RoPE on image inputs.
2. **Conv3d inflation correctness** (stage 2 inherits wrong features). → Identity test for T=1 input.
3. **SigLIP2 weight load mismatch** (silent backbone degradation). → Cosine-similarity regression test vs. HF reference.
4. **Video memory at So400m** — 16 frames × 256² × batch 16 × 1152-d = ~80GB activations without checkpointing. → Gradient checkpointing mandatory; may also need ZeRO-2 (Lightning's `fsdp` strategy).
5. **Latent-dim expansion discontinuity at stage 2** — train loss may spike. → `expand_latent_dim` preserves old outputs by construction (zero-init new channels); add an assertion test.
6. **Modality sampler starvation** — weighted alternating can leave one modality starved on small datasets. → Log per-modality step count each epoch.
7. **HF SigLIP2-So400m access** — may be gated. → Confirm user has HF token with access before Step 3.
8. **3D re-introduction later** — keep `ThreeDDataset` stubbed and `modality_ids` already including a "3d" slot so adding it later is a data-pipeline change, not an architecture change.

---

## Verification (end-to-end)

After full implementation, verify the system by:

1. **Unit tests**: `pytest tests/ -v` — all RoPE, inflation, patchify, decoder roundtrip, FSQ/VQ tests pass.
2. **Stage 1 smoke**: `python scripts/train.py fit --config configs/base.yaml --trainer.fast_dev_run=5` completes without error.
3. **Stage 1 → 2 → 4 chain**: run each stage to convergence (or reduced `max_steps` for validation), confirm checkpoints chain via `ckpt_path`.
4. **Metrics regression**: `python scripts/eval.py --ckpt runs/stage2/last.ckpt` reports:
   - Image rFID ≤ a threshold (baseline: AToken paper ~0.5 rFID; our Base target < 3.0, So400m target < 1.5).
   - Video rFID and temporal warp-PSNR within sane ranges.
5. **Visualization**: `python scripts/visualize.py --ckpt runs/stage4/last.ckpt` produces recon grids, latent PCA (modalities cluster in shared space), codebook histogram (entropy > 80% of uniform).
6. **DDP parity**: 1-GPU vs 8-GPU training curves overlap within statistical noise over 1K steps.
