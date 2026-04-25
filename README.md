# MAVT — Memory-Augmented Vision Tokenizer

A unified visual tokenizer for images, videos, and 3D assets operating in a shared 4D latent space. MAVT combines two architectural decisions:

- **4D Relational Graph Attention (RGAT)** — injects typed geometric relationships (spatial, temporal, cross-plane) directly into Transformer attention, replacing the implicit all-pairs weighting of standard self-attention with explicit structural priors.
- **Content-Detail Split** — decompresses the token stream into a semantic content channel and a high-frequency detail channel via learned slot attention, achieving ~2.9× token reduction without discarding information.

~276M parameters. Designed for 8× A100/H100 DDP training across three modalities in a progressive curriculum.

---

## Architecture

```
Input (image / video / 3D triplane)
        |
        v
Stage 1: Patchify
  Unified Conv3d  (kernel 2x16x16, stride 2x16x16)
  Image: causal zero-pad → equivalent to Conv2d
  Video: direct application, bidirectional
  3D:    per-plane patchification, plane_id tags
  Output: tokens (B, N, 1152)  +  4D positions (N, 4)
        |
        v
Stage 2: Hybrid Transformer-RGAT Backbone (12 blocks)
  Blocks 0-3:  StandardTransformer  [SigLIP2 init]
  Block  4:    RGAT4D               [zero-init output]
  Blocks 5-7:  StandardTransformer  [SigLIP2 init]
  Block  8:    RGAT4D               [zero-init output]
  Blocks 9-11: StandardTransformer  [SigLIP2 init]

  RGAT4D edge types (dense masked attention, no torch_geometric):
    Type 0  SPATIAL      same (t,z), |dx|<=2, |dy|<=2
    Type 1  TEMPORAL     same (x,y,z), |dt|<=1         [video only]
    Type 2  DEPTH        reserved
    Type 3  CROSS-PLANE  different plane_id, shares>=1 coord [3D only]
        |
        v
Stage 3: Content-Detail Split
  ContentExtractor (slot cross-attn, 2 layers) -> C tokens  (N_c = 0.25*N)
  Residual R = X - broadcast(C)
  DynamicsPooler  (slot cross-attn, 2 layers) -> D tokens  (N_d = 0.10*N)
  Output: [C ; D]  (B, N_c+N_d, 1152)      ~2.9x compression
        |
        v
Stage 4: VAE Latent Projection
  VAE head:      [C;D] -> mu, logvar -> z in R^256 (per token)
  MRL prefixes:  z32, z64, z128, z256 share the same token latent
        |
        v
Stage 5: Dual Latent Readout
  Reconstruction: UnifiedDetailExpander cross-attends target positions into full z256
  Understanding:  learned query attention-pools each MRL prefix -> s in R^768
        |
        v
Outputs: reconstructed pixel tensor  +  semantic embedding
```

### Token counts (full resolution)

| Modality | Input N | After C-D Split | Compression |
|----------|---------|-----------------|-------------|
| Image 256px | 256 | 89 | 2.9x |
| Video 8f x 128px | 512 | 179 | 2.9x |
| 3D triplane S=64 | 48 | 24 | 2.0x |

---

## Installation

**Using uv (recommended):**
```bash
bash setup_env.sh
source .venv/bin/activate
```

**Using pip directly:**
```bash
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.8, PyTorch >= 2.2, CUDA 12.x recommended.

---

## Quick Start

### Smoke test (CPU, synthetic data, ~30 seconds)
```bash
python3 smoke_test.py
```

Runs 20 unit tests covering RGAT zero-init, adjacency mask edge counts, C-D Split residual ratio, and full forward passes for all three modalities.

### Training — Stage 1 (image only, synthetic data)
```bash
python3 train.py fit \
  --config configs/train/stage1_image.yaml \
  --trainer.max_steps 500 \
  --trainer.accelerator gpu
```

### Training — Stage 1 (real data, single GPU)
```bash
python3 train.py fit \
  --config configs/train/stage1_image.yaml \
  --data.image_root /path/to/open-images
```

### Training — Stage 1 (DDP, 8 GPUs)
```bash
python3 train.py fit \
  --config configs/train/stage1_image.yaml \
  --trainer.devices 8 \
  --trainer.strategy ddp \
  --data.image_root /path/to/open-images
```

### Training — Stage 2 (resume from stage 1 checkpoint)
```bash
python3 train.py fit \
  --config configs/train/stage2_video.yaml \
  --ckpt_path checkpoints/stage1/mavt-stage1-best.ckpt \
  --data.image_root /path/to/open-images \
  --data.video_root /path/to/webvid
```

### Training — Stage 3 (all modalities)
```bash
python3 train.py fit \
  --config configs/train/stage3_3d.yaml \
  --ckpt_path checkpoints/stage2/mavt-stage2-best.ckpt \
  --data.image_root /path/to/open-images \
  --data.video_root /path/to/webvid \
  --data.threed_root /path/to/cap3d
```

### Evaluation
```bash
python3 evaluate.py \
  --ckpt checkpoints/stage1/mavt-stage1-best.ckpt \
  --modality image \
  --data_root /path/to/images \
  --resolution 256
```

### WandB logging
Append to any training command:
```bash
  --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
  --trainer.logger.init_args.project mavt \
  --trainer.logger.init_args.name stage1-image
```

---

## Training Curriculum

Three progressive stages following the spec:

| Stage | Modalities | Steps | LR | SigLIP2 | Notes |
|-------|-----------|-------|----|---------|-------|
| 1 | Image | 200K | 1e-4 | Frozen | Establish spatial features |
| 2 | + Video | 200K | 5e-5 | Last 4 blocks unfrozen | Add temporal structure |
| 3 | + 3D | 50K | 2e-5 | Fully unfrozen | Cross-plane edges |

Resume across stages by passing `--ckpt_path` to the next stage's config.

---

## Configuration

All model and training hyperparameters are YAML-configurable via Lightning CLI. Override any field on the command line:

```bash
# Change patch size
python3 train.py fit --config configs/train/stage1_image.yaml \
  --model.patch_size 8

# Change RGAT spatial radius
python3 train.py fit --config configs/train/stage1_image.yaml \
  --model.r_s 1

# Disable LPIPS (faster, less GPU memory)
python3 train.py fit --config configs/train/stage1_image.yaml \
  --model.use_lpips false
```

### Key model parameters (`configs/model/mavt_base.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 1152 | Token embedding dimension |
| `num_heads` | 16 | Attention heads |
| `num_blocks` | 12 | Backbone depth (RGAT at positions 4, 8) |
| `patch_size` | 16 | Spatial patch size in pixels |
| `latent_dim` | 256 | Maximum VAE latent dimension per token |
| `mrl_prefixes` | [32, 64, 128, 256] | Matryoshka channel prefixes supervised for understanding |
| `r_s` | 2 | RGAT spatial radius (5x5 window) |
| `r_t` | 1 | RGAT temporal radius (+-1 frame) |
| `use_gradient_checkpointing` | true | Saves ~30% GPU memory, +15% compute |

---

## Loss Function

```
L_total = w_mod * (w_l1 * L1 + w_lpips * LPIPS)
        + w_kl  * KL
        + w_clip * InfoNCE(visual, text)     [optional]
        + w_aux  * SlotDiversity
```

Default weights: `w_l1=1.0, w_lpips=0.1, w_kl=1e-4, w_clip=0.1, w_aux=0.01`.

`w_mod` is a per-modality inverse-EMA scale so harder modalities receive proportionally more gradient.

---

## C-D Split Monitoring

Three signals logged during training to detect failure modes:

| Metric | Target | Failure mode |
|--------|--------|-------------|
| `cd_slot_diversity` | <= 0.5 | > 0.9 → slot collapse |
| `cd_residual_ratio` | 0.3 – 0.5 | < 0.1 → content over-fits; > 0.7 → content fails |
| `detail_contribution` | >= 15% | Detail branch inactive |

---

## SigLIP2 Weight Initialization

To load pretrained SigLIP2 weights into the 10 Transformer backbone blocks:

```bash
python3 train.py fit \
  --config configs/train/stage1_image.yaml \
  --model.init_siglip2 true \
  --model.siglip2_model_name google/siglip2-base-patch16-224
```

This requires `pip install transformers` and HuggingFace Hub access. The two RGAT4D blocks are always randomly initialized with zero-init output projections (identity at step 0).

---

## Hardware Requirements

**Minimum (smoke test / development):**
- Any CPU, 4 GB RAM

**Recommended (full-scale training):**
- 8x A100 80GB or H100 80GB
- bf16 mixed precision
- Batch size: 32 image / 16 video / 32 3D per GPU
- Estimated wall-clock: ~10 days for 450K total steps

**Memory notes:**
- RGAT4D blocks operate on the full (B, N, N) attention matrix
- For video with N=2048, enable `use_gradient_checkpointing=true` and reduce batch size if OOM
- Adjacency masks are precomputed once per (modality, resolution) and cached on device

---

## Project Structure

```
Antoken/
├── pyproject.toml
├── setup_env.sh
├── train.py              # LightningCLI entry point
├── evaluate.py           # Evaluation script
├── smoke_test.py         # 20 unit tests (no GPU required)
├── quick_train_test.py   # End-to-end training loop check
├── configs/
│   ├── model/mavt_base.yaml
│   └── train/
│       ├── stage1_image.yaml
│       ├── stage2_video.yaml
│       └── stage3_3d.yaml
└── src/mavt/
    ├── model/
    │   ├── patchify.py           # Conv3d patchification, 4D position grids
    │   ├── rgat.py               # RGAT4DBlock, build_adjacency
    │   ├── transformer.py        # StandardTransformerBlock (SigLIP2-compatible)
    │   ├── backbone.py           # 12-block hybrid backbone, mask caching
    │   ├── content_detail_split.py  # SlotPooler, ContentDetailSplit
    │   ├── latent_heads.py       # VAEHead
    │   ├── decoder.py            # Reconstruction decoder, MRL UnderstandingDecoder
    │   └── mavt.py               # Full MAVT model
    ├── losses/losses.py          # MAVTLoss, ModalityEMAWeighter, infonce_loss
    ├── data/
    │   ├── datasets.py           # Synthetic, ImageFolder, Video, ThreeD datasets
    │   └── datamodule.py         # MAVTDataModule (3-stage curriculum)
    ├── training/
    │   └── lightning_module.py   # MAVTLightningModule, LR schedule, visualization
    └── evaluation/
        └── metrics.py            # PSNR, SSIM, temporal-PSNR, FIDTracker
```

---

## Data Formats

| Modality | Dataset tensor shape | Source |
|----------|---------------------|--------|
| Image | `(3, H, W)` float32, normalized [-1, 1] | Open Images V7, any image folder |
| Video | `(3, T, H, W)` float32, normalized | WebVid, Panda-70M, HMDB51, MSVD |
| 3D triplane | `(3, 3, S, S)` float32 | Cap3D, TRELLIS-SLAT preprocessing |

Video reconstruction target is temporally downsampled to `(3, T//t_patch, H, W)` to match the patch-grid temporal resolution of the encoder output.

---

## Ablation Variants

As specified in section 10.2 of the design doc:

| Variant | RGAT | C-D Split | Run |
|---------|------|-----------|-----|
| V0 (baseline) | No | No | `--model.num_blocks 12` with standard Transformer only |
| V1 | No | Yes | Remove RGAT blocks from backbone |
| V2 | Yes | No | Set `content_ratio=1.0` to disable split |
| V3 (target) | Yes | Yes | Default config |
