#!/bin/bash
#SBATCH --job-name=mavt-s3
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/stage3_%j.log
#SBATCH --error=logs/stage3_%j.err

# ============================================================================
# MAVT Stage 3: Image + Video + 3D
#   - SigLIP2 fully unfrozen
#   - LR = 2e-5
#   - Resume from Stage 2 checkpoint
#   - 3D data loaded directly from dataset/tripplane/<uid>/{oxoy,oxoz,oyoz}.png
#
# Prerequisites:
#   1. Stage 2 checkpoint exists
#   2. Triplane renders ready at $TRIPLANE_DIR (one subdir per object,
#      each containing oxoy.png, oxoz.png, oyoz.png)
#
# Usage:
#   sbatch train_stage3.sh
#   bash   train_stage3.sh   # interactive on GPU node
# ============================================================================

set -euo pipefail

# --- Single-GPU setup. SLURM (#SBATCH --gpus-per-node=1) exports
#     CUDA_VISIBLE_DEVICES itself; do not override here. Outside SLURM, set it
#     explicitly when invoking the script: `CUDA_VISIBLE_DEVICES=3 bash train_stage3.sh`.
NUM_GPUS=1

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_DIR"

IMAGE_SHARDS_DIR="$PROJECT_DIR/dataset/image10k/train"
VIDEO_SHARDS_DIR="$PROJECT_DIR/dataset/dataset_10m"
TRIPLANE_DIR="$PROJECT_DIR/dataset/tripplane"

# --- Stage 2 checkpoint (edit this path after Stage 2 completes) ---
STAGE2_CKPT="checkpoints/stage2/last.ckpt"
ALLOW_FROM_SCRATCH="${ALLOW_FROM_SCRATCH:-true}"

mkdir -p logs checkpoints/stage3

# --- Environment ---
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# --- WandB ---
export WANDB_PROJECT="${WANDB_PROJECT:-mavt-tokenizer}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-stage3-$(date +%Y%m%d-%H%M%S)}"

# --- Check 3D data availability ---
if [ -d "$TRIPLANE_DIR" ]; then
    N_3D=$(find "$TRIPLANE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
else
    N_3D=0
fi

echo "========================================"
echo "  MAVT Stage 3 — Image + Video + 3D"
echo "  GPUs: $NUM_GPUS  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "  Image shards:   $IMAGE_SHARDS_DIR"
echo "  Video shards:   $VIDEO_SHARDS_DIR"
echo "  Triplane dir:   $TRIPLANE_DIR"
echo "  3D objects:     $N_3D renders"
echo "  Checkpoint:     $STAGE2_CKPT"
echo "  WandB project:  $WANDB_PROJECT  run=$WANDB_RUN_NAME"
echo "========================================"

if [ "$N_3D" -eq 0 ]; then
    echo ""
    echo "[ERROR] No 3D triplane renders found in $TRIPLANE_DIR"
    echo "        Expected layout: $TRIPLANE_DIR/<uid>/{oxoy,oxoz,oyoz}.png"
    exit 1
fi

# Verify checkpoint exists
if [ ! -f "$STAGE2_CKPT" ]; then
    if [ "$ALLOW_FROM_SCRATCH" != "true" ]; then
        echo "[ERROR] Stage 2 checkpoint not found: $STAGE2_CKPT"
        echo "        Stage 3 should resume from Stage 2. Set ALLOW_FROM_SCRATCH=true only for debugging."
        exit 1
    fi
    echo "[WARN] Stage 2 checkpoint not found; training Stage 3 from pretrained SigLIP2 only."
    CKPT_ARG=""
else
    CKPT_ARG="--ckpt_path $STAGE2_CKPT"
fi

python train.py fit \
    --config configs/model/mavt_base.yaml \
    --config configs/train/universal_data/stage3_universal.yaml \
    --data.image_shards_dir "$IMAGE_SHARDS_DIR" \
    --data.video_shards_dir "$VIDEO_SHARDS_DIR" \
    --data.video_max_shards 209 \
    --data.triplane_dir "$TRIPLANE_DIR" \
    --data.active_modalities '["image", "video", "threed"]' \
    --data.image_resolution 256 \
    --data.video_frames 16 \
    --data.video_resolution 256 \
    --data.triplane_res 256 \
    --data.batch_size 4 \
    --data.num_workers 8 \
    --data.pin_memory true \
    --model.training_stage 3 \
    --model.init_siglip2 true \
    --model.use_lpips true \
    --model.warmup_steps 500 \
    --model.total_steps 200000 \
    --trainer.devices "$NUM_GPUS" \
    --trainer.strategy auto \
    --trainer.precision bf16-mixed \
    --trainer.max_steps 200000 \
    --trainer.accumulate_grad_batches 4 \
    --trainer.log_every_n_steps 50 \
    --trainer.val_check_interval 1.0 \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project "$WANDB_PROJECT" \
    --trainer.logger.init_args.name "$WANDB_RUN_NAME" \
    $CKPT_ARG
