#!/bin/bash
#SBATCH --job-name=mavt
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00             # đặt tối đa được phép ở cụm bạn
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
set -euo pipefail


# ============================================================================
# MAVT Stage 2: Image + Video
#   - SigLIP2 last 4 blocks unfrozen
#   - LR = 5e-5
#   - Resume from Stage 1 checkpoint
#   - Reads images from WDS shards, videos from video2dataset shard dirs
#
# Usage:
#   sbatch train_stage2.sh
#   bash   train_stage2.sh   # interactive on GPU node
# ============================================================================

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_DIR"

IMAGE_SHARDS_DIR="$PROJECT_DIR/dataset/image10k/train"
VIDEO_SHARDS_DIR="$PROJECT_DIR/dataset/dataset_10m"

# --- Stage 1 checkpoint (edit this path after Stage 1 completes) ---
STAGE1_CKPT="checkpoints/stage1/last.ckpt"
ALLOW_FROM_SCRATCH="${ALLOW_FROM_SCRATCH:-false}"

mkdir -p logs checkpoints/stage2

# --- Environment ---
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | tr '\n' ':' | sed 's/:$//')"
unset VIRTUAL_ENV
if [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

_PYTHON_PATH="$(command -v python)"
case "$_PYTHON_PATH" in
    */.venv/*) _ACTIVE_ENV=".venv ($_PYTHON_PATH)" ;;
    */miniconda3/*|*/anaconda3/*|*/conda/*) _ACTIVE_ENV="conda ${CONDA_DEFAULT_ENV:-?} ($_PYTHON_PATH)" ;;
    *) _ACTIVE_ENV="system ($_PYTHON_PATH)" ;;
esac

echo "========================================"
echo "  MAVT Stage 2 — Image + Video"
echo "  Env:  $_ACTIVE_ENV"
echo "  GPUs: $NUM_GPUS"
echo "  Image shards: $IMAGE_SHARDS_DIR"
echo "  Video shards: $VIDEO_SHARDS_DIR"
echo "  Checkpoint:   $STAGE1_CKPT"
echo "========================================"

# Verify checkpoint exists
if [ ! -f "$STAGE1_CKPT" ]; then
    if [ "$ALLOW_FROM_SCRATCH" != "true" ]; then
        echo "[ERROR] Stage 1 checkpoint not found: $STAGE1_CKPT"
        echo "        Stage 2 should resume from Stage 1. Set ALLOW_FROM_SCRATCH=true only for debugging."
        exit 1
    fi
    echo "[WARN] Stage 1 checkpoint not found; training Stage 2 from pretrained SigLIP2 only."
    CKPT_ARG=""
else
    CKPT_ARG="--ckpt_path $STAGE1_CKPT"
fi

python train.py fit \
    --config configs/model/mavt_base.yaml \
    --config configs/train/universal_data/stage2_universal.yaml \
    --data.image_shards_dir "$IMAGE_SHARDS_DIR" \
    --data.video_shards_dir "$VIDEO_SHARDS_DIR" \
    --data.video_max_shards 50 \
    --data.active_modalities '["image", "video"]' \
    --data.image_resolution 256 \
    --data.video_frames 16 \
    --data.video_resolution 256 \
    --data.batch_size 16 \
    --data.num_workers 8 \
    --data.pin_memory true \
    --model.training_stage 2 \
    --model.init_siglip2 true \
    --model.use_lpips true \
    --model.warmup_steps 500 \
    --model.total_steps 200000 \
    --trainer.devices "$NUM_GPUS" \
    --trainer.precision bf16-mixed \
    --trainer.max_steps 200000 \
    --trainer.accumulate_grad_batches 2 \
    --trainer.log_every_n_steps 50 \
    --trainer.val_check_interval 2000 \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project mavt \
    --trainer.logger.init_args.name stage2-video \
    $CKPT_ARG
