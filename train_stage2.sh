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

# --- Stage 1 best checkpoint (used as init for Stage 2) ---
# Overridable via env: STAGE1_CKPT=path/to/ckpt bash train_stage2.sh
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/stage1_5/balanced/mavt-stage1_5-balanced-step=0120000-val/loss=0.1238.ckpt}"

# --- Resume from a Stage 2 checkpoint instead of init from Stage 1 ---
# Set RESUME_CKPT=path/to/stage2.ckpt to do a true Lightning resume (restores
# optimizer / scheduler / global_step). Leave empty for fresh Stage 2 start.
RESUME_CKPT="${RESUME_CKPT:-}"

mkdir -p logs checkpoints/stage2

# --- Environment ---
# Use project .venv (uv-managed). Override with PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
export PATH="$PROJECT_DIR/.venv/bin:${PATH:-}"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

# Resolve init/resume strategy:
#   RESUME_CKPT (true Lightning resume; restores optim+scheduler+step)
#   else STAGE1_CKPT via model.init_from_ckpt (weights only, soft restart)
INIT_ARGS=()
CKPT_ARG=""
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    CKPT_ARG="--ckpt_path $RESUME_CKPT"
    MODE="resume from $RESUME_CKPT"
elif [ -f "$STAGE1_CKPT" ]; then
    INIT_ARGS+=( --model.init_from_ckpt "$STAGE1_CKPT" )
    MODE="init weights from $STAGE1_CKPT"
else
    echo "[WARN] Neither RESUME_CKPT nor STAGE1_CKPT exists. Training from scratch."
    MODE="from scratch"
fi

# Loss weights / data overrides — exportable for experiments
W_TEMP="${W_TEMP:-0.1}"             # temporal loss weight (video). 0 = off
W_SEM="${W_SEM:-0.3}"
W_LPIPS="${W_LPIPS:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-8}"       # effective batch = BATCH_SIZE × accumulate_grad_batches
NUM_WORKERS="${NUM_WORKERS:-4}"
VIDEO_MAX_SHARDS="${VIDEO_MAX_SHARDS:-0}"  # 0 = no cap

echo "========================================"
echo "  MAVT Stage 2 — Image + Video"
echo "  GPUs:        $NUM_GPUS"
echo "  Image shards:$IMAGE_SHARDS_DIR"
echo "  Video shards:$VIDEO_SHARDS_DIR (max=$VIDEO_MAX_SHARDS, 0=unlimited)"
echo "  Mode:        $MODE"
echo "  Batch size:  $BATCH_SIZE (× accumulate=2 → effective 16)"
echo "  w_temp=$W_TEMP w_sem=$W_SEM w_lpips=$W_LPIPS"
echo "========================================"

MAX_SHARDS_ARG=()
if [ "$VIDEO_MAX_SHARDS" != "0" ]; then
    MAX_SHARDS_ARG=( --data.video_max_shards "$VIDEO_MAX_SHARDS" )
fi

"$PYTHON_BIN" train.py fit \
    --config configs/model/mavt_base.yaml \
    --config configs/train/universal_data/stage2_universal.yaml \
    --config configs/train/universal_data/stage2_3_paths.yaml \
    --data.image_shards_dir "$IMAGE_SHARDS_DIR" \
    --data.video_shards_dir "$VIDEO_SHARDS_DIR" \
    "${MAX_SHARDS_ARG[@]}" \
    --data.active_modalities '["image", "video"]' \
    --model.active_modalities '["image", "video"]' \
    --data.image_resolution 256 \
    --data.video_frames 16 \
    --data.video_resolution 256 \
    --data.batch_size "$BATCH_SIZE" \
    --data.num_workers "$NUM_WORKERS" \
    --data.pin_memory true \
    --data.persistent_workers true \
    --data.prefetch_factor 4 \
    --model.training_stage 2 \
    --model.init_siglip2 true \
    --model.use_lpips true \
    --model.use_clip false \
    --model.w_lpips "$W_LPIPS" \
    --model.w_sem "$W_SEM" \
    --model.w_temp "$W_TEMP" \
    --model.warmup_steps 500 \
    --model.total_steps 200000 \
    "${INIT_ARGS[@]}" \
    --trainer.devices "$NUM_GPUS" \
    --trainer.precision bf16-mixed \
    --trainer.max_steps 200000 \
    --trainer.accumulate_grad_batches 2 \
    --trainer.log_every_n_steps 50 \
    --trainer.val_check_interval 2000 \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project mavt \
    --trainer.logger.init_args.name "stage2_decoder_v2_w3" \
    $CKPT_ARG
