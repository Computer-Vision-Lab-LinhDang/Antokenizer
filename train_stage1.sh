#!/bin/bash
#SBATCH --job-name=mavt-s1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_%j.log
#SBATCH --error=logs/stage1_%j.err

# ============================================================================
# MAVT Stage 1: Image Only
#   - SigLIP2 fully frozen
#   - LR = 1e-4
#   - Reads directly from WDS .tar shards
#
# Usage:
#   sbatch train_stage1.sh
#   bash   train_stage1.sh   # interactive on GPU node
# ============================================================================

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_DIR"

IMAGE_SHARDS_DIR="$PROJECT_DIR/dataset/image10k/train"

mkdir -p logs checkpoints/stage1

# --- Environment ---
if [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate base
fi

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

echo "========================================"
echo "  MAVT Stage 1 — Image Only"
echo "  GPUs: $NUM_GPUS"
echo "  Image shards: $IMAGE_SHARDS_DIR"
echo "========================================"

python train.py fit \
    --config configs/model/mavt_base.yaml \
    --config configs/train/universal_data/stage1_universal.yaml \
    --data.image_shards_dir "$IMAGE_SHARDS_DIR" \
    --data.active_modalities '["image"]' \
    --data.image_resolution 256 \
    --data.batch_size 32 \
    --data.num_workers 8 \
    --data.pin_memory true \
    --model.training_stage 1 \
    --model.init_siglip2 false \
    --model.use_lpips true \
    --model.use_clip false \
    --model.warmup_steps 1000 \
    --model.total_steps 200000 \
    --trainer.devices "$NUM_GPUS" \
    --trainer.precision bf16-mixed \
    --trainer.max_steps 200000 \
    --trainer.log_every_n_steps 50 \
    --trainer.val_check_interval 2000 \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project mavt \
    --trainer.logger.init_args.name stage1-image
