#!/bin/bash
#SBATCH --job-name=mavt-s1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=40G
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

UNIVERSAL_DATA_ROOT="${UNIVERSAL_DATA_ROOT:-${IMAGE_SHARDS_DIR:-$PROJECT_DIR/dataset/image10k/train}}"
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/train/universal_data/stage1_universal.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-false}"

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

PYTHON_CMD="${PYTHON_CMD:-python}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    else
        echo "[ERROR] Could not find python or python3 in PATH"
        exit 1
    fi
fi

CONFIG_INFO="$(
"$PYTHON_CMD" - "$STAGE1_CONFIG" <<'PY'
import re
import sys

try:
    import yaml

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model = cfg.get("model") or {}
    trainer = cfg.get("trainer") or {}
    callbacks = trainer.get("callbacks") or []

    init_siglip2 = model.get("init_siglip2", "unknown")
    if isinstance(init_siglip2, bool):
        init_siglip2 = str(init_siglip2).lower()

    siglip2_model_name = model.get("siglip2_model_name", "unknown")
    has_progress_bar = any(
        isinstance(cb, dict)
        and "RichProgressBar" in str(cb.get("class_path", ""))
        for cb in callbacks
    )

    print(init_siglip2)
    print(siglip2_model_name)
    print("yes" if has_progress_bar else "no")
except Exception:
    try:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()

        model_match = re.search(
            r"(?ms)^model:\s*\n(?P<body>(?:^[ \t].*\n|^\s*(?:#.*)?\n)+)",
            text,
        )
        model_body = model_match.group("body") if model_match else ""

        def read_model_value(key: str) -> str:
            match = re.search(rf"(?m)^[ \t]+{re.escape(key)}:\s*(.+?)\s*$", model_body)
            if not match:
                return "unknown"
            return match.group(1).strip().strip('"').strip("'")

        print(read_model_value("init_siglip2"))
        print(read_model_value("siglip2_model_name"))
        print("yes" if "RichProgressBar" in text else "no")
    except Exception:
        print("unknown")
        print("unknown")
        print("unknown")
PY
)"
CONFIG_INIT_SIGLIP2="$(printf '%s\n' "$CONFIG_INFO" | sed -n '1p')"
CONFIG_SIGLIP2_MODEL_NAME="$(printf '%s\n' "$CONFIG_INFO" | sed -n '2p')"
CONFIG_PROGRESS_BAR="$(printf '%s\n' "$CONFIG_INFO" | sed -n '3p')"

SIGLIP2_MODEL_NAME_ARG=()
if [ -n "${SIGLIP2_MODEL_NAME+x}" ]; then
    SIGLIP2_MODEL_NAME_EFFECTIVE="$SIGLIP2_MODEL_NAME"
    SIGLIP2_MODEL_NAME_SOURCE="env override"
    SIGLIP2_MODEL_NAME_ARG=(--model.siglip2_model_name "$SIGLIP2_MODEL_NAME")
else
    SIGLIP2_MODEL_NAME_EFFECTIVE="$CONFIG_SIGLIP2_MODEL_NAME"
    SIGLIP2_MODEL_NAME_SOURCE="config"
fi

INIT_SIGLIP2_ARG=()
if [ -n "${INIT_SIGLIP2+x}" ]; then
    case "$INIT_SIGLIP2" in
        true|false) ;;
        *)
            echo "[ERROR] INIT_SIGLIP2 must be 'true' or 'false' (got: $INIT_SIGLIP2)"
            exit 1
            ;;
    esac
    INIT_SIGLIP2_EFFECTIVE="$INIT_SIGLIP2"
    INIT_SIGLIP2_SOURCE="env override"
    INIT_SIGLIP2_ARG=(--model.init_siglip2 "$INIT_SIGLIP2")
else
    INIT_SIGLIP2_EFFECTIVE="$CONFIG_INIT_SIGLIP2"
    INIT_SIGLIP2_SOURCE="config"
fi

NUM_GPUS=$("$PYTHON_CMD" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
PYTHON_BIN="$(command -v "$PYTHON_CMD")"
if [ -d "$UNIVERSAL_DATA_ROOT" ]; then
    IMAGE_SHARD_COUNT=$(find "$UNIVERSAL_DATA_ROOT" -maxdepth 1 -type f -name '*.tar' | wc -l | tr -d ' ')
else
    IMAGE_SHARD_COUNT=0
fi

echo "========================================"
echo "  MAVT Stage 1 — Image Only"
echo "  Project dir:      $PROJECT_DIR"
echo "  Python:           $PYTHON_BIN"
echo "  GPUs:             $NUM_GPUS"
echo "  Dataset path:     $UNIVERSAL_DATA_ROOT"
echo "  Dataset exists:   $([ -d "$UNIVERSAL_DATA_ROOT" ] && echo yes || echo no)"
echo "  Image .tar shards: $IMAGE_SHARD_COUNT"
echo "  Config:           $STAGE1_CONFIG"
echo "  Pretrained model: $SIGLIP2_MODEL_NAME_EFFECTIVE ($SIGLIP2_MODEL_NAME_SOURCE)"
echo "  Init pretrained:  $INIT_SIGLIP2_EFFECTIVE ($INIT_SIGLIP2_SOURCE)"
echo "  Progress bar:     $CONFIG_PROGRESS_BAR (RichProgressBar config)"
echo "  Install deps:     $INSTALL_DEPS"
echo "========================================"

if [ "$INSTALL_DEPS" = "true" ]; then
    echo "[INFO] INSTALL_DEPS=true, running setup_env.sh before training..."
    bash setup_env.sh
    source .venv/bin/activate
    PYTHON_CMD="python"
    PYTHON_BIN="$(command -v "$PYTHON_CMD")"
    NUM_GPUS=$("$PYTHON_CMD" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    echo "[INFO] Active Python after install: $PYTHON_BIN"
    echo "[INFO] GPUs after install: $NUM_GPUS"
elif [ "$INSTALL_DEPS" != "false" ]; then
    echo "[ERROR] INSTALL_DEPS must be 'true' or 'false' (got: $INSTALL_DEPS)"
    exit 1
else
    echo "[INFO] INSTALL_DEPS=false, using the current environment."
fi

echo "[INFO] Starting Stage 1 training..."

"$PYTHON_CMD" train.py fit \
    --config configs/model/mavt_base.yaml \
    --config "$STAGE1_CONFIG" \
    --data.universal_data_root "$UNIVERSAL_DATA_ROOT" \
    --data.active_modalities '["image"]' \
    --data.image_resolution 256 \
    --data.batch_size 32 \
    --data.num_workers 8 \
    --data.pin_memory true \
    --model.training_stage 1 \
    "${SIGLIP2_MODEL_NAME_ARG[@]}" \
    "${INIT_SIGLIP2_ARG[@]}" \
    --model.use_lpips true \
    --model.warmup_steps 1000 \
    --model.total_steps 200000 \
    --trainer.devices "$NUM_GPUS" \
    --trainer.precision bf16-mixed \
    --trainer.max_steps 200000 \
    --trainer.log_every_n_steps 50 \
    --trainer.val_check_interval 2000 \
    --trainer.enable_progress_bar true \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project mavt \
    --trainer.logger.init_args.name stage1-image
