#!/usr/bin/env bash
# ============================================================================
# Parallel Dataset Download Wrapper for MAVT
# ============================================================================
# This script runs multiple dataset downloads in parallel by calling
# download_dataset.sh multiple times with --only flag in the background.
#
# Usage:
#   ./download_datasets_parallel.sh --data-root ./data/datasets
#
# Options:
#   --data-root DIR       Root directory for all datasets (default: ./data/datasets)
#   --num-proc  N         Parallel processes per dataset (default: 8)
#   --max-parallel N      Maximum datasets to download simultaneously (default: all)
#   --datasets LIST       Comma-separated list of datasets (default: all)
#                         Available: dfn,openimages,webvid,textvr,panda70m,objaverse,cap3d
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-./data/datasets}"
NUM_PROC=8
MAX_PARALLEL=0  # 0 means unlimited
DATASETS="webvid,textvr,openimages"  # Default: smaller datasets first

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)     DATA_ROOT="$2";     shift 2 ;;
        --num-proc)      NUM_PROC="$2";      shift 2 ;;
        --max-parallel)  MAX_PARALLEL="$2";  shift 2 ;;
        --datasets)      DATASETS="$2";      shift 2 ;;
        -h|--help)
            head -20 "$0" | tail -16
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\n\033[1;31m[ERR]\033[0m  $*" >&2; }

# ── Setup ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_dataset.sh"

if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    err "download_dataset.sh not found at: $DOWNLOAD_SCRIPT"
    exit 1
fi

if [ ! -x "$DOWNLOAD_SCRIPT" ]; then
    log "Making download_dataset.sh executable..."
    chmod +x "$DOWNLOAD_SCRIPT"
fi

mkdir -p "$DATA_ROOT"
LOG_DIR="${DATA_ROOT}/download_logs"
mkdir -p "$LOG_DIR"

log "============================================================"
log "Parallel Dataset Download"
log "============================================================"
log "Data root: $(realpath "$DATA_ROOT")"
log "Datasets: ${DATASETS}"
log "Parallel processes per dataset: ${NUM_PROC}"
log "Max parallel datasets: ${MAX_PARALLEL:-unlimited}"
log "Logs: ${LOG_DIR}"
log "============================================================"

# ── Parse dataset list ──────────────────────────────────────────────────────
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

# ── Launch downloads ────────────────────────────────────────────────────────
PIDS=()
declare -A PID_TO_DATASET

start_download() {
    local dataset="$1"
    local logfile="${LOG_DIR}/${dataset}.log"

    log "Starting download: $dataset → ${logfile}"

    "$DOWNLOAD_SCRIPT" \
        --data-root "$DATA_ROOT" \
        --num-proc "$NUM_PROC" \
        --only "$dataset" \
        > "$logfile" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    PID_TO_DATASET[$pid]="$dataset"

    log "  PID $pid: $dataset"
}

# Launch downloads (respecting max_parallel if set)
running=0
for dataset in "${DATASET_ARRAY[@]}"; do
    # Wait if we've hit the parallel limit
    if [ "$MAX_PARALLEL" -gt 0 ] && [ "$running" -ge "$MAX_PARALLEL" ]; then
        log "Waiting for a slot (max parallel: $MAX_PARALLEL)..."
        wait -n  # Wait for any job to finish
        running=$((running - 1))
    fi

    start_download "$dataset"
    running=$((running + 1))
    sleep 2  # Small delay to avoid overwhelming the system
done

log "\n============================================================"
log "All downloads launched. Waiting for completion..."
log "============================================================\n"

# ── Monitor progress ────────────────────────────────────────────────────────
while [ ${#PIDS[@]} -gt 0 ]; do
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        dataset="${PID_TO_DATASET[$pid]}"

        if ! kill -0 "$pid" 2>/dev/null; then
            # Process finished
            wait "$pid"
            exit_code=$?

            if [ $exit_code -eq 0 ]; then
                log "✓ Completed: $dataset"
            else
                warn "✗ Failed: $dataset (exit code: $exit_code)"
                warn "  Check log: ${LOG_DIR}/${dataset}.log"
            fi

            unset 'PIDS[$i]'
            unset 'PID_TO_DATASET[$pid]'
        fi
    done

    # Rebuild array to remove gaps
    PIDS=("${PIDS[@]}")

    if [ ${#PIDS[@]} -gt 0 ]; then
        echo -ne "\r  Still downloading: ${#PIDS[@]} datasets remaining...  "
        sleep 5
    fi
done

echo ""  # New line after progress

# ── Summary ─────────────────────────────────────────────────────────────────
log "\n============================================================"
log "Download Summary"
log "============================================================\n"

for dataset in "${DATASET_ARRAY[@]}"; do
    logfile="${LOG_DIR}/${dataset}.log"
    if [ -f "$logfile" ]; then
        if grep -q "Done\." "$logfile" || tail -5 "$logfile" | grep -q "→"; then
            log "  ✓ $dataset - Success"
        else
            warn "  ✗ $dataset - Check log for errors"
        fi
    else
        warn "  ? $dataset - No log file found"
    fi
done

log "\nAll parallel downloads complete!"
log "Data root: $(realpath "$DATA_ROOT")"
log "\nDisk usage:"
du -sh "${DATA_ROOT}"/* 2>/dev/null || true
log "\nLogs available at: ${LOG_DIR}"
