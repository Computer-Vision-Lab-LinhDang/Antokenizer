#!/usr/bin/env bash
# ============================================================================
# HuggingFace Dataset Downloader for MAVT
# ============================================================================
# Uses official HuggingFace CLI and Python tools for efficient downloading
# of datasets hosted on HuggingFace Hub.
#
# Prerequisites:
#   pip install huggingface_hub datasets
#   huggingface-cli login  (optional, for gated datasets)
#
# Usage:
#   ./download_hf_datasets.sh --data-root ./data/datasets
#
# Options:
#   --data-root DIR       Root directory for datasets (default: ./data/datasets)
#   --cache-dir DIR       HuggingFace cache directory (default: ~/.cache/huggingface)
#   --parallel N          Number of parallel downloads (default: 4)
#   --token TOKEN         HuggingFace token (or use huggingface-cli login)
#   --datasets LIST       Comma-separated dataset list (default: all HF datasets)
#                         Available: dfn,webvid,cap3d,textocr
#   --streaming           Use streaming mode (no full download, saves space)
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-./data/datasets}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
PARALLEL=4
HF_TOKEN="${HF_TOKEN:-}"
STREAMING=false
DATASETS="dfn,webvid,cap3d"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)     DATA_ROOT="$2";     shift 2 ;;
        --cache-dir)     CACHE_DIR="$2";     shift 2 ;;
        --parallel)      PARALLEL="$2";      shift 2 ;;
        --token)         HF_TOKEN="$2";      shift 2 ;;
        --datasets)      DATASETS="$2";      shift 2 ;;
        --streaming)     STREAMING=true;     shift   ;;
        -h|--help)
            head -25 "$0" | tail -21
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo -e "\n\033[1;34m[INFO]\033[0m $*"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\n\033[1;31m[ERR]\033[0m  $*" >&2; }

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        err "Required command not found: $1"
        err "Install hint: $2"
        return 1
    fi
}

# ── Setup ───────────────────────────────────────────────────────────────────
mkdir -p "$DATA_ROOT"
export HF_HOME="$CACHE_DIR"

log "============================================================"
log "HuggingFace Dataset Downloader for MAVT"
log "============================================================"
log "Data root: $(realpath "$DATA_ROOT")"
log "HF cache: $CACHE_DIR"
log "Parallel: $PARALLEL"
log "Streaming: $STREAMING"
log "Datasets: $DATASETS"
log "============================================================"

# Check prerequisites
check_cmd python3 "sudo apt-get install -y python3"
check_cmd pip "sudo apt-get install -y python3-pip"

# Check/install HuggingFace tools
log "Checking HuggingFace tools..."
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    log "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

if ! python3 -c "import datasets" 2>/dev/null; then
    log "Installing datasets library..."
    pip install -q datasets
fi

if ! command -v huggingface-cli &>/dev/null; then
    warn "huggingface-cli not found in PATH, but huggingface_hub is installed"
fi

# Setup authentication if token provided
if [ -n "$HF_TOKEN" ]; then
    log "Setting up HuggingFace authentication..."
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)"
fi

# ── Download functions ──────────────────────────────────────────────────────

download_dfn() {
    # DFN-2B: apf1/datafilteringnetworks_2b
    log "────────────────────────────────────────────────────────"
    log "Downloading DFN-2B (apf1/datafilteringnetworks_2b)"
    log "────────────────────────────────────────────────────────"

    DFN_DIR="${DATA_ROOT}/dfn_2b"
    mkdir -p "$DFN_DIR"

    python3 <<EOF
from huggingface_hub import snapshot_download
import os

repo_id = "apf1/datafilteringnetworks_2b"
local_dir = "${DFN_DIR}"

print(f"Downloading {repo_id} to {local_dir}...")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=${PARALLEL},
        resume_download=True,
    )
    print(f"✓ DFN-2B downloaded to {local_dir}")
except Exception as e:
    print(f"✗ Error downloading DFN-2B: {e}")
    exit(1)
EOF

    log "DFN-2B → ${DFN_DIR}"
}

download_webvid() {
    # WebVid-10M: TempoFunk/webvid-10M
    log "────────────────────────────────────────────────────────"
    log "Downloading WebVid-10M (TempoFunk/webvid-10M)"
    log "────────────────────────────────────────────────────────"

    WEBVID_DIR="${DATA_ROOT}/webvid_10m"
    mkdir -p "$WEBVID_DIR"

    if [ "$STREAMING" = true ]; then
        log "Using streaming mode - metadata only"
        python3 <<EOF
from datasets import load_dataset
import json

dataset = load_dataset("TempoFunk/webvid-10M", streaming=True, split="train")
print(f"✓ WebVid-10M loaded in streaming mode")

# Save sample for verification
sample = next(iter(dataset.take(1)))
with open("${WEBVID_DIR}/sample.json", "w") as f:
    json.dump(str(sample), f)
print(f"Sample saved to ${WEBVID_DIR}/sample.json")
EOF
    else
        python3 <<EOF
from huggingface_hub import snapshot_download

repo_id = "TempoFunk/webvid-10M"
local_dir = "${WEBVID_DIR}/metadata"

print(f"Downloading metadata from {repo_id}...")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.csv", "*.json", "*.parquet"],
        max_workers=${PARALLEL},
        resume_download=True,
    )
    print(f"✓ WebVid-10M metadata downloaded to {local_dir}")
    print("Note: Use video2dataset to download actual videos from URLs in metadata")
except Exception as e:
    print(f"✗ Error downloading WebVid-10M: {e}")
    exit(1)
EOF
    fi

    log "WebVid-10M → ${WEBVID_DIR}"
}

download_cap3d() {
    # Cap3D: tiange/Cap3D
    log "────────────────────────────────────────────────────────"
    log "Downloading Cap3D (tiange/Cap3D)"
    log "────────────────────────────────────────────────────────"

    CAP3D_DIR="${DATA_ROOT}/cap3d"
    mkdir -p "$CAP3D_DIR"

    python3 <<EOF
from huggingface_hub import snapshot_download

repo_id = "tiange/Cap3D"
local_dir = "${CAP3D_DIR}"

print(f"Downloading {repo_id} to {local_dir}...")

try:
    # Download captions and README first (small)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.csv", "*.md", "*.txt"],
        max_workers=${PARALLEL},
        resume_download=True,
    )
    print(f"✓ Cap3D captions downloaded")

    # Download point clouds (large - optional)
    # Uncomment to download:
    # snapshot_download(
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     local_dir=local_dir,
    #     allow_patterns=["PointCloud_pt_zips/*.zip"],
    #     max_workers=${PARALLEL},
    #     resume_download=True,
    # )
    print(f"✓ Cap3D downloaded to {local_dir}")
    print("Note: Point clouds not downloaded (uncomment in script to enable)")
except Exception as e:
    print(f"✗ Error downloading Cap3D: {e}")
    exit(1)
EOF

    log "Cap3D → ${CAP3D_DIR}"
}

download_textocr() {
    # TextOCR: facebook/textocr (similar to TextVR)
    log "────────────────────────────────────────────────────────"
    log "Downloading TextOCR (facebook/textocr)"
    log "────────────────────────────────────────────────────────"

    TEXTOCR_DIR="${DATA_ROOT}/textocr"
    mkdir -p "$TEXTOCR_DIR"

    python3 <<EOF
from datasets import load_dataset

dataset_name = "facebook/textocr"
local_dir = "${TEXTOCR_DIR}"

print(f"Downloading {dataset_name}...")

try:
    dataset = load_dataset(dataset_name, cache_dir="${CACHE_DIR}")
    dataset.save_to_disk(local_dir)
    print(f"✓ TextOCR downloaded to {local_dir}")
except Exception as e:
    print(f"✗ Error downloading TextOCR: {e}")
    exit(1)
EOF

    log "TextOCR → ${TEXTOCR_DIR}"
}

download_objaverse_hf() {
    # Objaverse metadata from HF
    log "────────────────────────────────────────────────────────"
    log "Downloading Objaverse metadata (allenai/objaverse)"
    log "────────────────────────────────────────────────────────"

    OBJAVERSE_DIR="${DATA_ROOT}/objaverse_xl"
    mkdir -p "$OBJAVERSE_DIR/metadata"

    python3 <<EOF
from huggingface_hub import snapshot_download

repo_id = "allenai/objaverse"
local_dir = "${OBJAVERSE_DIR}/metadata"

print(f"Downloading metadata from {repo_id}...")

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["*.json", "*.csv", "*.parquet"],
        max_workers=${PARALLEL},
        resume_download=True,
    )
    print(f"✓ Objaverse metadata downloaded to {local_dir}")
    print("Note: Use objaverse Python library to download actual 3D objects")
except Exception as e:
    print(f"✗ Error downloading Objaverse metadata: {e}")
    exit(1)
EOF

    log "Objaverse metadata → ${OBJAVERSE_DIR}"
}

# ── Main execution ──────────────────────────────────────────────────────────

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

log "\nStarting downloads...\n"

for dataset in "${DATASET_ARRAY[@]}"; do
    case "$dataset" in
        dfn)
            download_dfn
            ;;
        webvid)
            download_webvid
            ;;
        cap3d)
            download_cap3d
            ;;
        textocr)
            download_textocr
            ;;
        objaverse)
            download_objaverse_hf
            ;;
        *)
            warn "Unknown dataset: $dataset"
            ;;
    esac
done

# ── Summary ─────────────────────────────────────────────────────────────────
log "\n============================================================"
log "Download Summary"
log "============================================================\n"

for dataset in "${DATASET_ARRAY[@]}"; do
    case "$dataset" in
        dfn)        dir="${DATA_ROOT}/dfn_2b" ;;
        webvid)     dir="${DATA_ROOT}/webvid_10m" ;;
        cap3d)      dir="${DATA_ROOT}/cap3d" ;;
        textocr)    dir="${DATA_ROOT}/textocr" ;;
        objaverse)  dir="${DATA_ROOT}/objaverse_xl" ;;
        *)          continue ;;
    esac

    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        log "  ✓ $dataset → $dir ($size)"
    else
        warn "  ✗ $dataset → $dir (not found)"
    fi
done

log "\nAll HuggingFace downloads complete!"
log "Data root: $(realpath "$DATA_ROOT")"
log "HF cache: $CACHE_DIR"

log "\nNext steps:"
log "  1. For WebVid videos: Use video2dataset with metadata CSVs"
log "  2. For Objaverse 3D: Use objaverse Python library"
log "  3. For Open Images: Use original download_dataset.sh"
log "\nStart training:"
log "  python -m train.example_stage_training --data-root $DATA_ROOT"
