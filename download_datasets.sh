#!/usr/bin/env bash
# ============================================================================
# MAVT Dataset Download Script
# ============================================================================
# Downloads seven datasets for MAVT training / validation / testing:
#   1. apf1/datafilteringnetworks_2b  (image-text pairs via DataComp CommonPool)
#   2. Open Images V7                 (image classification / detection)
#   3. TempoFunk/webvid-10M           (video-text pairs)
#   4. TextVR                         (text-video retrieval benchmark)
#   5. Panda-70M                      (70M video-caption pairs)
#   6. allenai/objaverse-xl           (10M+ 3D objects)
#   7. tiange/Cap3D                   (3D captions for Objaverse/ShapeNet/ABO)
#
# Prerequisites:
#   - Python 3.8+
#   - pip install huggingface_hub datasets img2dataset gdown yt-dlp objaverse
#   - git-lfs               (for HuggingFace repos)
#   - aws CLI                (for Open Images CVDF download)
#   - Sufficient disk space  (~20TB+ for full datasets)
#
# Usage:
#   chmod +x scripts/download_datasets.sh
#   ./scripts/download_datasets.sh [OPTIONS]
#
# Options:
#   --data-root DIR       Root directory for all datasets (default: ./data/datasets)
#   --num-proc  N         Parallel download processes    (default: 8)
#   --skip-dfn            Skip DFN-2B
#   --skip-openimages     Skip Open Images V7
#   --skip-webvid         Skip WebVid-10M
#   --skip-textvr         Skip TextVR
#   --skip-panda70m       Skip Panda-70M
#   --skip-objaverse      Skip Objaverse-XL
#   --skip-cap3d          Skip Cap3D
#   --only DATASET        Download only one (dfn|openimages|webvid|textvr|panda70m|objaverse|cap3d)
# ============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-./data/datasets}"
NUM_PROC="${NUM_DOWNLOAD_PROC:-8}"
SKIP_DFN=false
SKIP_OPENIMAGES=false
SKIP_WEBVID=false
SKIP_TEXTVR=false
SKIP_PANDA70M=false
SKIP_OBJAVERSE=false
SKIP_CAP3D=false

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)       DATA_ROOT="$2";       shift 2 ;;
        --num-proc)        NUM_PROC="$2";        shift 2 ;;
        --skip-dfn)        SKIP_DFN=true;        shift   ;;
        --skip-openimages) SKIP_OPENIMAGES=true; shift   ;;
        --skip-webvid)     SKIP_WEBVID=true;     shift   ;;
        --skip-textvr)     SKIP_TEXTVR=true;     shift   ;;
        --skip-panda70m)   SKIP_PANDA70M=true;   shift   ;;
        --skip-objaverse)  SKIP_OBJAVERSE=true;  shift   ;;
        --skip-cap3d)      SKIP_CAP3D=true;      shift   ;;
        --only)
            SKIP_DFN=true; SKIP_OPENIMAGES=true; SKIP_WEBVID=true
            SKIP_TEXTVR=true; SKIP_PANDA70M=true; SKIP_OBJAVERSE=true; SKIP_CAP3D=true
            case "$2" in
                dfn)        SKIP_DFN=false ;;
                openimages) SKIP_OPENIMAGES=false ;;
                webvid)     SKIP_WEBVID=false ;;
                textvr)     SKIP_TEXTVR=false ;;
                panda70m)   SKIP_PANDA70M=false ;;
                objaverse)  SKIP_OBJAVERSE=false ;;
                cap3d)      SKIP_CAP3D=false ;;
                *) echo "Unknown dataset: $2"; exit 1 ;;
            esac
            shift 2 ;;
        -h|--help)
            head -30 "$0" | tail -25
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

# Google Drive download helper (uses gdown or curl fallback)
gdrive_download() {
    local FILE_ID="$1"
    local OUTPUT="$2"

    if [ -f "$OUTPUT" ]; then
        log "Already exists, skipping: $OUTPUT"
        return 0
    fi

    if command -v gdown &>/dev/null; then
        gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT" --fuzzy 2>&1 || {
            warn "gdown failed for ${FILE_ID}, trying curl fallback..."
            curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}&confirm=t" -o "$OUTPUT" 2>&1 || true
        }
    else
        log "gdown not found, using curl (may fail on large files)..."
        curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}&confirm=t" -o "$OUTPUT" 2>&1 || true
    fi
}

mkdir -p "$DATA_ROOT"
log "Data root: $(realpath "$DATA_ROOT")"
log "Parallel processes: ${NUM_PROC}"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  1. apf1/datafilteringnetworks_2b  (DFN-2B)                             ║
# ║  Splits: train only (filtering indices for CommonPool)                   ║
# ║  Ref: https://arxiv.org/abs/2309.17425                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_DFN" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "1/7  DFN-2B  (apf1/datafilteringnetworks_2b)"
    log "════════════════════════════════════════════════════════════════"

    DFN_DIR="${DATA_ROOT}/dfn_2b"
    mkdir -p "$DFN_DIR"

    # 1a. Clone DFN index files from HuggingFace
    check_cmd git     "brew install git / apt install git"
    check_cmd git-lfs "brew install git-lfs && git lfs install"

    DFN_INDEX_DIR="${DFN_DIR}/indices"
    if [ ! -d "${DFN_INDEX_DIR}/.git" ]; then
        log "Cloning DFN-2B indices..."
        git clone https://huggingface.co/datasets/apf1/datafilteringnetworks_2b "${DFN_INDEX_DIR}"
    else
        log "DFN-2B indices already cloned, pulling..."
        git -C "${DFN_INDEX_DIR}" pull
    fi

    # 1b. Clone DataComp tools
    DATACOMP_DIR="${DFN_DIR}/datacomp"
    if [ ! -d "${DATACOMP_DIR}/.git" ]; then
        log "Cloning DataComp repository..."
        git clone https://github.com/mlfoundations/datacomp.git "${DATACOMP_DIR}"
    fi

    log "Installing DataComp dependencies..."
    pip install -r "${DATACOMP_DIR}/requirements.txt" 2>/dev/null || warn "Some deps may have failed."

    # 1c. Download CommonPool metadata
    COMMONPOOL_DIR="${DFN_DIR}/commonpool"
    mkdir -p "$COMMONPOOL_DIR"

    log "Downloading CommonPool metadata (xlarge)..."
    log "  NOTE: Adjust --scale (small/medium/large/xlarge) to control size."
    python "${DATACOMP_DIR}/download_upstream.py" \
        --scale xlarge \
        --data_dir "${COMMONPOOL_DIR}" \
        --metadata_only 2>&1 || warn "CommonPool metadata download may have issues."

    # 1d. Reshard with DFN indices
    DFN_NPY=$(find "${DFN_INDEX_DIR}" -name "*.npy" -type f | head -1)
    if [ -n "$DFN_NPY" ]; then
        log "Resharding CommonPool with DFN indices..."
        python "${DATACOMP_DIR}/reshard.py" \
            --input_dir "${COMMONPOOL_DIR}" \
            --output_dir "${DFN_DIR}/resharded" \
            --sort_column uid \
            --index_file "${DFN_NPY}" 2>&1 || warn "Resharding had issues."
    else
        warn "No .npy index file found in ${DFN_INDEX_DIR}. Check manually."
    fi

    # 1e. Download images with img2dataset
    if command -v img2dataset &>/dev/null; then
        log "Downloading images with img2dataset..."
        img2dataset \
            --url_list "${DFN_DIR}/resharded" \
            --input_format "webdataset" \
            --output_folder "${DFN_DIR}/images" \
            --output_format "webdataset" \
            --processes_count "${NUM_PROC}" \
            --thread_count 64 \
            --image_size 256 \
            --resize_mode "center_crop" \
            --save_additional_columns '["txt","json"]' 2>&1 || warn "img2dataset had issues."
    else
        warn "img2dataset not found. Install: pip install img2dataset"
    fi

    log "DFN-2B → ${DFN_DIR}"
else
    log "Skipping DFN-2B (--skip-dfn)"
fi


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  2. Open Images V7                                                      ║
# ║  Splits: train (~9M imgs), validation (~42K), test (~125K)              ║
# ║  Ref: https://storage.googleapis.com/openimages/web/download_v7.html    ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_OPENIMAGES" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "2/7  Open Images V7"
    log "════════════════════════════════════════════════════════════════"

    OI_DIR="${DATA_ROOT}/open_images_v7"
    mkdir -p "${OI_DIR}"/{images/{train,validation,test},annotations,metadata}

    # 2a. Download images via CVDF S3 mirror (no credentials needed)
    check_cmd aws "pip install awscli"

    log "Downloading TRAIN images (~560GB, resumable)..."
    aws s3 --no-sign-request sync s3://open-images-dataset/train      "${OI_DIR}/images/train"      2>&1 || warn "Train sync issues."
    log "Downloading VALIDATION images..."
    aws s3 --no-sign-request sync s3://open-images-dataset/validation  "${OI_DIR}/images/validation"  2>&1 || warn "Val sync issues."
    log "Downloading TEST images..."
    aws s3 --no-sign-request sync s3://open-images-dataset/test        "${OI_DIR}/images/test"        2>&1 || warn "Test sync issues."

    # 2b. Annotations — image-level labels (V7, all splits)
    ANNO_DIR="${OI_DIR}/annotations"
    META_DIR="${OI_DIR}/metadata"

    log "Downloading annotations (all splits)..."

    # Human & machine image labels (train / val / test)
    for TYPE in human machine; do
        for SPLIT in train val test; do
            wget -nc -q -P "$ANNO_DIR" \
                "https://storage.googleapis.com/openimages/v7/oidv7-${SPLIT}-annotations-${TYPE}-imagelabels.csv" 2>/dev/null || true
        done
    done

    # Bounding boxes (train / val / test)
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv  2>/dev/null || true
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv   2>/dev/null || true
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv         2>/dev/null || true

    # Point labels (train / val / test)
    for SPLIT in train val test; do
        wget -nc -q -P "$ANNO_DIR" \
            "https://storage.googleapis.com/openimages/v7/oidv7-${SPLIT}-annotations-point-labels.csv" 2>/dev/null || true
    done

    # Visual relationships (train / val / test)
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv       2>/dev/null || true
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v6/oidv6-validation-annotations-vrd.csv  2>/dev/null || true
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv        2>/dev/null || true

    # Segmentation annotations (train / val / test)
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv       2>/dev/null || true
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv  2>/dev/null || true
    wget -nc -q -P "$ANNO_DIR" https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv        2>/dev/null || true

    # Image metadata with rotation (train / val / test)
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv    2>/dev/null || true
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv  2>/dev/null || true
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv              2>/dev/null || true

    # Class metadata
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv          2>/dev/null || true
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv  2>/dev/null || true
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/v7/oidv7-classes-trainable.txt           2>/dev/null || true
    wget -nc -q -P "$META_DIR" https://storage.googleapis.com/openimages/v7/oidv7-classes-segmentation.txt        2>/dev/null || true

    # 2c. Segmentation masks (train / val / test — each split 16 zip files)
    log "Downloading segmentation masks..."
    MASK_DIR="${OI_DIR}/masks"
    mkdir -p "${MASK_DIR}"/{train,validation,test}

    for HEX in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
        wget -nc -q -P "${MASK_DIR}/train"      "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-${HEX}.zip"            2>/dev/null || true
        wget -nc -q -P "${MASK_DIR}/validation"  "https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-${HEX}.zip"  2>/dev/null || true
        wget -nc -q -P "${MASK_DIR}/test"        "https://storage.googleapis.com/openimages/v5/test-masks/test-masks-${HEX}.zip"              2>/dev/null || true
    done

    log "Unzipping masks..."
    for SPLIT_DIR in "${MASK_DIR}/train" "${MASK_DIR}/validation" "${MASK_DIR}/test"; do
        for ZIP_FILE in "${SPLIT_DIR}"/*.zip; do
            [ -f "$ZIP_FILE" ] && unzip -n -q "$ZIP_FILE" -d "$SPLIT_DIR" && rm -f "$ZIP_FILE"
        done
    done

    log "Open Images V7 → ${OI_DIR}"
else
    log "Skipping Open Images V7 (--skip-openimages)"
fi


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  3. TempoFunk/webvid-10M                                                ║
# ║  Splits: train (~10.7M video-text pairs, metadata CSV)                  ║
# ║  Ref: https://huggingface.co/datasets/TempoFunk/webvid-10M              ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_WEBVID" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "3/7  WebVid-10M  (TempoFunk/webvid-10M)"
    log "════════════════════════════════════════════════════════════════"

    WEBVID_DIR="${DATA_ROOT}/webvid_10m"
    mkdir -p "$WEBVID_DIR"
    check_cmd python "Install Python 3.8+"

    # 3a. Download metadata CSVs via huggingface_hub
    log "Downloading WebVid-10M metadata (~2.9GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TempoFunk/webvid-10M',
    repo_type='dataset',
    local_dir='${WEBVID_DIR}/metadata',
    local_dir_use_symlinks=False,
)
print('Metadata download complete.')
" 2>&1 || {
        warn "huggingface_hub failed. Trying git clone..."
        check_cmd git-lfs "brew install git-lfs && git lfs install"
        if [ ! -d "${WEBVID_DIR}/metadata/.git" ]; then
            git clone https://huggingface.co/datasets/TempoFunk/webvid-10M "${WEBVID_DIR}/metadata"
        else
            git -C "${WEBVID_DIR}/metadata" pull
        fi
    }

    # 3b. Download actual videos with video2dataset
    if command -v video2dataset &>/dev/null; then
        log "Downloading videos with video2dataset..."
        CSV_FILES=$(find "${WEBVID_DIR}/metadata/data" -name "*.csv" -type f 2>/dev/null | sort)
        if [ -n "$CSV_FILES" ]; then
            for CSV_FILE in $CSV_FILES; do
                BASENAME=$(basename "$CSV_FILE" .csv)
                log "  Processing ${BASENAME}..."
                video2dataset \
                    --url_list "$CSV_FILE" \
                    --url_col "contentUrl" \
                    --caption_col "name" \
                    --output_folder "${WEBVID_DIR}/videos/${BASENAME}" \
                    --output_format "webdataset" \
                    --processes_count "${NUM_PROC}" \
                    --thread_count 32 \
                    --encode_formats '{"video": "mp4"}' \
                    --timeout 30 \
                    --retries 2 2>&1 || warn "Issues processing ${BASENAME}."
            done
        else
            warn "No CSV files found in ${WEBVID_DIR}/metadata/data"
        fi
    else
        warn "video2dataset not found. Install: pip install video2dataset"
        log "  Metadata downloaded to ${WEBVID_DIR}/metadata"
    fi

    log "WebVid-10M → ${WEBVID_DIR}"
else
    log "Skipping WebVid-10M (--skip-webvid)"
fi


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  4. TextVR  (Text-Video Retrieval benchmark)                            ║
# ║  Splits: train / val / test (inside the zip)                            ║
# ║  Source: Google Drive file 1RZefU1XqODCt2NH68P0V_DMAOiQjRlSn            ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_TEXTVR" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "4/7  TextVR  (TextVR_data.zip from Google Drive)"
    log "════════════════════════════════════════════════════════════════"

    TEXTVR_DIR="${DATA_ROOT}/textvr"
    mkdir -p "$TEXTVR_DIR"

    TEXTVR_ZIP="${TEXTVR_DIR}/TextVR_data.zip"
    TEXTVR_FILE_ID="1RZefU1XqODCt2NH68P0V_DMAOiQjRlSn"

    log "Downloading TextVR_data.zip from Google Drive..."
    gdrive_download "$TEXTVR_FILE_ID" "$TEXTVR_ZIP"

    if [ -f "$TEXTVR_ZIP" ]; then
        log "Extracting TextVR_data.zip..."
        unzip -n -q "$TEXTVR_ZIP" -d "$TEXTVR_DIR" 2>&1 || warn "Extraction had issues."
        log "  Contents:"
        ls -lh "$TEXTVR_DIR"/ 2>/dev/null || true
    else
        warn "TextVR_data.zip not found after download. Check Google Drive access."
    fi

    log "TextVR → ${TEXTVR_DIR}"
else
    log "Skipping TextVR (--skip-textvr)"
fi


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  5. Panda-70M                                                           ║
# ║  Splits: train-full (~70M), train-10M, train-2M, validation, test       ║
# ║  Ref: https://github.com/snap-research/Panda-70M  (CVPR 2024)          ║
# ║  CSV metadata on Google Drive, videos via video2dataset + yt-dlp        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_PANDA70M" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "5/7  Panda-70M"
    log "════════════════════════════════════════════════════════════════"

    PANDA_DIR="${DATA_ROOT}/panda_70m"
    mkdir -p "${PANDA_DIR}"/{csv,videos}

    # ── 5a. Download CSV metadata from Google Drive ──────────────────────
    # File IDs from the official repo:
    #   training_full:  1pbh8W3qgst9CD7nlPhsH9wmUSWjQlGdW
    #   training_10M:   1LLOFeYw9nZzjT5aA1Wj4oGi5yHUzwSk5
    #   training_2M:    1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB
    #   validation:     1uHR5iXS3Sftzw6AwEhyZ9RefipNzBAzt
    #   testing:        1BZ9L-157Au1TwmkwlJV8nZQvSRLIiFhq

    declare -A PANDA_FILES=(
        ["training_full"]="1pbh8W3qgst9CD7nlPhsH9wmUSWjQlGdW"
        ["training_10m"]="1LLOFeYw9nZzjT5aA1Wj4oGi5yHUzwSk5"
        ["training_2m"]="1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB"
        ["validation"]="1uHR5iXS3Sftzw6AwEhyZ9RefipNzBAzt"
        ["testing"]="1BZ9L-157Au1TwmkwlJV8nZQvSRLIiFhq"
    )

    log "Downloading Panda-70M CSV metadata (train-full, train-10M, train-2M, val, test)..."
    for SPLIT_NAME in training_full training_10m training_2m validation testing; do
        FILE_ID="${PANDA_FILES[$SPLIT_NAME]}"
        OUTPUT_FILE="${PANDA_DIR}/csv/panda70m_${SPLIT_NAME}.csv.zip"
        log "  Downloading ${SPLIT_NAME}..."
        gdrive_download "$FILE_ID" "$OUTPUT_FILE"
    done

    # Unzip any compressed CSVs (training files are zipped)
    log "Extracting CSV files..."
    for ZIP_FILE in "${PANDA_DIR}/csv"/*.zip; do
        if [ -f "$ZIP_FILE" ]; then
            unzip -n -q "$ZIP_FILE" -d "${PANDA_DIR}/csv" 2>&1 || true
        fi
    done

    # ── 5b. Clone Panda-70M repo (for video2dataset config) ─────────────
    PANDA_REPO="${PANDA_DIR}/repo"
    if [ ! -d "${PANDA_REPO}/.git" ]; then
        log "Cloning Panda-70M repository..."
        git clone https://github.com/snap-research/Panda-70M.git "${PANDA_REPO}"
    fi

    # Install their custom video2dataset fork
    log "Installing Panda-70M video2dataset fork..."
    pip install -e "${PANDA_REPO}/dataset_dataloading/video2dataset" 2>&1 || warn "video2dataset install had issues."

    # ── 5c. Download videos with video2dataset ───────────────────────────
    if command -v video2dataset &>/dev/null; then
        log "Downloading videos for each split..."

        PANDA_CONFIG="${PANDA_REPO}/dataset_dataloading/video2dataset/video2dataset/configs/panda70m.yaml"
        # Fallback if the config file doesn't exist at that path
        if [ ! -f "$PANDA_CONFIG" ]; then
            PANDA_CONFIG=$(find "${PANDA_REPO}" -name "panda*70*m*.yaml" -type f | head -1)
        fi

        for CSV_FILE in "${PANDA_DIR}/csv"/*.csv; do
            [ ! -f "$CSV_FILE" ] && continue
            BASENAME=$(basename "$CSV_FILE" .csv)
            SPLIT_OUT="${PANDA_DIR}/videos/${BASENAME}"
            mkdir -p "$SPLIT_OUT"

            log "  Downloading videos: ${BASENAME}..."
            if [ -n "$PANDA_CONFIG" ] && [ -f "$PANDA_CONFIG" ]; then
                video2dataset \
                    --url_list "$CSV_FILE" \
                    --url_col "url" \
                    --caption_col "caption" \
                    --clip_col "timestamp" \
                    --output_folder "$SPLIT_OUT" \
                    --save_additional_columns "[matching_score,desirable_filtering,shot_boundary_detection]" \
                    --config "$PANDA_CONFIG" 2>&1 || warn "Issues downloading ${BASENAME}."
            else
                video2dataset \
                    --url_list "$CSV_FILE" \
                    --url_col "url" \
                    --caption_col "caption" \
                    --clip_col "timestamp" \
                    --output_folder "$SPLIT_OUT" \
                    --output_format "webdataset" \
                    --processes_count "${NUM_PROC}" 2>&1 || warn "Issues downloading ${BASENAME}."
            fi
        done
    else
        warn "video2dataset not found. CSVs are downloaded; install video2dataset to fetch videos."
        warn "  cd ${PANDA_REPO}/dataset_dataloading/video2dataset && pip install -e ."
    fi

    log "Panda-70M → ${PANDA_DIR}"
else
    log "Skipping Panda-70M (--skip-panda70m)"
fi


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  6. Objaverse-XL  (allenai/objaverse-xl)                                ║
# ║  10M+ 3D objects from Sketchfab, GitHub, Thingiverse, etc.              ║
# ║  Downloaded via the `objaverse` Python package                          ║
# ║  Ref: https://arxiv.org/abs/2307.05663                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_OBJAVERSE" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "6/7  Objaverse-XL  (allenai/objaverse-xl)"
    log "════════════════════════════════════════════════════════════════"

    OBJ_DIR="${DATA_ROOT}/objaverse_xl"
    mkdir -p "${OBJ_DIR}"/{objects,metadata}
    check_cmd python "Install Python 3.8+"

    # 6a. Install the objaverse Python package
    log "Installing objaverse package..."
    pip install objaverse 2>&1 || warn "objaverse install had issues."

    # 6b. Download Objaverse 1.0 core (~800K objects) via Python API
    log "Downloading Objaverse 1.0 annotations and UIDs..."
    python -c "
import objaverse
import json, os

out_dir = '${OBJ_DIR}/metadata'

# Get all UIDs
uids = objaverse.load_uids()
print(f'Total Objaverse 1.0 UIDs: {len(uids)}')

# Save UID list
with open(os.path.join(out_dir, 'objaverse_uids.json'), 'w') as f:
    json.dump(uids, f)

# Get annotations (names, tags, categories, etc.)
annotations = objaverse.load_annotations(uids)
with open(os.path.join(out_dir, 'objaverse_annotations.json'), 'w') as f:
    json.dump(annotations, f, default=str)
print('Annotations saved.')    
" 2>&1 || warn "Objaverse metadata download had issues."

    # 6c. Download 3D objects (GLB files) — this is very large
    log "Downloading Objaverse 1.0 3D objects (GLB files)..."
    log "  NOTE: This downloads ~800K GLB files. Very large and slow."
    python -c "
import objaverse
import os

# Download all objects (downloads to ~/.objaverse by default)
# We use processes for parallel download
objects = objaverse.load_objects(
    uids=objaverse.load_uids(),
    download_processes=${NUM_PROC},
)
print(f'Downloaded {len(objects)} objects.')  

# Create a symlink for convenience
src = os.path.expanduser('~/.objaverse')
dst = '${OBJ_DIR}/objects'
if os.path.isdir(src) and not os.path.islink(dst):
    if not os.listdir(dst):  # empty dir
        os.rmdir(dst)
        os.symlink(src, dst)
        print(f'Symlinked {src} → {dst}')  
" 2>&1 || warn "Object download had issues. You can resume later."

    # 6d. Objaverse-XL metadata (full 10M+ index)
    log "Downloading Objaverse-XL alignment annotations..."
    python -c "
import objaverse.xl as oxl
import json, os

out_dir = '${OBJ_DIR}/metadata'

# Sketchfab annotations (the largest source)
sketchfab = oxl.get_annotations(download_dir=out_dir + '/xl_annotations')
print(f'Objaverse-XL Sketchfab annotations: {len(sketchfab)} entries')
" 2>&1 || warn "Objaverse-XL metadata had issues."

    log "Objaverse-XL → ${OBJ_DIR}"
else
    log "Skipping Objaverse-XL (--skip-objaverse)"
fi


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  7. Cap3D  (tiange/Cap3D)                                               ║
# ║  3D captions for Objaverse, Objaverse-XL, ABO, ShapeNet                 ║
# ║  Includes: captions CSV, point clouds (.pt/.ply), rendered images       ║
# ║  Ref: https://arxiv.org/abs/2306.07279                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
if [ "$SKIP_CAP3D" = false ]; then
    log "════════════════════════════════════════════════════════════════"
    log "7/7  Cap3D  (tiange/Cap3D)"
    log "════════════════════════════════════════════════════════════════"

    CAP3D_DIR="${DATA_ROOT}/cap3d"
    mkdir -p "${CAP3D_DIR}"/{captions,pointclouds,rendered_images}
    check_cmd python "Install Python 3.8+"

    # 7a. Download caption CSV files
    log "Downloading Cap3D caption files..."
    HF_CAP3D="https://huggingface.co/datasets/tiange/Cap3D/resolve/main"

    # Objaverse captions (full, ~255MB)
    wget -nc -q -P "${CAP3D_DIR}/captions" \
        "${HF_CAP3D}/Cap3D_automated_Objaverse_full.csv" 2>/dev/null || true
    # ABO captions (~3.4MB)
    wget -nc -q -P "${CAP3D_DIR}/captions" \
        "${HF_CAP3D}/Cap3D_automated_ABO.csv" 2>/dev/null || true
    # ShapeNet captions (~19.5MB)
    wget -nc -q -P "${CAP3D_DIR}/captions" \
        "${HF_CAP3D}/Cap3D_automated_ShapeNet.csv" 2>/dev/null || true

    log "  Caption files downloaded."

    # 7b. Download point clouds (.pt format, zipped)
    log "Downloading Cap3D point clouds (Objaverse)..."
    log "  NOTE: Multiple zip files, each several GB."

    python -c "
from huggingface_hub import HfApi, hf_hub_download
import os

api = HfApi()
repo_files = api.list_repo_files('tiange/Cap3D', repo_type='dataset')

# Download point cloud zips
pc_files = [f for f in repo_files if f.startswith('PointCloud_pt_zips/') and f.endswith('.zip')]
print(f'Found {len(pc_files)} point cloud zip files')  

for f in pc_files:
    print(f'  Downloading {f}...')  
    hf_hub_download(
        repo_id='tiange/Cap3D',
        filename=f,
        repo_type='dataset',
        local_dir='${CAP3D_DIR}/pointclouds',
    )
print('Point cloud downloads complete.')
" 2>&1 || warn "Point cloud download had issues."

    # 7c. Download rendered images (optional, very large)
    log "Downloading Cap3D rendered images (Objaverse)..."
    log "  NOTE: This is extremely large (100s of GB). Downloading index only."
    log "  To download all rendered images, use:"
    log "    huggingface-cli download tiange/Cap3D --include 'RenderedImage_perobj_zips/*' --repo-type dataset"

    # Download just the rendered image README/index for reference
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='tiange/Cap3D',
    filename='RenderedImage_perobj_zips/README.md',
    repo_type='dataset',
    local_dir='${CAP3D_DIR}/rendered_images',
)
print('Rendered images index downloaded.')
" 2>&1 || warn "Rendered image index download had issues."

    # 7d. Download point clouds for ABO and ShapeNet
    log "Downloading Cap3D point clouds (ABO & ShapeNet)..."
    python -c "
from huggingface_hub import HfApi, hf_hub_download
import os

api = HfApi()
repo_files = api.list_repo_files('tiange/Cap3D', repo_type='dataset')

# ABO point clouds
abo_files = [f for f in repo_files if f.startswith('PointCloud_zips_ABO/') and f.endswith('.zip')]
for f in abo_files:
    print(f'  Downloading {f}...')
    hf_hub_download(repo_id='tiange/Cap3D', filename=f, repo_type='dataset', local_dir='${CAP3D_DIR}/pointclouds')

# ShapeNet point clouds
sn_files = [f for f in repo_files if f.startswith('PointCloud_zips_ShapeNet/') and f.endswith('.zip')]
for f in sn_files:
    print(f'  Downloading {f}...')
    hf_hub_download(repo_id='tiange/Cap3D', filename=f, repo_type='dataset', local_dir='${CAP3D_DIR}/pointclouds')

print('ABO & ShapeNet point clouds complete.')
" 2>&1 || warn "ABO/ShapeNet point cloud download had issues."

    # 7e. Unzip point clouds
    log "Extracting point cloud zips..."
    for ZIP_FILE in $(find "${CAP3D_DIR}/pointclouds" -name "*.zip" -type f 2>/dev/null); do
        PARENT_DIR=$(dirname "$ZIP_FILE")
        unzip -n -q "$ZIP_FILE" -d "$PARENT_DIR" 2>&1 || true
    done

    log "Cap3D → ${CAP3D_DIR}"
else
    log "Skipping Cap3D (--skip-cap3d)"
fi


# ============================================================================
# Summary
# ============================================================================
log "════════════════════════════════════════════════════════════════"
log "All downloads complete! Summary:"
log "════════════════════════════════════════════════════════════════"
log ""
log "  Data root: $(realpath "$DATA_ROOT")"
[ "$SKIP_DFN"        = false ] && log "  1. DFN-2B       → ${DATA_ROOT}/dfn_2b"
[ "$SKIP_OPENIMAGES" = false ] && log "  2. Open Images  → ${DATA_ROOT}/open_images_v7  (train/val/test)"
[ "$SKIP_WEBVID"     = false ] && log "  3. WebVid-10M   → ${DATA_ROOT}/webvid_10m"
[ "$SKIP_TEXTVR"     = false ] && log "  4. TextVR       → ${DATA_ROOT}/textvr           (train/val/test in zip)"
[ "$SKIP_PANDA70M"   = false ] && log "  5. Panda-70M    → ${DATA_ROOT}/panda_70m        (train-full/10M/2M/val/test)"
[ "$SKIP_OBJAVERSE" = false ] && log "  6. Objaverse-XL → ${DATA_ROOT}/objaverse_xl     (800K core + 10M XL index)"
[ "$SKIP_CAP3D"     = false ] && log "  7. Cap3D        → ${DATA_ROOT}/cap3d             (captions + point clouds)"
log ""
log "  Disk usage:"
du -sh "${DATA_ROOT}"/* 2>/dev/null || true
log ""
log "Done. 🎉"
