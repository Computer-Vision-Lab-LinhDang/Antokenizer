#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_datasets.sh
# ---------------------------------------------------------------------------
# Tải 3 dataset từ HuggingFace Hub về thư mục ./dataset/:
#   1) ImageNet-1k  (ILSVRC/imagenet-1k)           -> 1/2 số shard parquet train
#   2) WebVid-10M   (TempoFunk/webvid-10M, CSV)    -> 1/2 số CSV partition train
#                   (đây là metadata: videoid + caption + URL — video MP4
#                    phải tải riêng từ contentUrl trong CSV)
#   3) Objaverse (LVIS ~46k objects, bản nhỏ)
#      -> dùng lại scripts/download_objaverse.py
#
# Yêu cầu:
#   pip install -U "huggingface_hub[cli]" datasets objaverse tqdm video2dataset
#   export HF_TOKEN=hf_xxx     # cần cho ImageNet-1k (gated)
#
# Tuỳ chọn (env):
#   SKIP_WEBVID_VIDEOS=1   # bỏ qua bước fetch MP4 bằng video2dataset
#   WEBVID_VIDEO_SIZE=256  # cạnh ngắn khi resize video (mặc định 256)
#   WEBVID_PROCESSES=8     # số process cho video2dataset
#   WEBVID_THREADS=16      # số thread cho video2dataset
#
# Cách dùng:
#   bash scripts/download_datasets.sh                 # tải cả 3
#   bash scripts/download_datasets.sh imagenet        # chỉ ImageNet
#   bash scripts/download_datasets.sh webvid          # chỉ WebVid
#   bash scripts/download_datasets.sh objaverse       # chỉ Objaverse
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset}"
mkdir -p "${DATASET_ROOT}"

# ── Defaults (cho phép override qua env) ─────────────────────────────────────
IMAGENET_REPO="${IMAGENET_REPO:-ILSVRC/imagenet-1k}"
WEBVID_REPO="${WEBVID_REPO:-TempoFunk/webvid-10M}"

IMAGENET_DIR="${DATASET_ROOT}/imagenet-1k"
WEBVID_DIR="${DATASET_ROOT}/webvid-10M"

NUM_PROC="${NUM_PROC:-8}"
HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ARG=(--token "${HF_TOKEN}")
fi

TARGET="${1:-all}"

log() { printf "\n[\033[1;36m%s\033[0m] %s\n" "$(date +%H:%M:%S)" "$*"; }

# ── Helper: tải 1/2 số shard (+ tuỳ chọn cộng thêm N shard) từ HF dataset repo
# Args: $1 repo_id  $2 local_dir  $3 file_prefix  $4 ext  $5 extra (mặc định 0)
download_half_shards() {
    local repo_id="$1"
    local local_dir="$2"
    local prefix="$3"
    local ext="${4:-parquet}"
    local extra="${5:-0}"

    mkdir -p "${local_dir}"

    log "Liệt kê shard từ ${repo_id} (prefix='${prefix}', ext='${ext}')..."
    HF_REPO="${repo_id}" HF_PREFIX="${prefix}" HF_EXT="${ext}" python3 - <<'PY' > "${local_dir}/.shards_all.txt"
import os
from huggingface_hub import HfApi
repo   = os.environ["HF_REPO"]
prefix = os.environ["HF_PREFIX"]
ext    = "." + os.environ["HF_EXT"].lstrip(".")
api = HfApi(token=os.environ.get("HF_TOKEN"))
files = api.list_repo_files(repo_id=repo, repo_type="dataset")
shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(ext))
for f in shards:
    print(f)
PY

    local total
    total=$(wc -l < "${local_dir}/.shards_all.txt" | tr -d ' ')
    if [[ "${total}" -eq 0 ]]; then
        log "Không tìm thấy shard nào với prefix='${prefix}' trong ${repo_id}. Bỏ qua."
        return 0
    fi

    local half=$(( (total + 1) / 2 ))
    local keep=$(( half + extra ))
    if (( keep > total )); then keep=${total}; fi
    if (( keep < 0 ));     then keep=0;       fi
    log "Tổng shard: ${total} → tải ${keep} (1/2=${half} + extra=${extra}, capped tại ${total})"
    head -n "${keep}" "${local_dir}/.shards_all.txt" > "${local_dir}/.shards_keep.txt"

    # Build --include flags từ danh sách shard giữ lại.
    local include_args=()
    while IFS= read -r f; do
        [[ -n "$f" ]] && include_args+=( --include "$f" )
    done < "${local_dir}/.shards_keep.txt"

    log "Tải ${keep} shard về ${local_dir}..."
    huggingface-cli download \
        "${repo_id}" \
        --repo-type dataset \
        --local-dir "${local_dir}" \
        --max-workers "${NUM_PROC}" \
        "${HF_TOKEN_ARG[@]}" \
        "${include_args[@]}"

    log "Hoàn tất ${repo_id} (${keep}/${total} shard)."
}

# ── 1) ImageNet-1k ───────────────────────────────────────────────────────────
download_imagenet() {
    log "=== ImageNet-1k: ${IMAGENET_REPO} → ${IMAGENET_DIR} ==="
    # Mặc định 1/2 + 100 shard (vd: 147 + 100 = 247/294 trên ILSVRC/imagenet-1k).
    # Override qua env IMAGENET_EXTRA_SHARDS.
    download_half_shards "${IMAGENET_REPO}" "${IMAGENET_DIR}" "data/train-" "parquet" "${IMAGENET_EXTRA_SHARDS:-100}"
    # Validation/test rất nhỏ → tải đầy đủ để có ground-truth eval.
    log "Tải toàn bộ validation/test (nhỏ) + metadata..."
    huggingface-cli download \
        "${IMAGENET_REPO}" \
        --repo-type dataset \
        --local-dir "${IMAGENET_DIR}" \
        --max-workers "${NUM_PROC}" \
        "${HF_TOKEN_ARG[@]}" \
        --include "data/val-*.parquet" \
        --include "data/validation-*.parquet" \
        --include "data/test-*.parquet" \
        --include "*.json" --include "*.txt" --include "README*" || true
}

# ── 2) WebVid-10M ────────────────────────────────────────────────────────────
fetch_webvid_videos() {
    # Dùng video2dataset để fetch MP4 thật từ contentUrl trong CSV.
    # Đầu vào: thư mục chứa các CSV partition đã được giữ lại (1/2 đầu).
    local csv_dir="${WEBVID_DIR}/data/train/partitions"
    local out_dir="${WEBVID_DIR}/videos_train"
    local merged="${WEBVID_DIR}/.train_half.csv"

    if ! command -v video2dataset >/dev/null 2>&1; then
        log "[WARN] không tìm thấy 'video2dataset'. Chạy: pip install video2dataset. Bỏ qua bước MP4."
        return 0
    fi

    # Chỉ giữ các CSV nằm trong danh sách .shards_keep.txt (1/2 đầu).
    if [[ ! -s "${WEBVID_DIR}/.shards_keep.txt" ]]; then
        log "[WARN] không có .shards_keep.txt — bỏ qua fetch MP4."
        return 0
    fi

    log "Gộp ${csv_dir} (1/2 partition) -> ${merged}"
    python3 - <<PY
import csv, os
keep = [l.strip() for l in open("${WEBVID_DIR}/.shards_keep.txt") if l.strip()]
root = "${WEBVID_DIR}"
out  = "${merged}"
header_written = False
n_rows = 0
with open(out, "w", newline="") as fout:
    writer = None
    for rel in keep:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                n_rows += 1
print(f"[merge] rows={n_rows} -> {out}")
PY

    local size="${WEBVID_VIDEO_SIZE:-256}"
    local nproc="${WEBVID_PROCESSES:-8}"
    local nthr="${WEBVID_THREADS:-16}"
    local cfg="${WEBVID_DIR}/.v2d_config.yaml"

    cat > "${cfg}" <<YAML
subsampling:
  ResizeSubsampler:
    video_size: ${size}
    resize_mode: ["scale", "crop"]
reading:
  yt_args:
    download_size: "480p"
  timeout: 60
  sampler: null
storage:
  number_sample_per_shard: 1000
  oom_shard_count: 5
distribution:
  processes_count: ${nproc}
  thread_count: ${nthr}
  distributor: multiprocessing
YAML

    log "video2dataset → ${out_dir} (size=${size}, proc=${nproc}, thr=${nthr})"
    video2dataset \
        --url_list="${merged}" \
        --input_format="csv" \
        --output_folder="${out_dir}" \
        --output_format="webdataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_dir,duration]' \
        --encode_formats='{"video": "mp4"}' \
        --config="${cfg}"
    log "Hoàn tất fetch MP4: ${out_dir}"
}

download_webvid() {
    log "=== WebVid-10M: ${WEBVID_REPO} → ${WEBVID_DIR} ==="
    # TempoFunk/webvid-10M chỉ chứa metadata CSV (videoid + caption + URL),
    # tổ chức theo data/train/partitions/XXXX.csv. Lấy 1/2 số partition.
    download_half_shards "${WEBVID_REPO}" "${WEBVID_DIR}" "data/train/partitions/" "csv"
    # Validation rất nhỏ → tải đầy đủ.
    log "Tải toàn bộ validation partitions (nhỏ)..."
    huggingface-cli download \
        "${WEBVID_REPO}" \
        --repo-type dataset \
        --local-dir "${WEBVID_DIR}" \
        --max-workers "${NUM_PROC}" \
        "${HF_TOKEN_ARG[@]}" \
        --include "data/val/partitions/*.csv" \
        --include "README*" || true

    if [[ "${SKIP_WEBVID_VIDEOS:-0}" != "1" ]]; then
        fetch_webvid_videos
    else
        log "SKIP_WEBVID_VIDEOS=1 → bỏ qua bước fetch MP4 bằng video2dataset."
    fi
}

# ── 3) Objaverse (LVIS subset) ───────────────────────────────────────────────
download_objaverse() {
    log "=== Objaverse (LVIS ~46k) → ./dataset/objaverse ==="
    # Tận dụng script đã có: scripts/download_objaverse.py
    python "${SCRIPT_DIR}/download_objaverse.py"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "${TARGET}" in
    all)        download_imagenet; download_webvid; download_objaverse ;;
    imagenet)   download_imagenet ;;
    webvid)     download_webvid ;;
    objaverse)  download_objaverse ;;
    *)
        echo "Usage: $0 [all|imagenet|webvid|objaverse]" >&2
        exit 1
        ;;
esac

log "Tất cả đã xong. Dataset nằm trong: ${DATASET_ROOT}"
