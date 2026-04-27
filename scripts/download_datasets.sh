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
#   WEBVID_VIDEO_SIZE=256  # dùng khi WEBVID_RESIZE_MODE != none
#   WEBVID_DOWNLOAD_SIZE=480 # preferred height cho yt-dlp fallback
#   WEBVID_SHARD_SIZE=1000 # số sample/shard của video2dataset
#   WEBVID_PROCESSES=8     # số process cho video2dataset
#   WEBVID_THREADS=16      # số thread cho video2dataset
#   WEBVID_OUTPUT_FORMAT=files|webdataset  # mặc định files, khớp loader trong repo
#   WEBVID_RESIZE_MODE=none|scale|scale,crop  # mặc định none: chỉ download, không resize
#   WEBVID_INCREMENTAL_MODE=incremental|overwrite
#   WEBVID_MAX_ROWS=N      # debug: chỉ tải N dòng metadata đầu tiên
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
    hf download \
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
    hf download \
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

    if ! python3 - <<'PY' >/dev/null 2>&1; then
from video2dataset import video2dataset  # noqa: F401
PY
        log "[WARN] python3 không import được 'video2dataset'. Chạy: pip install -U video2dataset. Bỏ qua bước MP4."
        return 0
    fi

    # Chỉ giữ các CSV nằm trong danh sách .shards_keep.txt (1/2 đầu).
    if [[ ! -s "${WEBVID_DIR}/.shards_keep.txt" ]]; then
        log "[WARN] không có .shards_keep.txt — bỏ qua fetch MP4."
        return 0
    fi

    log "Gộp ${csv_dir} (1/2 partition) -> ${merged}"
    local _merge_py
    _merge_py="$(mktemp /tmp/webvid_merge_XXXXXX.py)"
    cat > "${_merge_py}" <<'PY'
import csv
import os
import sys

root = os.environ["WEBVID_ROOT"]
out = os.environ["WEBVID_MERGED"]
max_rows = int(os.environ.get("WEBVID_MAX_ROWS") or 0)
keep_path = os.path.join(root, ".shards_keep.txt")
keep = [line.strip() for line in open(keep_path, encoding="utf-8") if line.strip()]
required = {"contentUrl", "name"}
n_rows = 0
n_missing_files = 0

with open(out, "w", newline="", encoding="utf-8") as fout:
    writer = None
    fieldnames = None
    for rel in keep:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            n_missing_files += 1
            continue
        with open(path, newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            if not reader.fieldnames:
                continue
            missing = required.difference(reader.fieldnames)
            if missing:
                raise SystemExit(f"[merge] CSV {path} thiếu cột bắt buộc: {sorted(missing)}")
            if writer is None:
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writeheader()
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"[merge] CSV {path} có header khác file đầu tiên")

            for row in reader:
                if not (row.get("contentUrl") or "").strip():
                    continue
                # PyArrow (used by video2dataset) does not handle multiline cell
                # values by default; collapse embedded newlines to a space.
                row = {k: (v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ") if isinstance(v, str) else v) for k, v in row.items()}
                writer.writerow(row)
                n_rows += 1
                if max_rows > 0 and n_rows >= max_rows:
                    break
        if max_rows > 0 and n_rows >= max_rows:
            break

if n_missing_files:
    print(f"[merge] warning: missing_files={n_missing_files}", file=sys.stderr)
if n_rows == 0:
    raise SystemExit("[merge] không có dòng hợp lệ để tải video")
limit_note = f" (limited by WEBVID_MAX_ROWS={max_rows})" if max_rows > 0 else ""
print(f"[merge] rows={n_rows}{limit_note} -> {out}")
PY
    WEBVID_ROOT="${WEBVID_DIR}" \
    WEBVID_MERGED="${merged}" \
    WEBVID_MAX_ROWS="${WEBVID_MAX_ROWS:-0}" \
    python3 "${_merge_py}"
    rm -f "${_merge_py}"

    local size="${WEBVID_VIDEO_SIZE:-256}"
    local download_size="${WEBVID_DOWNLOAD_SIZE:-480}"
    local shard_size="${WEBVID_SHARD_SIZE:-1000}"
    local nproc="${WEBVID_PROCESSES:-8}"
    local nthr="${WEBVID_THREADS:-16}"
    local output_format="${WEBVID_OUTPUT_FORMAT:-files}"
    local resize_mode="${WEBVID_RESIZE_MODE:-none}"
    local incremental_mode="${WEBVID_INCREMENTAL_MODE:-incremental}"
    local timeout="${WEBVID_TIMEOUT:-60}"
    local tmp_dir="${WEBVID_TMP_DIR:-/tmp}"

    log "video2dataset → ${out_dir} (format=${output_format}, shard=${shard_size}, size=${size}, proc=${nproc}, thr=${nthr})"
    local _v2d_py
    _v2d_py="$(mktemp /tmp/webvid_v2d_XXXXXX.py)"
    cat > "${_v2d_py}" <<'PY'
import csv
import inspect
import os

from video2dataset import video2dataset

if __name__ == "__main__":
    url_list = os.environ["WEBVID_URL_LIST"]
    output_folder = os.environ["WEBVID_OUTPUT_FOLDER"]
    output_format = os.environ["WEBVID_OUTPUT_FORMAT"]
    video_size = int(os.environ["WEBVID_VIDEO_SIZE"])
    download_size = int(os.environ["WEBVID_DOWNLOAD_SIZE"])
    shard_size = int(os.environ["WEBVID_SHARD_SIZE"])
    processes_count = int(os.environ["WEBVID_PROCESSES"])
    thread_count = int(os.environ["WEBVID_THREADS"])
    resize_mode = os.environ["WEBVID_RESIZE_MODE"].strip()
    incremental_mode = os.environ["WEBVID_INCREMENTAL_MODE"]
    timeout = int(os.environ["WEBVID_TIMEOUT"])
    tmp_dir = os.environ["WEBVID_TMP_DIR"]

    with open(url_list, newline="", encoding="utf-8") as fin:
        columns = next(csv.reader(fin))

    required = {"contentUrl", "name"}
    missing = required.difference(columns)
    if missing:
        raise SystemExit(f"video2dataset input thiếu cột bắt buộc: {sorted(missing)}")

    save_cols = [c for c in ("videoid", "page_dir", "duration", "page_idx") if c in columns]
    common_kwargs = {
        "url_list": url_list,
        "output_folder": output_folder,
        "output_format": output_format,
        "input_format": "csv",
        "url_col": "contentUrl",
        "caption_col": "name",
        "save_additional_columns": save_cols,
    }

    signature = inspect.signature(video2dataset)
    params = signature.parameters

    if "encode_formats" in params:
        common_kwargs["encode_formats"] = {"video": "mp4"}

    if "config" in params:
        subsampling = {}
        if resize_mode.lower() not in {"", "0", "none", "no", "false"}:
            subsampling["ResolutionSubsampler"] = {
                "args": {
                    "video_size": video_size,
                    "resize_mode": resize_mode,
                }
            }
        kwargs = {
            **common_kwargs,
            "config": {
                "subsampling": subsampling,
                "reading": {
                    "yt_args": {
                        "download_size": download_size,
                        "download_audio_rate": 44100,
                        "yt_metadata_args": None,
                    },
                    "timeout": timeout,
                    "sampler": None,
                },
                "storage": {
                    "number_sample_per_shard": shard_size,
                    "oom_shard_count": 5,
                    "captions_are_subtitles": False,
                },
                "distribution": {
                    "processes_count": processes_count,
                    "thread_count": thread_count,
                    "subjob_size": 1000,
                    "distributor": "multiprocessing",
                },
            },
            "incremental_mode": incremental_mode,
            "tmp_dir": tmp_dir,
        }
    else:
        legacy_kwargs = {
            **common_kwargs,
            "processes_count": processes_count,
            "thread_count": thread_count,
            "number_sample_per_shard": shard_size,
            "oom_shard_count": 5,
            "distributor": "multiprocessing",
            "subjob_size": 1000,
            "incremental_mode": incremental_mode,
            "timeout": timeout,
            "tmp_dir": tmp_dir,
        }
        if resize_mode.lower() not in {"", "0", "none", "no", "false"}:
            legacy_kwargs["video_size"] = video_size
            legacy_kwargs["resize_mode"] = [part.strip() for part in resize_mode.split(",") if part.strip()]
        kwargs = {key: value for key, value in legacy_kwargs.items() if key in params}

    print(
        "[v2d] api_params="
        f"{','.join(params)}; save_cols={save_cols}; output_format={output_format}; "
        f"resize_mode={resize_mode or 'none'}"
    )
    video2dataset(**kwargs)
PY
    WEBVID_URL_LIST="${merged}" \
    WEBVID_OUTPUT_FOLDER="${out_dir}" \
    WEBVID_OUTPUT_FORMAT="${output_format}" \
    WEBVID_VIDEO_SIZE="${size}" \
    WEBVID_DOWNLOAD_SIZE="${download_size}" \
    WEBVID_SHARD_SIZE="${shard_size}" \
    WEBVID_PROCESSES="${nproc}" \
    WEBVID_THREADS="${nthr}" \
    WEBVID_RESIZE_MODE="${resize_mode}" \
    WEBVID_INCREMENTAL_MODE="${incremental_mode}" \
    WEBVID_TIMEOUT="${timeout}" \
    WEBVID_TMP_DIR="${tmp_dir}" \
    python3 "${_v2d_py}"
    rm -f "${_v2d_py}"
    log "Hoàn tất fetch MP4: ${out_dir}"
}

download_webvid() {
    log "=== WebVid-10M: ${WEBVID_REPO} → ${WEBVID_DIR} ==="
    # TempoFunk/webvid-10M chỉ chứa metadata CSV (videoid + caption + URL),
    # tổ chức theo data/train/partitions/XXXX.csv. Lấy 1/2 số partition.
    # download_half_shards "${WEBVID_REPO}" "${WEBVID_DIR}" "data/train/partitions/" "csv"
    # Validation rất nhỏ → tải đầy đủ.
    log "Tải toàn bộ validation partitions (nhỏ)..."
    # hf download \
    #     "${WEBVID_REPO}" \
    #     --repo-type dataset \
    #     --local-dir "${WEBVID_DIR}" \
    #     --max-workers "${NUM_PROC}" \
    #     "${HF_TOKEN_ARG[@]}" \
    #     --include "data/val/partitions/*.csv" \
    #     --include "README*" || true

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
