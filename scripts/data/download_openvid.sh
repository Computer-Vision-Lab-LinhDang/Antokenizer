#!/usr/bin/env bash
# OpenVid-1M parts from HF (~39 GB, ~6k clips each) + caption CSV.
#   PARTS="0 1 2" ./download_openvid.sh
set -uo pipefail
OUT="${OUT:-$(dirname "$0")/../../data/datasets/openvid}"; PARTS="${PARTS:-0 1}"
PY="${PY:-$HOME/miniforge3/envs/mavt/bin/python}"
mkdir -p "$OUT/videos" "$OUT/zips"; cd "$OUT"
$PY - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download("nkp37/OpenVid-1M", "data/train/OpenVid-1M.csv", repo_type="dataset", local_dir=".")
print("captions csv ok")
PY
for p in $PARTS; do
  [ -f "zips/.done_$p" ] && { echo "part $p done"; continue; }
  echo "[$(date +%H:%M)] OpenVid_part$p.zip"
  $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download('nkp37/OpenVid-1M','OpenVid_part$p.zip',repo_type='dataset',local_dir='zips')" || { echo "download failed $p"; exit 1; }
  unzip -q -o "zips/OpenVid_part$p.zip" -d videos && rm -f "zips/OpenVid_part$p.zip" && touch "zips/.done_$p"
  echo "[$(date +%H:%M)] part $p: $(find videos -name '*.mp4' | wc -l) clips so far"
done
echo "OPENVID_DONE"
