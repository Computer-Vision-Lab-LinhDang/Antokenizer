#!/usr/bin/env bash
# Evaluate every new checkpoint with scripts/eval_gates.py on one GPU (default 7). Idempotent.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/scratch/vunguyen13/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
PY="${PY:-$HOME/miniforge3/envs/mavt/bin/python}"; GPU="${GPU:-7}"
mkdir -p runs/gates
while true; do
  for ck in $(ls -t runs/deepseek/*/checkpoints/*.ckpt runs/*/checkpoints/*.ckpt 2>/dev/null | sort -u); do
    tag=$(echo "$ck" | sed 's#runs/##; s#/checkpoints/#__#; s#\.ckpt$##; s#/#_#g')
    [ -f "runs/gates/$tag.json" ] && continue
    echo "[$(date +%H:%M)] eval $ck"
    HIP_VISIBLE_DEVICES=$GPU $PY scripts/eval_gates.py --ckpt "$ck" --n-images 1024 --cifar-train 5000 --cifar-test 1000 \
      --rgat-impl flex --out "runs/gates/$tag.json" > "runs/gates/$tag.log" 2>&1 \
      && grep -aE "eff_rank|pair_cos|align|knn_top1|psnr" "runs/gates/$tag.log" | tail -7
  done
  sleep 120
done
