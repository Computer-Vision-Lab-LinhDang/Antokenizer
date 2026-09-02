#!/usr/bin/env bash
# Build manifests for the downloaded subsets and decode-check a few samples.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/miniforge3/envs/mavt/bin/python}"; export PYTHONPATH="$PWD/src"
$PY -m mavt.data.manifest --out-dir data/manifests \
  --openimages-root data/datasets/open_images_v7/images/train \
  --openvid-root data/datasets/openvid/videos --openvid-csv data/datasets/openvid/data/train/OpenVid-1M.csv
wc -l data/manifests/*.jsonl
$PY - <<'PY'
import sys; sys.path.insert(0, "src")
from mavt.data.datasets import ManifestImageDataset, ManifestVideoDataset
im = ManifestImageDataset("data/manifests/openimages.jsonl", 256); s = im[0]; print("image sample", tuple(s["data"].shape), s["id"])
vd = ManifestVideoDataset("data/manifests/openvid.jsonl", n_frames=8, resolution=256)
for i in range(3):
    s = vd[i]; print("video sample", tuple(s["data"].shape), s["id"], "|", s["caption"][:60])
print("MANIFESTS_DONE")
PY
