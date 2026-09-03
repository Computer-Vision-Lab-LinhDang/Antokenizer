#!/usr/bin/env bash
# Cap3D captions (~280 MB) + CIFAR-100 (170 MB) for the kNN gate.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="${PY:-$HOME/miniforge3/envs/mavt/bin/python}"
mkdir -p "$ROOT/data/datasets/cap3d/captions" "$ROOT/data/eval/cifar100"
HF=https://huggingface.co/datasets/tiange/Cap3D/resolve/main
for f in Cap3D_automated_Objaverse_full.csv Cap3D_automated_ABO.csv Cap3D_automated_ShapeNet.csv; do wget -nc -q -P "$ROOT/data/datasets/cap3d/captions" "$HF/$f" && echo "cap3d: $f"; done
wget -nc -q -P "$ROOT/data/eval/cifar100" https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz && echo "cifar100 tar ok"
echo "SMALL_DONE"
