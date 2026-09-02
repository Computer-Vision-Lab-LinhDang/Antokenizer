#!/usr/bin/env bash
# Open Images V7 train shards from CVDF S3 (public, no aws cli needed). Each shard ~35-50 GB, ~105k imgs.
#   SHARDS="0 1 2" ./download_openimages.sh          # train_0..train_2  (~125 GB, ~320k images)
set -uo pipefail
OUT="${OUT:-$(dirname "$0")/../../data/datasets/open_images_v7}"; SHARDS="${SHARDS:-0 1}"
mkdir -p "$OUT/images/train" "$OUT/tars"; cd "$OUT"
for s in $SHARDS; do
  f="tars/train_$s.tar.gz"; [ -f "tars/.done_$s" ] && { echo "shard $s done"; continue; }
  echo "[$(date +%H:%M)] shard train_$s"
  wget -c -q --show-progress -O "$f" "https://open-images-dataset.s3.amazonaws.com/tar/train_$s.tar.gz" || { echo "download failed $s"; exit 1; }
  tar -xzf "$f" -C images/train --strip-components=1 && rm -f "$f" && touch "tars/.done_$s"
  echo "[$(date +%H:%M)] shard $s extracted: $(ls images/train | wc -l) images so far"
done
echo "OPENIMAGES_DONE"
