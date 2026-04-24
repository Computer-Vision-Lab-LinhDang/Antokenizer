import pyarrow.parquet as pq
import glob
from pathlib import Path

parquet_dir = "./dataset/datasets--ILSVRC--imagenet-1k/snapshots/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data/"  # sửa lại đường dẫn

corrupt = []
ok = []

for f in sorted(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)):
    try:
        pq.read_metadata(f)  # chỉ đọc metadata, nhanh
        ok.append(f)
    except Exception as e:
        print(f"CORRUPT: {f}\n  → {e}\n")
        corrupt.append(f)

print(f"\nOK: {len(ok)} | Corrupt: {len(corrupt)}")
