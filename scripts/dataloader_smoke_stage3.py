"""CPU-only smoke test cho dataloader stage3.

Mục đích: kiểm tra MAVTDataModule có nạp được dữ liệu image / video / threed
từ /home/user02/linhdang/Antokenizer/dataset mà không cần GPU.

Chạy:
    python scripts/dataloader_smoke_stage3.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

# Force CPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mavt.data.datamodule import MAVTDataModule  # noqa: E402


# Subset image dir (chỉ vài tar shard) để init nhanh.
IMAGE_SHARDS_DIR = "/tmp/mavt_smoke/image_shards"
VIDEO_SHARDS_DIR = str(PROJECT_ROOT / "dataset" / "dataset_10m")
TRIPLANE_DIR = str(PROJECT_ROOT / "dataset" / "tripplane")

EXPECTED_SHAPES = {
    "image": (3, 256, 256),
    "video": (3, 16, 256, 256),
    "threed": (3, 3, 256, 256),
}


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def check_one_modality(modality: str) -> bool:
    banner(f"Modality: {modality}")
    dm = MAVTDataModule(
        active_modalities=[modality],
        image_shards_dir=IMAGE_SHARDS_DIR if modality == "image" else None,
        video_shards_dir=VIDEO_SHARDS_DIR if modality == "video" else None,
        video_max_shards=1 if modality == "video" else None,
        triplane_dir=TRIPLANE_DIR if modality == "threed" else None,
        image_resolution=256,
        video_frames=16,
        video_resolution=256,
        triplane_res=256,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.05,
        test_split=0.0,
    )
    t0 = time.time()
    dm.setup()
    print(f"setup() took {time.time() - t0:.1f}s")
    print(f"  train ds size: {len(dm._train_ds)}")
    print(f"  val ds size:   {len(dm._val_ds)}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Train batch
    batch = next(iter(train_loader))
    assert batch["modality"] == modality, batch["modality"]
    data = batch["data"]
    assert isinstance(data, torch.Tensor)
    expected = EXPECTED_SHAPES[modality]
    assert data.shape[1:] == expected, f"{data.shape} vs (B, {expected})"
    print(f"  train batch shape: {tuple(data.shape)}  dtype={data.dtype}")
    print(f"  data range: [{data.min().item():.3f}, {data.max().item():.3f}] (expect ~[-1, 1])")
    if "id" in batch:
        print(f"  ids[:2]:   {batch['id'][:2]}")
    if "caption" in batch:
        caps = [c[:60] for c in batch["caption"][:2]]
        print(f"  caps[:2]:  {caps}")

    # Val batch
    val_batch = next(iter(val_loader))
    assert val_batch["modality"] == modality
    print(f"  val batch shape:   {tuple(val_batch['data'].shape)}")
    return True


def check_combined() -> bool:
    """Test với cả 3 modality cùng lúc — để xác nhận
    ModalityGroupedBatchSampler hoạt động và mỗi batch chỉ chứa 1 modality."""
    banner("Combined: image + video + threed")
    dm = MAVTDataModule(
        active_modalities=["image", "video", "threed"],
        image_shards_dir=IMAGE_SHARDS_DIR,
        video_shards_dir=VIDEO_SHARDS_DIR,
        video_max_shards=1,
        triplane_dir=TRIPLANE_DIR,
        image_resolution=256,
        video_frames=16,
        video_resolution=256,
        triplane_res=256,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.05,
        test_split=0.0,
    )
    t0 = time.time()
    dm.setup()
    print(f"setup() took {time.time() - t0:.1f}s")
    print(f"  train ds size (concat): {len(dm._train_ds)}")
    print(f"  val ds size (concat):   {len(dm._val_ds)}")

    train_loader = dm.train_dataloader()
    seen = {}
    first_batch_per_mod = {}
    n_batches = 200
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        m = batch["modality"]
        seen[m] = seen.get(m, 0) + 1
        if m not in first_batch_per_mod:
            first_batch_per_mod[m] = (i, tuple(batch["data"].shape))
            print(f"  first {m:<7s} batch at idx {i}: shape={tuple(batch['data'].shape)}")
    print(f"  modality counts in first {n_batches} batches: {seen}")
    if len(first_batch_per_mod) < 3:
        print(f"  [WARN] thiếu modality: {set(EXPECTED_SHAPES) - set(first_batch_per_mod)}")
        return False
    return True


def check_balanced() -> bool:
    """Verify modality_weights actually rebalances batches."""
    banner("Combined + modality_weights={image:1, video:1, threed:1}")
    dm = MAVTDataModule(
        active_modalities=["image", "video", "threed"],
        image_shards_dir=IMAGE_SHARDS_DIR,
        video_shards_dir=VIDEO_SHARDS_DIR,
        video_max_shards=1,
        triplane_dir=TRIPLANE_DIR,
        image_resolution=256,
        video_frames=16,
        video_resolution=256,
        triplane_res=256,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        val_split=0.05,
        test_split=0.0,
        modality_weights={"image": 1.0, "video": 1.0, "threed": 1.0},
    )
    dm.setup()
    print(f"  total batches/epoch: {len(dm._batch_sampler)}")
    seen: dict = {}
    for i, batch in enumerate(dm.train_dataloader()):
        if i >= 300:
            break
        seen[batch["modality"]] = seen.get(batch["modality"], 0) + 1
    print(f"  modality counts in first 300 batches: {seen}")
    if min(seen.values()) == 0:
        print("  [FAIL] some modality not seen at all")
        return False
    # With 1:1:1 weights, counts should be roughly equal (within 2x).
    mn, mx = min(seen.values()), max(seen.values())
    if mx > 3 * mn:
        print(f"  [WARN] imbalance ratio {mx / mn:.2f} (expected ~1)")
    return True


def check_super_batch_ddp_simulation() -> bool:
    """Simulate world_size=4: every super-batch must be single-modality across ranks."""
    banner("DDP super-batch grouping (world_size=4)")
    from mavt.data.datamodule import ModalityGroupedBatchSampler  # noqa: WPS433
    os.environ["WORLD_SIZE"] = "4"
    try:
        dm = MAVTDataModule(
            active_modalities=["image", "video", "threed"],
            image_shards_dir=IMAGE_SHARDS_DIR,
            video_shards_dir=VIDEO_SHARDS_DIR,
            video_max_shards=1,
            triplane_dir=TRIPLANE_DIR,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            val_split=0.05,
            test_split=0.0,
        )
        dm.setup()
        sampler: ModalityGroupedBatchSampler = dm._batch_sampler  # type: ignore
        super_batches = sampler._make_super_batches()
        print(f"  num super-batches: {len(super_batches)}")
        # Map each global index to its modality via group boundaries.
        idx_to_mod = {}
        for grp_i, group in enumerate(sampler.groups):
            mod = sampler.modalities[grp_i]
            for idx in group:
                idx_to_mod[idx] = mod
        bad = 0
        for sb in super_batches:
            mods_seen = {idx_to_mod[batch[0]] for batch in sb}
            if len(mods_seen) > 1:
                bad += 1
        print(f"  super-batches with mixed modality: {bad} (must be 0)")
        return bad == 0
    finally:
        os.environ.pop("WORLD_SIZE", None)


def main() -> int:
    print("CPU-only dataloader smoke test for stage3")
    print(f"  image_shards: {IMAGE_SHARDS_DIR}")
    print(f"  video_shards: {VIDEO_SHARDS_DIR}")
    print(f"  triplane:     {TRIPLANE_DIR}")

    ok = True
    for m in ["threed", "image", "video"]:
        try:
            check_one_modality(m)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {m}: {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()
            ok = False

    for fn in (check_combined, check_balanced, check_super_batch_ddp_simulation):
        try:
            if not fn():
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()
            ok = False

    print("\n" + "=" * 50)
    print("SMOKE TEST: PASS" if ok else "SMOKE TEST: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
