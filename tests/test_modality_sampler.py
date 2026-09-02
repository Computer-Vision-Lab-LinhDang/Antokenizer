"""ModalityGroupedBatchSampler v2: per-modality batch sizes + modality sampling weights.

Three-modality training (image 814k / video 140k / 3D ~46k) needs (a) smaller batches for the
token-heavy modalities (video 8 frames = 4x image tokens, 3D = 3 planes) and (b) control over how
often each modality appears — proportional sampling would show 3D in <1% of batches."""
from __future__ import annotations
import collections
import torch
from torch.utils.data import ConcatDataset, Dataset
from mavt.data.datamodule import ModalityGroupedBatchSampler


class _N(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i


def _cat(sizes):
    return ConcatDataset([_N(n) for n in sizes])


def test_per_modality_batch_sizes_and_single_modality_batches():
    cds = _cat([1000, 300, 50])
    s = ModalityGroupedBatchSampler(cds, batch_size=32, modalities=("image", "video", "threed"),
                                    modality_batch_sizes={"video": 8, "threed": 4})
    bounds = [(0, 1000), (1000, 1300), (1300, 1350)]
    sizes = collections.Counter()
    for b in s:
        mod = [i for i, (lo, hi) in enumerate(bounds) if lo <= b[0] < hi][0]
        assert all(bounds[mod][0] <= x < bounds[mod][1] for x in b), "batch mixes modalities"
        sizes[mod] = len(b)
    assert sizes == {0: 32, 1: 8, 2: 4}


def test_modality_weights_control_batch_frequency_with_oversampling():
    cds = _cat([1000, 300, 50])
    s = ModalityGroupedBatchSampler(cds, batch_size=10, modalities=("image", "video", "threed"),
                                    modality_batch_sizes={"threed": 5},
                                    modality_weights={"image": 0.5, "video": 0.3, "threed": 0.2})
    counts = collections.Counter()
    for b in s:
        counts["image" if b[0] < 1000 else "video" if b[0] < 1300 else "threed"] += 1
    total = sum(counts.values())
    assert total >= 100
    frac = {k: v / total for k, v in counts.items()}
    assert abs(frac["image"] - 0.5) < 0.03 and abs(frac["video"] - 0.3) < 0.03 and abs(frac["threed"] - 0.2) < 0.03, frac
    # 3D has only 50 items (10 batches of 5) but must be oversampled to ~20% of batches
    assert counts["threed"] > 10


def test_default_behaviour_unchanged_without_new_args():
    cds = _cat([100, 60])
    s = ModalityGroupedBatchSampler(cds, batch_size=10)
    batches = list(s)
    assert len(batches) == 16 and all(len(b) == 10 for b in batches)
    assert len(s) == 16
