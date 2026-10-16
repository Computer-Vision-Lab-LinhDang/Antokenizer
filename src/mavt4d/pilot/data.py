"""Verified, observation-only loading and source-grouped pilot partitions."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from mavt.data.splits import assign
from mavt4d.contracts import ObservationStep
from .config import PilotConfig

INPUT_KEYS = frozenset({"images", "observed", "frame_times", "intrinsics", "cam_to_world"})
MANIFEST_KEYS = frozenset({"record_id", "group_key", "split", "render_config_sha256", "input_path"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Only tensors are permitted; IDs, labels and ground truth stay outside.

    All eight variants from a source share camera calibration and timestamps,
    permitting a single explicit query-camera table for the batched decoder.
    RGB bounds are checked by the file loader, not here, to allow controlled
    perturbation tests. This object is training data, never recurrent state.
    """
    images: Tensor
    observed: Tensor
    frame_times: Tensor
    intrinsics: Tensor
    cam_to_world: Tensor

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, f.name), Tensor) for f in fields(self)):
            raise TypeError("TrainingBatch fields must be tensors")
        if self.images.ndim != 6 or self.images.shape[0] <= 0 or self.images.shape[1] != 1:
            raise ValueError("images: expected [B,1,T,3,H,W]")
        batch = self.images.shape[0]
        for f in fields(self):
            tensor = getattr(self, f.name)
            if tensor.shape[0] != batch or tensor.device != self.images.device:
                raise ValueError("batch tensor leading dimension/device mismatch")
        for index in range(batch):
            ObservationStep(**{f.name: getattr(self, f.name)[index] for f in fields(self)})
        for name in ("frame_times", "intrinsics", "cam_to_world"):
            value = getattr(self, name)
            if not torch.equal(value, value[:1].expand_as(value)):
                raise ValueError(f"{name}: source-group batch requires shared calibration and times")

    def to(self, device: str | torch.device) -> "TrainingBatch":
        return TrainingBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class PilotDataset:
    """Verify exported manifests, then read only input files on demand.

    Evaluation-manifest bytes are hashed for integrity but labels are never
    parsed here. Geometry target files are neither opened nor cached. Every
    cached input is verified before insertion and cloned when returned.
    """
    def __init__(self, root: str | Path, cfg: PilotConfig) -> None:
        self.root = Path(root).resolve()
        self.cfg = cfg
        self.summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        summary = self.summary
        expected_config = asdict(cfg.event_config)
        if summary.get("version") != cfg.dataset_version or summary.get("config") != expected_config:
            raise ValueError("dataset config differs from locked render configuration")
        expected_seeds = list(range(cfg.seed_start, cfg.seed_start + cfg.source_count))
        if summary.get("source_seeds") != expected_seeds:
            raise ValueError("dataset source seeds differ from locked config")
        render = {k: v for k, v in expected_config.items() if k != "val_fraction"}
        fingerprint = hashlib.sha256((json.dumps(render, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
        if summary.get("render_config_sha256") != fingerprint:
            raise ValueError("render config fingerprint mismatch")
        if summary.get("audits") != {"identical_final_two": True, "different_histories": True,
                                     "reversal_frame_multiset": True, "target_visible": True}:
            raise ValueError("dataset does not contain completed required fixture audits")
        hashes = summary.get("sha256", {})
        for name, filename in (("manifest", "manifest_v1.jsonl"), ("evaluation", "eval_targets_v1.jsonl")):
            if hashes.get(name) != file_sha256(self.root / filename):
                raise ValueError(f"{name} checksum mismatch")
        rows = [json.loads(line) for line in (self.root / "manifest_v1.jsonl").read_text(encoding="utf-8").splitlines()]
        grouped: dict[str, list] = defaultdict(list)
        seen_ids, seen_paths = set(), set()
        expected_groups = {f"synthetic-event:source-{seed}" for seed in expected_seeds}
        for row in rows:
            if set(row) != MANIFEST_KEYS:
                raise ValueError("input manifest violates observation-only whitelist")
            record_id, group = row["record_id"], row["group_key"]
            if not isinstance(record_id, str) or len(record_id) != 64 or any(c not in "0123456789abcdef" for c in record_id):
                raise ValueError("invalid opaque record ID")
            if record_id in seen_ids or row["input_path"] in seen_paths:
                raise ValueError("duplicate input record or path")
            seen_ids.add(record_id)
            seen_paths.add(row["input_path"])
            if group not in expected_groups or row["split"] != assign(group, cfg.val_fraction):
                raise ValueError("source group or original split mismatch")
            if row["render_config_sha256"] != fingerprint:
                raise ValueError("manifest render config fingerprint mismatch")
            if row["input_path"] != f"inputs/{record_id}.pt":
                raise ValueError("input path must match opaque record ID")
            path = self.root / row["input_path"]
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError("input file escapes dataset root or is absent")
            if row["input_path"] not in hashes.get("inputs", {}):
                raise ValueError("input checksum absent")
            grouped[group].append(MappingProxyType(row))
        if set(grouped) != expected_groups or any(len(records) != 8 for records in grouped.values()):
            raise ValueError("dataset must contain eight distinct variants per requested source")
        if set(hashes.get("inputs", {})) != seen_paths:
            raise ValueError("input checksum table differs from manifest")
        self._records = {key: tuple(value) for key, value in sorted(grouped.items())}
        self._cache: dict[str, dict[str, Tensor]] = {}
        train, dev, val = [], [], []
        for group, records in self._records.items():
            if records[0]["split"] == "val":
                val.append(group)
            else:
                fraction = int(hashlib.sha256((cfg.dev_salt + group).encode()).hexdigest(), 16) / 2**256
                (dev if fraction < cfg.dev_fraction else train).append(group)
        self.train_groups, self.dev_groups, self.val_groups = tuple(train), tuple(dev), tuple(val)
        if not self.train_groups:
            raise ValueError("configuration has no training sources")
        self.overfit_group = self.train_groups[0]
        observed_counts = {
            "samples": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
            "sources": {s: sum(rs[0]["split"] == s for rs in self._records.values()) for s in ("train", "val")},
        }
        if summary.get("counts") != observed_counts:
            raise ValueError("summary counts differ from verified manifest")

    def records_for_group(self, group_key: str) -> tuple[Mapping, ...]:
        """Stable manifest order; audit metadata must not be supplied to a model."""
        return self._records[group_key]

    def load_group(self, group_key: str, device: str | torch.device = "cpu") -> TrainingBatch:
        inputs = []
        for row in self.records_for_group(group_key):
            relative = row["input_path"]
            if relative not in self._cache:
                path = self.root / relative
                if file_sha256(path) != self.summary["sha256"]["inputs"][relative]:
                    raise ValueError(f"input checksum mismatch: {relative}")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or set(payload) != INPUT_KEYS or any(not isinstance(value, Tensor) for value in payload.values()):
                    raise ValueError("input payload violates tensor-only whitelist")
                images = payload["images"]
                if images.shape != (1, self.cfg.T, 3, self.cfg.H, self.cfg.W) or images.dtype != torch.float32:
                    raise ValueError("input RGB shape/dtype differs from config")
                ObservationStep(**payload)
                if not bool(torch.isfinite(images).all()) or bool((images.abs() > 1).any()):
                    raise ValueError("input RGB must be finite in [-1,1]")
                if not bool(payload["observed"].all()):
                    raise ValueError("exported fixtures must start fully observed before masking")
                expected_times = torch.arange(self.cfg.T, dtype=torch.float64)[None] * self.cfg.dt
                if not torch.allclose(payload["frame_times"], expected_times, atol=1e-12, rtol=0):
                    raise ValueError("input frame times differ from render config")
                self._cache[relative] = {name: value.detach().contiguous().clone() for name, value in payload.items()}
            inputs.append(self._cache[relative])
        # stack allocates new storage even on CPU, so callers cannot mutate cache.
        return TrainingBatch(**{name: torch.stack([item[name] for item in inputs]).to(device) for name in INPUT_KEYS})
