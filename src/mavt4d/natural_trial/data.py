"""Read verified compact RGB exports; recording IDs never enter the model."""
import json
from pathlib import Path

import numpy as np
import torch

from .prepare import sha256, split_for_source


class NaturalDataset:
    def __init__(self, root, cfg):
        self.root = Path(root).resolve()
        self.summary = json.loads((self.root / "summary.json").read_text())
        if self.summary.get("schema") != "natural-epic-trial-v1":
            raise ValueError("unknown natural dataset schema")
        for key in ("frames", "height", "width"):
            if self.summary[key] != getattr(cfg, key):
                raise ValueError("natural export dimensions mismatch")
        if sha256(self.root / "manifest.jsonl") != self.summary["manifest_sha256"]:
            raise ValueError("natural manifest checksum mismatch")
        rows = [json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()]
        self.records = {row["record_id"]: row for row in rows}
        if len(rows) != len(self.records):
            raise ValueError("duplicate natural clip")
        self.cache = {}
        for row in rows:
            if row["split"] != split_for_source(row["source"]):
                raise ValueError("source-disjoint split mismatch")
            path = (self.root / row["input"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError("natural clip path escapes export or is missing")
        self.train = tuple(row["record_id"] for row in rows if row["split"] == "train")
        self.dev = tuple(row["record_id"] for row in rows if row["split"] == "dev")
        if not self.train or not self.dev:
            raise ValueError("natural trial requires train and dev")
        self.fingerprint = sha256(self.root / "summary.json")
        self.cfg = cfg

    def batch(self, ids, device):
        images, times = [], []
        for key in ids:
            row = self.records[key]
            if key not in self.cache:
                path = self.root / row["input"]
                if sha256(path) != row["sha256"]:
                    raise ValueError("natural input checksum mismatch")
                with np.load(path, allow_pickle=False) as data:
                    rgb, t = data["images"].copy(), data["frame_times"].copy()
                if rgb.dtype != np.uint8 or rgb.shape != (self.cfg.frames, 3, self.cfg.height, self.cfg.width):
                    raise ValueError("natural input image shape/dtype mismatch")
                if t.dtype != np.float64 or t.shape != (self.cfg.frames,) or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
                    raise ValueError("natural timestamp shape/order invalid")
                self.cache[key] = rgb, t
            rgb, t = self.cache[key]
            images.append(torch.from_numpy(rgb).float().div(127.5).sub(1))
            times.append(torch.from_numpy(t))
        return torch.stack(images).to(device), torch.stack(times).to(device)
