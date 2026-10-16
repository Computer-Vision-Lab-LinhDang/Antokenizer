"""Rank-zero online W&B progress, local evidence, and checkpoint artifacts.

Credentials are obtained by the W&B SDK from the caller's environment. This
module never calls login, writes a key, records the environment, or uploads
source files, host metadata, console output, or dataset paths.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


class RunLogger:
    def __init__(self, output: str | Path, *, rank: int, config: Mapping[str, Any],
                 run_id: str | None = None, sdk=None, enabled: bool = True) -> None:
        self.output = Path(output)
        self.rank = rank
        self.run = None
        self.sdk = None
        self.run_id = run_id
        self.url = None
        self._last_step = -1
        if rank != 0:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        if not enabled:
            return
        if os.environ.get("WANDB_MODE", "online") != "online":
            raise ValueError("production W&B logging requires online mode")
        self.sdk = sdk if sdk is not None else importlib.import_module("wandb")
        self.run = self.sdk.init(
            project="semantic-router", name="semantic router", mode="online",
            id=run_id, resume="must" if run_id else None,
            dir=str(self.output), config=dict(config),
            settings={"disable_code": True, "disable_git": True,
                      "x_disable_meta": True, "x_disable_stats": True,
                      "x_disable_machine_info": True, "console": "off",
                      "init_timeout": 60},
        )
        if self.run is None:
            raise RuntimeError("W&B did not return an online run")
        self.run_id, self.url = self.run.id, self.run.url
        self._last_step = int(getattr(self.run, "step", 0)) - 1
        (self.output / "wandb-run.json").write_text(
            json.dumps({"id": self.run_id, "url": self.url}, sort_keys=True) + "\n",
            encoding="utf-8")

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if self.rank != 0:
            return
        step = max(step, self._last_step + 1)
        self._last_step = step
        values = dict(metrics)
        for name, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"nonfinite logging metric: {name}")
        local = {"utc": datetime.now(timezone.utc).isoformat(), "step": step, **values}
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(local, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        if self.run is not None:
            self.run.log(values, step=step)

    def checkpoint(self, path: str | Path, *, metadata: Mapping[str, Any]) -> None:
        if self.rank != 0 or self.run is None:
            return
        path = Path(path)
        artifact = self.sdk.Artifact(
            name=f"hybrid-{self.run_id}-checkpoint", type="model", metadata=dict(metadata))
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=["latest", f"{metadata['phase']}-{metadata['step']}"]
                              if "phase" in metadata else ["latest", f"step-{metadata['step']}"])

    def finish(self, *, exit_code: int = 0) -> None:
        if self.rank == 0 and self.run is not None:
            self.run.finish(exit_code=exit_code)
