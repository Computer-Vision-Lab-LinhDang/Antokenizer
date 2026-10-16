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
        progress = self.output / "progress.jsonl"
        if progress.exists():
            # This is only the local journal sequence. It is deliberately not
            # used as W&B's transport step, whose resumed server tail can be
            # ahead of the SDK's initially reported run.step.
            with progress.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        self._last_step = max(self._last_step, int(json.loads(line)["step"]))
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
        self.run.define_metric("progress/logical_step")
        self.run.define_metric("*", step_metric="progress/logical_step", step_sync=True)
        (self.output / "wandb-run.json").write_text(
            json.dumps({"id": self.run_id, "url": self.url}, sort_keys=True) + "\n",
            encoding="utf-8")

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if self.rank != 0:
            return
        logical_step = step
        step = max(step, self._last_step + 1)
        self._last_step = step
        values = dict(metrics)
        values["progress/logical_step"] = logical_step
        for name, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"nonfinite logging metric: {name}")
        local = {"utc": datetime.now(timezone.utc).isoformat(), "step": step, **values}
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(local, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        if self.run is not None:
            # Omit step: W&B's backend owns its monotonic history counter.
            # Passing a resumed optimizer/local counter can silently drop
            # events when the server already has later history records.
            self.run.log(values)

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
