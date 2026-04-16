from __future__ import annotations

from mavt.compat import LearningRateMonitor, ModelCheckpoint


def default_callbacks() -> list:
    return [
        ModelCheckpoint(save_last=True, monitor="val/loss", mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]
