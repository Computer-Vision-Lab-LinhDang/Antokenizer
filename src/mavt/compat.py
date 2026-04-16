from __future__ import annotations

import torch.nn as nn

try:  # pragma: no cover - exercised only when lightning imports cleanly
    from lightning import LightningDataModule, LightningModule
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.cli import LightningCLI
except Exception:  # pragma: no cover - fallback for lightweight environments
    class LightningModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def save_hyperparameters(self, *args, **kwargs) -> None:
            return None

        def log_dict(self, *args, **kwargs) -> None:
            return None

    class LightningDataModule:
        pass

    class LightningCLI:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("LightningCLI requires the lightning package and its dependencies.")

    class ModelCheckpoint:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class LearningRateMonitor:
        def __init__(self, *args, **kwargs) -> None:
            pass
