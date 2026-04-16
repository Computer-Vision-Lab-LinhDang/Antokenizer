from mavt.compat import LightningCLI
from mavt.data.datamodule import UnifiedDataModule
from mavt.model.antoken import AToken


class AntokenCLI(LightningCLI):
    """Small LightningCLI wrapper for repo defaults."""


def main(args=None) -> None:
    AntokenCLI(
        AToken,
        UnifiedDataModule,
        args=args,
        save_config_kwargs={"overwrite": True},
    )
