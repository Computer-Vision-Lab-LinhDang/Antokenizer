from __future__ import annotations

import argparse

from mavt.data.datamodule import UnifiedDataModule
from mavt.model.antoken import AToken
from mavt.utils.vis import save_reconstruction_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    model = AToken.load_from_checkpoint(args.checkpoint)
    model.eval()
    datamodule = UnifiedDataModule(data_root=args.data_root, stage=args.stage)
    datamodule.setup("test")
    loader = datamodule.test_dataloader()[0]
    batch = next(iter(loader))
    outputs = model(batch)
    modality = batch["modality"][0] if isinstance(batch["modality"], list) else batch["modality"]
    save_reconstruction_grid(batch[modality], outputs["reconstruction"], args.output)


if __name__ == "__main__":
    main()
