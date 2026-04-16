from __future__ import annotations

import argparse

import torch

from mavt.data.datamodule import UnifiedDataModule
from mavt.model.antoken import AToken
from mavt.utils.metrics import psnr, ssim_like


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    model = AToken.load_from_checkpoint(args.checkpoint)
    model.eval()
    datamodule = UnifiedDataModule(data_root=args.data_root, stage=args.stage)
    datamodule.setup("test")

    for loader in datamodule.test_dataloader():
        for batch in loader:
            with torch.no_grad():
                outputs = model(batch)
            modality = batch["modality"][0] if isinstance(batch["modality"], list) else batch["modality"]
            target = batch[modality]
            prediction = outputs["reconstruction"]
            print(
                {
                    "modality": modality,
                    "psnr": float(psnr(prediction, target)),
                    "ssim_like": float(ssim_like(prediction, target)),
                }
            )
            break


if __name__ == "__main__":
    main()
