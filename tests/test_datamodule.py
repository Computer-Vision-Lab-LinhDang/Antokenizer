import json
from pathlib import Path

import torch
from PIL import Image

from mavt.data.datamodule import UnifiedDataModule


def test_datamodule_image_batch(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "captions").mkdir()
    array = (torch.rand(32, 32, 3) * 255).byte().numpy()
    Image.fromarray(array).save(tmp_path / "images" / "sample.jpg")
    (tmp_path / "captions" / "images.json").write_text(json.dumps({"sample": "caption"}))

    datamodule = UnifiedDataModule(
        data_root=str(tmp_path),
        stage=1,
        batch_size=1,
        num_workers=0,
        image_size=32,
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["image"].shape == (1, 3, 32, 32)
    assert batch["modality"][0] == "image"
