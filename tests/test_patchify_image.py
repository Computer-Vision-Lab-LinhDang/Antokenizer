import torch

from mavt.encoder.patch_embed import SpaceTimePatchEmbed
from mavt.patchify.image import ImagePatchifier


def test_image_patchifier_shapes():
    patchifier = ImagePatchifier(SpaceTimePatchEmbed(embed_dim=32, patch_size=8, temporal_patch_size=2))
    image = torch.randn(2, 3, 32, 32)
    tokens, positions = patchifier(image)

    assert tokens.shape == (2, 16, 32)
    assert positions.shape == (2, 16, 4)
    assert positions[0, 0].tolist() == [0, 0, 0, 0]
