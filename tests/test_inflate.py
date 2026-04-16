import torch
from torch import nn

from mavt.encoder.patch_embed import SpaceTimePatchEmbed


def test_inflate_conv2d_matches_image_path():
    conv2d = nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=True)
    patch = SpaceTimePatchEmbed.from_conv2d(conv2d, temporal_patch_size=2)
    image = torch.randn(2, 3, 16, 16)

    expected = conv2d(image).flatten(2).transpose(1, 2)
    actual, _ = patch.forward_image(image)
    torch.testing.assert_close(actual, expected)
