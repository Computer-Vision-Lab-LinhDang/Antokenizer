import torch

from mavt.encoder.patch_embed import SpaceTimePatchEmbed
from mavt.patchify.video import VideoPatchifier


def test_video_patchifier_shapes():
    patchifier = VideoPatchifier(SpaceTimePatchEmbed(embed_dim=32, patch_size=8, temporal_patch_size=2))
    video = torch.randn(2, 3, 4, 32, 32)
    tokens, positions = patchifier(video)

    assert tokens.shape == (2, 32, 32)
    assert positions.shape == (2, 32, 4)
    assert positions[0, 0].tolist() == [0, 0, 0, 0]


def test_video_patchifier_tiles_long_inputs():
    video = torch.randn(1, 3, 40, 16, 16)
    tiles = list(VideoPatchifier.iter_tiles(video, tile_frames=16, stride=8))
    assert tiles[0][0] == 0
    assert tiles[-1][1].shape[2] == 16
