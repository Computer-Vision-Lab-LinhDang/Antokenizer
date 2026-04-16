import torch

from mavt.decoder.image import ImageDecoder
from mavt.decoder.video import VideoDecoder


def test_image_decoder_roundtrip_shape():
    decoder = ImageDecoder(latent_dim=16, embed_dim=64, patch_size=8, depth=2, num_heads=4)
    latents = torch.randn(2, 16, 16)
    batch = {"image": torch.randn(2, 3, 32, 32)}
    output = decoder(latents, batch)
    assert output.shape == batch["image"].shape


def test_video_decoder_roundtrip_shape():
    decoder = VideoDecoder(
        latent_dim=16,
        embed_dim=64,
        patch_size=8,
        temporal_patch_size=2,
        depth=2,
        num_heads=4,
    )
    latents = torch.randn(2, 32, 16)
    batch = {"video": torch.randn(2, 3, 4, 32, 32)}
    output = decoder(latents, batch)
    assert output.shape == batch["video"].shape
