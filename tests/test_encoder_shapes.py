import torch

from mavt.encoder.unified_encoder import UnifiedEncoder


def test_encoder_shapes_for_image_and_video():
    encoder = UnifiedEncoder(
        embed_dim=64,
        depth=2,
        num_heads=4,
        patch_size=8,
        temporal_patch_size=2,
    )

    image_batch = {"image": torch.randn(2, 3, 32, 32), "modality": "image"}
    image_out = encoder(image_batch)
    assert image_out["tokens"].shape == (2, 16, 64)
    assert image_out["attn_mask"].all()

    video_batch = {"video": torch.randn(2, 3, 4, 32, 32), "modality": "video"}
    video_out = encoder(video_batch)
    assert video_out["tokens"].shape == (2, 32, 64)
    assert video_out["attn_mask"].shape == (2, 32, 32)
