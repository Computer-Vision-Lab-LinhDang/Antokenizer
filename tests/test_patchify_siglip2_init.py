"""Patchify must reproduce SigLIP2's embedding layer exactly at init (needs the 4 GB model)."""
import pytest
import torch
from conftest import requires_hf
pytestmark = requires_hf
MODEL = "google/siglip2-so400m-patch16-384"


@pytest.fixture(scope="module")
def siglip_emb():
    from transformers import AutoModel
    return AutoModel.from_pretrained(MODEL).vision_model.embeddings.eval()


def _enc():
    from mavt.model.patchify import PatchifyEncoder
    e = PatchifyEncoder(embed_dim=1152, patch_size=16, t_patch=2).eval(); e.init_from_siglip2(MODEL); return e


def test_image_tokens_match_siglip2_at_native_384(siglip_emb):
    enc = _enc(); x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        ours, _, _ = enc(x, "image"); ref = siglip_emb(x)
    assert ours.shape == ref.shape and torch.allclose(ours, ref, atol=1e-4)


def test_image_tokens_at_256_use_interpolated_table(siglip_emb):
    import torch.nn.functional as F
    enc = _enc(); x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        ours, _, _ = enc(x, "image")
        conv = siglip_emb.patch_embedding(x).flatten(2).transpose(1, 2)
        tab = siglip_emb.position_embedding.weight.view(1, 24, 24, -1).permute(0, 3, 1, 2)
        ref = conv + F.interpolate(tab, size=(16, 16), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(1, 256, -1)
    assert torch.allclose(ours, ref, atol=1e-4)


def test_video_first_chunk_equals_second_frame_embedding_at_init(siglip_emb):
    enc = _enc(); v = torch.randn(1, 3, 2, 64, 64)
    with torch.no_grad():
        ours, _, _ = enc(v, "video"); img, _, _ = enc(v[:, :, 1], "image")
    assert torch.allclose(ours, img, atol=1e-4)


def test_plane_tokens_reuse_image_pipeline_at_init(siglip_emb):
    enc = _enc(); planes = torch.randn(1, 3, 3, 64, 64)
    with torch.no_grad():
        ours, _, pid = enc(planes, "threed"); xy, _, _ = enc(planes[:, 0], "image")
    assert torch.allclose(ours[:, :16], xy, atol=1e-4) and pid.tolist() == [0]*16 + [1]*16 + [2]*16
