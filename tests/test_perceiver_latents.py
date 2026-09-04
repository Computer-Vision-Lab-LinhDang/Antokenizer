"""Chế độ Perceiver/TiTok: nối latent token học được vào chuỗi patch ngay sau patchify,
cho chúng đi qua CÙNG các block self-attention của trunk, rồi chỉ giữ latent làm z.

Lý do (docs/slot_attention_failure.md): mọi module pooling riêng đều có thang nhiệt độ riêng
để hiệu chỉnh sai; latent đi trong residual stream đã LayerNorm của trunk thì không có chỗ nào
để bão hoà. TiTok đạt rFID 2.21 với 32 token theo đúng cách này."""
from __future__ import annotations
import pytest
import torch
from mavt.model.backbone import HybridBackbone
from mavt.model.mavt import MAVT

TINY = dict(embed_dim=64, num_heads=4, num_blocks=2, patch_size=16, t_patch=2, latent_dim=8,
            semantic_dim=16, dec_dim=64, num_dec_attn_blocks=1, use_gradient_checkpointing=False)


def test_backbone_appends_latent_tokens_and_keeps_patch_count():
    b = HybridBackbone(dim=32, num_heads=4, num_blocks=2, num_latent_tokens=6)
    N = 16
    x = torch.randn(2, N, 32)
    pos = torch.zeros(N, 4, dtype=torch.long); pos[:, 1] = torch.arange(N) // 4; pos[:, 2] = torch.arange(N) % 4
    out = b(x, pos, torch.zeros(N, dtype=torch.long), "image")
    assert out.shape == (2, N + 6, 32), out.shape


def test_latent_tokens_are_learned_and_get_gradient():
    b = HybridBackbone(dim=32, num_heads=4, num_blocks=2, num_latent_tokens=6)
    assert b.latent_tokens is not None and b.latent_tokens.requires_grad
    N = 16
    pos = torch.zeros(N, 4, dtype=torch.long); pos[:, 1] = torch.arange(N) // 4; pos[:, 2] = torch.arange(N) % 4
    out = b(torch.randn(2, N, 32), pos, torch.zeros(N, dtype=torch.long), "image")
    out[:, N:].pow(2).mean().backward()
    assert b.latent_tokens.grad is not None and b.latent_tokens.grad.abs().sum() > 0


def test_backbone_without_latents_is_unchanged():
    b = HybridBackbone(dim=32, num_heads=4, num_blocks=2)
    assert b.latent_tokens is None
    N = 16
    pos = torch.zeros(N, 4, dtype=torch.long); pos[:, 1] = torch.arange(N) // 4; pos[:, 2] = torch.arange(N) % 4
    assert b(torch.randn(2, N, 32), pos, torch.zeros(N, dtype=torch.long), "image").shape == (2, N, 32)


@pytest.mark.parametrize("modality,shape", [("image", (1, 3, 64, 64)),
                                            ("video", (1, 3, 8, 64, 64)),
                                            ("threed", (1, 3, 3, 64, 64))])
def test_mavt_perceiver_mode_z_has_exactly_K_tokens_and_recon_matches_input(modality, shape):
    torch.manual_seed(0)
    m = MAVT(**TINY, num_latent_tokens=12)
    m.prepare_for_modalities([{"modality": modality, "resolution": 64, "frames": 8}])
    m.eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        out = m(x, modality, decode=True)
    assert out.z.shape == (1, 12, TINY["latent_dim"]), out.z.shape
    assert out.reconstruction.shape == x.shape
    assert out.semantic.shape == (1, TINY["semantic_dim"])


def test_perceiver_mode_bypasses_the_content_detail_split():
    """Không còn phép tách, nên không được để lại số hạng loss nào của nó."""
    torch.manual_seed(0)
    m = MAVT(**TINY, num_latent_tokens=12)
    m.prepare_for_modalities([{"modality": "image", "resolution": 64}])
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 64), "image", decode=False)
    assert float(out.cd_metrics["slot_diversity"]) == 0.0
    assert float(out.cd_metrics.get("content_recon_error", 0.0)) == 0.0
    assert out.latent_token_types is None and out.latent_positions is None


def test_default_model_still_uses_the_split():
    m = MAVT(**TINY)
    m.prepare_for_modalities([{"modality": "image", "resolution": 64}]); m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 64), "image", decode=False)
    assert out.latent_token_types is not None and out.z.shape[1] > 12
