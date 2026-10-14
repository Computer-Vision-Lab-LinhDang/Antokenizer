"""D6 audit — ghim trạng thái bốn bug H2, H6, H8, H17 của MAVT v3 (E2.1 trên research board).

Quy ước: mỗi test FAIL khi bug còn, PASS khi đã sửa. Không dùng xfail — test đỏ là trạng thái thật.
Trạng thái tại 7a7dc82 (feat/v3-pipeline-local):
  H2  đã sửa (93bbe90)                                        -> test xanh
  H6  còn, tiềm ẩn (config hiện tại vuông, một bố cục mỗi N)  -> test đỏ
  H8  còn (chỉ ảnh hưởng chế độ slot, perceiver bỏ qua)       -> test đỏ
  H17 còn (cả hai chế độ, kể cả baseline perceiver 25.57 dB)  -> test đỏ
"""
from __future__ import annotations

import torch

from mavt.losses.losses import slot_diversity_loss
from mavt.model.backbone import HybridBackbone
from mavt.model.content_detail_split import ContentDetailSplit
from mavt.model.mavt import MAVT
from mavt.model.patchify import PatchifyEncoder
from mavt.model.rgat import build_adjacency

TINY = dict(embed_dim=64, num_heads=4, num_blocks=2, patch_size=16, t_patch=2, latent_dim=8,
            semantic_dim=16, dec_dim=64, num_dec_attn_blocks=1, use_gradient_checkpointing=False)


def test_h2_content_weights_normalise_over_slot_axis():
    """H2 — sai trục normalize. Dòng: content_detail_split.py:185, softmax(C @ x^T, dim=-2).

    Trọng số patch->slot phải cộng bằng 1 theo trục SLOT cho mỗi patch (x_approx là tổ hợp lồi).
    Bản 2026-09-02 chuẩn hoá theo trục patch (x_approx ~0.2x norm của x, cos 0.027); sửa ở 93bbe90.
    Dùng N_c != N để bản lỗi không thể vô tình đúng.

    Các chỗ normalize khác đã kiểm, đúng trục: rgat.py:167 softmax theo key; graph_attention.py:211
    softmax theo láng giềng (dim=2 của (B,N,K,H)); content_reconstruction_error trung bình theo patch;
    F.normalize(dim=-1) trong losses/cd_split; LayerNorm ở latent_heads.SemanticHead là code chết
    (MAVT dùng UnderstandingDecoder). Đã có test tương tự: test_content_branch.py.
    """
    torch.manual_seed(0)
    x, C = torch.randn(2, 16, 32), torch.randn(2, 5, 32)
    w = ContentDetailSplit._content_weights(x, C)                   # (B, N_c, N)
    assert w.shape == (2, 5, 16)
    per_patch = w.sum(dim=1)
    assert torch.allclose(per_patch, torch.ones_like(per_patch), atol=1e-5), "H2: softmax sai trục"


def test_h6_mask_cache_is_keyed_by_layout_not_only_n():
    """H6 — trùng cache key. Dòng: backbone.py:85 (_mask_cache) và :94 key = (modality, N).

    Ảnh 16x16 và 8x32 patch cùng N=256 nên lấy chung mask: đầu vào thứ hai chạy RGAT trên đồ thị
    láng giềng của bố cục thứ nhất. Va chạm thực tế khác: video (Tp,Hp,Wp)=(4,8,8) và (1,16,16).
    Ảnh hưởng cả baseline perceiver vì block RGAT vẫn đọc graph đã cache. Hiện tiềm ẩn: mọi config
    train dùng CenterCrop vuông và một độ phân giải mỗi modality, nên mỗi (modality, N) chỉ có một
    bố cục. Sẽ thành lỗi mỗi step khi bật multi-res / aspect ratio / token masking.
    """
    b = HybridBackbone(dim=32, num_heads=4, num_blocks=2, rgat_impl="dense")
    pos_a = PatchifyEncoder._image_positions(16, 16, "cpu")
    pos_b = PatchifyEncoder._image_positions(8, 32, "cpu")
    pid = torch.full((256,), -1, dtype=torch.long)
    b._get_masks(pos_a, pid, "image")
    adj_got, _ = b._get_masks(pos_b, pid, "image")
    adj_want, _ = build_adjacency(pos_b, pid, "image", b.r_s, b.r_t)
    wrong = (adj_got != adj_want).sum().item()
    assert wrong == 0, f"H6: 8x32 dùng mask của 16x16 ({wrong}/{adj_want.numel()} ô cạnh sai)"


def test_h8_slot_diversity_loss_has_gradient():
    """H8 — slot_diversity là no-op. Dòng: content_detail_split.py:306 (torch.no_grad) ->
    lightning_module.py:313 -> losses.py:336 slot_diversity_loss -> tổng loss với w_aux.

    Metric tính trong no_grad nên không có grad_fn: số hạng w_aux * l_div cộng vào loss nhưng
    gradient về slot bằng 0 — cơ chế chống collapse duy nhất không làm gì. Chỉ ảnh hưởng chế độ
    slot; perceiver (num_latent_tokens>0) đặt metric = hằng 0 nên không dùng tới.
    """
    torch.manual_seed(0)
    s = ContentDetailSplit(dim=32, num_heads=4, num_slot_layers=1)
    s.prepare_poolers(16, 16)
    _, metrics = s(torch.randn(2, 64, 32))
    div = metrics["slot_diversity"]
    assert div.requires_grad, "H8: slot_diversity tính trong torch.no_grad() -> loss không có gradient"
    slot_diversity_loss(div).backward()
    g = s._content_poolers["16_16"].slots.grad
    assert g is not None and g.abs().sum() > 0, "H8: gradient về slot bằng 0"


def test_h17_semantic_signal_keeps_a_token_axis():
    """H17 — một vector toàn cục làm nút cổ chai. Dòng: decoder.py:500-504
    (UnderstandingDecoder: một query attention-pool -> (B, semantic_dim)), mavt.py:199.

    Toàn bộ giám sát ngữ nghĩa lên z đi qua MỘT vector mỗi mẫu (ví dụ 768 số so với 4096 số latent
    ở perceiver 128x32). Head không có positional encoding, nên còn mù bố cục. w_dense không tính:
    dense_distill_loss lấy out.backbone_tokens (trước VAE), không đi qua z.
    Tiêu chí sửa: tín hiệu ngữ nghĩa suy từ z giữ trục token (semantic.ndim == 3, >1 token).
    Đúng ở cả hai chế độ (slot và perceiver = baseline 25.57 dB).
    """
    for k in (0, 12):
        torch.manual_seed(0)
        m = MAVT(**TINY, num_latent_tokens=k)
        m.prepare_for_modalities([{"modality": "image", "resolution": 64}])
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 64, 64), "image", decode=False)
        assert out.semantic.ndim == 3 and out.semantic.shape[1] > 1, (
            f"H17 (num_latent_tokens={k}): semantic {tuple(out.semantic.shape)} là một vector toàn cục "
            f"cho z {tuple(out.z.shape)}")
