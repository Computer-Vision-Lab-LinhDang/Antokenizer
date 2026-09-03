"""Nhánh content phải thực sự xấp xỉ feature (đo 2026-09-03: nó bắt 6.3% năng lượng,
trần PCA-64 trong ảnh là 91%; xoá nó chỉ mất 1.16 dB dù chiếm 50% ngân sách latent ở window 2).

Hai lỗi cấu trúc: (1) softmax chuẩn hoá theo trục patch nên x_approx không phải tổ hợp lồi
per-patch và không thể đạt scale của x; (2) không có số hạng loss nào ép slot mang thông tin."""
from __future__ import annotations
import torch
import pytest
from mavt.model.content_detail_split import ContentDetailSplit
from mavt.losses.losses import MAVTLoss


def _split(dim=32, n_c_ratio=0.25):
    torch.manual_seed(0)
    s = ContentDetailSplit(dim=dim, num_heads=4, num_slot_layers=1,
                           local_detail_window_size=2, local_detail_temporal_window_size=1)
    s.prepare_poolers(int(64 * n_c_ratio), int(64 * n_c_ratio))
    return s


def test_content_weights_form_a_per_patch_convex_combination():
    """Mỗi patch phải được dựng lại bằng tổ hợp lồi của các slot: trọng số theo slot cộng lại bằng 1."""
    s = _split()
    x = torch.randn(2, 64, 32)
    w = s._content_weights(x, s._get_content_pooler(16, 16)(x))     # (B, N_c, N)
    per_patch = w.sum(dim=-2)                                        # cộng theo slot, cho mỗi patch
    assert torch.allclose(per_patch, torch.ones_like(per_patch), atol=1e-5), per_patch[0, :5]


def test_split_reports_content_reconstruction_error():
    s = _split()
    x = torch.randn(2, 64, 32)
    _, metrics, _, _ = s(x, content_ratio=0.25, detail_ratio=0.25, return_metadata=True)
    assert "content_recon_error" in metrics
    e = float(metrics["content_recon_error"])
    assert 0.0 <= e, e


def test_content_error_is_trainable_and_decreases_on_low_rank_input():
    """Với input hạng thấp, tối ưu pooler phải giảm được sai số xấp xỉ — nếu không, đường gradient hỏng."""
    s = _split()
    torch.manual_seed(1)
    basis = torch.randn(4, 32)
    x = (torch.randn(2, 64, 4) @ basis)                              # hạng 4, 16 slot thừa sức bao
    opt = torch.optim.Adam(s.parameters(), lr=1e-2)
    def err():
        _, m, _, _ = s(x, content_ratio=0.25, detail_ratio=0.25, return_metadata=True)
        return m["content_recon_error"]
    e0 = float(err())
    for _ in range(60):
        opt.zero_grad(); e = err(); e.backward(); opt.step()
    e1 = float(err())
    assert e1 < e0 * 0.7, f"sai số không giảm: {e0:.3f} → {e1:.3f}"


def test_loss_exposes_content_term_and_adds_it_to_total():
    lf = MAVTLoss(w_content=0.5, use_lpips=False)
    kw = dict(pred=torch.rand(2, 3, 32, 32), target=torch.rand(2, 3, 32, 32),
              loss_kl=torch.tensor(0.0), slot_diversity=torch.tensor(0.0), modality="image")
    out = lf(**kw, content_recon_error=torch.tensor(0.8))
    base = lf(**kw)
    assert "loss_content" in out and float(out["loss_content"]) == pytest.approx(0.8)
    assert float(out["loss"]) > float(base["loss"])
