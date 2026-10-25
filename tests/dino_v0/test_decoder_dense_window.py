"""The dense-masked C read of the decoder equals the original gather-the-window read (fwd and grads)."""
import torch

from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.decoder import UV_MASK, QueryDecoder, gather_window, query_coords
from mavt.dino_v0.rope import modality_mask


def _gather_forward(dec, zc, zd, qc, qm, modality):
    """QueryDecoder.forward as it was before the dense read (window gathered per query)."""
    B, Q = qc.shape[:2]
    c = dec.in_c_image(zc) if modality == "image" else dec.in_c_video(zc)
    d = None if zd is None else dec.in_d(zd)
    delta = dec.compute_delta(d, qc)
    centre = qc[..., 1:3] + delta
    idx, valid, cell = gather_window(centre, dec.grid, dec.r)
    q_m = qm * torch.tensor(UV_MASK, dtype=qm.dtype)
    q_pos = torch.cat([qc[..., :1], centre, qc[..., 3:]], -1)
    W = idx.shape[-1]
    k_pos = torch.cat([qc[..., :1].unsqueeze(2).expand(-1, -1, W, -1), cell, torch.zeros_like(cell[..., :1])], -1)
    k_m = q_m.unsqueeze(2).expand(-1, -1, W, -1)
    h = dec.query.expand(B, Q, -1)
    for blk in dec.blocks:
        hn = blk.n_q(h)
        rc = blk.read_c.forward_gather(hn, blk.n_c(c), idx, valid, q_pos, k_pos, q_m, k_m)
        if d is None:
            h = h + rc
        else:
            rd = blk.read_d(blk.n_qd(h), blk.n_d(d))
            o = torch.sigmoid(blk.gate(torch.cat([hn, rc], -1)))
            h = h + o * rc + (1 - o) * rd
        hs = blk.n_s(h)
        h = h + blk.self_attn(hs, hs, qc, qm, qc, qm)
        h = h + blk.mlp(blk.n_m(h))
    h = dec.norm(h)
    return dict(rgb=dec.rgb_head(h), feat=dec.feat_head(h), delta=delta)


def test_dense_window_matches_gather():
    torch.manual_seed(0)
    cfg = tiny_config()
    dec = QueryDecoder(cfg, feat_dim=16).double()                # float64 (RoPE still rotates in float32)
    torch.nn.init.normal_(dec.offset[-1].weight, std=0.5)       # non-zero Δ: centres off the grid cells
    B, N = 3, cfg.n_patches
    for modality, cc in (("image", cfg.zr_c_image), ("video", cfg.zr_c_video)):
        zc = torch.randn(B, N, cc, dtype=torch.float64, requires_grad=True)
        zd = None if modality == "image" else torch.randn(B, cfg.d_tokens, cfg.zr_d, dtype=torch.float64, requires_grad=True)
        idx = torch.rand(B, N).argsort(-1)[:, :5]
        qc = query_coords(cfg.grid, 2.0, B, idx).double()
        qm = modality_mask(modality, qc.shape[1]).unsqueeze(0).expand(B, -1, -1).double()
        new = dec(zc, zd, qc, qm, modality)
        ref = _gather_forward(dec, zc, zd, qc, qm, modality)
        assert new["delta"].abs().max() > 0.05 or modality == "image"
        for k in ("rgb", "feat"):
            torch.testing.assert_close(new[k], ref[k], atol=1e-6, rtol=1e-5)
        w = torch.randn_like(new["rgb"])
        ins = [zc] + ([zd] if zd is not None else []) + [dec.offset[0].weight, dec.blocks[0].read_c.kv.weight]
        g_new = torch.autograd.grad((new["rgb"] * w).sum(), ins, allow_unused=True)
        g_ref = torch.autograd.grad((ref["rgb"] * w).sum(), ins, allow_unused=True)
        for a, b in zip(g_new, g_ref):
            if a is None:
                assert b is None
                continue
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
