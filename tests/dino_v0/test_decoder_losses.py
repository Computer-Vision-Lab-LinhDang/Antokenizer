import torch

from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.decoder import (QueryDecoder, gather_window, image_to_patches, patches_to_image,
                                  query_coords)
from mavt.dino_v0.losses import feat_loss, gate_reg, kl, rgb_l1, sem_loss, var_cov
from mavt.dino_v0.rope import modality_mask

FEAT = 32


def _inputs(cfg, modality, B=2, Q=None):
    torch.manual_seed(0)
    N, G = cfg.n_patches, cfg.grid
    cc = cfg.zr_c_image if modality == "image" else cfg.zr_c_video
    zc = torch.randn(B, N, cc)
    zd = None if modality == "image" else torch.randn(B, cfg.d_tokens, cfg.zr_d)
    idx = None if Q is None else torch.randperm(N)[:Q]
    qc = query_coords(G, 1.0 if modality == "video" else 0.0, B, idx)
    qm = modality_mask(modality, qc.shape[1]).unsqueeze(0).expand(B, -1, -1)
    return zc, zd, qc, qm


def test_shapes_image():
    cfg = tiny_config()
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "image")
    out = dec(zc, zd, qc, qm, "image")
    B, Q = qc.shape[:2]
    assert out["rgb"].shape == (B, Q, cfg.patch * cfg.patch * 3)
    assert out["feat"].shape == (B, Q, FEAT)
    assert out["delta"].shape == (B, Q, 2) and torch.all(out["delta"] == 0)
    assert out["gate"].shape == (B, Q, 1) and torch.all(out["gate"] == 1)


def test_shapes_video_and_zero_delta_at_init():
    cfg = tiny_config()
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "video", Q=cfg.train_queries_per_frame)
    out = dec(zc, zd, qc, qm, "video")
    B, Q = qc.shape[:2]
    assert out["rgb"].shape == (B, Q, cfg.patch * cfg.patch * 3)
    assert out["feat"].shape == (B, Q, FEAT)
    assert out["gate"].shape == (B, Q, 1)
    assert torch.all(out["delta"] == 0)
    assert torch.all((out["gate"] > 0) & (out["gate"] < 1))


def test_delta_receives_gradient_through_rope():
    cfg = tiny_config()
    dec = QueryDecoder(cfg, FEAT)
    with torch.no_grad():
        dec.offset[-1].weight.normal_(0, 1e-3)
    zc, zd, qc, qm = _inputs(cfg, "video")
    out = dec(zc, zd, qc, qm, "video")
    out["delta"].retain_grad()
    rgb_l1(out["rgb"], torch.zeros_like(out["rgb"])).backward()
    assert out["delta"].grad is not None and out["delta"].grad.abs().sum() > 0
    assert dec.offset[-1].weight.grad.abs().sum() > 0


def test_delta_override_is_used():
    cfg = tiny_config()
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "video")
    dlt = torch.full_like(qc[..., 1:3], 1.0 / cfg.grid)
    a = dec(zc, zd, qc, qm, "video")
    b = dec(zc, zd, qc, qm, "video", delta_override=dlt)
    assert torch.equal(b["delta"], dlt)
    assert not torch.allclose(a["rgb"], b["rgb"])


def test_gather_shift_one_cell_right():
    G, r = 8, 1
    qc = query_coords(G, 0.0, 1)                                 # all cells
    centre = qc[..., 1:3]
    shifted = centre + torch.tensor([1.0 / G, 0.0])
    i0, v0, c0 = gather_window(centre, G, r)
    i1, v1, c1 = gather_window(shifted, G, r)
    # interior cells: every window index moves by exactly one column
    row, col = 3, 3
    q = row * G + col
    assert v0[0, q].all() and v1[0, q].all()
    assert torch.equal(i1[0, q], i0[0, q] + 1)
    assert torch.allclose(c1[0, q, :, 0], c0[0, q, :, 0] + 1.0 / G)
    assert torch.equal(i0[0, q].view(3, 3)[1, 1], torch.tensor(q))
    # right edge: the centre cell is clamped, the window is masked past the edge
    qe = row * G + (G - 1)
    assert torch.equal(i1[0, qe], i0[0, qe])
    assert v0[0, qe].sum() == 6


def test_patch_round_trip():
    img = torch.randn(2, 3, 32, 32)
    p = image_to_patches(img, 8)
    assert p.shape == (2, 16, 8 * 8 * 3)
    assert torch.equal(patches_to_image(p, 4, 8), img)
    # row-major: patch 1 is the top row, second column
    assert torch.equal(p[0, 1].view(8, 8, 3).permute(2, 0, 1), img[0, :, :8, 8:16])


def test_losses():
    torch.manual_seed(0)
    const = torch.ones(2, 16, 8)
    rand = torch.randn(2, 64, 8)
    assert var_cov(const) > var_cov(rand)
    z = torch.zeros(3, 5)
    assert kl(z, z).abs().item() == 0.0
    x = torch.randn(4, 32)
    assert sem_loss(x, x).item() < 1e-5
    assert feat_loss(x, x).item() < 1e-5
    assert gate_reg(torch.ones(2, 3, 1)).item() == 0.0
    assert rgb_l1(x, x).item() == 0.0
