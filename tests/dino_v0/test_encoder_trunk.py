"""Tier A (FrameEncoder) and tier B (Trunk) of MAVT on DINOv3 v0, with the stub encoder on CPU."""
import torch

from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.encoder import FrameEncoder, PixelStem
from mavt.dino_v0.rope import grid_coords, modality_mask
from mavt.dino_v0.trunk import Trunk


def _coords(cfg, T, modality="video"):
    c = torch.stack([grid_coords(cfg.grid, t=float(t)) for t in range(T)])
    m = torch.stack([modality_mask(modality, cfg.n_patches) for _ in range(T)])
    return c, m


def test_frame_encoder_shapes_and_frozen():
    torch.manual_seed(0)
    cfg = tiny_config()
    enc = FrameEncoder(cfg).train()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    out = enc(x)
    N, D = cfg.n_patches, cfg.stub_dino_dim
    assert N == 16 and enc.dino.out_dim == D
    assert out["x"].shape == (2, N, cfg.dim)
    assert out["feat_target"].shape == (2, N, D) and out["cls"].shape == (2, D)
    assert PixelStem(cfg)(x).shape == (2, N, cfg.pix_dim)
    assert all(not p.requires_grad for p in enc.dino.parameters())
    assert not enc.dino.training and not enc.dino.model.training
    assert not out["feat_target"].requires_grad and not out["cls"].requires_grad
    assert out["x"].requires_grad
    e = out["mean_energy"].item()
    assert out["mean_energy"].dim() == 0 and 0.0 <= e <= 1.0
    out["x"].sum().backward()
    assert all(p.grad is None for p in enc.dino.parameters())
    assert enc.proj.weight.grad is not None and enc.stem.net[0].weight.grad is not None


def test_trunk_shapes():
    cfg = tiny_config()
    trunk = Trunk(cfg)
    for T in (1, 3):
        c, m = _coords(cfg, T)
        x = torch.randn(2, T, cfg.n_patches, cfg.dim)
        assert trunk(x, c, m).shape == x.shape
    ck = Trunk(cfg, use_checkpoint=True).train()
    x = torch.randn(1, 3, cfg.n_patches, cfg.dim, requires_grad=True)
    ck(x, *_coords(cfg, 3)).sum().backward()
    assert x.grad is not None


def test_trunk_causal():
    torch.manual_seed(0)
    cfg = tiny_config()
    trunk = Trunk(cfg).eval()
    c, m = _coords(cfg, 4)
    x = torch.randn(1, 4, cfg.n_patches, cfg.dim)
    y = trunk(x, c, m)
    x2 = x.clone()
    x2[:, 2:] += torch.randn_like(x2[:, 2:])
    y2 = trunk(x2, c, m)
    assert torch.allclose(y[:, :2], y2[:, :2], atol=1e-5)
    assert not torch.allclose(y[:, 2:], y2[:, 2:], atol=1e-3)


def test_trunk_window_locality():
    torch.manual_seed(0)
    cfg = tiny_config(trunk_layers=1, temporal_window=1)
    trunk = Trunk(cfg).eval()
    c, m = _coords(cfg, 2)
    x = torch.randn(1, 2, cfg.n_patches, cfg.dim)
    y = trunk(x, c, m)
    q = 0                                       # cell (0, 0); window = cells 0, 1, 4, 5 of frame 0
    for j, inside in ((5, True), (1, True), (2, False), (15, False)):
        x2 = x.clone()
        x2[0, 0, j] += torch.randn(cfg.dim)
        y2 = trunk(x2, c, m)
        changed = not torch.allclose(y[0, 1, q], y2[0, 1, q], atol=1e-5)
        assert changed == inside, (j, inside)


def test_rope_image_mask_ignores_t():
    torch.manual_seed(0)
    cfg = tiny_config()
    trunk = Trunk(cfg).eval()
    x = torch.randn(2, 1, cfg.n_patches, cfg.dim)
    c, m = _coords(cfg, 1, "image")
    c2 = c.clone()
    c2[..., 0] = 37.5
    assert torch.allclose(trunk(x, c, m), trunk(x, c2, m), atol=1e-5)
    # with the video mask the t value matters only through differences, so shift one frame of two
    cv, mv = _coords(cfg, 2, "video")
    cv2 = cv.clone()
    cv2[1, :, 0] += 3.0
    xv = torch.randn(1, 2, cfg.n_patches, cfg.dim)
    assert not torch.allclose(trunk(xv, cv, mv)[:, 1], trunk(xv, cv2, mv)[:, 1], atol=1e-4)
