"""C / D state and latent heads of MAVT on DINOv3, v0 (tiny config, CPU)."""
import pytest
import torch

from mavt.dino_v0.cd_state import CDState, CBuilder, DState
from mavt.dino_v0.config import DinoV0Config, tiny_config
from mavt.dino_v0.latent import LatentHeads, SemanticPool, budget
from mavt.dino_v0.rope import grid_coords, modality_mask


def _inputs(cfg, B, T, requires_grad=False):
    N = cfg.n_patches
    X = torch.randn(B, T, N, cfg.dim, requires_grad=requires_grad)
    coords = torch.stack([grid_coords(cfg.grid, t=float(t)) for t in range(T)])
    mod = "image" if T == 1 else "video"
    mask = modality_mask(mod, N).expand(T, N, 4)
    return X, coords, mask


def _grad_frames(out, X):
    """Per-frame L1 norm of d(out.sum()) / dX -> [T]."""
    (g,) = torch.autograd.grad(out.sum(), X, retain_graph=True, allow_unused=True)
    if g is None:
        return torch.zeros(X.shape[1])
    return g.abs().sum(dim=(0, 2, 3))


def test_shapes_image():
    torch.manual_seed(0)
    cfg = tiny_config()
    m = CDState(cfg)
    X, coords, mask = _inputs(cfg, 2, 1)
    out = m(X, coords, mask)
    assert out["C"].shape == (2, 1, cfg.n_patches, cfg.dim)
    assert out["seg"].tolist() == [0]
    assert out["D"] is None and out["D_hat"] is None and out["D_next"] is None


def test_shapes_video():
    torch.manual_seed(0)
    cfg = tiny_config(keyframe_every=4)
    m = CDState(cfg)
    X, coords, mask = _inputs(cfg, 2, 6)
    out = m(X, coords, mask)
    M, d = cfg.d_tokens, cfg.d_dim
    assert out["C"].shape == (2, 2, cfg.n_patches, cfg.dim)
    assert out["seg"].dtype == torch.long and out["seg"].tolist() == [0, 0, 0, 0, 1, 1]
    for key in ("D", "D_hat", "D_next"):
        assert out[key].shape == (2, 6, M, d)
    assert m.d_state.up(out["D"]).shape == (2, 6, M, cfg.dim)


def test_mask_by_autograd():
    torch.manual_seed(0)
    cfg = tiny_config(keyframe_every=4)
    m = CDState(cfg)
    X, coords, mask = _inputs(cfg, 2, 6, requires_grad=True)
    out = m(X, coords, mask)

    # C_0 reads only its keyframe X_0; C_1 only X_4
    g = _grad_frames(out["C"][:, 0], X)
    assert g[0] > 0 and torch.all(g[1:] == 0)
    g = _grad_frames(out["C"][:, 1], X)
    assert g[4] > 0 and torch.all(g[torch.arange(6) != 4] == 0)

    # C never reads D
    (gd0,) = torch.autograd.grad(out["C"].sum(), m.d_state.d0, retain_graph=True, allow_unused=True)
    assert gd0 is None or torch.all(gd0 == 0)

    # D_t is causal: no grad from X_{t'} with t' > t, and it does read X_t
    for t in range(6):
        g = _grad_frames(out["D"][:, t], X)
        assert g[t] > 0
        assert torch.all(g[t + 1:] == 0), t

    # D_hat_t is made before X_t arrives: no grad from X_t or later, even at a keyframe
    # (the keyframe prediction reads the previous segment's C, not the C built from X_t)
    for t in range(1, 6):
        g = _grad_frames(out["D_hat"][:, t], X)
        assert torch.all(g[t:] == 0), t
    assert not out["D_hat"][:, 0].requires_grad or torch.all(_grad_frames(out["D_hat"][:, 0], X) == 0)

    # D_next_t reads up to X_t only
    for t in range(6):
        g = _grad_frames(out["D_next"][:, t], X)
        assert torch.all(g[t + 1:] == 0)


def test_c_neighbourhood():
    """With one block, a C cell reads X_k only within its 3 x 3 neighbourhood (self-attn mixes C0 only)."""
    torch.manual_seed(0)
    cfg = tiny_config(c_blocks=1)
    cb = CBuilder(cfg)
    N = cfg.n_patches
    x = torch.randn(1, N, cfg.dim, requires_grad=True)
    c = cb(x, grid_coords(cfg.grid), modality_mask("image", N))
    (g,) = torch.autograd.grad(c[0, 0].sum(), x)          # cell (0, 0)
    g = g[0].abs().sum(-1).view(cfg.grid, cfg.grid)
    assert torch.all(g[:2, :2] > 0)
    assert torch.all(g[2:] == 0) and torch.all(g[:, 2:] == 0)


def test_budget_full_config():
    cfg = DinoV0Config()
    assert budget(cfg, "image") == 12288
    assert budget(cfg, "video") == 12288


def test_latent_heads():
    torch.manual_seed(0)
    cfg = tiny_config()
    B, N, M = 2, cfg.n_patches, cfg.d_tokens
    heads = LatentHeads(cfg)
    C = torch.randn(B, N, cfg.dim)
    D = torch.randn(B, M, cfg.d_dim)
    z, mu, lv = heads.encode_c(C, "image")
    assert z.shape == (B, N, cfg.zr_c_image) and mu.shape == lv.shape == z.shape
    z, mu, lv = heads.encode_c(C, "video")
    assert z.shape == (B, N, cfg.zr_c_video)
    zd, mud, lvd = heads.encode_d(D)
    assert zd.shape == (B, M, cfg.zr_d)
    kl = LatentHeads.kl(mu, lv) + LatentHeads.kl(mud, lvd)
    assert kl.dim() == 0 and torch.isfinite(kl)
    assert not torch.equal(z, mu)                          # training: sampled
    heads.eval()
    z, mu, _ = heads.encode_c(C, "image")
    assert torch.equal(z, mu)
    zd, mud, _ = heads.encode_d(D)
    assert torch.equal(zd, mud)
    # logvar clamp
    big = torch.full((1, 1, cfg.d_dim), 1e4)
    _, _, lv = heads.encode_d(big)
    assert lv.min() >= -30 and lv.max() <= 20


def test_semantic_pool():
    torch.manual_seed(0)
    cfg = tiny_config()
    pool = SemanticPool(cfg)
    C = torch.randn(2, cfg.n_patches, cfg.dim)
    D_up = DState(cfg).up(torch.randn(2, cfg.d_tokens, cfg.d_dim))
    assert pool(C, D_up).shape == (2, cfg.zs_dim)
    assert pool(C, None).shape == (2, cfg.zs_dim)


@pytest.mark.parametrize("flag", ["use_track", "use_event_memory"])
def test_v1_not_implemented(flag):
    cfg = tiny_config(**{flag: True})
    for cls in (CDState, CBuilder, DState):
        with pytest.raises(NotImplementedError, match="v1"):
            cls(cfg)
