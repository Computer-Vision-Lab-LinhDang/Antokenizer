"""v1 phase-1 decoder options: per-query Δ (delta_mode "query"), conv rgb head (rgb_head "conv"),
warp-only decode; default flags must stay the v0 decoder (same keys, same init, same outputs)."""
import pytest
import torch
import torch.nn as nn

from mavt.dino_v0.camera import ZeroLinear
from mavt.dino_v0.config import DinoV0Config, tiny_config
from mavt.dino_v0.decoder import (UV_MASK, DecoderBlock, QueryDecoder, fourier_uv, image_to_patches,
                                  query_coords, window_bias)
from mavt.dino_v0.rope import AxialRoPE, grid_coords, modality_mask

FEAT = 16
CONV = dict(rgb_head="conv", conv_channels=(32, 16, 16, 8, 8), conv_res_blocks=1)   # 4 x2 stages: 4 -> 64


class _V0Decoder(nn.Module):
    """Frozen copy of the v0 QueryDecoder (__init__ / forward as of commit 57aba35)."""

    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.ray_q = ZeroLinear(6, cfg.dec_dim)
        d = cfg.dec_dim
        self.cfg, self.grid, self.r = cfg, cfg.grid, cfg.read_window
        self.in_c_image = nn.Linear(cfg.zr_c_image, d)
        self.in_c_video = nn.Linear(cfg.zr_c_video, d)
        self.in_d = nn.Linear(cfg.zr_d, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.n_fourier = 8
        self.offset = nn.Sequential(nn.Linear(d + 2 + 4 * self.n_fourier, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.rope = AxialRoPE(d // cfg.dec_heads)
        self.blocks = nn.ModuleList([DecoderBlock(cfg, self.rope) for _ in range(cfg.dec_blocks)])
        self.norm = nn.LayerNorm(d)
        self.rgb_head = nn.Linear(d, cfg.patch * cfg.patch * 3)
        self.feat_head = nn.Linear(d, feat_dim)

    def forward(self, zc, zd, q_coords, q_mask, modality, q_rays=None):
        B, Q = q_coords.shape[:2]
        c = self.in_c_image(zc) if modality == "image" else self.in_c_video(zc)
        d = None if zd is None else self.in_d(zd)
        if d is None:
            delta = torch.zeros_like(q_coords[..., 1:3])
        else:
            dm = d.mean(1, keepdim=True).expand(-1, Q, -1)
            delta = self.offset(torch.cat([dm, fourier_uv(q_coords[..., 1:3], self.n_fourier).to(dm.dtype)], -1))
        centre = q_coords[..., 1:3] + delta
        q_m = q_mask * torch.tensor(UV_MASK, device=q_mask.device, dtype=q_mask.dtype)
        q_pos = torch.cat([q_coords[..., :1], centre, q_coords[..., 3:]], -1)
        N = c.shape[1]
        c_pos = grid_coords(self.grid, 0.0, device=q_coords.device).to(q_coords.dtype)
        c_pos = torch.cat([q_coords[:, :1, :1].expand(-1, N, -1), c_pos[None, :, 1:].expand(B, -1, -1)], -1)
        c_m = q_m[:, :1].expand(-1, N, -1)
        win = (window_bias(centre, self.grid, self.r, c.dtype), q_pos, q_m, c_pos, c_m)
        h = self.query.expand(B, Q, -1).to(c.dtype)
        if q_rays is not None:
            h = h + self.ray_q(q_rays.to(c.dtype))
        gate = None
        for blk in self.blocks:
            h, gate = blk(h, c, d, win, q_coords, q_mask)
        h = self.norm(h)
        return dict(rgb=self.rgb_head(h), feat=self.feat_head(h), delta=delta, gate=gate)


def _inputs(cfg, modality, B=2, idx=None, t=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = cfg.n_patches
    cc = cfg.zr_c_image if modality == "image" else cfg.zr_c_video
    zc = torch.randn(B, N, cc, generator=g)
    zd = None if modality == "image" else torch.randn(B, cfg.d_tokens, cfg.zr_d, generator=g)
    qc = query_coords(cfg.grid, t, B, idx)
    qm = modality_mask(modality, qc.shape[1]).unsqueeze(0).expand(B, -1, -1)
    return zc, zd, qc, qm


def _flat_inputs(cfg, B, T, seed=0):
    """B * T flattened video frames: row b * T + t carries frame t's coords."""
    g = torch.Generator().manual_seed(seed)
    N = cfg.n_patches
    zc = torch.randn(B * T, N, cfg.zr_c_video, generator=g)
    zd = torch.randn(B * T, cfg.d_tokens, cfg.zr_d, generator=g)
    qc = torch.stack([query_coords(cfg.grid, float(t), B) for t in range(T)], 1).flatten(0, 1)   # [B*T,N,4]
    qm = modality_mask("video", N).unsqueeze(0).expand(B * T, -1, -1)
    return zc, zd, qc, qm


def _nonzero_delta(dec, std=0.3):
    last = dec.offset[-1] if dec.cfg.delta_mode == "mean" else dec.delta_mlp[-1]
    with torch.no_grad():
        last.weight.normal_(0, std)


# ---- default flags = v0 --------------------------------------------------------------------

def test_default_flags_match_v0_bit_identical():
    cfg = tiny_config(dec_blocks=2)
    torch.manual_seed(0)
    ref = _V0Decoder(cfg, FEAT)
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    sr, sd = ref.state_dict(), dec.state_dict()
    assert list(sr) == list(sd)                                 # same keys, same order
    assert all(torch.equal(sr[k], sd[k]) for k in sr)           # same init RNG stream
    _nonzero_delta(ref)
    dec.load_state_dict(ref.state_dict(), strict=True)          # v0 checkpoint loads strict
    ref.load_state_dict(dec.state_dict(), strict=True)
    for modality in ("image", "video"):
        for idx in (None, torch.randperm(cfg.n_patches)[:5]):
            zc, zd, qc, qm = _inputs(cfg, modality, B=3, idx=idx)
            rays = torch.randn(3, qc.shape[1], 6)
            for r in (None, rays):
                a, b = ref(zc, zd, qc, qm, modality, q_rays=r), dec(zc, zd, qc, qm, modality, q_rays=r)
                assert set(b) == {"rgb", "image", "feat", "delta", "gate"} and b["image"] is None
                for k in a:
                    assert torch.equal(a[k], b[k]), (modality, k)
            if modality == "video":
                assert b["delta"].abs().max() > 0.05


def test_flags_create_params_only_when_on():
    keys = lambda **kw: set(QueryDecoder(tiny_config(**kw), FEAT).state_dict())
    v0 = keys()
    q = keys(delta_mode="query")
    cv = keys(**CONV)
    assert not any(k.startswith(("delta_", "rgb_conv")) for k in v0)
    assert not any(k.startswith("offset") for k in q) and any(k.startswith("delta_read") for k in q)
    assert not any(k.startswith("rgb_head") for k in cv) and any(k.startswith("rgb_conv") for k in cv)
    assert q - {k for k in q if k.startswith("delta_")} == v0 - {k for k in v0 if k.startswith("offset")}
    with pytest.raises(AssertionError):
        QueryDecoder(tiny_config(delta_mode="nope"), FEAT)


# ---- conv rgb head -------------------------------------------------------------------------

@pytest.mark.parametrize("delta_mode", ["mean", "query"])
def test_conv_head_shapes(delta_mode):
    cfg = tiny_config(delta_mode=delta_mode, **CONV)
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    N, P, S = cfg.n_patches, cfg.patch * cfg.patch * 3, cfg.image_size
    zc, zd, qc, qm = _inputs(cfg, "image", B=2, t=0.0)
    out = dec(zc, zd, qc, qm, "image")
    assert out["image"].shape == (2, 3, S, S) and out["rgb"].shape == (2, N, P)
    assert torch.equal(out["rgb"], image_to_patches(out["image"], cfg.patch))
    assert out["feat"].shape == (2, N, FEAT)
    assert out["delta"].shape == (2, N, 2) and torch.all(out["delta"] == 0)
    assert out["gate"].shape == (2, N, 1) and torch.all(out["gate"] == 1)
    B, T = 2, 3                                                  # B * T flattened frames
    zc, zd, qc, qm = _flat_inputs(cfg, B, T)
    zc.requires_grad_(True)
    out = dec(zc, zd, qc, qm, "video")
    assert out["image"].shape == (B * T, 3, S, S) and out["rgb"].shape == (B * T, N, P)
    assert out["feat"].shape == (B * T, N, FEAT) and out["delta"].shape == (B * T, N, 2)
    assert out["gate"].shape == (B * T, N, 1)
    out["rgb"].abs().mean().backward()
    assert zc.grad.abs().sum() > 0
    assert dec.rgb_conv.conv_in.weight.grad.abs().sum() > 0


def test_conv_head_asserts_full_grid():
    cfg = tiny_config(**CONV)
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "video", idx=torch.randperm(cfg.n_patches)[:5])
    with pytest.raises(AssertionError):
        dec(zc, zd, qc, qm, "video")
    with pytest.raises(AssertionError):                          # 3 x2 stages cannot reach 4 -> 64
        QueryDecoder(tiny_config(rgb_head="conv", conv_channels=(16, 8, 8, 8)), FEAT)


def test_conv_head_param_count_real_config():
    cfg = DinoV0Config(dino_repo=None, rgb_head="conv")
    dec = QueryDecoder(cfg, cfg.dino_dim)
    n = sum(p.numel() for p in dec.rgb_conv.parameters())
    print(f"\nconv rgb head @ dec_dim {cfg.dec_dim}, conv_channels {cfg.conv_channels}, "
          f"res_blocks {cfg.conv_res_blocks}: {n / 1e6:.2f}M params")
    assert len(cfg.conv_channels) - 1 == 4                       # 16 -> 256 in 4 x2 stages
    assert 10e6 < n < 60e6


@pytest.mark.parametrize("flags", [dict(), dict(delta_mode="query"), dict(delta_mode="query", **CONV)])
def test_flattened_batch_equals_per_frame(flags):
    """Rows are independent: one call on B * T frames == T calls on B frames."""
    cfg = tiny_config(**flags)
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    _nonzero_delta(dec)
    B, T = 2, 3
    zc, zd, qc, qm = _flat_inputs(cfg, B, T)
    flat = dec(zc, zd, qc, qm, "video")
    for t in range(T):
        rows = torch.arange(B) * T + t
        one = dec(zc[rows], zd[rows], qc[rows], qm[rows], "video")
        for k in ("rgb", "feat", "delta", "gate"):
            torch.testing.assert_close(flat[k][rows], one[k], atol=1e-5, rtol=1e-4)
    assert flat["delta"].abs().max() > 0.01


# ---- per-query Δ ---------------------------------------------------------------------------

def test_query_delta_zero_at_init_then_per_query():
    cfg = tiny_config(delta_mode="query")
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "video", B=2)
    out = dec(zc, zd, qc, qm, "video")
    assert out["delta"].shape == (2, cfg.n_patches, 2) and torch.all(out["delta"] == 0)
    assert torch.all(dec(zc, None, qc, modality_mask("video", qc.shape[1]).expand(2, -1, -1), "video")["delta"] == 0)

    reads = []
    dec.delta_read.register_forward_hook(lambda m, i, o: reads.append(o))
    _nonzero_delta(dec, std=0.1)
    zd = zd.clone().requires_grad_(True)
    delta = dec(zc, zd, qc, qm, "video")["delta"]
    assert (delta - delta.mean(1, keepdim=True)).abs().max() > 1e-3     # Δ differs between queries
    rd = reads[-1]                                                      # [B, Q, D]: each query's own D read
    assert (rd - rd.mean(1, keepdim=True)).abs().max() > 1e-3
    g, = torch.autograd.grad((delta * torch.randn_like(delta)).sum(), zd)
    assert g.abs().sum() > 0                                            # zd -> Δ
    # the D-driven part of Δ is per query: changing zd moves different queries differently
    with torch.no_grad():
        dd = dec(zc, zd + torch.randn_like(zd), qc, qm, "video")["delta"] - delta
    assert (dd - dd.mean(1, keepdim=True)).abs().max() > 1e-4


def test_query_delta_rays_and_grad_to_last_layer():
    cfg = tiny_config(delta_mode="query")
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "video", B=2)
    rays = torch.randn(2, cfg.n_patches, 6)
    out = dec(zc, zd, qc, qm, "video", q_rays=rays)
    out["rgb"].abs().mean().backward()
    assert dec.delta_mlp[-1].weight.grad.abs().sum() > 0               # zero-init Δ still learns
    _nonzero_delta(dec)
    dec.zero_grad()
    out = dec(zc, zd, qc, qm, "video", q_rays=rays)
    out["rgb"].abs().mean().backward()
    assert dec.delta_ray.weight.grad.abs().sum() > 0                    # zero-init ray input learns


# ---- warp-only decode ----------------------------------------------------------------------

def _perturb_read_d(dec):
    with torch.no_grad():
        for blk in dec.blocks:
            for m in (blk.read_d, blk.n_qd, blk.n_d, blk.gate):
                for p in m.parameters():
                    p.add_(torch.randn_like(p))


@pytest.mark.parametrize("delta_mode", ["mean", "query"])
def test_warp_only(delta_mode):
    cfg = tiny_config(delta_mode=delta_mode, dec_blocks=2)
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    zc, zd, qc, qm = _inputs(cfg, "video", B=2)
    zd2 = zd + torch.randn_like(zd)
    # Δ = 0 at init: D has no path to the warp-only output at all
    a, b = dec(zc, zd, qc, qm, "video", warp_only=True), dec(zc, zd2, qc, qm, "video", warp_only=True)
    assert torch.equal(a["rgb"], b["rgb"]) and torch.all(a["gate"] == 1)

    _nonzero_delta(dec)
    w = dec(zc, zd, qc, qm, "video", warp_only=True)
    full = dec(zc, zd, qc, qm, "video")
    assert torch.all(w["gate"] == 1) and w["gate"].shape == (2, cfg.n_patches, 1)
    assert torch.equal(w["delta"], full["delta"]) and w["delta"].abs().max() > 0.05
    assert not torch.allclose(w["rgb"], full["rgb"])
    assert not torch.allclose(dec(zc, zd2, qc, qm, "video", warp_only=True)["rgb"], w["rgb"])   # zd acts via Δ

    _perturb_read_d(dec)                                                # readD / gate weights do nothing
    w2 = dec(zc, zd, qc, qm, "video", warp_only=True)
    for k in ("rgb", "feat", "delta", "gate"):
        assert torch.equal(w[k], w2[k]), k
    assert not torch.allclose(dec(zc, zd, qc, qm, "video")["rgb"], full["rgb"])

    zd = zd.clone().requires_grad_(True)                                # gradient to D only through Δ
    dec.zero_grad()
    dec(zc, zd, qc, qm, "video", warp_only=True)["rgb"].abs().mean().backward()
    assert zd.grad.abs().sum() > 0
    assert all(p.grad is None for blk in dec.blocks for p in blk.read_d.parameters())


def test_warp_only_without_zd_is_a_no_op():
    cfg = tiny_config(delta_mode="query")
    torch.manual_seed(0)
    dec = QueryDecoder(cfg, FEAT)
    for modality in ("image", "video"):
        zc, _, qc, qm = _inputs(cfg, modality, B=2)
        a, b = dec(zc, None, qc, qm, modality), dec(zc, None, qc, qm, modality, warp_only=True)
        for k in ("rgb", "feat", "delta", "gate"):
            assert torch.equal(a[k], b[k])
