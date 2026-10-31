"""ROCm speed: the two gathers whose backward dominated a training step (profile 2026-10-31) become matmuls.
- geometry.gather_rows(t, idx) = t[b, idx] as a one-hot bmm (backward: a matmul, not index_put_ with accumulate);
  GrainSplat reads its windows through it.
- PrevRead reads its 9 taps with geometry.read_coarse (bilinear weights @ tokens) instead of F.grid_sample.
Both must give the values and gradients of the old code. Tiny config, CPU."""
import pytest
import torch
import torch.nn.functional as F

from mavt.grain_prior import geometry as g
from mavt.grain_prior import tiny_config
from mavt.grain_prior.readout import PrevRead

from .test_splat_projected import PROJ, _grains, _splat


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_rows_equals_indexing_values_and_gradients(dtype):
    torch.manual_seed(0)
    B, G = 3, 11
    t = torch.randn(B, G, 4, 5, dtype=dtype, requires_grad=True)
    idx = torch.randint(0, G, (B, 6, 7))
    bi = torch.arange(B).view(-1, 1, 1)
    ref = t[bi, idx]
    out = g.gather_rows(t, idx)
    assert out.dtype == dtype and out.shape == ref.shape and torch.equal(out, ref)
    w = torch.randn_like(ref.float())
    t32 = t.detach().float().requires_grad_(True)                     # the exact gradient: fp32 indexing
    (gr,) = torch.autograd.grad((t32[bi, idx] * w).sum(), t32)
    (go,) = torch.autograd.grad((out.float() * w).sum(), t)
    # bf16: duplicates are summed in fp32 by the bmm, then rounded once (index_put_ sums them in bf16)
    tol = dict(rtol=1e-2, atol=1e-2) if dtype == torch.bfloat16 else dict(rtol=0, atol=1e-6)
    assert torch.allclose(go.float(), gr, **tol)


def test_gather_rows_keeps_a_non_finite_row_to_itself():
    t = torch.arange(12.0).view(1, 4, 3)
    t[0, 2] = float("inf")
    idx = torch.tensor([[0, 1, 3, 1]])
    assert torch.equal(g.gather_rows(t, idx), t[0, idx[0]].unsqueeze(0))       # row 2 is not read: no NaN spread


def test_splat_reads_its_windows_through_gather_rows(monkeypatch):
    calls = []
    real = g.gather_rows
    monkeypatch.setattr(g, "gather_rows", lambda t, idx: calls.append(1) or real(t, idx))
    sp = _splat()
    cfg = sp.cfg
    u, sigma, depth, valid = _grains(cfg, shift=0.3)
    B, G = u.shape[:2]
    out, _ = sp(torch.randn(B, G, cfg.d_f), u, depth, sigma, valid, torch.ones(B),
                torch.randn(B, cfg.n_coarse, cfg.d_coarse), torch.randn(B, cfg.n_coarse, cfg.d_coarse),
                g.cell_centres(cfg.query_grid).unsqueeze(0).expand(B, -1, -1))
    assert calls and torch.isfinite(out).all()


def _prev_read_reference(pr, F_prev, u_q, u_now, u_prev, valid_now, valid_prev, sigma):
    """The PrevRead read before 2026-10-31 (F.grid_sample on the projected previous feature)."""
    B = F_prev.shape[0]
    c = pr.centres(u_q, u_now, u_prev, valid_now, valid_prev, sigma)
    img = pr.proj(F_prev.detach().to(pr.proj.weight.dtype)).transpose(1, 2).reshape(B, -1, pr.grid, pr.grid)
    pts = (c.unsqueeze(2) + pr.offsets) * 2 - 1
    taps = F.grid_sample(img.float(), pts, mode="bilinear", padding_mode="border", align_corners=False)
    return pr.mix(taps.permute(0, 2, 3, 1).flatten(2).to(pr.mix.weight.dtype))


def test_prev_read_matches_grid_sample_and_does_not_call_it(monkeypatch):
    torch.manual_seed(0)
    cfg = tiny_config(read_prev_frame=True)
    pr = PrevRead(cfg, 16)
    torch.nn.init.normal_(pr.mix.weight, 0, 0.1)
    F_prev = torch.randn(2, cfg.n_cells, 16)
    uq = torch.rand(2, 9, 2) * 1.2 - 0.1                                      # some queries off the grid (border)
    u_now = torch.rand(2, 5, 2).requires_grad_(True)
    u_prev = torch.rand(2, 5, 2)
    valid, s = torch.ones(2, 5, dtype=torch.bool), torch.full((2, 5), 0.5)
    ref = _prev_read_reference(pr, F_prev, uq, u_now, u_prev, valid, valid, s)
    (g_ref_u,) = torch.autograd.grad(ref.sum(), u_now, retain_graph=True)
    g_ref_w = torch.autograd.grad(ref.sum(), pr.proj.weight)[0]
    monkeypatch.setattr(F, "grid_sample", lambda *a, **k: (_ for _ in ()).throw(AssertionError("grid_sample")))
    out = pr(F_prev, uq, u_now, u_prev, valid, valid, s)
    assert torch.allclose(out, ref, atol=1e-5)
    (g_u,) = torch.autograd.grad(out.sum(), u_now, retain_graph=True)
    g_w = torch.autograd.grad(out.sum(), pr.proj.weight)[0]
    assert torch.allclose(g_u, g_ref_u, atol=1e-4) and torch.allclose(g_w, g_ref_w, atol=1e-4)


def test_grain_delta_footprint_sample_is_detached_from_the_grain_position(monkeypatch):
    """The RGB footprint of 3.13 is read at a detached position: no grid_sampler backward (the code still reads the
    pixels where the grain went; the transport is trained by the other terms)."""
    from mavt.grain_prior.delta import GrainDelta
    cfg = tiny_config(grain_df_k=3)
    gd = GrainDelta(cfg, 16)
    seen = []
    real = F.grid_sample
    monkeypatch.setattr(F, "grid_sample", lambda x, pts, **k: seen.append(pts.requires_grad) or real(x, pts, **k))
    B, G = 2, cfg.n_grains
    u = torch.rand(B, G, 2, requires_grad=True)
    code = gd.encode(torch.randn(B, cfg.n_cells, 16), torch.randn(B, cfg.n_cells, 16), torch.rand(B, 3, 64, 64),
                     torch.randn(B, G, cfg.d_f), u, torch.ones(B, G, dtype=torch.bool))
    assert seen == [False]
    code.sum().backward()
    assert u.grad is not None and u.grad.abs().sum() > 0                 # the bilinear feature reads still train u
