"""geometry.read_coarse / losses.sample_cells without grid_sample: bilinear weights (align_corners=False, border
padding, token centres at (j + 0.5) / side) as a [B, M, side^2] matrix times the tokens. Same values and gradients
(tokens and positions) as F.grid_sample; the backward is two bmm instead of grid_sampler_2d_backward (atomic adds
into a 4 x 4 map on ROCm: the top GPU op of a training step)."""
import torch
import torch.nn.functional as F

from mavt.grain_prior import geometry as g
from mavt.grain_prior.losses import sample_cells


def _ref_read(tokens, u, side):
    B, _, C = tokens.shape
    img = tokens.transpose(1, 2).reshape(B, C, side, side)
    grid = (u.to(img.dtype) * 2 - 1).unsqueeze(1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(2).transpose(1, 2)


def test_bilinear_weights_sum_to_one_and_have_four_taps():
    u = torch.rand(2, 50, 2) * 1.4 - 0.2                     # inside and outside the unit square
    W = g.bilinear_weights(u, 4)
    assert W.shape == (2, 50, 16)
    assert torch.allclose(W.sum(-1), torch.ones(2, 50), atol=1e-6)
    assert ((W > 0).sum(-1) <= 4).all() and (W >= 0).all()


def test_read_coarse_matches_grid_sample_values_and_gradients():
    torch.manual_seed(0)
    for side, M in ((4, 300), (16, 64)):
        tokens = torch.randn(3, side * side, 8, requires_grad=True)
        u = (torch.rand(3, M, 2) * 1.4 - 0.2).requires_grad_(True)
        tokens2, u2 = tokens.detach().clone().requires_grad_(True), u.detach().clone().requires_grad_(True)
        out, ref = g.read_coarse(tokens, u, side), _ref_read(tokens2, u2, side)
        assert torch.allclose(out, ref, atol=1e-5), (side, (out - ref).abs().max())
        w = torch.randn_like(out)
        (out * w).sum().backward()
        (ref * w).sum().backward()
        assert torch.allclose(tokens.grad, tokens2.grad, atol=1e-5)
        assert torch.allclose(u.grad, u2.grad, atol=1e-4), (u.grad - u2.grad).abs().max()


def test_read_coarse_keeps_the_token_dtype():
    tokens = torch.randn(2, 16, 8).to(torch.bfloat16)
    out = g.read_coarse(tokens, torch.rand(2, 10, 2), 4)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 10, 8)


def test_sample_cells_matches_grid_sample():
    torch.manual_seed(1)
    x = torch.randn(2, 16 * 16, requires_grad=True)
    u = torch.rand(2, 40, 2) * 1.2 - 0.1
    out = sample_cells(x, u, 16)
    img = x.detach().view(2, 1, 16, 16)
    ref = F.grid_sample(img, (u * 2 - 1).unsqueeze(1), mode="bilinear", padding_mode="border", align_corners=False).view(2, -1)
    assert out.shape == (2, 40) and torch.allclose(out, ref, atol=1e-5)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
