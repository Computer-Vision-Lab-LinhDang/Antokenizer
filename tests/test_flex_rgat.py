"""FlexRGAT4D contracts (need a GPU for the FlexAttention kernel)."""
import pytest
import torch
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs a GPU")
from mavt.model.patchify import PatchifyEncoder
from mavt.model.graph_attention import SparseRGAT4D, build_graph
from mavt.model.flex_rgat import FlexRGAT4D, build_flex_graph


def _tokens(x, modality, dim):
    enc = PatchifyEncoder(embed_dim=dim, patch_size=16, t_patch=2).eval()
    with torch.no_grad():
        return enc(x, modality)


def _randomise_biases(m):
    with torch.no_grad():
        for p in (m.edge_bias, m.rel_t, m.rel_x, m.rel_y, m.rel_z):
            p.normal_(0, 0.5)
        m.out_proj.weight.normal_(0, 0.02)


def test_flex_cpu_fallback_matches_sparse():
    """On CPU FlexRGAT4D must fall back to the gather path with identical results."""
    torch.manual_seed(0)
    dim, heads = 64, 4
    _, pos, pid = _tokens(torch.randn(1, 3, 4, 64, 64), "video", dim)
    ref = SparseRGAT4D(dim, heads).eval(); _randomise_biases(ref)
    flex = FlexRGAT4D(dim, heads).eval(); flex.load_state_dict(ref.state_dict())
    gs = build_graph(pos, pid, "video", 2, 1); gf = build_flex_graph(pos, pid, "video", 2, 1)
    x = torch.randn(2, pos.shape[0], dim)
    with torch.no_grad():
        assert torch.allclose(ref(x, gs), flex(x, gf), atol=1e-5)


@cuda
def test_flex_matches_sparse_shared_kv_with_biases():
    torch.manual_seed(0)
    dim, heads = 128, 8
    _, pos, pid = _tokens(torch.randn(1, 3, 8, 128, 128), "video", dim)
    ref = SparseRGAT4D(dim, heads).cuda().eval(); _randomise_biases(ref)
    flex = FlexRGAT4D(dim, heads).cuda().eval(); flex.load_state_dict(ref.state_dict())
    gs = build_graph(pos, pid, "video", 2, 1, plane_local_spatial=True, cross_mode="projection")
    gf = build_flex_graph(pos, pid, "video", 2, 1, plane_local_spatial=True, cross_mode="projection")
    x = torch.randn(2, 256, dim, device="cuda")
    with torch.no_grad():
        assert torch.allclose(ref(x, gs), flex(x, gf), atol=2e-3)


@cuda
def test_biases_receive_gradient_through_flex_kernel():
    dim, heads = 128, 8
    _, pos, pid = _tokens(torch.randn(1, 3, 256, 256), "image", dim)
    blk = FlexRGAT4D(dim, heads).cuda()
    with torch.no_grad():
        blk.out_proj.weight.normal_(0, 0.02)
    g = build_flex_graph(pos, pid, "image", 2, 1)
    x = torch.randn(2, 256, dim, device="cuda", requires_grad=True)
    blk(x, g).square().mean().backward()
    assert blk.edge_bias.grad is not None and blk.edge_bias.grad.abs().sum() > 0
    assert blk.rel_x.grad is not None and blk.rel_x.grad.abs().sum() > 0


@cuda
def test_video16f_block_memory_under_4gb():
    dim, heads = 1152, 16
    _, pos, pid = _tokens(torch.randn(1, 3, 16, 256, 256), "video", dim)
    blk = FlexRGAT4D(dim, heads).cuda(); g = build_flex_graph(pos, pid, "video", 2, 1)
    x = torch.randn(4, 2048, dim, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        blk(x, g).float().mean().backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        blk(x, g).float().mean().backward()
    assert torch.cuda.max_memory_allocated() / 1e9 < 4.0


def test_backbone_flex_flag_runs():
    from mavt.model.backbone import HybridBackbone
    dim = 128
    bb = HybridBackbone(dim=dim, num_heads=8, num_blocks=4, rgat_impl="flex",
                        edge_plane_local=True, edge_cross_mode="projection")
    tok, pos, pid = _tokens(torch.randn(2, 3, 3, 128, 128), "threed", dim)
    assert bb(tok, pos, pid, "threed").shape == tok.shape


@cuda
def test_repeated_forward_does_not_recompile_and_is_fast():
    """Same graph, many steps: score_mod identity is stable so flex compiles once."""
    import time
    dim, heads = 1152, 16
    _, pos, pid = _tokens(torch.randn(1, 3, 256, 256), "image", dim)
    blk = FlexRGAT4D(dim, heads).cuda(); g = build_flex_graph(pos, pid, "image", 2, 1)
    x = torch.randn(8, 256, dim, device="cuda", requires_grad=True)
    assert blk._score_mod_for(g, x.device) is blk._score_mod_for(g, x.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        blk(x, g).float().mean().backward()                      # compile
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            blk(x, g).float().mean().backward()
        torch.cuda.synchronize()
    per_step_ms = (time.time() - t0) / 10 * 1000
    assert per_step_ms < 100, f"{per_step_ms:.0f} ms/step — recompiling or eager fallback"
