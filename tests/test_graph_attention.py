"""Contracts for sparse relational graph attention over the 4D lattice."""
import torch
from mavt.model.patchify import PatchifyEncoder
from mavt.model.rgat import RGAT4DBlock, build_adjacency
from mavt.model.graph_attention import SparseRGAT4D, build_graph, graph_from_dense


def _tokens(x, modality, dim=64):
    enc = PatchifyEncoder(embed_dim=dim, patch_size=16, t_patch=2).eval()
    with torch.no_grad():
        return enc(x, modality)


def test_sparse_matches_dense_rgat_on_same_edges():
    torch.manual_seed(0)
    dim, heads = 64, 4
    tok, pos, pid = _tokens(torch.randn(1, 3, 64, 64), "image", dim)
    adj, masks = build_adjacency(pos, pid, "image", r_s=2, r_t=1)
    dense = RGAT4DBlock(dim, heads).eval()
    with torch.no_grad():
        dense.out_proj.weight.normal_(0, 0.02)
    sparse = SparseRGAT4D(dim, heads, per_type_kv=True).eval()
    sparse.load_from_dense(dense)
    g = graph_from_dense(masks)
    x = torch.randn(2, 16, dim)
    with torch.no_grad():
        y_dense = dense(x, adj, masks); y_sparse = sparse(x, g)
    assert torch.allclose(y_dense, y_sparse, atol=1e-5), (y_dense - y_sparse).abs().max()


def test_scales_to_long_video_without_dense_tensors():
    dim, heads = 128, 8
    _, pos, pid = _tokens(torch.randn(1, 3, 16, 256, 256), "video", dim)   # N = 2048
    g = build_graph(pos, pid, "video", r_s=2, r_t=1)
    assert g.nbr_idx.shape[0] == 2048 and g.nbr_idx.shape[1] <= 40
    blk = SparseRGAT4D(dim, heads)
    x = torch.randn(2, 2048, dim, requires_grad=True)
    y = blk(x, g); y.mean().backward()
    assert y.shape == x.shape and x.grad is not None


def test_plane_local_spatial_gives_2d_neighbourhood_on_all_planes():
    _, pos, pid = _tokens(torch.randn(1, 3, 3, 128, 128), "threed")
    g = build_graph(pos, pid, "threed", r_s=2, r_t=1, plane_local_spatial=True)
    spatial = (g.nbr_type == 0) & g.valid
    per_plane = [spatial[pid == p].sum(1).float().mean().item() for p in range(3)]
    assert all(k > 12 for k in per_plane), per_plane


def test_projection_cross_plane_edges_are_geometric_and_small():
    _, pos, pid = _tokens(torch.randn(1, 3, 3, 128, 128), "threed")
    g_any = build_graph(pos, pid, "threed", cross_mode="shared_axis")
    g_proj = build_graph(pos, pid, "threed", cross_mode="projection")
    k_any = ((g_any.nbr_type == 3) & g_any.valid).sum(1).float().mean().item()
    k_proj = ((g_proj.nbr_type == 3) & g_proj.valid).sum(1).float().mean().item()
    assert k_proj < k_any * 0.5 and 8 <= k_proj <= 24, (k_any, k_proj)
