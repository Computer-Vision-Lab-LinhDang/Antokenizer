"""Graph types, spatial edges and same-pixel track edges."""
from __future__ import annotations

import pytest
import torch

from mavt4d.graph.spatial import build_same_pixel_track_edges, build_spatial_edges
from mavt4d.graph.types import (EDGE_ATTR_DIM, CorrespondenceGraph, EdgeSet, EdgeType, Provenance, TokenLayout,
                                batch_graphs)


def _rc(layout: TokenLayout, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(frame, r, c) of camera token indices."""
    hw = layout.Hp * layout.Wp
    return idx // hw, (idx % hw) // layout.Wp, idx % layout.Wp


def test_layout_indexing_roundtrip():
    L = TokenLayout(V=2, T=3, Hp=4, Wp=5, S=2, K=6)
    assert (L.n_cam, L.n_hub, L.num_nodes) == (120, 12, 132)
    v, t, r, c = torch.meshgrid(*(torch.arange(n) for n in (2, 3, 4, 5)), indexing="ij")
    assert torch.equal(L.cam_index(v, t, r, c), torch.arange(L.n_cam).reshape(2, 3, 4, 5))
    s, k = torch.meshgrid(torch.arange(2), torch.arange(6), indexing="ij")
    assert torch.equal(L.hub_index(s, k), L.n_cam + torch.arange(L.n_hub).reshape(2, 6))
    assert L.cam_index(1, 2, 3, 4) == L.n_cam - 1 and L.hub_index(1, 5) == L.num_nodes - 1


def test_spatial_edges_window_and_cap():
    L = TokenLayout(V=2, T=2, Hp=6, Wp=7)
    e = build_spatial_edges(L, radius=2, cap=24)
    deg = torch.bincount(e.dst, minlength=L.n_cam).reshape(2, 2, 6, 7)
    assert (deg[:, :, 2:4, 2:5] == 24).all()
    assert (deg[:, :, [0, 0, -1, -1], [0, -1, 0, -1]] == 8).all()
    assert (deg[:, :, 0, 3] == 14).all()
    fs, rs, cs = _rc(L, e.src)
    fd, rd, cd = _rc(L, e.dst)
    assert (e.src != e.dst).all() and torch.equal(fs, fd)  # no self-loops, never across view-frames
    assert ((rs - rd).abs() <= 2).all() and ((cs - cd).abs() <= 2).all()
    capped = torch.bincount(build_spatial_edges(L, radius=2, cap=8).dst, minlength=L.n_cam).reshape(2, 2, 6, 7)
    assert (capped[:, :, 1:5, 1:6] == 8).all()


def test_spatial_provenance_grid_attr_zero():
    e = build_spatial_edges(TokenLayout(V=1, T=1, Hp=4, Wp=7))
    assert e.attr.shape == (len(e.src), EDGE_ATTR_DIM) and (e.attr == 0).all()
    assert (e.provenance == Provenance.GRID).all() and (e.weight == 1).all()
    assert e.src.dtype == e.dst.dtype == e.provenance.dtype == torch.int64 and e.weight.dtype == torch.float32


def test_same_pixel_track_edges():
    L = TokenLayout(V=2, T=3, Hp=2, Wp=3)
    e = build_same_pixel_track_edges(L, cap=2)
    hw, T = L.Hp * L.Wp, L.T
    fs, fd = e.src // hw, e.dst // hw
    assert torch.equal(e.src % hw, e.dst % hw)  # same (r, c)
    assert torch.equal(fs // T, fd // T) and ((fs % T - fd % T).abs() == 1).all()  # same camera, |dt| = 1
    deg = torch.bincount(e.dst, minlength=L.n_cam).reshape(2, 3, 2, 3)
    assert (deg[:, 0] == 1).all() and (deg[:, 1] == 2).all() and (deg[:, 2] == 1).all()
    assert (e.provenance == Provenance.GRID).all() and (e.attr == 0).all()


def test_empty_edgeset_shapes():
    e = EdgeSet.empty()
    assert e.src.shape == e.dst.shape == e.weight.shape == e.provenance.shape == (0,)
    assert e.attr.shape == (0, EDGE_ATTR_DIM)
    assert e.src.dtype == e.dst.dtype == e.provenance.dtype == torch.int64
    assert e.weight.dtype == e.attr.dtype == torch.float32


def test_batch_graphs_offsets():
    L = TokenLayout(V=1, T=2, Hp=2, Wp=3, S=1, K=4)
    g1 = CorrespondenceGraph(L, {EdgeType.SPATIAL: build_spatial_edges(L)})
    g2 = CorrespondenceGraph(L, {EdgeType.SPATIAL: build_spatial_edges(L, radius=1, cap=2),
                                 EdgeType.TRACK: build_same_pixel_track_edges(L)})
    b = batch_graphs([g1, g2])
    assert b.layout == L and b.num_graphs == 2
    for et in EdgeType:
        e1, e2, eb = g1.edges[et], g2.edges[et], b.edges[et]
        assert torch.equal(eb.src, torch.cat([e1.src, e2.src + L.num_nodes]))
        assert torch.equal(eb.dst, torch.cat([e1.dst, e2.dst + L.num_nodes]))
        assert torch.equal(b.in_degree(et), torch.cat([g1.in_degree(et), g2.in_degree(et)]))
    assert b.in_degree(EdgeType.TRACK).shape == (2 * L.num_nodes,)
    with pytest.raises(ValueError, match="layout"):
        batch_graphs([g1, CorrespondenceGraph(TokenLayout(V=1, T=2, Hp=2, Wp=4), {})])


def test_graph_always_has_five_keys():
    L = TokenLayout(V=1, T=1, Hp=2, Wp=3)
    g = CorrespondenceGraph(L, {EdgeType.SPATIAL: build_spatial_edges(L)})
    for graph in (g, batch_graphs([g, g]), g.to("cpu")):
        assert set(graph.edges) == set(EdgeType)
        assert all(len(graph.edges[et].src) == 0 for et in EdgeType if et != EdgeType.SPATIAL)
        assert graph.in_degree(EdgeType.NB3D).sum() == 0
