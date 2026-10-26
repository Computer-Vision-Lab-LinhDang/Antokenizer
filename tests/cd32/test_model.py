"""C-D 3.2 model on the tiny config: shapes, the asymmetric mask, causality, degeneration, gradients."""
import math

import pytest
import torch

from mavt.cd32 import MAVTCD32, tiny_config
from mavt.cd32 import geometry as geo


def _model(**kw):
    torch.manual_seed(0)
    return MAVTCD32(tiny_config(**kw))


def _orbit(B, V, size):
    c2w = torch.zeros(B, V, 4, 4)
    for v in range(V):
        a = 2 * math.pi * v / V
        o = torch.tensor([2 * math.cos(a), 2 * math.sin(a), 0.3])
        z = -o / o.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, -1.0]))
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        c2w[:, v, :3, 0], c2w[:, v, :3, 1], c2w[:, v, :3, 2], c2w[:, v, :3, 3] = x, y, z, o
        c2w[:, v, 3, 3] = 1
    f = size / 2 / math.tan(0.3456)
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]).expand(B, V, 3, 3).clone()
    return dict(c2w=c2w, K=K)


@pytest.mark.parametrize("modality", ["image", "video", "threed"])
def test_step_finite_and_backward(modality):
    m = _model()
    if modality == "image":
        x, cam = torch.rand(2, 3, 64, 64) * 2 - 1, None
    else:
        T = 8 if modality == "video" else 4
        x, cam = torch.rand(2, 3, T, 64, 64) * 2 - 1, (_orbit(2, 4, 64) if modality == "threed" else None)
    out = m(x, modality, cam, return_outputs=True)
    assert torch.isfinite(out["loss"])
    assert all(torch.isfinite(v) for v in out["parts"].values())
    out["loss"].backward()
    assert out["outputs"]["rgb"].shape == m.to_frames(x, modality).shape
    grads = {n: p.grad for n, p in m.named_parameters() if p.requires_grad}
    assert grads["decoder.feat_head.1.weight"].abs().sum() > 0
    assert grads["backbone.c_blocks.0.qkv.weight"].abs().sum() > 0
    if modality == "video":
        for n in ("d_heads.pose.weight", "d_heads.flow.weight", "d_heads.link.weight", "enc_r.out.weight",
                  "decoder.innov.out.weight", "decoder.prev.out.weight"):
            assert grads[n] is not None and grads[n].abs().sum() > 0, n
    assert not any(n.startswith("encoder.dino.") for n, p in m.named_parameters() if p.requires_grad)


def test_video_latent_shapes_and_budget():
    cfg = tiny_config()
    m = _model().eval()
    x = torch.rand(1, 3, 8, 64, 64) * 2 - 1
    with torch.no_grad():
        o = m(x, "video", return_outputs=True)["outputs"]
    S, K, N = 2, cfg.keyframe_every, cfg.n_cells
    assert o["c"].shape == (S, N, cfg.d_c_video) and o["rho"].shape == (S, N) and o["fov"].shape == (S,)
    assert o["xi"].shape == (S, K - 1, 6) and o["M"].shape == (S, K - 1, cfg.flow_grid, cfg.flow_grid, 3)
    assert o["link"].shape == (S, 7) and o["tokens"].shape == (S, K - 1, cfg.L_max, cfg.d_r)
    assert (o["L"] == cfg.L_max).all()                               # eval: full prefix
    assert torch.allclose(o["rho"].median(-1).values, torch.zeros(S), atol=1e-6)   # median depth 1
    assert torch.allclose(o["fov"], torch.full((S,), cfg.fov_init), atol=1e-4)     # init 50 deg


def test_c_never_reads_d_or_other_frames():
    """Changing every non-key frame leaves C (content, rho, focal) of each segment unchanged."""
    m = _model().eval()
    x = torch.rand(1, 3, 8, 64, 64) * 2 - 1
    y = x.clone()
    y[:, :, [1, 2, 3, 5, 6, 7]] = torch.rand_like(y[:, :, [1, 2, 3, 5, 6, 7]]) * 2 - 1
    with torch.no_grad():
        a = m(x, "video", return_outputs=True)["outputs"]
        b = m(y, "video", return_outputs=True)["outputs"]
    for k in ("c", "rho", "fov"):
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    assert not torch.allclose(a["xi"], b["xi"]) or not torch.allclose(a["tokens"], b["tokens"])


def _perturb_heads(m):
    with torch.no_grad():                    # zero-init heads would hide any dependence
        for lin in (m.d_heads.pose, m.d_heads.flow, m.d_heads.link):
            lin.weight.normal_(0, 0.02)


def test_d_is_sparse_causal_within_a_segment():
    """D of frame t reads C, frame t-1 and frame t only: changing the last frame leaves earlier D unchanged."""
    m = _model().eval()
    _perturb_heads(m)
    x = torch.rand(1, 3, 4, 64, 64) * 2 - 1
    y = x.clone()
    y[:, :, 3] = torch.rand_like(y[:, :, 3]) * 2 - 1
    with torch.no_grad():
        a = m(x, "video", return_outputs=True)["outputs"]
        b = m(y, "video", return_outputs=True)["outputs"]
    assert torch.allclose(a["xi"][:, :2], b["xi"][:, :2], atol=1e-6)
    assert torch.allclose(a["tokens"][:, :2], b["tokens"][:, :2], atol=1e-6)
    assert not torch.allclose(a["xi"][:, 2], b["xi"][:, 2])


def test_segments_do_not_read_the_future():
    """Changing segment 1 leaves everything of segment 0 unchanged (the link of segment 1 reads C_0, not back)."""
    m = _model().eval()
    _perturb_heads(m)
    x = torch.rand(1, 3, 8, 64, 64) * 2 - 1
    y = x.clone()
    y[:, :, 4:] = torch.rand_like(y[:, :, 4:]) * 2 - 1
    with torch.no_grad():
        a = m(x, "video", return_outputs=True)["outputs"]
        b = m(y, "video", return_outputs=True)["outputs"]
    for k in ("c", "rho", "xi", "M", "tokens"):
        assert torch.allclose(a[k][0], b[k][0], atol=1e-6), k
    assert torch.allclose(a["rgb"][:, :4], b["rgb"][:, :4], atol=1e-5)
    assert not torch.allclose(a["link"][1], b["link"][1])


def test_image_degenerates_to_identity_read_and_rho_gets_no_rec_gradient():
    cfg = tiny_config(w_smooth=0.0)
    torch.manual_seed(0)
    m = MAVTCD32(cfg)
    with torch.no_grad():
        m.c_heads.rho_mlp.fc2.weight.normal_(0, 0.1)                 # a non-trivial depth map
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    out = m(x, "image")
    out["loss"].backward()
    assert m.c_heads.rho_mlp.fc2.weight.grad is None or m.c_heads.rho_mlp.fc2.weight.grad.abs().sum() == 0
    # the projection is the identity: every query's nearest neighbour is its own cell, at distance 0
    G = cfg.grid
    rho = torch.randn(2, G * G)
    K = geo.intrinsics_from_focal(torch.full((2,), 20.0), G)
    p, z, valid, cull = m.decoder.project(rho, K, torch.eye(3).expand(2, 3, 3), torch.zeros(2, 3))
    idx, d2, R2, _ = geo.knn_read(geo.cell_centres(G), p, valid, cfg.knn)
    assert torch.equal(idx[..., 0], torch.arange(G * G).expand(2, -1)) and d2[..., 0].max() < 1e-6


def test_l0_has_no_innovation_and_innovation_changes_the_frame():
    m = _model().eval()
    with torch.no_grad():
        m.decoder.innov.out.weight.normal_(0, 0.05)
    x = torch.rand(1, 3, 4, 64, 64) * 2 - 1
    with torch.no_grad():
        full = m(x, "video", L=torch.full((1, 3), m.cfg.L_max), return_outputs=True)["outputs"]["rgb"]
        none = m(x, "video", L=torch.zeros(1, 3, dtype=torch.long), return_outputs=True)["outputs"]["rgb"]
        h = m.decoder.innov(torch.randn(1, m.cfg.n_cells, m.cfg.dec_dim), torch.rand(1, m.cfg.n_cells),
                            torch.randn(1, m.cfg.L_max, m.cfg.d_r), torch.zeros(1, dtype=torch.long))
    assert h.abs().max() == 0
    assert torch.allclose(full[:, 0], none[:, 0])                   # keyframe: C only
    assert not torch.allclose(full[:, 1:], none[:, 1:])


def test_threed_uses_known_poses_without_d_geom():
    m = _model()
    x = torch.rand(2, 3, 4, 64, 64) * 2 - 1
    out = m(x, "threed", _orbit(2, 4, 64), return_outputs=True)
    assert out["diag"]["numbers"] <= m.cfg.segment_numbers("threed")
    out["loss"].backward()
    assert all(p.grad is None for p in m.d_heads.parameters())     # pose / flow heads unused
    assert m.c_heads.alpha.grad is not None                        # cross-view rho refinement in the graph


def test_shared_streams_option_b():
    m = _model(shared_streams=True)
    assert m.backbone.d_blocks is m.backbone.c_blocks
    out = m(torch.rand(1, 3, 4, 64, 64) * 2 - 1, "video")
    assert torch.isfinite(out["loss"])


def test_gradient_checkpointing_matches():
    torch.manual_seed(0)
    a = MAVTCD32(tiny_config())
    torch.manual_seed(0)
    b = MAVTCD32(tiny_config(), use_checkpoint=True)
    x = torch.rand(1, 3, 8, 64, 64) * 2 - 1
    losses = []
    for m in (a, b):
        m.train()
        torch.manual_seed(1)                                         # same noise and L draws
        loss = m(x, "video")["loss"]
        loss.backward()
        losses.append(loss.detach())
    assert torch.allclose(losses[0], losses[1], atol=1e-5)
    ga, gb = a.backbone.c_blocks[0].qkv.weight.grad, b.backbone.c_blocks[0].qkv.weight.grad
    assert torch.allclose(ga, gb, atol=1e-5)


def test_bf16_autocast_keeps_geometry_float32():
    m = _model()
    rho = torch.randn(1, m.cfg.n_cells)
    K = geo.intrinsics_from_focal(torch.tensor([4.0]), m.cfg.grid)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        p, z, valid, cull = m.decoder.project(rho, K, torch.eye(3)[None], torch.zeros(1, 3))
        R, tau = geo.pose_matrix(torch.randn(1, 6))
        out = m(torch.rand(1, 3, 4, 64, 64) * 2 - 1, "threed", _orbit(1, 4, 64))
    assert p.dtype == z.dtype == R.dtype == torch.float32
    assert torch.isfinite(out["loss"])


def test_lpips_keys_are_not_missing_on_load():
    m = _model()
    m.lpips = torch.nn.Linear(1, 1)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("lpips.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "lpips" in k] and not unexpected
