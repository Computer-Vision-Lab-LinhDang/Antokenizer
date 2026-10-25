"""v1 speed changes against the v0 math (CPU, fp32 unless stated): RoPE tables, the 4-D trunk attention,
SDPA in the C / D Attention, the mixed-modality step (+ DDP, world size 1), fused AdamW resume, compile."""
import argparse
import importlib.util
import os
import socket

import pytest
import torch
import torch.nn.functional as F

from mavt.dino_v0.cd_state import Attention, CBuilder
from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.model import MAVTDinoV0
from mavt.dino_v0.rope import AxialRoPE, grid_coords, modality_mask, rope_tables
from mavt.dino_v0.trunk import Trunk, _shift

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _coords(cfg, T, modality="video"):
    c = torch.stack([grid_coords(cfg.grid, t=float(t)) for t in range(T)])
    m = torch.stack([modality_mask(modality, cfg.n_patches) for _ in range(T)])
    return c, m


def _close(a, b, tol):
    """Max abs difference, scaled by max|b| for large tensors (parameter grads sum over many tokens)."""
    return (a - b).abs().max().item() <= tol * max(1.0, b.abs().max().item())


# ---------------------------------------------------------------------------- v0 references
def _v0_rotate(x, ang):
    """AxialRoPE.rotate of v0: float32, interleaved pairs (0::2, 1::2), stack + flatten."""
    if ang.dim() == 2:
        ang = ang.unsqueeze(0)
    cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)


def _v0_qkv(L, x, ang, ang_prev):
    B, T, N, D = x.shape
    q, k, v = L.qkv(x).view(B, T, N, 3, L.h, L.hd).permute(3, 0, 1, 4, 2, 5)
    rot = lambda z, a: _v0_rotate(z.reshape(B * T, L.h, N, L.hd), a.repeat(B, 1, 1)).view(B, T, L.h, N, L.hd)
    q, k_cur, k_prev = rot(q, ang), rot(k, ang), rot(_shift(k, 1), ang_prev)
    return q, torch.cat([k_cur, k_prev], 3), torch.cat([v, _shift(v, 1)], 3)


def _v0_trunk(trunk, x, c, m):
    """Trunk.forward of v0: one 5-D SDPA (math path) over keys [x_t ; x_{t-1}] with the bool mask [T,1,N,2N]."""
    B, T, N, D = x.shape
    ang = trunk.layers[0].rope.angles(c, m)
    ang_prev = _shift(ang, 0)
    am = trunk.attn_mask(T, N)
    stats = []
    for L in trunk.layers:
        h = L.norm1(x)
        q, k, v = _v0_qkv(L, h, ang, ang_prev)
        if L.R > 0:
            a, st = L._rel_core(h, q, k, v, am)
            stats.append(st)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
            a = L.out(o.transpose(2, 3).reshape(B, T, N, D))
        x = x + a
        x = x + L.mlp(L.norm2(x))
    return trunk.norm(x), stats


def _v0_attention(att, xq, xkv, ang, mask):
    """cd_state.Attention of v0 with a mask: explicit float32 q k^T + mask, softmax."""
    B, Nq, D = xq.shape
    q = att.q(xq).view(B, Nq, att.heads, att.head_dim).transpose(1, 2)
    k, v = att.kv(xkv).view(B, -1, 2, att.heads, att.head_dim).permute(2, 0, 3, 1, 4)
    q, k = _v0_rotate(q, ang), _v0_rotate(k, ang)
    a = (q.float() @ k.float().transpose(-2, -1)) * att.head_dim ** -0.5 + mask.float()
    o = (a.softmax(-1) @ v.float()).to(v.dtype)
    return att.out(o.transpose(1, 2).reshape(B, Nq, D))


# ---------------------------------------------------------------------------- RoPE
def test_rope_matches_v0():
    torch.manual_seed(0)
    rope = AxialRoPE(64)
    c, m = _coords(tiny_config(image_size=256), 3)
    ang = rope.angles(c, m)[1].clone().requires_grad_(True)               # [N, 32], decoder-style grad on angles
    x = torch.randn(2, 4, 256, 64, requires_grad=True)
    w = torch.randn(2, 4, 256, 64)
    for new in (rope.rotate(x, ang), rope.rotate(x, rope_tables(ang)), rope.rotate(x, rope_tables(ang, torch.float32))):
        ref = _v0_rotate(x, ang)
        assert (new - ref).abs().max() <= 1e-6                           # same pairing, float rounding only
        g_new = torch.autograd.grad((new * w).sum(), (x, ang))
        g_ref = torch.autograd.grad((ref * w).sum(), (x, ang))
        for a, b in zip(g_new, g_ref):
            assert _close(a, b, 1e-6)
    # bf16 tables (trunk / C under autocast): within one bf16 rounding step of |x| of v0's fp32 rotation
    xb = (torch.randn(2, 4, 256, 64) * 4).bfloat16()
    new = rope.rotate(xb, rope_tables(ang.detach(), torch.bfloat16))
    ref = _v0_rotate(xb, ang.detach())
    assert new.dtype == torch.bfloat16
    assert (new.float() - ref.float()).abs().max() <= 2 ** -7 * xb.float().abs().max()


# ---------------------------------------------------------------------------- trunk
def _trunk_pair_check(cfg, T, modality, train=False, checkpoint=False):
    torch.manual_seed(0)
    trunk = Trunk(cfg, use_checkpoint=checkpoint).train(train)
    c, m = _coords(cfg, T, modality)
    x = torch.randn(2, T, cfg.n_patches, cfg.dim)
    w = torch.randn_like(x)
    params = [p for p in trunk.parameters()]
    outs = []
    for fn in (lambda xx: (trunk(xx, c, m), trunk._rel_stats), lambda xx: _v0_trunk(trunk, xx, c, m)):
        xx = x.clone().requires_grad_(True)
        torch.manual_seed(1)                                             # same Gumbel noise (R2, train)
        y, stats = fn(xx)
        loss = (y * w).sum()
        if stats:
            loss = loss - torch.stack(stats)[:, 0].mean()
        outs.append((y.detach(), torch.autograd.grad(loss, [xx] + params, allow_unused=True)))
    (y1, g1), (y0, g0) = outs
    assert (y1 - y0).abs().max() <= 1e-5
    for a, b in zip(g1, g0):
        assert (a is None) == (b is None)
        if a is not None:
            assert _close(a, b, 1e-5)


@pytest.mark.parametrize("T,modality", [(1, "image"), (4, "video")])
def test_trunk_matches_v0(T, modality):
    _trunk_pair_check(tiny_config(), T, modality)
    _trunk_pair_check(tiny_config(image_size=128, temporal_window=2), T, modality, train=True, checkpoint=True)


@pytest.mark.parametrize("T", [1, 3])
def test_trunk_r2_matches_v0(T):
    cfg = tiny_config(rgat_relations=4, rgat_rank=8)
    _trunk_pair_check(cfg, T, "video", train=True)
    _trunk_pair_check(cfg, T, "video", train=False)


def test_trunk_mask_cached_and_same_window():
    cfg = tiny_config(image_size=128)
    trunk = Trunk(cfg)
    N = cfg.n_patches
    b1 = trunk._cached("bias", 0, N, torch.device("cpu"), torch.float32)
    assert trunk._cached("bias", 0, N, torch.device("cpu"), torch.float32) is b1     # built once
    am = trunk.attn_mask(3, N)[1, 0]                                     # v0, frame t >= 1: [cur | prev window]
    assert torch.equal(b1[:, :N] == 0, am[:, N:]) and torch.all(b1[:, N:] == 0)       # v1: [prev window | cur]
    assert torch.isinf(b1).any() and not torch.isnan(b1).any()


# ---------------------------------------------------------------------------- C / D attention
def test_cd_attention_sdpa_matches_explicit():
    torch.manual_seed(0)
    cfg = tiny_config()
    cb = CBuilder(cfg)
    att = Attention(cfg.dim, cfg.heads)
    N = cfg.n_patches
    xq = torch.randn(3, N, cfg.dim, requires_grad=True)
    xkv = torch.randn(3, N, cfg.dim, requires_grad=True)
    ang = cb.rope.angles(grid_coords(cfg.grid), modality_mask("image", N))
    w = torch.randn(3, N, cfg.dim)
    ins = [xq, xkv] + list(att.parameters())
    for a_in in (ang, rope_tables(ang, torch.float32)):
        new = att(xq, xkv, cb.rope, a_in, a_in, attn_mask=cb.nbr)
        ref = _v0_attention(att, xq, xkv, ang, cb.nbr)
        assert (new - ref).abs().max() <= 1e-6
        g_new = torch.autograd.grad((new * w).sum(), ins, retain_graph=True)
        g_ref = torch.autograd.grad((ref * w).sum(), ins)
        for a, b in zip(g_new, g_ref):
            assert _close(a, b, 1e-5)
        # blocked keys get exactly zero gradient (cell 0 reads only its 2 x 2 corner of X_k)
        (g,) = torch.autograd.grad(new[0, 0].sum(), xkv)
        g = g[0].abs().sum(-1).view(cfg.grid, cfg.grid)
        assert torch.all(g[2:] == 0) and torch.all(g[:, 2:] == 0) and torch.all(g[:2, :2] > 0)


# ---------------------------------------------------------------------------- mixed step / DDP
def _batches(tr, cfg, T=3, seed=0):
    out = {}
    for m, B in (("image", 2), ("video", 2), ("threed", 1)):
        out[m] = tr.to_inputs(next(tr.synthetic_iter(m, B, cfg, T, seed)), m, torch.device("cpu"))
    return out


def test_mixed_step_every_param_gets_grad():
    tr = _script("train_dino_v0")
    cfg = tiny_config()
    torch.manual_seed(0)
    model = MAVTDinoV0(cfg).train()
    w = tr.mixed_weights(None, tr.MODALITIES)
    assert w == {m: pytest.approx(1 / 3) for m in tr.MODALITIES}
    assert tr.mixed_weights("image:2,video:1,threed:1", {"image", "video"}) == pytest.approx(dict(image=2 / 3, video=1 / 3))
    step = tr.MixedStep(model, w)
    out = step(_batches(tr, cfg))
    out["loss"].backward()
    never = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert never == []                                                   # nothing for DDP to ignore
    assert torch.allclose(out["loss"].detach(), sum(out["parts"][f"{m}/loss"] for m in tr.MODALITIES) / 3)
    assert {"image/loss_rgb", "video/loss_fut", "threed/loss_fut", "video/diag_gate_mean"} <= set(out["parts"]) | set(out["diag"])
    # the probe agrees, and does flag what an image-only step leaves unused
    args = argparse.Namespace(n_frames=3, n_views=2, seed=0)
    assert tr.probe_unused(step, cfg, args, torch.device("cpu"), False) == []
    only_image = tr.probe_unused(tr.MixedStep(model, dict(image=1.0)), cfg, args, torch.device("cpu"), False)
    assert "model.cd.d_state.d0" in only_image and "model.latent.c_video.weight" in only_image
    assert all(p.grad is None for p in model.parameters())               # probe leaves no grads behind


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mixed_step_ddp_world1(tmp_path):
    """DDP(find_unused_parameters=False, static_graph, gradient_as_bucket_view, no buffer sync) around the mixed
    step with activation checkpointing: two steps run and the grads equal those of the bare module."""
    import torch.distributed as dist
    tr = _script("train_dino_v0")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        cfg = tiny_config()
        w = tr.mixed_weights(None, tr.MODALITIES)
        torch.manual_seed(0)
        ref = tr.MixedStep(MAVTDinoV0(cfg, use_checkpoint=True).train(), w)
        torch.manual_seed(0)
        step = tr.MixedStep(MAVTDinoV0(cfg, use_checkpoint=True).train(), w)
        ddp, desc = tr.mixed_ddp(step, [], None, static_graph=True)
        assert "find_unused_parameters=False static_graph=True" in desc and "buffer sync=False" in desc
        for it in range(2):
            batches = _batches(tr, cfg, seed=it)
            grads = []
            for mod, params in ((ref, ref.model.parameters()), (ddp, step.model.parameters())):
                mod.zero_grad(set_to_none=True)
                torch.manual_seed(10 + it)
                mod(batches)["loss"].backward()
                grads.append([p.grad for p in params if p.requires_grad])
            for a, b in zip(*grads):
                assert b is not None and _close(b, a, 1e-6)
    finally:
        dist.destroy_process_group()


def test_trainer_mixed_ddp_main(tmp_path, monkeypatch):
    """main() under a 1-rank gloo group (as torchrun --nproc_per_node 1 would launch it), mixed steps."""
    tr = _script("train_dino_v0")
    for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
                     MASTER_PORT=str(_free_port())).items():
        monkeypatch.setenv(k, v)
    out = tmp_path / "run"
    assert tr.main(["--dry-run", "--mix-modalities", "--use-checkpoint", "--out", str(out)]) == 0
    import json
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert [r["modality"] for r in recs] == ["mixed"] * 3 and all("video/loss_fut" in r for r in recs)


# ---------------------------------------------------------------------------- optimizer / compile / profile
def test_fused_adamw_resume():
    tr = _script("train_dino_v0")
    args = argparse.Namespace(wd=0.05, lr=1e-3, warmup=1, steps=10, min_lr_ratio=0.1)
    m = torch.nn.Linear(4, 4)
    opt, _ = tr.build_optim(m, args, torch.device("cpu"))
    assert not opt.param_groups[0].get("fused")                          # fused only on GPU
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    opt2, _ = tr.build_optim(m, args, torch.device("cpu"))
    opt2.load_state_dict(opt.state_dict())                               # a v0 (non-fused) checkpoint
    tr.set_fused(opt2, True)
    st = opt2.state[m.weight]["step"]
    assert all(g["fused"] for g in opt2.param_groups) and st.dtype == torch.float32 and st.device == m.weight.device
    try:
        m(torch.randn(2, 4)).sum().backward()
        opt2.step()                                                      # fused CPU AdamW exists in recent torch
    except RuntimeError as ex:
        pytest.skip(f"fused AdamW not available on CPU here: {ex}")


def test_apply_compile_regions_eager_backend():
    import torch._dynamo
    tr = _script("train_dino_v0")
    cfg = tiny_config()
    torch.manual_seed(0)
    model = MAVTDinoV0(cfg).eval()
    torch.manual_seed(0)
    plain = MAVTDinoV0(cfg).eval()
    saved = (torch._dynamo.config.suppress_errors, torch._dynamo.config.cache_size_limit)
    try:
        done = tr.apply_compile(model, "eager", log=lambda s: None)
        regions = [n for n, m in model.named_modules() if type(m).__name__ in tr.COMPILE_REGIONS]
        assert done == regions and len(regions) >= cfg.trunk_layers + cfg.c_blocks + cfg.d_blocks + cfg.dec_blocks
        x = torch.rand(1, 3, 3, 64, 64) * 2 - 1
        with torch.no_grad():
            a, b = model.reconstruct(x, "video"), plain.reconstruct(x, "video")
        assert torch.allclose(a, b, atol=1e-6)
    finally:
        torch._dynamo.config.suppress_errors, torch._dynamo.config.cache_size_limit = saved
        torch._dynamo.reset()


def test_profile_script_cpu_smoke(tmp_path, capsys):
    import yaml
    cfg = {k: v for k, v in vars(tiny_config()).items() if k not in ("extra", "conv_channels", "forecast_horizons")}
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(dict(cfg, train=dict(batch_image=2))))
    prof = _script("profile_dino_v0")
    assert prof.main(["--config", str(p), "--device", "cpu", "--modality", "mixed", "--batch-image", "2",
                      "--batch-video", "1", "--batch-3d", "1", "--n-frames", "2", "--n-views", "2", "--steps", "1"]) == 0
    out = capsys.readouterr().out
    assert "image 2 + video 1 + threed 1" in out and "ms/step" in out
