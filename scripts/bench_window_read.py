#!/usr/bin/env python3
"""Micro-benchmark of the decoder's C read at training shapes: the dense masked SDPA over all N cells
(window_read "dense") against the gathered (2r+1)^2-cell read (window_read "gather"), fwd + bwd in bf16
autocast, plus an unmasked dense SDPA as the speed-of-light reference. Read-only, synthetic tensors.

  HIP_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/bench_window_read.py --rows 688 --iters 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from mavt.dino_v0.decoder import UV_MASK, _WindowCrossAttn, gather_window, query_coords, window_bias  # noqa: E402
from mavt.dino_v0.rope import AxialRoPE, grid_coords, modality_mask  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=688, help="decoded frames per decoder call (v1 run: 688)")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--window", type=int, default=2, help="read_window r: (2r+1)^2 cells")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    dev = torch.device(a.device)
    gpu = dev.type == "cuda"
    torch.manual_seed(0)
    N, B, r = a.grid * a.grid, a.rows, a.window
    attn = _WindowCrossAttn(a.dim, a.heads, AxialRoPE(a.dim // a.heads)).to(dev)
    x = torch.randn(B, N, a.dim, device=dev, requires_grad=True)
    c = torch.randn(B, N, a.dim, device=dev, requires_grad=True)
    qc = query_coords(a.grid, 1.0, B, device=dev)
    qm = modality_mask("video", N, device=dev).unsqueeze(0).expand(B, -1, -1)
    centre = qc[..., 1:3] + torch.randn(B, N, 2, device=dev) * 0.3 / a.grid      # Δ ~ 0.3 cell
    q_m = qm * torch.tensor(UV_MASK, device=dev)
    q_pos = torch.cat([qc[..., :1], centre, qc[..., 3:]], -1)
    c_pos = grid_coords(a.grid, 0.0, device=dev)
    c_pos = torch.cat([qc[:, :1, :1].expand(-1, N, -1), c_pos[None, :, 1:].expand(B, -1, -1)], -1)
    c_m = q_m[:, :1].expand(-1, N, -1)
    bias = window_bias(centre, a.grid, r, torch.bfloat16 if gpu else torch.float32)
    idx, valid, cell = gather_window(centre, a.grid, r)
    W = idx.shape[-1]
    k_pos = torch.cat([qc[..., :1].unsqueeze(2).expand(-1, -1, W, -1), cell, torch.zeros_like(cell[..., :1])], -1)
    k_m = q_m.unsqueeze(2).expand(-1, -1, W, -1)

    def unmasked():
        """Same projections and RoPE, SDPA without a mask: the fused-kernel reference (not a valid read)."""
        B_, Q, D = x.shape
        q = attn.rope(attn.q(x).view(B_, Q, attn.h, attn.hd).transpose(1, 2), q_pos, q_m)
        k, v = attn.kv(c).view(B_, N, 2, attn.h, attn.hd).permute(2, 0, 3, 1, 4)
        out = torch.nn.functional.scaled_dot_product_attention(q, attn.rope(k, c_pos, c_m), v)
        return attn.o(out.transpose(1, 2).reshape(B_, Q, D))

    modes = {
        "dense (masked SDPA, v1)": lambda: attn.forward(x, c, bias, q_pos, q_m, c_pos, c_m),
        "gather (%d cells)" % W: lambda: attn.forward_gather(x, c, idx, valid, q_pos, k_pos, q_m, k_m),
        "dense unmasked (reference)": unmasked,
    }
    print(f"rows {B} x N {N} x dim {a.dim}, {a.heads} heads, window {W} cells, {'bf16 autocast' if gpu else 'fp32 cpu'}")
    for name, fn in modes.items():
        def step():
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=gpu):
                y = fn()
            y.float().square().mean().backward()
        for _ in range(3):
            step()
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            step()
        if gpu:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / a.iters * 1e3
        mem = f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if gpu else ""
        print(f"  {name:30s} {ms:8.2f} ms / fwd+bwd{mem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
