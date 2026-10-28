"""Muon for the hidden matrices + AdamW for the rest, in one optimizer, and the schedule / clipping helpers of the
trainer (scripts/train_dino_v0.py --optimizer muon --schedule wsd --clip-mode median).

Muon (Jordan et al. 2024): momentum 0.95 (Nesterov), the update orthogonalised by five Newton-Schulz iterations and
scaled by 0.2 sqrt(max(m, n)) so its RMS matches AdamW's (Liu et al. 2025, "Muon is scalable"); both kinds then
share one lr. Fused weights (q / k / v, k / v, the R value maps of 2.2) are orthogonalised per block: a group's
`splits` cuts the output axis into equal blocks. Decoupled weight decay p <- p (1 - lr wd) on every kind.

Groups: dict(params, kind = muon | io | scalar | attn_scalar, splits (muon), lr, weight_decay, lr_scale). The kind
only selects the update rule; io / scalar / attn_scalar are AdamW (betas 0.9 / 0.95, eps 1e-8).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T of g = U S V^T (2-D, quintic iteration, bf16 on GPU as in Muon)."""
    a, b, c = NS_COEFFS
    x = g.to(torch.bfloat16 if g.is_cuda else torch.float32)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    if tall:
        x = x.T
    return x.to(g.dtype)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, groups, lr: float = 1e-3, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, betas=betas, eps=eps,
                        weight_decay=weight_decay, kind="io", splits=1, lr_scale=1.0)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            if g["kind"] == "muon":
                self._muon(g)
            else:
                self._adamw(g)
        return loss

    def _muon(self, g):
        lr, wd, mom = g["lr"], g["weight_decay"], g["momentum"]
        for p in g["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            st = self.state[p]
            buf = st.get("momentum_buffer")
            if buf is None:
                buf = st["momentum_buffer"] = torch.zeros_like(grad)
            buf.mul_(mom).add_(grad)
            u = grad.add(buf, alpha=mom) if g["nesterov"] else buf
            shape = u.shape
            u2 = u.reshape(shape[0], -1)                                 # 1 x 1 convolutions: [out, in]
            blocks = u2.split(shape[0] // g["splits"], 0)
            upd = torch.cat([newton_schulz(blk, g["ns_steps"]) * (0.2 * math.sqrt(max(blk.shape))) for blk in blocks], 0)
            if wd:
                p.mul_(1 - lr * wd)
            p.add_(upd.reshape(shape).to(p.dtype), alpha=-lr)

    def _adamw(self, g):
        lr, wd, (b1, b2), eps = g["lr"], g["weight_decay"], g["betas"], g["eps"]
        for p in g["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            st = self.state[p]
            if not st:
                st["step"] = 0
                st["exp_avg"], st["exp_avg_sq"] = torch.zeros_like(p), torch.zeros_like(p)
            st["step"] += 1
            t = st["step"]
            st["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
            if wd:
                p.mul_(1 - lr * wd)
            denom = (st["exp_avg_sq"] / (1 - b2 ** t)).sqrt_().add_(eps)
            p.addcdiv_(st["exp_avg"], denom, value=-lr / (1 - b1 ** t))


def wsd_factor(step: int, warmup: int, steps: int, decay_steps: int, min_ratio: float,
               decay_start: Optional[int] = None) -> float:
    """Warmup-stable-decay: linear warmup, constant, then a linear decay of decay_steps to min_ratio that starts
    at decay_start (default steps - decay_steps). A decay branch: resume a stable checkpoint at step k with
    --decay-start k --steps k + D in another --out."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    d0 = steps - decay_steps if decay_start is None or decay_start < 0 else decay_start
    if step < d0:
        return 1.0
    prog = min(1.0, (step - d0) / max(1, decay_steps))
    return 1.0 - (1.0 - min_ratio) * prog


class MedianClip:
    """Adaptive clip: threshold = mult x median of the last `window` pre-clip grad norms (fallback before the window
    fills). update(gn) records gn and returns whether it is above the threshold in force."""

    def __init__(self, mult: float, window: int, fallback: float):
        self.mult, self.window, self.fallback = mult, window, fallback
        self.hist: list = []
        self.n_clip, self.n = 0, 0

    def threshold(self) -> float:
        if len(self.hist) < self.window:
            return self.fallback
        s = sorted(self.hist)
        k = len(s) // 2
        med = s[k] if len(s) % 2 else 0.5 * (s[k - 1] + s[k])
        return self.mult * med

    def update(self, gn: float) -> bool:
        clipped = gn > self.threshold()
        if math.isfinite(gn):
            self.hist = (self.hist + [gn])[-self.window:]
        self.n, self.n_clip = self.n + 1, self.n_clip + int(clipped)
        return clipped

    def frac(self) -> float:
        """Fraction of clipped steps since the last call (resets)."""
        f = self.n_clip / max(1, self.n)
        self.n_clip, self.n = 0, 0
        return f

    def state_dict(self) -> dict:
        return dict(hist=list(self.hist), n_clip=self.n_clip, n=self.n)

    def load_state_dict(self, sd: dict) -> None:
        self.hist, self.n_clip, self.n = list(sd["hist"]), sd["n_clip"], sd["n"]
