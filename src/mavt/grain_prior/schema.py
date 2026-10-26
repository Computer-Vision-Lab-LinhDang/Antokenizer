"""State schema, header (32 scalars), serializer and validation (sections 3, 11).

A state S is a dict of batched tensors:
    h       [B, 32]                         header
    u, rho, sigma, f                        grains: [B, G, 2], [B, G], [B, G], [B, G, d_f]
    c0      [B, 16, d_coarse], kappa0 [B, 4]
    zb      [B, 16, d_coarse]               boundary memory (serialized only when the state holds a target)
    recs    list (<= queue) of dict(xi [B, 6], M [B, n_m, 3], kappa [B, 4], b [B, 16 x d_b] as [B, n_memb],
            r [B, n_slots, d_r], L [B] long), oldest first; only r[:, :L] is valid and serialized

Header layout (index: field): 0 version, 1 mode, 2 n_targets, 3 coordinate_mode (0 anchor median gauge,
1 metric), 4 camera_mode (0 estimated, 1 given), 5 anchor time (s, episode), 6 boundary offset t_b - t_0,
7-9 target offsets t_i - t_0, 10-13 view ids (anchor, targets), 14-16 lengths L_i, 17-18 width, height,
19 segment id, 20-22 provenance (0 observed, 1 predicted), 23 anchor observed, 24-31 reserved (zero).
"""
from __future__ import annotations

import torch

from .config import HEADER, MODES, VERSION, GrainPriorConfig

H_VERSION, H_MODE, H_NT, H_COORD, H_CAM, H_T0, H_TB = 0, 1, 2, 3, 4, 5, 6
H_TT, H_VIEW, H_LEN, H_W, H_H, H_SEG, H_PROV, H_OBS, H_RES = 7, 10, 14, 17, 18, 19, 20, 23, 24


def mode_code(mode: str) -> int:
    return MODES.index(mode)


def new_header(cfg: GrainPriorConfig, B: int, mode: torch.Tensor, coord: torch.Tensor, cam: torch.Tensor,
               device) -> torch.Tensor:
    """mode / coord / cam [B] codes -> header [B, 32] of an anchor-only state."""
    h = torch.zeros(B, HEADER, device=device)
    h[:, H_VERSION] = VERSION
    h[:, H_MODE], h[:, H_COORD], h[:, H_CAM] = mode.float(), coord.float(), cam.float()
    h[:, H_W] = h[:, H_H] = cfg.image_size
    h[:, H_OBS] = 1
    return h


def record_numbers(cfg: GrainPriorConfig, rec: dict) -> torch.Tensor:
    return 6 + 3 * cfg.n_motion + 4 + cfg.n_memb + rec["L"] * cfg.d_r


def payload(cfg: GrainPriorConfig, S: dict) -> torch.Tensor:
    """Scalars written by serialize, per sample [B]."""
    n = torch.full((S["h"].shape[0],), HEADER + cfg.anchor_numbers(), device=S["h"].device, dtype=torch.long)
    if S["recs"]:
        n = n + cfg.n_coarse * cfg.d_coarse
        for rec in S["recs"]:
            n = n + record_numbers(cfg, rec)
    return n


def serialize(cfg: GrainPriorConfig, S: dict) -> list:
    """State -> one flat float32 vector per sample (no padding; r cut to its valid prefix)."""
    out = []
    for b in range(S["h"].shape[0]):
        parts = [S["h"][b], torch.cat([S["u"][b], S["rho"][b, :, None], S["sigma"][b, :, None], S["f"][b]], -1),
                 S["c0"][b], S["kappa0"][b]]
        if S["recs"]:
            parts.append(S["zb"][b])
        for rec in S["recs"]:
            L = int(rec["L"][b])
            parts += [rec["xi"][b], rec["M"][b], rec["kappa"][b], rec["b"][b], rec["r"][b, :L]]
        out.append(torch.cat([p.detach().float().reshape(-1) for p in parts]))
    return out


def validate(cfg: GrainPriorConfig, flat: torch.Tensor) -> None:
    """Shapes, finite values, enums, lengths, reserved zero and the scalar count of one serialized state."""
    if flat.dim() != 1 or flat.dtype != torch.float32:
        raise ValueError("a state is a 1-D float32 vector")
    if not torch.isfinite(flat).all():
        raise ValueError("non-finite value in the state")
    h = flat[:HEADER]
    if int(h[H_VERSION]) != VERSION:
        raise ValueError(f"version {int(h[H_VERSION])} != {VERSION}")
    if not 0 <= int(h[H_MODE]) < len(MODES) or int(h[H_COORD]) not in (0, 1) or int(h[H_CAM]) not in (0, 1):
        raise ValueError("invalid mode / coordinate / camera code")
    nt = int(h[H_NT])
    if not 0 <= nt <= cfg.queue:
        raise ValueError(f"n_targets {nt} outside 0..{cfg.queue}")
    Ls = [int(x) for x in h[H_LEN:H_LEN + nt]]
    if any(not 0 <= L <= cfg.n_slots for L in Ls) or any(h[H_LEN + nt:H_LEN + cfg.queue] != 0):
        raise ValueError("invalid lengths")
    if (h[H_RES:] != 0).any():
        raise ValueError("reserved header fields must be zero")
    if flat.numel() != cfg.state_numbers(Ls):
        raise ValueError(f"{flat.numel()} scalars, the header implies {cfg.state_numbers(Ls)}")


def deserialize(cfg: GrainPriorConfig, flats: list) -> dict:
    """Flat vectors (one per sample, same n_targets) -> batched state; r is zero-padded to n_slots."""
    for x in flats:
        validate(cfg, x)
    G, dc, nc = cfg.n_grains, cfg.d_coarse, cfg.n_coarse
    rows = []
    for x in flats:
        i = 0

        def take(n):
            nonlocal i
            v = x[i:i + n]
            i += n
            return v

        h = take(HEADER)
        gr = take(G * cfg.d_grain).view(G, cfg.d_grain)
        s = dict(h=h, u=gr[:, :2], rho=gr[:, 2], sigma=gr[:, 3], f=gr[:, 4:], c0=take(nc * dc).view(nc, dc),
                 kappa0=take(4), recs=[])
        nt = int(h[H_NT])
        s["zb"] = take(nc * dc).view(nc, dc) if nt else s["c0"].clone()
        for j in range(nt):
            L = int(h[H_LEN + j])
            rec = dict(xi=take(6), M=take(3 * cfg.n_motion).view(cfg.n_motion, 3), kappa=take(4),
                       b=take(cfg.n_memb), L=torch.tensor(L, device=x.device))
            r = torch.zeros(cfg.n_slots, cfg.d_r, device=x.device)
            r[:L] = take(L * cfg.d_r).view(L, cfg.d_r)
            rec["r"] = r
            s["recs"].append(rec)
        rows.append(s)
    nts = {len(s["recs"]) for s in rows}
    if len(nts) != 1:
        raise ValueError("a batch must share n_targets")
    st = lambda k: torch.stack([s[k] for s in rows])
    S = {k: st(k) for k in ("h", "u", "rho", "sigma", "f", "c0", "kappa0", "zb")}
    S["recs"] = [{k: torch.stack([s["recs"][j][k] for s in rows]) for k in rows[0]["recs"][j]}
                 for j in range(nts.pop())]
    return S
