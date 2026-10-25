"""The closed state Z of C-D 4.1: header, serializer and payload count.

Z (batched dict): h [B, 16], c [B, N, d_c], rho [B, N], kappa0 [B, 4], and for the T = n_obs - 1 targets
xi [B, T, 6], M [B, T, n_motion, 3], kappa [B, T, 4], a [B, T, R_a] (tucker only), r [B, T, n_slots, d_r]
(slots >= L zero), L [B, T]. An image has T = 0 and no target keys.

Header, 16 FP32 slots in this order: mode (0 image, 1 video, 2 static_views), n_obs, time_offsets[4],
view_ids[4], L_target[3], coordinate_mode (0 anchor-normalised, 1 metric), camera_mode (0 estimated,
1 externally calibrated), version. Unused entries are 0. Time offsets are in frame steps of the clip as it
was sampled (the loaders do not carry seconds); views of a static scene share time 0.

serialize() writes, per sample, exactly the scalars of the ledger in the order
    h, c, rho, kappa0, then per target xi, M, kappa, [a], r[:L]
so a target with L slots costs 6 + 48 + 4 (+ 32) + 24 L and padding is never sent.
"""
from __future__ import annotations

import torch

from .config import MODES, VERSION, CD41Config


def make_header(cfg: CD41Config, mode: str, n_obs: int, L: torch.Tensor, cam_given: bool) -> torch.Tensor:
    """L [B, n_obs - 1] long -> header [B, 16] float32."""
    B, dev = L.shape[0], L.device
    h = torch.zeros(B, cfg.header_slots, device=dev)
    h[:, 0] = MODES.index(mode)
    h[:, 1] = n_obs
    if mode == "video":
        h[:, 2:2 + n_obs] = torch.arange(n_obs, device=dev, dtype=torch.float32)
    if mode == "static_views":
        h[:, 6:6 + n_obs] = torch.arange(n_obs, device=dev, dtype=torch.float32)
    if L.shape[1]:
        h[:, 10:10 + L.shape[1]] = L.float()
    h[:, 13] = 1.0 if mode == "static_views" else 0.0
    h[:, 14] = 1.0 if cam_given else 0.0
    h[:, 15] = VERSION
    return h


def target_numbers(cfg: CD41Config, L: torch.Tensor) -> torch.Tensor:
    return 6 + 3 * cfg.n_motion + 4 + (cfg.rank_a if cfg.tucker else 0) + L * cfg.d_r


def payload_numbers(cfg: CD41Config, Z: dict) -> torch.Tensor:
    """Scalars sent per sample [B] (header included)."""
    B = Z["c"].shape[0]
    n = torch.full((B,), float(cfg.header_slots + cfg.anchor_numbers), device=Z["c"].device)
    if "L" in Z and Z["L"].shape[1]:
        n = n + target_numbers(cfg, Z["L"].float()).sum(1)
    return n


def serialize(cfg: CD41Config, Z: dict) -> list:
    """Batched state -> list of B flat FP32 tensors, each exactly the payload of its sample."""
    out = []
    B = Z["c"].shape[0]
    T = Z["xi"].shape[1]
    for b in range(B):
        parts = [Z["h"][b], Z["c"][b].flatten(), Z["rho"][b], Z["kappa0"][b]]
        for t in range(T):
            parts += [Z["xi"][b, t], Z["M"][b, t].flatten(), Z["kappa"][b, t]]
            if cfg.tucker:
                parts.append(Z["a"][b, t])
            L = int(Z["L"][b, t])
            parts.append(Z["r"][b, t, :L].flatten())
        out.append(torch.cat([p.float().reshape(-1) for p in parts]))
    return out


def deserialize(cfg: CD41Config, flats: list) -> dict:
    """Inverse of serialize(): list of flat tensors (same n_obs) -> batched state; r zero-padded to n_slots."""
    N, nm, Ra, dr = cfg.n_cells, cfg.n_motion, cfg.rank_a, cfg.d_r
    rows = []
    for v in flats:
        i = 0

        def take(k):
            nonlocal i
            s = v[i:i + k]
            i += k
            return s
        h = take(cfg.header_slots)
        n_obs = int(round(h[1].item()))
        z = dict(h=h, c=take(N * cfg.d_c).view(N, cfg.d_c), rho=take(N), kappa0=take(4), xi=[], M=[], kappa=[], a=[],
                 r=[], L=[])
        for t in range(n_obs - 1):
            z["xi"].append(take(6))
            z["M"].append(take(3 * nm).view(nm, 3))
            z["kappa"].append(take(4))
            if cfg.tucker:
                z["a"].append(take(Ra))
            L = int(round(h[10 + t].item()))
            r = torch.zeros(cfg.n_slots, dr, device=v.device)
            r[:L] = take(L * dr).view(L, dr)
            z["r"].append(r)
            z["L"].append(L)
        if i != v.numel():
            raise ValueError(f"payload has {v.numel()} scalars, the header describes {i}")
        rows.append(z)
    if len({int(round(z['h'][1].item())) for z in rows}) != 1:
        raise ValueError("deserialize batches states with the same n_obs")
    dev = flats[0].device
    Z = {k: torch.stack([z[k] for z in rows]) for k in ("h", "c", "rho", "kappa0")}
    T = len(rows[0]["xi"])
    if T == 0:
        Z["xi"] = torch.zeros(len(rows), 0, 6, device=dev)
        return Z
    for k in ("xi", "M", "kappa", "r"):
        Z[k] = torch.stack([torch.stack(z[k]) for z in rows])
    if cfg.tucker:
        Z["a"] = torch.stack([torch.stack(z["a"]) for z in rows])
    Z["L"] = torch.tensor([z["L"] for z in rows], dtype=torch.long, device=dev)
    return Z
