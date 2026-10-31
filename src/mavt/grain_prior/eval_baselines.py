"""Baselines and protocols the decoded-clip scores are read against (eval and training diagnostics).

    copy_last               the previous ground-truth frame as the target (the anchor for the first target): what a
                            decoder that only carries the last observation forward would score
    strip_target_posterior  the state with every record's innovation removed (r = 0, L = 0)
    no_memory               the model decoding with z = c_0 at every record (memory_step is the identity)

novel view protocol = no_memory(model).decode(strip_target_posterior(cfg, S)): grains + c_0 + the record's camera
and motion only. On posed views (cameras copied) nothing of the target view is read; on video the estimated
camera and motion of the record are still posteriors of the target, only b_i and r_i are removed.
"""
from __future__ import annotations

import contextlib

import torch

from . import schema as sc


def copy_last(frames: torch.Tensor, q: int) -> torch.Tensor:
    """frames [B, T, ...] of a clip whose targets are the last q frames -> [B, q, ...], frame t - 1 for target t."""
    T = frames.shape[1]
    return frames[:, T - q - 1:T - 1]


def strip_target_posterior(cfg, S: dict) -> dict:
    """A copy of S whose records carry no innovation (r = 0, L = 0; the header lengths follow) and no grain feature
    update (dfc = 0, 3.13: read from the target, so a target-free decode must not see it). S is untouched."""
    recs = [dict(r, L=torch.zeros_like(r["L"]), r=torch.zeros_like(r["r"])) for r in S["recs"]]
    recs = [dict(r, dfc=torch.zeros_like(r["dfc"])) if "dfc" in r else r for r in recs]
    h = S["h"].clone()
    h[:, sc.H_LEN:sc.H_LEN + cfg.queue] = 0
    return dict(S, recs=recs, h=h)


@contextlib.contextmanager
def no_memory(model):
    """Decode with z = c_0 at every record (cfg.memory off), restored on exit."""
    old = model.cfg.memory
    model.cfg.memory = False
    try:
        yield model
    finally:
        model.cfg.memory = old
