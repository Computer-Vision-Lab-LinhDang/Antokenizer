"""Query-readout pilot: source-only encoding, one target camera, RGB loss on sampled target rays.

The encoder sees only the source image (and its intrinsics); the target image, mask and camera enter
the loss and the ray query, never the state. Legacy, shared_ray_v1 and shared_field_v1 share this step.
"""
import torch

from . import geometry as g
from .config import config_from_dict
from .evaluate_shared import legacy_queries
from .model import MAVTGrainPrior
from .query import RayQuery

ENCODER = ("stems.", "stream_c.", "anchor.")
FROZEN_ALWAYS = ("stems.dino.",)
FRESH = ("readout.", "rgb.", "query_decoder.")   # reader tensors a warm start never copies


def reader_prefixes(cfg):
    """Modules on the pilot decode path: the reader, its RGB head and the replayed appearance."""
    return ("readout.", "rgb.", "appear.") if cfg.readout_version == "legacy" else ("query_decoder.", "appear.")


def configure_pilot(model, frozen_encoder=True):
    """Train the reader (and the encoder unless frozen_encoder); everything off the pilot path is frozen,
    so every trainable parameter gets a gradient each step (DDP without find_unused_parameters).
    -> names of the trainable parameters."""
    train = reader_prefixes(model.cfg) + (() if frozen_encoder else ENCODER)
    names = []
    for n, p in model.named_parameters():
        on = n.startswith(train) and not n.startswith(FROZEN_ALWAYS)
        p.requires_grad_(on)
        if on:
            names.append(n)
    model.train()
    return names


def _sample_rays(mask, rays, seed):
    """rays distinct pixel indices per sample [B, rays], uniform and fixed by seed (mask weights the loss)."""
    B, H, W = mask.shape
    if not 0 < rays <= H * W:
        raise ValueError("rays must be in [1, H * W]")
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(B, H * W, generator=gen).argsort(-1)[:, :rays].to(mask.device)


def pilot_step(model, source, source_kappa, target, camera, mask, rays=1024, seed=0):
    """source / target [B, 3, H, W] in [-1, 1]; source_kappa [B, 4] or None; camera: target Camera
    (canonical -> camera); mask [B, H, W] valid target pixels. -> dict(loss, l1, rays, readout_version)."""
    cfg = model.cfg
    if target.shape != source.shape or mask.shape != (target.shape[0],) + target.shape[2:]:
        raise ValueError("target must match source, mask must be [B, H, W]")
    H = target.shape[2]
    idx = _sample_rays(mask, rays, seed)
    uv = g.cell_centres(H, target.device).reshape(-1, 2)[idx]                      # pixel centres [B, R, 2]
    tgt = ((target.float() + 1) / 2).flatten(2).transpose(1, 2).gather(1, idx[..., None].expand(-1, -1, 3))
    m = mask.float().flatten(1).gather(1, idx)[..., None]
    S = model.encode_source(source, source_kappa)
    if cfg.readout_version == "legacy":
        rgb = legacy_queries(model, S, camera, uv)["rgb"]
    else:
        view = model.prepare_state(S, mode="novel_view")
        q = RayQuery(camera, uv, cfg.render_near, cfg.render_far, cfg.render_samples)
        rgb = model.decode_packet(model.read_rays(view, q), ("rgb",))["rgb"]
    l1 = ((rgb.float() - tgt).abs() * m).sum() / (3 * m.sum()).clamp_min(1)
    return dict(loss=cfg.w_rgb * l1, l1=l1.detach(), rays=rays, readout_version=cfg.readout_version)


def warm_start_query_encoder(ckpt_path, readout_version, **overrides):
    """New model with readout_version whose encoder is copied from a legacy checkpoint; the reader starts
    from its fresh init. An override that changes the shape of a copied tensor raises. -> (model, receipt)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old = ck.get("cfg", {})
    if old.get("readout_version", "legacy") != "legacy":
        raise ValueError("warm start reads a legacy checkpoint")
    cfg = config_from_dict(old, **{"readout_version": readout_version, "innovation_semantics": "disabled",
                                   "w_lpips": 0.0, **overrides})
    cfg.check()
    model = MAVTGrainPrior(cfg)
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in ck.get("model", ck).items()}
    own = model.state_dict()
    copied = []
    for k, v in sd.items():
        if k.startswith(FRESH) or k not in own:
            continue
        if own[k].shape != v.shape:
            raise ValueError(f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}")
        own[k] = v
        copied.append(k)
    missing = [k for k in own if not k.startswith(FRESH) and k not in sd]
    if missing:
        raise ValueError(f"checkpoint lacks encoder tensors: {missing[:5]}")
    model.load_state_dict(own)
    return model, dict(source=str(ckpt_path), source_step=ck.get("step"), readout_version=readout_version,
                       innovation_semantics=cfg.innovation_semantics, copied=len(copied),
                       new_reader_initialized=True, fresh_prefixes=FRESH)
