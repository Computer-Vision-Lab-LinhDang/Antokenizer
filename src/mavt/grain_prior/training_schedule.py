"""Explicit per-module learning rates and stateless update-indexed schedules.

This module does not change the existing ImageNet/CRSR trainer defaults. Callers opt in.
"""
import math

ENCODER_PREFIXES = ('stems.', 'stream_c.', 'anchor.')


def optimizer_groups(model, reader_lr=1e-4, encoder_lr=1e-5, weight_decay=.05):
    """Every trainable parameter occurs once. Biases, norm gains, and scalar temperatures
    get no decay. Frozen DINO is excluded by configure_pilot, before calling this function.
    """
    if not 0 < encoder_lr <= reader_lr or weight_decay < 0:
        raise ValueError('require 0 < encoder_lr <= reader_lr and weight_decay >= 0')
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = 'encoder' if name.startswith(ENCODER_PREFIXES) else 'reader'
        decay = weight_decay if parameter.ndim >= 2 else 0.
        key = (role,decay)
        grouped.setdefault(key,[]).append(parameter)
    if not grouped:
        raise ValueError('no trainable parameters')
    return [dict(params=params,role=role,weight_decay=decay,
                 lr=encoder_lr if role=='encoder' else reader_lr,
                 base_lr=encoder_lr if role=='encoder' else reader_lr)
            for (role,decay),params in grouped.items()]


def lr_multiplier(step,total_steps,warmup,schedule='cosine',min_ratio=.1):
    """step is 1-based optimizer-update index. At warmup: 1; at total_steps: min_ratio.

    No restarts; a fixed schedule avoids making evaluation-driven changes across arms.
    Constant-after-warmup is provided as a separate registered control.
    """
    if not 0 <= warmup < total_steps or not 1 <= step <= total_steps:
        raise ValueError('invalid update index or warmup length')
    if schedule not in ('constant','cosine') or not 0 <= min_ratio <= 1:
        raise ValueError('invalid schedule/min_ratio')
    if warmup and step <= warmup:
        return step/warmup
    if schedule == 'constant':
        return 1.
    # With no warmup, update 1 starts at the peak; final update reaches the floor.
    start = warmup if warmup else 1
    fraction = (step-start)/max(1,total_steps-start)
    return min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*fraction))


def set_learning_rates(optimizer,step,total_steps,warmup,schedule='cosine',min_ratio=.1):
    multiplier = lr_multiplier(step,total_steps,warmup,schedule,min_ratio)
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr']*multiplier
    return multiplier
