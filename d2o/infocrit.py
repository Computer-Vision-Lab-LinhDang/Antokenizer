from __future__ import annotations

"""Information criteria for model selection in token merging.

Ported from dHT (NeurIPS 2025 Spotlight) with extensions for multi-scale TAO integration.

Key Formulas:
    NLL = n * log(2π * Π_c σ²_c + ε)  # Gaussian negative log-likelihood
    k = 1 / (2n - 2√n)                 # Degrees of freedom for 2D regions

    AIC  = NLL + 2k
    BIC  = NLL + k * log(n)            # Statistical consistency (recommended)
    AICc = AIC + 2k(k+1)/(n-k-1)       # Small sample correction
    CIC  = NLL + k * (log(n) + 1)      # Conservative

Returns negative IC so that HIGHER is BETTER for maximization.

References:
    - dHT: https://github.com/google-research/dht
    - BIC consistency: Schwarz (1978), "Estimating the Dimension of a Model"
"""
import torch
from torch import Tensor
from typing import Literal

__all__ = [
    'gaussian_2nll',
    'infocrit',
    'infocrit_multi_scale',
    'compare_ic_for_merge',
]


def gaussian_2nll(s2: Tensor, mV: Tensor, iota: float = 1.0) -> Tensor:
    """Compute Gaussian negative log-likelihood for multivariate diagonal covariance.

    Args:
        s2: Variance tensor, shape [nV, C] - diagonal covariance per vertex per channel
        mV: Normalized region sizes, shape [nV] - must be mV/total_pixels
        iota: Stability constant, prevents log(0). Default 1.0 (from dHT)

    Returns:
        NLL values, shape [nV]

    Formula:
        NLL_i = n_i * log(2π * Π_c σ²_ic + ε)

        Where:
        - n_i = mV[i] (normalized region size)
        - σ²_ic = s2[i, c] (variance for region i, channel c)
        - Π_c = product over channels (assumes independence)
        - ε = iota (numerical stability)

    Note:
        The product over channels treats channels as independent Gaussians.
        This is a reasonable assumption for image features.
    """
    # Product over channels: (2π * s2).prod(-1) gives Π_c(2π * σ²_c)
    # Add iota for numerical stability, then log
    return mV * (2 * torch.pi * s2).prod(-1).add(iota).log()


def _df(mV: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute degrees of freedom for image regions.

    Formula: k = 1 / (2n - 2√n + eps)

    This accounts for 2D spatial correlation in image regions:
    - Larger regions (higher n) → smaller penalty per parameter
    - Square root term accounts for boundary effects

    Args:
        mV: Region sizes, shape [nV]
        eps: Stability constant to prevent division by zero

    Returns:
        Degrees of freedom, shape [nV]
    """
    denom = 2 * mV - 2 * mV.sqrt() + eps
    return 1 / denom.clamp(min=eps)


def _aic_fn(mV: Tensor) -> Tensor:
    """AIC penalty: 2k"""
    return 2 * _df(mV)


def _bic_fn(mV: Tensor) -> Tensor:
    """BIC penalty: k * log(n)

    BIC provides statistical consistency for segmentation:
    - As n → ∞, BIC selects the true model with probability 1
    - Recommended for image tokenization (proven in dHT paper)
    """
    return mV.log() * _df(mV)


def _cic_fn(mV: Tensor) -> Tensor:
    """CIC penalty: k * (log(n) + 1)

    More conservative than BIC, higher penalty.
    """
    return mV.log().add(1) * _df(mV)


def _aicc_fn(mV: Tensor, eps: float = 1e-6) -> Tensor:
    """AICc penalty: AIC + bias correction for small samples.

    Formula: 2k + 2k(k+1)/(n-k-1+eps)

    Recommended when n/k < 40 (small sample sizes).
    """
    k = _df(mV)
    # Add eps and clamp to prevent division by zero or negative
    denom = (mV - k - 1).clamp(min=eps)
    return 2 * k + 2 * (k ** 2 + k) / denom


_infodict = {
    'aic': _aic_fn,
    'aicc': _aicc_fn,
    'bic': _bic_fn,
    'cic': _cic_fn,
}


def infocrit(
    s2: Tensor,
    mV: Tensor,
    H: int,
    W: int,
    mode: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
    iota: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Compute information criteria for model selection.

    This is the main function for computing IC scores used in merge decisions.

    Args:
        s2: Variance tensor, shape [nV, C] - per-region per-channel variance
        mV: Region sizes (absolute pixel counts), shape [nV]
        H: Image/scale height
        W: Image/scale width
        mode: Information criterion type:
            - 'aic': Akaike IC (balanced)
            - 'bic': Bayesian IC (consistency, recommended)
            - 'aicc': AIC with small sample correction
            - 'cic': Conservative IC
        iota: NLL stability constant (default 1.0)
        eps: Division stability constant (default 1e-8)

    Returns:
        Negative IC values, shape [nV]
        HIGHER is BETTER (ready for maximization)

    Formula:
        IC = -NLL - penalty(k)
        Return: -IC = NLL + penalty

    Example:
        >>> s2 = torch.randn(512, 3).abs()  # 512 regions, 3 channels (RGB)
        >>> mV = torch.randint(10, 100, (512,)).float()  # Region sizes
        >>> ic = infocrit(s2, mV, H=256, W=256, mode='bic')
        >>> # ic.shape = [512], higher values indicate better regions
    """
    nc = H * W  # Total pixel count for normalization
    nll = gaussian_2nll(s2, mV / nc, iota)  # Normalize mV by total pixels
    fn = _infodict.get(mode, _bic_fn)
    k = fn(mV + eps)  # Add eps for numerical stability
    return (nll + k).neg()  # Negative so higher is better


def infocrit_multi_scale(
    s2_list: list[Tensor],
    mV_list: list[Tensor],
    H_list: list[int],
    W_list: list[int],
    mode: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
    iota: float = 1.0,
    eps: float = 1e-8,
) -> list[Tensor]:
    """Compute IC for multiple scales simultaneously.

    Convenience function for TAO's multi-scale tokenization.

    Args:
        s2_list: List of variance tensors, one per scale
        mV_list: List of region sizes, one per scale
        H_list: List of heights per scale
        W_list: List of widths per scale
        mode, iota, eps: Same as infocrit()

    Returns:
        List of IC tensors, one per scale

    Example:
        >>> # 5 scales: 2x2, 4x4, 8x8, 16x16, 32x32
        >>> ic_list = infocrit_multi_scale(
        ...     s2_list=[s2_1, s2_2, s2_3, s2_4, s2_5],
        ...     mV_list=[mV_1, mV_2, mV_3, mV_4, mV_5],
        ...     H_list=[2, 4, 8, 16, 32],
        ...     W_list=[2, 4, 8, 16, 32],
        ...     mode='bic'
        ... )
    """
    return [
        infocrit(s2, mV, H, W, mode, iota, eps)
        for s2, mV, H, W in zip(s2_list, mV_list, H_list, W_list)
    ]


def compare_ic_for_merge(
    ic_fine: Tensor,
    ic_coarse: Tensor,
    ic_merged: Tensor,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
) -> tuple[Tensor, Tensor]:
    """Compare IC values to decide whether to merge.

    Core decision function for IC-guided token merging.

    Args:
        ic_fine: IC values for fine scale tokens, shape [N]
        ic_coarse: IC values for coarse scale tokens, shape [N]
                   (must be aligned with fine via fine_to_coarse_map)
        ic_merged: IC values for hypothetically merged tokens, shape [N]
        reduction: How to aggregate:
            - 'mean': Average IC (default)
            - 'sum': Total IC
            - 'none': Per-token comparison

    Returns:
        should_merge: Boolean tensor indicating merge decision
        ic_improvement: IC improvement from merging (merged - separate)

    Logic:
        Merge IF: IC(merged) > IC(fine) + IC(coarse)

        This ensures merging only occurs when it improves
        the overall information criterion.
    """
    ic_separate = ic_fine + ic_coarse
    ic_improvement = ic_merged - ic_separate

    if reduction == 'mean':
        should_merge = ic_improvement.mean() > 0
    elif reduction == 'sum':
        should_merge = ic_improvement.sum() > 0
    else:  # 'none'
        should_merge = ic_improvement > 0

    return should_merge, ic_improvement
