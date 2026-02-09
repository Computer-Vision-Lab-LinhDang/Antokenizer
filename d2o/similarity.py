from __future__ import annotations

"""Similarity functions for token merging.

Ported from dHT with extensions for TAO's multi-scale cross-scale merging.

Includes:
    - Gaussian similarity: exp(-||f_i - f_j||²) - content-based, from dHT
    - Cosine similarity: (f_i · f_j) / (||f_i|| ||f_j||) - semantic-based, from D2O
    - Hybrid similarity: α * Gaussian + β * Cosine - combines both

References:
    - dHT: https://github.com/google-research/dht
    - D2O: "D2O: Dynamic Discriminative Operations" (ICLR 2025)
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Literal

__all__ = [
    'gaussian_similarity',
    'cosine_similarity',
    'hybrid_similarity',
    'get_similarity_fn',
    'compute_edge_features',
    'compute_cross_scale_similarity',
]


def gaussian_similarity(fV: Tensor, E: Tensor) -> Tensor:
    """Compute Gaussian similarity between edge endpoints.

    Formula: sim(i, j) = exp(-||f_i - f_j||²)

    Args:
        fV: Vertex features, shape [nV, C]
        E: Edge indices, shape [2, k] where E[0] = sources, E[1] = targets

    Returns:
        Similarity scores, shape [k], range (0, 1]

    Note:
        This is the core similarity from dHT. It provides:
        - Smooth decay with feature distance
        - Differentiable for gradient-based optimization
        - No hyperparameters (σ=1, assumes normalized features)
        - Natural probabilistic interpretation

    Implementation:
        Uses efficient in-place operations for memory optimization.
        fV[E] → diff → pow(2) → sum → neg_() → exp_()
    """
    # fV[E]: shape [2, k, C] - features of edge endpoints
    # diff: shape [k, C] - differences (f_j - f_i)
    # squared L2 → negative → exp
    return fV[E].diff(1, 0)[0].pow(2).sum(-1).neg_().exp_()


def cosine_similarity(fV: Tensor, E: Tensor) -> Tensor:
    """Compute cosine similarity between edge endpoints.

    Formula: sim(i, j) = (f_i · f_j) / (||f_i|| ||f_j||)

    Args:
        fV: Vertex features, shape [nV, C]
        E: Edge indices, shape [2, k]

    Returns:
        Similarity scores, shape [k], range [0, 1] (mapped from [-1, 1])

    Note:
        This measures angular similarity, useful for semantic features
        where magnitude may vary. Maps from [-1, 1] to [0, 1] for
        consistency with Gaussian similarity.
    """
    fV_i = F.normalize(fV[E[0]], dim=-1)  # shape [k, C]
    fV_j = F.normalize(fV[E[1]], dim=-1)  # shape [k, C]
    # Dot product, map [-1, 1] → [0, 1]
    return (fV_i * fV_j).sum(-1).add(1).mul(0.5)


def hybrid_similarity(
    fV: Tensor,
    E: Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tensor:
    """Compute hybrid Gaussian + Cosine similarity.

    Formula: sim = α * gaussian(f_i, f_j) + β * cosine(f_i, f_j)

    Args:
        fV: Vertex features, shape [nV, C]
        E: Edge indices, shape [2, k]
        alpha: Weight for Gaussian component (content similarity)
        beta: Weight for Cosine component (semantic similarity)

    Returns:
        Weighted similarity scores, shape [k]

    Note:
        alpha + beta should equal 1.0 for normalized output.
        Default 0.5/0.5 balances content and semantic similarity.

        - Gaussian captures local content similarity (from dHT)
        - Cosine captures global semantic similarity (from D2O)
        - Hybrid combines both for robust merging
    """
    g_sim = gaussian_similarity(fV, E)
    c_sim = cosine_similarity(fV, E)
    return alpha * g_sim + beta * c_sim


_simdict = {
    'gaussian': gaussian_similarity,
    'cosine': cosine_similarity,
    'hybrid': hybrid_similarity,
}


def get_similarity_fn(
    similarity: Literal['gaussian', 'cosine', 'hybrid']
) -> Callable[[Tensor, Tensor], Tensor]:
    """Get similarity function by name.

    Args:
        similarity: Name of similarity function

    Returns:
        Callable similarity function
    """
    return _simdict.get(similarity, gaussian_similarity)


def compute_edge_features(
    fV: Tensor,
    E: Tensor,
    mV: Tensor,
    similarity: Literal['gaussian', 'cosine', 'hybrid'] = 'gaussian',
    bb: Optional[Tensor] = None,
    cmp: float = 0.0,
    center: float = 0.5,
    projs2: Optional[Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tensor:
    """Compute edge features with optional compactness regularization.

    Ported from dHT with extensions for TAO.

    Args:
        fV: Vertex features, shape [nV, C]
        E: Edge indices, shape [2, k]
        mV: Region sizes, shape [nV]
        similarity: Similarity function name
        bb: Bounding boxes, shape [4, nV] as [ymin, xmin, ymax, xmax]
        cmp: Compactness factor [0, 1], 0 = disabled
        center: Initial value for edge features
        projs2: Variance projection for self-edges, shape [nV, 1]
        alpha, beta: Hybrid similarity weights

    Returns:
        Edge features, shape [k]

    Note:
        Self-edges (E[0] == E[1]) are handled specially:
        - If projs2 provided: use sigmoid(projs2) as self-edge weight
        - This allows vertices to control their own merge likelihood

        Compactness regularization (if cmp > 0 and bb provided):
        - Favors merges that maintain compact bounding boxes
        - Formula: 4*cmp*size / (2 + height + width)²
        - Prevents "snake-like" elongated regions
    """
    fE = fV.new_full((E.shape[1],), center)  # Initialize with center value
    nloop = torch.ne(*E)  # Non-self edges mask
    loop = ~nloop  # Self-edges mask

    # Compute similarity for non-loop edges
    if similarity == 'hybrid':
        fE[nloop] = hybrid_similarity(fV, E[:, nloop], alpha, beta)
    elif similarity == 'cosine':
        fE[nloop] = cosine_similarity(fV, E[:, nloop])
    else:  # gaussian (default)
        fE[nloop] = gaussian_similarity(fV, E[:, nloop])

    # Optional: Use variance projection for self-edges
    if projs2 is not None:
        # projs2 shape: [nV, 1], sigmoid maps to [0, 1]
        # Self-edge similarity = learned merge likelihood
        fE[loop] = projs2.sigmoid()[E[0, loop], 0]

    # Optional: Compactness regularization
    if cmp > 0 and bb is not None:
        nE = E[:, nloop]  # Non-loop edges
        l0 = E[0, loop]  # Self-edge indices
        mVf = mV.to(fE.dtype)
        ymin, xmin, ymax, xmax = bb.to(fE.dtype)

        fC = torch.zeros_like(fE)

        # Self-edges: compactness of individual regions
        # Higher value for more compact (square-ish) regions
        fC[loop] = (
            4 * cmp * mVf[l0]
            / (2 + ymax[l0] - ymin[l0] + xmax[l0] - xmin[l0]).pow(2)
        )

        # Non-loop edges: compactness of merged regions
        # Computes merged bounding box and evaluates compactness
        fC[nloop] = (
            4 * cmp * mVf[nE].sum(0)
            / (
                2
                + ymax[nE].max(0)[0] - ymin[nE].min(0)[0]
                + xmax[nE].max(0)[0] - xmin[nE].min(0)[0]
            ).pow(2)
        )

        # Weighted combination: (1-cmp)*similarity + cmp*compactness
        fE.mul_(1 - cmp).add_(fC)

    return fE


def compute_cross_scale_similarity(
    fine_features: Tensor,
    coarse_features: Tensor,
    fine_to_coarse_map: Tensor,
    similarity: Literal['gaussian', 'cosine', 'hybrid'] = 'hybrid',
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tensor:
    """Compute similarity between fine and coarse scale tokens.

    TAO extension for cross-scale token merging.

    Args:
        fine_features: Fine-scale features, shape [N_fine, C]
        coarse_features: Coarse-scale features, shape [N_coarse, C]
        fine_to_coarse_map: Mapping from fine to coarse, shape [N_fine]
            fine_to_coarse_map[i] = j means fine token i maps to coarse token j
        similarity: Similarity function name
        alpha, beta: Hybrid weights

    Returns:
        Similarity scores, shape [N_fine]

    Example:
        >>> # 16x16 fine scale → 8x8 coarse scale (2x2 pooling)
        >>> fine = torch.randn(256, 64)   # 16*16 = 256 tokens
        >>> coarse = torch.randn(64, 64)  # 8*8 = 64 tokens
        >>> # Each 2x2 patch maps to 1 coarse token
        >>> mapping = torch.arange(256) // 4  # Simplified mapping
        >>> sim = compute_cross_scale_similarity(fine, coarse, mapping)
        >>> # sim.shape = [256]
    """
    # Get coarse parent features for each fine token
    coarse_parents = coarse_features[fine_to_coarse_map]  # [N_fine, C]

    if similarity == 'gaussian':
        # Gaussian: exp(-||fine - coarse||²)
        diff = fine_features - coarse_parents
        return (-diff.pow(2).sum(-1)).exp()

    elif similarity == 'cosine':
        # Cosine: normalize then dot product
        fine_norm = F.normalize(fine_features, dim=-1)
        coarse_norm = F.normalize(coarse_parents, dim=-1)
        return (fine_norm * coarse_norm).sum(-1).add(1).mul(0.5)

    else:  # hybrid (default)
        # Compute both and combine
        diff = fine_features - coarse_parents
        gaussian_sim = (-diff.pow(2).sum(-1)).exp()

        fine_norm = F.normalize(fine_features, dim=-1)
        coarse_norm = F.normalize(coarse_parents, dim=-1)
        cosine_sim = (fine_norm * coarse_norm).sum(-1).add(1).mul(0.5)

        return alpha * gaussian_sim + beta * cosine_sim


def build_spatial_mapping(
    H_fine: int,
    W_fine: int,
    H_coarse: int,
    W_coarse: int,
    device: torch.device,
) -> Tensor:
    """Build spatial mapping from fine scale to coarse scale.

    Utility function for TAO's multi-scale tokenization.

    Args:
        H_fine, W_fine: Fine scale dimensions
        H_coarse, W_coarse: Coarse scale dimensions
        device: Target device (typically inferred from feature tensors)

    Returns:
        Mapping tensor, shape [H_fine * W_fine]

    Assumes 2x2 downsampling between scales:
        Token (i, j) in fine → token (i//2, j//2) in coarse

    Example:
        >>> mapping = build_spatial_mapping(16, 16, 8, 8, device=torch.device('cpu'))
        >>> mapping.shape  # torch.Size([256])
        >>> mapping[0]     # 0 (top-left fine → top-left coarse)
        >>> mapping[1]     # 0 (still top-left coarse due to 2x2 pooling)
    """
    if H_fine != 2 * H_coarse or W_fine != 2 * W_coarse:
        raise ValueError(
            f"build_spatial_mapping requires 2x2 downsampling between scales.\n"
            f"Got: fine={H_fine}x{W_fine}, coarse={H_coarse}x{W_coarse}\n"
            f"Expected: fine={2*H_coarse}x{2*W_coarse}"
        )

    # Create 2D grid for fine scale
    i_fine = torch.arange(H_fine, device=device).view(-1, 1).expand(H_fine, W_fine)
    j_fine = torch.arange(W_fine, device=device).view(1, -1).expand(H_fine, W_fine)

    # Map to coarse coordinates
    i_coarse = i_fine // 2
    j_coarse = j_fine // 2

    # Flatten to 1D coarse indices
    coarse_idx = i_coarse * W_coarse + j_coarse

    return coarse_idx.view(-1)  # [H_fine * W_fine]
