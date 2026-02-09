"""
AToken package initialization.

Provides convenience imports for Stage 1 image pipeline components
and the D3PM diffusion generator.

Modules:
    core: Core tensor utilities (SparseTensor4D, patchify, RoPE)
    d2o: Dynamic token merging with information criteria (IC-guided merging)
    model: Neural network components (encoder, decoder, blocks)
    diffusion: D3PM discrete diffusion modules
    tasks: Training tasks (reconstruction, semantic, diffusion)
    train: Training utilities (Trainer, EMA, DataLoader, degradation)
    infer: Inference utilities (encoding, decoding, generation)
    losses: Loss functions (reconstruction, perceptual, CLIP)
    scripts: Training and inference scripts

Example:
    >>> from atoken.diffusion import DiffusionGenerator
    >>> from atoken.train import create_diffusion_dataloader
    >>> from atoken.infer import DiffusionInference
    >>>
    >>> # Create model
    >>> generator = DiffusionGenerator(d_model=768)
    >>>
    >>> # Create dataloader
    >>> loader = create_diffusion_dataloader("/path/to/images", batch_size=8)
    >>>
    >>> # Run inference
    >>> inference = DiffusionInference(generator)
    >>> restored = inference.restore(degraded_image)
"""

from . import core, d2o, diffusion, infer, losses, model, scripts, tasks, train

__all__ = [
    "core",
    "d2o",
    "diffusion",
    "model",
    "losses",
    "tasks",
    "train",
    "infer",
    "scripts",
]
