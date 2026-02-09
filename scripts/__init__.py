"""Training and inference scripts for AToken diffusion models.

This module provides command-line tools for:
- Training the D3PM diffusion generator
- Running inference for image/video restoration
- Evaluating model performance

Example:
    # From command line
    python -m atoken.scripts.train_diffusion train --data_dir /path/to/images

    # From Python
    from atoken.scripts.train_diffusion import train, create_model
    model = create_model(d_model=768)
"""

from .train_diffusion import (
    create_model,
    train,
    demo_inference,
)

__all__ = [
    "create_model",
    "train",
    "demo_inference",
]
