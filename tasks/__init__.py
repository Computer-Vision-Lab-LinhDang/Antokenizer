"""Task adapters for different modalities."""

from .image_rec import ImageReconstructionTask
from .image_sem import ImageSemanticTask
from .object_cls import ObjectClassificationTask
from .diffusion_gen import DiffusionGenerationTask, DiffusionGenerationTaskV2

__all__ = [
	"ImageReconstructionTask",
	"ImageSemanticTask",
	"ObjectClassificationTask",
	"DiffusionGenerationTask",
	"DiffusionGenerationTaskV2",
]
