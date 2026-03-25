"""pytest conftest: make the repo root importable as 'atoken'.

module5_encoder.py uses `from atoken.core.rope4d import apply_rope_4d`.
The repo root (Antokenizer/) has an __init__.py, so we register it under
the 'atoken' alias before any test imports happen.

SigLIP2 weights may be cached locally, which would cause a dimension
mismatch with the small embed_dim used in tests (64 vs 1152).  We force
the fallback Conv2d path by patching from_pretrained to always raise.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register repo root as 'atoken' package.
spec = importlib.util.spec_from_file_location(
    "atoken",
    REPO_ROOT / "__init__.py",
    submodule_search_locations=[str(REPO_ROOT)],
)
_pkg = importlib.util.module_from_spec(spec)
sys.modules["atoken"] = _pkg
spec.loader.exec_module(_pkg)

# Force SigLIP2 to always fall back to the learnable Conv2d so that tests
# use the embed_dim specified in the test config (64) rather than the real
# SigLIP2 output dimension (1152), even when weights are locally cached.
_siglip_patcher = patch(
    "transformers.SiglipVisionModel.from_pretrained",
    side_effect=RuntimeError("siglip2 disabled in test environment"),
)
_siglip_patcher.start()

# Mirror sub-packages under the 'atoken.*' namespace.
# 'train' is optional — skip if its heavy deps (e.g. webdataset) are absent.
for _sub in ("core", "mavt", "losses"):
    sys.modules[f"atoken.{_sub}"] = importlib.import_module(_sub)

try:
    sys.modules["atoken.train"] = importlib.import_module("train")
except Exception:
    pass
