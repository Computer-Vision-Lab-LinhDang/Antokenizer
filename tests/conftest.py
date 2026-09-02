import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Tests that download SigLIP2-SO400M (~4 GB) only run when explicitly enabled.
RUN_HF = os.environ.get("MAVT_RUN_HF_TESTS") == "1"
requires_hf = pytest.mark.skipif(not RUN_HF, reason="set MAVT_RUN_HF_TESTS=1 to run SigLIP2 download tests")
