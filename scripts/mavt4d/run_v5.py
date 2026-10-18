#!/usr/bin/env python3
"""Explicit natural v5 runner; allocation and credentials are provisioned outside."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mavt4d.object_state.v5.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
