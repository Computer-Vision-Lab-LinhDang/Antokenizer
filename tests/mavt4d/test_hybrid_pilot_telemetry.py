"""Telemetry must be bounded, numeric and honest about unavailable counters."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from mavt4d.hybrid_pilot.telemetry import collect_gpu_metrics


def test_gpu_metrics_keep_zero_but_omit_unavailable_and_nonfinite():
    payload = [{"gpu": 0, "utilization_pct": 0, "power_w": 275,
                "vram_used_mb": 200, "vram_total_mb": 294592,
                "unsupported": "N/A", "other": float("nan")}]
    with patch("shutil.which", return_value="/opt/rocm/bin/amd-smi"), \
         patch("subprocess.run", return_value=SimpleNamespace(
             stdout=json.dumps(payload), returncode=0)) as run:
        result = collect_gpu_metrics(timeout=2.5)
    assert result["system/gpu_0/utilization_pct"] == 0
    assert result["system/gpu_0/power_w"] == 275
    assert result["system/telemetry_available"] == 1
    assert not any("unsupported" in key or "other" in key for key in result)
    assert run.call_args.kwargs["timeout"] == 2.5
    assert run.call_args.kwargs.get("shell", False) is False


def test_missing_tool_is_not_reported_as_zero_gpu_load():
    with patch("shutil.which", return_value=None):
        assert collect_gpu_metrics() == {"system/telemetry_available": 0.0}


def test_broken_collector_is_nonfatal_and_does_not_log_raw_output():
    with patch("shutil.which", return_value="/usr/bin/amd-smi"), \
         patch("subprocess.run", return_value=SimpleNamespace(
             stdout="unexpected sensitive diagnostic", returncode=1)):
        assert collect_gpu_metrics() == {"system/telemetry_available": 0.0}
