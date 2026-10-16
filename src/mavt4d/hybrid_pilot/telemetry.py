"""Read-only AMD counters in a bounded subprocess, without extra packages.

The installed amd-smi Python library is used directly because some guest CLI
versions fail in their metric argument parser. No device settings are changed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


_COLLECTOR = """
import json,sys
sys.path.insert(0,sys.argv[1])
from amdsmi import amdsmi_interface as a
a.amdsmi_init(a.AmdSmiInitFlags.INIT_AMD_GPUS)
rows=[]
try:
    for gpu,handle in enumerate(a.amdsmi_get_processor_handles()):
        row={'gpu':gpu}
        for getter,fields in [
            ('amdsmi_get_gpu_activity',{'gfx_activity':'utilization_pct'}),
            ('amdsmi_get_power_info',{'current_socket_power':'power_w'}),
            ('amdsmi_get_gpu_vram_usage',{'vram_used':'vram_used_mb','vram_total':'vram_total_mb'})]:
            try:
                values=getattr(a,getter)(handle)
                row.update({name:values[key] for key,name in fields.items() if key in values})
            except Exception:
                pass
        rows.append(row)
    print(json.dumps(rows))
finally:
    a.amdsmi_shut_down()
"""
_COUNTERS = frozenset({"utilization_pct", "power_w", "vram_used_mb", "vram_total_mb"})


def collect_gpu_metrics(*, timeout: float = 3.0) -> dict[str, float]:
    """Return available numeric counters; missing readings are never zero-filled.

    Call on rank zero at logging boundaries. This measures device-wide activity;
    it is not an attribution of board power or memory to this training process.
    """
    unavailable = {"system/telemetry_available": 0.0}
    executable = shutil.which("amd-smi")
    if not executable:
        return unavailable
    # Resolving the executable locates the corresponding versioned installation.
    library = Path(executable).resolve().parent.parent.parent / "share" / "amd_smi"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _COLLECTOR, str(library)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode:
            return unavailable
        rows = json.loads(result.stdout)
        if not isinstance(rows, list):
            return unavailable
        metrics: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get("gpu")) is not int or row["gpu"] < 0:
                continue
            for counter in _COUNTERS:
                value = row.get(counter)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    metrics[f"system/gpu_{row['gpu']}/{counter}"] = float(value)
        return {"system/telemetry_available": float(bool(metrics)), **metrics}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return unavailable
