"""Filter read-only board counters to this explicitly assigned GPU set."""
import os

from mavt4d.hybrid_pilot.telemetry import collect_gpu_metrics


def selected_gpu_metrics():
    raw = os.environ.get("MAVT_PHYSICAL_GPU_IDS", os.environ.get("ROCR_VISIBLE_DEVICES", ""))
    parts = raw.split(",")
    if not parts or any(not part.strip().isdigit() for part in parts):
        return {"system/telemetry_available": 0.}
    ids = {int(part) for part in parts}
    allowed = tuple(f"system/gpu_{index}/" for index in sorted(ids))
    metrics = collect_gpu_metrics()
    selected = {name: value for name, value in metrics.items() if name.startswith(allowed)}
    return {"system/telemetry_available": float(bool(selected)), **selected}
