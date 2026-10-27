"""Versioned query pilot configuration and source/target manifest admission."""
from .config import GrainPriorConfig

BASELINE_REVISION = "a7aba58d5f344b3fe14fd0bd6f98bdb42f194594"
PROTOCOL_VERSION = "shared-query-pilot-v1"


def pilot_config(readout_version, **overrides):
    values = dict(stratum=1, d_f=12, rgb_from="hidden", read_qk_norm=True,
                  sigma_min=0.15, image_fov_fixed=True, w_lpips=0.0,
                  readout_version=readout_version, measurement_version=2,
                  innovation_semantics="disabled")
    values.update(overrides)
    cfg = GrainPriorConfig(**values)
    cfg.check()
    return cfg


def validate_manifest(rows):
    """No target may be a source; group identity is supplied by the dataset split audit.

    This validates declared grouping, not the truth of an arbitrary group_id.
    """
    groups, scenes = {}, set()
    if not rows:
        raise ValueError("empty manifest")
    for row in rows:
        for key in ("scene_id", "group_id", "split", "source_ids", "target_ids"):
            if not row.get(key):
                raise ValueError(f"missing {key}")
        group, split = row["group_id"], row["split"]
        if split not in ("train", "val"):
            raise ValueError("only registered train/val splits are supported")
        if group in groups and groups[group] != split:
            raise ValueError("group crosses train/val split")
        if set(row["source_ids"]) & set(row["target_ids"]):
            raise ValueError("target view is also a source")
        groups[group] = split
        scenes.add(row["scene_id"])
    return dict(scenes=len(scenes), groups=len(groups), version=PROTOCOL_VERSION)
