#!/usr/bin/env python3
"""Export audited temporal-event fixtures into a new, versioned directory.

Model inputs and evaluation targets are deliberately written to separate files.
The output directory must not already exist, even when it is empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from mavt4d.data.event_synthetic import (
    EVENT_CONFIGURATIONS,
    EVENT_CORES,
    EventSceneConfig,
    make_event_fixture,
)


_REVERSAL_PAIRS = (("left_then_right", "right_then_left"), ("clockwise", "counterclockwise"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _tensor_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


def _render_fingerprint(cfg: EventSceneConfig) -> str:
    # Split membership is bookkeeping; version and every rendering field define
    # the observation content. Never include filesystem locations or Git state.
    render_config = {name: value for name, value in asdict(cfg).items() if name != "val_fraction"}
    return hashlib.sha256(_json_line(render_config).encode("utf-8")).hexdigest()


def _record_id(render_fingerprint: str, seed: int, core: str, configuration: str) -> str:
    identity = json.dumps([render_fingerprint, seed, core, configuration], separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _frame_multiset(images: torch.Tensor) -> list[str]:
    # Include all views in each frame's digest; there is one view in this version.
    return sorted(
        hashlib.sha256(images[:, t].contiguous().numpy().tobytes()).hexdigest()
        for t in range(images.shape[1])
    )


def _audit_group(fixtures: dict, cfg: EventSceneConfig, seed: int) -> None:
    """Fail on an observed invariant violation rather than exporting an invalid suite."""
    if set(fixtures) != {(core, config) for core in EVENT_CORES for config in EVENT_CONFIGURATIONS}:
        raise ValueError("event fixture crossing is incomplete")
    groups, splits = set(), set()
    for (core, configuration), fixture in fixtures.items():
        scene, target = fixture.scene, fixture.evaluation
        scene.validate()
        if (target.version, target.source_seed, target.core, target.configuration) != (
            cfg.version, seed, core, configuration
        ):
            raise ValueError("fixture evaluation identity does not match its request")
        groups.add(target.group_key)
        splits.add(target.split)
        if scene.images.shape != (1, cfg.T, 3, cfg.H, cfg.W):
            raise ValueError("fixture RGB shape does not match configuration")
        if not torch.isfinite(scene.images).all() or scene.images.abs().max() > 1:
            raise ValueError("fixture RGB must be finite and in [-1, 1]")
        if not torch.isfinite(target.positions).all():
            raise ValueError("fixture positions must be finite")
        visible = target.gt.instance == target.target_instance_id
        if not bool((visible.flatten(2).sum(-1) >= 4).all()):
            raise ValueError("target must remain visible with at least four pixels in every frame")
        if not torch.isfinite(target.gt.depth).all() or not bool((target.gt.depth[visible] > 0).all()):
            raise ValueError("target depth must be finite and positive")
    if len(groups) != 1 or len(splits) != 1 or not splits <= {"train", "val"}:
        raise ValueError("all variants of one source must share a group and split")

    for configuration in EVENT_CONFIGURATIONS:
        for first, second in combinations(EVENT_CORES, 2):
            a, b = fixtures[first, configuration], fixtures[second, configuration]
            if not torch.equal(a.scene.images[:, -2:], b.scene.images[:, -2:]):
                raise ValueError("final-two RGB suffix differs across event cores")
            if not torch.equal(a.evaluation.positions[-2:], b.evaluation.positions[-2:]):
                raise ValueError("final-two position suffix differs across event cores")
            if torch.equal(a.scene.images[:, :-2], b.scene.images[:, :-2]):
                raise ValueError("event cores must have different rendered histories")
            for name in ("frame_times", "intrinsics", "cam_to_world"):
                if not torch.equal(getattr(a.scene, name), getattr(b.scene, name)):
                    raise ValueError(f"{name} must not identify the event core")
        for first, second in _REVERSAL_PAIRS:
            a, b = fixtures[first, configuration], fixtures[second, configuration]
            if not torch.equal(a.scene.images[:, :-2].flip(1), b.scene.images[:, :-2]):
                raise ValueError("reversal pair must have exactly reversed rendered history")
            if _frame_multiset(a.scene.images) != _frame_multiset(b.scene.images):
                raise ValueError("reversal pair must have the same frame multiset")


def _input_payload(fixture) -> dict[str, torch.Tensor]:
    scene = fixture.scene
    return {
        "images": _tensor_copy(scene.images),
        "observed": torch.ones(
            scene.images.shape[:2] + scene.images.shape[-2:], dtype=torch.bool
        ),
        "frame_times": _tensor_copy(scene.frame_times),
        "intrinsics": _tensor_copy(scene.intrinsics),
        "cam_to_world": _tensor_copy(scene.cam_to_world),
    }


def _target_payload(fixture) -> dict:
    target = fixture.evaluation
    return {
        "positions": _tensor_copy(target.positions),
        "boxes": [
            {name: _tensor_copy(getattr(box, name)) for name in ("center", "size", "yaw", "instance_id")}
            for box in target.boxes
        ],
        "gt": {
            name: _tensor_copy(getattr(target.gt, name))
            for name in ("world_xyz", "depth", "instance", "valid")
        },
    }


def _contact_sheet(fixtures: dict, cfg: EventSceneConfig, seed: int, path: Path) -> None:
    scale, margin, label_width, header = 2, 8, 150, 44
    cell_w, cell_h = cfg.W * scale, cfg.H * scale
    row_height = cell_h + margin
    width = label_width + cfg.T * (cell_w + margin) + margin
    height = header + len(EVENT_CORES) * row_height + 32
    canvas = Image.new("RGB", (width, height), "#17202b")
    draw = ImageDraw.Draw(canvas)
    configuration = EVENT_CONFIGURATIONS[0]
    draw.text((8, 6), f"{cfg.version} | source {seed} | {configuration} | green = identical final two frames", fill="white")
    for t in range(cfg.T):
        x = label_width + t * (cell_w + margin)
        draw.text((x, 25), f"t={t}", fill="#cbd5e1")
    for row, core in enumerate(EVENT_CORES):
        y = header + row * row_height
        draw.text((8, y + cell_h // 2), core, fill="white")
        images = fixtures[core, configuration].scene.images[0]
        for t in range(cfg.T):
            x = label_width + t * (cell_w + margin)
            pixels = ((images[t].permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
            frame = Image.fromarray(pixels).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            canvas.paste(frame, (x, y))
            if t >= cfg.T - 2:
                draw.rectangle((x - 2, y - 2, x + cell_w + 1, y + cell_h + 1), outline="#56e39f", width=3)
    draw.text((8, height - 23), "Evaluation preview only: row labels are never included in model tensor files.", fill="#cbd5e1")
    canvas.save(path, format="PNG")


def build_event_suite(out: Path, cfg: EventSceneConfig, source_seeds) -> dict:
    """Render a complete source x core x configuration crossing and audit every group.

    A failed build may leave an incomplete new directory, but never a success
    summary. It is intentionally not removed or reused on a later invocation.
    """
    if not isinstance(cfg, EventSceneConfig):
        raise TypeError("cfg must be EventSceneConfig")
    cfg = EventSceneConfig(**asdict(cfg))  # Revalidate before creating output.
    seeds = tuple(source_seeds)
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("source_seeds must be a nonempty collection of unique integers")
    if any(seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise ValueError("source_seeds must be in [0, 2**63 - 1]")
    seeds = tuple(sorted(seeds))
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    (out / "targets").mkdir()

    sample_counts, source_counts = Counter(), Counter()
    input_hashes, target_hashes = {}, {}
    render_fingerprint = _render_fingerprint(cfg)
    manifest = out / "manifest_v1.jsonl"
    evaluation = out / "eval_targets_v1.jsonl"
    with manifest.open("x", encoding="utf-8") as inputs, evaluation.open("x", encoding="utf-8") as targets:
        for seed in seeds:
            fixtures = {
                (core, configuration): make_event_fixture(cfg, seed, core, configuration)
                for core in EVENT_CORES
                for configuration in EVENT_CONFIGURATIONS
            }
            _audit_group(fixtures, cfg, seed)
            source_counts[next(iter(fixtures.values())).evaluation.split] += 1
            if seed == seeds[0]:
                _contact_sheet(fixtures, cfg, seed, out / "contact_sheet.png")
            for (core, configuration), fixture in fixtures.items():
                target = fixture.evaluation
                record_id = _record_id(render_fingerprint, seed, core, configuration)
                input_path = f"inputs/{record_id}.pt"
                target_path = f"targets/{record_id}.pt"
                torch.save(_input_payload(fixture), out / input_path)
                torch.save(_target_payload(fixture), out / target_path)
                input_hashes[input_path] = _sha256(out / input_path)
                target_hashes[target_path] = _sha256(out / target_path)
                sample_counts[target.split] += 1
                common = {
                    "record_id": record_id, "group_key": target.group_key, "split": target.split,
                    "render_config_sha256": render_fingerprint,
                }
                inputs.write(_json_line({**common, "input_path": input_path}))
                targets.write(_json_line({
                    **common, "version": target.version, "source_seed": target.source_seed,
                    "core": core, "configuration": configuration,
                    "target_instance_id": target.target_instance_id, "target_path": target_path,
                }))

    summary = {
        "version": cfg.version,
        "config": asdict(cfg),
        "render_config_sha256": render_fingerprint,
        "source_seeds": list(seeds),
        "counts": {
            "samples": {split: sample_counts[split] for split in ("train", "val")},
            "sources": {split: source_counts[split] for split in ("train", "val")},
        },
        "audits": {
            "identical_final_two": True, "different_histories": True,
            "reversal_frame_multiset": True, "target_visible": True,
        },
        "sha256": {
            "manifest": _sha256(manifest), "evaluation": _sha256(evaluation),
            "inputs": input_hashes, "targets": target_hashes,
        },
    }
    (out / "README.md").write_text(
        f"# {cfg.version} synthetic event fixtures\n\n"
        "![Temporal histories and common final tubelet](contact_sheet.png)\n\n"
        "`manifest_v1.jsonl` points to observation-only tensor files in `inputs/`. "
        "`eval_targets_v1.jsonl` contains labels and points to geometry in `targets/`. "
        "Do not pass record IDs, group keys, split labels, evaluation rows or target files to a model.\n\n"
        "All four event cores share the exact final two images for a source and configuration. "
        "Reversal pairs also share the exact frame multiset; their labels require temporal order. "
        "The contact sheet shows one source and configuration; summary audits cover every source.\n\n"
        "These are controlled fixtures, not a trained model or evidence of learned event semantics. "
        "The complete input file contains the full clip: use a causal prefix adapter before recurrent evaluation.\n",
        encoding="utf-8",
    )
    # This file is the completion marker and is written only after all audits and artifacts succeed.
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.source_count <= 0:
        parser.error("--source-count must be positive")
    if args.seed_start < 0 or args.seed_start + args.source_count - 1 > 2**63 - 1:
        parser.error("seed range must be inside [0, 2**63 - 1]")
    cfg = EventSceneConfig(T=args.frames, H=args.height, W=args.width, val_fraction=args.val_fraction)
    torch.set_num_threads(1)
    summary = build_event_suite(args.output, cfg, range(args.seed_start, args.seed_start + args.source_count))
    print(json.dumps({"version": summary["version"], "counts": summary["counts"], "audits": summary["audits"]}, indent=2))


if __name__ == "__main__":
    main()
