#!/usr/bin/env python3
"""Train/val leakage audit for the v3 web data: is any scene / source video / asset on both sides?

MAVTDataModule.setup() splits every modality with ``random_split(ds, [n_train, n_val(, n_test)])``:
per *sample*, from the global torch RNG, never written to disk. The run's exact permutation is
therefore not recoverable (it depends on DDP vs single-GPU seeding, on the sizes of the modalities
drawn before it, and on how many 3D objects were rendered at launch). It does not need to be: a
per-sample random split leaks every group with >= 2 members whatever the permutation, so we
simulate the split at the config's fractions over a few seeds and count, per grouping key:

    image  : Open Images manifest  -> key = exact file content (size -> first 64 KiB -> full sha1)
    video  : OpenVid manifest      -> key = source video parsed from the clip name
                                      <ytid>_<scene>_<s>to<e>, celebv_<ytid>_<n>[_<m>],
                                      magictime_<ytid>[_<nnnn>]                      -> yt:<ytid>
                                      magictime_<tiktok id>[_<nnnn>]                 -> tt:<id>
                                      <pixabay|pexels|mixkit>_<words>_<id>_<seg>     -> <site>:<id>
    threed : renders/<uid>/ dir    -> key = exact content of oxoy+oxoz+oyoz (same asset, two uids)

    python scripts/data/check_split.py video data/manifests/openvid_v2.jsonl --val-split 0.002
    python scripts/data/check_split.py image data/manifests/openimages_v2.jsonl --workers 32
    python scripts/data/check_split.py threed data/datasets/objaverse_lvis/3d_objects/renders

``--split-mode group_hash --val-split <f>`` scores the fixed split of mavt.data.splits instead (the
datamodule default since feat/data-g1): items are hashed by the same id the datamodule uses, leaks
are still counted by the content/source keys above, so exact duplicates under two ids still show.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

PLANES = ("oxoy", "oxoz", "oyoz")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from mavt.data.splits import SPLIT_SALT, assign, group_key, video_source  # noqa: E402


def _by(items: Sequence, key: Callable) -> Dict[Hashable, List]:
    out: Dict[Hashable, List] = defaultdict(list)
    for it in items:
        out[key(it)].append(it)
    return out


def _sha1(files: Sequence[str], limit: Optional[int]) -> str:
    h = hashlib.sha1()
    for fp in files:
        with open(fp, "rb") as f:
            if limit:
                h.update(f.read(limit))
            else:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def content_groups(items: Dict[str, Sequence[str]], sizes: Optional[Dict[str, Optional[int]]] = None,
                   workers: int = 16) -> Dict[str, str]:
    """item -> representative item of its exact-duplicate group. Only files whose size collides are
    read, first 64 KiB then in full, so a 1M-image manifest costs a fraction of its bytes.
    ponytail: exact duplicates only; near-duplicates (re-encodes, crops) need a perceptual hash."""
    sizes = sizes or {}
    size = {k: sizes.get(k) or sum(os.path.getsize(f) for f in fs) for k, fs in items.items()}
    cands = [g for g in _by(list(items), size.__getitem__).values() if len(g) > 1]
    for limit in (1 << 16, None):
        flat = [k for g in cands for k in g]
        with ThreadPoolExecutor(workers) as ex:
            digest = dict(zip(flat, ex.map(lambda k: _sha1(items[k], limit), flat)))
        cands = [sub for g in cands for sub in _by(g, digest.__getitem__).values() if len(sub) > 1]
    rep = {k: k for k in items}
    for g in cands:
        for k in g:
            rep[k] = g[0]
    return rep


def simulate_split(n: int, val_split: float, test_split: float, seed: int) -> List[str]:
    """Same sizes and layout as MAVTDataModule.setup()'s random_split: a uniform permutation,
    first n_train = train, next n_val = val, rest = test."""
    n_val = max(1, int(n * val_split))
    n_test = max(1, int(n * test_split)) if test_split > 0 else 0
    n_train = n - n_val - n_test
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    assign = [""] * n
    for rank, i in enumerate(perm):
        assign[i] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return assign


def leak_table(keys: Sequence[Hashable], assign: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Per held-out split: items, groups, items / groups whose key also has a train member."""
    train_keys = {k for k, s in zip(keys, assign) if s == "train"}
    out: Dict[str, Dict[str, int]] = {}
    for split in sorted(set(assign) - {"train"}):
        ks = [k for k, s in zip(keys, assign) if s == split]
        out[split] = {"items": len(ks), "groups": len(set(ks)),
                      "leaked_items": sum(k in train_keys for k in ks),
                      "leaked_groups": len({k for k in ks if k in train_keys})}
    return out


# --------------------------------------------------------------------------- per-modality keys

def _read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def video_keys(manifest: str) -> Tuple[List[str], Dict]:
    keys, origins, unparsed = [], defaultdict(set), []
    for r in _read_jsonl(manifest):
        key, origin = video_source(r["path"])
        if key is None:
            unparsed.append(r["path"])
            key = "?" + r["path"]                        # own group, but counted and shown in notes
        keys.append(key)
        origins[key].add(origin)
    stock_sites = defaultdict(set)
    for k in origins:
        site, _, num = k.partition(":")
        if site in ("pixabay", "pexels", "mixkit"):
            stock_sites[num].add(site)
    return keys, {
        "unparsed_names": len(unparsed),
        "unparsed_examples": [os.path.basename(p) for p in unparsed[:5]],
        "yt_ids_in_both_celebv_and_panda": sum({"celebv", "panda"} <= o for o in origins.values()),
        "stock_ids_under_two_sites": sum(len(s) > 1 for s in stock_sites.values()),
        "groups_per_origin": dict(Counter(o for os_ in origins.values() for o in os_)),
    }


def image_keys(manifest: str, workers: int) -> Tuple[List[str], Dict]:
    recs = _read_jsonl(manifest)
    paths = [r["path"] for r in recs]
    rep = content_groups({p: [p] for p in paths}, {r["path"]: r.get("bytes") for r in recs}, workers)
    stems = Counter(os.path.splitext(os.path.basename(p))[0] for p in paths)
    return [rep[p] for p in paths], {"duplicate_paths": len(paths) - len(set(paths)),
                                     "duplicate_ids": sum(c - 1 for c in stems.values() if c > 1)}


def threed_uids(renders_dir: str) -> List[str]:
    """Objects exactly as UniversalThreeDDataset lists them (sorted, all three planes present)."""
    return sorted(d for d in os.listdir(renders_dir)
                  if not d.startswith(".") and os.path.isdir(os.path.join(renders_dir, d))
                  and all(os.path.exists(os.path.join(renders_dir, d, f"{p}.png")) for p in PLANES))


def item_ids(kind: str, source: str) -> List[str]:
    """Per-sample ids in the order the *_keys functions return keys (what the datamodule hashes)."""
    return threed_uids(source) if kind == "threed" else [r["path"] for r in _read_jsonl(source)]


def hash_assign(kind: str, ids: Sequence[str], val_split: float, test_split: float, salt: str) -> List[str]:
    """The datamodule's split_mode="group_hash" assignment (mavt.data.splits)."""
    return [assign(group_key(kind, i), val_split, test_split, salt) for i in ids]


def threed_keys(renders_dir: str, workers: int) -> Tuple[List[str], Dict]:
    uids = threed_uids(renders_dir)
    items = {u: [os.path.join(renders_dir, u, f"{p}.png") for p in PLANES] for u in uids}
    rep = content_groups(items, workers=workers)
    expected = {f"{p}.png" for p in PLANES}
    extra = sum(bool(set(os.listdir(os.path.join(renders_dir, u))) - expected) for u in uids)
    return [rep[u] for u in uids], {"dirs_with_files_besides_planes": extra}


# --------------------------------------------------------------------------- report

def summarize(keys: Sequence[Hashable], val_split: float, test_split: float, seeds: Sequence[int],
              fixed_assign: Optional[Sequence[str]] = None) -> Dict:
    """Leak counts over simulated random splits (one per seed), or over one fixed assignment."""
    sizes = Counter(keys)
    hist = Counter(min(c, 10) for c in sizes.values())
    runs = ([leak_table(keys, fixed_assign)] if fixed_assign is not None else
            [leak_table(keys, simulate_split(len(keys), val_split, test_split, s)) for s in seeds])
    held = {split: {f: [min(r[split][f] for r in runs), sum(r[split][f] for r in runs) / len(runs),
                        max(r[split][f] for r in runs)] for f in runs[0][split]}
            for split in runs[0]}
    return {"items": len(keys), "groups": len(sizes),
            "items_in_multi_member_groups": sum(c for c in sizes.values() if c > 1),
            "largest_group": max(sizes.values(), default=0),
            "group_size_hist": {("10+" if k == 10 else str(k)): v for k, v in sorted(hist.items())},
            "held_out": held, "seeds": [] if fixed_assign is not None else list(seeds),
            "val_split": val_split, "test_split": test_split}


def print_report(kind: str, src: str, s: Dict, notes: Dict) -> None:
    multi = s["items_in_multi_member_groups"]
    print(f"## {kind}: {src}")
    print(f"items {s['items']:,} | groups {s['groups']:,} | items sharing a group {multi:,} "
          f"({100 * multi / max(1, s['items']):.2f}%) | largest group {s['largest_group']}")
    print("group size histogram (size: groups):", ", ".join(f"{k}: {v:,}" for k, v in s["group_size_hist"].items()))
    how = f"seeds {s['seeds']}" if s["seeds"] else f"group_hash split, salt {s.get('salt')!r}"
    print(f"\nsplit: val {s['val_split']}, test {s['test_split']}, {how} (min / mean / max)\n")
    print("| split | items | groups | items whose group is also in train | groups also in train |")
    print("|---|---|---|---|---|")
    for split, r in s["held_out"].items():
        pct = 100 * r["leaked_items"][1] / max(1, r["items"][1])
        print(f"| {split} | {r['items'][1]:,.0f} | {r['groups'][1]:,.1f} | "
              f"{r['leaked_items'][0]:,} / {r['leaked_items'][1]:,.1f} / {r['leaked_items'][2]:,} ({pct:.1f}%) | "
              f"{r['leaked_groups'][0]:,} / {r['leaked_groups'][1]:,.1f} / {r['leaked_groups'][2]:,} |")
    print("\nnotes:", json.dumps(notes, ensure_ascii=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["image", "video", "threed"])
    ap.add_argument("source", help="manifest .jsonl (image/video) or renders/ dir (threed)")
    ap.add_argument("--val-split", type=float, default=0.002, help="v3 configs use 0.002")
    ap.add_argument("--test-split", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=16, help="threads for hashing size-colliding files")
    ap.add_argument("--json", help="also write the summary here")
    ap.add_argument("--split-mode", choices=["random", "group_hash"], default="random",
                    help="random = simulate the old per-sample random_split; group_hash = the fixed split")
    ap.add_argument("--salt", default=SPLIT_SALT)
    a = ap.parse_args(argv)
    if a.kind == "video":
        keys, notes = video_keys(a.source)
    elif a.kind == "image":
        keys, notes = image_keys(a.source, a.workers)
    else:
        keys, notes = threed_keys(a.source, a.workers)
    if not keys:
        print(f"no items in {a.source}", file=sys.stderr)
        return 1
    fixed = (hash_assign(a.kind, item_ids(a.kind, a.source), a.val_split, a.test_split, a.salt)
             if a.split_mode == "group_hash" else None)
    s = summarize(keys, a.val_split, a.test_split, a.seeds, fixed)
    if fixed is not None:
        s["salt"] = a.salt
    print_report(a.kind, a.source, s, notes)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"kind": a.kind, "source": a.source, **s, "notes": notes}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
