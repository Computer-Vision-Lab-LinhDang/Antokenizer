"""Deterministic grouped train/val/test split.

Every sample is assigned from ``sha1(salt + group key)``, never from an RNG permutation, so

  * all members of a group (the clips cut from one source video, one Open Images id, one 3D uid)
    land on the same side: no clip in val has a sibling clip in train;
  * the assignment of a sample never changes when the manifest grows, shrinks or is reordered,
    so stage 1 -> stage 3 and resumes with more data keep the same val set;
  * raising a fraction only moves samples train -> val (or -> test), never back.

Fractions are fractions of *groups*. ``DEFAULT_VAL`` sizes val to a few hundred groups per
modality on the v3 data (OpenVid 15k source videos, Open Images 0.2-1.1M ids, LVIS 46k objects).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

SPLIT_SALT = "mavt-split-v1"
DEFAULT_VAL: Dict[str, float] = {"image": 0.005, "video": 0.02, "threed": 0.01}

_YTID = r"[A-Za-z0-9_-]{11}"                                       # YouTube ids are exactly 11 chars, may hold "_"/"-"
_YT = re.compile(rf"^(?P<id>{_YTID})_\d+_\d+to\d+$")              # Panda-70M: <ytid>_<scene>_<start>to<end>
_CELEBV = re.compile(rf"^celebv_(?P<id>{_YTID})(?:_\d+)+$")        # CelebV-HQ: celebv_<ytid>_<n>[_<m>]
_MAGICTIME = re.compile(rf"^magictime_(?:(?P<tt>\d{{15,20}})|(?P<id>{_YTID}))(?:_\d+)*$")  # <ytid|tiktok id>[_<nnnn>...]
_STOCK = re.compile(r"^(?P<site>pixabay|pexels|mixkit)_(?:.*_)?(?P<id>\d+)_\d+$")  # <site>_<words>_<id>_<seg>


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def video_source(path: str) -> Tuple[Optional[str], str]:
    """(source key, origin tag) of an OpenVid clip; key None = name pattern not recognised."""
    stem = _stem(path)
    m = _CELEBV.match(stem)
    if m:
        return f"yt:{m['id']}", "celebv"
    m = _MAGICTIME.match(stem)
    if m:
        return (f"tt:{m['tt']}" if m["tt"] else f"yt:{m['id']}"), "magictime"
    m = _STOCK.match(stem)
    if m:
        return f"{m['site']}:{m['id']}", m["site"]
    m = _YT.match(stem)
    if m:
        return f"yt:{m['id']}", "panda"
    return None, "unparsed"


def group_key(modality: str, item_id: str) -> str:
    """Group of one sample. ``item_id`` is the manifest path (image/video) or the 3D uid.
    ponytail: exact ids only; the same picture/asset under two ids is not merged (check_split.py
    measures how often that happens)."""
    if modality == "video":
        key, _ = video_source(item_id)
        return key if key is not None else f"clip:{_stem(item_id)}"   # unparsed name -> own group
    if modality == "image":
        return f"img:{_stem(item_id)}"                                  # Open Images id = file stem
    if modality == "threed":
        return f"obj:{os.path.basename(os.path.normpath(item_id))}"
    raise ValueError(f"unknown modality {modality!r}")


def bucket(key: str, salt: str = SPLIT_SALT) -> float:
    """Uniform position of a group in [0, 1), a pure function of (salt, key)."""
    return int(hashlib.sha1(f"{salt}\0{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def assign(key: str, val_frac: float, test_frac: float = 0.0, salt: str = SPLIT_SALT) -> str:
    if not (0.0 <= val_frac and 0.0 <= test_frac and val_frac + test_frac < 1.0):
        raise ValueError(f"need 0 <= val, test and val + test < 1, got {val_frac}, {test_frac}")
    u = bucket(key, salt)
    return "val" if u < val_frac else "test" if u < val_frac + test_frac else "train"


def split_indices(keys: Sequence[str], val_frac: float, test_frac: float = 0.0,
                  salt: str = SPLIT_SALT) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    cache: Dict[str, str] = {}
    for i, k in enumerate(keys):
        s = cache.get(k)
        if s is None:
            s = cache[k] = assign(k, val_frac, test_frac, salt)
        out[s].append(i)
    return out


def fraction_for(modality: str, frac: Union[float, Mapping[str, float], None],
                 default: Mapping[str, float]) -> float:
    """``val_split``/``test_split`` may be one float for every modality or a per-modality dict."""
    if frac is None:
        return default.get(modality, 0.0)
    if isinstance(frac, Mapping):
        return float(frac.get(modality, default.get(modality, 0.0)))
    return float(frac)


def item_ids(ds) -> Optional[List[str]]:
    """Per-index ids of the datasets in mavt.data.datasets, or None when the dataset does not
    expose them (WebDataset shards, synthetic): the caller then falls back to a random split."""
    if hasattr(ds, "records"):                 # Manifest{Image,Video}Dataset
        return [r["path"] for r in ds.records]
    if hasattr(ds, "obj_dirs"):                # UniversalThreeDDataset
        return [d.name for d in ds.obj_dirs]
    if hasattr(ds, "paths"):                   # Universal{Image,Video}Dataset
        return [str(p) for p in ds.paths]
    return None
