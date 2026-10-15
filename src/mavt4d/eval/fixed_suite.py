"""E1.3 fixed test suite v1: mine roadmap §1.4 clip categories from nuScenes val annotations.

Pure logic over small per-keyframe records (no file IO). The loader lives in
scripts/mavt4d/build_fixed_suite.py. A clip is a window of K consecutive keyframes (2 Hz) inside one
scene; categories are decided from annotator attributes first and world displacement second:

  moving instance : moving attribute on >= half its annotations in the window AND displacement >= 3 m
                    (vehicle.moving, pedestrian.moving; cycle.with_rider counts only through that
                    displacement requirement)
  static instance : no vehicle.moving / pedestrian.moving attribute AND displacement <= 1 m
  near            : min xy distance to the LIDAR_TOP ego position over the instance's annotations
                    in the window (occlusion: max over the three frames of the pattern)

Ego pose is the LIDAR_TOP keyframe's ego_pose; all distances and displacements are in the xy plane.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

SUITE = "fixed_suite_v1"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")  # ring
LIDAR = "LIDAR_TOP"

STATIC = "static_scene_moving_camera"
FIXED = "fixed_camera_moving_object"
BOTH = "moving_camera_moving_object"
OCCL = "occlusion_reappearance"
EVENT = "short_event"
PAIR = "same_final_position_different_direction"
MISSING = "missing_modality"
RARE = "rare_category"
CATEGORIES = (STATIC, FIXED, BOTH, OCCL, EVENT, PAIR, MISSING, RARE)  # roadmap §1.4 order (1)-(8)
WINDOW_CATEGORIES = (STATIC, FIXED, BOTH, OCCL, EVENT, RARE)  # decided per window

STRICT_MOVING = frozenset({"vehicle.moving", "pedestrian.moving"})
MOVING_ATTRS = STRICT_MOVING | {"cycle.with_rider"}
EVENT_STATES = {  # attribute -> (group, state); an event is a state change within one group
    "vehicle.moving": ("vehicle", "moving"), "vehicle.stopped": ("vehicle", "still"),
    "vehicle.parked": ("vehicle", "still"), "pedestrian.moving": ("pedestrian", "moving"),
    "pedestrian.standing": ("pedestrian", "still"),
}
MISSING_VARIANTS = ("lidar_dropped", "half_cameras_dropped", "cam_front_only")
HALF_CAMERAS_KEPT = ("CAM_FRONT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT")  # every other camera around the ring

# --- every threshold of the suite, in one place --------------------------------------------------------
_MOTION = {"moving_attr_frac": 0.5, "moving_min_disp_m": 3.0, "static_max_disp_m": 1.0}
THRESHOLDS: dict[str, dict] = {
    STATIC: {**_MOTION, "ego_min_disp_m": 5.0, "min_static_instances": 1},
    FIXED: {**_MOTION, "ego_max_dev_m": 0.5, "ego_max_dyaw_deg": 5.0, "near_m": 40.0, "min_moving": 1},
    BOTH: {**_MOTION, "ego_min_disp_m": 5.0, "near_m": 40.0, "min_moving": 2},
    OCCL: {"near_m": 50.0, "visible_min_bin": 3, "occluded_bin": 1, "occluded_lidar_pts": 0},
    EVENT: {"near_m": 40.0},
    PAIR: {**_MOTION, "near_m": 40.0, "max_final_dist_m": 2.0, "min_heading_diff_deg": 90.0},
    MISSING: {"bases_per_category": 2, "variants": list(MISSING_VARIANTS)},
    RARE: {"max_train_share": 0.005, "near_m": 40.0},
}
PER_SCENE = 2

_BASE_NOTE = ("nuScenes val keyframes only (2 Hz, no sweeps); ego = LIDAR_TOP keyframe ego_pose; distances "
              "and displacements are xy metres in the global frame.")
NOTES = {
    STATIC: "Every instance with >= 2 annotations in the window is static.",
    FIXED: "Ego nearly still (max deviation from the first keyframe and max |yaw change|).",
    BOTH: "Ego translation is first-to-last keyframe.",
    OCCL: ("visibility bins (1: 0-40%, ..., 4: 80-100%) are aggregated over all 6 cameras, so an occlusion here "
           "need not be an occlusion in any single camera."),
    EVENT: "Annotator attribute change of one instance between consecutive keyframes.",
    PAIR: ("Annotation-mined approximate pair (final position in the last keyframe's ego frame, displacement "
           "heading in that frame, yaw only); synthetic T6 will supply exact pairs."),
    MISSING: ("Derived from another selected clip: missing = mask False, never zero values; the referenced "
              "tokens are kept so a loader can still verify what it drops."),
    RARE: ("Class has < 0.5% of TRAIN instances. Stands in for 'asset outside the dynamics set' only if these "
           "classes are excluded from dynamics training."),
}

_SALT = "mavt-split-v1"


def bucket(key: str, salt: str = _SALT) -> float:
    """Same as mavt.data.splits.bucket (replicated: importing mavt runs mavt/__init__, which builds the
    torch model). tests/mavt4d/test_fixed_suite.py checks the two agree."""
    return int(hashlib.sha1(f"{salt}\0{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


@dataclass(frozen=True, slots=True)
class Ann:
    instance: str
    category: str
    xyz: tuple[float, float, float]
    size: tuple[float, float, float]  # (w, l, h)
    rotation: tuple[float, float, float, float]  # wxyz
    vis: int  # visibility bin 1..4 (0 = missing)
    attrs: tuple[str, ...]
    lidar_pts: int


@dataclass(frozen=True, slots=True)
class Keyframe:
    sample_token: str
    timestamp_us: int
    ego_xyz: tuple[float, float, float]
    ego_rot: tuple[float, float, float, float]  # wxyz
    lidar_sd: str
    cam_sd: dict[str, str]
    anns: tuple[Ann, ...]


@dataclass(frozen=True, slots=True)
class Scene:
    name: str
    token: str
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True)
class Cand:
    category: str
    scene: Scene
    start: int
    instances: tuple[str, ...]
    criteria: dict = field(hash=False)
    pair_id: str | None = None


# --- geometry -------------------------------------------------------------------------------------------

def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pose_matrix(translation, rotation) -> np.ndarray:
    """(4, 4) float64 transform from translation (3,) and quaternion wxyz."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_rot(rotation), np.asarray(translation, dtype=np.float64)
    return T


def box_corners(center, size, rotation) -> np.ndarray:
    """(8, 3) corners in the devkit order: 0-3 front face (+l/2 along box x = heading), 4-7 back face.
    ``size`` is (w, l, h): w along box y, l along box x, h along box z."""
    w, l, h = size
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return np.stack([x, y, z], 1) @ quat_to_rot(rotation).T + np.asarray(center, dtype=np.float64)


def _wrap_deg(a: float) -> float:
    return abs((a + 180.0) % 360.0 - 180.0)


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


# --- per-window measurements ----------------------------------------------------------------------------

def window_starts(n: int, k: int, stride: int) -> list[int]:
    return list(range(0, n - k + 1, stride))


def rare_classes(train_counts: dict[str, int], max_share: float = THRESHOLDS[RARE]["max_train_share"]) -> dict:
    total = sum(train_counts.values())
    return {c: {"train_instances": n, "train_share": n / total}
            for c, n in sorted(train_counts.items()) if total and n / total < max_share}


def _dist(a: Ann, kf: Keyframe) -> float:
    return math.hypot(a.xyz[0] - kf.ego_xyz[0], a.xyz[1] - kf.ego_xyz[1])


@dataclass(frozen=True)
class _Track:
    instance: str
    category: str
    frames: tuple[int, ...]
    anns: tuple[Ann, ...]
    dists: tuple[float, ...]
    disp: float
    moving_frac: float
    moving: bool
    static: bool

    @property
    def near(self) -> float:
        return min(self.dists)


def _tracks(frames: tuple[Keyframe, ...]) -> list[_Track]:
    by = defaultdict(list)
    for t, kf in enumerate(frames):
        for a in kf.anns:
            by[a.instance].append((t, a, _dist(a, kf)))
    m = THRESHOLDS[FIXED]
    out = []
    for inst, seq in sorted(by.items()):
        first, last = seq[0][1], seq[-1][1]
        disp = math.hypot(last.xyz[0] - first.xyz[0], last.xyz[1] - first.xyz[1])
        frac = sum(any(x in MOVING_ATTRS for x in a.attrs) for _, a, _ in seq) / len(seq)
        strict = any(x in STRICT_MOVING for _, a, _ in seq for x in a.attrs)
        out.append(_Track(inst, first.category, tuple(t for t, _, _ in seq), tuple(a for _, a, _ in seq),
                          tuple(d for _, _, d in seq), disp, frac,
                          moving=len(seq) >= 2 and frac >= m["moving_attr_frac"] and disp >= m["moving_min_disp_m"],
                          static=not strict and disp <= m["static_max_disp_m"]))
    return out


def _ego(frames: tuple[Keyframe, ...]) -> dict:
    p = np.array([kf.ego_xyz[:2] for kf in frames])
    yaw0 = math.degrees(yaw_of(frames[0].ego_rot))
    return {"ego_disp_m": _r(np.linalg.norm(p[-1] - p[0])),
            "ego_max_dev_m": _r(np.linalg.norm(p - p[0], axis=1).max()),
            "ego_max_dyaw_deg": _r(max(_wrap_deg(math.degrees(yaw_of(kf.ego_rot)) - yaw0) for kf in frames))}


def _occlusion(tr: _Track) -> list[int] | None:
    """[i, j, k]: visible (bin >= 3), then occluded (bin 1 or 0 lidar points), then visible again."""
    th = THRESHOLDS[OCCL]
    vis = [a.vis >= th["visible_min_bin"] for a in tr.anns]
    occ = [a.vis == th["occluded_bin"] or a.lidar_pts <= th["occluded_lidar_pts"] for a in tr.anns]
    for j in range(1, len(tr.anns) - 1):
        if not occ[j]:
            continue
        before = [i for i in range(j) if vis[i] and not occ[i]]
        after = [k for k in range(j + 1, len(tr.anns)) if vis[k] and not occ[k]]
        if before and after:
            i, k = before[-1], after[0]
            if max(tr.dists[i], tr.dists[j], tr.dists[k]) <= th["near_m"]:
                return [i, j, k]
    return None


def _event(tr: _Track) -> dict | None:
    """First state change (moving <-> still) between consecutive annotations of one instance."""
    prev = None
    for idx, a in enumerate(tr.anns):
        st = next((EVENT_STATES[x] + (x,) for x in a.attrs if x in EVENT_STATES), None)
        if st and prev and st[0] == prev[0] and st[1] != prev[1] and tr.frames[idx] == tr.frames[idx - 1] + 1:
            return {"instance": tr.instance, "frame": tr.frames[idx], "from": prev[2], "to": st[2],
                    "dist_m": _r(tr.dists[idx])}
        prev = st if st else prev
    return None


def _moving_info(tr: _Track) -> dict:
    return {"instance": tr.instance, "category": tr.category, "disp_m": _r(tr.disp),
            "moving_frac": _r(tr.moving_frac), "min_dist_m": _r(tr.near)}


def classify_window(scene: Scene, start: int, k: int, rare: dict) -> dict[str, tuple[list[str], dict]]:
    """Window categories it satisfies -> (key instance tokens, measured criteria)."""
    frames = scene.keyframes[start:start + k]
    tracks = _tracks(frames)
    ego = _ego(frames)
    out: dict[str, tuple[list[str], dict]] = {}
    by_near = sorted(tracks, key=lambda t: (t.near, t.instance))

    th = THRESHOLDS[STATIC]
    tracked = [t for t in by_near if len(t.anns) >= 2]
    if (ego["ego_disp_m"] >= th["ego_min_disp_m"] and len(tracked) >= th["min_static_instances"]
            and all(t.static for t in tracked)):
        out[STATIC] = ([t.instance for t in tracked], {**ego, "n_static": len(tracked),
                                                      "max_static_disp_m": _r(max(t.disp for t in tracked))})

    th = THRESHOLDS[FIXED]
    movers = [t for t in by_near if t.moving and t.near <= th["near_m"]]
    if (ego["ego_max_dev_m"] <= th["ego_max_dev_m"] and ego["ego_max_dyaw_deg"] <= th["ego_max_dyaw_deg"]
            and len(movers) >= th["min_moving"]):
        out[FIXED] = ([t.instance for t in movers], {**ego, "moving": [_moving_info(t) for t in movers]})

    th = THRESHOLDS[BOTH]
    movers = [t for t in by_near if t.moving and t.near <= th["near_m"]]
    if ego["ego_disp_m"] >= th["ego_min_disp_m"] and len(movers) >= th["min_moving"]:
        out[BOTH] = ([t.instance for t in movers], {**ego, "moving": [_moving_info(t) for t in movers]})

    occ = []
    for t in by_near:
        ijk = _occlusion(t)
        if ijk:
            occ.append((t, {"instance": t.instance, "category": t.category,
                            "frames": [t.frames[x] for x in ijk], "vis": [a.vis for a in t.anns],
                            "lidar_pts": [a.lidar_pts for a in t.anns], "annotated_frames": list(t.frames),
                            "max_dist_m": _r(max(t.dists[x] for x in ijk))}))
    if occ:
        out[OCCL] = ([t.instance for t, _ in occ], {**ego, "instances": [c for _, c in occ]})

    events = [(t, e) for t in by_near if t.near <= THRESHOLDS[EVENT]["near_m"] and (e := _event(t))]
    if events:
        out[EVENT] = ([t.instance for t, _ in events], {**ego, "instances": [e for _, e in events]})

    rares = [t for t in by_near if t.category in rare and t.near <= THRESHOLDS[RARE]["near_m"]]
    if rares:
        out[RARE] = ([t.instance for t in rares], {**ego, "instances": [
            {"instance": t.instance, "category": t.category, "train_share": _r(rare[t.category]["train_share"], 5),
             "min_dist_m": _r(t.near)} for t in rares]})
    return out


# --- pairs: same final position, different motion direction ---------------------------------------------

@dataclass(frozen=True)
class _PairCand:
    scene: Scene
    start: int
    instance: str
    final_xy: tuple[float, float]  # in the window's last-keyframe ego frame (x forward, y left)
    heading_deg: float  # displacement direction in that frame
    disp: float
    final_dist: float


def pair_candidates(scene: Scene, start: int, k: int) -> list[_PairCand]:
    frames = scene.keyframes[start:start + k]
    last = frames[-1]
    yaw = yaw_of(last.ego_rot)
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for t in _tracks(frames):
        if not t.moving or t.frames[-1] != k - 1 or t.dists[-1] > THRESHOLDS[PAIR]["near_m"]:
            continue
        a0, a1 = t.anns[0], t.anns[-1]
        rx, ry = a1.xyz[0] - last.ego_xyz[0], a1.xyz[1] - last.ego_xyz[1]
        dx, dy = a1.xyz[0] - a0.xyz[0], a1.xyz[1] - a0.xyz[1]
        out.append(_PairCand(scene, start, t.instance, (c * rx + s * ry, -s * rx + c * ry),
                             math.degrees(math.atan2(-s * dx + c * dy, c * dx + s * dy)), t.disp, t.dists[-1]))
    return out


def _row_id(category: str, scene: Scene, start: int) -> str:
    return f"{category}:{scene.name}:{start:02d}"


def select_pairs(cands: list[_PairCand], per_category: int, k: int, per_scene: int = PER_SCENE) -> list[Cand]:
    """Pairs of windows from different scenes; quota counts rows (2 per pair); each window used once."""
    th = THRESHOLDS[PAIR]
    cell = th["max_final_dist_m"]
    grid = defaultdict(list)
    for i, a in enumerate(cands):
        grid[(math.floor(a.final_xy[0] / cell), math.floor(a.final_xy[1] / cell))].append(i)
    pairs = []
    for (gx, gy), idx in grid.items():
        for i in idx:
            a = cands[i]
            for j in (j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in grid.get((gx + dx, gy + dy), ())):
                b = cands[j]
                if j <= i or a.scene.name == b.scene.name:
                    continue
                d = math.dist(a.final_xy, b.final_xy)
                dh = _wrap_deg(a.heading_deg - b.heading_deg)
                if d <= th["max_final_dist_m"] and dh >= th["min_heading_diff_deg"]:
                    pairs.append((a, b, d, dh) if (a.scene.name, a.start) < (b.scene.name, b.start) else (b, a, d, dh))

    def key(p):
        a, b = p[0], p[1]
        sa, sb = a.scene.keyframes[a.start].sample_token, b.scene.keyframes[b.start].sample_token
        return bucket(f"{PAIR}:{a.scene.name}:{sa}:{a.instance}|{b.scene.name}:{sb}:{b.instance}"), \
            a.scene.name, a.start, a.instance, b.scene.name, b.start, b.instance

    taken: dict[str, list[int]] = defaultdict(list)
    out: list[Cand] = []
    for a, b, d, dh in sorted(pairs, key=key):
        if len(out) + 2 > per_category:
            break
        if not all(_free(taken[x.scene.name], x.start, k, per_scene) for x in (a, b)):
            continue
        pid = f"{PAIR}:pair-{len(out) // 2:03d}"
        ids = [_row_id(PAIR, x.scene, x.start) for x in (a, b)]
        for x, partner in ((a, ids[1]), (b, ids[0])):
            taken[x.scene.name].append(x.start)
            out.append(Cand(PAIR, x.scene, x.start, (x.instance,), {
                "instance": x.instance, "final_ego_xy": [_r(v) for v in x.final_xy], "heading_deg": _r(x.heading_deg),
                "disp_m": _r(x.disp), "final_dist_m": _r(x.final_dist), "partner_id": partner,
                "final_pos_diff_m": _r(d), "heading_diff_deg": _r(dh)}, pid))
    return out


# --- selection and rows -----------------------------------------------------------------------------------

def _free(starts: list[int], start: int, k: int, per_scene: int) -> bool:
    return len(starts) < per_scene and all(abs(start - s) >= k for s in starts)


def select(cands: list[Cand], per_category: int, k: int, per_scene: int = PER_SCENE) -> list[Cand]:
    """Up to ``per_category`` windows ordered by bucket(category:scene:start_sample_token), at most
    ``per_scene`` per scene, no two overlapping windows of one scene."""
    def key(c: Cand):
        return bucket(f"{c.category}:{c.scene.name}:{c.scene.keyframes[c.start].sample_token}"), c.scene.name, c.start

    taken: dict[str, list[int]] = defaultdict(list)
    out = []
    for c in sorted(cands, key=key):
        if len(out) >= per_category:
            break
        if _free(taken[c.scene.name], c.start, k, per_scene):
            taken[c.scene.name].append(c.start)
            out.append(c)
    return out


def make_row(c: Cand, k: int, version: str) -> dict:
    frames = c.scene.keyframes[c.start:c.start + k]
    return {
        "suite": SUITE, "id": _row_id(c.category, c.scene, c.start), "category": c.category,
        "source": "nuscenes", "version": version, "split": "val", "scene_name": c.scene.name,
        "scene_token": c.scene.token, "fps": 2, "keyframe_only": True,
        "sample_tokens": [f.sample_token for f in frames], "timestamps_us": [f.timestamp_us for f in frames],
        "cameras": list(CAMERAS), "camera_sd_tokens": {cam: [f.cam_sd[cam] for f in frames] for cam in CAMERAS},
        "lidar_sd_tokens": [f.lidar_sd for f in frames], "instance_tokens": list(c.instances),
        "criteria": c.criteria, "thresholds": THRESHOLDS[c.category], "provenance": "annotation-heuristic",
        "mask": None, "pair_id": c.pair_id, "requires_train_exclusion": c.category == RARE,
        "notes": f"{_BASE_NOTE} {NOTES[c.category]}",
    }


def _mask(variant: str, k: int) -> dict:
    keep = {"lidar_dropped": CAMERAS, "half_cameras_dropped": HALF_CAMERAS_KEPT, "cam_front_only": ("CAM_FRONT",)}
    return {"lidar": [variant == "half_cameras_dropped"] * k,
            "cameras": {cam: [cam in keep[variant]] * k for cam in CAMERAS}}


def missing_modality_rows(rows: list[dict], per_category: int, k: int) -> list[dict]:
    """Every variant of the first ``bases_per_category`` selected clips of each other category, whole
    clips only, at most ``per_category`` rows."""
    n = THRESHOLDS[MISSING]["bases_per_category"]
    bases = [r for cat in CATEGORIES if cat != MISSING for r in [r for r in rows if r["category"] == cat][:n]]
    out = []
    for b in bases[:per_category // len(MISSING_VARIANTS)]:
        for v in MISSING_VARIANTS:
            out.append({**b, "id": f"{MISSING}:{v}:{b['id']}", "category": MISSING, "mask": _mask(v, k),
                        "criteria": {"base_id": b["id"], "base_category": b["category"], "variant": v},
                        "thresholds": THRESHOLDS[MISSING], "pair_id": None,
                        "notes": f"{_BASE_NOTE} {NOTES[MISSING]}"})
    return out


def mine(scenes: list[Scene], k: int, stride: int, per_category: int, rare: dict, version: str) -> tuple[list[dict], dict]:
    """All suite rows (CATEGORIES order) and candidate counts per category before selection."""
    by_cat: dict[str, list[Cand]] = defaultdict(list)
    pool: list[_PairCand] = []
    n_windows = 0
    for sc in scenes:
        for s in window_starts(len(sc.keyframes), k, stride):
            n_windows += 1
            for cat, (inst, crit) in classify_window(sc, s, k, rare).items():
                by_cat[cat].append(Cand(cat, sc, s, tuple(inst), crit))
            pool += pair_candidates(sc, s, k)
    chosen = {cat: select(by_cat[cat], per_category, k) for cat in WINDOW_CATEGORIES}
    chosen[PAIR] = select_pairs(pool, per_category, k)
    rows = [make_row(c, k, version) for cat in CATEGORIES if cat != MISSING for c in chosen[cat]]
    miss = missing_modality_rows(rows, per_category, k)
    rows = [r for cat in CATEGORIES for r in (miss if cat == MISSING else rows) if r["category"] == cat]
    stats = {"windows": n_windows, "candidates": {cat: len(by_cat[cat]) for cat in WINDOW_CATEGORIES},
             "pair_instance_candidates": len(pool)}
    return rows, stats
