#!/usr/bin/env python3
"""Count annotated instances in the actual input frustum before choosing slot tier.

Only official train scenes minus the v5 log-group dev split are counted. The
nuScenes devkit's BoxVisibility.ANY convention is pinned explicitly; occlusion
and unannotated background are not measured by this census.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")


def visible_instances(tables, sample, cameras):
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
    from pyquaternion import Quaternion
    result = set()
    for camera in cameras:
        sensor = tables.keyframe_data[sample][camera]
        record = tables.sample_data[sensor]
        world_to_camera = np.linalg.inv(tables.ego_to_global(sensor) @ tables.sensor_to_ego(sensor))
        rotation = Quaternion(matrix=world_to_camera[:3, :3])
        intrinsic = tables.raw_intrinsics(sensor)
        for annotation in tables.annotations_of[sample]:
            row = tables.sample_annotation[annotation]
            box = Box(row['translation'], row['size'], Quaternion(row['rotation']))
            box.rotate(rotation)
            box.translate(world_to_camera[:3, 3])
            if box_in_image(box, intrinsic, (record['width'], record['height']), BoxVisibility.ANY):
                result.add(row['instance_token'])
    return result


def iter_json_array(stream, *, chunk_size=1024 * 1024):
    """Incrementally decode a table; memory is bounded by a chunk plus one row."""
    decoder = json.JSONDecoder()
    buffer, pos, started, finished, expecting_value = "", 0, False, False, True
    while True:
        chunk = stream.read(chunk_size)
        eof = not chunk
        buffer, pos = buffer[pos:] + chunk, 0
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos == len(buffer):
                break
            if not started:
                if buffer[pos] != '[':
                    raise ValueError('JSON table must be an array')
                pos, started = pos + 1, True
                continue
            if finished:
                raise ValueError('trailing JSON table data')
            if buffer[pos] == ']':
                pos, finished = pos + 1, True
                continue
            if not expecting_value:
                if buffer[pos] != ',':
                    raise ValueError('missing JSON array separator')
                pos, expecting_value = pos + 1, True
                continue
            try:
                row, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError('incomplete JSON table') from error
                break
            yield row
            pos, expecting_value = end, False
        if eof:
            if not started or not finished:
                raise ValueError('incomplete JSON table')
            return


def _rows(metadata, name):
    with (metadata / 'v1.0-trainval' / (name + '.json')).open() as stream:
        yield from iter_json_array(stream)


def summarize_camera_counts(counts):
    """One observed camera per draw: flatten sample×camera, never union objects."""
    def summarize(values):
        histogram = Counter(values)
        total = sum(histogram.values())
        return {"total_steps": total, "steps_over_16": sum(n for k, n in histogram.items() if k > 16),
                "steps_over_32": sum(n for k, n in histogram.items() if k > 32),
                "histogram": dict(sorted(histogram.items()))}
    result = summarize([count for values in counts.values() for count in values])
    if not result["total_steps"]:
        raise ValueError("no eligible training camera-keyframes in slot census")
    result["fraction_over_16"] = result["steps_over_16"] / result["total_steps"]
    result["per_camera"] = {camera: summarize(values) for camera, values in counts.items()}
    return result


def census(root, *, cameras=CAMERAS):
    from mavt.data.splits import bucket
    from mavt4d.data.nuscenes import _se3, load_official_split
    from mavt4d.object_state.v5.natural import nuscenes_metadata_fingerprint
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
    from pyquaternion import Quaternion
    root = Path(root)
    metadata = root if (root / 'v1.0-trainval').is_dir() else root / 'metadata'
    official = set(load_official_split('v1.0-trainval', 'train'))
    eligible = {r['token'] for r in _rows(metadata, 'scene') if r['name'] in official
                and bucket(r['log_token'], 'mavt-v5-nusc-log-dev-v1') >= .10}
    groups = defaultdict(list)
    for row in _rows(metadata, 'sample'):
        if row['scene_token'] in eligible:
            groups[row['scene_token']].append((row['timestamp'], row['token']))
    groups = {g: [token for _, token in sorted(rows)] for g, rows in groups.items() if len(rows) >= 4}
    samples = {token for group in groups.values() for token in group}
    sensors = {r['token']: r['channel'] for r in _rows(metadata, 'sensor') if r['channel'] in cameras}
    calibrated = {r['token']: r for r in _rows(metadata, 'calibrated_sensor') if r['sensor_token'] in sensors}
    # The full sample_data/ego_pose/annotation tables dominate RAM in the generic
    # reader. Retain only camera-keyframe records, then stream annotations once.
    keyframes = defaultdict(list)
    ego_ids = set()
    for row in _rows(metadata, 'sample_data'):
        if row['is_key_frame'] and row['sample_token'] in samples and row['calibrated_sensor_token'] in calibrated:
            keyframes[row['sample_token']].append({k: row[k] for k in
                ('ego_pose_token', 'calibrated_sensor_token', 'width', 'height')})
            ego_ids.add(row['ego_pose_token'])
    ego = {r['token']: _se3(r['rotation'], r['translation']) for r in _rows(metadata, 'ego_pose') if r['token'] in ego_ids}
    camera_geometry = defaultdict(list)
    for sample, rows in keyframes.items():
        for row in rows:
            calibration = calibrated[row['calibrated_sensor_token']]
            transform = np.linalg.inv(ego[row['ego_pose_token']] @ _se3(calibration['rotation'], calibration['translation']))
            camera_geometry[sample].append((sensors[calibration['sensor_token']], Quaternion(matrix=transform[:3, :3]), transform[:3, 3],
                np.asarray(calibration['camera_intrinsic']), (row['width'], row['height'])))
    del ego, keyframes
    identities = defaultdict(set)
    for row in _rows(metadata, 'sample_annotation'):
        sample = row['sample_token']
        if sample not in camera_geometry:
            continue
        for camera, rotation, translation, intrinsic, size in camera_geometry[sample]:
            box = Box(row['translation'], row['size'], Quaternion(row['rotation']))
            box.rotate(rotation)
            box.translate(translation)
            if box_in_image(box, intrinsic, size, BoxVisibility.ANY):
                identities[(sample, camera)].add(row['instance_token'])
    counts, digest = {camera: [] for camera in cameras}, hashlib.sha256()
    for group, sample_tokens in sorted(groups.items()):
        for sample in sample_tokens:
            available = {camera for camera, *_ in camera_geometry[sample]}
            if available != set(cameras):
                raise ValueError('census sample lacks a configured input camera')
            for camera in cameras:
                ids = sorted(identities[(sample, camera)])
                counts[camera].append(len(ids))
                digest.update(json.dumps([group, sample, camera, ids], separators=(',', ':')).encode())
    summary = summarize_camera_counts(counts)
    return {'schema': 'v5-slot-census-v2', 'source': 'nuscenes', 'split': 'train',
            'groups': len(groups), 'sample_steps': sum(map(len, groups.values())),
            'input_views': list(cameras), 'input_distribution': 'uniform_single_camera', **summary,
            'annotation': 'gt_box_in_image_any',
            'metadata_sha256': nuscenes_metadata_fingerprint(metadata, 'v1.0-trainval', official),
            'instance_census_sha256': digest.hexdigest(),
            'recommended_tier': {'slots': 32, 'content_tokens': 4, 'dynamic_tokens': 4} if summary['fraction_over_16'] > .2
                                else {'slots': 16, 'content_tokens': 8, 'dynamic_tokens': 8},
            'unmeasured': ['occlusion', 'unannotated_objects', 'background_regions'],
            'unmeasured_sources': ['openimages', 'openvid', 'dl3dv', 'arkitscenes']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', action='append', dest='cameras')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('versioned census output already exists')
    result = census(args.root, cameras=args.cameras or CAMERAS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({'groups': result['groups'], 'steps': result['total_steps'],
                      'fraction_over_16': result['fraction_over_16'], 'recommended_tier': result['recommended_tier']}))


if __name__ == '__main__':
    main()
