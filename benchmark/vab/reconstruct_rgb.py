"""Offline sparse reconstruction of retained public RGB; imports no simulator."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from cascade.perception.rgb_triangulation import RgbView, TriangulationPolicy, match_rgb_features, triangulate_pairs


def read_json(path, maximum):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('RGB artifact must be a bounded regular file')
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError('RGB artifact exceeded read bound')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def reconstruct(directory, *, policy=TriangulationPolicy(), max_features=512, ratio=.7):
    directory = Path(directory)
    calibration, calibration_hash = read_json(directory/'calibration.json', 65536)
    calculated = hashlib.sha256(json.dumps(calibration['recipe'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if calculated != calibration['sha256']:
        raise ValueError('RGB calibration digest differs')
    recipe = calibration['recipe']
    if (recipe['schema'] != 'cascade.vab-public-rgb-calibration.v1'
            or recipe['pixel_center_offset_uv'] != [.5, .5]
            or set(recipe['cameras']) != {'agentview', 'robot0_eye_in_hand'}):
        raise ValueError('original VAB camera calibration recipe required')
    files = sorted(directory.glob('[0-9][0-9][0-9][0-9].json'))
    if not 1 <= len(files) <= 256 or [p.name for p in files] != [f'{i:04d}.json' for i in range(len(files))]:
        raise ValueError('complete bounded RGB archive required')
    artifacts = {str(directory/'calibration.json'): calibration_hash}
    results, binding, previous = [], None, None
    for index, path in enumerate(files):
        data, artifacts[str(path)] = read_json(path, 256*1024)
        if data['schema'] != 'cascade.vab-public-rgb-pair.v1' or data['physical_admission'] is not False:
            raise ValueError('unknown public RGB record')
        returned_step, returned_time = data['returned_solver_step'], data['returned_simulation_time_s']
        if (type(returned_step) is not int or not 0 <= returned_step < 2**53
                or type(returned_time) not in (float, int) or not math.isfinite(returned_time) or returned_time < 0):
            raise ValueError('invalid public-observation return clock')
        if data['status'] == 'unverified':
            if index != 0 or data['views'] != [] or data['reason'] != 'reset images precede render observer':
                raise ValueError('unexpected missing RGB render provenance')
            results.append({'file': path.name, 'status': 'unverified', 'reason': data['reason']})
            continue
        if data['status'] != 'observed' or len(data['views']) != 2:
            raise ValueError('two observed RGB views required')
        views = []
        for item in data['views']:
            if item['rgb_encoding'] != 'base64':
                raise ValueError('unknown RGB encoding')
            value = dict(item['view'])
            value['rgb'] = base64.b64decode(value['rgb'], validate=True)
            view = RgbView(**value)
            if view.sha256 != item['capture_sha256'] or view.calibration_sha256 != calculated:
                raise ValueError('RGB capture bytes or calibration identity differ')
            if view.model_identity_sha256 != calibration['recipe']['model_identity_sha256']:
                raise ValueError('RGB model differs from calibration')
            if view.camera_id not in recipe['cameras']:
                raise ValueError('RGB camera is absent from calibration')
            fovy = recipe['cameras'][view.camera_id]['fovy_deg']
            if type(fovy) not in (float, int) or not math.isfinite(fovy) or not 1 < fovy < 179:
                raise ValueError('invalid declared camera field of view')
            focal = 64/math.tan(math.radians(fovy)/2)
            if ((view.width, view.height) != (128, 128)
                    or view.intrinsics != (focal, 0., 64., 0., focal, 64., 0., 0., 1.)
                    or list(view.pixel_center_offset_uv) != recipe['pixel_center_offset_uv']
                    or view.world_frame_id != 'vab_world'):
                raise ValueError('RGB geometry differs from the declared camera recipe')
            if view.solver_step > returned_step or view.simulation_time_s > returned_time:
                raise ValueError('RGB capture is later than its public-observation return')
            views.append(view)
        left, right = views
        if [v.camera_id for v in views] != ['agentview', 'robot0_eye_in_hand']:
            raise ValueError('original VAB camera pair required')
        current = (left.model_identity_sha256, left.epoch, left.world_frame_id)
        if binding is not None and current != binding:
            raise ValueError('RGB archive changes model, epoch or world frame')
        if previous is not None and (left.solver_step <= previous[0] or left.simulation_time_s <= previous[1]):
            raise ValueError('RGB archive replays or rewinds capture clock')
        binding, previous = current, (left.solver_step, left.simulation_time_s)
        triangulate_pairs(left, right, [], policy=policy)  # Admission before optional image processing.
        pairs = match_rgb_features(left, right, max_features=max_features, ratio=ratio)
        result = triangulate_pairs(left, right, pairs, policy=policy)
        results.append({'file': path.name, 'status': 'analyzed', 'capture_step': left.solver_step,
                        'capture_time_s': left.simulation_time_s, 'reconstruction': result})
    if sorted(directory.glob('[0-9][0-9][0-9][0-9].json')) != files:
        raise ValueError('RGB archive membership changed during reconstruction')
    for path, expected in artifacts.items():
        maximum = 65536 if Path(path).name == 'calibration.json' else 256*1024
        if read_json(Path(path), maximum)[1] != expected:
            raise ValueError('RGB archive changed during reconstruction')
    return {'schema': 'cascade.vab-offline-rgb-reconstruction.v1', 'captures': results,
            'artifact_sha256': artifacts, 'feature_recipe': {'method': 'SIFT mutual ratio',
            'max_features': max_features, 'ratio': ratio}, 'models_constructed': 0,
            'physics_steps_advanced': 0, 'position_error_m': None, 'physical_admission': False,
            'object_identity_verified': False, 'complete_geometry_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new reconstruction output required')
    result = reconstruct(args.directory)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, allow_nan=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
