"""One robot, one world, one client process: launch its shared-owner world (robots=1), build its
CASCADE RobotRuntime, signal READY, wait for the fleet START file, walk its route, save walk.json.

Usage: fleet_one.py RUN_DIR ROBOT_ID INDEX DISTANCE GPU BASE_PORT POLICY MAX_STEPS MAX_WALL_S CAMERA_EVERY RESOLUTION ROW_EVERY
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_lib as S  # noqa: E402
import route_walk as R  # noqa: E402

sys.path.insert(0, str(S.REPO / 'src'))


def main():
    (run, robot_id, index, distance, gpu, base_port, policy, max_steps, max_wall_s,
     camera_every, resolution, row_every) = sys.argv[1:13]
    run, index, distance = Path(run), int(index), float(distance)
    out = run / robot_id
    out.mkdir(exist_ok=True)
    for key in tuple(os.environ):
        if key.startswith('CASCADE_'):
            os.environ.pop(key)
    for key in ('BELIEFS', 'GRASP_MEMORY', 'ENVELOPE', 'SPATIAL_MEMORY'):
        os.environ['CASCADE_' + key + '_PATH'] = str(out / 'stores' / f'{key.lower()}.json')
    rep = {'robot_id': robot_id, 'distance_m': distance, 'steps': [], 'physical_admission': False, 'gpu': gpu,
           'client_pid': os.getpid()}
    record = runtime = None
    try:
        record = S.launch_owner(out / 'owner', gpu, robots=1, spacing=1.0, layout='line', route_m=distance,
                                base_port=int(base_port), policy=policy, max_wall_s=float(max_wall_s),
                                max_steps=int(max_steps), camera_every=int(camera_every), resolution=resolution,
                                physics_row_every=int(row_every), gc_policy=None)
        marker = S.wait_listening(record, 900)
        rep['listening_after_s'] = round(time.monotonic() - record['launched_monotonic'], 1)
        rep['scene_model_sha256'] = marker['scene_model_sha256']
        (wire_id, entry), = marker['robots'].items()
        config_dir = out / 'configs'
        rep['model_identity_sha256'] = S.write_robot_config(record, wire_id, entry, policy, config_dir=config_dir)
        cfg, base = S.load_cfg(wire_id, entry['hello']['epoch'], config_dir=config_dir)
        S.save(out / 'resolved-config.json', cfg.as_dict())
        rep['passive_samples'] = len(S.passive_ready(base))
        from cascade.apps.robot_runtime import build_robot_runtime
        runtime, _ = build_robot_runtime(cfg, out / 'runtime')
        rep['before'] = R.state_row(runtime)
        (out / 'READY').write_text(json.dumps({'at': time.time()}))
        deadline = time.monotonic() + 1200
        while not (run / 'START').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('fleet start file never appeared')
            time.sleep(0.2)
        t0 = time.monotonic()
        rep['walk_started_wall'] = time.time()
        result = runtime.execute('locomotion.walk_distance', {'distance_m': distance})
        wall = time.monotonic() - t0
        rep['after'] = R.state_row(runtime)
        step = {'index': 0, 'tool': 'locomotion.walk_distance', 'args': {'distance_m': distance},
                'wall_s': round(wall, 2), 'confirmed': S.confirmed(result), **S.compact(result)}
        step['verifier_samples'] = R.samples_to_jsonl(result, out / 'verifier-samples.jsonl')
        rep['steps'].append(step)
        print(json.dumps({'duck': robot_id, **{k: step[k] for k in ('confirmed', 'post_status', 'measured_distance_m',
                                                                    'wall_s', 'error', 'post_reason')}}, default=str), flush=True)
    except BaseException:
        rep['error'] = traceback.format_exc()
        print(robot_id, rep['error'][-1200:], flush=True)
    finally:
        if runtime is not None:
            try:
                rep['runtime_close'] = runtime.close()
            except BaseException as exc:
                rep['runtime_close_error'] = repr(exc)
        if record is not None:
            time.sleep(6)  # a few seconds of post-walk physics in the recording
            rep['termination'] = S.terminate(record)
        S.save(out / 'walk.json', rep)


if __name__ == '__main__':
    import cascade
    if not Path(cascade.__file__).resolve().is_relative_to(S.REPO / 'src'):
        raise RuntimeError(f'foreign CASCADE imported: {cascade.__file__}')
    main()
