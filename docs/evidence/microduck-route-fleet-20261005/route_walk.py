"""Route gate: N MicroDucks in one shared world walk long independent routes.

Diagnostic only, no LLM: CASCADE RobotRuntime -> SafeBase walk_distance -> the
shared owner's per-robot endpoint, judged by the independent verifier under the
route profile. Never resets or retries inside an episode; continues to the next
robot-independent step only through each robot's own runtime.

Usage: route_walk.py RUN_NAME --robots N --policy PROFILE --distance D
                     [--distances d0,d1,...] [--gpu gpu1] [--base-port 18100]
                     [--spacing 1.0] [--max-steps 14000] [--max-wall-s 1500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_lib as S  # noqa: E402

sys.path.insert(0, str(S.REPO / 'src'))


def state_row(runtime):
    r = runtime.execute('locomotion.get_base_state', {})
    s = (r.get('state') or r.get('result') or r) if isinstance(r, dict) else {}
    return {k: s.get(k) for k in ('sim_time_s', 'position_world', 'orientation_wxyz', 'controller_status',
                                   'fallen', 'linear_velocity_world')}


def samples_to_jsonl(result, path):
    rows = (result.get('measured') or {}).get('samples') or []
    with open(path, 'w') as f:
        for s in rows:
            f.write(json.dumps({k: s.get(k) for k in ('sim_time_s', 'wall_time_s', 'position_world', 'orientation_wxyz',
                                                       'linear_velocity_world', 'angular_velocity_body',
                                                       'controller_status', 'fallen', 'latched')}) + '\n')
    return len(rows)


def run_robot(robot_id, entry, record, run, policy, distance, start_barrier, report, done=None):
    out = run / robot_id
    out.mkdir()
    entry_report = {'robot_id': robot_id, 'distance_m': distance, 'steps': [], 'physical_admission': False}
    report[robot_id] = entry_report
    runtime = None
    try:
        entry_report['model_identity_sha256'] = S.write_robot_config(record, robot_id, entry, policy)
        cfg, base = S.load_cfg(robot_id, entry['hello']['epoch'])
        S.save(out / 'resolved-config.json', cfg.as_dict())
        entry_report['passive_samples'] = len(S.passive_ready(base))
        from cascade.apps.robot_runtime import build_robot_runtime
        runtime, _ = build_robot_runtime(cfg, out / 'runtime')
        entry_report['before'] = state_row(runtime)
        if hasattr(start_barrier, 'wait') and not hasattr(start_barrier, 'parties'):
            start_barrier.wait(timeout=3600)  # wave event: set by the driver when this wave may start
        else:
            start_barrier.wait(timeout=300)
        t0 = time.monotonic()
        result = runtime.execute('locomotion.walk_distance', {'distance_m': distance})
        wall = time.monotonic() - t0
        entry_report['after'] = state_row(runtime)
        step = {'index': 0, 'tool': 'locomotion.walk_distance', 'args': {'distance_m': distance},
                'wall_s': round(wall, 2), 'confirmed': S.confirmed(result), **S.compact(result)}
        step['verifier_samples'] = samples_to_jsonl(result, out / 'verifier-samples.jsonl')
        entry_report['steps'].append(step)
        print(json.dumps({'duck': robot_id, **{k: step[k] for k in ('confirmed', 'post_status', 'measured_distance_m',
                                                                    'wall_s', 'error', 'post_reason')}}, default=str), flush=True)
    except BaseException:
        entry_report['error'] = traceback.format_exc()
        print(robot_id, entry_report['error'][-1500:], flush=True)
    finally:
        if done is not None:
            done.set()
        if runtime is not None:
            try:
                entry_report['runtime_close'] = runtime.close()
            except BaseException as exc:
                entry_report['runtime_close_error'] = repr(exc)
        S.save(out / 'walk.json', entry_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_name')
    ap.add_argument('--robots', type=int, required=True)
    ap.add_argument('--policy', choices=sorted(S.POLICIES), required=True)
    ap.add_argument('--distance', type=float, default=5.0)
    ap.add_argument('--distances', default=None, help='comma list overriding --distance per robot')
    ap.add_argument('--gpu', default='gpu1')
    ap.add_argument('--base-port', type=int, default=18100)
    ap.add_argument('--spacing', type=float, default=1.0)
    ap.add_argument('--layout', default='line')
    ap.add_argument('--route-m', type=float, default=5.0)
    ap.add_argument('--max-steps', type=int, default=14000)
    ap.add_argument('--max-wall-s', type=float, default=1500.)
    ap.add_argument('--camera-every', type=int, default=20)
    ap.add_argument('--resolution', default='1920x1080')
    ap.add_argument('--physics-row-every', type=int, default=10)
    ap.add_argument('--gc-policy', default='freeze-startup-heap')
    ap.add_argument('--listen-timeout', type=float, default=900.)
    ap.add_argument('--profile-phases', action='store_true')
    ap.add_argument('--waves', type=int, default=1, help='command the robots in this many sequential groups')
    args = ap.parse_args()
    run = HERE / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    S.ensure_configs()
    for key in tuple(os.environ):
        if key.startswith('CASCADE_'):
            os.environ.pop(key)
    for key in ('BELIEFS', 'GRASP_MEMORY', 'ENVELOPE', 'SPATIAL_MEMORY'):
        os.environ['CASCADE_' + key + '_PATH'] = str(run / 'stores' / f'{key.lower()}.json')
    distances = ([float(x) for x in args.distances.split(',')] if args.distances
                 else [args.distance] * args.robots)
    assert len(distances) == args.robots
    experiment = {'repo': str(S.REPO), 'bundle': str(S.BUNDLE), 'bundle_sha256': S.BUNDLE_SHA,
                  'policy_profile': args.policy, 'policy': str(S.POLICIES[args.policy][0]),
                  'policy_sha256': S.POLICIES[args.policy][1], 'base_profile': S.BASE_PROFILE,
                  'limits': json.loads(S.LIMITS.read_text()), 'distances': distances, 'args': vars(args),
                  'scope': 'shared-world route diagnostic, RobotRuntime, no LLM; not physical admission'}
    S.save(run / 'experiment.json', experiment)
    report = {}
    record = None
    try:
        record = S.launch_owner(run / 'owner', args.gpu, robots=args.robots, spacing=args.spacing,
                                layout=args.layout, route_m=args.route_m, base_port=args.base_port,
                                policy=args.policy, max_wall_s=args.max_wall_s, max_steps=args.max_steps,
                                camera_every=args.camera_every, resolution=args.resolution,
                                physics_row_every=args.physics_row_every, gc_policy=args.gc_policy or None,
                                profile_phases=args.profile_phases)
        t_launch = time.monotonic()
        marker = S.wait_listening(record, args.listen_timeout)
        experiment['listening_after_s'] = round(time.monotonic() - t_launch, 1)
        experiment['scene_model_sha256'] = marker['scene_model_sha256']
        S.save(run / 'experiment.json', experiment)
        robots = sorted(marker['robots'])
        assert len(robots) == args.robots, robots
        waves = max(1, args.waves)
        wave_of = {robot_id: i * waves // args.robots for i, robot_id in enumerate(robots)}
        wave_events = [threading.Event() for _ in range(waves)]
        done_events = {robot_id: threading.Event() for robot_id in robots}
        experiment['waves'] = {str(w): [r for r in robots if wave_of[r] == w] for w in range(waves)}
        S.save(run / 'experiment.json', experiment)
        threads = []
        for robot_id, distance in zip(robots, distances):
            t = threading.Thread(target=run_robot, args=(robot_id, marker['robots'][robot_id], record, run,
                                                         args.policy, distance, wave_events[wave_of[robot_id]],
                                                         report, done_events[robot_id]), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.5)
        time.sleep(5)  # every runtime built and passively ready before the first wave
        for w in range(waves):
            experiment.setdefault('wave_started_wall', {})[str(w)] = time.time()
            wave_events[w].set()
            for robot_id in experiment['waves'][str(w)]:
                done_events[robot_id].wait(timeout=1500)
        for t in threads:
            t.join()
    except BaseException:
        experiment['driver_error'] = traceback.format_exc()
        print(experiment['driver_error'][-2000:], flush=True)
    finally:
        if record is not None:
            # Let the owner publish a few more seconds of post-walk physics, then close it cleanly.
            time.sleep(5)
            experiment['termination'] = S.terminate(record)
        summary = []
        for robot_id in sorted(report):
            r = report[robot_id]
            for step in r['steps']:
                summary.append({'duck': robot_id, 'distance_m': r['distance_m'], 'confirmed': step['confirmed'],
                                'post_status': step['post_status'], 'post_reason': step['post_reason'],
                                'measured_distance_m': step['measured_distance_m'], 'post_metrics': step['post_metrics'],
                                'wall_s': step['wall_s'], 'sim_s': (None if step['after_sim_t'] is None or step['before_sim_t'] is None
                                                                      else round(step['after_sim_t'] - step['before_sim_t'], 2)),
                                'error': step['error'], 'before_pos': step['before_pos'], 'after_pos': step['after_pos']})
            if not r['steps']:
                summary.append({'duck': robot_id, 'distance_m': r['distance_m'], 'confirmed': False,
                                'error': (r.get('error') or '')[-400:]})
        experiment['summary'] = summary
        S.save(run / 'experiment.json', experiment)
        print(json.dumps(summary, indent=1, default=str), flush=True)


if __name__ == '__main__':
    import cascade
    if not Path(cascade.__file__).resolve().is_relative_to(S.REPO / 'src'):
        raise RuntimeError(f'foreign CASCADE imported: {cascade.__file__}')
    main()
