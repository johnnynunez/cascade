"""Fleet route gate: N MicroDucks, N shared-owner worlds (one robot each), independent long routes.

Why N worlds: with twelve robots in ONE world the owner steps 5 ms of physics in ~100 ms of
wall and twelve closed-loop clients saturate its RPC threads, so the unchanged 0.5 s state-age /
RPC and 0.4 s progress limits trip at ~0.8-2.6 m (runs g2d/g2e). One robot per world steps in
~11 ms and a 5 m walk is confirmed (g1). Every robot still has its own RobotRuntime, verifier,
policy and command; no limit is relaxed. Not physical admission.

Usage: fleet_walk.py RUN_NAME --robots 12 --policy rough_walk_e --distances 5,4.5,... [--gpus gpu0,gpu1]
"""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument('run_name')
    ap.add_argument('--robots', type=int, default=12)
    ap.add_argument('--policy', choices=sorted(S.POLICIES), default='rough_walk_e')
    ap.add_argument('--distances', default='5,4.5,5.5,4,5,5.5,3.5,5,4.5,5.5,5,4')
    ap.add_argument('--gpus', default='gpu0,gpu1')
    ap.add_argument('--base-port', type=int, default=18300)
    ap.add_argument('--max-steps', type=int, default=40000)
    ap.add_argument('--max-wall-s', type=float, default=1500.)
    ap.add_argument('--camera-every', type=int, default=20)
    ap.add_argument('--resolution', default='1280x720')
    ap.add_argument('--physics-row-every', type=int, default=4)
    ap.add_argument('--listen-timeout', type=float, default=900.)
    ap.add_argument('--stagger-s', type=float, default=3.)
    args = ap.parse_args()
    run = HERE / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    S.ensure_configs()
    for key in tuple(os.environ):
        if key.startswith('CASCADE_'):
            os.environ.pop(key)
    for key in ('BELIEFS', 'GRASP_MEMORY', 'ENVELOPE', 'SPATIAL_MEMORY'):
        os.environ['CASCADE_' + key + '_PATH'] = str(run / 'stores' / f'{key.lower()}.json')
    distances = [float(x) for x in args.distances.split(',')][:args.robots]
    assert len(distances) == args.robots
    gpus = args.gpus.split(',')
    experiment = {'repo': str(S.REPO), 'bundle': str(S.BUNDLE), 'bundle_sha256': S.BUNDLE_SHA,
                  'policy_profile': args.policy, 'policy': str(S.POLICIES[args.policy][0]),
                  'policy_sha256': S.POLICIES[args.policy][1], 'base_profile': S.BASE_PROFILE,
                  'limits': json.loads(S.LIMITS.read_text()), 'distances': distances, 'args': vars(args),
                  'scope': 'fleet route diagnostic: one shared-owner world per robot, RobotRuntime, no LLM; not physical admission',
                  'robot_ids': [f'duck{i:02d}' for i in range(args.robots)]}
    S.save(run / 'experiment.json', experiment)
    import subprocess
    env = dict(os.environ)
    env['PYTHONPATH'] = str(S.REPO / 'src')
    procs, logs = [], []
    for i in range(args.robots):
        robot_id = f'duck{i:02d}'
        (run / robot_id).mkdir(exist_ok=True)
        log = open(run / robot_id / 'client.log', 'w')
        argv = [sys.executable, str(HERE / 'fleet_one.py'), str(run), robot_id, str(i), str(distances[i]),
                gpus[i % len(gpus)], str(args.base_port + i), args.policy, str(args.max_steps), str(args.max_wall_s),
                str(args.camera_every), args.resolution, str(args.physics_row_every)]
        procs.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env))
        logs.append(log)
        time.sleep(args.stagger_s)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        ready = [(run / f'duck{i:02d}' / 'READY').exists() for i in range(args.robots)]
        alive = [p.poll() is None for p in procs]
        if all(r or not a for r, a in zip(ready, alive)):
            break
        time.sleep(1)
    experiment['ready'] = [(run / f'duck{i:02d}' / 'READY').exists() for i in range(args.robots)]
    experiment['walks_released_wall'] = time.time()
    (run / 'START').write_text(json.dumps({'at': time.time()}))
    for p, log in zip(procs, logs):
        p.wait()
        log.close()
    report = {}
    for i in range(args.robots):
        w = run / f'duck{i:02d}' / 'walk.json'
        if w.exists():
            report[f'duck{i:02d}'] = json.load(open(w))
        else:
            report[f'duck{i:02d}'] = {'robot_id': f'duck{i:02d}', 'distance_m': distances[i], 'steps': [],
                                      'error': 'client process left no walk.json', 'gpu': gpus[i % len(gpus)]}
    summary = []
    for robot_id in sorted(report):
        r = report[robot_id]
        for step in r['steps']:
            summary.append({'duck': robot_id, 'distance_m': r['distance_m'], 'confirmed': step['confirmed'],
                            'post_status': step['post_status'], 'post_reason': step['post_reason'],
                            'measured_distance_m': step['measured_distance_m'],
                            'metrics': (step['post_metrics'] or {}).get('metrics'),
                            'wall_s': step['wall_s'], 'error': step['error'],
                            'before_pos': step['before_pos'], 'after_pos': step['after_pos'], 'gpu': r['gpu']})
        if not r['steps']:
            summary.append({'duck': robot_id, 'distance_m': r['distance_m'], 'confirmed': False,
                            'error': (r.get('error') or '')[-400:], 'gpu': r['gpu']})
    experiment['summary'] = summary
    S.save(run / 'experiment.json', experiment)
    print(json.dumps(summary, indent=1, default=str), flush=True)


if __name__ == '__main__':
    import cascade
    if not Path(cascade.__file__).resolve().is_relative_to(S.REPO / 'src'):
        raise RuntimeError(f'foreign CASCADE imported: {cascade.__file__}')
    main()
