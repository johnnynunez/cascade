"""Keynote showcase run: one shared owner world, N MicroDucks following a presenter proxy (choreography).

No clients, no command admission: the owner evaluates each robot's scripted twist in-process.
Usage: showcase.py RUN_NAME --robots 12 --choreography keynote_12.json [--gpu gpu1] [--sim-s 70]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent / 'HERMES_MICRODUCK_ROUTE_20261005'
sys.path.insert(0, str(ROUTE))
import shared_lib as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_name')
    ap.add_argument('--robots', type=int, default=12)
    ap.add_argument('--choreography', default=str(HERE / 'keynote_12.json'))
    ap.add_argument('--policy', choices=sorted(S.POLICIES), default='rough_walk_e')
    ap.add_argument('--gpu', default='gpu1')
    ap.add_argument('--sim-s', type=float, default=70.)
    ap.add_argument('--max-wall-s', type=float, default=3300.)
    ap.add_argument('--camera-every', type=int, default=20)   # 10 captures per sim second
    ap.add_argument('--resolution', default='1920x1080')
    ap.add_argument('--physics-row-every', type=int, default=10)
    ap.add_argument('--mem', default='64G')
    args = ap.parse_args()
    run = HERE / args.run_name
    run.mkdir(parents=True, exist_ok=False)
    choreo = json.loads(Path(args.choreography).read_text())
    (run / 'choreography.json').write_text(json.dumps(choreo, indent=1))
    record = S.launch_owner(run / 'owner', args.gpu, mem=args.mem, robots=args.robots, spacing=0., layout='choreography',
                            route_m=0., base_port=None, policy=args.policy, max_wall_s=args.max_wall_s,
                            max_steps=int(round(args.sim_s * 200)), camera_every=args.camera_every,
                            resolution=args.resolution, physics_row_every=args.physics_row_every,
                            gc_policy='freeze-startup-heap', choreography=args.choreography)
    S.save(run / 'launch.json', {'record': record, 'args': vars(args), 'bundle_sha256': S.BUNDLE_SHA,
                                 'policy_sha256': S.POLICIES[args.policy][1],
                                 'scope': 'showcase choreography; oracle presenter pose; scripted twists; not physical admission'})
    out = Path(record['out'])
    receipt = out / 'receipt.json'
    deadline = time.monotonic() + args.max_wall_s + 300
    while time.monotonic() < deadline and not receipt.exists():
        time.sleep(10)
    print('receipt', receipt.exists(), flush=True)
    if receipt.exists():
        r = json.loads(receipt.read_text())
        print(json.dumps({k: r.get(k) for k in ('completed', 'steps', 'error', 'exit_code', 'wall_duration_s', 'scope')},
                         default=str), flush=True)
        robots = r.get('robots') or {}
        for name, row in sorted(robots.items()):
            st = row.get('last_state') or {}
            pos = st.get('position')
            print(name, 'steps', row.get('steps'), 'controller', st.get('controller'), 'fault', st.get('fault'),
                  'pos', None if pos is None else [round(v, 3) for v in pos], flush=True)


if __name__ == '__main__':
    main()
