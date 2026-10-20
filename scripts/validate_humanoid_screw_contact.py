"""Standalone G1 wrist-driver thread-contact fixture, any requested turns (Newton).

The humanoid analogue of ``validate_screw_contact.py``: no runtime, owner or
domain; the scene's wrist program runs until the measured nut rotation reaches
``--turns``, then the wrist torque is latched to exact zero and the run
continues for ``--rest`` seconds. The unchanged ``verify_threading`` judges the
whole post-command stream with ``ThreadContract(requested_turns=--turns)``.
Use it for requests the mounted domain does not admit (e.g. two turns); the
one-turn route is ``validate_humanoid_fastening.py``.
"""
import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from cascade.sim.newton_humanoid_screw import SIMPLIFICATIONS, HumanoidThreadingScene
from cascade.sim.threading_verification import ThreadContract, ThreadSample, verify_threading

SOURCES = ('scripts/validate_humanoid_screw_contact.py', 'src/cascade/sim/newton_humanoid_screw.py',
           'src/cascade/sim/threading_verification.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=REPO / 'assets/factory/nut_bolt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--turns', type=float, default=2.)
    parser.add_argument('--rest', type=float, default=1.)
    parser.add_argument('--max-seconds', type=float, default=40.)
    parser.add_argument('--no-drive', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.turns) or not 0 < args.turns <= 5:
        parser.error('--turns must be in (0, 5]')
    args.output.mkdir(parents=True, exist_ok=False)
    before = {n: hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in SOURCES}
    scene = HumanoidThreadingScene(args.assets, args.robot_asset, args.cache, device=args.device,
                                   drive=not args.no_drive)
    names = {f.name for f in fields(ThreadSample)}
    samples, stopped_at, started = [], None, time.monotonic()
    with (args.output / 'samples.jsonl').open('w') as output:
        while scene.time_s < args.max_seconds:
            scene.step()
            row = scene.observe()
            if scene.time_s >= .8 and scene._started_angle is None:
                scene.command(turns=args.turns)
            if scene._started_angle is not None:
                samples.append(ThreadSample(**{k: v for k, v in row.items() if k in names}))
            if stopped_at is None and scene._started_angle is not None and scene.turns_so_far() >= args.turns:
                scene.stop()
                stopped_at = {'time_s': scene.time_s, 'step': scene.step_id, 'turns': scene.turns_so_far()}
            output.write(json.dumps(row, allow_nan=False) + '\n')
            if stopped_at is not None and scene.time_s - stopped_at['time_s'] >= args.rest:
                break
    if before != {n: hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in SOURCES}:
        raise RuntimeError('source changed during the physical experiment')
    result = verify_threading(samples, ThreadContract(pitch_m=scene.pitch_m, requested_turns=args.turns))
    if scene._maximum_nut_angular_speed > 10.:
        result.update(status='unverified', threading_verified=False,
                      reason='substep nut speed exceeded the independently observed 10 rad/s sampling bound')
    tail = [json.loads(line) for line in (args.output / 'samples.jsonl').read_text().splitlines()[-30:]]
    result.update(wall_s=time.monotonic() - started, requested_turns=args.turns, stop=stopped_at,
                  strokes=scene.program.strokes, regrips=scene.program.regrips,
                  rest_tail={'wrist_command_nm': [r['wrist_roll_command_nm'] for r in tail[-1:]],
                             'max_wrist_speed_rad_s': max(abs(r['wrist_roll_speed_rad_s']) for r in tail)},
                  maximum_nut_angular_speed_rad_s=scene._maximum_nut_angular_speed,
                  peak_nut_speed_by_phase=scene.peak_nut_speed_by_phase, no_drive=args.no_drive,
                  simplifications=SIMPLIFICATIONS, asset_hashes=scene.asset_hashes,
                  robot_asset_sha256=hashlib.sha256(args.robot_asset.read_bytes()).hexdigest(),
                  body_labels=list(scene.model.body_label), source_sha256=before, device=str(scene.model.device),
                  controls='G1 joint position actuators, right wrist-roll torque and grasp weld state only',
                  mechanism='free nut / fixed bolt SDF mesh contacts; hand-held six-flat socket driver')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('body_labels', 'source_sha256')}, indent=2), flush=True)
    return 0 if result['threading_verified'] != args.no_drive else 1


if __name__ == '__main__':
    raise SystemExit(main())
