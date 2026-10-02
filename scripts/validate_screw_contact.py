"""Record SO-101/Factory contact physics and verify actual nut advancement."""
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
from cascade.sim.newton_screw_contact import ThreadingScene
from cascade.sim.threading_verification import ThreadSample, ThreadContract, verify_threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=REPO / 'assets/factory/nut_bolt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--turns', type=float, default=1.)
    parser.add_argument('--no-drive', action='store_true')
    parser.add_argument('--misaligned', action='store_true')
    parser.add_argument('--substeps', type=int, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
        parser.error('--seconds must be finite and between 1 and 120')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = (Path(__file__), REPO/'src/cascade/sim/newton_screw_contact.py',
               REPO/'src/cascade/sim/threading_verification.py')
    source_before = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/'source-before.json').write_text(json.dumps(source_before,indent=2)+'\n')
    (args.output / 'summary.json').write_text(json.dumps({'threading_verified':False,'status':'running'}))
    try:
        return run(args, sources, source_before)
    except BaseException as exc:
        (args.output/'summary.json').write_text(json.dumps({'threading_verified':False,
            'seating_verified':False, 'status':'error', 'error':f'{type(exc).__name__}: {exc}',
            'source_before':source_before},indent=2)+'\n')
        raise


def run(args, sources, source_before):
    scene = ThreadingScene(args.assets, args.robot_asset, args.cache, drive=not args.no_drive,
                           misaligned=args.misaligned, substeps=args.substeps)
    names = {field.name for field in fields(ThreadSample)}
    samples, rows = [], []
    started = time.monotonic()
    with (args.output / 'samples.jsonl').open('w') as output:
        while scene.time_s < args.seconds:
            scene.step()
            row = scene.observe()
            if scene.time_s >= .8 and scene._started_angle is None:
                scene.command(turns=args.turns)
            if scene._started_angle is not None:
                samples.append(ThreadSample(**{key:value for key,value in row.items() if key in names}))
            rows.append(row)
            output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
            if len(rows) % 60 == 0: print(json.dumps({k:v for k,v in row.items() if k != 'body_poses_xyzw'}),flush=True)
    source_after = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if source_before != source_after:
        raise RuntimeError('source changed during the physical experiment')
    result = verify_threading(samples, ThreadContract(pitch_m=scene.pitch_m, requested_turns=args.turns))
    if scene._maximum_nut_angular_speed > 10.:
        result.update(status='unverified', threading_verified=False,
                      reason='substep nut speed exceeded the independently observed 10 rad/s sampling bound')
    result.update(wall_s=time.monotonic()-started, asset_hashes=scene.asset_hashes,
                  device=str(scene.model.device), body_labels=list(scene.model.body_label),
                  controls='SO-101 joint position actuators and socket spindle torque only',
                  mechanism='free nut / fixed bolt SDF mesh contacts; physical six-flat socket',
                  no_drive=args.no_drive, misaligned=args.misaligned, substeps=args.substeps,
                  source_sha256=source_after, source_unchanged=True,
                  motor_mode='disabled_zero_torque' if args.no_drive else 'velocity_servo_then_brake',
                  maximum_nut_angular_speed_rad_s=scene._maximum_nut_angular_speed,
                  robot_source_sha256={str(path.relative_to(args.robot_asset.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(args.robot_asset.parent.rglob('*')) if path.is_file()})
    (args.output / 'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['threading_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
