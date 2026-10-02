"""Negative control: the complete seating fixture with exactly zero spindle torque."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from cascade.sim.newton_screw_seating import SeatingScene
from cascade.sim.threading_verification import ThreadContract, ThreadSample, verify_threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=REPO/'assets/factory/nut_bolt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *(REPO/'src/cascade/sim'/name for name in
               ('newton_screw_contact.py', 'newton_screw_seating.py', 'threading_verification.py'))]
    before = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/'source-before.json').write_text(json.dumps(before, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps({'status':'running','negative_control_passed':False}))
    try:
        scene = SeatingScene(args.assets, args.robot_asset, args.cache, drive=False)
        samples, rows = [], []
        with (args.output/'samples.jsonl').open('w') as output, (args.output/'substeps.jsonl').open('w') as substeps:
            while scene.time_s < 8.:
                scene.step()
                row = scene.observe()
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush()
                if scene.time_s >= .8 and scene._started_angle is None:
                    scene.command(turns=1.)
                for sample in scene._substep_samples:
                    substeps.write(json.dumps(sample, allow_nan=False)+'\n')
                    if scene._started_angle is not None: samples.append(ThreadSample(**sample))
                substeps.flush()
                if row['tool_fixture_interference_seen']:
                    raise RuntimeError('negative control encountered tool interference')
        after = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        if before != after: raise RuntimeError('source changed during zero-drive control')
        verdict = verify_threading(samples, ThreadContract(pitch_m=scene.pitch_m, requested_turns=1.))
        checks = verdict.get('checks', {})
        passed = (verdict['status'] == 'refuted'
                  and checks.get('requested_fastener_rotation') is False
                  and checks.get('axial_advance') is False
                  and checks.get('axis_alignment') is True and checks.get('fixed_fixture') is True
                  and all(row['motor_torque_nm'] == 0. for row in rows)
                  and scene._maximum_nut_angular_speed <= 10.)
        result = {'status':'confirmed' if passed else 'refuted', 'negative_control_passed':passed,
            'motor_mode':'disabled_exact_zero_torque', 'threading':verdict, 'seating_verified':False,
            'source_sha256':after, 'source_unchanged':True, 'asset_hashes':scene.asset_hashes,
            'scene':'factory_seating_v3', 'body_labels':list(scene.model.body_label),
            'maximum_nut_angular_speed_rad_s':scene._maximum_nut_angular_speed,
            'robot_source_sha256':{str(p.relative_to(args.robot_asset.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.robot_asset.parent.rglob('*')) if p.is_file()}}
        (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, indent=2), flush=True)
        return 0 if passed else 1
    except BaseException as exc:
        (args.output/'summary.json').write_text(json.dumps({'status':'error', 'negative_control_passed':False,
            'error':f'{type(exc).__name__}: {exc}', 'source_before':before}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
