"""Continue contact-driven threading to a measured shoulder seat and motor-off rest."""
import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from cascade.sim.newton_screw_seating import SeatingScene
from cascade.sim.seating_verification import loaded_shoulder, verify_factory_seating
from cascade.sim.threading_verification import ThreadContract, ThreadSample, verify_threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=REPO/'assets/factory/nut_bolt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=100.)
    parser.add_argument('--substeps', type=int, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
        parser.error('--seconds must be finite and in [1, 120]')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *(REPO/'src/cascade/sim'/name for name in
               ('newton_screw_contact.py', 'newton_screw_seating.py', 'threading_verification.py',
                'seating_verification.py'))]
    before = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/'source-before.json').write_text(json.dumps(before, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps({'status':'running', 'seating_verified':False}))
    try:
        return run(args, sources, before)
    except BaseException as exc:
        (args.output/'summary.json').write_text(json.dumps({'status':'error', 'seating_verified':False,
            'error':f'{type(exc).__name__}: {exc}', 'source_before':before}, indent=2)+'\n')
        raise


def run(args, sources, before):
    scene = SeatingScene(args.assets, args.robot_asset, args.cache, substeps=args.substeps)
    names = {field.name for field in fields(ThreadSample)}
    rows, samples, seat_rows = [], [], []
    seat_since = motor_off_at = None
    termination = 'deadline_without_measured_seat'
    started = time.monotonic()
    with (args.output/'samples.jsonl').open('w') as output, (args.output/'substeps.jsonl').open('w') as substeps:
        while scene.time_s < args.seconds:
            scene.step()
            row = scene.observe()
            if scene.time_s >= .8 and scene._started_angle is None:
                scene.command(turns=20.)
            for sample in scene._substep_samples:
                substeps.write(json.dumps(sample, allow_nan=False)+'\n')
                if scene._started_angle is not None and motor_off_at is None:
                    samples.append(ThreadSample(**{k:v for k,v in sample.items() if k in names}))
            substeps.flush()
            row['phase'] = 'motor_off_rest' if motor_off_at is not None else 'threading_to_shoulder'
            rows.append(row)
            output.write(json.dumps(row, allow_nan=False)+'\n'); output.flush()
            if len(rows) % 60 == 0:
                print(json.dumps({k:row[k] for k in ('time_s','angle_rad','axial_position_m',
                    'nut_bottom_gap_m','shoulder_contacts','shoulder_contact_force_n','motor_torque_nm',
                    'nut_angular_speed_rad_s','tool_fixture_contacts','phase')}), flush=True)
            if row['tool_fixture_interference_seen']:
                scene.drive = False
                termination = 'tool_fixture_interference'
                break
            loaded_seat = loaded_shoulder(row)
            if motor_off_at is None:
                if loaded_seat:
                    if seat_since is None: seat_since = scene.time_s
                    seat_rows.append(row)
                    if scene.time_s-seat_since >= .5:
                        motor_off_at = scene.time_s
                        scene.drive = False  # Exactly zero spindle actuation in the rest stage.
                else:
                    seat_since = None
                    seat_rows.clear()
            elif scene.time_s-motor_off_at >= 2.:
                termination = 'motor_off_rest_completed'
                break
    after = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if before != after: raise RuntimeError('source changed during physical seating experiment')
    threading = verify_threading(samples, ThreadContract(pitch_m=scene.pitch_m, requested_turns=15.))
    rest = [row for row in rows if motor_off_at is not None and row['time_s'] >= motor_off_at+1.]
    verdict = verify_factory_seating(rows, threading, motor_off_at)
    checks = verdict['checks']
    confirmed = verdict['seating_verified'] and termination == 'motor_off_rest_completed'
    result = {'status':'confirmed' if confirmed else verdict['status'],
        'seating_verified':confirmed, 'preload_verified':False,
        'reason':verdict.get('reason'),
        'checks':checks, 'termination':termination, 'threading':threading,
        'illustrative_motor_limit_nm':scene.motor_limit_nm, 'motor_off_time_s':motor_off_at,
        'final_nut_position_m':rows[-1]['fastener_position_m'],
        'final_nut_bottom_gap_m':rows[-1]['nut_bottom_gap_m'],
        'loaded_seat_samples':len(seat_rows), 'retained_rest_samples':len(rest),
        'maximum_nut_angular_speed_rad_s':scene._maximum_nut_angular_speed,
        'elapsed_simulation_s':scene.time_s, 'wall_s':time.monotonic()-started,
        'source_sha256':after, 'source_unchanged':True, 'asset_hashes':scene.asset_hashes,
        'device':str(scene.model.device), 'substeps':args.substeps, 'scene':'factory_seating_v3',
        'seat_fixture':{'id':'fixed_annular_seat_3mm', 'inner_radius_m':.011,
            'outer_radius_m':.018, 'thickness_m':.003, 'top_z_m':.023, 'fixed_to_bolt_fixture':True},
        'thread_friction_coefficient':.1, 'socket_radial_clearance_m':.0003,
        'socket_contact_solref':[.015,1.],
        'socket_offset_m':scene.socket_offset.tolist(),
        'socket_inserts':{'count':6, 'passive_only':True, 'mass_kg_each':.002,
            'stiffness_n_m':2000., 'damping_n_s_m':4., 'spring_reference_m':-.00035,
            'range_m':[-.0006,.0008], 'nominal_contact_preload_n_each':.1},
        'body_labels':list(scene.model.body_label),
        'robot_source_sha256':{str(p.relative_to(args.robot_asset.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.robot_asset.parent.rglob('*')) if p.is_file()}}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['seating_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
