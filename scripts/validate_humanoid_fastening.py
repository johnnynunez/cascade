"""Humanoid fastening episodes through the ordinary configuration/runtime route.

A Unitree G1 (newton-assets 29-DoF + Dex3 MJCF) holds a vertical hex driver
from above with a three-finger contact grasp (thumb opposing index + middle; no
weld) and turns the pre-engaged Factory M20 nut by rotating the hand about the
tool axis with the whole arm (with re-grips), in Newton with SDF thread contact. Each episode builds a fresh composed runtime
from a robot profile (``kind: fastening``, recipe ``g1_wrist_driver_m20_v1``)
via ``load_robot_config`` + ``build_robot_runtime`` and calls ``reset_stop``
and ``fastening.turn_screw(turns=1, direction="tighten")`` through
``RobotRuntime.execute`` (the call the MCP server makes; stdio/LLM are not
exercised). The unchanged ``FasteningDomain`` judges every command.

Run in a standalone Newton environment (newton 1.6, warp, mujoco, mujoco-warp,
scipy, trimesh), not cascade's ``.venv``::

  <newton-env>/bin/python scripts/validate_humanoid_fastening.py \\
      --robot-asset <newton-assets>/unitree_g1/mjcf/g1_29dof_with_hand_rev_1_0.xml \\
      --output runs/humanoid-fastening-01 --device cuda:0 --episodes 1,1,2

``--episodes`` lists commands per episode (``2`` = two consecutive one-turn
commands, additionally judged as one two-turn request by ``verify_threading``).
``--no-drive`` is a negative control (exact zero wrist-roll torque);
``--fingers-open`` is the grasp negative control: the hand performs the same
turn with its fingers open, so nothing but contact could turn the driver.
Every summary reports per-stroke thumb/index/middle normal forces, slip and
transmitted torque (``stroke_grip_stats``).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

PROFILE = 'g1_wrist_driver_fastening'
SOURCES = ('scripts/validate_humanoid_fastening.py', 'src/cascade/sim/newton_humanoid_screw.py',
           'src/cascade/sim/humanoid_fastening.py', 'src/cascade/apps/humanoid_fastening_runtime.py',
           'src/cascade/apps/factory_runtime.py', 'src/cascade/skills/fastening_runtime.py',
           'src/cascade/control/fastening.py', 'src/cascade/sim/threading_verification.py',
           'src/cascade/sim/factory_owner.py')


def _digests():
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in SOURCES}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + '\n')


def _profile(args, digest):
    return {'kind': 'fastening', 'recipe': 'g1_wrist_driver_m20_v1', 'assets': str(args.assets.resolve()),
            'robot_asset': str(args.robot_asset.resolve()), 'device': args.device,
            'model_identity_sha256': digest, 'gc_policy': 'freeze-startup-heap'}


def _config_dir(output, profile):
    """Private pinned copy of the shipped profile (shipped stays device/digest null)."""
    config = output / 'config'
    (config / 'robots').mkdir(parents=True, exist_ok=True)
    (config / 'llm').mkdir(exist_ok=True)
    shutil.copy(REPO / 'configs/llm/mock.yaml', config / 'llm/mock.yaml')
    lines = ['version: 1', 'robot_id: unitree_g1_29dof_dex3_wrist_driver', 'domains:', '  fastening:']
    lines += [f'    {key}: {json.dumps(value)}' for key, value in profile.items()]
    (config / 'robots' / f'{PROFILE}.yaml').write_text('\n'.join(lines) + '\n')
    return config


def _rows(directory):
    with (directory / 'solves.jsonl').open() as stream:
        for line in stream:
            yield json.loads(line)


def _summarize_command(result):
    pre = result.get('pre_stop_threading') or {}
    final = result.get('final_threading') or {}
    rest = result.get('rest') or {}
    return {'verified': result.get('verified'), 'ok': result.get('ok'),
            'postcondition': (result.get('postcondition') or {}).get('status'),
            'reason': (result.get('postcondition') or {}).get('reason'), 'error': result.get('error'),
            'rotation_before_stop_turns': (pre.get('measured') or {}).get('turns'),
            'final_rotation_turns': (final.get('measured') or {}).get('turns'),
            'axial_advance_m': (final.get('measured') or {}).get('axial_advance_m'),
            'max_pitch_error_m': (final.get('measured') or {}).get('max_pitch_error_m'),
            'max_radial_offset_m': (final.get('measured') or {}).get('max_radial_offset_m'),
            'max_axis_tilt_rad': (final.get('measured') or {}).get('max_axis_tilt_rad'),
            'contact_rotation_fraction': (final.get('measured') or {}).get('contact_rotation_fraction'),
            'pre_stop_checks': pre.get('checks'), 'final_checks': final.get('checks'),
            'rest': {k: rest.get(k) for k in ('status', 'window_sim_s', 'samples', 'zero_spindle_effort')}}


def _stream_stats(directory, admissions):
    """Contact/torque statistics from the retained raw solves (no re-simulation)."""
    from cascade.sim.humanoid_fastening import grip_row
    grip_rows = []
    stats = {'solves': 0, 'max_abs_wrist_torque_nm': 0., 'max_thread_load_n': 0., 'max_tool_wall_load_n': 0.,
             'max_socket_floor_load_n': 0., 'max_nut_angular_speed_rad_s': 0., 'max_joint_speed_rad_s': 0.,
             'regrips': 0, 'strokes': 0, 'phases': {}}
    samples = []
    first = min(a['admission_step'] for a in admissions) if admissions else None
    for raw in _rows(directory):
        solve = raw['solve']
        stats['solves'] += 1
        stats['max_abs_wrist_torque_nm'] = max(stats['max_abs_wrist_torque_nm'], abs(solve['spindle_effort_nm']))
        stats['max_thread_load_n'] = max(stats['max_thread_load_n'], raw['thread_load_n'])
        stats['max_tool_wall_load_n'] = max(stats['max_tool_wall_load_n'], raw['tool_wall_load_n'])
        stats['max_socket_floor_load_n'] = max(stats['max_socket_floor_load_n'], raw['socket_floor_load_n'])
        stats['max_nut_angular_speed_rad_s'] = max(stats['max_nut_angular_speed_rad_s'], solve['fastener_angular_speed_rad_s'])
        stats['max_joint_speed_rad_s'] = max(stats['max_joint_speed_rad_s'], max(map(abs, solve['joint_velocity_rad_s'])))
        stats['regrips'] = max(stats['regrips'], raw['program']['regrips'])
        stats['strokes'] = max(stats['strokes'], raw['program']['strokes'])
        phase = raw['program']['phase']
        stats['phases'][phase] = stats['phases'].get(phase, 0) + 1
        grip_rows.append(grip_row(phase, solve['simulation_time_s'], raw, raw['hand_phi_rad'],
                                  raw['nut_turns_unwrapped']))
        if first is not None and solve['step'] > first:
            samples.append(solve)
    from cascade.sim.humanoid_fastening import stroke_grip_stats
    stats['strokes_grip'] = stroke_grip_stats(grip_rows)
    stats['failed_grasp_strokes'] = sum(not s['grasp_ok'] for s in stats['strokes_grip'])
    return stats, samples


def _two_turn_verdict(samples, binding):
    from cascade.sim.threading_verification import ThreadContract, ThreadSample, verify_threading
    rows = [ThreadSample(epoch=s['epoch'], step=s['step'], time_s=s['simulation_time_s'],
                         fastener_id=binding['fastener_id'], fixture_id=binding['fixture_id'],
                         fastener_position_m=s['fastener_position_m'], fastener_quaternion_xyzw=s['fastener_quaternion_xyzw'],
                         fixture_position_m=s['fixture_position_m'], fixture_quaternion_xyzw=s['fixture_quaternion_xyzw'],
                         thread_contacts=s['thread_contacts'], tool_contacts=s['tool_contacts']) for s in samples]
    return verify_threading(rows, ThreadContract(pitch_m=.0025, requested_turns=2.))


def run_episode(args, index, commands, digest, sources):
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import load_robot_config
    episode = args.output / f'episode-{index:02d}'
    episode.mkdir()
    config = _config_dir(episode, _profile(args, digest))
    cfg = load_robot_config(PROFILE, config_dir=config)
    started = time.monotonic()
    runtime, _ = build_robot_runtime(cfg, episode / 'run')
    built = time.monotonic() - started
    calls = []
    try:
        for command in range(commands):
            t0 = time.monotonic()
            reset = runtime.execute('reset_stop', {}, operator=True)  # this script is the operator
            t1 = time.monotonic()
            result = runtime.execute('fastening.turn_screw', {'turns': 1.0, 'direction': 'tighten'})
            calls.append({'reset_stop': reset, 'reset_wall_s': t1 - t0, 'turn_wall_s': time.monotonic() - t1,
                          'result': result})
            if not result.get('verified'):
                break
    finally:
        closure = runtime.close()
    domain_dir = episode / 'run/domains/fastening'
    admissions = [c['result'].get('admission') for c in calls if c['result'].get('admission')]
    stats, samples = _stream_stats(domain_dir, admissions)
    summary = {'episode': index, 'commands_requested': commands, 'build_wall_s': built,
               'readiness': json.loads((domain_dir / 'readiness.json').read_text()),
               'commands': [dict(_summarize_command(c['result']), reset_ok=c['reset_stop'].get('ok'),
                                 turn_wall_s=c['turn_wall_s']) for c in calls],
               'verified_commands': sum(bool(c['result'].get('verified')) for c in calls),
               'stream': stats, 'closure_ok': bool(closure.get('ok')) if isinstance(closure, dict) else closure,
               'model_identity_sha256': digest, 'source_sha256': sources, 'no_drive': args.no_drive,
               'fingers_open': args.fingers_open}
    if commands == 2 and len(calls) == 2:
        summary['two_turn_verify_threading'] = _two_turn_verdict(
            samples, {'fastener_id': 'factory_nut_m20_loose', 'fixture_id': 'fixed_factory_bolt_m20_loose'})
    _write(episode / 'calls.json', calls)
    _write(episode / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'readiness')}, indent=1,
                     default=str), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-asset', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=REPO / 'assets/factory/nut_bolt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--episodes', default='1,1,2')
    parser.add_argument('--no-drive', action='store_true')
    parser.add_argument('--fingers-open', action='store_true')
    args = parser.parse_args()
    episodes = [int(x) for x in args.episodes.split(',')]
    if not episodes or any(n not in (1, 2) for n in episodes):
        parser.error('--episodes is a comma list of 1 or 2 commands per episode')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = _digests()
    _write(args.output / 'source-before.json', sources)
    from cascade.apps.humanoid_fastening_runtime import prepare_humanoid_model
    from cascade.sim.newton_humanoid_screw import SIMPLIFICATIONS
    if args.fingers_open:
        # Grasp negative control only: the same turn program with the fingers open.
        from cascade.sim.newton_humanoid_screw import HumanoidThreadingScene
        HumanoidThreadingScene.grip_enabled = False
    if args.no_drive:
        # Negative control only: every wrist command is exact zero torque.
        from cascade.sim.newton_humanoid_screw import HumanoidThreadingScene
        HumanoidThreadingScene.wrist_torque = lambda self, roll_ref, roll, roll_speed: 0.
    t0 = time.monotonic()
    model = prepare_humanoid_model(_profile(args, None), args.output / 'prepare-sdf-cache')
    digest = model.binding.model_sha256
    _write(args.output / 'prepare.json', {'model_identity_sha256': digest, 'wall_s': time.monotonic() - t0,
                                          'document': model.document, 'binding': json.loads(json.dumps(
                                              model.binding.__dict__, default=list)), 'limits': model.limits.__dict__})
    del model
    results = []
    for index, commands in enumerate(episodes, 1):
        try:
            results.append(run_episode(args, index, commands, digest, sources))
        except Exception as exc:
            results.append({'episode': index, 'error': f'{type(exc).__name__}: {exc}'})
            print('episode failed', index, exc, flush=True)
    if _digests() != sources:
        raise RuntimeError('source changed during the physical experiment')
    _write(args.output / 'summary.json', {'simplifications': SIMPLIFICATIONS, 'episodes': results,
                                          'model_identity_sha256': digest, 'source_sha256': sources,
                                          'no_drive': args.no_drive, 'fingers_open': args.fingers_open})
    ok = all('error' not in r and r.get('verified_commands') == r.get('commands_requested') for r in results)
    return 0 if ok != (args.no_drive or args.fingers_open) else 1


if __name__ == '__main__':
    raise SystemExit(main())
