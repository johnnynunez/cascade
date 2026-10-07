"""Hermes-owned harness: first owned Unitree H2 episodes on PhysX (Isaac Sim 6.2 build).

Diagnostic, not physical admission. One ``scripts/isaac_h2_bridge.py`` owner (GPU 0)
publishes on the MOBILE wire; CASCADE's RobotRuntime -> SafeBase ``walk_velocity`` /
``turn`` drives it, judged by the independent verifier under the candidate profile
``h2_velocity_candidate`` (configs/bases). No verifier limit is relaxed to pass.

Usage: h2_episode.py RUN_NAME [--commands "vx,vy,wz,dur;..."] [--max-wall-s 900]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent  # HERMES_AUDIT_20261007
REPO = Path(os.environ.get('CASCADE_H2_REPO', LAB / 'cascade'))
RELEASE = Path('/home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release')
GPU = 'GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3'  # gpu0: x86 test rig split, never a Spark default
sys.path.insert(0, str(REPO / 'src'))


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    os.replace(tmp, path)


def launch_owner(run_dir, *, port, max_wall_s, max_steps, camera_every, camera_follow=True):
    run_dir = Path(run_dir)
    out = run_dir / 'kit'
    argv = [str(RELEASE / 'python.sh'), str(REPO / 'scripts/isaac_h2_bridge.py'), '--release', str(RELEASE),
            '--policy', str(REPO / 'runs/.install-cache/h2/policy.pt'),
            '--limits', str(REPO / 'configs/h2/controller-limits-candidate.json'), '--robot-id', 'h2',
            '--source', 'isaac-h2', '--device', 'cuda:0', '--port', str(port), '--out', str(out),
            '--max-wall-s', str(max_wall_s), '--max-steps', str(max_steps), '--camera-every', str(camera_every),
            '--overview-resolution', '1280x720', '--physics-row-every', '4',
            '--camera-eye', '3.2', '-4.2', '1.8', '--camera-target', '0.3', '0.0', '0.75']
    if camera_follow:
        argv.append('--camera-follow')
    env = {'HOME': os.environ['HOME'], 'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': GPU,
           'ISAACSIM_PATH': str(RELEASE)}
    log = run_dir / 'owner.log'
    script = run_dir / 'run.sh'
    script.write_text('#!/bin/bash\n' + f'cd {shlex.quote(str(REPO))} || exit 99\n'
                      + '/usr/bin/env -i ' + ' '.join(shlex.quote(f'{k}={v}') for k, v in sorted(env.items()))
                      + f' /usr/bin/timeout --kill-after=20s {int(max_wall_s + 200)}s '
                      + ' '.join(shlex.quote(a) for a in argv) + f' > {shlex.quote(str(log))} 2>&1\n'
                      + f'code=$?\necho "EXIT=$code" >> {shlex.quote(str(log))}\nexit $code\n')
    script.chmod(0o700)
    unit = f'hermes-h2-owner-{int(time.time())}'
    subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}', '-p', 'MemoryMax=64G',
                    '/bin/bash', str(script)], check=True, capture_output=True, text=True)
    record = dict(unit=unit, run_dir=str(run_dir), out=str(out), argv=argv, port=port, gpu=GPU,
                  launched_wall=time.time())
    save(run_dir / 'launch.json', record)
    return record


def unit_active(unit):
    r = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True)
    return r.stdout.strip() in ('active', 'activating', 'deactivating')


def wait_listening(record, timeout_s=400):
    marker = Path(record['out']) / 'BRIDGE_LISTENING.json'
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if marker.exists():
            try:
                return json.loads(marker.read_text())
            except ValueError:
                pass
        if not unit_active(record['unit']):
            raise RuntimeError('owner exited before listening; see owner.log / kit/receipt.json')
        time.sleep(0.5)
    raise TimeoutError('owner not listening')


def owner_pid(record):
    needle = record['out']
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            args = [c.decode(errors='replace') for c in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')]
        except OSError:
            continue
        if any(a.endswith('isaac_h2_bridge.py') for a in args) and needle in args and 'python3' in os.path.basename(args[0]):
            return int(pid)
    return None


def terminate(record, timeout_s=240):
    pid = owner_pid(record)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and unit_active(record['unit']):
        time.sleep(0.5)
    if unit_active(record['unit']):
        subprocess.run(['systemctl', '--user', 'stop', record['unit']], capture_output=True)
    return dict(pid=pid, unit_active_after=unit_active(record['unit']))


def write_robot_config(run_dir, record, marker):
    """Private config dir: the H2 base pinned to THIS owner's identity and endpoint."""
    import shutil
    import yaml
    configs = Path(run_dir) / 'configs'
    if not configs.exists():
        shutil.copytree(REPO / 'configs', configs)
    identity = json.loads((Path(record['out']) / 'model-identity.json').read_text())
    hello = marker['hello']
    if hello['model_identity_sha256'] != identity['model_identity_sha256']:
        raise RuntimeError('hello identity differs from the owner receipt')
    base = {'extends': 'h2_velocity_candidate', 'robot_id': 'h2', 'source': 'isaac-h2', 'kind': 'h2',
            'device': hello['device'], 'asset_sha256': hello['asset_sha256'], 'policy_sha256': hello['policy_sha256'],
            'model_identity_sha256': identity['model_identity_sha256'],
            'support_contract': identity['support_contract'], 'bridge_port': marker['port'], 'epoch': hello['epoch']}
    (configs / 'bases' / 'h2_episode.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
    robot = {'version': 1, 'robot_id': 'h2', 'domains': {'locomotion': {'kind': 'locomotion', 'bases': ['h2_episode']}}}
    (configs / 'robots' / 'h2.yaml').write_text(yaml.safe_dump(robot, sort_keys=False))
    return configs, identity['model_identity_sha256']


def load_cfg(configs, epoch):
    from cascade.config import load_robot_config
    cfg = load_robot_config('h2', llm='mock', config_dir=configs)
    resolved = cfg._data['domains']['locomotion']['resolved']
    bases = resolved.get('bases') or [resolved['base']]
    for base in bases:
        base['epoch'] = epoch
    if 'base' in resolved:
        resolved['base']['epoch'] = epoch
    return cfg, bases[0]


def passive_ready(base, sim_seconds=1.0, wall_s=120):
    from cascade.sim.base_truth import BaseTruthReader
    reader = BaseTruthReader(base)
    samples, target = [], None
    deadline = time.monotonic() + wall_s
    try:
        while time.monotonic() < deadline:
            state = reader()
            samples.append({'at': time.monotonic(), 'error': reader.last_error, 'state': state.as_dict() if state else None})
            if state:
                if state.fallen or state.latched or state.controller_status != 'ready':
                    raise RuntimeError(f'robot not ready: {state.as_dict()}')
                if target is None:
                    target = state.sim_time_s + sim_seconds
                if state.sim_time_s >= target:
                    return samples
            time.sleep(0.1)
        raise TimeoutError('passive preparation deadline')
    finally:
        reader.close()


def state_row(runtime):
    r = runtime.execute('locomotion.get_base_state', {})
    s = (r.get('state') or r.get('result') or r) if isinstance(r, dict) else {}
    return {k: s.get(k) for k in ('sim_time_s', 'position_world', 'orientation_wxyz', 'controller_status', 'fallen',
                                   'linear_velocity_world')}


def compact(result):
    post = result.get('postcondition') or {}
    measured = result.get('measured') or {}
    before, after = measured.get('before'), measured.get('after')

    def pos(s):
        return None if not s else (s.get('position_world') or s.get('position'))
    return {'ok': result.get('ok'), 'execution_ok': result.get('execution_ok'), 'outcome': result.get('outcome'),
            'reason': result.get('reason'), 'error': result.get('error'), 'post_status': post.get('status'),
            'post_reason': post.get('reason') or post.get('reasons'),
            'post_metrics': {k: v for k, v in post.items() if k not in ('status', 'reason', 'reasons', 'evidence', 'samples')},
            'command': result.get('command'), 'before_pos': pos(before), 'after_pos': pos(after),
            'before_sim_t': None if not before else before.get('sim_time_s'),
            'after_sim_t': None if not after else after.get('sim_time_s'),
            'samples': len(measured.get('samples') or [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name')
    ap.add_argument('--commands', default='0.3,0,0,3;0,0,0.5,2;0.3,0,0,3;0,0.2,0,2;-0.3,0,0,2',
                    help='semicolon list of vx,vy,wz,duration_s for walk_velocity')
    ap.add_argument('--port', type=int, default=18400)
    ap.add_argument('--max-wall-s', type=float, default=900.)
    ap.add_argument('--max-steps', type=int, default=400000)
    ap.add_argument('--camera-every', type=int, default=10)
    ap.add_argument('--pause-s', type=float, default=1.0, help='wall pause between commands (robot stands)')
    args = ap.parse_args()
    run = HERE / 'runs' / args.name
    run.mkdir(parents=True, exist_ok=False)
    report = {'name': args.name, 'physical_admission': False, 'steps': [], 'repo': str(REPO)}
    record = launch_owner(run, port=args.port, max_wall_s=args.max_wall_s, max_steps=args.max_steps,
                          camera_every=args.camera_every)
    runtime = None
    try:
        marker = wait_listening(record)
        report['hello'] = marker['hello']
        configs, report['model_identity_sha256'] = write_robot_config(run, record, marker)
        cfg, base = load_cfg(configs, marker['hello']['epoch'])
        save(run / 'resolved-config.json', cfg.as_dict())
        report['passive_samples'] = len(passive_ready(base))
        from cascade.apps.robot_runtime import build_robot_runtime
        runtime, _ = build_robot_runtime(cfg, run / 'runtime')
        report['before'] = state_row(runtime)
        for index, spec in enumerate(s for s in args.commands.split(';') if s.strip()):
            vx, vy, wz, duration = (float(v) for v in spec.split(','))
            call = {'vx': vx, 'vy': vy, 'wz': wz, 'duration_s': duration}
            t0 = time.monotonic()
            result = runtime.execute('locomotion.walk_velocity', call)
            wall = time.monotonic() - t0
            step = {'index': index, 'tool': 'locomotion.walk_velocity', 'args': call, 'wall_s': round(wall, 2),
                    'confirmed': result.get('ok') is True and result.get('execution_ok') is True
                    and (result.get('postcondition') or {}).get('status') == 'confirmed', **compact(result)}
            rows = (result.get('measured') or {}).get('samples') or []
            with (run / f'verifier-samples-{index}.jsonl').open('w') as f:
                for s in rows:
                    f.write(json.dumps({k: s.get(k) for k in ('sim_time_s', 'wall_time_s', 'position_world',
                                                               'orientation_wxyz', 'linear_velocity_world',
                                                               'angular_velocity_body', 'controller_status', 'fallen',
                                                               'latched')}) + '\n')
            step['after'] = state_row(runtime)
            report['steps'].append(step)
            print(json.dumps({'step': index, **{k: step[k] for k in ('args', 'confirmed', 'post_status', 'post_reason',
                                                                       'wall_s', 'error', 'before_pos', 'after_pos')}},
                             default=str), flush=True)
            save(run / 'episode.json', report)
            if step['after'].get('fallen') or step['after'].get('controller_status') in ('fault', 'disabled'):
                report['stopped_early'] = step['after']
                break
            time.sleep(args.pause_s)
    except BaseException:
        report['error'] = traceback.format_exc()
        print(report['error'][-2000:], flush=True)
    finally:
        if runtime is not None:
            try:
                report['runtime_close'] = runtime.close()
            except BaseException as exc:
                report['runtime_close_error'] = repr(exc)
        report['owner_terminate'] = terminate(record)
        try:
            receipt = json.loads((Path(record['out']) / 'receipt.json').read_text())
            report['owner_receipt'] = {k: receipt.get(k) for k in ('completed', 'end_reason', 'error', 'steps',
                                                                   'policy_evaluations', 'frame_count',
                                                                   'wall_duration_s', 'exit_code', 'last_fall_evidence')}
        except Exception as exc:  # noqa: BLE001
            report['owner_receipt_error'] = repr(exc)
        save(run / 'episode.json', report)
        print(json.dumps({k: report.get(k) for k in ('owner_receipt', 'error', 'stopped_early')}, default=str)[:1500])


if __name__ == '__main__':
    main()
