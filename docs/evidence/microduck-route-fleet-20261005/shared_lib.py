"""Hermes-owned harness: N MicroDucks in ONE shared Isaac 6.2/Newton world, long routes.

Source: CASCADE worktree ``integ`` (origin/main + hull fix + long-walk + route candidate).
One `scripts/isaac_microduck_shared.py` owner solves every robot; per-robot loopback
endpoints (``--serve-base-port``) are driven by CASCADE RobotRuntime/FleetRuntime.
Diagnostic, not physical admission; no verifier limit is relaxed beyond the route
profile under test.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

W = Path(__file__).resolve().parent
LAB = W.parent
REPO = LAB / 'HERMES_MICRODUCK_FLEET_20261004' / 'integ'
RELEASE = Path('/home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release')
RECIPE = 'isaac62_48b2d951'
BUNDLE = LAB / 'MICRODUCK/external/isaaclab-usd-allcollisions-v1'
BUNDLE_SHA = 'ba7cdd9ed68552fc83aefaf881a148970203bb08fb003e0d52715f42a8263079'
POLICIES = {
    'rough_walk_e': (LAB / 'LOCOMOTION_NEXT/research/microduck-rough-walk-e/policy.onnx',
                     '5aa423bd693e431b19e2ead77f99cbae6184e40a529eb2f7c1b4f85bb7f57040'),
    'isaaclab_velocity_rough': (LAB / 'MICRODUCK/external/isaaclab-policies-20260901/single/velocity_rough.onnx',
                                '36bad8c2aca4c4e7293d6b989ff1d8f9993e31e532a9a9acb5cc1cc56750ade4'),
    'isaaclab_velocity_flat': (LAB / 'MICRODUCK/external/isaaclab-policies-20260901/single/velocity_flat.onnx',
                               '2c88bdd80a031efce26084b5b67c3713a7e3355cd06fbb1802cef4c06150b507'),
}
BAM_ROOT = LAB / 'MICRODUCK/research/isaaclab-pr8161/source'
BAM_PROFILE = 'official_infer_nominal_no_delay'
ORT = LAB / 'MICRODUCK/runtime-python'
LIMITS = REPO / 'configs/microduck/controller-limits-route.json'
WARP_CACHE = LAB / 'HERMES_MICRODUCK_FLEET_20261004' / 'warp-cache'
CONFIGS = W / 'configs'
GPUS = {'gpu0': 'GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3',
        'gpu1': 'GPU-4c811d02-a79b-2837-425e-5f62ac3c578a'}
BASE_PROFILE = 'microduck_distance_route'


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    os.replace(tmp, path)


def owner_argv(out, *, robots, spacing, layout, route_m, base_port, policy, max_wall_s, max_steps,
               camera_every=20, resolution='1920x1080', physics_row_every=10, gc_policy='freeze-startup-heap',
               ground_visual_m=40.0, profile_phases=False, choreography=None):
    policy_path, policy_sha = POLICIES[policy]
    argv = [str(RELEASE / 'python.sh'), str(REPO / 'scripts/isaac_microduck_shared.py'),
            '--engine', 'newton', '--release', str(RELEASE), '--sdk-recipe', RECIPE,
            '--bundle', str(BUNDLE), '--bundle-sha256', BUNDLE_SHA,
            '--policy', str(policy_path), '--policy-sha256', policy_sha, '--policy-profile', policy,
            '--bam-source-root', str(BAM_ROOT), '--bam-profile', BAM_PROFILE,
            '--python-extra-path', str(ORT), '--robot-id', 'microduck', '--source', 'isaac-microduck',
            '--device', 'cuda:0', '--port', '0', '--limits', str(LIMITS), '--out', str(out),
            '--camera-every', str(camera_every), '--max-jpeg-bytes', str(4*1024**2),
            '--overview-resolution', resolution, '--ground-visual-m', str(ground_visual_m),
            '--max-wall-s', str(max_wall_s), '--max-steps', str(max_steps),
            '--solver-cuda-graph', '--reuse-solved-read',
            '--robots', str(robots), '--spacing', str(spacing), '--layout', layout, '--route-m', str(route_m),
            '--physics-row-every', str(physics_row_every)]
    if base_port is not None:
        argv += ['--serve-base-port', str(base_port)]
    if choreography is not None:
        argv += ['--choreography', str(choreography)]
    if gc_policy:
        argv += ['--gc-policy', gc_policy]
    if profile_phases:
        argv += ['--profile-phases']
    return argv


def clean_env(gpu_uuid, run_dir):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith('CASCADE_') or key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX',
                                               'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_DEVICE_ORDER'):
            env.pop(key)
    stores = Path(run_dir) / 'stores'
    stores.mkdir(parents=True, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=gpu_uuid,
               WARP_CACHE_PATH=str(WARP_CACHE), ISAACSIM_PATH=str(RELEASE),
               CASCADE_BELIEFS_PATH=str(stores / 'beliefs.json'),
               CASCADE_GRASP_MEMORY_PATH=str(stores / 'grasp-memory.json'),
               CASCADE_ENVELOPE_PATH=str(stores / 'envelope.json'),
               CASCADE_SPATIAL_MEMORY_PATH=str(stores / 'spatial-memory.json'))
    return env


def launch_owner(run_dir, gpu, *, mem='64G', cpu='800%', **kwargs):
    """Start the shared owner as a detached user service; returns its record."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    WARP_CACHE.mkdir(exist_ok=True)
    out = run_dir / 'kit'
    argv = owner_argv(out, **kwargs)
    env = clean_env(GPUS[gpu], run_dir)
    log = run_dir / 'owner.log'
    outer = int(kwargs['max_wall_s'] + 200)
    script = run_dir / 'run.sh'
    script.write_text('#!/bin/bash\n'
                      f'cd {shlex.quote(str(REPO))} || exit 99\n'
                      '/usr/bin/env -i ' + ' '.join(shlex.quote(f'{k}={v}') for k, v in sorted(env.items()))
                      + f' /usr/bin/timeout --kill-after=20s {outer}s '
                      + ' '.join(shlex.quote(a) for a in argv)
                      + f' > {shlex.quote(str(log))} 2>&1\n'
                      f'code=$?\necho "EXIT=$code" >> {shlex.quote(str(log))}\nexit $code\n')
    script.chmod(0o700)
    unit = f'hermes-route-owner-{int(time.time())}'
    subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}', '-p', f'MemoryMax={mem}',
                    '-p', f'CPUQuota={cpu}', '/bin/bash', str(script)], check=True,
                   capture_output=True, text=True)
    record = dict(gpu=gpu, gpu_uuid=GPUS[gpu], unit=unit, run_dir=str(run_dir), out=str(out), argv=argv,
                  launched_wall=time.time(), launched_monotonic=time.monotonic(), **kwargs)
    save(run_dir / 'launch.json', record)
    return record


def unit_active(unit):
    r = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True)
    return r.stdout.strip() in ('active', 'activating', 'deactivating')


def log_tail(record, n=20):
    log = Path(record['run_dir']) / 'owner.log'
    try:
        return '\n'.join(log.read_text(errors='replace').splitlines()[-n:])
    except OSError:
        return ''


def wait_listening(record, timeout_s):
    marker = Path(record['out']) / 'BRIDGE_LISTENING.json'
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if marker.exists():
            try:
                return json.loads(marker.read_text())
            except ValueError:
                pass
        if not unit_active(record['unit']) and not marker.exists():
            raise RuntimeError(f"owner exited before listening:\n{log_tail(record)}")
        time.sleep(0.5)
    raise TimeoutError(f"owner not listening after {timeout_s}s:\n{log_tail(record)}")


def owner_pid(record):
    needle = record['out']
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        args = [c.decode(errors='replace') for c in cmd]
        if any(a.endswith('isaac_microduck_shared.py') for a in args) and needle in args \
                and 'python3' in os.path.basename(args[0]):
            return int(pid)
    return None


def terminate(record, timeout_s=180):
    """SIGTERM the owned owner (its signal path writes receipts), then wait."""
    pid = owner_pid(record)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and unit_active(record['unit']):
        time.sleep(0.5)
    if unit_active(record['unit']):
        subprocess.run(['systemctl', '--user', 'stop', record['unit']], capture_output=True)
    return dict(pid=pid, unit_active_after=unit_active(record['unit']), log_tail=log_tail(record, 5))


def ensure_configs(config_dir=None):
    config_dir = Path(config_dir or CONFIGS)
    if not config_dir.exists():
        shutil.copytree(REPO / 'configs', config_dir)
    return config_dir


def write_robot_config(record, robot_id, entry, policy, config_dir=None):
    """Pin this robot's base to ITS identity in the shared owner's receipt and its own endpoint."""
    import yaml
    CONFIGS = ensure_configs(config_dir)  # noqa: N806 - per-world config namespace when given
    out = Path(record['out'])
    identity = json.loads((out / 'model-identity.json').read_text())['robots'][robot_id]
    hello = entry['hello']
    if hello['model_identity_sha256'] != identity['model_identity_sha256']:
        raise RuntimeError(f'{robot_id}: hello identity differs from the owner receipt')
    base_name = f'{robot_id}_route'
    base = {'extends': BASE_PROFILE, 'robot_id': robot_id, 'source': 'isaac-microduck',
            'device': hello['device'], 'asset_sha256': hello['asset_sha256'], 'policy_sha256': POLICIES[policy][1],
            'model_identity_sha256': identity['model_identity_sha256'],
            'support_contract': identity['support_contract'], 'bridge_port': entry['port'],
            'epoch': hello['epoch']}
    (CONFIGS / 'bases' / f'{base_name}.yaml').write_text(yaml.safe_dump(base, sort_keys=False))
    robot = {'version': 1, 'robot_id': robot_id,
             'domains': {'locomotion': {'kind': 'locomotion', 'bases': [base_name]}}}
    (CONFIGS / 'robots' / f'{robot_id}.yaml').write_text(yaml.safe_dump(robot, sort_keys=False))
    return identity['model_identity_sha256']


def load_cfg(robot_id, epoch, config_dir=None):
    from cascade.config import load_robot_config
    cfg = load_robot_config(robot_id, llm='mock', config_dir=Path(config_dir or CONFIGS))
    resolved = cfg._data['domains']['locomotion']['resolved']
    bases = resolved.get('bases') or [resolved['base']]
    for base in bases:
        base['epoch'] = epoch
    if 'base' in resolved:
        resolved['base']['epoch'] = epoch
    return cfg, bases[0]


def passive_ready(base, sim_seconds=1.5, wall_s=120):
    """Independent read-only channel: wait until physics advanced and the robot is upright."""
    from cascade.sim.base_truth import BaseTruthReader
    reader = BaseTruthReader(base)
    samples, target = [], None
    deadline = time.monotonic() + wall_s
    try:
        while time.monotonic() < deadline:
            state = reader()
            samples.append({'at': time.monotonic(), 'error': reader.last_error,
                            'state': state.as_dict() if state else None})
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


def confirmed(result):
    return (result.get('ok') is True and result.get('execution_ok') is True
            and result.get('postcondition', {}).get('status') == 'confirmed')


def compact(result):
    """Everything worth keeping from a RobotRuntime result, without the sample arrays."""
    post = result.get('postcondition') or {}
    measured = result.get('measured') or {}
    before, after = measured.get('before'), measured.get('after')
    def pos(s):
        return None if not s else (s.get('position_world') or s.get('position'))
    return {'ok': result.get('ok'), 'execution_ok': result.get('execution_ok'), 'outcome': result.get('outcome'),
            'reason': result.get('reason'), 'error': result.get('error'),
            'post_status': post.get('status'), 'post_reason': post.get('reason') or post.get('reasons'),
            'post_metrics': {k: v for k, v in post.items() if k not in ('status', 'reason', 'reasons', 'evidence', 'samples')},
            'measured_distance_m': result.get('measured_distance_m'),
            'measured_angle_rad': result.get('measured_angle_rad'),
            'requested_distance_m': result.get('requested_distance_m'),
            'command': result.get('command'), 'before_pos': pos(before), 'after_pos': pos(after),
            'before_sim_t': None if not before else before.get('sim_time_s'),
            'after_sim_t': None if not after else after.get('sim_time_s'),
            'samples': len(measured.get('samples') or [])}
