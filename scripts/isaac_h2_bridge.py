#!/usr/bin/env python3
"""Unitree H2 PhysX owner (Isaac Sim 6.2 build). Run with <release>/python.sh.

One process owns one H2: it steps PhysX at the bundle's 200 Hz, lets NVIDIA's
deployment chain run the pinned Velocity-H2-History-v0 policy every fourth
solve, and publishes every completed state on the MOBILE wire so SafeBase and
the independent verifier work unchanged. BRIDGE_LISTENING is transport
availability, never physical acceptance or admission.

Use an outer process-group deadline (GNU timeout --kill-after=15s N) for Kit
startup/native hangs; --max-wall-s bounds the control episode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('scripts/isaac_h2_bridge.py', 'scripts/isaac_microduck_bridge.py', 'src/cascade/sim/h2_physx.py',
                'src/cascade/sim/h2_stepper.py', 'src/cascade/sim/h2_identity.py', 'src/cascade/sim/mobile_bridge.py',
                'src/cascade/sim/mobile_identity.py', 'src/cascade/sim/microduck_state.py',
                'src/cascade/sim/microduck_stepper.py', 'src/cascade/control/h2_policy_contract.py',
                'src/cascade/control/mobile_base.py', 'src/cascade/control/mobile_support.py',
                'src/cascade/control/mobile_telemetry.py', 'src/cascade/apps/signal_stop.py',
                'configs/h2/bundle.json', 'assets/h2/bundle/env.yaml', 'assets/h2/bundle/IO_descriptors.yaml')
CONTROLLER_LIMITS = ('max_linear_speed', 'max_angular_speed', 'max_duration_s', 'lease_s',
                     'max_state_age_s', 'max_action_wall_s')
HEADING_HOLD = ('heading_hold_kp', 'heading_hold_ki')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--release', type=Path, required=True, help='prepared Isaac Sim release with python.sh')
    p.add_argument('--policy', type=Path, required=True, help='policy.pt fetched by scripts/h2_assets.py')
    p.add_argument('--limits', type=Path, required=True, help='explicit controller limits JSON')
    p.add_argument('--robot-id', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--device', required=True, help='explicit CUDA device (pin the GPU with CUDA_VISIBLE_DEVICES)')
    p.add_argument('--port', type=int, required=True, help='explicit private loopback port; 0 allowed for tests')
    p.add_argument('--out', type=Path, required=True, help='new run directory, never overwritten')
    p.add_argument('--max-wall-s', type=float, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--camera-every', type=int, default=20, help='overview capture interval in completed steps')
    p.add_argument('--overview-resolution', default='1280x720')
    p.add_argument('--camera-eye', type=float, nargs=3, default=(3.5, -3.5, 1.9),
                   help='world position, or offset from the robot x/y with --camera-follow')
    p.add_argument('--camera-target', type=float, nargs=3, default=(0.5, 0.0, 0.8))
    p.add_argument('--camera-follow', action='store_true', help='chase camera: eye/target offsets follow the robot')
    p.add_argument('--ground-visual-m', type=float, default=40.0)
    p.add_argument('--max-jpeg-bytes', type=int, default=2 * 1024**2)
    p.add_argument('--physics-row-every', type=int, default=1)
    p.add_argument('--check-only', action='store_true', help='offline admission only; no Kit, socket or writes')
    return p.parse_args(argv)


def bind_repo():
    import sys
    sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
    import cascade
    if Path(cascade.__file__).resolve() != REPO / 'src/cascade/__init__.py':
        raise RuntimeError('foreign CASCADE checkout imported')


def load_limits(path):
    import math
    from cascade.sim.microduck_newton import strict_json
    data = strict_json(Path(path).read_bytes())
    required = set(CONTROLLER_LIMITS)
    if not isinstance(data, dict) or set(data) not in (required, required | set(HEADING_HOLD)):
        raise ValueError('explicit complete controller limits required; no unknown keys')
    for key, value in data.items():
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or (value == 0 and key not in HEADING_HOLD)):
            raise ValueError(f'positive finite limit required: {key}')
    return data


def admit(args):
    """Offline validation only; no Kit import, no socket."""
    import hashlib
    import json
    import math
    import re
    from cascade.control.h2_policy_contract import (BUNDLE_MANIFEST, H2PolicyContract, load_bundle_manifest,
                                                    sha256_of, vendored_path, verify_pinned_file)
    from cascade.config import CONFIG_DIR
    from cascade.sim.h2_identity import contract_record
    if type(args.port) is not int or not 0 <= args.port <= 65535:
        raise ValueError('explicit port must be in 0..65535')
    if re.fullmatch(r'cuda(:\d+)?', args.device) is None:
        raise ValueError('explicit CUDA device required; no CPU fallback')
    for key in ('max_steps', 'camera_every', 'max_jpeg_bytes', 'physics_row_every'):
        value = getattr(args, key)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not math.isfinite(args.max_wall_s) or args.max_wall_s <= 0:
        raise ValueError('positive finite max_wall_s required')
    if re.fullmatch(r'\d{2,4}x\d{2,4}', str(args.overview_resolution)) is None:
        raise ValueError('overview resolution must be WIDTHxHEIGHT')
    if not 2. <= float(args.ground_visual_m) <= 1000.:
        raise ValueError('ground visual size must be in 2..1000 m')
    for key in ('robot_id', 'source'):
        value = getattr(args, key)
        if not value or value != value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise ValueError(f'invalid {key}')
    for key in ('release', 'policy', 'out', 'limits'):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    if not (args.release / 'python.sh').is_file():
        raise ValueError('--release must be a prepared Isaac release with python.sh')
    if args.out.exists():
        raise ValueError('--out already exists; preserve previous receipts')
    manifest = load_bundle_manifest()
    contract = H2PolicyContract.from_bundle()
    policy_sha = verify_pinned_file(args.policy, manifest['files']['policy.pt']['sha256'], 'policy.pt')
    limits = load_limits(args.limits)
    lo, hi = contract.command_ranges['lin_vel_x']
    if limits['max_linear_speed'] > min(-lo, hi) or limits['max_angular_speed'] > contract.command_ranges['ang_vel_z'][1]:
        raise ValueError('controller limits exceed the training command ranges')
    manifest_path = CONFIG_DIR / BUNDLE_MANIFEST
    return {
        'bundle': manifest, 'manifest_path': str(manifest_path), 'manifest_sha256': sha256_of(manifest_path),
        'policy': str(args.policy), 'policy_sha256': policy_sha,
        'env_yaml': str(vendored_path(manifest, 'env.yaml')),
        'descriptor_yaml': str(vendored_path(manifest, 'IO_descriptors.yaml')),
        'asset_sha256': manifest['files']['H2.usda']['sha256'],
        'contract_record': contract_record(contract), 'limits': limits, 'limits_sha256': sha256_of(args.limits),
        'source_sha256': {f: sha256_of(REPO / f) for f in SOURCE_FILES},
    }, contract


class OverviewFrames:
    """Diagnostic overview frames for the wire and the run directory (not a sensor)."""

    def __init__(self, identity, *, max_jpeg_bytes, clock=None):
        import threading
        import time
        self.identity = {k: identity[k] for k in ('robot_id', 'source', 'epoch', 'engine', 'device',
                                                 'asset_sha256', 'policy_sha256', 'model_identity_sha256')}
        self.max_jpeg_bytes = int(max_jpeg_bytes)
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._frame = None
        self._captured_at = None
        self._closed = False

    def publish(self, rgb, *, step, sim_time_s, captured_at):
        import base64
        from cascade.sim.microduck_stepper import FrameCache
        jpeg = FrameCache.encode(rgb)
        if len(jpeg) > self.max_jpeg_bytes:
            raise ValueError('JPEG exceeded bound')
        frame = {**self.identity, 'camera': 'overview', 'step': int(step), 'sim_time_s': float(sim_time_s),
                 'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
                 'rgb_jpeg_b64': base64.b64encode(jpeg).decode('ascii')}
        with self._lock:
            if self._closed:
                raise RuntimeError('frame cache closed')
            if self._frame is not None and (step <= self._frame['step'] or sim_time_s <= self._frame['sim_time_s']):
                raise ValueError('capture clocks must advance')
            self._frame, self._captured_at = frame, captured_at
        return jpeg

    def __call__(self, request):
        if request.get('camera') != 'overview':
            raise ValueError('unknown camera; explicit overview required')
        with self._lock:
            if self._closed or self._frame is None:
                raise RuntimeError('no captured frame available')
            return {'frame': {**self._frame, 'producer_age_s': max(0., self.clock() - self._captured_at)}}

    def close(self):
        with self._lock:
            self._closed, self._frame = True, None


def run(args, admission, contract, *, backend_factory=None, server_factory=None, stop_requested=lambda: False,
        signals=None):
    """Bounded lifecycle; injected CPU backends are test fixtures only."""
    import hashlib
    import json
    import sys
    import time
    from contextlib import nullcontext
    import isaac_microduck_bridge as shared  # receipt helpers shared with the MicroDuck owner
    from cascade.apps.signal_stop import SignalRequest
    from cascade.sim.h2_physx import KitH2PhysxBackend
    from cascade.sim.h2_stepper import H2Stepper
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer

    defer = signals.defer if signals is not None else nullcontext
    write_json = shared.write_json

    def checkpoint():
        if signals is not None:
            signals.checkpoint(persistent=True)
    with defer():
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        result = dict(completed=False, physical_acceptance=False, steps=0, policy_evaluations=0, frame_count=0,
                      teardown_errors=[])
        backend = controller = stepper = server = frames = None
        configuration = {'arguments': vars(args), 'admission': admission}
        result['configuration_sha256'] = hashlib.sha256(json.dumps(
            configuration, sort_keys=True, allow_nan=False, default=shared.json_default).encode()).hexdigest()
        write_json(out / 'startup.json', {**configuration, 'configuration_sha256': result['configuration_sha256']})
    try:
        checkpoint()
        backend = (backend_factory or KitH2PhysxBackend)(args, admission, contract)
        backend.signals = signals
        backend.open()
        checkpoint()
        with defer():
            from cascade.sim.h2_identity import build_h2_model_identity
            identity = build_h2_model_identity(admission, backend.receipt, repo=REPO)
            result['model_identity_sha256'] = identity['model_identity_sha256']
            write_json(out / 'model-identity.json', identity)
            controller = MobileBridgeController(
                robot_id=args.robot_id, source=args.source, engine='physx', device=args.device, kind='h2',
                asset_sha256=admission['asset_sha256'], policy_sha256=admission['policy_sha256'],
                model_identity_sha256=identity['model_identity_sha256'],
                support_contract=identity['support_contract'],
                physics_dt=contract.physics_dt, policy_dt=contract.control_dt,
                **{k: admission['limits'][k] for k in CONTROLLER_LIMITS + HEADING_HOLD if k in admission['limits']})
            remaining = args.max_wall_s - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError('wall deadline expired during bootstrap')
            stepper = H2Stepper(backend, controller, contract, max_steps=args.max_steps, max_wall_s=remaining,
                                checkpoint=checkpoint)
            frames = OverviewFrames(controller.hello(), max_jpeg_bytes=args.max_jpeg_bytes)
            server = (server_factory or MobileBridgeServer)(controller, port=args.port, frame_callback=frames)
            stepper.start()
        checkpoint()
        # Prime the renderer on the unscored bootstrap state and DISCARD those pixels.
        warmup = backend.capture()
        checkpoint()
        if (warmup['step'], warmup['sim_time_s']) != (stepper.initial_step, stepper.initial_time):
            raise RuntimeError('camera warmup advanced physics')
        result['discarded_bootstrap_capture'] = {'step': warmup['step'], 'sim_time_s': warmup['sim_time_s'],
                                                 'published': False}
        del warmup
        write_json(out / 'runtime.json', {'hello': controller.hello(), 'backend': backend.receipt,
                                          'actual_physics_dt': stepper.dt,
                                          'actual_policy_dt': contract.decimation * stepper.dt})
        (out / 'frames').mkdir()
        with ((out / 'physics.jsonl').open('x') as trace, (out / 'policy.jsonl').open('x') as policies,
              (out / 'frames.jsonl').open('x') as frame_rows):
            def row(stream, value):
                stream.write(json.dumps(value, allow_nan=False, default=shared.json_default) + '\n')
                stream.flush()
            while stepper.steps < args.max_steps:
                checkpoint()
                if stop_requested():
                    result['end_reason'] = 'signal/lifecycle shutdown'
                    break
                if time.monotonic() - started >= args.max_wall_s:
                    result['end_reason'] = 'max_wall_s'
                    break
                try:
                    sample = stepper.tick()
                finally:
                    for record in stepper.policy_records:
                        row(policies, record)
                checkpoint()
                if stepper.steps == 1 or stepper.steps % args.physics_row_every == 0 or stepper.steps == args.max_steps:
                    row(trace, {**sample, 'episode_step': stepper.steps, 'controller': controller.state()['state']})
                if stepper.steps == 1 or stepper.steps % args.camera_every == 0 or stepper.steps == args.max_steps:
                    capture = backend.capture()
                    checkpoint()
                    if capture['step'] != sample['step'] or capture['sim_time_s'] != sample['sim_time']:
                        raise RuntimeError('capture not bound to the last published completed state')
                    jpeg = frames.publish(**{k: v for k, v in capture.items() if k != 'renders'})
                    filename = f"frames/overview_{sample['step']:09d}.jpg"
                    with (out / filename).open('xb') as image:
                        image.write(jpeg)
                    row(frame_rows, {'step': capture['step'], 'sim_time_s': capture['sim_time_s'], 'file': filename,
                                     'sha256': hashlib.sha256(jpeg).hexdigest(), 'captured_at': capture['captured_at'],
                                     'renders': capture.get('renders'), 'physics_ticks_during_capture': 0})
                    result['frame_count'] += 1
                if stepper.steps == 1:
                    server.start()
                    marker = {'status': 'BRIDGE_LISTENING', 'physical_acceptance': False,
                              'host': server.address[0], 'port': server.address[1],
                              'hello': controller.hello(), 'configuration_sha256': result['configuration_sha256']}
                    write_json(out / 'BRIDGE_LISTENING.json', marker)
                    print('BRIDGE_LISTENING ' + json.dumps(marker, allow_nan=False), flush=True)
            result.update(completed=True, end_reason=result.get('end_reason', 'max_steps'))
    except SignalRequest as exc:
        result.update(completed=False, signal=exc.signum, end_reason='signal/lifecycle shutdown')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if stepper is not None:
            try:
                stepper.fail(exc)
            except Exception as containment_error:
                result['containment_error'] = str(containment_error)
    finally:
        with defer():
            if stepper is not None:
                result.update(steps=stepper.steps, policy_evaluations=stepper.policy_evaluations,
                              policy_attempts=stepper.policy_attempts)
                if stepper.last is not None:
                    result['last_fall_evidence'] = stepper.last.get('fall_evidence')
            if controller is not None:
                controller.stop(latch=True)
                result['last_state'] = controller.state()
            for resource in (server, frames, stepper if stepper is not None else backend):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        result['teardown_errors'].append(str(exc))
            if backend is not None:
                result['backend'] = backend.receipt
            if result['teardown_errors']:
                result['completed'] = False
            result['wall_duration_s'] = time.monotonic() - started
            result.update(receipt_revision=0, receipt_phase='before_sdk_shutdown',
                          sdk_shutdown='requested_after_receipt; verify process exit externally')
            shared._resolve_outcome(result, signals)
            shared._persist(result, 'initial receipt', lambda: write_json(out / 'receipt.json', result))
            shared._persist(result, 'receipt reconciliation', lambda: shared._refresh_receipt(out, result, signals))
            shared._resolve_outcome(result, signals)
            teardown = {'sdk_close_called': False, 'sdk_close_returned': False, 'reason': 'no backend/app handle acquired'}
            if backend is not None:
                try:
                    returned = backend.shutdown(exit_code=result['exit_code'])
                except Exception as exc:
                    result.update(completed=False, sdk_shutdown='failed', receipt_phase='sdk_shutdown_failed')
                    result['teardown_errors'].append(str(exc))
                    teardown = {'sdk_close_returned': False, 'error': str(exc)}
                else:
                    if returned is True:
                        result.update(sdk_shutdown='returned', receipt_phase='sdk_shutdown_returned')
                        teardown = {'sdk_close_called': True, 'sdk_close_returned': True}
                    else:
                        result.update(sdk_shutdown='no_owned_app', receipt_phase='no_sdk_handle')
            shared._persist(result, 'teardown record', lambda: write_json(out / 'teardown.json', teardown))
            shared._persist(result, 'final receipt reconciliation',
                            lambda: shared._refresh_receipt(out, result, signals))
    return result


def main(argv=None):
    import json
    args = parse_args(argv)
    try:
        bind_repo()
        admission, contract = admit(args)
        if args.check_only:
            print(json.dumps({'ok': True, 'physical_acceptance': False, 'policy_sha256': admission['policy_sha256'],
                              'asset_sha256': admission['asset_sha256'], 'joints': len(contract.joint_names),
                              'policy_joints': len(contract.policy_joint_names),
                              'observation_width': contract.observation_width, 'limits': admission['limits']}))
            return 0
        from cascade.apps.signal_stop import SignalRequest, StopSignals
        with StopSignals(protect_registration=True) as signals:
            try:
                result = run(args, admission, contract, signals=signals)
            except SignalRequest as exc:
                return 128 + exc.signum
        print(json.dumps({k: result[k] for k in ('completed', 'physical_acceptance', 'steps', 'policy_evaluations',
                                                 'teardown_errors')}))
        return result['exit_code']
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'physical_acceptance': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
