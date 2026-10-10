#!/usr/bin/env python3
"""Opt-in MicroDuck Newton bridge. Run with --release/python.sh; no PhysX fallback.

Use an outer process-group deadline, e.g. GNU timeout --kill-after=15s 240s,
for Kit startup/native hangs. --max-wall-s also bounds the control episode.
BRIDGE_LISTENING is transport availability, never physical acceptance/READY.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True, choices=['newton'], help='PhysX BAM is unsupported')
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--sdk-recipe', choices=['isaac62_48b2d951'], default=None,
                   help='explicit exact internal SDK recipe; default keeps stable Newton admission')
    model = p.add_mutually_exclusive_group(required=True)
    model.add_argument('--asset', type=Path)
    model.add_argument('--bundle', type=Path)
    p.add_argument('--bundle-sha256', required=True, help='offline-admitted receipt.json SHA-256')
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--policy-sha256', required=True)
    p.add_argument('--policy-profile', choices=['velstand', 'rough_walk_e', 'isaaclab_velocity_flat', 'isaaclab_velocity_rough'], default='velstand',
                   help='explicit reviewed checkpoint; alternative profiles are not physical admission')
    p.add_argument('--target-profile', choices=['direct-v1', 'robotd-targets-v1'], default='direct-v1',
                   help='explicit output transform, separately bound from checkpoint; no physical admission')
    p.add_argument('--integrator-profile', choices=['sdk-default', 'euler-v1'], default='sdk-default',
                   help='opt-in Euler authoring with effective native readback; default leaves SDK selection untouched')
    p.add_argument('--handoff-profile', choices=['stand-on-zero-twist-v1'], default=None,
                   help='opt-in: evaluate the standing policy whenever the commanded twist is zero; requires --standing-policy/--standing-policy-sha256')
    p.add_argument('--standing-policy', type=Path, help='official VelStand ONNX for the opt-in standing handoff')
    p.add_argument('--standing-policy-sha256', help='independent SHA-256 of that standing policy file')
    p.add_argument('--bam-source-root', type=Path, required=True)
    p.add_argument('--bam-profile', required=True)
    p.add_argument('--python-extra-path', type=Path, action='append', default=[],
                   help='explicit directory containing ONLY the optional onnxruntime package')
    p.add_argument('--robot-id', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--device', required=True, help='explicit CUDA device, no CPU fallback')
    p.add_argument('--port', type=int, required=True, help='explicit private loopback port; 0 allowed for tests')
    p.add_argument('--limits', type=Path, required=True, help='explicit controller/fall/solver limits JSON')
    p.add_argument('--out', type=Path, required=True, help='new run directory, never overwritten')
    p.add_argument('--max-wall-s', type=float, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--camera-every', type=int, default=20, help='overview capture interval in completed steps')
    p.add_argument('--camera-rgbd', action='store_true', help='opt-in registered RGB-D; changes effective model identity')
    p.add_argument('--camera-mount', type=Path, help='explicit rigid-render-camera JSON; requires RGB-D and SDK48')
    p.add_argument('--camera-mount-sha256', help='independent SHA256 of the exact camera mount file')
    p.add_argument('--max-jpeg-bytes', type=int, default=2*1024**2)
    p.add_argument('--ground-visual-m', type=float, default=None,
                   help='opt-in display size of the ground plane (metres); the collider stays infinite')
    p.add_argument('--overview-resolution', default=None,
                   help='opt-in overview WIDTHxHEIGHT (default 640x480); recorded in the receipt and identity')
    p.add_argument('--solver-cuda-graph', action='store_true',
                   help='explicit reviewed SDK solver graph; BAM/checkpoints stay outside capture')
    p.add_argument('--reuse-solved-read', action='store_true',
                   help='reuse detached same-solve state/support; requires bound solver graph buffers')
    p.add_argument('--private-rtx-cache', action='store_true',
                   help='exact SDK recipe only: exclusive writable RTX/PSO cache under --out')
    p.add_argument('--rtx-cache-seed', type=Path,
                   help='optional closed, independently frozen manifest.json + data seed directory')
    p.add_argument('--rtx-cache-seed-sha256', help='independently pinned seed manifest SHA-256')
    p.add_argument('--check-only', action='store_true', help='offline admission only; no Kit, socket or writes')
    # Absent unless given, so the recorded arguments of a default run are unchanged.
    p.add_argument('--state-history', type=int, default=argparse.SUPPRESS,
                   help='opt-in (B72): keep the last N completed states (1..256) for the reader op state_history')
    return p.parse_args(argv)


def state_history(args) -> int:
    """The opt-in producer state history size (B72); 0 when --state-history is absent."""
    if not hasattr(args, 'state_history'):
        return 0
    from cascade.sim.mobile_bridge import MAX_STATE_HISTORY
    if type(args.state_history) is not int or not 1 <= args.state_history <= MAX_STATE_HISTORY:
        raise ValueError(f'--state-history must be an integer in 1..{MAX_STATE_HISTORY}')
    return args.state_history


CONTROLLER_LIMITS = ('max_linear_speed', 'max_angular_speed', 'max_duration_s', 'lease_s',
                     'max_state_age_s', 'max_action_wall_s')
FALL_LIMITS = ('min_height_m', 'max_height_m', 'max_tilt_rad')
# Optional together: bridge-side heading hold for straight commands (absent = open loop).
HEADING_HOLD = ('heading_hold_kp', 'heading_hold_ki')


def selected_target_contract(args, admission):
    from cascade.sim.microduck_policy_admission import target_contract, verify_target_contract
    value = verify_target_contract(admission.get('target_contract'), args.policy_sha256)
    if value != target_contract(args.policy_sha256, args.target_profile):
        raise ValueError('selected target profile differs from admission')
    return value


def create_policy(args, admission, factory=None):
    from cascade.control.microduck_policy import MicroduckPolicy
    from cascade.sim.microduck_policy_admission import verify_target_contract
    expected = selected_target_contract(args, admission)
    policy = (factory or MicroduckPolicy)(args.policy, args.policy_sha256,
                                         target_profile=args.target_profile)
    if verify_target_contract(getattr(policy, 'target_contract', None), args.policy_sha256) != expected:
        raise ValueError('constructed policy target contract differs from admission')
    handoff = admission.get('handoff')
    if handoff is not None:
        from cascade.control.microduck_handoff import StandingHandoff
        standing = (factory or MicroduckPolicy)(args.standing_policy, args.standing_policy_sha256,
                                               target_profile=args.target_profile)
        policy = StandingHandoff(policy, standing, profile=handoff['profile'])
        if (policy.contract()['standing_policy_sha256'] != handoff['standing_policy_sha256']
                or policy.target_contract != expected):
            raise ValueError('constructed standing handoff differs from admission')
    return policy


def bind_repo():
    import sys
    sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
    import cascade
    if Path(cascade.__file__).resolve() != REPO / 'src/cascade/__init__.py':
        raise RuntimeError('foreign CASCADE checkout imported')
    for name, module in list(sys.modules.items()):
        if name.startswith('cascade.') and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(REPO / 'src'):
                raise RuntimeError(f'foreign CASCADE dependency: {name}')


def load_limits(path):
    import math
    from cascade.sim.microduck_newton import strict_json
    data = strict_json(Path(path).read_bytes())
    required = set(CONTROLLER_LIMITS + FALL_LIMITS + ('max_contacts', 'max_constraints'))
    if not isinstance(data, dict) or set(data) not in (required, required | set(HEADING_HOLD)):
        raise ValueError('explicit complete controller/fall/solver limits required; no unknown keys; '
                         'heading hold needs both gains')
    for key, value in data.items():
        # The heading-hold integral gain may be zero (proportional-only hold).
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (value == 0 and key != 'heading_hold_ki')):
            raise ValueError(f'positive finite limit required: {key}')
    for key, cap in (('max_contacts', 512), ('max_constraints', 2400)):
        if type(data[key]) is not int or data[key] > cap:
            raise ValueError(f'{key} must fit the frozen solver capacity {cap}')
    if data['min_height_m'] >= data['max_height_m'] or data['max_tilt_rad'] >= math.pi:
        raise ValueError('invalid fall limits')
    return data


def validate_extra_paths(paths):
    result = []
    for path in paths:
        path = Path(path).expanduser().absolute()
        if not path.is_dir() or {p.name for p in path.iterdir()} != {'onnxruntime'} or not (path / 'onnxruntime').is_dir():
            raise ValueError('--python-extra-path may contain ONLY onnxruntime, never a venv/USD/NumPy tree')
        result.append(str(path))
    return result


def json_default(value):
    from cascade.control.mobile_support import SupportObservation
    if type(value) is SupportObservation:
        return value.as_observation_dict()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def write_json(path, value):
    import json
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        stream.write('\n')


def _resolve_outcome(result, signals):
    """Keep the first observed termination, including later close failures."""
    if signals is not None:
        result['signal_registration_attempts'] = signals.registration_attempts
        if signals.signum is not None:
            result.setdefault('signal', signals.signum)
    if 'signal' in result:
        result.update(completed=False, end_reason='signal/lifecycle shutdown')
    if result['teardown_errors']:
        result['completed'] = False
    result['exit_code'] = 128 + result['signal'] if 'signal' in result else (0 if result['completed'] else 1)


def _persist(result, label, action):
    """Record a persistence failure without letting it skip mandatory steps.

    Only the exact exception types the CLI treats as reportable are contained.
    The receipt keeps the outcome already resolved in memory; the failure is
    reported separately so it cannot be mistaken for a lifecycle result.
    """
    try:
        return action()
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        result.setdefault('persistence_errors', []).append(f'{label}: {type(exc).__name__}: {exc}')
        return None


def _refresh_receipt(out, result, signals):
    """Atomically reconcile an owned receipt, preserving prior exact bytes.

    A first signal can arrive during read, encoding, comparison or publication.
    Recheck the notification scalar after that work. Native close that never
    returns still requires an external supervisor; no Python finalizer follows it.
    """
    import json
    import os
    path = out / 'receipt.json'
    if not path.exists():
        # The initial publication failed; publish the resolved outcome now.
        _resolve_outcome(result, signals)
        write_json(path, result)
        return
    for _ in range(3):
        _resolve_outcome(result, signals)
        def encode():
            return (json.dumps(result, indent=2, sort_keys=True, allow_nan=False,
                               default=json_default) + '\n').encode()
        previous, desired = path.read_bytes(), encode()
        if previous == desired:
            if signals is None or signals.signum is None or 'signal' in result:
                return
            _resolve_outcome(result, signals)
        revision = result.get('receipt_revision', 0)
        with (out / f'receipt-history-{revision:03d}.json').open('xb') as stream:
            stream.write(previous)
        result['receipt_revision'] = revision + 1
        temporary = out / '.receipt-update.json'
        with temporary.open('xb') as stream:
            stream.write(encode())
        os.replace(temporary, path)
    raise RuntimeError('receipt outcome did not stabilize within its bounded publication')


def run(args, admission, *, backend_factory=None, policy_factory=None, server_factory=None,
        stop_requested=lambda: False, signals=None):
    """Execute a bounded lifecycle; injected CPU backends are test fixtures only."""
    import base64
    import hashlib
    import json
    import sys
    import time
    from contextlib import nullcontext
    from cascade.apps.signal_stop import SignalRequest
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
    from cascade.sim.microduck_newton import KitNewtonBackend
    from cascade.sim.microduck_integrator import selected as selected_integrator
    from cascade.sim.microduck_stepper import FrameCache, MicroduckStepper

    defer = signals.defer if signals is not None else nullcontext
    selected_target_contract(args, admission)  # Refuse drift before SDK/output ownership.
    integrator_selection = selected_integrator(args, admission)
    def checkpoint():
        if signals is not None:
            signals.checkpoint(persistent=True)
    with defer():
        # Exclusive creation is the ownership boundary; never truncate old receipts.
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        result: dict[str, Any] = dict(completed=False, physical_acceptance=False, steps=0, policy_evaluations=0,
                      frame_count=0, support_probe_count=0, teardown_errors=[])
        backend = controller = stepper = server = cache = None
        original_path = list(sys.path)
        configuration = {'arguments': vars(args), 'admission': admission}
        config_bytes = json.dumps(configuration, sort_keys=True, allow_nan=False, default=json_default).encode()
        result['configuration_sha256'] = hashlib.sha256(config_bytes).hexdigest()
        write_json(out / 'startup.json', {**configuration, 'configuration_sha256': result['configuration_sha256']})
        experience = out / 'microduck.newton.kit'
        with experience.open('x') as stream:
            stream.write(admission['experience_text'])
        result['experience_sha256'] = hashlib.sha256(experience.read_bytes()).hexdigest()
    try:
        checkpoint()
        with defer():
            sys.path.extend(validate_extra_paths(args.python_extra_path))
            backend = (backend_factory or KitNewtonBackend)(args, admission, experience)
            backend.signals = signals
        checkpoint()
        # open owns only the passive SDK acquisition deferral. Active native
        # initialization must not inherit a defer spanning the entire method.
        backend.open()
        checkpoint()
        with defer():
            from cascade.sim.mobile_identity import build_model_identity

            model_identity = build_model_identity(admission, backend.receipt, repo=REPO,
                                                   runtime_scene=out / 'runtime-scene.usda')
            if integrator_selection is not None:
                backend.verify_integrator_identity()
            result['model_identity_sha256'] = model_identity['model_identity_sha256']
            write_json(out / 'model-identity.json', model_identity)
            policy = create_policy(args, admission, policy_factory)
            controller = MobileBridgeController(robot_id=args.robot_id, source=args.source,
                engine='newton', device=args.device, asset_sha256=admission['asset_sha256'],
                policy_sha256=args.policy_sha256,
                model_identity_sha256=model_identity['model_identity_sha256'],
                support_contract=model_identity['support_contract'],
                # Wire v1 nominal dt is retained for existing strict clients; the
                # measured float32 dt is separately frozen in the runtime receipt.
                physics_dt=.005, policy_dt=.020, state_history=state_history(args),
                **{k: admission['limits'][k] for k in CONTROLLER_LIMITS + HEADING_HOLD if k in admission['limits']})
            remaining = args.max_wall_s - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError('wall deadline expired during bootstrap')
            stepper = MicroduckStepper(backend, controller, policy, backend.bam, max_steps=args.max_steps,
                max_wall_s=remaining, checkpoint=checkpoint,
                **{k: admission['limits'][k] for k in FALL_LIMITS})
            if getattr(args, 'camera_rgbd', False):
                from cascade.sim.mobile_rgbd import RgbdFrameCache
                height, width = getattr(backend, 'overview_shape', (480, 640))
                cache = RgbdFrameCache(controller.hello(),
                    calibration=backend.receipt['rgbd_camera']['calibration'],
                    max_jpeg_bytes=args.max_jpeg_bytes, max_pixels=height * width)
                if admission.get('camera_mount') is not None:
                    backend.bind_capture_identity(controller.hello())
            else:
                height, width = getattr(backend, 'overview_shape', (480, 640))
                cache = FrameCache(controller.hello(), max_jpeg_bytes=args.max_jpeg_bytes, max_pixels=height * width)
            server = (server_factory or MobileBridgeServer)(controller, port=args.port, frame_callback=cache)
            stepper.start()
        checkpoint()
        # Cold RTX initialization can outlast the consumer's frame-age bound.
        # Prime it on unscored HOME/bootstrap state, then DISCARD these pixels.
        # The first published image still follows a new controlled solve;
        # neither old pixels nor their timestamps are rejuvenated.
        warmup = backend.capture()
        checkpoint()
        if (warmup['step'] != stepper.initial_step or warmup['sim_time_s'] != stepper.initial_time
                or backend.physics_clock != (stepper.initial_step, stepper.initial_time)):
            raise RuntimeError('camera warmup advanced physics or returned mismatched clocks')
        result['discarded_bootstrap_capture'] = {'step': warmup['step'],
            'sim_time_s': warmup['sim_time_s'], 'published': False, 'reason': 'unscored renderer initialization'}
        del warmup
        write_json(out / 'runtime.json', {'hello': controller.hello(), 'backend': backend.receipt,
                                          'actual_physics_dt': stepper.dt, 'actual_policy_dt': 4*stepper.dt})
        (out / 'frames').mkdir()
        last_policy_attempt = -1
        with ((out / 'physics.jsonl').open('x') as trace,
              (out / 'policy.jsonl').open('x') as policies,
              (out / 'frames.jsonl').open('x') as frames,
              (out / 'support-probe.jsonl').open('x') as probes):
            def row(stream, value):
                stream.write(json.dumps(value, allow_nan=False, default=json_default) + '\n')
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
                    # A cancelled inference and its bounded retry share a
                    # physical step. Do not deduplicate away the discarded input.
                    for record in stepper.policy_records:
                        if record['attempt'] <= last_policy_attempt:
                            raise RuntimeError('policy attempt trace repeated/regressed')
                        row(policies, record)
                        last_policy_attempt = record['attempt']
                checkpoint()
                row(trace, {**sample, 'episode_step': stepper.steps, 'bam': backend.bam.telemetry(),
                            'controller': controller.state()['state']})
                if stepper.steps == 1 or stepper.steps % args.camera_every == 0 or stepper.steps == args.max_steps:
                    probe = getattr(backend, 'support_probe', None)
                    if probe is not None:  # Software lifecycle doubles have no native solver.
                        evidence = probe()
                        row(probes, {**evidence, 'model_identity_sha256': result['model_identity_sha256'],
                                     'epoch': controller.hello()['epoch']})
                        result['support_probe_count'] += 1
                        if (evidence.get('passed') is not True or evidence.get('step') != sample['step']
                                or evidence.get('sim_time_s') != sample['sim_time']):
                            raise RuntimeError('support-force probe failed or belongs to another completed step')
                    capture = backend.capture()
                    checkpoint()
                    if capture['step'] != sample['step'] or capture['sim_time_s'] != sample['sim_time']:
                        raise RuntimeError('capture not bound to last published completed state')
                    pose_evidence = capture.pop('pose_evidence', None)
                    cache.publish(**capture)
                    wire = cache({'camera': 'overview'})['frame']
                    jpeg = base64.b64decode(wire['rgb_jpeg_b64'], validate=True)
                    filename = f"frames/overview_{sample['step']:09d}.jpg"
                    with (out / filename).open('xb') as image:
                        image.write(jpeg)
                    frame_evidence = {}
                    if pose_evidence is not None:
                        pose_file = f"frames/overview_{sample['step']:09d}.pose.json"
                        write_json(out / pose_file, pose_evidence | {'capture_pose': capture['capture_pose']})
                        frame_evidence = {'pose_file': pose_file, 'pose_sha256': hashlib.sha256((out/pose_file).read_bytes()).hexdigest()}
                    row(frames, {k: v for k, v in wire.items() if k != 'rgb_jpeg_b64'} | frame_evidence | {
                        'file': filename, 'sha256': hashlib.sha256(jpeg).hexdigest(),
                        'render_times': capture['render_times'], 'physics_ticks_during_capture': 0})
                    if getattr(args, 'camera_rgbd', False):
                        rgbd = cache({'camera': 'overview', 'modality': 'rgbd'})['rgbd']
                        write_json(out / 'frames' / f"overview_{sample['step']:09d}.rgbd.json", rgbd)
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
                if hasattr(policy, 'telemetry'):
                    result['handoff'] = policy.telemetry()
                result.update(steps=stepper.steps, policy_evaluations=stepper.policy_evaluations,
                              policy_attempts=stepper.policy_attempts, policy_commits=stepper.policy_commits)
            if controller is not None:
                controller.stop(latch=True)
                result['last_state'] = controller.state()
            for resource in (server, cache, stepper if stepper is not None else backend):
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
            result['sdk_shutdown'] = 'requested_after_receipt; verify process exit externally'
            sys.path[:] = original_path
            result.update(receipt_revision=0, receipt_phase='before_sdk_shutdown')
            _resolve_outcome(result, signals)
            # Persistence failures are recorded, never allowed to skip the
            # mandatory SDK shutdown or to override the resolved outcome.
            _persist(result, 'initial receipt', lambda: write_json(out / 'receipt.json', result))
            # A signal during initial persistence must affect the requested SDK
            # exit, not merely a later Python return that the SDK may never allow.
            _persist(result, 'receipt reconciliation', lambda: _refresh_receipt(out, result, signals))
            _resolve_outcome(result, signals)
            if backend is not None:
                try:
                    sdk_returned = backend.shutdown(exit_code=result['exit_code'])
                except Exception as exc:
                    result.update(completed=False, sdk_shutdown='failed', receipt_phase='sdk_shutdown_failed')
                    result['teardown_errors'].append(str(exc))
                    teardown = {'sdk_close_returned': False, 'error': str(exc)}
                else:
                    if sdk_returned is True:
                        result.update(sdk_shutdown='returned', receipt_phase='sdk_shutdown_returned')
                        teardown = {'sdk_close_called': True, 'sdk_close_returned': True}
                    elif sdk_returned is False:
                        result.update(sdk_shutdown='no_owned_app', receipt_phase='no_sdk_handle')
                        teardown = {'sdk_close_called': False, 'sdk_close_returned': False,
                                    'reason': 'no owned SimulationApp handle; partial startup cleanup unverified'}
                    else:
                        result.update(sdk_shutdown='unverified', receipt_phase='backend_shutdown_returned')
                        teardown = {'sdk_close_called': None, 'sdk_close_returned': None,
                                    'reason': 'backend returned without an explicit SDK-close attestation'}
            else:
                result.update(sdk_shutdown='no_owned_app', receipt_phase='no_sdk_handle')
                teardown = {'sdk_close_called': False, 'sdk_close_returned': False,
                            'reason': 'no backend/app handle acquired'}
            # The SDK attestation is decided above; failing to persist it is a
            # separate persistence error, never a false sdk_close_returned=false.
            _persist(result, 'teardown record', lambda: write_json(out / 'teardown.json', teardown))
            # Returning/failing close and teardown writes are also signal
            # boundaries. Preserve earlier receipts instead of erasing evidence.
            _persist(result, 'final receipt reconciliation', lambda: _refresh_receipt(out, result, signals))
    return result


def admit(args):
    """Offline validation only; do not import Kit/ONNX/Newton or open a socket."""
    import math
    import re
    from cascade.control.newton_bam import SOURCE_SHA256, _validated_params
    from cascade.sim.microduck_newton import (experience_text, sha256,
                                              strict_json, verify_bundle)
    from cascade.sim.microduck_policy_admission import admit_policy, target_contract
    from cascade.sim.microduck_integrator import contract as integrator_contract
    integrator = integrator_contract(getattr(args, 'integrator_profile', 'sdk-default'))
    if args.engine != 'newton':
        raise ValueError('PhysX BAM unsupported; no fallback')
    state_history(args)  # opt-in producer state history (B72), refused before any file is read
    mount_path, mount_sha = getattr(args, 'camera_mount', None), getattr(args, 'camera_mount_sha256', None)
    camera_mount = None
    if mount_path is not None or mount_sha is not None:
        if (mount_path is None or mount_sha is None or not args.camera_rgbd
                or args.sdk_recipe != 'isaac62_48b2d951'):
            raise ValueError('camera mount requires paired file/SHA, RGB-D and exact SDK48 recipe')
        from cascade.sim.mobile_camera_pose import admit_mount
        camera_mount = admit_mount(mount_path, mount_sha)
    if args.reuse_solved_read and not args.solver_cuda_graph:
        raise ValueError('same-solve read reuse requires bound solver graph buffers')
    if type(args.port) is not int or not 0 <= args.port <= 65535:
        raise ValueError('explicit port must be in 0..65535')
    if re.fullmatch(r'cuda:\d+', args.device) is None:
        raise ValueError('explicit CUDA device required; no CPU fallback')
    for key in ('max_steps', 'camera_every', 'max_jpeg_bytes'):
        value = getattr(args, key)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if args.max_jpeg_bytes > 4*1024**2:
        raise ValueError('max_jpeg_bytes exceeds bounded frame protocol')
    if not math.isfinite(args.max_wall_s) or args.max_wall_s <= 0:
        raise ValueError('positive finite max_wall_s required')
    for key in ('robot_id', 'source'):
        value = getattr(args, key)
        if not value or value != value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise ValueError(f'invalid {key}')
    for key in ('release', 'asset', 'bundle', 'policy', 'standing_policy', 'bam_source_root', 'out', 'limits'):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.expanduser().absolute())
    if not (args.release / 'python.sh').is_file():
        raise ValueError('--release must be a prepared Isaac release with python.sh')
    sdk_recipe = None
    if args.sdk_recipe is not None:
        from cascade.sim.microduck_sdk import admit_release
        sdk_recipe = admit_release(args.release, args.sdk_recipe)
    if args.out.exists():
        raise ValueError('--out already exists; preserve previous receipts')
    from cascade.sim.private_rtx_cache import admit as admit_rtx_cache
    rtx_cache = admit_rtx_cache(args)
    extras = validate_extra_paths(args.python_extra_path)
    args.python_extra_path = [Path(p) for p in extras]
    bundle = args.bundle or args.asset.parent.parent
    for root in (bundle, args.release, args.bam_source_root, *args.python_extra_path):
        if args.out.resolve().is_relative_to(root.resolve()):
            raise ValueError('output must not modify an input/SDK directory')
    admitted = verify_bundle(bundle, expected_sha256=args.bundle_sha256, asset=args.asset)
    policy_admission = admit_policy(args.policy, args.policy_sha256, args.policy_profile)
    targets = target_contract(args.policy_sha256, args.target_profile)
    handoff = None
    handoff_args = (getattr(args, 'handoff_profile', None), getattr(args, 'standing_policy', None),
                    getattr(args, 'standing_policy_sha256', None))
    if any(value is not None for value in handoff_args):
        from cascade.control.microduck_handoff import HANDOFF_PROFILES, HANDOFF_RULE
        if any(value is None for value in handoff_args) or args.handoff_profile not in HANDOFF_PROFILES:
            raise ValueError('standing handoff requires --handoff-profile, --standing-policy and --standing-policy-sha256 together')
        if args.standing_policy_sha256 == args.policy_sha256:
            raise ValueError('standing handoff requires a standing policy different from the motion policy')
        standing_admission = admit_policy(args.standing_policy, args.standing_policy_sha256, 'velstand')
        handoff = {'profile': args.handoff_profile, 'rule': HANDOFF_RULE, 'motion_policy_sha256': args.policy_sha256,
                   'standing_policy_sha256': args.standing_policy_sha256, 'standing_policy_profile': 'velstand',
                   'standing_policy_admission': standing_admission}
    bam_sources = {}
    for relative, expected in SOURCE_SHA256.items():
        path = args.bam_source_root / relative
        if sha256(path) != expected:
            raise ValueError(f'BAM source SHA mismatch: {relative}')
        bam_sources[relative] = expected
    config_path = REPO / 'assets/microduck/newton-bam.json'
    config = strict_json(config_path.read_bytes())
    if config.get('default_profile', 'missing') is not None or args.bam_profile not in config['profiles']:
        raise ValueError('explicit known BAM profile required; no implicit default')
    params = _validated_params(config['profiles'][args.bam_profile])
    files = ('scripts/isaac_microduck_bridge.py', 'scripts/isaac_runtime.py', 'scripts/isaac_camera_readback.py',
             'scripts/convert_microduck.py', 'src/cascade/sim/microduck_newton.py',
             'src/cascade/sim/microduck_stepper.py', 'src/cascade/sim/microduck_state.py',
             'src/cascade/sim/mobile_bridge.py', 'src/cascade/sim/mobile_identity.py',
             'src/cascade/sim/mobile_rgbd.py',
             'src/cascade/sim/mobile_camera_pose.py',
             'src/cascade/sim/mobile_camera_encoding.py',
             'src/cascade/sensing/models.py',
             'src/cascade/sim/microduck_contact_support.py', 'src/cascade/control/mobile_base.py',
             'src/cascade/control/mobile_support.py', 'src/cascade/control/mobile_telemetry.py',
             'src/cascade/apps/signal_stop.py',
             'src/cascade/control/newton_bam.py', 'src/cascade/control/microduck_policy.py',
             'src/cascade/sim/microduck_policy_admission.py', 'assets/microduck/policy-candidates.json',
             'src/cascade/sim/microduck_solver_graph.py',
             'src/cascade/sim/microduck_integrator.py',
             'src/cascade/sim/microduck_sdk.py',
             'src/cascade/sim/private_rtx_cache.py',
             'src/cascade/control/microduck_actuator.py', 'src/cascade/control/microduck_handoff.py',
             'assets/microduck/manifest.json',
             'assets/microduck/isaaclab-microduck-usd-manifest.json', 'scripts/admit_microduck_usd.py',
             'assets/microduck/newton-bam.json', 'configs/isaac/microduck.newton.kit')
    admitted.update(limits=load_limits(args.limits), limits_sha256=sha256(args.limits),
                    bam_params=params, bam_config_sha256=sha256(config_path), bam_source_sha256=bam_sources,
                    policy_sha256=args.policy_sha256, policy_admission=policy_admission,
                    target_contract=targets, integrator_contract=integrator,
                    source_sha256={f: sha256(REPO/f) for f in files},
                    experience_text=experience_text(args.release, sdk_recipe=args.sdk_recipe))
    if sdk_recipe is not None:
        admitted['sdk_recipe'] = sdk_recipe
    if handoff is not None:
        admitted['handoff'] = handoff
    if camera_mount is not None:
        from cascade.sim.mobile_camera_encoding import encoding_policy
        admitted['camera_mount'] = camera_mount
        admitted['camera_pose_encoding'] = encoding_policy(args.release,args.sdk_recipe)
    if rtx_cache is not None:
        admitted['private_rtx_cache'] = rtx_cache
    return admitted


def main(argv=None):
    import json
    args = parse_args(argv)
    try:
        bind_repo()
        admission = admit(args)
        if args.check_only:
            print(json.dumps({'ok': True, 'physical_acceptance': False,
                'asset_sha256': admission['asset_sha256'], 'asset_receipt_sha256': admission['asset_receipt_sha256'],
                'output_count': len(admission['receipt']['outputs']), 'bam_config_sha256': admission['bam_config_sha256'],
                'target_contract': admission['target_contract'],
                'integrator_contract': admission['integrator_contract'],
                **({'handoff': {k: admission['handoff'][k] for k in ('profile', 'standing_policy_sha256')}} if 'handoff' in admission else {}),
                **({'private_rtx_cache': admission['private_rtx_cache']} if 'private_rtx_cache' in admission else {})}))
            return 0
        from cascade.apps.signal_stop import SignalRequest, StopSignals
        with StopSignals(protect_registration=True) as signals:
            try:
                result = run(args, admission, signals=signals)
            except SignalRequest as exc:
                # A request after run() has returned cannot dispatch more work.
                return 128 + exc.signum
        print(json.dumps({k: result[k] for k in ('completed', 'physical_acceptance', 'steps', 'policy_evaluations', 'teardown_errors')}))
        return result['exit_code']
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'physical_acceptance': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
