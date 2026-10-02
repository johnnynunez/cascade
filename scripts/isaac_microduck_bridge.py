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
    model = p.add_mutually_exclusive_group(required=True)
    model.add_argument('--asset', type=Path)
    model.add_argument('--bundle', type=Path)
    p.add_argument('--bundle-sha256', required=True, help='offline-admitted receipt.json SHA-256')
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--policy-sha256', required=True)
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
    p.add_argument('--max-jpeg-bytes', type=int, default=2*1024**2)
    p.add_argument('--check-only', action='store_true', help='offline admission only; no Kit, socket or writes')
    return p.parse_args(argv)


CONTROLLER_LIMITS = ('max_linear_speed', 'max_angular_speed', 'max_duration_s', 'lease_s',
                     'max_state_age_s', 'max_action_wall_s')
FALL_LIMITS = ('min_height_m', 'max_height_m', 'max_tilt_rad')


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
    if not isinstance(data, dict) or set(data) != set(CONTROLLER_LIMITS + FALL_LIMITS + ('max_contacts', 'max_constraints')):
        raise ValueError('explicit complete controller/fall/solver limits required; no unknown keys')
    for key, value in data.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
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
    from cascade.control.microduck_policy import MicroduckPolicy
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
    from cascade.sim.microduck_newton import KitNewtonBackend
    from cascade.sim.microduck_stepper import FrameCache, MicroduckStepper

    defer = signals.defer if signals is not None else nullcontext
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
        if signals is not None:
            signals.checkpoint()
        with defer():
            sys.path.extend(validate_extra_paths(args.python_extra_path))
            backend = (backend_factory or KitNewtonBackend)(args, admission, experience)
            backend.open()
        if signals is not None:
            signals.checkpoint()
        with defer():
            from cascade.sim.mobile_identity import build_model_identity

            model_identity = build_model_identity(admission, backend.receipt, repo=REPO,
                                                   runtime_scene=out / 'runtime-scene.usda')
            result['model_identity_sha256'] = model_identity['model_identity_sha256']
            write_json(out / 'model-identity.json', model_identity)
            policy = (policy_factory or MicroduckPolicy)(args.policy, args.policy_sha256)
            controller = MobileBridgeController(robot_id=args.robot_id, source=args.source,
                engine='newton', device=args.device, asset_sha256=admission['asset_sha256'],
                policy_sha256=args.policy_sha256,
                model_identity_sha256=model_identity['model_identity_sha256'],
                support_contract=model_identity['support_contract'],
                # Wire v1 nominal dt is retained for existing strict clients; the
                # measured float32 dt is separately frozen in the runtime receipt.
                physics_dt=.005, policy_dt=.020,
                **{k: admission['limits'][k] for k in CONTROLLER_LIMITS})
            remaining = args.max_wall_s - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError('wall deadline expired during bootstrap')
            stepper = MicroduckStepper(backend, controller, policy, backend.bam, max_steps=args.max_steps,
                max_wall_s=remaining, **{k: admission['limits'][k] for k in FALL_LIMITS})
            cache = FrameCache(controller.hello(), max_jpeg_bytes=args.max_jpeg_bytes, max_pixels=640*480)
            server = (server_factory or MobileBridgeServer)(controller, port=args.port, frame_callback=cache)
            stepper.start()
        if signals is not None:
            signals.checkpoint()
        # Cold RTX initialization can outlast the consumer's frame-age bound.
        # Prime it on unscored HOME/bootstrap state, then DISCARD these pixels.
        # The first published image still follows a new controlled solve;
        # neither old pixels nor their timestamps are rejuvenated.
        warmup = backend.capture()
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
                    if capture['step'] != sample['step'] or capture['sim_time_s'] != sample['sim_time']:
                        raise RuntimeError('capture not bound to last published completed state')
                    cache.publish(**capture)
                    wire = cache({'camera': 'overview'})['frame']
                    jpeg = base64.b64decode(wire['rgb_jpeg_b64'], validate=True)
                    filename = f"frames/overview_{sample['step']:09d}.jpg"
                    with (out / filename).open('xb') as image:
                        image.write(jpeg)
                    row(frames, {k: v for k, v in wire.items() if k != 'rgb_jpeg_b64'} | {
                        'file': filename, 'sha256': hashlib.sha256(jpeg).hexdigest(),
                        'render_times': capture['render_times'], 'physics_ticks_during_capture': 0})
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
            if signals is not None:
                result['signal_registration_attempts'] = signals.registration_attempts
            if signals is not None and signals.signum is not None:
                result.update(completed=False, signal=signals.signum, end_reason='signal/lifecycle shutdown')
            result['exit_code'] = 128 + result['signal'] if 'signal' in result else (0 if result['completed'] else 1)
            write_json(out / 'receipt.json', result)
            # SimulationApp.close fast-shutdown terminates the process, often without
            # returning to Python. Preserve the failure exit code and receipt first.
            if backend is not None:
                try:
                    backend.shutdown(exit_code=result['exit_code'])
                    write_json(out / 'teardown.json', {'sdk_close_returned': True})
                except Exception as exc:
                    result['completed'] = False
                    result['exit_code'] = 1
                    result['teardown_errors'].append(str(exc))
                    write_json(out / 'teardown.json', {'sdk_close_returned': False, 'error': str(exc)})
    return result


def admit(args):
    """Offline validation only; do not import Kit/ONNX/Newton or open a socket."""
    import math
    import re
    from cascade.control.newton_bam import SOURCE_SHA256, _validated_params
    from cascade.sim.microduck_newton import (digest_token, experience_text, sha256,
                                              source_manifest, strict_json, verify_bundle)
    if args.engine != 'newton':
        raise ValueError('PhysX BAM unsupported; no fallback')
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
    for key in ('release', 'asset', 'bundle', 'policy', 'bam_source_root', 'out', 'limits'):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.expanduser().absolute())
    if not (args.release / 'python.sh').is_file():
        raise ValueError('--release must be a prepared Isaac release with python.sh')
    if args.out.exists():
        raise ValueError('--out already exists; preserve previous receipts')
    extras = validate_extra_paths(args.python_extra_path)
    args.python_extra_path = [Path(p) for p in extras]
    bundle = args.bundle or args.asset.parent.parent
    for root in (bundle, args.release, args.bam_source_root, *args.python_extra_path):
        if args.out.resolve().is_relative_to(root.resolve()):
            raise ValueError('output must not modify an input/SDK directory')
    admitted = verify_bundle(bundle, expected_sha256=args.bundle_sha256, asset=args.asset)
    digest_token(args.policy_sha256)
    manifest, _ = source_manifest()
    policies = [r for r in manifest['files'] if r['path'] == 'microduck-policies/velstand.onnx']
    if (len(policies) != 1 or policies[0]['sha256'] != args.policy_sha256
            or sha256(args.policy) != args.policy_sha256 or args.policy.stat().st_size != policies[0]['size']):
        raise ValueError('explicit policy must match pinned velstand ONNX SHA/size; no checkpoint substitution')
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
             'src/cascade/sim/microduck_contact_support.py', 'src/cascade/control/mobile_base.py',
             'src/cascade/control/mobile_support.py',
             'src/cascade/apps/signal_stop.py',
             'src/cascade/control/newton_bam.py', 'src/cascade/control/microduck_policy.py',
             'src/cascade/control/microduck_actuator.py', 'assets/microduck/manifest.json',
             'assets/microduck/newton-bam.json', 'configs/isaac/microduck.newton.kit')
    admitted.update(limits=load_limits(args.limits), limits_sha256=sha256(args.limits),
                    bam_params=params, bam_config_sha256=sha256(config_path), bam_source_sha256=bam_sources,
                    policy_sha256=args.policy_sha256, source_sha256={f: sha256(REPO/f) for f in files},
                    experience_text=experience_text(args.release))
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
                'output_count': len(admission['receipt']['outputs']), 'bam_config_sha256': admission['bam_config_sha256']}))
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
