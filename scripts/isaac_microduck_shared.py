#!/usr/bin/env python3
"""Bounded shared-world MicroDuck runtime, with opt-in private endpoints.

Accepts the single-robot bridge's explicit asset/SDK/policy flags plus --robots
(1..12) and --spacing (at least 2 m). --port must be 0. The default foundation
exposes no command socket; --serve-base-port explicitly enables one independent
loopback endpoint per robot (0 chooses ephemeral ports). No task admission claim.
Use an outer process-group deadline for native startup/teardown hangs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
import time
from pathlib import Path

from isaac_microduck_bridge import (CONTROLLER_LIMITS, FALL_LIMITS, REPO, admit,
    bind_repo, json_default, parse_args, write_json, _persist, _refresh_receipt, _resolve_outcome)


def run(args, admission, signals):
    from cascade.apps.signal_stop import SignalRequest
    from cascade.control.microduck_policy import MicroduckPolicy
    from cascade.sim.mobile_bridge import MobileBridgeController
    from cascade.sim.microduck_shared import SharedMicroduckStepper
    from cascade.sim.microduck_shared_native import SharedKitNewtonBackend, SharedRobotView
    from cascade.sim.microduck_stepper import MicroduckStepper
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    started = time.monotonic()
    result = {'completed': False, 'physical_acceptance': False, 'steps': 0,
              'scope': 'shared-scene zero-command foundation', 'teardown_errors': []}
    owner = fleet = endpoints = profile = None
    steppers = []
    previous_path = list(sys.path)
    try:
        write_json(out / 'startup.json', {'arguments': vars(args), 'admission': admission})
        experience = out / 'microduck.shared.kit'
        with experience.open('x') as f:
            f.write(admission['experience_text'])
        sys.path.extend(map(str, args.python_extra_path))
        owner = SharedKitNewtonBackend(args, admission, experience)
        owner.signals = signals
        owner.open()
        if getattr(args, 'profile_phases', False):
            owner.receipt['configuration']['phase_profile'] = 'owner-thread-inclusive-v1'
        identity = owner.bind_identity(repo=REPO, runtime_scene=out / 'runtime-scene.usda')
        write_json(out / 'model-identity.json', identity)
        remaining = args.max_wall_s - (time.monotonic() - started)
        for binding, actuator in zip(owner.layout.robots, owner.actuators):
            signals.checkpoint(persistent=True)
            controller = MobileBridgeController(robot_id=binding.robot_id, source=args.source,
                engine='newton', device=args.device, asset_sha256=admission['asset_sha256'],
                policy_sha256=args.policy_sha256, model_identity_sha256=binding.model_identity_sha256,
                support_contract=binding.support_contract(), physics_dt=.005, policy_dt=.020,
                **{k: admission['limits'][k] for k in CONTROLLER_LIMITS})
            view = SharedRobotView(owner, binding, controller.hello()['epoch'])
            steppers.append(MicroduckStepper(view, controller,
                MicroduckPolicy(args.policy, args.policy_sha256), actuator,
                max_steps=args.max_steps, max_wall_s=remaining,
                checkpoint=lambda: signals.checkpoint(persistent=True),
                **{k: admission['limits'][k] for k in FALL_LIMITS}))
        fleet = SharedMicroduckStepper(owner, steppers, layout=owner.layout)
        if getattr(args, 'profile_phases', False):
            from cascade.sim.microduck_timing import PhaseProfile, instrument_owner
            profile = PhaseProfile(out / 'timing.jsonl')
            instrument_owner(profile, owner, steppers)
        fleet.start()
        if args.serve_base_port is not None:
            from cascade.sim.microduck_admission import SharedEndpoints
            endpoints = SharedEndpoints(steppers, base_port=args.serve_base_port,
                                        max_jpeg_bytes=args.max_jpeg_bytes)
            result['scope'] = 'shared-scene bounded command candidate; physical outcomes unverified'
        before = owner.physics_clock
        warmup = owner.capture()
        if owner.physics_clock != before or (warmup['step'], warmup['sim_time_s']) != before:
            raise RuntimeError('camera warmup changed shared physics clock')
        write_json(out / 'runtime.json', {'backend': owner.receipt,
            'robots': {s.identity['robot_id']: s.controller.hello() for s in steppers}})
        def row(stream, value):
            with profile.span('write.' + Path(stream.name).name) if profile else nullcontext():
                stream.write(json.dumps(value, allow_nan=False, default=json_default) + '\n')
                stream.flush()
        with ((out / 'physics.jsonl').open('x') as physics,
              (out / 'policy.jsonl').open('x') as policies,
              (out / 'frames.jsonl').open('x') as frames,
              (out / 'support-probe.jsonl').open('x') as probes):
            attempts = {s.identity['robot_id']: -1 for s in steppers}
            i = 0
            while i < args.max_steps:
                signals.checkpoint(persistent=True)
                if time.monotonic() - started >= args.max_wall_s:
                    raise RuntimeError('shared episode wall deadline expired')
                with profile.attempt(owner) if profile else nullcontext() as timing:
                    try:
                        if endpoints is not None:
                            endpoints.admission.drain()
                        samples = fleet.tick()
                        if timing is not None:
                            timing['outcome'] = 'withheld' if samples is None else 'solved'
                    finally:
                        for stepper in steppers:
                            for record in stepper.policy_records:
                                robot = stepper.identity['robot_id']
                                if record['attempt'] > attempts[robot]:
                                    row(policies, record)
                                    attempts[robot] = record['attempt']
                    if samples is None:
                        result['withheld_ticks'] = fleet.withheld_ticks
                        continue
                    result['steps'] = i + 1
                    with profile.span('record.physics') if profile else nullcontext():
                        row(physics, {'step': owner.physics_clock[0], 'sim_time_s': owner.physics_clock[1],
                            'robots': {s.identity['robot_id']: {**samples[s.identity['robot_id']],
                                'bam': s.actuator.telemetry(), 'controller': s.controller.state()}
                                for s in steppers}})
                    if i == 0 or (i+1) % args.camera_every == 0 or i+1 == args.max_steps:
                        with profile.span('camera.overview') if profile else nullcontext():
                            probe = owner.support_probe()
                            row(probes, probe)
                            if probe.get('passed') is not True:
                                raise RuntimeError('shared native contact-force probe failed')
                            capture = owner.capture()
                            if (capture['step'], capture['sim_time_s']) != owner.physics_clock:
                                raise RuntimeError('overview capture does not match shared solve')
                            if endpoints is not None:
                                endpoints.publish_capture(capture)
                            import cv2
                            ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(capture['rgb'], cv2.COLOR_RGB2BGR))
                            if not ok:
                                raise RuntimeError('overview JPEG encoding failed')
                            path = out / 'frames' / f'overview_{capture["step"]:09d}.jpg'
                            with path.open('xb') as image:
                                image.write(jpeg.tobytes())
                            row(frames, {k:v for k,v in capture.items() if k != 'rgb'} | {
                                'file': path.relative_to(out).as_posix(), 'sha256': hashlib.sha256(jpeg).hexdigest(),
                                'scene_model_sha256': identity['scene_model_sha256'],
                                'physics_ticks_during_capture': 0})
                            if endpoints is not None and not endpoints.started:
                                marker = {'robots': endpoints.start(), 'physical_acceptance': False,
                                          'scene_model_sha256': identity['scene_model_sha256']}
                                write_json(out / 'BRIDGE_LISTENING.json', marker)
                                print('BRIDGE_LISTENING ' + json.dumps(marker), flush=True)
                    if (i+1) % 100 == 0:
                        print(json.dumps({'robots': args.robots, 'completed_steps': i+1}), flush=True)
                    i += 1
        result['completed'] = True
    except SignalRequest as exc:
        result.update(signal=exc.signum, error='signal/lifecycle shutdown')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        with signals.defer():
            for resource in (endpoints, profile, fleet if fleet is not None else owner):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as exc:
                        result['teardown_errors'].append(type(exc).__name__)
            result['robots'] = {s.identity['robot_id']: {'steps': s.steps,
                'policy_commits': s.policy_commits, 'last_state': s.controller.state()}
                for s in steppers if s.started}
            result['backend'] = owner.receipt if owner else None
            result['withheld_ticks'] = fleet.withheld_ticks if fleet else 0
            if profile is not None:
                result['phase_profile'] = {'file': 'timing.jsonl', 'attempts': profile.attempts,
                                          'errors': list(profile.errors)}
            result['wall_duration_s'] = time.monotonic() - started
            if result['teardown_errors'] or signals.signum is not None:
                result['completed'] = False
            result.update(receipt_revision=0, receipt_phase='before_sdk_shutdown')
            _resolve_outcome(result, signals)
            sys.path[:] = previous_path
            _persist(result, 'initial receipt', lambda: write_json(out / 'receipt.json', result))
            _persist(result, 'receipt reconciliation', lambda: _refresh_receipt(out, result, signals))
            _resolve_outcome(result, signals)
            if owner is not None:
                try:
                    returned = owner.shutdown(result['exit_code'])
                except Exception as exc:
                    returned = False
                    result['teardown_errors'].append(type(exc).__name__)
                result['sdk_close_returned'] = returned
                _persist(result, 'teardown', lambda: write_json(out / 'teardown.json', {'sdk_close_returned': returned}))
                _persist(result, 'final receipt reconciliation', lambda: _refresh_receipt(out, result, signals))
    return result


def main(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--robots', type=int, required=True)
    extra.add_argument('--spacing', type=float, required=True)
    extra.add_argument('--serve-base-port', type=int, default=None)
    extra.add_argument('--profile-phases', action='store_true')
    options, rest = extra.parse_known_args(argv)
    args = parse_args(rest)
    args.robots, args.spacing = options.robots, options.spacing
    args.serve_base_port = options.serve_base_port
    args.profile_phases = options.profile_phases
    bind_repo()
    from cascade.apps.signal_stop import StopSignals
    from cascade.sim.microduck_shared_native import placements
    placements(args.robots, args.spacing)
    if args.port != 0:
        raise ValueError('use --port 0 and opt in separately with --serve-base-port')
    if args.serve_base_port is not None and not 0 <= args.serve_base_port <= 65536-args.robots:
        raise ValueError('invalid per-robot loopback port range')
    if args.sdk_recipe is None:
        raise ValueError('shared native runtime requires an explicit SDK recipe')
    if not args.reuse_solved_read or not args.solver_cuda_graph or args.camera_rgbd:
        raise ValueError('shared foundation requires graph/read reuse and RGB overview only')
    admission = admit(args)
    for path in ('scripts/isaac_microduck_shared.py', 'src/cascade/sim/microduck_shared.py',
                 'src/cascade/sim/microduck_shared_native.py', 'src/cascade/sim/microduck_admission.py',
                 'src/cascade/sim/microduck_timing.py'):
        admission['source_sha256'][path] = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    if args.check_only:
        print(json.dumps({'ok': True, 'robots': args.robots, 'physical_acceptance': False}))
        return 0
    with StopSignals(protect_registration=True) as signals:
        result = run(args, admission, signals)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
