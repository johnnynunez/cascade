"""Manual MicroDuck control cadence; no Kit imports or pose writes.

Backend.read() returns one complete native snapshot; step() solves exactly one
step. Reads and camera RPCs never call tick(). CUDA/Kit acceptance is separate
from the software-only tests of this orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from cascade.apps.signal_stop import SignalRequest
from cascade.control.microduck_policy import POLICY_JOINTS, observation
from cascade.sim.microduck_state import body_frame_vectors


@dataclass
class StagedTick:
    sample: dict
    identity: dict
    policy_slot: bool
    command: np.ndarray
    candidate: tuple | None

    @property
    def prepared(self):
        return self.sample, self.identity, self.policy_slot


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def clock_tolerance(t):
    # Native time sums float32 dt in a double. Allow double summation ULPs,
    # NOT an accumulating nominal-decimal dt error or one whole physical tick.
    return max(1e-10, 32 * math.ulp(max(1., abs(t))))


def validated_sample(sample, *, min_height_m, max_height_m, max_tilt_rad):
    if type(sample.get('step')) is not int or sample['step'] < 0:
        raise ValueError('invalid native step')
    t = sample.get('sim_time')
    if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
        raise ValueError('invalid native time')
    if tuple(sample.get('joint_names', ())) != POLICY_JOINTS:
        raise ValueError('unexpected policy joint names/order')
    result = dict(sample)
    for key in ('q', 'dq'):
        a = sample.get(key)
        if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != (14,) or not np.isfinite(a).all():
            raise ValueError(f'{key} must be finite native float32[14]')
        result[key] = a.copy()
    for key, size in (('position', 3), ('orientation_wxyz', 4), ('linear_velocity', 3),
                      ('angular_velocity', 3), ('gravity_body', 3)):
        a = np.asarray(sample.get(key))
        if a.shape != (size,) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
            raise ValueError(f'invalid physics {key}')
        result[key] = a.astype(float).tolist()
    vectors = body_frame_vectors(result['orientation_wxyz'], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.])
    if not np.allclose(vectors['gravity_body'], result['gravity_body'], rtol=0, atol=2e-6):
        raise ValueError('incoherent gravity/orientation')
    contacts = sample.get('contacts')
    if not isinstance(contacts, (tuple, list)) or any(not isinstance(x, str) or not x or len(x) > 256 for x in contacts):
        raise ValueError('explicit observed contact names required')
    if len(set(contacts)) != len(contacts):
        raise ValueError('duplicate contacts')
    result['contacts'] = tuple(contacts)
    result['fallen'] = (not min_height_m <= result['position'][2] <= max_height_m
                        or math.acos(float(np.clip(-result['gravity_body'][2], -1., 1.))) > max_tilt_rad)
    return result


class MicroduckStepper:
    def __init__(self, backend, controller, policy, actuator, *, max_steps,
                 max_wall_s, min_height_m, max_height_m, max_tilt_rad,
                 clock=time.monotonic, checkpoint=None):
        self.backend, self.controller = backend, controller
        self.policy, self.actuator = policy, actuator
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError('max_steps must be a positive integer')
        self.max_steps = max_steps
        self.max_wall_s = positive(max_wall_s, 'max_wall_s')
        self.fall_limits = dict(min_height_m=positive(min_height_m, 'min_height_m'),
                                max_height_m=positive(max_height_m, 'max_height_m'),
                                max_tilt_rad=positive(max_tilt_rad, 'max_tilt_rad'))
        if min_height_m >= max_height_m or max_tilt_rad >= math.pi:
            raise ValueError('invalid fall limits')
        self.clock = clock
        if checkpoint is not None and not callable(checkpoint):
            raise ValueError('checkpoint must be callable or absent')
        self.checkpoint = checkpoint
        self.dt = positive(backend.dt, 'actual physics dt')
        if not math.isclose(self.dt, float(np.float32(.005)), rel_tol=0, abs_tol=1e-12):
            raise ValueError('MicroDuck requires the measured float32 0.005 timestep')
        self.steps = self.policy_evaluations = 0
        self.policy_attempts = self.policy_commits = 0
        self.policy_records = []  # current tick's attempts, including discards/failures
        self.policy_target_generation = None
        self.started = self.closed = False
        self.failure = ''

    def _read(self):
        read = getattr(self.backend, '_read_for_stepper', self.backend.read)
        return validated_sample(read(), **self.fall_limits)

    def _publish(self, sample, balance):
        value = {k: v for k, v in sample.items() if k != 'gravity_body'}
        value['q'], value['dq'] = value['q'].tolist(), value['dq'].tolist()
        value['balance_active'] = bool(balance and not value['fallen'])
        self.controller.publish(value)

    def fail(self, exc):
        self.failure = str(exc)
        reason = ('MicroDuck containment: ' + str(exc)).replace('\n', ' ')[:240]
        self.controller.fault(reason)
        self.backend.contain(reason)  # pause, NOT a physical successful stop

    def start(self):
        if self.started or self.closed:
            raise RuntimeError('stepper already started/closed')
        try:
            self.start_wall = self.clock()
            hello = self.controller.hello()
            self.identity = {k: hello[k] for k in ('robot_id', 'source', 'epoch', 'engine', 'device', 'asset_sha256', 'policy_sha256', 'model_identity_sha256')}
            self.last = self._read()
            self.initial_step = self.last['step']
            self.initial_time = self.last['sim_time']
            if self.last['fallen']:
                raise RuntimeError('initial pose violates explicit fall bounds')
            # HOME/FK reset is NOT a completed controlled physics sample. In
            # particular, bootstrap contacts still belong to the pre-HOME solve.
            # Do not publish them or admit commands until tick() completes.
            self.started = True
        except Exception as exc:
            self.fail(exc)
            raise

    def _control_snapshot(self, sim_time):
        command = np.zeros(13, dtype=np.float32)
        with self.controller._lock:
            command[:3] = self.controller.control_at(sim_time)
            identity = self.controller.hello()
            if any(identity[k] != v for k, v in self.identity.items()):
                raise RuntimeError('controller identity/epoch changed during episode')
        return command, identity

    def _check_wall(self, *, checkpoint=True):
        # Gate new work after a slow call returns; this cannot interrupt an
        # already-running native operation. An external process timeout is
        # still required to bound uncooperative inference/GPU calls.
        if checkpoint and self.checkpoint is not None:
            self.checkpoint()
        elapsed = self.clock() - self.start_wall
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed >= self.max_wall_s:
            raise RuntimeError('wall duration limit reached/clock regressed')

    def _policy_slot(self, sample, command, identity, *, stage_only=False, retry=0):
        """Commit at most ONE action per regular 50Hz slot.

        A stop that crosses inference invalidates that speculative computation.
        One bounded retry uses the same completed physical state and last
        committed RAW action, but the latest intent. No extra solve or policy
        cadence is inserted, and physical actuator delay history is not erased.
        Once the short memory-only commit has linearized, its solve is in flight;
        a later stop cannot undo it. No permission lock covers ONNX or GPU work.
        """
        # An opt-in handoff selects the network from the intent this attempt
        # observes; a retry re-selects with the latest intent. The previous action
        # is read after selection: it is the last committed raw action either way.
        select = getattr(self.policy, 'select', None)
        for attempt in range(1 if stage_only else 2):
            self._check_wall()
            selection = select(command) if select is not None else None
            previous = self.policy.previous_action
            obs = observation(sample['q'], sample['dq'], sample['angular_velocity'],
                              sample['gravity_body'], previous, command)
            record = {k: identity[k] for k in ('robot_id', 'source', 'epoch', 'generation', 'model_identity_sha256')}
            record.update(observation_step=sample['step'], observation_sim_time_s=sample['sim_time'],
                          observation=obs[0].tolist(), commands=command.tolist(), status='pending',
                          attempt=self.policy_attempts, policy_slot=self.steps // 4,
                          retry=retry+attempt, committed=False)
            if selection is not None:
                record.update(selection)
            self.policy_records.append(record)
            self.last_policy = record
            self.policy_attempts += 1
            try:
                action = self.policy.preview(obs)
                targets = self.policy.targets(action)
                self.policy_evaluations += 1
                record.update(raw_action=action.tolist(), targets=targets.tolist())
                clock_step, clock_time = self.backend.physics_clock
                if (type(clock_step) is not int or clock_step != sample['step']
                        or type(clock_time) not in (int, float) or not math.isfinite(clock_time)
                        or abs(clock_time - sample['sim_time']) > clock_tolerance(sample['sim_time'])):
                    raise RuntimeError('physics clock changed during inference; stale input not committed')
                if stage_only:
                    self._check_wall()
                    record['status'] = 'staged'
                    return action, targets, record
                # Only bounded policy/target history work is committed under
                # this lock. Uploads and BAM/Kit calls follow after releasing it.
                with self.controller._lock:
                    current_command, current = self._control_snapshot(sample['sim_time'])
                    self._check_wall()  # also covers waiting for the permission lock
                    if (current['generation'] == identity['generation']
                            and np.array_equal(current_command, command)):
                        record['commit_outcome'] = 'in_progress'
                        self.policy.commit(action)
                        self.policy_commits += 1
                        self.policy_target_generation = identity['generation']
                        record.update(status='evaluated', committed=True, commit_outcome='returned',
                                      commit_generation=identity['generation'])
                    else:
                        record.update(status='discarded', reason='command invalidated during inference',
                                      observed_generation=current['generation'])
                if record['committed']:
                    self._check_wall()
                    self.actuator.set_targets(targets)
                    return
                command, identity = current_command, current
            except SignalRequest as exc:
                # Unwind to the lifecycle owner, without committing a pending
                # candidate or treating an operator interrupt as physical rest.
                record.update(status='interrupted', signal=exc.signum)
                if record.get('commit_outcome') == 'in_progress':
                    # The copy may have happened before Python delivered the
                    # signal; never describe that ambiguous history as discarded.
                    record.update(committed=None, commit_outcome='unknown_due_to_interruption')
                raise
            except Exception as exc:
                record.update(status='failed', error=str(exc))
                raise
        raise RuntimeError('repeated command invalidation during inference; no action committed')

    def _tick_inputs(self, *, reset_records=True):
        if not self.started or self.closed or self.failure:
            raise RuntimeError('stepper not running; lifecycle restart required')
        if reset_records:
            self.policy_records = []
        self._check_wall()
        if self.steps >= self.max_steps:
            raise RuntimeError('step limit reached')
        if self.backend.dt != self.dt:
            raise RuntimeError('physics dt changed during episode')
        sample = self._read()
        if sample['fallen']:
            raise RuntimeError('physics state is fallen')
        if sample['step'] != self.last['step'] or sample['sim_time'] != self.last['sim_time']:
            raise RuntimeError('uncommanded physics step/time change')
        command, identity = self._control_snapshot(sample['sim_time'])
        policy_slot = self.steps % 4 == 0
        return sample, command, identity, policy_slot

    def _stage_tick(self, *, retry=0):
        """Shared owner only: no history, target or native control mutation."""
        sample, command, identity, policy_slot = self._tick_inputs(reset_records=retry == 0)
        candidate = self._policy_slot(sample, command, identity, stage_only=True, retry=retry) if policy_slot else None
        return StagedTick(sample, identity, policy_slot, command, candidate)

    def _prepare_tick(self, *, prepare_actuator=True):
        sample, command, identity, policy_slot = self._tick_inputs()
        if policy_slot:
            self._policy_slot(sample, command, identity)
        prepared = sample, identity, policy_slot
        if prepare_actuator:
            self._prepare_actuator(prepared)
        return prepared

    def _prepare_actuator(self, prepared, *, snapshot=None):
        sample, _, _ = prepared
        # Between 50Hz slots retain the last COMMITTED balancing target and
        # the model's physical delay, rather than cutting torque or silently
        # changing cadence. A stop immediately changes intent; it is not an
        # instantaneous physical-rest claim. Stamp the held target generation.
        # A shared-scene cohort passes one host snapshot of the world per step so
        # twelve adapters do not each sync the device for the same checks.
        self._check_wall()
        if snapshot is None:
            self.actuator.before_step(self.dt)
        else:
            self.actuator.before_step(self.dt, snapshot=snapshot)
        if self.backend.physics_clock != (sample['step'], sample['sim_time']):
            raise RuntimeError('policy/actuator advanced physics clock before solve')
        self._check_wall()

    def _validate_tick(self, prepared):
        sample, identity, policy_slot = prepared
        result = self._read()
        if result['step'] != sample['step'] + 1:
            raise RuntimeError('solve must advance exactly one native step')
        expected_time = self.initial_time + (self.steps + 1) * self.dt
        if abs(result['sim_time'] - expected_time) > clock_tolerance(expected_time):
            raise RuntimeError('physics time differs from frozen native dt')
        result.update(policy_target_generation=self.policy_target_generation,
                      permission_generation_at_sample=identity['generation'],
                      policy_target_held=not policy_slot)
        return result

    def _commit_tick(self, prepared, result):
        _, _, policy_slot = prepared
        if policy_slot:
            self.last_policy['first_step_after_commit'] = result['step']
        self.steps += 1
        self.last = result
        self._publish(result, True)
        return result

    def tick(self):
        if not self.started or self.closed or self.failure:
            raise RuntimeError('stepper not running; lifecycle restart required')
        try:
            prepared = self._prepare_tick()
            self.backend.step()
            result = self._validate_tick(prepared)
            return self._commit_tick(prepared, result)
        except Exception as exc:
            self.fail(exc)
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.controller.stop(latch=True)
            try:
                self.backend.contain('bounded lifecycle shutdown; no physical stop verdict')
            finally:
                self.backend.close()


def validate_render_times(times, sim_time_s):
    """Validate both annotators, never manufacture a capture timestamp.

    Renderer clocks quantize to ns and use decimal scene dt; native Newton
    accumulates float32 dt. Four float32 ULPs plus 2ns cover that representation
    difference, capped below a quarter of a physics step so a stale tick fails.
    """
    try:
        fabric = times['rpFabricTime']
        numerator = fabric['fabricFrameTimeNumerator']
        denominator = fabric['fabricFrameTimeDenominator']
        simulation = times['IsaacReadSimulationTime']['simulationTime']
        for value in (numerator, denominator, simulation, sim_time_s):
            if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError('invalid render clock')
        if denominator <= 0 or numerator < 0 or simulation < 0 or sim_time_s < 0:
            raise ValueError('invalid render clock bounds')
        tolerance = min(.005 / 4, 2e-9 + 4 * np.finfo(np.float32).eps * max(.005, abs(sim_time_s)))
        if any(abs(float(t) - sim_time_s) > tolerance for t in (numerator / denominator, simulation)):
            raise ValueError('stale render clock does not match completed physical step')
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError('missing or invalid render-product clocks') from exc


class FrameCache:
    """One immutable encoded overview frame; no Kit/encoder work on RPC threads."""
    def __init__(self, identity, *, max_jpeg_bytes, max_pixels, clock=time.monotonic):
        import threading
        self.identity = {k: identity[k] for k in ('robot_id', 'source', 'epoch', 'engine', 'device',
                                                 'asset_sha256', 'policy_sha256', 'model_identity_sha256')}
        for value, cap in ((max_jpeg_bytes, 4 * 1024**2), (max_pixels, 16 * 1024**2)):
            if type(value) is not int or not 0 < value <= cap:
                raise ValueError('invalid frame resource bound')
        self.max_jpeg_bytes, self.max_pixels = max_jpeg_bytes, max_pixels
        self.clock = clock
        self._lock = threading.Lock()
        self._frame = None
        self._closed = False

    def publish(self, rgb, *, step, sim_time_s, captured_at, render_times):
        import base64
        import cv2
        if type(step) is not int or step < 0:
            raise ValueError('invalid capture step')
        validate_render_times(render_times, sim_time_s)
        if type(captured_at) not in (int, float) or not math.isfinite(captured_at) or captured_at > self.clock():
            raise ValueError('invalid capture wall clock')
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or not 0 < rgb.shape[0] * rgb.shape[1] <= self.max_pixels):
            raise ValueError('capture must be bounded uint8 RGB')
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok or len(encoded) > self.max_jpeg_bytes:
            raise ValueError('JPEG encoding failed/exceeded bound')
        frame = {**self.identity, 'camera': 'overview', 'step': step, 'sim_time_s': float(sim_time_s),
                 'width': rgb.shape[1], 'height': rgb.shape[0],
                 'rgb_jpeg_b64': base64.b64encode(encoded).decode('ascii')}
        with self._lock:
            if self._closed:
                raise RuntimeError('frame cache closed')
            if self._frame is not None and (step <= self._frame['step'] or sim_time_s <= self._frame['sim_time_s']):
                raise ValueError('capture clocks must advance; cannot restamp/rejuvenate pixels')
            self._frame, self._captured_at = frame, captured_at

    def __call__(self, request):
        if request.get('camera') != 'overview':
            raise ValueError('unknown camera; explicit overview required')
        with self._lock:
            if self._closed or self._frame is None:
                raise RuntimeError('no captured frame available')
            age = self.clock() - self._captured_at
            if not math.isfinite(age) or age < 0:
                raise RuntimeError('capture age clock regressed')
            return {'frame': {**self._frame, 'producer_age_s': age}}

    def close(self):
        with self._lock:
            self._closed = True
            self._frame = None
