"""Owner cadence for the Unitree H2 bundle: one PhysX solve per tick, inference every fourth.

No Kit import. ``backend`` is the single physics owner (``KitH2PhysxBackend`` or a
software double in tests):

* ``backend.dt`` — measured physics step (must equal the contract's 0.005 s);
* ``backend.physics_clock`` — ``(step, sim_time)`` of the last completed solve;
* ``backend.read()`` — one complete native snapshot of the completed state;
* ``backend.policy_due`` — whether the next ``control`` call runs inference
  (the deployment chain's own decimation counter, never assumed);
* ``backend.control(command)`` — hand the admitted planar twist to the deployment
  chain for the NEXT solve (inference and target write on a due tick, nothing on
  the others) and return its record;
* ``backend.step()`` — solve exactly one physics step, nothing else;
* ``backend.contain(reason)`` — pause after a fault (never a physical stop verdict).

The controller is the only command source. The published state is always a
completed solve; a fall or an inconsistent clock faults the episode instead of
being smoothed over. Mirrors ``MicroduckStepper`` deliberately so the verifier,
``SafeBase`` and the MOBILE wire stay unchanged.
"""
from __future__ import annotations

import math
import time

import numpy as np

from cascade.apps.signal_stop import SignalRequest
from cascade.control.h2_policy_contract import H2PolicyContract
from cascade.sim.microduck_state import body_frame_vectors
from cascade.sim.microduck_stepper import clock_tolerance, positive

# A pelvis this high above the training stand (1.015 m) is a thrown or exploded
# robot, not a walking one; the bound is a state-validity check, not a limit.
MAX_PELVIS_HEIGHT_M = 1.6


def validated_sample(sample: dict, contract: H2PolicyContract) -> dict:
    """Validate one native snapshot against the pinned contract; derive ``fallen``."""
    if type(sample.get('step')) is not int or sample['step'] < 0:
        raise ValueError('invalid native step')
    t = sample.get('sim_time')
    if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
        raise ValueError('invalid native time')
    if tuple(sample.get('joint_names', ())) != contract.policy_joint_names:
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
    solved = sample.get('support_contacts')
    if not isinstance(solved, (list, tuple)) or len(solved) > 4096 or any(not isinstance(c, dict) for c in solved):
        raise ValueError('explicit solved support contacts required (a list, possibly empty)')
    result['support_contacts'] = tuple(dict(c) for c in solved)
    height = result['position'][2]
    tilt = math.acos(float(np.clip(-result['gravity_body'][2], -1., 1.)))
    illegal = sorted(set(contacts) & set(contract.illegal_contact_bodies))
    result['fall_evidence'] = {'pelvis_height_m': height, 'tilt_rad': tilt, 'illegal_contacts': illegal,
                               'criteria': {'min_pelvis_height_m': contract.fall_pelvis_height_m,
                                            'max_tilt_rad': contract.fall_tilt_rad,
                                            'max_pelvis_height_m': MAX_PELVIS_HEIGHT_M}}
    result['fallen'] = (not contract.fall_pelvis_height_m <= height <= MAX_PELVIS_HEIGHT_M
                        or tilt > contract.fall_tilt_rad or bool(illegal))
    return result


class H2Stepper:
    def __init__(self, backend, controller, contract: H2PolicyContract, *, max_steps, max_wall_s,
                 clock=time.monotonic, checkpoint=None):
        self.backend, self.controller, self.contract = backend, controller, contract
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError('max_steps must be a positive integer')
        self.max_steps = max_steps
        self.max_wall_s = positive(max_wall_s, 'max_wall_s')
        self.clock = clock
        if checkpoint is not None and not callable(checkpoint):
            raise ValueError('checkpoint must be callable or absent')
        self.checkpoint = checkpoint
        self.dt = positive(backend.dt, 'actual physics dt')
        if not math.isclose(self.dt, contract.physics_dt, rel_tol=0, abs_tol=1e-9):
            raise ValueError('H2 requires the bundle physics timestep')
        self.steps = self.policy_evaluations = self.policy_attempts = 0
        self.policy_records = []  # current tick's control records
        self.last = None
        self.start_wall = None
        self.started = self.closed = False
        self.failure = ''

    def _read(self):
        return validated_sample(self.backend.read(), self.contract)

    def _publish(self, sample):
        from cascade.control.mobile_support import SupportObservation
        value = {k: v for k, v in sample.items() if k not in ('gravity_body', 'fall_evidence', 'support_contacts')}
        value['q'], value['dq'] = value['q'].tolist(), value['dq'].tolist()
        # Complete solved reactions of this very solve, bound to the episode identity.
        value['support'] = SupportObservation(
            version=1, status='known', reason='PhysX solved normal reactions between every robot link and the ground',
            epoch=self.identity['epoch'], step=sample['step'], sim_time_s=sample['sim_time'],
            model_identity_sha256=self.identity['model_identity_sha256'], contacts=sample['support_contacts'])
        # The deployment chain is alive and the robot is not fallen. This is the
        # producer's attestation that the balance policy keeps running; it proves
        # neither rest nor stability, the independent verifier does.
        value['balance_active'] = not sample['fallen']
        self.controller.publish(value)

    def fail(self, exc):
        self.failure = str(exc)
        reason = ('H2 containment: ' + str(exc)).replace('\n', ' ')[:240]
        self.controller.fault(reason)
        self.backend.contain(reason)  # pause, NOT a physical successful stop

    def start(self):
        if self.started or self.closed:
            raise RuntimeError('stepper already started/closed')
        try:
            self.start_wall = self.clock()
            hello = self.controller.hello()
            self.identity = {k: hello[k] for k in ('robot_id', 'source', 'epoch', 'engine', 'device',
                                                   'asset_sha256', 'policy_sha256', 'model_identity_sha256')}
            self.last = self._read()
            self.initial_step = self.last['step']
            self.initial_time = self.last['sim_time']
            if self.last['fallen']:
                raise RuntimeError('initial pose violates the training fall criteria: '
                                   + str(self.last['fall_evidence']))
            # The default-state reset is not a completed controlled sample; nothing is
            # published and no command is admitted until tick() completes a solve.
            self.started = True
        except Exception as exc:
            self.fail(exc)
            raise

    def _control_snapshot(self, sim_time):
        command = np.zeros(3, dtype=np.float32)
        with self.controller._lock:
            command[:] = self.controller.control_at(sim_time)
            identity = self.controller.hello()
            if any(identity[k] != v for k, v in self.identity.items()):
                raise RuntimeError('controller identity/epoch changed during episode')
        return command, identity

    def _check_wall(self):
        if self.checkpoint is not None:
            self.checkpoint()
        elapsed = self.clock() - self.start_wall
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed >= self.max_wall_s:
            raise RuntimeError('wall duration limit reached/clock regressed')

    def _prepare_tick(self):
        if not self.started or self.closed or self.failure:
            raise RuntimeError('stepper not running; lifecycle restart required')
        self.policy_records = []
        self._check_wall()
        if self.steps >= self.max_steps:
            raise RuntimeError('step limit reached')
        if self.backend.dt != self.dt:
            raise RuntimeError('physics dt changed during episode')
        sample = self._read()
        if sample['fallen']:
            raise RuntimeError('physics state is fallen: ' + str(sample['fall_evidence']))
        if sample['step'] != self.last['step'] or sample['sim_time'] != self.last['sim_time']:
            raise RuntimeError('uncommanded physics step/time change')
        command, identity = self._control_snapshot(sample['sim_time'])
        due = bool(self.backend.policy_due)
        if due != (self.steps % self.contract.decimation == 0):
            raise RuntimeError('deployment chain decimation drifted from the owner cadence')
        record = {k: identity[k] for k in ('robot_id', 'source', 'epoch', 'generation', 'model_identity_sha256')}
        record.update(observation_step=sample['step'], observation_sim_time_s=sample['sim_time'],
                      commands=command.tolist(), policy_slot=self.steps // self.contract.decimation,
                      inference=due, attempt=self.policy_attempts, status='pending')
        self.policy_records.append(record)
        if due:
            self.policy_attempts += 1
        self._check_wall()
        try:
            # The deployment chain reads the same completed state the controller
            # snapshot belongs to; a clock change here means it stepped physics.
            chain = self.backend.control(command)
            if self.backend.physics_clock != (sample['step'], sample['sim_time']):
                raise RuntimeError('control advanced the physics clock before the solve')
        except SignalRequest as exc:
            record.update(status='interrupted', signal=exc.signum)
            raise
        except Exception as exc:
            record.update(status='failed', error=str(exc))
            raise
        if isinstance(chain, dict):
            record.update(chain)
        if due:
            self.policy_evaluations += 1
        record['status'] = 'evaluated' if due else 'held'
        self._check_wall()
        return sample, identity, due

    def _validate_tick(self, prepared):
        sample, identity, due = prepared
        result = self._read()
        if result['step'] != sample['step'] + 1:
            raise RuntimeError('solve must advance exactly one native step')
        expected_time = self.initial_time + (self.steps + 1) * self.dt
        if abs(result['sim_time'] - expected_time) > clock_tolerance(expected_time) + 1e-9:
            raise RuntimeError('physics time differs from the frozen native dt')
        result.update(permission_generation_at_sample=identity['generation'], policy_target_held=not due)
        return result

    def tick(self):
        if not self.started or self.closed or self.failure:
            raise RuntimeError('stepper not running; lifecycle restart required')
        try:
            prepared = self._prepare_tick()
            self.backend.step()
            result = self._validate_tick(prepared)
            self.steps += 1
            self.last = result
            self._publish(result)  # a fallen result faults inside publish, after it is recorded
            return result
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
