"""Optional measured placement aiming, with no motion or release authority.

The arm is still the only physics writer. A tool-to-body transform predicts
rigid transport in scratch; it creates no joint, weld or simulator constraint.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from ..safety.trajectory import PLAN_BUDGET_S
from .held_observation import HeldObservationInvalid
from .mujoco_withdrawal import coordinate_binding


def _backend(arm):
    from ..control.lazy_arm import LazyArm
    raw = getattr(arm, 'raw', None)
    return raw.__dict__.get('_arm') if isinstance(raw, LazyArm) else raw


def capture(runtime, q, *, deadline=None):
    """Opt in only with a configured native region; never activate LazyArm."""
    if runtime.cfg.arm.get('mj_delivery_area') is None:
        return None
    try:
        return MeasuredAim(runtime, q, deadline=deadline)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HeldObservationInvalid('placement attachment channel unavailable: '+str(exc)) from exc


class MeasuredAim:
    def __init__(self, runtime, q, *, deadline=None):
        from ..control.mujoco_arm import MujocoArm
        from ..sim.mujoco_placement import PlacementHistory, state_digest
        from ..sim.truth import _match_label
        self.runtime = runtime
        self.deadline = min(time.monotonic()+PLAN_BUDGET_S,
                            float('inf') if deadline is None else deadline)
        self.arm, self.raw = runtime.arm, _backend(runtime.arm)
        if (not isinstance(self.raw, MujocoArm) or self.raw._engine is None
                or self.raw._engine.kind != 'mjc' or self.arm.motion_planner is not None):
            raise HeldObservationInvalid('placement attachment requires the materialized native MuJoCo driver')
        self.world = self.raw.world
        self.history = getattr(self.world, 'placement_history', None)
        if (type(self.history) is not PlacementHistory or self.history.arm is not self.raw
                or self.history.world is not self.world):
            raise HeldObservationInvalid('placement attachment lacks its bound model/epoch observer')
        self.harness = self.arm.harness
        self._stream_started = False
        self.generation = self.harness._halt_generation
        self.cancellation = self.harness._observation_cancel_generation
        with self.world.lock:
            self.history.guard()
            self.channel = self._channel()
            self.coordinates = coordinate_binding(runtime, self.raw)
            self.epoch, self.identity = self.history.epoch, self.history.identity
            self.labels = self._labels()
            self.state = state_digest(self.world.data)
            self.q = self.world.data.qpos[self.raw._qadr].copy()
            if not np.array_equal(self.q, q):
                raise HeldObservationInvalid('placement attachment differs from the measured planning joints')
            name = _match_label(self.labels[1] or self.labels[0],
                                {n: n for n in self.history.bodies})
            if name is None:
                raise HeldObservationInvalid('placement attachment needs one observed held-body identity')
            self.name, (body, _, _) = name, self.history.bodies[name]
            self.history.geometry()  # Detached FK of the same final qpos, no forward/step.
            tcp = np.asarray(runtime.kin.fk(self.q), dtype=float)
            if (tcp.shape != (4, 4) or not np.isfinite(tcp).all()
                    or not np.array_equal(tcp[3], [0., 0., 0., 1.])
                    or not np.allclose(tcp[:3, :3].T @ tcp[:3, :3], np.eye(3), atol=1e-8, rtol=0)
                    or abs(np.linalg.det(tcp[:3, :3])-1.) > 1e-8):
                raise HeldObservationInvalid('placement attachment TCP frame is unavailable')
            pose = np.eye(4)
            pose[:3, :3] = self.history.scratch.xmat[body].reshape(3, 3)
            pose[:3, 3] = self.history.scratch.xpos[body]
            if np.linalg.norm(pose[:2, 3]-tcp[:2, 3]) > .12:
                raise HeldObservationInvalid('object is outside the existing held-offset sanity gate')
            self._attachment = np.linalg.inv(tcp) @ pose
            self.clock = float(self.world.data.time)
            if not np.isfinite(self.clock) or self.clock < 0:
                raise HeldObservationInvalid('placement attachment final-state clock is invalid')
            self.guard()

    def _labels(self):
        return (self.runtime.held_object, self.runtime._held_det_label)

    def _channel(self):
        return (self.runtime.arm, getattr(self.runtime.arm, 'raw', None),
                _backend(self.runtime.arm), self.raw._engine, self.raw.world,
                self.world.model, self.world.data, self.world.lock, self.raw._ctrl,
                getattr(self.world, 'placement_history', None), self.runtime.kin,
                self.runtime.cfg, self.runtime.arm.harness)

    @property
    def translation(self):
        return self._attachment[:3, 3].copy()

    def guard(self, runtime=None, q=None):
        from ..sim.mujoco_placement import state_digest
        if (runtime is not None and runtime is not self.runtime
                or any(a is not b for a, b in zip(self.channel, self._channel()))):
            raise HeldObservationInvalid('placement attachment channel binding changed')
        with self.world.lock:
            if (coordinate_binding(self.runtime, self.raw) != self.coordinates
                    or self.history.epoch != self.epoch or self.history.identity != self.identity
                    or self._labels() != self.labels):
                raise HeldObservationInvalid('placement attachment model/epoch/coordinate binding changed')
            try:
                self.history.guard()
            except ValueError as exc:
                raise HeldObservationInvalid('placement attachment model/state changed: '+str(exc)) from exc
            if (state_digest(self.world.data) != self.state
                    or q is not None and not np.array_equal(q, self.q)):
                raise HeldObservationInvalid('placement attachment measured state changed before consumption')
            self.harness._check_halt_generation(self.generation)
            if (self.harness._observation_cancel_generation != self.cancellation
                    or self.harness.estopped or self.harness.halted is not None):
                raise HeldObservationInvalid('placement attachment was cancelled')
            if time.monotonic() >= self.deadline:
                raise HeldObservationInvalid('placement attachment planning deadline expired')

    def receipt(self):
        return {'scope': 'measured rigid attachment prediction for geometric aiming only',
                'physical_task_verdict': False, 'release_authority': False,
                'object': self.name, 'epoch': self.epoch, 'model_sha256': self.identity,
                'final_state_time_s': self.clock, 'state_sha256': self.state,
                'generation': self.generation, 'cancellation_token': self.cancellation,
                'tool_to_body': self._attachment.tolist()}

    def transport_guard(self):
        """Keep identity/cancellation during legitimate motion, not old qpos."""
        if any(a is not b for a, b in zip(self.channel, self._channel())):
            raise HeldObservationInvalid('placement transport channel binding changed')
        with self.world.lock:
            if (coordinate_binding(self.runtime, self.raw) != self.coordinates
                    or self.history.epoch != self.epoch or self.history.identity != self.identity
                    or self._labels() != self.labels):
                raise HeldObservationInvalid('placement transport model/epoch binding changed')
            self.history.guard()
            clock = float(self.world.data.time)
            if not np.isfinite(clock) or clock < self.clock:
                raise HeldObservationInvalid('placement transport clock regressed')
            self.harness._check_halt_generation(self.generation)
            self.harness.check_motion_cancellation(self.cancellation)

    def before_stream(self):
        # This runs after both SafeArm and the backend read their start state.
        # The first stream consumes the original snapshot/deadline. Later
        # segments preserve its cancellation context through normal motion.
        if not self._stream_started:
            self.guard()
            self._stream_started = True
        else:
            self.transport_guard()

    def motion_arguments(self):
        return {'_halt_generation': self.generation, '_cancellation_token': self.cancellation,
                'before_stream': self.before_stream}

    def admit_release(self, withdrawal):
        from .mujoco_withdrawal import Withdrawal
        self.transport_guard()
        if (type(withdrawal) is not Withdrawal or withdrawal.arm is not self.arm
                or withdrawal.world is not self.world or withdrawal.generation != self.generation
                or withdrawal.cancellation != self.cancellation
                or withdrawal._bound_history is not self.history
                or withdrawal._bound_epoch != self.epoch
                or withdrawal.model is not self.world.model):
            raise HeldObservationInvalid('release did not retain the observed motion context')


def _geometry_binding(geometry):
    values = [geometry.target, geometry.rotation, geometry.hover, geometry.retreat_target]
    values += [None if item is None else item.q
               for item in (geometry.lift, geometry.pre, geometry.low, geometry.retreat)]
    return tuple(None if v is None else (np.asarray(v).dtype.str,
                                        np.asarray(v).shape, np.asarray(v).tobytes()) for v in values)


class RegionPlan:
    """Private in-call handoff; a report cannot reconstruct motion authority."""
    def __init__(self, aim, geometry, report):
        self.aim = aim
        self.geometry = deepcopy(geometry)
        self.report = deepcopy(report)
        self._geometry = _geometry_binding(self.geometry)
        self._target = tuple(report['target'])
        self._release_z = float(geometry.target[2])

    def consume(self, runtime, x, y, release_z, q):
        self.aim.guard(runtime, q)
        if ((x, y) != self._target or release_z != self._release_z
                or _geometry_binding(self.geometry) != self._geometry):
            raise HeldObservationInvalid('region placement target or prepared geometry changed')
        return self.geometry
