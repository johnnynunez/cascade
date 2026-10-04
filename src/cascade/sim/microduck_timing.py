"""Opt-in owner-thread spans; diagnostic clocks never advance physics."""
from contextlib import contextmanager
from functools import wraps
import json
import threading
import time


class PhaseProfile:
    """Inclusive, nested timings; overlapping durations must not be summed.

    Only the owning thread records. Wrappers outside an attempt are transparent.
    Each row reports the preceding profile write cost; the footer reports the
    final one. Footer/close overhead itself is excluded, never called physics.
    """
    def __init__(self, path, *, clock=time.monotonic_ns):
        self.stream = path.open('x')
        self.clock, self.thread = clock, threading.get_ident()
        self.current = None
        self.attempts = 0
        self.previous_write = None
        self.restores, self.errors = [], []
        self.closed = False

    @contextmanager
    def span(self, phase, robot=None):
        current = self.current
        if current is None or threading.get_ident() != self.thread:
            yield
            return
        item = dict(id=len(current['spans']), parent=current['stack'][-1] if current['stack'] else None,
                    phase=phase, robot_id=robot, start_monotonic_ns=self.clock(), error_type=None)
        current['spans'].append(item)
        current['stack'].append(item['id'])
        try:
            yield
        except BaseException as exc:
            item['error_type'] = type(exc).__name__
            raise
        finally:
            item['end_monotonic_ns'] = self.clock()
            item['duration_ns'] = item['end_monotonic_ns'] - item['start_monotonic_ns']
            current['stack'].pop()

    def wrap(self, obj, name, phase, robot=None):
        original = getattr(obj, name)
        local = name in vars(obj)
        @wraps(original)
        def timed(*args, **kwargs):
            with self.span(phase, robot):
                return original(*args, **kwargs)
        setattr(obj, name, timed)
        self.restores.append((obj, name, original, local))

    @contextmanager
    def attempt(self, owner):
        if self.current is not None or self.closed or threading.get_ident() != self.thread:
            raise RuntimeError('profile attempt requires its idle owner thread')
        current = dict(kind='attempt', attempt=self.attempts, clock_before=owner.physics_clock,
            start_monotonic_ns=self.clock(), outcome='incomplete', error_type=None, spans=[], stack=[])
        self.attempts += 1
        self.current = current
        primary = None
        try:
            yield current
        except BaseException as exc:
            primary = exc
            current.update(outcome='error', error_type=type(exc).__name__)
            raise
        finally:
            self.current = None
            try:
                current['end_monotonic_ns'] = self.clock()
                current['duration_ns'] = current['end_monotonic_ns'] - current['start_monotonic_ns']
                current['clock_after'] = owner.physics_clock
                current.pop('stack')
                current['previous_profile_write'] = self.previous_write
                started = self.clock()
                self.stream.write(json.dumps(current, allow_nan=False) + '\n')
                self.stream.flush()
                self.previous_write = dict(attempt=current['attempt'], duration_ns=self.clock()-started)
            except BaseException as exc:
                self.errors.append(type(exc).__name__)
                if primary is None:
                    raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        first = None
        for obj, name, original, local in reversed(self.restores):
            try:
                if local:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
            except BaseException as exc:
                self.errors.append(type(exc).__name__)
                first = first or exc
        try:
            self.stream.write(json.dumps(dict(kind='footer', attempts=self.attempts,
                last_profile_write=self.previous_write, errors=self.errors,
                scope='owner-thread inclusive spans; not additive; footer/close cost excluded')) + '\n')
            self.stream.flush()
        except BaseException as exc:
            self.errors.append(type(exc).__name__)
            first = first or exc
        finally:
            try:
                self.stream.close()
            except BaseException as exc:
                self.errors.append(type(exc).__name__)
                first = first or exc
        if first is not None:
            raise first


def instrument_owner(profile, owner, steppers):
    """Install only in the explicitly profiled shared CLI, after model binding."""
    from . import microduck_contact_support as support, microduck_shared_native as native

    profile.wrap(support, 'read_support', 'support.decode')
    profile.wrap(native, 'read_native_states', 'native.capture')
    for name, phase in (('_read_completed_scene', 'completed.read'), ('step', 'solve'),
                        ('capture', 'camera.capture'), ('support_probe', 'support.probe')):
        profile.wrap(owner, name, phase)
    for stepper in steppers:
        robot = stepper.backend.binding.robot_id
        for name, phase in (('_stage_tick', 'policy.prepare'), ('_validate_tick', 'completed.validate'),
                            ('_commit_tick', 'publication')):
            profile.wrap(stepper, name, phase, robot)
        profile.wrap(stepper.actuator, 'set_targets', 'bam.targets', robot)
        profile.wrap(stepper.actuator, 'before_step', 'bam.before_step', robot)
