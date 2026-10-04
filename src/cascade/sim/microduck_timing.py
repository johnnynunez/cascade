"""Opt-in owner-thread spans; diagnostic clocks never advance physics."""
from contextlib import contextmanager
from collections import deque
from functools import wraps
import gc
from itertools import count
import json
import threading
import time


class _GCEvents:
    """Bounded passive callbacks, including collections outside owner attempts.

    Absolute times and event IDs survive batch boundaries; starts/stops are not
    paired here. Callback intervals include scheduling and other callbacks, not
    isolated CPU time. No locks, I/O or collector settings change in callbacks.
    """
    def __init__(self, clock, limit=4096):
        self.clock, self.limit = clock, limit
        self.pending, self.tickets = deque(maxlen=limit), count()
        self.active, self.registered = True, True
        self.drains = self.recorded = self.errors = 0
        self.initial = dict(enabled=gc.isenabled(), thresholds=gc.get_threshold())
        self.callback = self._record  # retain this exact bound method for removal
        gc.callbacks.append(self.callback)

    def _record(self, phase, info):
        ticket = next(self.tickets)
        if not self.active:
            return
        try:
            event = dict(id=ticket, phase=phase, error_type=None,
                monotonic_ns=self.clock(), thread_id=threading.get_ident(),
                generation=info['generation'], collected=info.get('collected'),
                uncollectable=info.get('uncollectable'))
        except BaseException as exc:
            event = dict(id=ticket, phase=phase, error_type=type(exc).__name__)
        # This queue is never replaced. A suspended callback cannot retain an
        # already serialized batch. CPython deque append/popleft and count next
        # are atomic; callbacks never acquire a lock or wait for the owner.
        self.pending.append(event)

    def drain(self):
        events = [self.pending.popleft() for _ in range(len(self.pending))]
        # The owner also consumes one ticket per drain. Everything issued before
        # this fence must be recorded, dropped by the bound, or still in flight.
        fence = next(self.tickets)
        observed = fence - self.drains
        self.drains += 1
        self.recorded += len(events)
        self.errors += sum(event['error_type'] is not None for event in events)
        self.registered &= any(item is self.callback for item in gc.callbacks)
        return dict(events=events, fence_id=fence, observed=observed,
            recorded=self.recorded, dropped_or_pending=observed-self.recorded,
            errors=self.errors, callback_registered=self.registered)

    def close(self):
        self.active = False
        try:
            batch = self.drain()
        finally:
            for index in range(len(gc.callbacks)-1, -1, -1):
                if gc.callbacks[index] is self.callback:
                    del gc.callbacks[index]
        return batch, dict((key, value) for key, value in batch.items() if key != 'events') | dict(
            event_limit=self.limit,
            accounting_complete=not batch['dropped_or_pending'] and not self.errors and self.registered,
            initial=self.initial, final=dict(enabled=gc.isenabled(), thresholds=gc.get_threshold()),
            scope='process callbacks; absolute clocks; intervals are not isolated CPU time')


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
        self.gc = _GCEvents(clock)

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
                current['gc'] = self.gc.drain()
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
            gc_batch, gc_summary = self.gc.close()
            self.stream.write(json.dumps(dict(kind='footer', attempts=self.attempts,
                last_profile_write=self.previous_write, errors=self.errors,
                gc=gc_batch, gc_summary=gc_summary,
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
