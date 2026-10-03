"""Bounded, opt-in copies of existing motion observations. Never control input."""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from ..sim.target_receipts import vector

_ACTIVE = ContextVar('motion_evidence', default=None)
_PHASE = ContextVar('motion_evidence_phase', default=None)
_STREAM = ContextVar('motion_evidence_stream', default=None)


def submission_coverage(events):
    """Post-action diagnostic matching only; never change the task verdict."""
    requests, acknowledgements, observed = {}, {}, {}
    problems = []
    try:
        for row in events:
            data = row['data']
            if row['kind'] == 'target_request':
                key = data['command_id']
                if not isinstance(key, str) or not key or key in requests:
                    raise ValueError('missing or reused command id')
                requests[key] = data
            elif row['kind'] == 'target_ack':
                key = data['command_id']
                ack = data['queued_receipt']
                request = requests[key]
                if (not isinstance(ack, dict) or type(ack.get('version')) is not int or ack['version'] != 1
                        or type(ack.get('sequence')) is not int or ack['sequence'] <= 0
                        or ack.get('command_id') != key or ack.get('status') != 'queued'
                        or not ack.get('instance') or not ack.get('epoch') or not ack.get('robot_id')
                        or ack['robot_id'] != request['robot_id']
                        or data['endpoint'] != request['endpoint']
                        or ack.get('requested_asset_q') != vector(request['q_asset'], '<f8')):
                    raise ValueError('missing or mismatched queued receipt')
                acknowledgements[key] = ack
            elif row['kind'] == 'isaac_state':
                packet = data.get('target_receipts')
                if not isinstance(packet, dict):
                    continue
                coverage = packet['coverage']
                if any(coverage.get(k, 0) for k in ('evicted', 'diagnostic_errors', 'superseded_before_setter_receipt')):
                    problems.append('producer reports lost or superseded evidence')
                write = packet.get('last_written')
                if not isinstance(write, dict) or write.get('command_id') not in acknowledgements:
                    continue
                key = write['command_id']
                ack, request = acknowledgements[key], requests[key]
                first = write['first_write']
                clock = first['prior_completed_update']
                if (type(packet.get('version')) is not int or packet['version'] != 1
                        or data['endpoint'] != request['endpoint']
                        or any(write.get(k) != ack[k] or packet.get(k) != ack[k]
                               for k in ('instance', 'epoch', 'robot_id'))
                        or write['sequence'] != ack['sequence']
                        or first['status'] != 'setter_returned'
                        or first['arm_target'] != vector(request['q_asset'], '<f4')
                        or first['full_target'] != vector(first['full_target']['values'], '<f4')
                        or len(first['joint_names']) != len(request['q_asset'])
                        or not isinstance(clock, dict) or clock.get('engine') != 'physx'
                        or clock.get('clock') != 'SimulationManager'
                        or clock.get('epoch') != ack['epoch'] or clock.get('robot_id') != ack['robot_id']
                        or type(clock.get('physics_step')) is not int or clock['physics_step'] < 0
                        or type(clock.get('sim_time')) not in (int, float) or clock['sim_time'] < 0
                        or clock.get('physics_dt_s') != data['physics_clock']['physics_dt_s']
                        or data['physics_clock']['epoch'] != ack['epoch']
                        or data['physics_clock']['robot_id'] != ack['robot_id']
                        or data['physics_clock']['physics_step'] < clock['physics_step']
                        or first['producer_monotonic_s'] > write['last_write']['producer_monotonic_s']
                        or write['write_count'] < 1 or write['submission_failures']):
                    raise ValueError('missing or mismatched first setter receipt')
                if key in observed and observed[key] != first:
                    raise ValueError('first setter receipt changed on a repeated write')
                observed[key] = first
    except Exception as exc:
        problems.append(f'{type(exc).__name__}: {exc}')
    missing = sorted(set(requests) - set(observed))
    return {'complete': bool(requests) and not missing and not problems,
            'requested': len(requests), 'acknowledged': len(acknowledgements),
            'first_setter_observed': len(observed), 'missing_command_ids': missing,
            'problems': sorted(set(problems)), 'physical_acceptance': False}


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())  # Exact floats, not TraceLogger's rounded arrays.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _message(exc):
    try:
        return str(exc), True
    except BaseException:
        # A diagnostic formatter must not replace the original action exception.
        return '<unavailable: exception message formatting failed>', False


def _bounded_error(exc):
    return f'{type(exc).__name__[:96]}: {_message(exc)[0][:400]}'[:500]


def _exception_fields(exc):
    message, available = _message(exc)
    if not available and _ACTIVE.get() is not None:
        _ACTIVE.get().error(RuntimeError('exception message formatting failed'))
    return {'exception_type': type(exc).__name__, 'error': message}


class _LifecycleJournal:
    """A bounded projection of existing events, never submission/physical proof."""
    KINDS = frozenset(('skill_begin', 'skill_end', 'skill_raised', 'phase_begin',
                       'phase_end', 'stream_begin', 'stream_returned', 'stream_raised'))
    FIELD_LIMITS = {'phase': 96, 'stream': 64, 'data.exception_type': 96, 'data.error': 512}

    def __init__(self, max_events, max_bytes):
        self.max_events, self.max_bytes = max_events, max_bytes
        self.rows = deque()
        self.bytes = self.observed = self.evicted = self.dropped = self.errors = self.truncated = 0
        self.first_error = None

    @property
    def incomplete(self):
        return bool(self.evicted or self.dropped or self.errors or self.truncated)

    def record(self, kind, captured, phase, stream, data):
        if kind not in self.KINDS:
            return
        self.observed += 1
        if not self.max_events or not self.max_bytes:
            self.dropped += 1
            return
        try:
            if type(captured) not in (int, float) or not math.isfinite(captured):
                raise ValueError('nonfinite or invalid lifecycle capture time')
            truncated = []

            def text(value, key):
                if value is None and key in ('phase', 'stream'):
                    return None
                if type(value) is not str:
                    raise TypeError(f'invalid lifecycle {key}')
                limit = self.FIELD_LIMITS[key]
                if len(value) > limit:
                    truncated.append(key)
                return value[:limit]

            row = {'kind': kind, 'monotonic_s': captured,
                   'phase': text(phase, 'phase'), 'stream': text(stream, 'stream'),
                   'data': {}, 'truncated_fields': truncated}
            if kind == 'stream_returned' and 'settled' in data:
                if type(data['settled']) is not bool:
                    raise TypeError('invalid lifecycle settled flag')
                row['data']['settled'] = data['settled']
            if kind in ('skill_raised', 'stream_raised'):
                for key in ('exception_type', 'error'):
                    if key in data:
                        row['data'][key] = text(data[key], 'data.' + key)
            self.truncated += bool(truncated)
            encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
            size = len(encoded.encode())
            if size > self.max_bytes:
                self.dropped += 1
                return  # An oversized row must not evict the useful existing tail.
            while len(self.rows) >= self.max_events or self.bytes + size > self.max_bytes:
                _, previous_size = self.rows.popleft()
                self.bytes -= previous_size
                self.evicted += 1
            self.rows.append((encoded, size))
            self.bytes += size
        except Exception as exc:
            self.errors += 1
            if self.first_error is None:
                self.first_error = _bounded_error(exc)
            raise

    def snapshot(self, *, include_events=False):
        result = {'version': 1, 'diagnostic_only': True, 'physical_acceptance': False,
                  'enabled': bool(self.max_events and self.max_bytes),
                  'max_events': self.max_events, 'max_bytes': self.max_bytes,
                  'retained_events': len(self.rows), 'retained_bytes': self.bytes,
                  'observed_events': self.observed, 'evicted_events': self.evicted,
                  'dropped_events': self.dropped, 'logging_errors': self.errors,
                  'first_error': self.first_error, 'truncated_events': self.truncated,
                  'field_limits_chars': dict(self.FIELD_LIMITS),
                  'complete': bool(self.max_events and self.max_bytes) and not self.incomplete,
                  'scope': 'bounded lifecycle projection; return/end does not establish physical closure'}
        if include_events:
            result['events'] = [json.loads(row) for row, _ in self.rows]
        return result


class Recording:
    def __init__(self, skill, arm, *, max_events=30000, max_bytes=16 * 1024 * 1024):
        self.id = uuid.uuid4().hex
        self.skill, self.arm = skill, arm
        self.max_events, self.max_bytes = max_events, max_bytes
        reserve_events = max(0, min(128, max_events // 4))
        reserve_bytes = max(0, min(64 * 1024, max_bytes // 4))
        if not reserve_events or not reserve_bytes:
            reserve_events = reserve_bytes = 0
        self.lifecycle = _LifecycleJournal(reserve_events, reserve_bytes)
        self.prefix_max_events = max_events - reserve_events
        self.prefix_max_bytes = max_bytes - reserve_bytes
        self.events, self.bytes = [], 0
        self.dropped = self.errors = 0
        self.first_error = None

    def error(self, exc):
        self.errors += 1
        if self.first_error is None:
            self.first_error = _bounded_error(exc)

    def record(self, kind, data):
        try:
            captured, phase, stream = time.monotonic(), _PHASE.get(), _STREAM.get()
        except Exception as exc:
            self.error(exc)
            return
        try:
            self.lifecycle.record(kind, captured, phase, stream, data)
        except Exception as exc:
            self.error(exc)
        try:
            row = json.dumps({'kind': kind, 'monotonic_s': captured,
                              'phase': phase, 'stream': stream,
                              'data': _plain(data)}, separators=(',', ':'), allow_nan=False)
            size = len(row.encode())
            if len(self.events) >= self.prefix_max_events or self.bytes + size > self.prefix_max_bytes:
                self.dropped += 1
                return
            self.events.append(row)
            self.bytes += size
        except Exception as exc:
            self.error(exc)

    def finish(self, directory):
        summary = {'recording_id': self.id, 'logging_complete': False,
                   'submission_coverage': 'not_adjudicated', 'physical_acceptance': False}
        try:
            path = Path(directory) / (self.id + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            events = [json.loads(r) for r in self.events]
            coverage = submission_coverage(events)
            logging_complete = not (self.errors or self.dropped or self.lifecycle.incomplete)
            if not logging_complete:
                coverage['complete'] = False
            repo = Path(__file__).resolve().parents[3]
            source_files = ('scripts/isaac_bridge.py', 'src/cascade/sim/target_receipts.py',
                            'src/cascade/sim/bridge_client.py', 'src/cascade/control/isaac_arm.py',
                            'src/cascade/control/simulation_motion.py',
                            'src/cascade/control/motion_evidence.py', 'src/cascade/skills/runtime.py')
            source = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in source_files}
            document = {'version': 1, 'recording_id': self.id, 'pid': os.getpid(),
                        'skill': self.skill, 'arm': self.arm, 'events': events,
                        'lifecycle_journal': self.lifecycle.snapshot(include_events=True),
                        'buffer_budget': {'max_events': self.max_events, 'max_bytes': self.max_bytes,
                                          'prefix_max_events': self.prefix_max_events,
                                          'prefix_max_bytes': self.prefix_max_bytes,
                                          'prefix_retained_bytes': self.bytes,
                                          'scope': 'serialized event UTF-8 bytes; document metadata excluded'},
                        'submission_coverage': coverage,
                        'source_at_flush': {'scope': 'on-disk source; not imported-bytecode attestation',
                                            'repo': str(repo), 'sha256': source},
                        'logging_errors': self.errors, 'first_error': self.first_error,
                        'dropped_events': self.dropped, 'logging_complete': logging_complete,
                        'scope': 'existing commands/observations only; submission is not motion or physical acceptance'}
            raw = (json.dumps(document, separators=(',', ':'), allow_nan=False) + '\n').encode()
            with path.open('xb') as stream:
                stream.write(raw)
            summary.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                           logging_complete=document['logging_complete'], submission_coverage=coverage)
        except Exception as exc:
            self.error(exc)
        summary.update(logging_errors=self.errors, first_error=self.first_error, dropped_events=self.dropped)
        summary['lifecycle_journal'] = self.lifecycle.snapshot()
        return summary


def active():
    return _ACTIVE.get() is not None


def event(kind, **data):
    recording = _ACTIVE.get()
    if recording is not None:
        try:
            recording.record(kind, data)
        except Exception as exc:
            recording.error(exc)


def command_id():
    return uuid.uuid4().hex if active() else None


@contextmanager
def record_skill(skill, arm, trace_context, *, enabled):
    directory = os.environ.get('CASCADE_MOTION_EVIDENCE_DIR') if enabled else None
    if not directory or active():
        yield
        return
    recording = Recording(skill, arm)
    token = _ACTIVE.set(recording)
    try:
        event('skill_begin', skill=skill, arm=arm)
        yield
    except BaseException as exc:
        event('skill_raised', **_exception_fields(exc))
        raise
    finally:
        event('skill_end')
        _ACTIVE.reset(token)
        # No hardware access and no flush while fn()/watcher-paused context runs.
        trace_context['motion_evidence'] = recording.finish(directory)


def phase(name):
    """Host code only: tool arguments cannot select a diagnostic phase."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not active():
                return fn(*args, **kwargs)
            token = _PHASE.set(name)
            event('phase_begin', name=name)
            try:
                return fn(*args, **kwargs)
            finally:
                event('phase_end', name=name)
                _PHASE.reset(token)
        return wrapped
    return decorate


def stream(fn):
    @wraps(fn)
    def wrapped(self, start, target, *args, **kwargs):
        if not active():
            return fn(self, start, target, *args, **kwargs)
        token = _STREAM.set(uuid.uuid4().hex)
        event('stream_begin', q_start=start, q_goal=target,
              clock=self.last_state.physics_clock if self.last_state is not None else None)
        try:
            result = fn(self, start, target, *args, **kwargs)
            event('stream_returned', settled=result)
            return result
        except BaseException as exc:
            event('stream_raised', **_exception_fields(exc))
            raise
        finally:
            _STREAM.reset(token)
    return wrapped
