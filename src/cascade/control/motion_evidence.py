"""Bounded, opt-in copies of existing motion observations. Never control input."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
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


class Recording:
    def __init__(self, skill, arm, *, max_events=30000, max_bytes=16 * 1024 * 1024):
        self.id = uuid.uuid4().hex
        self.skill, self.arm = skill, arm
        self.max_events, self.max_bytes = max_events, max_bytes
        self.events, self.bytes = [], 0
        self.dropped = self.errors = 0
        self.first_error = None

    def error(self, exc):
        self.errors += 1
        if self.first_error is None:
            self.first_error = f'{type(exc).__name__}: {exc}'[:500]

    def record(self, kind, data):
        try:
            row = json.dumps({'kind': kind, 'monotonic_s': time.monotonic(),
                              'phase': _PHASE.get(), 'stream': _STREAM.get(),
                              'data': _plain(data)}, separators=(',', ':'), allow_nan=False)
            size = len(row.encode())
            if len(self.events) >= self.max_events or self.bytes + size > self.max_bytes:
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
            if self.errors or self.dropped:
                coverage['complete'] = False
            repo = Path(__file__).resolve().parents[3]
            source_files = ('scripts/isaac_bridge.py', 'src/cascade/sim/target_receipts.py',
                            'src/cascade/sim/bridge_client.py', 'src/cascade/control/isaac_arm.py',
                            'src/cascade/control/simulation_motion.py',
                            'src/cascade/control/motion_evidence.py', 'src/cascade/skills/runtime.py')
            source = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in source_files}
            document = {'version': 1, 'recording_id': self.id, 'pid': os.getpid(),
                        'skill': self.skill, 'arm': self.arm, 'events': events,
                        'submission_coverage': coverage,
                        'source_at_flush': {'scope': 'on-disk source; not imported-bytecode attestation',
                                            'repo': str(repo), 'sha256': source},
                        'logging_errors': self.errors, 'first_error': self.first_error,
                        'dropped_events': self.dropped, 'logging_complete': not self.errors and not self.dropped,
                        'scope': 'existing commands/observations only; submission is not motion or physical acceptance'}
            raw = (json.dumps(document, separators=(',', ':'), allow_nan=False) + '\n').encode()
            with path.open('xb') as stream:
                stream.write(raw)
            summary.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                           logging_complete=document['logging_complete'], submission_coverage=coverage)
        except Exception as exc:
            self.error(exc)
        summary.update(logging_errors=self.errors, first_error=self.first_error, dropped_events=self.dropped)
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
        event('skill_raised', exception_type=type(exc).__name__, error=str(exc))
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
            event('stream_raised', exception_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            _STREAM.reset(token)
    return wrapped
