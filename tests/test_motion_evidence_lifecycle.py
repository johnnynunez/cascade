"""Diagnostic tail retention cannot manufacture submission or physical proof."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control import motion_evidence as evidence
from cascade.control import simulation_motion
from cascade.sim.target_receipts import TargetReceipts
from cascade.types import SafetyViolation
from test_motion_evidence import Sim


def finish(recording, tmp_path):
    summary = recording.finish(tmp_path)
    return summary, json.loads(Path(summary['path']).read_text())


def byte_count(rows):
    return sum(len(json.dumps(r, separators=(',', ':'), allow_nan=False).encode()) for r in rows)


def assert_bounds(document, max_events, max_bytes):
    tail = document['lifecycle_journal']
    prefix = document['events']
    assert len(prefix) + len(tail['events']) <= max_events
    assert byte_count(prefix) + byte_count(tail['events']) <= max_bytes
    assert tail['retained_bytes'] == byte_count(tail['events'])
    assert tail['retained_events'] == len(tail['events'])
    assert tail['diagnostic_only'] is True and tail['physical_acceptance'] is False
    assert document['buffer_budget']['max_events'] == max_events
    assert document['buffer_budget']['max_bytes'] == max_bytes


def test_saturated_prefix_retains_return_exception_phase_and_skill_end(tmp_path):
    rec = evidence.Recording('pick', 'arm', max_events=64, max_bytes=16384)
    rec.record('skill_begin', {})
    for _ in range(100):
        rec.record('isaac_state', {'target_receipts': 'x' * 1000})
    prefix_before = list(rec.events)
    for kind, data in [('stream_returned', {'settled': False}),
                       ('stream_raised', {'exception_type': 'ValueError', 'error': 'original'}),
                       ('phase_end', {}), ('skill_raised', {'exception_type': 'ValueError', 'error': 'original'}),
                       ('skill_end', {})]:
        rec.record(kind, data)
    summary, doc = finish(rec, tmp_path)
    assert rec.events[:len(prefix_before)] == prefix_before
    tail = doc['lifecycle_journal']
    assert [r['kind'] for r in tail['events']][-5:] == [
        'stream_returned', 'stream_raised', 'phase_end', 'skill_raised', 'skill_end']
    assert tail['events'][-5]['data']['settled'] is False
    assert tail['events'][-4]['data']['error'] == 'original'
    assert summary['logging_complete'] is False
    assert summary['submission_coverage']['complete'] is False
    assert summary['dropped_events'] > 0
    assert 'events' not in summary['lifecycle_journal']  # Compact trace receipt.
    assert_bounds(doc, 64, 16384)


@pytest.mark.parametrize('max_events,max_bytes', [(0, 0), (1, 4096), (3, 3), (4, 4), (8, 2048), (40, 4096), (600, 262144)])
def test_total_budget_scales_and_tail_evictions_are_counted(tmp_path, max_events, max_bytes):
    rec = evidence.Recording('test', 'arm', max_events=max_events, max_bytes=max_bytes)
    for i in range(200):
        rec.record('phase_end', {'ignored_large_field': 'x'*1000, 'index': i})
    rec.record('skill_end', {})
    summary, doc = finish(rec, tmp_path)
    tail = doc['lifecycle_journal']
    assert tail['observed_events'] == 201
    assert tail['observed_events'] == (tail['retained_events'] + tail['evicted_events'] +
                                       tail['dropped_events'] + tail['logging_errors'])
    assert tail['max_events'] <= min(128, max_events//4)
    assert tail['max_bytes'] <= min(65536, max_bytes//4)
    assert not tail['complete'] and not summary['logging_complete']
    if tail['events']:
        assert tail['events'][-1]['kind'] == 'skill_end'
    assert_bounds(doc, max_events, max_bytes)


def test_default_reservation_is_inside_old_limits_and_does_not_duplicate_targets(tmp_path):
    rec = evidence.Recording('test', 'arm')
    rec.record('stream_begin', {'q_start': np.array([.123456789012345]),
                                'q_goal': np.arange(10000), 'clock': {'large': list(range(10000))}})
    _, doc = finish(rec, tmp_path)
    tail = doc['lifecycle_journal']
    assert (tail['max_events'], tail['max_bytes']) == (128, 65536)
    assert doc['buffer_budget']['prefix_max_events'] == 30000 - 128
    assert doc['buffer_budget']['prefix_max_bytes'] == 16*1024*1024 - 65536
    assert tail['events'][0]['data'] == {}
    assert doc['events'][0]['data']['q_start'] == [.123456789012345]
    assert_bounds(doc, 30000, 16*1024*1024)


def test_tail_byte_eviction_retains_latest_bounded_errors_and_reports_truncation(tmp_path):
    rec = evidence.Recording('test', 'arm', max_events=1000, max_bytes=32768)
    phase = evidence._PHASE.set('p'*1000)
    stream = evidence._STREAM.set('s'*1000)
    try:
        for i in range(20):
            rec.record('stream_raised', {'exception_type': 'E'*1000, 'error': f'{i}:'+'💥'*10000})
    finally:
        evidence._STREAM.reset(stream)
        evidence._PHASE.reset(phase)
    summary, doc = finish(rec, tmp_path)
    tail = doc['lifecycle_journal']
    row = tail['events'][-1]
    assert row['data']['error'].startswith('19:')
    assert len(row['data']['error']) <= 512
    assert len(row['phase']) <= 96 and len(row['stream']) <= 64
    assert len(row['data']['exception_type']) <= 96
    assert set(row['truncated_fields']) == {'phase', 'stream', 'data.error', 'data.exception_type'}
    assert tail['truncated_events'] == 20 and tail['evicted_events'] > 0
    assert tail['retained_events'] < tail['max_events']  # Byte cap, not event cap.
    assert not tail['complete'] and not summary['logging_complete']
    assert_bounds(doc, 1000, 32768)


@pytest.mark.parametrize('loss', ['eviction', 'truncation', 'serialization'])
def test_lifecycle_loss_never_promotes_an_otherwise_complete_submission(tmp_path, loss):
    rec = evidence.Recording('test', 'arm')
    clock = dict(version=1, engine='physx', clock='SimulationManager', robot_id='/robot',
                 epoch='epoch', physics_step=1, sim_time=.01, physics_dt_s=.01)
    ledger = TargetReceipts('/robot', ['joint'], 'epoch')
    ledger.completed_update(clock)
    ack = ledger.queued([.125], 'command')
    ledger.setter_returned(ack['sequence'], np.asarray([.125], np.float32), [0])
    rec.record('target_request', {'command_id': 'command', 'q_asset': [.125],
                                 'robot_id': '/robot', 'endpoint': ['fake', 1]})
    rec.record('target_ack', {'command_id': 'command', 'queued_receipt': ack, 'endpoint': ['fake', 1]})
    rec.record('isaac_state', {'target_receipts': ledger.snapshot(), 'endpoint': ['fake', 1],
                              'physics_clock': clock})
    assert evidence.submission_coverage([json.loads(r) for r in rec.events])['complete'] is True
    if loss == 'eviction':
        for _ in range(130):
            rec.record('phase_end', {})
    elif loss == 'truncation':
        rec.record('stream_raised', {'error': 'x'*1000})
    else:
        rec.record('stream_returned', {'settled': float('nan')})
    rec.record('skill_end', {})
    summary, doc = finish(rec, tmp_path)
    assert evidence.submission_coverage(doc['events'])['complete'] is True
    assert summary['dropped_events'] == 0  # Prefix retained every serializable row.
    assert summary['logging_complete'] is False
    assert summary['submission_coverage']['complete'] is False
    assert summary['submission_coverage']['first_setter_observed'] == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), object()])
def test_invalid_settled_value_is_not_coerced_to_a_success(tmp_path, bad):
    rec = evidence.Recording('test', 'arm')
    rec.record('stream_returned', {'settled': bad})
    rec.record('skill_end', {})
    summary, doc = finish(rec, tmp_path)
    tail = doc['lifecycle_journal']
    assert [row['kind'] for row in tail['events']] == ['skill_end']
    assert tail['logging_errors'] == 1
    assert not summary['logging_complete'] and not summary['submission_coverage']['complete']


def test_one_captured_clock_per_event_and_nonfinite_capture_fails_closed(monkeypatch, tmp_path):
    stamps = iter([12.25, float('nan'), 13.5])
    calls = []
    def clock():
        value = next(stamps); calls.append(value); return value
    monkeypatch.setattr(evidence, 'time', SimpleNamespace(monotonic=clock))
    rec = evidence.Recording('test', 'arm')
    rec.record('skill_begin', {})
    rec.record('phase_end', {})
    rec.record('skill_end', {})
    summary, doc = finish(rec, tmp_path)
    assert len(calls) == 3
    assert [r['monotonic_s'] for r in doc['events']] == [12.25, 13.5]
    assert [r['monotonic_s'] for r in doc['lifecycle_journal']['events']] == [12.25, 13.5]
    assert doc['lifecycle_journal']['logging_errors'] == 1
    assert summary['logging_complete'] is False


def test_detail_serialization_failure_does_not_lose_compact_close(tmp_path):
    rec = evidence.Recording('test', 'arm')
    rec.record('stream_returned', {'settled': True, 'ignored_unserializable': object()})
    rec.record('skill_end', {})
    summary, doc = finish(rec, tmp_path)
    assert [r['kind'] for r in doc['events']] == ['skill_end']
    assert doc['lifecycle_journal']['events'][0]['data'] == {'settled': True}
    assert summary['logging_errors'] == 1 and not summary['logging_complete']


def test_tail_serialization_failure_is_bounded_and_does_not_drop_detailed_event(monkeypatch, tmp_path):
    rec = evidence.Recording('test', 'arm')
    dumps = evidence.json.dumps
    def broken_projection(value, **kwargs):
        if isinstance(value, dict) and 'truncated_fields' in value:
            raise ValueError('tail encoding failed' + 'x'*10000)
        return dumps(value, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(evidence.json, 'dumps', broken_projection)
        rec.record('skill_end', {})
    summary, doc = finish(rec, tmp_path)
    assert [r['kind'] for r in doc['events']] == ['skill_end']
    assert doc['lifecycle_journal']['events'] == []
    assert doc['lifecycle_journal']['logging_errors'] == 1
    assert len(doc['lifecycle_journal']['first_error']) <= 500
    assert not summary['logging_complete'] and summary['logging_errors'] == 1


@pytest.mark.parametrize('bad_message', [False, True])
@pytest.mark.parametrize('base', [Exception, BaseException])
def test_original_stream_exception_survives_saturation_and_bad_message(monkeypatch, tmp_path, bad_message, base):
    class OriginalError(base):
        def __str__(self):
            if bad_message:
                raise RuntimeError('broken exception formatter')
            return 'original ' + 'x'*10000
    original = OriginalError()
    recording = evidence.Recording('test', 'arm', max_events=128, max_bytes=32768)
    monkeypatch.setattr(evidence, 'Recording', lambda *_a, **_kw: recording)
    monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(tmp_path))
    class Motion:
        last_state = None
        @evidence.stream
        def move(self, start, target):
            for _ in range(100):
                evidence.event('isaac_state', payload='x'*1000)
            raise original
    @evidence.phase('home')
    def action():
        Motion().move([0.], [1.])
    context = {}
    with pytest.raises(OriginalError) as raised:
        with evidence.record_skill('test', 'arm', context, enabled=True):
            action()
    assert raised.value is original and not evidence.active()
    doc = json.loads(Path(context['motion_evidence']['path']).read_text())
    rows = doc['lifecycle_journal']['events']
    assert [r['kind'] for r in rows][-4:] == ['stream_raised', 'phase_end', 'skill_raised', 'skill_end']
    assert rows[-4]['phase'] == 'home' and rows[-4]['stream'] is not None
    assert not context['motion_evidence']['logging_complete']


@pytest.mark.parametrize('where', ['record', 'finish'])
def test_diagnostic_error_with_broken_formatter_does_not_mask_original_base_exception(monkeypatch, tmp_path, where):
    class DiagnosticError(Exception):
        def __str__(self):
            raise BaseException('formatter must not replace control exception')
    class ControlStop(BaseException):
        pass
    original = ControlStop('original stop')
    def fail(*_a, **_kw):
        raise DiagnosticError()
    if where == 'record':
        monkeypatch.setattr(evidence, '_plain', fail)
    else:
        monkeypatch.setattr(Path, 'mkdir', fail)
    monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(tmp_path))
    context = {}
    with pytest.raises(ControlStop) as raised:
        with evidence.record_skill('test', 'arm', context, enabled=True):
            raise original
    assert raised.value is original and not evidence.active()
    summary = context['motion_evidence']
    assert summary['logging_errors'] > 0 and summary['logging_complete'] is False
    assert len(summary['first_error']) <= 500
    assert 'exception message formatting failed' in summary['first_error']


@pytest.mark.parametrize('slow,jump,cancel', [(0,1,False), (100,1,False), (0,200,False), (0,1,True), (0,0,False)])
def test_saturated_recording_preserves_entire_motion_sequence(monkeypatch, tmp_path, slow, jump, cancel):
    cls = evidence.Recording
    monkeypatch.setattr(evidence, 'Recording', lambda *a, **kw: cls(*a, max_events=16, max_bytes=2048, **kw))
    runs = []
    for enabled in (False, True):
        monkeypatch.setenv('CASCADE_MOTION_EVIDENCE_DIR', str(tmp_path))
        sim = Sim(monkeypatch, slow, jump, cancel)
        if jump == 0:
            sim.motion_wall_timeout_s = .12
        approve = lambda a,b,dt: sim.calls.append(('approve', a.tolist(), b.tolist(), dt))
        context = {}
        with evidence.record_skill('test', 'arm', context, enabled=enabled):
            try:
                outcome = simulation_motion.SimulationMotion(sim, approve=approve).stream(
                    np.array([.1, -.1]), .2, 50., .045, 1., None)
            except SafetyViolation as exc:
                outcome = str(exc)
        runs.append((sim.calls, outcome))
    assert runs[0] == runs[1]
    doc = json.loads(Path(context['motion_evidence']['path']).read_text())
    assert doc['lifecycle_journal']['events'][-1]['kind'] == 'skill_end'
    assert not doc['logging_complete']
    assert_bounds(doc, 16, 2048)
