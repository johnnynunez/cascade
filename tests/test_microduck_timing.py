"""Diagnostic spans preserve calls, clocks, thread ownership and failures."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from cascade.sim import microduck_timing as timing
from cascade.sim.microduck_timing import PhaseProfile


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 10
        return self.now


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_nested_inclusive_spans_restore_methods_and_report_their_own_write_cost(tmp_path):
    path, clock = tmp_path/'timing.jsonl', Clock()
    owner = SimpleNamespace(physics_clock=(2, .01), solve=lambda: 'same return')
    original = owner.solve
    profile = PhaseProfile(path, clock=clock)
    profile.wrap(owner, 'solve', 'solve')
    initial_clock = clock.now
    assert owner.solve() == 'same return' and clock.now == initial_clock  # outside an attempt
    for expected in ('withheld', 'solved'):
        with profile.attempt(owner) as attempt:
            with profile.span('outer', 'duck'):
                assert owner.solve() == 'same return'
            attempt['outcome'] = expected
            if expected == 'solved':
                owner.physics_clock = (3, .015)
    profile.close()
    profile.close()
    assert owner.solve is original
    first, second, footer = rows(path)
    assert first['clock_before'] == first['clock_after'] == [2, .01]
    assert first['outcome'] == 'withheld' and second['outcome'] == 'solved'
    assert second['clock_after'] == [3, .015]
    outer, inner = first['spans']
    assert outer['parent'] is None and inner['parent'] == outer['id']
    assert outer['duration_ns'] > inner['duration_ns'] > 0
    assert first['previous_profile_write'] is None
    assert second['previous_profile_write'] == {'attempt': 0, 'duration_ns': 10}
    assert footer['last_profile_write'] == {'attempt': 1, 'duration_ns': 10}
    assert footer['attempts'] == 2 and not footer['errors']


def test_reader_thread_is_not_instrumented_or_blocked_by_owner_attempt(tmp_path):
    path = tmp_path/'timing.jsonl'
    owner = SimpleNamespace(physics_clock=(2, .01), read=lambda: 'observed')
    profile = PhaseProfile(path)
    profile.wrap(owner, 'read', 'read')
    with profile.attempt(owner) as attempt, ThreadPoolExecutor(1) as pool:
        assert pool.submit(owner.read).result(timeout=1) == 'observed'
        assert not attempt['spans']
        assert owner.read() == 'observed'
        attempt['outcome'] = 'withheld'
    profile.close()
    assert [span['phase'] for span in rows(path)[0]['spans']] == ['read']
    assert owner.physics_clock == (2, .01)


def test_native_failure_is_recorded_without_changing_exception_or_clock(tmp_path):
    class Owner:
        physics_clock = (2, .01)
        def solve(self):
            raise failure
    failure = RuntimeError('native fault')
    path, owner = tmp_path/'timing.jsonl', Owner()
    profile = PhaseProfile(path)
    profile.wrap(owner, 'solve', 'solve')
    with pytest.raises(RuntimeError) as caught, profile.attempt(owner):
        owner.solve()
    assert caught.value is failure
    profile.close()
    assert 'solve' not in vars(owner)  # restore class method, not an instance shadow
    record = rows(path)[0]
    assert record['outcome'] == 'error' and record['error_type'] == 'RuntimeError'
    assert record['spans'][0]['error_type'] == 'RuntimeError'
    assert record['clock_before'] == record['clock_after'] == [2, .01]


def test_profile_write_failure_does_not_replace_primary_native_exception(tmp_path):
    class Broken:
        closed = False
        def write(self, value):
            raise OSError('disk unavailable')
        def close(self):
            self.closed = True
    profile = PhaseProfile(tmp_path/'timing.jsonl')
    profile.stream.close()
    profile.stream = Broken()
    owner = SimpleNamespace(physics_clock=(2, .01), solve=lambda: None)
    original = owner.solve
    profile.wrap(owner, 'solve', 'solve')
    failure = RuntimeError('original native failure')
    with pytest.raises(RuntimeError) as caught, profile.attempt(owner):
        raise failure
    assert caught.value is failure and profile.errors == ['OSError']
    with pytest.raises(OSError):
        profile.close()
    assert owner.solve is original and profile.stream.closed


def test_profile_write_failure_without_native_failure_is_not_a_success(tmp_path):
    profile = PhaseProfile(tmp_path/'timing.jsonl')
    profile.stream.close()
    with pytest.raises(ValueError, match='closed'), profile.attempt(SimpleNamespace(physics_clock=(2, .01))):
        pass
    assert profile.errors == ['ValueError']
    with pytest.raises(ValueError, match='closed'):
        profile.close()


@pytest.fixture(autouse=True)
def fake_gc(monkeypatch):
    def forbidden(*args):
        pytest.fail('passive profiling must not control or enumerate the collector')
    collector = SimpleNamespace(callbacks=[], enabled=True, thresholds=(700, 10, 10),
        isenabled=lambda: collector.enabled, get_threshold=lambda: collector.thresholds,
        get_count=lambda: (1, 2, 3), get_stats=lambda: [dict(collections=1, collected=0, uncollectable=0) for _ in range(3)],
        enable=forbidden, disable=forbidden, collect=forbidden, freeze=forbidden,
        unfreeze=forbidden, set_threshold=forbidden, get_objects=forbidden)
    monkeypatch.setattr(timing, 'gc', collector)
    return collector


INFO = dict(generation=2, collected=4, uncollectable=0)


def test_gen2_trigger_copies_bounded_frames_without_retaining_locals(fake_gc):
    import weakref
    class Payload:
        pass
    capture = timing._GCEvents(Clock())
    def emit(depth):
        if depth:
            return emit(depth-1)
        payload = Payload()
        reference = weakref.ref(payload)
        capture.callback('start', INFO)
        return reference
    reference = emit(20)
    assert reference() is None
    capture.callback('stop', INFO)
    capture.callback('start', dict(INFO, generation=1))
    batch, summary = capture.close()
    start, stop, minor = batch['events']
    trigger = start['trigger']
    assert len(trigger['frames']) == 16 and trigger['truncated']
    assert not trigger['strings_truncated']
    assert all(set(frame) == {'path', 'function', 'line'} for frame in trigger['frames'])
    assert trigger['frames'][0]['function'] == 'emit'
    assert start['monotonic_ns'] <= trigger['start_monotonic_ns'] < trigger['end_monotonic_ns']
    assert trigger['duration_ns'] == trigger['end_monotonic_ns']-trigger['start_monotonic_ns']
    assert 'trigger' not in stop and 'trigger' not in minor
    assert summary['counters_complete']
    for snapshot in (summary['counters_initial'], summary['counters_final']):
        assert snapshot['values']['count'] == [1, 2, 3]
        assert snapshot['duration_ns'] == snapshot['end_monotonic_ns']-snapshot['start_monotonic_ns'] >= 0


def test_trigger_strings_are_bounded_and_failure_keeps_callback_accounting(monkeypatch, fake_gc):
    capture = timing._GCEvents(Clock())
    exec(compile("capture.callback('start', INFO)", 'x'*600, 'exec'),
         {'capture': capture, 'INFO': INFO})
    trigger = capture.pending[0]['trigger']
    assert trigger['strings_truncated'] and trigger['frames'][0]['path'] == 'x'*512
    def fail():
        raise RuntimeError('stack unavailable')
    monkeypatch.setattr(timing, '_trigger_stack', fail)
    capture.callback('start', INFO)
    batch, summary = capture.close()
    assert batch['events'][-1]['error_type'] == 'RuntimeError'
    assert batch['events'][-1]['generation'] == 2 and batch['events'][-1]['monotonic_ns'] > 0
    assert summary['observed'] == summary['recorded'] == 2 and summary['errors'] == 1
    assert not summary['accounting_complete'] and not fake_gc.callbacks


def test_c_callback_without_python_caller_keeps_complete_event_metadata(monkeypatch, fake_gc):
    capture = timing._GCEvents(Clock())
    monkeypatch.setattr(timing.sys, '_getframe', lambda depth: SimpleNamespace(f_back=None))
    capture.callback('start', INFO)
    batch, summary = capture.close()
    event = batch['events'][0]
    assert event['error_type'] is None and event['generation'] == 2 and event['monotonic_ns'] > 0
    assert event['trigger']['frames'] == [] and not event['trigger']['python_caller_present']
    assert not event['trigger']['truncated'] and summary['accounting_complete']


def test_counter_failure_is_explicit_and_does_not_leave_callback_installed(fake_gc):
    foreign = lambda *args: None
    fake_gc.callbacks.append(foreign)
    capture = timing._GCEvents(Clock())
    def fail():
        raise RuntimeError('counter unavailable')
    fake_gc.get_stats = fail
    _, summary = capture.close()
    assert summary['counters_initial']['error_type'] is None
    assert summary['counters_final']['error_type'] == 'RuntimeError'
    assert summary['counters_final']['values'] is None and not summary['counters_complete']
    assert fake_gc.callbacks == [foreign]


def test_gc_events_cross_attempts_and_threads_without_forcing_pairs(tmp_path, fake_gc):
    path = tmp_path/'timing.jsonl'
    foreign = lambda *args: None
    fake_gc.callbacks.append(foreign)
    profile = PhaseProfile(path, clock=Clock())
    callback = profile.gc.callback
    callback('start', INFO)  # before the owner starts its first attempt
    owner = SimpleNamespace(physics_clock=(2, .01))
    with profile.attempt(owner) as attempt:
        attempt['outcome'] = 'withheld'
    with ThreadPoolExecutor(1) as pool:
        pool.submit(callback, 'stop', INFO).result(timeout=1)
    with profile.attempt(owner) as attempt:
        attempt['outcome'] = 'withheld'
    callback('start', INFO)  # deliberately no stop before closure
    later_foreign = lambda *args: None
    fake_gc.callbacks.append(later_foreign)
    fake_gc.enabled, fake_gc.thresholds = False, (600, 9, 9)  # external changes
    profile.close()
    profile.close()
    assert fake_gc.callbacks == [foreign, later_foreign]
    assert not fake_gc.enabled and fake_gc.thresholds == (600, 9, 9)
    first, second, footer = rows(path)
    events = [row['gc']['events'][0] for row in (first, second, footer)]
    assert [event['id'] for event in events] == [0, 2, 4]
    assert [row['gc']['fence_id'] for row in (first, second, footer)] == [1, 3, 5]
    assert [event['phase'] for event in events] == ['start', 'stop', 'start']
    assert events[0]['monotonic_ns'] < first['start_monotonic_ns']
    assert first['end_monotonic_ns'] < events[1]['monotonic_ns'] < second['start_monotonic_ns']
    assert events[1]['thread_id'] != events[0]['thread_id'] == events[2]['thread_id']
    assert all(event['generation'] == 2 and event['collected'] == 4 for event in events)
    summary = footer['gc_summary']
    assert summary['observed'] == summary['recorded'] == 3
    assert summary['dropped_or_pending'] == summary['errors'] == 0
    assert summary['accounting_complete']
    assert summary['initial'] == {'enabled': True, 'thresholds': [700, 10, 10]}
    assert summary['final'] == {'enabled': False, 'thresholds': [600, 9, 9]}
    callback('stop', INFO)  # a saved callback cannot write after close
    assert not profile.gc.active and not profile.gc.pending


def test_gc_bound_and_errors_remain_explicit_across_drains(fake_gc):
    clock = Clock()
    capture = timing._GCEvents(clock, limit=2)
    capture.callback('start', INFO)
    capture.callback('stop', INFO)
    capture.callback('start', INFO)  # bounded loss is explicit, with an ID gap
    capture.callback('stop', INFO)
    first = capture.drain()
    assert first['observed'] == 4 and first['dropped_or_pending'] == 2 and first['errors'] == 0
    assert [event['id'] for event in first['events']] == [2, 3]
    capture.callback('start', {})  # diagnostic callback failure must not escape
    capture.callback('stop', INFO)
    last, summary = capture.close()
    assert last['observed'] == 6 and last['errors'] == 1 and last['dropped_or_pending'] == 2
    assert [event['id'] for event in last['events']] == [5, 6]
    assert last['events'][0]['error_type'] == 'KeyError'
    assert summary['observed'] == 6 and summary['recorded'] == 4
    assert summary['dropped_or_pending'] == 2 and summary['errors'] == 1
    assert not summary['accounting_complete']
    assert fake_gc.callbacks == []


def test_gc_callback_clock_failure_and_profile_io_keep_primary_error_and_cleanup(tmp_path, fake_gc):
    profile = PhaseProfile(tmp_path/'timing.jsonl')
    foreign = lambda *args: None
    fake_gc.callbacks.append(foreign)
    def broken_clock():
        raise RuntimeError('diagnostic clock')
    profile.gc.clock = broken_clock
    profile.gc.callback('start', INFO)
    assert profile.gc.pending[0]['error_type'] == 'RuntimeError'
    profile.stream.close()
    primary = RuntimeError('native failure')
    with pytest.raises(RuntimeError) as caught, profile.attempt(SimpleNamespace(physics_clock=(2, .01))):
        raise primary
    assert caught.value is primary
    with pytest.raises(ValueError):
        profile.close()
    assert fake_gc.callbacks == [foreign] and not profile.gc.active


def test_gc_empty_profile_and_external_callback_removal_are_visible(tmp_path, fake_gc):
    path = tmp_path/'timing.jsonl'
    profile = PhaseProfile(path)
    fake_gc.callbacks.remove(profile.gc.callback)
    profile.close()
    footer, = rows(path)
    assert footer['attempts'] == 0 and footer['gc']['events'] == []
    assert footer['gc']['callback_registered'] is False
    assert footer['gc_summary']['observed'] == 0
    assert not footer['gc_summary']['accounting_complete']


@pytest.mark.parametrize('close_in_flight', [False, True])
def test_gc_callback_resuming_after_owner_drain_is_retained_or_explicitly_incomplete(fake_gc, close_in_flight):
    from threading import Event, get_ident
    entered, release = Event(), Event()
    owner_thread = get_ident()
    def suspended_clock():
        if get_ident() != owner_thread:
            entered.set()
            assert release.wait(timeout=2)
        return 123
    capture = timing._GCEvents(suspended_clock)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(capture.callback, 'start', INFO)
        try:
            assert entered.wait(timeout=2)
            if close_in_flight:
                first, summary = capture.close()
            else:
                first = capture.drain()
            serialized = json.dumps(first)
        finally:
            release.set()
        pending.result(timeout=2)
    assert json.dumps(first) == serialized  # callbacks cannot mutate a drained row
    assert first['observed'] == first['dropped_or_pending'] == 1 and first['events'] == []
    if close_in_flight:
        assert not summary['accounting_complete'] and summary['recorded'] == 0
    else:
        last, summary = capture.close()
        assert last['events'][0]['monotonic_ns'] == 123
        assert summary['observed'] == summary['recorded'] == 1
        assert summary['accounting_complete']


def test_gc_callback_entering_after_final_fence_cannot_append_unaccounted_events(fake_gc):
    import inspect
    import sys
    from threading import Event
    entered, release = Event(), Event()
    capture = timing._GCEvents(Clock())
    source, start = inspect.getsourcelines(capture._record)
    line = start + next(i for i, text in enumerate(source) if 'ticket = next' in text)
    def trace(frame, event, arg):
        if frame.f_code is capture._record.__code__ and event == 'line' and frame.f_lineno == line:
            entered.set()
            assert release.wait(timeout=2)
        return trace
    def late_callback():
        sys.settrace(trace)
        try:
            capture.callback('start', INFO)
        finally:
            sys.settrace(None)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(late_callback)
        try:
            assert entered.wait(timeout=2)
            batch, summary = capture.close()
            serialized = json.dumps(batch)
        finally:
            release.set()
        pending.result(timeout=2)
    assert not capture.pending and json.dumps(batch) == serialized
    assert summary['observed'] == 0 and summary['accounting_complete']
