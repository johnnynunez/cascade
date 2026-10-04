"""Diagnostic spans preserve calls, clocks, thread ownership and failures."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

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
    assert owner.solve() == 'same return' and clock.now == 0  # outside an attempt
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
