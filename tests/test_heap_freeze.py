"""The explicit startup-heap freeze bounds the collector's traversal set and nothing else."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO
from cascade.sim import heap_freeze
from cascade.sim.heap_freeze import StartupHeapFreeze


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 10
        return self.now


def fake_collector(*, tracked=1200, foreign_frozen=0):
    """Scripted collector for policy tests: records call order; retuning is forbidden.

    Shared with the launcher and Factory integration tests so they never freeze
    the real test interpreter.
    """
    def forbidden(*args):
        pytest.fail('the startup heap policy must not enable, disable or retune the collector')
    state = SimpleNamespace(calls=[], enabled=True, thresholds=(700, 10, 10), callbacks=[object()],
                            frozen=foreign_frozen, tracked=tracked)
    def collect(generation=2):
        state.calls.append(('collect', generation))
        return 7
    def freeze():
        state.calls.append(('freeze',))
        state.frozen, state.tracked = state.frozen+state.tracked, 0
    def unfreeze():
        state.calls.append(('unfreeze',))
        state.frozen, state.tracked = 0, state.frozen
    state.collect, state.freeze, state.unfreeze = collect, freeze, unfreeze
    state.isenabled = lambda: state.enabled
    state.get_threshold = lambda: state.thresholds
    state.get_count = lambda: (1, 2, 3)
    state.get_stats = lambda: [dict(collections=1, collected=0, uncollectable=0) for _ in range(3)]
    state.get_freeze_count = lambda: state.frozen
    state.get_objects = lambda generation=None: [object()] * state.tracked
    state.enable = state.disable = state.set_threshold = forbidden
    return state


@pytest.fixture
def collector():
    return fake_collector()


def test_apply_collects_before_freezing_and_records_unchanged_settings(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    receipt = policy.apply()
    assert collector.calls == [('collect', 2), ('freeze',)] and policy.state == 'frozen'
    assert receipt['policy'] == heap_freeze.POLICY == 'freeze-startup-heap-v1'
    assert receipt['implementation_sha256'] == hashlib.sha256(Path(heap_freeze.__file__).read_bytes()).hexdigest()
    assert receipt['collect'] == dict(duration_ns=10, unreachable=7, frozen_after_collect=0)
    assert receipt['freeze'] == dict(duration_ns=10, frozen=1200, foreign_frozen_before=0)
    assert receipt['before']['frozen'] == 0 and receipt['after']['frozen'] == 1200
    assert receipt['settings'] == dict(enabled=True, thresholds=[700, 10, 10], callbacks=1)
    assert receipt['process']['pid'] > 0 and receipt['process']['interpreter']
    assert collector.enabled and collector.thresholds == (700, 10, 10) and len(collector.callbacks) == 1
    assert 'Not a control-deadline' in receipt['scope']
    assert policy.receipt is receipt and policy.released is None and policy.apply_error is None


def test_release_unfreezes_then_collects_and_reports_the_retained_count(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    policy.apply()
    released = policy.release()
    # measured full collection while still frozen, then unfreeze, then the ordinary collection
    assert collector.calls[2:] == [('collect', 2), ('unfreeze',), ('collect', 2)] and policy.state == 'released'
    assert released['frozen_collect'] == dict(duration_ns=10, unreachable=7, tracked_after=0)
    assert released['unfrozen'] == released['policy_frozen'] == 1200 and released['after']['frozen'] == 0
    assert released['frozen_changed_since_apply'] == 0 and released['apply_receipt_complete'] is True
    assert released['collect'] == dict(duration_ns=10, unreachable=7)
    assert released['settings'] == dict(enabled=True, thresholds=[700, 10, 10], callbacks=1)
    with pytest.raises(RuntimeError, match='already released'):
        policy.release()
    assert collector.calls.count(('unfreeze',)) == 1


def test_release_without_collections_is_explicit(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    policy.apply()
    released = policy.release(collect=False, measure_frozen=False)
    assert released['collect'] is None and released['frozen_collect'] is None
    assert collector.calls == [('collect', 2), ('freeze',), ('unfreeze',)]


def test_apply_twice_and_release_before_apply_are_refused(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    with pytest.raises(RuntimeError, match='never applied'):
        policy.release()
    policy.apply()
    with pytest.raises(RuntimeError, match='already applied'):
        policy.apply()
    assert collector.calls == [('collect', 2), ('freeze',)]


def test_disabled_collector_is_refused_before_any_collector_call(collector):
    collector.enabled = False
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    with pytest.raises(RuntimeError, match='collector enabled'):
        policy.apply()
    assert collector.calls == [] and policy.state == 'idle' and policy.receipt is None


def test_settings_changed_during_collection_leave_the_heap_frozen_for_release(collector):
    original = collector.collect
    def disabling_collect(generation=2):
        collector.enabled = False  # e.g. a foreign gc callback retuning the collector
        return original(generation)
    collector.collect = disabling_collect
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    with pytest.raises(RuntimeError, match='settings changed'):
        policy.apply()
    # The collector operation happened; only the receipt failed. Release still restores it.
    assert collector.calls == [('collect', 2), ('freeze',)] and policy.frozen and policy.receipt is None
    assert policy.apply_error == 'RuntimeError'
    released = policy.release()
    assert collector.calls[2:] == [('collect', 2), ('unfreeze',), ('collect', 2)] and collector.frozen == 0
    assert released['apply_receipt_complete'] is False and released['apply_error'] == 'RuntimeError'
    assert released['policy_frozen'] is None and released['frozen_changed_since_apply'] is None
    assert released['unfrozen'] == 1200


class Interrupt(BaseException):
    """Stands in for the launcher's SignalRequest landing mid-call."""


def test_signal_right_after_freeze_is_recoverable_by_release(collector):
    clock = Clock()
    ticks = iter([0, 0, 1])  # before-collect, after-collect clock reads succeed; the post-freeze one is interrupted
    def interrupted_clock():
        if next(ticks):
            raise Interrupt()
        return clock()
    policy = StartupHeapFreeze(clock=interrupted_clock, collector=collector)
    with pytest.raises(Interrupt):
        policy.apply()
    assert collector.calls == [('collect', 2), ('freeze',)] and policy.frozen
    assert policy.receipt is None and policy.apply_error == 'Interrupt'
    policy.clock = clock
    released = policy.release()
    assert collector.frozen == 0 and released['apply_receipt_complete'] is False and released['unfrozen'] == 1200


def test_interrupt_inside_freeze_call_still_owns_the_frozen_heap(collector):
    """A signal delivered as freeze() returns: state was published before the call."""
    original = collector.freeze
    def interrupted_freeze():
        original()
        raise Interrupt()
    collector.freeze = interrupted_freeze
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    with pytest.raises(Interrupt):
        policy.apply()
    assert policy.state == 'freezing' and policy.frozen and policy.receipt is None and policy.apply_error == 'Interrupt'
    assert collector.frozen == 1200
    released = policy.release()  # no frozen measurement in an uncertain state; the unfreeze still happens
    assert released['frozen_collect'] is None and released['apply_receipt_complete'] is False
    assert collector.frozen == 0 and policy.state == 'released'


def test_interrupt_inside_unfreeze_call_allows_a_harmless_retry(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    policy.apply()
    original = collector.unfreeze
    def interrupted_unfreeze():
        original()
        raise Interrupt()
    collector.unfreeze = interrupted_unfreeze
    with pytest.raises(Interrupt):
        policy.release()
    assert policy.state == 'releasing' and policy.frozen and collector.frozen == 0 and policy.release_error == 'Interrupt'
    collector.unfreeze = original
    released = policy.release()  # retry: unfreeze is idempotent, receipt completes
    assert policy.state == 'released' and released['unfrozen'] == 0 and collector.calls.count(('unfreeze',)) == 2
    with pytest.raises(RuntimeError, match='already released'):
        policy.release()


def test_failure_after_unfreeze_keeps_collector_restored_and_receipt_absence_explicit(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    policy.apply()
    original = collector.collect
    def failing_collect(generation=2):
        raise OSError('synthetic failure after unfreeze')
    collector.collect = failing_collect
    with pytest.raises(OSError, match='after unfreeze'):
        policy.release(measure_frozen=False)
    assert collector.calls[2:] == [('unfreeze',)] and collector.frozen == 0
    assert policy.state == 'released' and policy.released is None and policy.release_error == 'OSError'
    with pytest.raises(RuntimeError, match='already released'):
        policy.release()
    collector.collect = original
    assert collector.calls.count(('unfreeze',)) == 1


def test_pre_release_snapshot_failure_still_unfreezes(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    policy.apply()
    collector.get_count = lambda: (_ for _ in ()).throw(RuntimeError('counters unavailable'))
    with pytest.raises(RuntimeError, match='counters unavailable'):
        policy.release()
    assert collector.calls[-1] == ('unfreeze',) and collector.frozen == 0 and policy.state == 'released'


def test_frozen_measurement_failure_still_unfreezes(collector):
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    policy.apply()
    original = collector.collect
    def failing(generation=2):
        raise MemoryError('synthetic measurement failure while frozen')
    collector.collect = failing
    with pytest.raises(MemoryError):
        policy.release()
    collector.collect = original
    assert collector.calls[-1] == ('unfreeze',) and collector.frozen == 0 and policy.state == 'released'
    assert policy.released is None and policy.release_error == 'MemoryError'


def test_foreign_frozen_objects_are_recorded_not_refused(collector):
    collector.frozen = 375  # CPython 3.12 parks immortal objects in the permanent generation
    policy = StartupHeapFreeze(clock=Clock(), collector=collector)
    receipt = policy.apply()
    assert receipt['freeze'] == dict(duration_ns=10, frozen=1200, foreign_frozen_before=375)
    assert receipt['collect']['frozen_after_collect'] == 375 and receipt['after']['frozen'] == 1575
    released = policy.release()
    assert released['unfrozen'] == 1575 and released['policy_frozen'] == 1200  # unfreeze is interpreter-wide


MECHANISM = r'''
import gc, json, sys, weakref
sys.path.insert(0, sys.argv[1])
from cascade.sim.heap_freeze import StartupHeapFreeze
class Node: pass
startup = Node(); startup.ref = startup
garbage = Node(); garbage.ref = garbage
dead = weakref.ref(garbage); del garbage
callbacks, settings = tuple(gc.callbacks), (gc.isenabled(), gc.get_threshold())
foreign = gc.get_freeze_count()
policy = StartupHeapFreeze()
receipt = policy.apply()
out = dict(foreign=foreign, foreign_recorded=receipt['freeze']['foreign_frozen_before'],
    frozen_after_collect_ge_foreign=receipt['collect']['frozen_after_collect'] >= foreign,
    garbage_dead=dead() is None, policy_frozen=receipt['freeze']['frozen'],
    delta_matches=receipt['freeze']['frozen'] == gc.get_freeze_count()-receipt['collect']['frozen_after_collect'],
    startup_listed_after_freeze=any(o is startup for o in gc.get_objects()))
later = Node(); later.ref = later
out['later_listed'] = any(o is later for o in gc.get_objects())
out['startup_tracked'] = gc.is_tracked(startup)
out['settings_unchanged'] = (gc.isenabled(), gc.get_threshold()) == settings and tuple(gc.callbacks) == callbacks
released = policy.release()
out.update(frozen_collect_ns=released['frozen_collect']['duration_ns'], unfreeze_collect_ns=released['collect']['duration_ns'],
    frozen_tracked_after=released['frozen_collect']['tracked_after'],
    unfrozen_le_after=released['unfrozen'] <= receipt['after']['frozen'],
    change_consistent=released['frozen_changed_since_apply'] == released['unfrozen']-receipt['after']['frozen'] <= 0,
    frozen_after_release_lt_unfrozen=released['after']['frozen'] < released['unfrozen'],
    startup_listed_after_release=any(o is startup for o in gc.get_objects()),
    settings_unchanged_after=(gc.isenabled(), gc.get_threshold()) == settings and tuple(gc.callbacks) == callbacks)
print(json.dumps(out))
'''


def test_real_collector_mechanism_in_a_fresh_interpreter():
    """Frozen objects leave every generation list; new objects stay tracked; unfreeze restores them.

    Runs in a subprocess so the test interpreter's own heap is never frozen.
    """
    proc = subprocess.run([sys.executable, '-c', MECHANISM, str(REPO / 'src')], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out['foreign_recorded'] == out['foreign'] >= 0
    assert out['frozen_after_collect_ge_foreign'] and out['garbage_dead']
    assert out['policy_frozen'] > 0 and out['delta_matches']
    assert not out['startup_listed_after_freeze'] and out['later_listed'] and out['startup_tracked']
    assert out['settings_unchanged'] and out['settings_unchanged_after']
    assert out['unfrozen_le_after'] and out['change_consistent'] and out['frozen_after_release_lt_unfrozen']
    assert out['startup_listed_after_release']
    # the frozen heap is outside the measured collection: far fewer tracked objects and far less time
    assert out['frozen_tracked_after'] < out['policy_frozen']
    assert out['frozen_collect_ns'] < out['unfreeze_collect_ns']
