"""Python-only profiling must not become simulation or motion authority."""
from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import runpy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO, load_isaac_bridge_definitions

HELPER = REPO / 'scripts/isaac_python_spans.py'
SPANS = runpy.run_path(str(HELPER))
PythonSpans = SPANS['PythonSpans']


class Backend:
    def __init__(self, fail=None):
        self.events, self.stack, self.fail = [], [], fail

    def begin_with_location(self, mask, name, function, filepath, line):
        if self.fail == 'begin':
            raise RuntimeError('broken begin')
        self.events.append(('begin', name))
        self.stack.append(name)

    def end(self, mask):
        name = self.stack.pop()
        self.events.append(('end', name))
        if self.fail == 'end':
            raise RuntimeError('broken end')


def test_disabled_helper_import_and_scopes_never_import_carb(monkeypatch):
    imported = []
    real_import = builtins.__import__

    def checked(name, *args, **kwargs):
        imported.append(name)
        if name.startswith(('carb', 'omni', 'isaacsim')):
            raise AssertionError('disabled profiling imported Kit')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', checked)
    monkeypatch.delenv('CASCADE_ISAAC_PYTHON_SPANS', raising=False)
    monkeypatch.delenv('CASCADE_ISAAC_PYTHON_TIMINGS', raising=False)
    helper = runpy.run_path(str(HELPER))
    spans = helper['from_environment']('nonexistent-source-is-not-read')
    with spans.zone('bridge.loop'):
        with spans.zone('nested'):
            pass
    spans.anchor_once()
    spans.sample_clock_if_due()
    spans.report()
    assert not spans.enabled and spans.anchor is None
    assert not any(name.startswith(('carb', 'omni', 'isaacsim')) for name in imported)


@pytest.mark.parametrize('failure', [None, 'begin', 'end'])
@pytest.mark.parametrize('native_error', [ValueError, KeyboardInterrupt])
def test_profiler_error_never_replaces_native_exception(failure, native_error):
    backend = Backend(failure)
    spans = PythonSpans(enabled=True, backend=backend)
    native = native_error('the original operation')
    with pytest.raises(native_error) as caught:
        with spans.zone('outer'):
            with spans.zone('inner'):
                raise native
    assert caught.value is native
    assert backend.stack == []
    assert bool(spans.error_count) == bool(failure)
    if failure:
        previous = list(backend.events)
        with spans.zone('after-invalid'):
            pass
        assert backend.events == previous


def test_anchor_is_unique_balanced_same_thread_and_brackets_native_zone():
    events, output = [], []
    times = iter([700, 711])
    backend = Backend()
    original_begin, original_end = backend.begin_with_location, backend.end
    backend.begin_with_location = lambda *args: (events.append('begin'), original_begin(*args))[-1]
    backend.end = lambda *args: (events.append('end'), original_end(*args))[-1]

    def clock():
        events.append('clock')
        return next(times)

    spans = PythonSpans(enabled=True, backend=backend, clock_ns=clock,
                        source_path='/source.py', emit=lambda line, **kw: output.append(line))
    spans.anchor_once()
    spans.anchor_once()
    assert events == ['clock', 'begin', 'end', 'clock']
    assert backend.events == [('begin', 'bridge.clock_anchor'), ('end', 'bridge.clock_anchor')]
    assert spans.anchor['before_monotonic_ns'] == 700
    assert spans.anchor['after_monotonic_ns'] == 711
    assert spans.anchor['native_thread_id'] == threading.get_native_id()
    assert len(output) == 1
    spans.report()
    report = json.loads(output[-1].removeprefix('[bridge-python-spans] '))
    assert report['diagnostic_valid'] and report['anchor'] == spans.anchor


def test_factory_checks_backend_and_hashes_sources_only_when_enabled(monkeypatch):
    backend = Backend()
    backend.is_profiler_active = lambda: True
    spec = importlib.util.spec_from_file_location('isolated_isaac_spans', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setenv('CASCADE_ISAAC_PYTHON_SPANS', '1')
    monkeypatch.setattr(helper.importlib, 'import_module', lambda name: backend)
    spans = helper.from_environment(str(REPO / 'scripts/isaac_bridge.py'))
    assert len(spans.source_sha256) == len(spans.helper_sha256) == 64
    assert backend.events == [('begin', 'bridge.profiler_preflight'), ('end', 'bridge.profiler_preflight')]
    backend.is_profiler_active = lambda: False
    unavailable = helper.from_environment(str(REPO / 'scripts/isaac_bridge.py'))
    assert unavailable.error_count == 1
    assert unavailable.first_error['stage'] == 'initialization'


def _loop(*, enabled, failure=None, stopped=False, finger_step=None, partial=False, timings=False):
    events, written = [], []
    backend = Backend(failure)
    spans = PythonSpans(enabled=enabled, backend=backend, timings=timings)
    spans.anchor_once()
    targets = np.array([[0., -1., -1., 0., .1, .2, .025, .035]])

    def get():
        events.append('get')
        return SimpleNamespace(numpy=to_numpy)

    def to_numpy():
        events.append('numpy')
        return targets

    def put(tgt):
        events.append('set')
        written.append(tgt.copy())

    counter = [0]

    def update():
        events.append('update')
        counter[0] += 1

    env = dict(np=np, _python_spans=spans, _profile_zone=spans.zone,
               _camera_video=None, _REQUIRE_CUDA=False,
               _tl=SimpleNamespace(is_playing=lambda: True), _was_playing=True,
               _state_lock=threading.Lock(), _targets=dict(q=None if partial else [1, 2, 3, 4, 5, 6],
               grip_frac=.5, stopped=stopped), _FINGER_STEP=finger_step,
               ARM_IDX=list(range(6)), GRIP_IDX=[6, 7], lower=np.zeros(8), upper=np.ones(8) * .05,
               art=SimpleNamespace(get_dof_position_targets=get, set_dof_position_targets=put),
               step=0, app=SimpleNamespace(is_running=lambda: counter[0] < 1),
               _bridge_should_stop=lambda: False, _camera_capture_due=lambda step: True,
               _update_wrist_cam=lambda: events.append('wrist'), _step_with_frame_history=update,
               _refresh_frames=lambda: events.append('refresh'), _run_exec_jobs=lambda: events.append('jobs'))
    tree = ast.parse((REPO / 'scripts/isaac_bridge.py').read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                and 'app.is_running()' in ast.unparse(n.test))
    exec(compile(ast.Module(body=[loop], type_ignores=[]), 'real_bridge_loop', 'exec'), env)
    return events, written, backend, targets


@pytest.mark.parametrize('stopped,finger_step,partial', [
    (False, None, False), (False, .001, False), (False, None, True), (True, None, False)])
@pytest.mark.parametrize('failure', [None, 'begin', 'end'])
def test_real_loop_preserves_get_numpy_set_order_targets_and_stop(stopped, finger_step, partial, failure):
    args = dict(stopped=stopped, finger_step=finger_step, partial=partial)
    off = _loop(enabled=False, **args)
    on = _loop(enabled=True, failure=failure, **args)
    assert on[0] == off[0] == ([] if stopped else ['get', 'numpy', 'set']) + ['wrist', 'update', 'refresh', 'jobs']
    assert len(on[1]) == len(off[1]) == (0 if stopped else 1)
    if not stopped:
        np.testing.assert_array_equal(on[1][0], off[1][0])
        np.testing.assert_allclose(on[1][0][0, :6], off[3][0, :6] if partial else np.arange(1, 7))
        np.testing.assert_allclose(on[1][0][0, 6:], [.025, .034] if finger_step else [.025, .025])
    assert on[2].stack == []
    if not failure:
        names = [name for kind, name in on[2].events if kind == 'begin']
        assert names.count('bridge.loop') == 1
        assert names.count('bridge.targets.get') == (0 if stopped else 1)


def test_real_exec_queue_keeps_fifo_done_and_nested_scopes():
    backend = Backend()
    spans = PythonSpans(enabled=True, backend=backend)
    order = []
    env = dict(_exec_lock=threading.Lock(), _exec_jobs=[], _profile_zone=spans.zone,
               order=order)
    load_isaac_bridge_definitions({'_run_exec_jobs'}, env)
    done = [threading.Event() for _ in range(3)]
    holders = [{}, {}, {}]

    def first():
        order.append('first')
        with spans.zone('bridge.app_update'):
            pass
        return {'ok': True}

    def failure():
        order.append('failure')
        raise ValueError('native job failed')

    env['_exec_jobs'][:] = [(first, holders[0], done[0]),
                            ('order.append("code")', holders[1], done[1]),
                            (failure, holders[2], done[2])]
    env['_run_exec_jobs']()
    assert order == ['first', 'code', 'failure']
    assert all(event.is_set() for event in done)
    assert holders[0]['resp']['ok'] and holders[1]['resp']['ok']
    assert not holders[2]['resp']['ok']
    assert 'ValueError: native job failed' in holders[2]['resp']['error']
    assert backend.stack == [] and env['_exec_jobs'] == []


def test_profiler_factory_is_called_after_simulationapp_and_bundled():
    source = (REPO / 'scripts/isaac_bridge.py').read_text()
    assert source.index('app = SimulationApp(') < source.index('_python_spans = _python_spans_from_environment(')
    for path in ('deploy/brev/build_bundle.py', 'deploy/brev/prepare_bundle.py',
                 'src/cascade/grasping/evidence.py', 'demo/scene_identity.py'):
        assert '"scripts/isaac_python_spans.py"' in (REPO / path).read_text()


class Clock:
    def __init__(self):
        self.now, self.calls = 1000, 0

    def __call__(self):
        self.calls += 1
        return self.now


def _sampling_spans():
    clock, records, backend = Clock(), [], Backend()
    spans = PythonSpans(enabled=True, backend=backend, clock_ns=clock,
                        source_path='/reviewed/bridge.py',
                        emit=lambda line, **kw: records.append(json.loads(
                            line.removeprefix('[bridge-python-spans] '))))
    spans.source_sha256, spans.helper_sha256 = 'a' * 64, 'b' * 64
    spans.anchor_once()
    return spans, clock, records, backend


def test_disabled_periodic_sampling_does_not_read_clock_backend_or_log():
    def forbidden(*args, **kwargs):
        pytest.fail('Disabled sampling performed work')

    spans = PythonSpans(clock_ns=forbidden, cpu_clock_ns=forbidden, emit=forbidden)
    for _ in range(3):
        spans.sample_clock_if_due()
        with spans.zone('disabled'):
            pass
    spans.report()
    assert spans.clock_sample_count == 0 and spans.last_clock_sample is None


def test_periodic_samples_are_unique_bracketed_and_never_catch_up():
    spans, clock, records, backend = _sampling_spans()
    spans.sample_clock_if_due()
    clock.now += 999_999_999
    spans.sample_clock_if_due()
    assert spans.clock_sample_count == 1
    clock.now += 20_000_000_000  # A slow iteration adds one sample, never a burst.
    spans.sample_clock_if_due()
    spans.sample_clock_if_due()
    clock.now += 1_000_000_000
    spans.sample_clock_if_due()
    samples = [r for r in records if r['event'] == 'clock_sample']
    assert [r['seq'] for r in samples] == [1, 2, 3]
    for sample in samples:
        assert sample['zone'] == f"bridge.clock_sample.{sample['seq']:06d}"
        assert sample['diagnostic_valid'] and sample['error_count'] == 0
        assert sample['before_monotonic_ns'] <= sample['after_monotonic_ns']
        for key in ('pid', 'native_thread_id', 'source_path', 'source_sha256', 'helper_sha256'):
            assert sample[key] == spans.anchor[key]
        assert backend.events.count(('begin', sample['zone'])) == 1
        assert backend.events.count(('end', sample['zone'])) == 1
    assert backend.stack == []
    assert clock.calls == 2 + 5 + 3  # Initial bracket; five polls; three end clocks.
    spans.report()
    assert records[-1]['clock_sample_count'] == 3
    assert records[-1]['last_clock_sample'] == spans.last_clock_sample


def test_periodic_bracket_encloses_zone_and_spacing_starts_after_end():
    spans, clock, records, backend = _sampling_spans()
    original_begin, original_end = backend.begin_with_location, backend.end
    observed = []

    def begin(*args):
        clock.now += 10
        observed.append(clock.now)
        original_begin(*args)

    def end(*args):
        clock.now += 10
        observed.append(clock.now)
        original_end(*args)

    backend.begin_with_location, backend.end = begin, end
    spans.sample_clock_if_due()
    sample = records[-1]
    assert sample['before_monotonic_ns'] <= observed[0] <= observed[1] <= sample['after_monotonic_ns']
    clock.now = sample['before_monotonic_ns'] + 1_000_000_000
    spans.sample_clock_if_due()
    assert spans.clock_sample_count == 1
    clock.now = sample['after_monotonic_ns'] + 1_000_000_000
    spans.sample_clock_if_due()
    assert spans.clock_sample_count == 2


def test_periodic_name_budget_is_bounded_and_cap_invalidates_only_diagnostics():
    spans, clock, records, backend = _sampling_spans()
    keys = set(vars(spans))
    for _ in range(2048):
        spans.sample_clock_if_due()
        clock.now += 1_000_000_000
    assert spans.clock_sample_count == 2048 and spans.error_count == 0
    spans.sample_clock_if_due()
    assert spans.clock_sample_count == 2048 and spans.error_count == 1
    assert records[-1]['event'] == 'diagnostic_error'
    assert records[-1]['first_error']['stage'] == 'clock_sample'
    assert not records[-1]['diagnostic_valid']
    previous = (clock.calls, len(records), len(backend.events))
    spans.sample_clock_if_due()
    with spans.zone('after-cap'):
        native_operation = 'still executed'
    assert native_operation == 'still executed'
    assert previous == (clock.calls, len(records), len(backend.events))
    assert set(vars(spans)) == keys and backend.stack == []
    assert not any(isinstance(value, list) for value in vars(spans).values())


@pytest.mark.parametrize('fault', ['begin', 'end', 'clock', 'regression', 'thread', 'emit'])
def test_periodic_errors_do_not_replace_native_exception_or_emit_more_zones(fault):
    spans, clock, records, backend = _sampling_spans()
    if fault in ('begin', 'end'):
        backend.fail = fault
    elif fault == 'clock':
        spans.clock_ns = lambda: (_ for _ in ()).throw(OSError('clock failed'))
    elif fault == 'regression':
        clock.now -= 1
    elif fault == 'thread':
        spans.anchor['native_thread_id'] = -1
    else:
        spans.emit = lambda *a, **kw: (_ for _ in ()).throw(OSError('log failed'))
    native = ValueError('native operation')
    with pytest.raises(ValueError) as caught:
        spans.sample_clock_if_due()
        raise native
    assert caught.value is native and spans.error_count
    previous = list(backend.events)
    spans.sample_clock_if_due()
    assert backend.events == previous and backend.stack == []
    if fault != 'emit':
        assert any(r['event'] == 'diagnostic_error' for r in records)


def test_periodic_clock_regression_after_a_not_due_poll_invalidates():
    spans, clock, records, _ = _sampling_spans()
    spans.sample_clock_if_due()
    clock.now += 500_000_000
    spans.sample_clock_if_due()
    clock.now -= 1
    spans.sample_clock_if_due()
    assert spans.error_count and records[-1]['event'] == 'diagnostic_error'


def test_inclusive_timing_totals_preserve_nested_exception_and_bound_output():
    wall, cpu, output = Clock(), Clock(), []
    spans = PythonSpans(timings=True, clock_ns=wall, cpu_clock_ns=cpu,
                        emit=lambda line, **kw: output.append(json.loads(line.split(' ', 1)[1])))
    original = KeyboardInterrupt('native cancellation')
    with pytest.raises(KeyboardInterrupt) as caught:
        with spans.zone('outer'):
            wall.now += 20
            cpu.now += 5
            with spans.zone('inner'):
                wall.now += 50
                cpu.now += 10
                raise original
    assert caught.value is original
    with spans.zone('outer'):
        wall.now += 30
        cpu.now += 7
    spans.sample_clock_if_due()
    assert not output
    wall.now += 10_000_000_000
    spans.sample_clock_if_due()
    spans.sample_clock_if_due()
    assert len(output) == 1
    outer, inner = output[0]['zones']['outer'], output[0]['zones']['inner']
    assert outer == {'count': 2, 'exceptions': 1, 'wall_total_ns': 100,
        'wall_min_ns': 30, 'wall_max_ns': 70, 'thread_cpu_total_ns': 22,
        'thread_cpu_min_ns': 7, 'thread_cpu_max_ns': 15}
    assert inner['wall_total_ns'] == 50 and inner['thread_cpu_total_ns'] == 10
    assert output[0]['diagnostic_valid'] and 'nested totals overlap' in output[0]['scope']
    for _ in range(1000):
        with spans.zone('inner'):
            wall.now += 1
            cpu.now += 1
    spans.report()
    assert len(output) == 2 and output[-1]['final']
    assert len(spans.timing_totals) == 2 and output[-1]['zones']['inner']['count'] == 1001


@pytest.mark.parametrize('stopped', [False, True])
def test_timing_only_preserves_real_loop_actions(stopped):
    off = _loop(enabled=False, stopped=stopped)
    on = _loop(enabled=False, timings=True, stopped=stopped)
    assert on[0] == off[0] and on[2].events == []
    assert len(on[1]) == len(off[1])
    if not stopped:
        np.testing.assert_array_equal(on[1][0], off[1][0])


def test_timing_only_factory_never_imports_native_profiler(monkeypatch):
    monkeypatch.delenv('CASCADE_ISAAC_PYTHON_SPANS', raising=False)
    monkeypatch.setenv('CASCADE_ISAAC_PYTHON_TIMINGS', '1')
    helper = runpy.run_path(str(HELPER))
    monkeypatch.setattr(helper['importlib'], 'import_module',
                        lambda name: pytest.fail('Timing-only mode imported a profiler'))
    spans = helper['from_environment'](str(REPO / 'scripts/isaac_bridge.py'))
    assert not spans.enabled and spans.timings and not spans.error_count
    assert len(spans.source_sha256) == len(spans.helper_sha256) == 64
    with spans.zone('bridge.loop'):
        pass
    assert spans.timing_totals['bridge.loop']['count'] == 1
    assert spans.timing_totals['bridge.loop']['wall_total_ns'] >= 0


@pytest.mark.parametrize('fault', ['capacity', 'clock', 'thread', 'emit'])
def test_timing_fault_only_invalidates_diagnostics(fault):
    wall, cpu, output = Clock(), Clock(), []
    spans = PythonSpans(timings=True, clock_ns=wall, cpu_clock_ns=cpu,
                        emit=lambda line, **kw: output.append(line))
    if fault == 'capacity':
        for i in range(32):
            with spans.zone(str(i)):
                pass
    elif fault == 'thread':
        spans.timing_thread_id = -1
    elif fault == 'emit':
        spans.emit = lambda *a, **kw: (_ for _ in ()).throw(OSError('broken sink'))
    native = ValueError('original operation')
    with pytest.raises(ValueError) as caught:
        with spans.zone('failing'):
            if fault == 'clock':
                wall.now -= 1
            if fault == 'emit':
                spans.report()
            raise native
    assert caught.value is native and spans.error_count
    assert len(spans.timing_totals) <= 32
    calls = wall.calls, cpu.calls
    with spans.zone('after-error'):
        pass
    assert (wall.calls, cpu.calls) == calls
    spans.report()
    if fault != 'emit':
        final = json.loads(output[-1].split(' ', 1)[1])
        # An invalid clock/thread cannot author a new trustworthy timestamp.
        assert final['event'] == ('diagnostic_error' if fault in ('clock', 'thread') else 'timing_summary')
        assert not final['diagnostic_valid']
