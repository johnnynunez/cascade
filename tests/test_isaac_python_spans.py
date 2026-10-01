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
    helper = runpy.run_path(str(HELPER))
    spans = helper['from_environment']('nonexistent-source-is-not-read')
    with spans.zone('bridge.loop'):
        with spans.zone('nested'):
            pass
    spans.anchor_once()
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


def _loop(*, enabled, failure=None, stopped=False, finger_step=None, partial=False):
    events, written = [], []
    backend = Backend(failure)
    spans = PythonSpans(enabled=enabled, backend=backend)
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

    env = dict(np=np, _profile_zone=spans.zone, _camera_video=None, _REQUIRE_CUDA=False,
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
                 'src/cascade/grasping/evidence.py'):
        assert '"scripts/isaac_python_spans.py"' in (REPO / path).read_text()
