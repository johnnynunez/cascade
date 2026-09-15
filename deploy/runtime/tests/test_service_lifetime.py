"""Managed services outlive old leases and still stop through their owners."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import runtime
import web_runtime

ELAPSED = 19 * 3600


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', HERE.parents[1])
    monkeypatch.setattr(runtime, 'HERE', HERE)
    monkeypatch.setenv('PAAI_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(runtime, 'remote_only', lambda: None)
    monkeypatch.setattr(runtime, 'app_python', lambda: Path(sys.executable))
    deployment = {'environment': {'OPENCLAW_PROFILE': 'lifetime-test'},
                  'launch_state': str(tmp_path / 'launch'),
                  'web_bind': '127.0.0.1'}
    runtime.atomic(runtime.state_dir() / 'deployment.json', deployment)
    with patch.dict(os.environ):
        yield deployment


@pytest.mark.parametrize('legacy_lease', ['expired', 'missing'])
@pytest.mark.parametrize('ending', ['term', 'interrupt', 'child_exit'])
def test_supervisor_outlives_legacy_lease_and_reaps_only_after_stop(managed, legacy_lease, ending):
    from cascade.apps import process_owner

    lease = runtime.state_dir() / 'lease.json'
    if legacy_lease == 'expired':
        runtime.atomic(lease, {'until': 1})
    initial_lease = lease.read_bytes() if lease.exists() else None
    clock = SimpleNamespace(now=1000.0)
    signals, records, children, logs, exits = {}, {}, {}, [], {}
    supervisor = {'pid': os.getpid(), 'argv': ['supervisor'], 'cwd': str(runtime.ROOT)}
    descendant = {'pid': 9000, 'argv': ['owned-descendant'], 'cwd': str(runtime.ROOT)}
    cleanup = MagicMock()
    health_calls = []

    def launch(command, **kwargs):
        role, pid = command[-3], 5000 + len(records)
        records[pid] = {'pid': pid, 'argv': command, 'cwd': str(kwargs['cwd'])}
        child = SimpleNamespace(pid=pid, poll=lambda: exits.get(role), wait=MagicMock())
        children[role] = child
        logs.append(kwargs['stdout'])
        return child

    class Stopped:
        value = False

        def set(self):
            self.value = True

        def is_set(self):
            return self.value

        def wait(self, seconds):
            assert seconds == 12
            assert len(health_calls) <= 2
            cleanup.assert_not_called()
            if len(health_calls) == 1:
                clock.now += ELAPSED
                if ending == 'child_exit':
                    exits['cameras'] = 1
            else:
                signum = signal.SIGTERM if ending == 'term' else signal.SIGINT
                signals[signum](signum, None)
            return self.value

    stopped = Stopped()

    def healthy():
        health_calls.append(clock.now)
        cleanup.assert_not_called()
        return {'memory': {'available_mib': 2048}}

    fake_time = SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now,
                                sleep=lambda _: None)
    with ExitStack() as stack:
        replacements = {
            'time': fake_time, 'threading': SimpleNamespace(Event=lambda: stopped),
            'identity': lambda pid: supervisor if pid == os.getpid() else records.get(pid),
            'process_tree': lambda _: [supervisor, *records.values(), descendant],
            'terminate_tree': cleanup, 'brain_probe': lambda _: None,
            'native_brain_probe': lambda _: None, 'simulator_probe': lambda: None,
            'camera_probe': lambda: None, 'http': lambda *a, **k: 200,
            'deep_probe': lambda _: None, 'smoke': lambda: None, 'health': healthy,
        }
        for name, value in replacements.items():
            stack.enter_context(patch.object(runtime, name, value))
        stack.enter_context(patch.object(runtime.ctypes, 'CDLL', return_value=SimpleNamespace(prctl=lambda *a: 0)))
        stack.enter_context(patch.object(runtime.signal, 'signal', side_effect=lambda sig, fn: signals.setdefault(sig, fn)))
        stack.enter_context(patch.object(runtime.subprocess, 'Popen', side_effect=launch))
        stack.enter_context(patch.object(process_owner, 'load_owner', return_value={}))
        stack.enter_context(patch.object(process_owner, 'register_process'))
        result = runtime.serve(60)

    receipt = runtime.read(runtime.state_dir() / 'supervisor.json')
    if ending == 'child_exit':
        assert result == 1
        assert receipt['phase'] == 'failed'
        assert receipt['reason'] == 'A required component exited'
        assert health_calls == [1000.0]
        assert not stopped.is_set()
    else:
        assert health_calls == [1000.0, 1000.0 + ELAPSED]
        assert stopped.is_set()
        assert result == 0 and receipt['phase'] == 'stopped'
    cleanup.assert_called_once_with([*records.values(), descendant])
    assert len(children) == 5
    for child in children.values():
        child.wait.assert_called_once_with(timeout=5)
    assert all(log.closed for log in logs)
    assert (lease.read_bytes() if lease.exists() else None) == initial_lease


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_camera_worker_outlives_18_hours_until_signal(managed, signum):
    signals = {}
    rig = SimpleNamespace(state=lambda: {}, open=MagicMock(), close=MagicMock())
    server = SimpleNamespace(start=MagicMock(), stop=MagicMock())
    clock = SimpleNamespace(now=0)

    class Stopped:
        value = False

        def set(self):
            self.value = True

        def wait(self, timeout=None):
            clock.now += ELAPSED
            rig.close.assert_not_called()
            server.stop.assert_not_called()
            if timeout is not None and clock.now >= timeout:
                return False
            signals[signum](signum, None)
            return self.value

    stopped = Stopped()
    cameras = SimpleNamespace(RecoveringViewRig=MagicMock(return_value=rig),
                              KitchenStreamServer=MagicMock(return_value=server))
    with patch.dict(sys.modules, {'serve_isaac_view': cameras}), \
            patch.object(runtime, 'threading', SimpleNamespace(Event=lambda: stopped)), \
            patch.object(runtime.signal, 'signal', side_effect=lambda sig, fn: signals.setdefault(sig, fn)):
        result = runtime.worker('cameras')
    assert clock.now > 18 * 3600
    assert stopped.value, 'The camera worker exited solely because its lifetime elapsed'
    assert result == 0
    cameras.RecoveringViewRig.assert_called_once_with(stop_event=stopped)
    rig.open.assert_called_once()
    server.start.assert_called_once()
    rig.close.assert_called_once()
    server.stop.assert_called_once()


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_web_worker_outlives_18_hours_until_signal(managed, signum):
    runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())
    started = 0
    run_web = web_runtime.run_web

    async def start_site():
        nonlocal started
        started += 1

    async def accelerated(*args, **kwargs):
        loop = asyncio.get_running_loop()
        original_time = loop.time
        offset = 0
        signals = {}
        with patch.object(loop, 'time', side_effect=lambda: original_time() + offset), \
                patch.object(loop, 'add_signal_handler', side_effect=lambda sig, fn: signals.setdefault(sig, fn)):
            task = asyncio.create_task(run_web(*args, **kwargs))
            try:
                for _ in range(20):
                    await asyncio.sleep(0)
                    if started == 2:
                        break
                assert started == 2
                offset = ELAPSED
                for _ in range(8):
                    await asyncio.sleep(0)
                assert not task.done(), 'The web worker exited solely because its lifetime elapsed'
                runner.cleanup.assert_not_awaited()
                signals[signum]()
                return await asyncio.wait_for(task, timeout=1)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    site = MagicMock(return_value=SimpleNamespace(start=start_site))
    with patch.object(web_runtime.web, 'AppRunner', return_value=runner), \
            patch.object(web_runtime.web, 'TCPSite', site), \
            patch.object(web_runtime, 'run_web', side_effect=accelerated):
        assert runtime.worker('web') == 0
    runner.cleanup.assert_awaited_once()
