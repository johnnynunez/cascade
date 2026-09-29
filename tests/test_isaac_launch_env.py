"""Isaac launch environment/signal adapter; real children but no Kit/GPU."""
from __future__ import annotations

import importlib.util
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cleanup_support(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("isaac_cleanup_support", ROOT / "scripts/install_support.py")
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    return support


@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.parametrize("rows,stopped", [
    ("424243 424242 Z\n900001 900001 S\n", True),
    ("900001 900001 S\n", True),
    ("424243 424242 S\n900001 900001 Z\n", False),
    ("424243 424242 Z\n424244 424242 S\n", False),
    ("424243 424242 Zunknown\n", False),
    ("424243 424242 Z", False),
    ("424243 424242\n", False),
    ("", False),
])
def test_owned_group_permission_error_requires_complete_exit_evidence(
        monkeypatch, denied_signal, rows, stopped):
    support = cleanup_support(monkeypatch)
    denied = PermissionError(errno.EPERM, "owned group signal denied")
    signals = []
    waits = []
    process = SimpleNamespace(pid=424242, wait=lambda **kwargs: waits.append(kwargs) or 17)

    def killpg(pgid, number):
        assert pgid == process.pid, "cleanup signalled a foreign group"
        signals.append(number)
        if number == denied_signal:
            raise denied

    def query(command, **kwargs):
        assert command == ["ps", "-A", "-o", "pid=,pgid=,stat="]
        assert kwargs["timeout"] > 0 and kwargs["capture_output"] and kwargs["text"]
        assert kwargs["env"]["LC_ALL"] == "C"
        return SimpleNamespace(returncode=0, stdout=rows, stderr="")

    monkeypatch.setattr(support.os, "killpg", killpg)
    monkeypatch.setattr(support.subprocess, "run", query)
    if stopped:
        support.stop_group(process)
        assert waits
    else:
        with pytest.raises(PermissionError) as error:
            support.stop_group(process)
        assert error.value is denied
    assert denied_signal in signals


@pytest.mark.parametrize("failure", [
    SimpleNamespace(returncode=1, stdout="", stderr=""),
    SimpleNamespace(returncode=0, stdout="424243 424242 Z\n", stderr="query incomplete"),
    subprocess.TimeoutExpired("ps", 5, output="424243 424242 Z\n"),
    PermissionError(errno.EACCES, "cannot query processes"),
    UnicodeError("invalid process table"),
])
def test_owned_group_query_failure_preserves_signal_error(monkeypatch, failure):
    support = cleanup_support(monkeypatch)
    denied = PermissionError(errno.EPERM, "original signal denial")
    def killpg(pgid, number):
        if number == signal.SIGKILL:
            raise denied
    def query(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(support.os, "killpg", killpg)
    monkeypatch.setattr(support.subprocess, "run", query)
    with pytest.raises(PermissionError) as error:
        support.stop_group(SimpleNamespace(pid=424242, wait=lambda **kwargs: 17))
    assert error.value is denied


def test_owned_group_other_permission_errors_are_not_suppressed(monkeypatch):
    support = cleanup_support(monkeypatch)
    denied = PermissionError(errno.EACCES, "access denied")
    def killpg(*args):
        raise denied
    monkeypatch.setattr(support.os, "killpg", killpg)
    monkeypatch.setattr(support.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected process query"))
    with pytest.raises(PermissionError) as error:
        support.stop_group(SimpleNamespace(pid=424242, wait=lambda **kwargs: 17))
    assert error.value is denied


def adapter():
    path = ROOT / "scripts/isaac_launch.py"
    assert path.exists(), "missing isolated source launcher"
    spec = importlib.util.spec_from_file_location("isaac_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isaac_environment_does_not_inherit_agent_python_or_profiler(tmp_path):
    module = adapter()
    source = tmp_path / "source"
    env = module.clean_environment({
        "HOME": str(tmp_path), "USER": "operator", "DISPLAY": ":1", "XAUTHORITY": "/auth",
        "XDG_RUNTIME_DIR": "/runtime", "OMNI_KIT_ACCEPT_EULA": "YES", "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONEXE": "/agent/python", "PYTHONHOME": "/agent", "PYTHONPATH": "/agent/src",
        "VIRTUAL_ENV": "/agent", "CONDA_PREFIX": "/conda", "LD_LIBRARY_PATH": "/agent/lib",
        "LD_PRELOAD": "/profiler/injected.so", "NSYS_PROFILING_SESSION_ID": "agent-session",
        "CASCADE_PHYSICS_DEVICE": "cuda:0", "ISAACSIM_PATH": str(source),
        "SECRET_API_TOKEN": "should-not-leak",
    }, source=str(source))
    for key in ("PYTHONEXE", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH", "NSYS_PROFILING_SESSION_ID", "SECRET_API_TOKEN"):
        assert key not in env
    assert env.get("LD_PRELOAD") != "/profiler/injected.so"
    assert env["DISPLAY"] == ":1" and env["XAUTHORITY"] == "/auth"
    assert env["ISAACSIM_PATH"] == str(source)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CASCADE_PHYSICS_DEVICE"] == "cuda:0"
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_environment_adapter_does_not_create_eula_consent():
    module = adapter()
    env = module.clean_environment({"HOME": "/home/test"}, source=None)
    assert "OMNI_KIT_ACCEPT_EULA" not in env


def test_launch_adapter_runs_real_child_with_exit_status_and_sanitized_env(tmp_path):
    adapter()
    child = tmp_path / "child.py"
    child.write_text('import os,json,sys; print(json.dumps(dict(os.environ))); sys.exit(17)\n')
    result = subprocess.run([sys.executable, str(ROOT / "scripts/isaac_launch.py"), "--python", sys.executable, "--", str(child)],
                            env={**os.environ, "PYTHONEXE": "/agent/python", "VIRTUAL_ENV": "/agent",
                                 "CASCADE_REQUIRE_CUDA": "1", "CASCADE_ISAAC_DT": "0.008333333333333333",
                                 "PAAI_CAMERA_VIDEO_CONFIG": str(tmp_path / "optional-video.json")},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 17
    env = json.loads(result.stdout)
    assert "PYTHONEXE" not in env and "VIRTUAL_ENV" not in env
    assert env["CASCADE_REQUIRE_CUDA"] == "1"
    assert env["CASCADE_ISAAC_DT"] == "0.008333333333333333"
    assert "PAAI_CAMERA_VIDEO_CONFIG" not in env


def test_term_reaches_the_source_wrappers_child_process(tmp_path):
    adapter()
    marker = tmp_path / "child.json"
    worker = tmp_path / "worker.py"
    worker.write_text('import os,json,pathlib,time\n' + f'pathlib.Path({str(marker)!r}).write_text(json.dumps({{"pid":os.getpid()}}))\n' + 'time.sleep(60)\n')
    shell = tmp_path / "python.sh"
    # Like Isaac's python.sh: it spawns Python, it does NOT exec it.
    shell.write_text(f'#!/bin/bash\n"{sys.executable}" "$@"\n')
    shell.chmod(0o755)
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/isaac_launch.py"), "--python", str(shell), "--", str(worker)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        pid = json.loads(marker.read_text())["pid"]
        process.terminate()
        assert process.wait(timeout=10) != 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
            if not status or "Z" in status:
                break
            time.sleep(0.02)
        assert not status or "Z" in status, "stopping source wrapper left its Kit child alive"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("ignore_term", [False, True])
def test_failed_launcher_group_cleanup_stops_managed_isaac(tmp_path, monkeypatch, ignore_term):
    support = cleanup_support(monkeypatch)
    marker = tmp_path / "kit.json"
    adapter_marker = tmp_path / "adapter.pid"
    worker = tmp_path / "synthetic_kit.py"
    worker.write_text(
        "import json,os,pathlib,signal,sys,time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") +
        "path = pathlib.Path(sys.argv[1])\n"
        "path.with_suffix('.tmp').write_text(json.dumps({\"pid\":os.getpid(),\"group\":os.getpgrp()}))\n"
        "path.with_suffix('.tmp').replace(path)\n"
        "time.sleep(60)\n"
    )
    launcher = tmp_path / "failed_launcher.py"
    launcher.write_text(
        "import pathlib,subprocess,sys,time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "adapter = subprocess.Popen([sys.executable,sys.argv[2],'--python',sys.executable,'--',sys.argv[3],str(marker)])\n"
        "pathlib.Path(sys.argv[4]).write_text(str(adapter.pid))\n"
        "deadline = time.monotonic() + 5\n"
        "while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)\n"
        "if not marker.exists(): raise RuntimeError('synthetic Kit failed to start')\n"
        "raise SystemExit(17)\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(launcher), str(marker), str(ROOT / "scripts/isaac_launch.py"),
         str(worker), str(adapter_marker)], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
    kit_pid = None
    try:
        assert process.wait(timeout=10) == 17
        identity = json.loads(marker.read_text())
        kit_pid = identity["pid"]
        # Exercise the real supervisor after its direct launcher has exited.
        support.stop_group(process)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = subprocess.run(["ps", "-p", str(kit_pid), "-o", "stat="],
                                    capture_output=True, text=True).stdout.strip()
            if not status or "Z" in status:
                break
            time.sleep(.02)
        assert not status or "Z" in status, "failed launcher cleanup orphaned its managed Isaac process"
        assert kit_pid == int(adapter_marker.read_text())
        assert identity["group"] == process.pid
        assert unrelated.poll() is None, "cleanup stopped an unrelated process"
    finally:
        # Only this fixture's group and recorded synthetic Kit can be signalled.
        try:
            support.stop_group(process)
        finally:
            if kit_pid is None and marker.exists():
                kit_pid = json.loads(marker.read_text())["pid"]
            if kit_pid is not None:
                status = subprocess.run(["ps", "-p", str(kit_pid), "-o", "stat="],
                                        capture_output=True, text=True, timeout=5).stdout.strip()
                if status and not status.startswith("Z"):
                    try:
                        os.kill(kit_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait(timeout=5)
            unrelated.terminate()
            unrelated.wait(timeout=5)
