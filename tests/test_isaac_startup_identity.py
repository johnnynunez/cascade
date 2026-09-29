"""Final-interpreter handshake across real execs; no Isaac or GPU imports."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="Isaac startup identity uses Linux /proc kernel births")

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    spec = importlib.util.spec_from_file_location("startup_adapter", ROOT / "scripts/isaac_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unsupported_readiness_platform_fails_before_process_lookup(tmp_path, monkeypatch, capsys):
    module = adapter()
    monkeypatch.setattr(module.sys, "platform", "unsupported")
    with pytest.raises(RuntimeError, match="requires Linux /proc"):
        module.wait_ready(42, tmp_path / "marker")
    monkeypatch.setattr(module.sys, "argv", ["isaac_launch.py", "--wait-ready", "42", "--ready-file", str(tmp_path / "marker")])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "requires Linux /proc" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def bridge_prefix(tmp_path):
    """Run the actual bridge entry through its marker, stopping before Kit."""
    source = (ROOT / "scripts/isaac_bridge.py").read_text()
    end = source.index("\npublish_ready()") + len("\npublish_ready()")
    bridge = tmp_path / "isaac_bridge.py"
    bridge.write_text(source[:end] + "\ntime.sleep(60)\n")
    shutil.copy2(ROOT / "scripts/isaac_launch.py", tmp_path / "isaac_launch.py")
    return bridge


def until(predicate, process, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert process.poll() is None, "fixture child exited before its gate"
        assert time.monotonic() < deadline
        time.sleep(.01)


def test_registration_waits_for_final_exec_and_keeps_original_kernel_birth(tmp_path, monkeypatch):
    from cascade.apps import process_owner as owners
    module = adapter()
    bridge = bridge_prefix(tmp_path)
    entered, release = tmp_path / "entered", tmp_path / "release"
    python = tmp_path / "managed-python"
    python.write_text(f"#!{sys.executable}\nimport os,pathlib,sys,time\n"
                      f"pathlib.Path({str(entered)!r}).touch()\n"
                      f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)\n"
                      "os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n")
    python.chmod(0o700)
    marker = tmp_path / "interpreter.json"
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/isaac_launch.py"),
        "--python", str(python), "--ready-file", str(marker), "--", str(bridge)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    state = tmp_path / "state"
    owner = owners.load_owner(state, ROOT, "fixture", create=True)
    observed = threading.Event()
    real_identity = module.kernel_identity
    def identity(pid):
        result = real_identity(pid)
        observed.set()
        return result
    monkeypatch.setattr(module, "kernel_identity", identity)
    try:
        until(entered.exists, process)
        before = real_identity(process.pid)
        assert "managed-python" in owners.process_identity(process.pid)["command"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(module.wait_ready, process.pid, marker, timeout=3)
            assert observed.wait(timeout=1)
            assert not result.done() and not marker.exists()
            assert owners.records(state) == []
            release.touch()
            ready = result.result(timeout=4)
        assert ready["launcher_pid"] == ready["bridge_pid"] == process.pid
        assert ready["launcher_birth"] == ready["bridge_birth"] == before["birth"]
        assert marker.stat().st_mode & 0o777 == 0o600
        record = owners.register_process(state, owner, process.pid, "isaac_bridge")
        assert "managed-python" not in record["command"]
        assert "isaac_launch.py" not in record["command"]
        assert str(bridge) in record["command"]
        assert owners.is_live(record, owner)
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)


def test_source_marker_distinguishes_stable_adapter_from_final_bridge(tmp_path):
    module = adapter()
    bridge = bridge_prefix(tmp_path)
    shell = tmp_path / "python.sh"
    shell.write_text(f'#!/bin/bash\n"{sys.executable}" "$@"\n')
    shell.chmod(0o700)
    marker = tmp_path / "interpreter.json"
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/isaac_launch.py"),
        "--python", str(shell), "--ready-file", str(marker), "--", str(bridge)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        ready = module.wait_ready(process.pid, marker, timeout=5)
        assert ready["launcher_pid"] == process.pid
        assert ready["bridge_pid"] != process.pid
        assert module.kernel_identity(ready["bridge_pid"])["birth"] == ready["bridge_birth"]
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)


@pytest.mark.parametrize("field,value", [("launcher_pid", 1), ("launcher_birth", "previous-process"),
                                         ("bridge_birth", "previous-bridge")])
def test_stale_marker_never_authorizes_registration(tmp_path, field, value):
    module = adapter()
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    marker = tmp_path / "interpreter.json"
    try:
        identity = module.kernel_identity(process.pid)
        record = {"launcher_pid": process.pid, "bridge_pid": process.pid,
                  "launcher_birth": identity["birth"], "bridge_birth": identity["birth"], field: value}
        marker.write_text(json.dumps(record))
        marker.chmod(0o600)
        with pytest.raises(RuntimeError, match="marker does not match|identity changed"):
            module.wait_ready(process.pid, marker, timeout=.1)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_wait_rejects_pid_reuse_instead_of_adopting_new_birth(tmp_path, monkeypatch):
    module = adapter()
    identities = iter([{"pid": 42, "birth": "first", "state": "R"},
                       {"pid": 42, "birth": "second", "state": "R"}])
    monkeypatch.setattr(module, "kernel_identity", lambda pid: next(identities))
    with pytest.raises(RuntimeError, match="PID was reused"):
        module.wait_ready(42, tmp_path / "missing", timeout=.1)


def test_timeout_is_bounded_and_does_not_restart_or_signal_child(tmp_path):
    module = adapter()
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            module.wait_ready(process.pid, tmp_path / "missing", timeout=.05)
        assert time.monotonic() - started < 1
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_launcher_reports_actual_early_child_exit_without_registration(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/isaac_launch.py", scripts / "isaac_launch.py")
    (scripts / "isaac_bridge.py").write_text("raise SystemExit(23)\n")
    source = (ROOT / "scripts/launch.sh").read_text()
    begin = source.index('            isaac_ready_dir="$(mktemp')
    end = source.index('            ownerctl record --pid "$bridge_pid"', begin)
    shell = ('set -euo pipefail\n'
             'die() { printf "%s\\n" "$*" >&2; exit 1; }\n'
             'isaac_args=()\n' + source[begin:end])
    result = subprocess.run(["bash", "-c", shell], env={**os.environ,
        "REPO": str(tmp_path), "STATE_DIR": str(tmp_path), "PY": sys.executable, "ISAAC_PY": sys.executable},
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "exited with status 23 before interpreter readiness" in result.stderr
    assert len(list(tmp_path.glob("isaac-startup.*"))) == 1
    assert not list(tmp_path.glob("isaac-startup.*/interpreter.json"))
