"""Real process identities/stdio MCP startup; no simulator or GPU claims."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]


def owner_module():
    assert importlib.util.find_spec("cascade.apps.process_owner") is not None, "Missing per-profile process ownership registry"
    from cascade.apps import process_owner
    return process_owner


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("test process did not publish its ownership record")


def test_mcp_startup_registers_live_identity_and_unique_run_dir(tmp_path):
    owners = owner_module()
    state = owners.profile_state_dir(tmp_path, "demo-a")
    owner = owners.load_owner(state, REPO, "demo-a", create=True)
    env = {**os.environ, "CASCADE_PREWARM": "0", "CASCADE_OPENCLAW_PROFILE": "demo-a"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "cascade.apps.mcp_server", "--launch-owner", owner["owner"], "--launch-state-dir", str(state)],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO,
    )
    try:
        records = wait_for(lambda: owners.live_records(state, owner, role="mcp"))
        assert len(records) == 1
        record = records[0]
        assert record["pid"] == proc.pid
        assert record["repo"] == str(REPO.resolve())
        assert record["birth"] and record["command"]
        assert "--launch-owner " + owner["owner"] in record["command"]
        assert Path(record["run_dir"]).parent == REPO / "runs"
        assert Path(record["run_dir"]).name.startswith(f"mcp_{proc.pid}_")
        assert owners.is_live(record, owner)
        assert not owners.is_live({**record, "birth": "stale-pid"}, owner)
        assert not owners.is_live({**record, "command": record["command"] + " impostor"}, owner)
        assert not owners.is_live({**record, "owner": "someone-else"}, owner)
        proc.stdin.write('{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["id"] == 1
    finally:
        proc.communicate(timeout=10)
    assert owners.live_records(state, owner, role="mcp") == []
    # The process test leaves no repo log artifacts behind.
    import shutil
    shutil.rmtree(record["run_dir"])


def test_profiles_and_unprofiled_never_share_an_owner(tmp_path):
    owners = owner_module()
    states = [owners.profile_state_dir(tmp_path, p) for p in ("", "default", "demo-a", "demo-b")]
    assert len(set(states)) == 4
    a = owners.load_owner(states[2], REPO, "demo-a", create=True)
    assert owners.load_owner(states[2], REPO, "demo-a") == a
    with pytest.raises(ValueError, match="owner|profile"):
        owners.load_owner(states[2], REPO, "demo-b")
    with pytest.raises(ValueError, match="profile"):
        owners.profile_state_dir(tmp_path, "../escape")
    assert owners.load_owner(states[3], REPO, "demo-b") is None
    assert not states[3].exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin kernel start-time contract")
def test_mac_birth_identity_uses_kernel_microseconds_not_ps_calendar_seconds():
    import ctypes
    import struct

    owners = owner_module()
    buf = ctypes.create_string_buffer(136)  # struct proc_bsdinfo, PROC_PIDTBSDINFO
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    assert lib.proc_pidinfo(os.getpid(), 3, 0, buf, len(buf)) == len(buf)
    sec, usec = struct.unpack_from("=QQ", buf.raw, 120)
    identity = owners.process_identity(os.getpid())
    assert identity["birth"] == f"darwin:{sec}:{usec}", "second-resolution ps lstart can collide after PID reuse"


def shutdown_child(*, ignore_term):
    behavior = "signal.SIG_IGN" if ignore_term else "lambda *args: sys.exit(0)"
    child = subprocess.Popen([sys.executable, "-c",
        "import os,signal,sys,time; "
        f"signal.signal(signal.SIGTERM,{behavior}); os.write(1,b'R'); time.sleep(60)"],
        stdout=subprocess.PIPE)
    assert select.select([child.stdout], [], [], 5)[0]
    assert child.stdout.read(1) == b"R"
    return child


def test_down_escalates_only_owned_ignoring_child_and_keeps_foreign_and_stale_pids(tmp_path, monkeypatch):
    owners = owner_module()
    state = owners.profile_state_dir(tmp_path, "demo-a")
    owner = owners.load_owner(state, REPO, "demo-a", create=True)
    foreign_state = owners.profile_state_dir(tmp_path, "demo-b")
    foreign_owner = owners.load_owner(foreign_state, REPO, "demo-b", create=True)
    children = [shutdown_child(ignore_term=True) for _ in range(3)]
    owned, foreign, stale = children
    original_kill, original_stop = owners.os.kill, owners.stop_owned_process
    signals = []
    def record_signal(pid, number):
        signals.append((pid, number))
        return original_kill(pid, number)
    try:
        owners.register_process(state, owner, owned.pid, "isaac_bridge")
        owners.register_process(foreign_state, foreign_owner, foreign.pid, "isaac_bridge")
        receipt = owners.register_process(state, owner, stale.pid, "qwen")
        (state / "qwen.pid").write_text(json.dumps({**receipt, "birth": "previous-process"}))
        monkeypatch.setattr(owners.os, "kill", record_signal)
        monkeypatch.setattr(owners, "stop_owned_process", lambda record, expected:
                            original_stop(record, expected, term_timeout_s=.1, kill_timeout_s=2))
        assert owners.stop_owned(state, owner) == [owned.pid]
        assert owned.wait(timeout=3) == -signal.SIGKILL
        assert foreign.poll() is None and stale.poll() is None
        assert signals == [(owned.pid, signal.SIGTERM), (owned.pid, signal.SIGKILL)]
        assert owners.stop_owned(state, owner) == []
        assert (state / "isaac_bridge.pid").exists(), "retain the inert receipt for diagnosis"
    finally:
        for child in children:
            if child.poll() is None:
                original_kill(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            child.stdout.close()


def test_graceful_owned_child_never_receives_sigkill(tmp_path, monkeypatch):
    owners = owner_module()
    owner = owners.load_owner(tmp_path, REPO, "fixture", create=True)
    child = shutdown_child(ignore_term=False)
    original_kill = owners.os.kill
    signals = []
    def record_signal(pid, number):
        signals.append((pid, number))
        return original_kill(pid, number)
    try:
        record = owners.register_process(tmp_path, owner, child.pid, "isaac_bridge")
        monkeypatch.setattr(owners.os, "kill", record_signal)
        owners.stop_owned_process(record, owner, term_timeout_s=2, kill_timeout_s=1)
        assert child.wait(timeout=3) == 0
        assert signals == [(child.pid, signal.SIGTERM)]
    finally:
        if child.poll() is None:
            original_kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        child.stdout.close()


def test_shutdown_rechecks_birth_before_escalating_reused_pid(monkeypatch):
    owners = owner_module()
    owner = {"owner": "fixture", "repo": "/fixture", "profile": "fixture", "state_dir": "/fixture/state"}
    record = {**owner, "pid": 424242, "birth": "original", "command": "fixture-child", "role": "isaac_bridge"}
    current = {key: record[key] for key in ("pid", "birth", "command")}
    signals = []
    monkeypatch.setattr(owners, "process_identity", lambda pid: dict(current))
    def replace_after_term(pid, number):
        signals.append((pid, number))
        current["birth"] = "replacement"
    monkeypatch.setattr(owners.os, "kill", replace_after_term)
    owners.stop_owned_process(record, owner, term_timeout_s=0, kill_timeout_s=0)
    assert signals == [(record["pid"], signal.SIGTERM)]


def test_shutdown_keeps_real_failure_and_receipt_if_pid_survives_both_signals(tmp_path, monkeypatch):
    owners = owner_module()
    owner = owners.load_owner(tmp_path, REPO, "fixture", create=True)
    record = {**owner, "pid": 424242, "birth": "original", "command": "fixture-child", "role": "isaac_bridge"}
    receipt = tmp_path / "isaac_bridge.pid"
    receipt.write_text(json.dumps(record))
    before = receipt.read_bytes()
    signals = []
    monkeypatch.setattr(owners, "is_live", lambda *args: True)
    monkeypatch.setattr(owners.os, "kill", lambda pid, number: signals.append((pid, number)))
    with pytest.raises(ValueError, match="did not stop after SIGTERM and SIGKILL; receipt retained"):
        owners.stop_owned_process(record, owner, term_timeout_s=0, kill_timeout_s=0)
    assert signals == [(record["pid"], signal.SIGTERM), (record["pid"], signal.SIGKILL)]
    assert receipt.read_bytes() == before
