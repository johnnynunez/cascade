"""Real process identities/stdio MCP startup; no simulator or GPU claims."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
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
