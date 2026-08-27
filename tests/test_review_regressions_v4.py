"""Regression pins for the 2026-07-20 adversarial review of the MCP stop
channel (booth prep). One test per confirmed defect, root cause in the
docstring — convention per tests/test_review_regressions*.py.
"""

import json
import signal
import time

import pytest

from conftest import has_pinocchio
from test_mcp_server import McpClient, _tool_payload

pytestmark = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")


def test_stop_during_startup_latches_after_build(tmp_path, monkeypatch):
    """CRITICAL: stop_now() used to answer ok/stopped:true while the runtime
    was still building and latch NOTHING — the queued motion then ran after
    the stop was acknowledged. _stop_pending must be set before reading
    _runtime and applied by _ensure_runtime after assigning it, so every
    interleaving is caught by one side."""
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    monkeypatch.setenv("CASCADE_VIEW", "0")
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    try:
        # stop acknowledged BEFORE any runtime exists (mid-build in prod)
        result = server.stop_now()
        payload = json.loads(result["content"][0]["text"])
        assert payload["stopped"]
        # the ack must promise the latch, not claim nothing can ever move
        assert "stop latched" in payload["note"]

        # the build must come up already stopped...
        runtime = server._ensure_runtime()
        assert runtime.arm.harness.estopped

        # ...and motion must be rejected until staff clears it
        res = server.call_tool("move_home", {})
        assert res["isError"]
        assert "e-stop" in json.loads(res["content"][0]["text"])["error"]

        server.reset_now()
        assert not runtime.arm.harness.estopped
        assert server._stop_pending is False
    finally:
        server.shutdown()


def test_cancel_before_dispatch_drops_the_call(tmp_path):
    """A notifications/cancelled that raced ahead of its tools/call used to
    be lost in the window between the worker's cancelled-ids check and its
    _inflight publication (TOCTOU): the cancelled MOTION call then executed.
    Now {check set / publish} and {read _inflight / add} share a lock, and a
    call cancelled before dispatch is dropped without a response."""
    c = McpClient(str(tmp_path / "run"))
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})

        # cancel the NEXT id, then send the motion call that will get it
        c.notify("notifications/cancelled", {"requestId": c._id + 1, "reason": "user"})
        c.send("tools/call", {"name": "move_home", "arguments": {}})

        # the dropped call gets no response: the next frame answers ping
        ping_id = c.send("ping")
        resp = c.recv()
        assert resp["id"] == ping_id, (
            f"cancelled call was executed/answered anyway: {resp}"
        )
    finally:
        c.close()


def test_garbage_and_batch_frames_do_not_kill_stop_channel(tmp_path):
    """A valid-JSON non-object frame (e.g. a JSON-RPC batch or bare string)
    used to raise AttributeError in the reader thread — killing the stdio
    stop channel and tearing the server down mid-session. The reader must
    survive any frame and unwrap batches."""
    c = McpClient(str(tmp_path / "run"))
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})

        c.proc.stdin.write('[1, 2, 3]\n')          # batch of non-objects
        c.proc.stdin.write('"just a string"\n')    # bare JSON scalar
        c.proc.stdin.write(
            '[{"jsonrpc": "2.0", "id": 9901, "method": "ping"}]\n'
        )  # batch containing a real request -> must be unwrapped
        c.proc.stdin.flush()

        resp = c.recv()
        assert resp["id"] == 9901 and resp["result"] == {}
        # and the normal path still works: the reader thread survived
        assert c.request("ping")["result"] == {}
    finally:
        c.close()


def test_exec_lock_serializes_chat_against_worker(tmp_path, monkeypatch):
    """CRITICAL pin (2026-07-20 second review): the MCP-mode dashboard
    reflex chat and the worker's skill execution share _exec_lock -- two
    threads must never stream the arm concurrently (it would defeat the
    per-waypoint velocity gate). The chat refuses while a command runs."""
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    monkeypatch.setenv("CASCADE_ARM", "mock")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("CASCADE_STREAM", "0")
    monkeypatch.setenv("CASCADE_VIEW", "0")
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    try:
        runtime = server._ensure_runtime()

        with server._exec_lock:  # simulate a worker mid-motion
            server._reflex_chat_task("wave")  # must refuse, not queue
        assert any("busy" in ev.text for ev in runtime.memory.events())
        assert runtime.last_path is None  # nothing executed

        server._reflex_chat_task("wave")  # lock free: executes LLM-free
        assert runtime.last_path == "reflex"
    finally:
        server.shutdown()


def test_sigusr1_is_the_staff_reset_channel(tmp_path):
    """With reset_stop hidden from attendees (CASCADE_HIDE_TOOLS), staff had NO
    way to clear a latched e-stop short of restarting the server. SIGUSR1
    now clears it (shell access to the rig == staff)."""
    c = McpClient(str(tmp_path / "run"),
                  extra_env={"CASCADE_HIDE_TOOLS": "reset_stop"})
    try:
        c.request("initialize", {"protocolVersion": "2025-06-18"})
        c.request("tools/call", {"name": "get_observation", "arguments": {}})

        payload, _ = _tool_payload(
            c.request("tools/call", {"name": "emergency_stop", "arguments": {}})
        )
        assert payload["stopped"]
        payload, is_err = _tool_payload(
            c.request("tools/call", {"name": "reset_stop", "arguments": {}})
        )
        assert is_err  # attendees cannot clear it

        c.proc.send_signal(signal.SIGUSR1)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            payload, is_err = _tool_payload(
                c.request("tools/call", {"name": "move_home", "arguments": {}})
            )
            if not is_err:
                break
            time.sleep(0.2)
        assert not is_err, f"e-stop still latched after SIGUSR1: {payload}"
    finally:
        c.close()
