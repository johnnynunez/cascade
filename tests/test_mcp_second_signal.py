"""A second signal that lands while the MCP worker handles the first one ends in the right path (#436 review).

The serial worker handles a signal in its `except SignalRequest` branch: stop_now(), then for SIGUSR1 the
operator reset (reset_now(), which drives an fr3 home) and its server.log line, then the reply to the call
it interrupted. Main ran that branch outside any `try`: a second signal there raised a SignalRequest that
escaped the loop. A real mock-arm server sent SIGUSR1 then SIGTERM back to back exited 1 with a traceback
instead of 143 (no e-stop latch from the SIGTERM path), and the SIGUSR1 line was lost.

Now every signal is handled inside the loop's `try`: a later signal folds into the one being handled
(SIGTERM always wins; a newer SIGINT / SIGUSR1 starts over with its own stop), the operator still gets a
SIGUSR1 line, and nothing is ever wrapped in `signals.defer()` (an fr3 reset drives home: a SIGTERM must
still cut it).
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from conftest import REPO, has_pinocchio

TERM, INT, USR1 = int(signal.SIGTERM), int(signal.SIGINT), int(signal.SIGUSR1)
NAMES = {TERM: "SIGTERM", INT: "SIGINT", USR1: "SIGUSR1"}


def _land(signum) -> None:
    """Deliver `signum` to this process and run its handler here, on the main thread (as a real signal
    that lands while this code runs): the handler raises at the loop's next bytecode boundary."""
    os.kill(os.getpid(), signum)
    for _ in range(100_000):
        pass


def _fake_server(events, *, land=None):
    """The worker's view of a server. `land=(where, signum)`: the first time `where` runs ("stop_now",
    "reset_now" or "reset_line"), `signum` lands inside it (after a reset_now did its work)."""
    landed = []

    def maybe_land(where):
        if land is not None and land[0] == where and not landed:
            landed.append(where)
            _land(land[1])

    def stop_now(**_kw):
        events.append("stop")
        maybe_land("stop_now")
        return {"ok": True}

    def reset_now():
        events.append("reset")
        result = {"content": [{"type": "text", "text": json.dumps({"ok": True, "domains": {}})}]}
        maybe_land("reset_now")
        return result

    server = SimpleNamespace(
        stop_now=stop_now, reset_now=reset_now, _cancel_lock=threading.Lock(), _cancelled_ids=set(), _inflight=None,
        _bounded=False, _dispatch_context=threading.local(), _stop_lock=threading.Lock(),
        _queued_arm_error=lambda name: None)
    return server, maybe_land


def _run(server, frames, monkeypatch, *, first, maybe_land=None):
    """The real worker loop on the main thread inside the real StopSignals; frame 1's tool call is cut by
    `first` (it lands inside the call, as a signal during a motion). Returns (rc or the escaped signal, replies)."""
    from cascade.apps import mcp_server
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    replies = []

    def handle(srv, msg):
        if msg.get("id") == 1:
            _land(first)
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": "{}"}]}}

    monkeypatch.setattr(mcp_server, "handle_message", handle)
    if maybe_land is not None:
        real_line = mcp_server._operator_reset_line

        def line(result):
            text = real_line(result)
            maybe_land("reset_line")
            return text

        monkeypatch.setattr(mcp_server, "_operator_reset_line", line)
    inbox, eof = queue.Queue(), object()
    for frame in frames:
        inbox.put((frame, replies.append))
    inbox.put(eof)
    with StopSignals(staff_reset=True) as signals:
        try:
            rc = mcp_server._worker_loop(server, signals, inbox, eof)
        except SignalRequest as escaped:
            rc = f"escaped: SignalRequest({NAMES.get(escaped.signum, escaped.signum)})"
    return rc, replies


def _call(ident):
    return {"jsonrpc": "2.0", "id": ident, "method": "tools/call", "params": {"name": "move_home", "arguments": {}}}


# first signal, where the second lands, the second -> exit code, stop/reset sequence, SIGUSR1 lines
CASES = {
    "usr1-then-term-in-the-reset": (USR1, "reset_now", TERM, 143, ["stop", "reset", "stop"],
                                    ["SIGUSR1 operator reset interrupted by SIGTERM"]),
    "usr1-then-term-in-its-stop": (USR1, "stop_now", TERM, 143, ["stop", "stop"],
                                   ["SIGUSR1 operator reset interrupted by SIGTERM"]),
    "usr1-then-term-in-the-line": (USR1, "reset_line", TERM, 143, ["stop", "reset", "stop"],
                                   ["SIGUSR1 operator reset: stop cleared"]),
    "usr1-twice": (USR1, "reset_now", USR1, 0, ["stop", "reset", "stop", "reset"],
                   ["SIGUSR1 operator reset interrupted by SIGUSR1", "SIGUSR1 operator reset: stop cleared"]),
    "term-then-usr1-in-its-stop": (TERM, "stop_now", USR1, 143, ["stop", "stop"], []),
    "int-then-term-in-its-stop": (INT, "stop_now", TERM, 143, ["stop", "stop"], []),
    "int-then-usr1-in-its-stop": (INT, "stop_now", USR1, 0, ["stop", "stop", "reset"],
                                  ["SIGUSR1 operator reset: stop cleared"]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_second_signal_while_the_first_is_handled(case, monkeypatch, capsys):
    """Main: the second signal escaped the worker (exit 1, traceback, no SIGTERM path). Now SIGTERM wins
    whenever it comes; a later SIGINT / SIGUSR1 is handled after the first (its own stop first)."""
    first, where, second, want_rc, want_events, want_lines = CASES[case]
    events = []
    server, maybe_land = _fake_server(events, land=(where, second))
    capsys.readouterr()
    rc, replies = _run(server, [_call(1), _call(2)], monkeypatch, first=first, maybe_land=maybe_land)
    err = capsys.readouterr().err
    assert rc == want_rc, (rc, events, err)
    assert events == want_events, events
    lines = [line for line in err.splitlines() if "SIGUSR1" in line]
    assert len(lines) == len(want_lines) and all(want in got for want, got in zip(want_lines, lines)), lines
    if want_rc == 143:
        assert replies == [], replies  # SIGTERM: the server exits; the transports end every waiting slot
    else:
        # the interrupted call gets its one "re-issue" reply, the next call runs normally
        assert [r["id"] for r in replies] == [1, 2], replies
        assert replies[0]["result"]["isError"] is True and "interrupted by signal" in json.dumps(replies[0])


def test_a_sigterm_after_a_completed_sigusr1_reset_still_logs_the_reset(monkeypatch, capsys):
    """The reset finished, its line not yet printed, when the SIGTERM landed: the operator still reads
    what the reset did, then the server takes the SIGTERM path."""
    events = []
    server, maybe_land = _fake_server(events, land=("reset_line", TERM))
    capsys.readouterr()
    rc, _ = _run(server, [_call(1)], monkeypatch, first=USR1, maybe_land=maybe_land)
    err = capsys.readouterr().err
    assert rc == 143 and events == ["stop", "reset", "stop"], (rc, events)
    assert "SIGUSR1 operator reset: stop cleared" in err and "then SIGTERM" in err, err


def test_a_sigterm_whose_request_a_tool_swallowed_still_takes_the_sigterm_path(monkeypatch):
    """A second signal that cuts a `finally` turns the first SignalRequest into another exception (e.g.
    "RuntimeError: release unlocked lock"), and a tool's error handling ("never kill the server on one
    bad message") swallows it. The handler recorded the SIGTERM (`signals.signum`): the worker takes
    the SIGTERM path at its next turn instead of serving on."""
    from cascade.apps import mcp_server
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    events = []
    server, _ = _fake_server(events)

    def handle(srv, msg):
        if msg["id"] == 1:
            try:
                _land(TERM)
            except SignalRequest:
                pass  # as a tool's `except Exception` around a replaced request
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": "{}"}]}}

    monkeypatch.setattr(mcp_server, "handle_message", handle)
    inbox, eof, replies = queue.Queue(), object(), []
    for i in (1, 2, 3):
        inbox.put((_call(i), replies.append))
    inbox.put(eof)
    with StopSignals(staff_reset=True) as signals:
        rc = mcp_server._worker_loop(server, signals, inbox, eof)
    assert rc == 143 and events == ["stop"], (rc, events)
    assert [r["id"] for r in replies] == [1], replies  # nothing after the SIGTERM ran


def test_a_signal_request_the_inbox_wait_replaced_is_still_that_signal(monkeypatch, capsys):
    """queue.Queue.get's Condition.wait re-acquires its lock in a `finally`: a second signal there turns
    the first request into "RuntimeError: release unlocked lock" (the requests in its context). The
    worker reads the signal behind it (SIGTERM wins), never crashes on it."""
    from cascade.apps import mcp_server
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    events = []
    server, _ = _fake_server(events)

    eof = object()

    class Inbox:
        raised = False

        def get(self, block=True, timeout=None):
            if self.raised:
                return eof  # a worker that served on ends here (rc 0)
            self.raised = True
            try:
                try:
                    raise SignalRequest(USR1)
                finally:
                    raise SignalRequest(TERM)
            except SignalRequest:
                raise RuntimeError("release unlocked lock")

    capsys.readouterr()
    with StopSignals(staff_reset=True) as signals:
        try:
            rc = mcp_server._worker_loop(server, signals, Inbox(), eof)
        except BaseException as escaped:  # noqa: BLE001
            rc = f"escaped: {escaped!r}"
    assert rc == 143 and events == ["stop"], (rc, events)
    assert "SIGUSR1 operator reset interrupted by SIGTERM" in capsys.readouterr().err  # the reset never ran


# ── a signal whose handler runs at one of the loop's back edges ─────────────────────────────────────

def _exposed_back_edges(code):
    """The backward jumps of `code` where a pending handler's exception lands outside every `try`.
    CPython 3.12 runs a pending handler at a backward jump AFTER jumping: the exception is located one
    code unit before the jump's target (`while True:` + `try:` in its body: before the `try`)."""
    import dis

    entries = dis.Bytecode(code).exception_entries
    return [(ins.offset, ins.argval) for ins in dis.get_instructions(code)
            if ins.opname == "JUMP_BACKWARD" and not any(e.start <= ins.argval - 2 < e.end for e in entries)]


def test_the_worker_loop_jumps_back_outside_its_try_only_where_the_handler_only_records():
    """A loop whose only bytecode boundary is its back edge, `while True: try: ...`: a signal there
    escapes 20 of 20 (the helper below); `try: while True: ...` catches it 20 of 20. Main's worker
    jumped back outside its `try` after every frame, every idle wake and every handled signal. Now
    only the back edge after a handled signal does, and the handler only records across it."""
    from cascade.apps import mcp_server
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    def exposed(stop):
        while True:
            try:
                if stop[0]:
                    return "done"
            except SignalRequest:
                return "caught"

    def covered(stop):
        try:
            while True:
                if stop[0]:
                    return "done"
        except SignalRequest:
            return "caught"

    seen = {}
    with StopSignals(staff_reset=True):
        for loop in (exposed, covered):
            outcomes = []
            for _ in range(5):
                stop = [False]

                def kick():
                    time.sleep(0.01)
                    os.kill(os.getpid(), signal.SIGUSR1)  # its handler runs on the main thread, in the loop
                    time.sleep(0.05)
                    stop[0] = True

                kicker = threading.Thread(target=kick, daemon=True)
                kicker.start()
                try:
                    outcomes.append(loop(stop))
                except SignalRequest:
                    outcomes.append("escaped")
                kicker.join(5)
            seen[loop.__name__] = (set(outcomes), _exposed_back_edges(loop.__code__))
    assert seen["exposed"][0] == {"escaped"} and len(seen["exposed"][1]) == 1, seen
    assert seen["covered"][0] == {"caught"} and seen["covered"][1] == [], seen
    exposed_edges = _exposed_back_edges(mcp_server._worker_loop.__code__)
    assert len(exposed_edges) == 1, f"back edges outside the worker's try: {exposed_edges}"


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="sys.monitoring needs Python 3.12")
def test_a_sigterm_at_the_back_edge_after_a_handled_signal_takes_the_sigterm_path(monkeypatch):
    """The one back edge outside the worker's `try` (after a handled signal): a SIGTERM whose handler
    runs exactly there (a sys.monitoring JUMP hook delivers it) is recorded, then handled at the top of
    the loop: STOP, 143. Main: it escaped the worker."""
    from cascade.apps import mcp_server
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    monkeypatch.setattr(mcp_server, "SIGNAL_WAKE_S", 0.01, raising=False)
    code = mcp_server._worker_loop.__code__
    edges = _exposed_back_edges(code)
    assert edges
    outcomes = {}
    for edge in edges:
        events = []
        server, _ = _fake_server(events)

        def handle(srv, msg):
            if msg.get("id") == 1:
                _land(USR1)  # a handled SIGUSR1: the loop goes back over that edge
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": "{}"}]}}

        monkeypatch.setattr(mcp_server, "handle_message", handle)
        inbox, eof, fired = queue.Queue(), object(), []
        for i in (1, 2):
            inbox.put((_call(i), lambda resp: None))
        inbox.put(eof)
        mon = sys.monitoring
        tool = next(i for i in range(6) if mon.get_tool(i) is None)

        def on_jump(c, src, dst, edge=edge, fired=fired):
            if c is code and (src, dst) == edge and not fired:
                fired.append(1)
                os.kill(os.getpid(), signal.SIGTERM)  # its handler runs here: as at the jump itself

        mon.use_tool_id(tool, "cascade-test-back-edge")
        try:
            mon.register_callback(tool, mon.events.JUMP, on_jump)
            mon.set_local_events(tool, code, mon.events.JUMP)
            with StopSignals(staff_reset=True) as signals:
                try:
                    rc = mcp_server._worker_loop(server, signals, inbox, eof)
                except SignalRequest as escaped:
                    rc = f"escaped: SignalRequest({NAMES.get(escaped.signum, escaped.signum)})"
        finally:
            mon.set_local_events(tool, code, 0)
            mon.register_callback(tool, mon.events.JUMP, None)
            mon.free_tool_id(tool)
        outcomes[edge] = (bool(fired), rc, events)
    assert all(fired and rc == 143 and events[-1] == "stop" for fired, rc, events in outcomes.values()), outcomes


# ── the real server process ───────────────────────────────────────────────────────────────────────────

def _env(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    env.update(CASCADE_CAMERA="mock", CASCADE_ARM="mock", CASCADE_STREAM="0", CASCADE_PREWARM="0",
               CASCADE_RUN_DIR=str(tmp_path / "run"), PYTHONDONTWRITEBYTECODE="1")
    return env


@pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")
@pytest.mark.parametrize("round_", range(3))
def test_sigusr1_then_sigterm_back_to_back_to_a_real_arm_server(tmp_path, round_):
    """The #436 review row: a mock-arm server, e-stopped, gets SIGUSR1 then SIGTERM back to back. Main:
    exit 1 with a SignalRequest traceback, the SIGUSR1 line lost. Now: the SIGTERM path (e-stop latch,
    teardown receipt, exit 143) and a SIGUSR1 line in server.log."""
    from test_mcp_lost_signal import _readline

    proc = subprocess.Popen([sys.executable, "-m", "cascade.apps.mcp_server"], cwd=REPO, env=_env(tmp_path),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        frames = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                  {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_observation", "arguments": {}}},
                  {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "emergency_stop", "arguments": {}}}]
        for frame in frames:
            proc.stdin.write((json.dumps(frame) + "\n").encode())
            proc.stdin.flush()
            reply = _readline(proc, 120)
            while '"id"' not in reply:  # notifications first
                reply = _readline(proc, 120)
            assert json.loads(reply)["id"] == frame["id"], reply
        os.kill(proc.pid, signal.SIGUSR1)
        os.kill(proc.pid, signal.SIGTERM)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc = "still running 30 s after the SIGTERM"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        err = proc.stderr.read().decode(errors="replace")
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()
    assert rc == 128 + signal.SIGTERM, (rc, err[-2500:])
    assert "Traceback" not in err and "SignalRequest" not in err, err[-2500:]
    assert "EMERGENCY STOP latched" in err, err[-2500:]
    assert any("SIGUSR1 operator reset" in line for line in err.splitlines()), err[-2500:]
    receipt = json.loads((tmp_path / "run" / "teardown.json").read_text())
    assert receipt["ok"] is True and receipt["complete"] is True, receipt
