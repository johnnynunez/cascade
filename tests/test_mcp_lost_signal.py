"""A stop signal the MCP server's main thread never woke up for is still handled (#435 review, ARM CI).

CPython runs a Python signal handler on the main thread, at its next bytecode boundary. A signal that
lands after the main thread released the GIL and before its wait started (or that the kernel delivers
to another thread) only sets CPython's C-level flag: a wait with no timeout never returns, so the
handler never runs. For the MCP server that wait was the serial worker's idle `inbox.get()`: a SIGTERM
there latched no e-stop, ran no teardown, wrote no receipt, and the server stayed up until a second
signal or SIGKILL (a CI orphan stayed alive 48 min; a gdb repro stopped the main thread at the futex
entry and hung 4 of 4, HTTP and HTTPS).

These tests build exactly that state without gdb: the signal goes to a helper thread
(`signal.pthread_kill` on itself). The C handler runs there and trips the flag; the main thread, idle
in the worker's queue, is never interrupted. Now the worker waits in bounded slices
(`mcp_server.SIGNAL_WAKE_S`), so the handler runs within one slice: STOP latched, the teardown and its
receipt, exit 143.
"""

from __future__ import annotations

import json
import os
import queue
import select
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO, has_pinocchio

#: The worker would wait forever on main: past this the test wakes it (and fails).
RESCUE_S = 3.0
TOKEN = "cascade-test-token-" + "k" * 24


def _main_thread_waits_in_the_queue(timeout_s: float = 10.0) -> bool:
    """True once the main thread's stack is inside `queue.get` (the worker's idle wait)."""
    main = threading.main_thread().ident
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, names = sys._current_frames().get(main), []
        while frame is not None:
            names.append((frame.f_code.co_name, frame.f_globals.get("__name__")))
            frame = frame.f_back
        if ("get", "queue") in names:
            return True
        time.sleep(0.01)
    return False


def _trip_on_a_helper_thread(signum) -> threading.Thread:
    """`signum` delivered to a helper thread once the main thread idles in the worker's queue: the C
    handler trips CPython's flag there and the main thread's wait is never interrupted."""
    def run():
        assert _main_thread_waits_in_the_queue()
        time.sleep(0.3)  # settled in its futex wait
        signal.pthread_kill(threading.get_ident(), signum)

    helper = threading.Thread(target=run, daemon=True, name="test-signal-on-a-helper-thread")
    helper.start()
    return helper


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
                         ids=["SIGTERM", "SIGINT", "SIGUSR1"])
def test_a_signal_that_never_woke_the_idle_worker_is_handled(signum):
    """SIGTERM: STOP latched, exit 143. SIGINT: STOP latched, the worker serves on. SIGUSR1 (staff
    reset): STOP then reset, serves on. Main: none of it until something else wakes the worker."""
    from cascade.apps.mcp_server import _worker_loop
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    inbox, eof, events = queue.Queue(), object(), []
    server = SimpleNamespace(stop_now=lambda: events.append(("stop", time.monotonic())),
                             reset_now=lambda: events.append(("reset", time.monotonic())))
    rescued = threading.Event()

    def rescue():
        # main: the worker waits forever; wake it (any queued item) so the test ends
        if not any(e[0] == "stop" for e in events):
            time.sleep(RESCUE_S)
        if not any(e[0] == "stop" for e in events):
            rescued.set()
        inbox.put(eof)

    with StopSignals(staff_reset=True) as signals:
        t0 = time.monotonic()
        helper = _trip_on_a_helper_thread(signum)
        waker = threading.Thread(target=rescue, daemon=True)
        if signum != signal.SIGTERM:
            # the loop goes on serving after a SIGINT / SIGUSR1: end it once the STOP is in
            def end_after_the_stop():
                deadline = time.monotonic() + RESCUE_S
                while not any(e[0] == "stop" for e in events) and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not any(e[0] == "stop" for e in events):
                    rescued.set()
                time.sleep(0.1)
                inbox.put(eof)

            waker = threading.Thread(target=end_after_the_stop, daemon=True)
        waker.start()
        try:
            code = _worker_loop(server, signals, inbox, eof)
        except SignalRequest as late:  # main: the handler ran only once the rescue woke the worker
            code = f"late {late.signum}"
        helper.join(5)
        waker.join(10)
    assert not rescued.is_set(), f"the {signal.Signals(signum).name} was lost: nothing ran until the worker was woken"
    stops = [t for kind, t in events if kind == "stop"]
    assert len(stops) == 1 and stops[0] - t0 < 2.0, events
    if signum == signal.SIGTERM:
        assert code == 128 + signal.SIGTERM
    else:
        assert code == 0
    assert [kind for kind, _ in events] == (["stop", "reset"] if signum == signal.SIGUSR1 else ["stop"])


def test_the_idle_worker_wakes_in_bounded_slices(monkeypatch):
    """Every idle wait of the worker carries a finite timeout no longer than SIGNAL_WAKE_S: a wait
    without one can miss a signal for good."""
    from cascade.apps import mcp_server

    waits, eof = [], object()

    class Inbox:
        def get(self, block=True, timeout=None):
            waits.append(timeout)
            if len(waits) < 3:
                raise queue.Empty
            return eof

    assert 0 < mcp_server.SIGNAL_WAKE_S <= 0.5
    assert mcp_server._worker_loop(SimpleNamespace(), SimpleNamespace(checkpoint=lambda: None), Inbox(), eof) == 0
    assert waits == [mcp_server.SIGNAL_WAKE_S] * 3, waits


def test_a_queued_frame_is_served_at_once_not_at_the_next_wake(monkeypatch):
    """The slices cost no latency: a frame put while the worker idles is handled on its arrival."""
    from cascade.apps import mcp_server

    inbox, eof, served = queue.Queue(), object(), []
    server = SimpleNamespace(_cancel_lock=threading.Lock(), _cancelled_ids=set(), _inflight=None,
                             _bounded=False, _dispatch_context=threading.local(), _stop_lock=threading.Lock(),
                             _queued_arm_error=lambda name: None)
    lags = []

    def feed():
        for i in range(5):
            assert _main_thread_waits_in_the_queue()
            time.sleep(0.05)
            inbox.put(({"jsonrpc": "2.0", "id": i, "method": "ping"},
                       lambda resp, t=time.monotonic(): lags.append(time.monotonic() - t)))
            while len(lags) <= i:
                time.sleep(0.001)
        inbox.put(eof)

    monkeypatch.setattr(mcp_server, "handle_message",
                        lambda srv, msg: served.append(msg["id"]) or {"jsonrpc": "2.0", "id": msg["id"], "result": {}})
    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    assert mcp_server._worker_loop(server, SimpleNamespace(checkpoint=lambda: None), inbox, eof) == 0
    feeder.join(10)
    assert served == list(range(5))
    # a worker that only looked every SIGNAL_WAKE_S would answer ~0.1-0.2 s late on average
    assert sorted(lags)[2] < mcp_server.SIGNAL_WAKE_S / 2, lags


def _main_thread_in(function: str, module: str, timeout_s: float = 10.0) -> bool:
    main = threading.main_thread().ident
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(main)
        while frame is not None:
            if frame.f_code.co_name == function and frame.f_globals.get("__name__") == module:
                return True
            frame = frame.f_back
        time.sleep(0.01)
    return False


def test_a_sigterm_while_the_first_call_waits_for_the_prewarm_build_is_handled(monkeypatch, tmp_path):
    """The worker's first tool call waits for the prewarm build (`_init_lock`). Main: a lock acquire
    with no timeout, so a SIGTERM whose wakeup was lost waited out the whole build (forever, for a
    build stuck in a camera open). Now the call waits in short sleeps and the signal runs at once."""
    from cascade.apps import demo, mcp_server
    from cascade.apps.signal_stop import SignalRequest, StopSignals

    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(demo, "build_runtime", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no build here")))
    server = mcp_server.McpSkillServer()
    stops = []
    monkeypatch.setattr(server, "stop_now", lambda **_kw: stops.append(time.monotonic()))
    building, release = threading.Event(), threading.Event()

    def build():  # the prewarm build, in flight
        with server._init_lock:
            building.set()
            release.wait(RESCUE_S)

    def trip():
        assert _main_thread_in("_ensure_runtime", "cascade.apps.mcp_server")
        time.sleep(0.3)
        signal.pthread_kill(threading.get_ident(), signal.SIGTERM)

    builder = threading.Thread(target=build, daemon=True)
    builder.start()
    assert building.wait(5)
    inbox, eof, replies = queue.Queue(), object(), []
    inbox.put(({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "world_state", "arguments": {}}}, replies.append))
    with StopSignals(staff_reset=True) as signals:
        server._signals = signals
        t0 = time.monotonic()
        helper = threading.Thread(target=trip, daemon=True)
        helper.start()
        try:
            code = mcp_server._worker_loop(server, signals, inbox, eof)
        except SignalRequest as late:
            code = f"late {late.signum}"
        finally:
            took = time.monotonic() - t0
            release.set()
            builder.join(5)
            helper.join(5)
    assert code == 128 + signal.SIGTERM and len(stops) == 1, (code, stops)
    assert took < 2.0, f"the SIGTERM waited {took:.1f} s for the build"


def test_the_teardown_waits_for_a_listener_that_never_stops_only_so_long(monkeypatch):
    """`httpd.shutdown()` waits for serve_forever without a bound: the teardown asks from a helper
    thread, waits TRANSPORT_STOP_S, then closes the socket and goes on."""
    from cascade.apps import mcp_server

    monkeypatch.setattr(mcp_server, "TRANSPORT_STOP_S", 0.3)
    stuck, closed = threading.Event(), []
    httpd = SimpleNamespace(shutdown=lambda: stuck.wait(30), server_close=lambda: closed.append(1))
    listener = threading.Thread(target=stuck.wait, args=(30,), daemon=True)
    listener.start()
    t0 = time.monotonic()
    try:
        mcp_server._stop_listener(httpd, listener)
        took = time.monotonic() - t0
    finally:
        stuck.set()
        listener.join(5)
    assert 0.25 <= took < 2.0 and closed == [1], (took, closed)


# ── the real server process: stdio, HTTP and HTTPS ──────────────────────────────────────────────────

#: Runs the server's `__main__` as `python -m` does, beside a helper thread that, once the test
#: creates the trigger file and the main thread has idled a moment, delivers SIGTERM to ITSELF.
#: argv: trigger file, sent file, then the server's own argv.
_LAUNCH = r'''
import os, runpy, signal, sys, threading, time
trigger, sent = sys.argv[1], sys.argv[2]
sys.argv = ["cascade.apps.mcp_server"] + sys.argv[3:]

def trip():
    while not os.path.exists(trigger):
        time.sleep(0.02)
    time.sleep(0.5)  # the main thread is settled in the worker's idle wait
    signal.pthread_kill(threading.get_ident(), signal.SIGTERM)
    open(sent, "w").close()

threading.Thread(target=trip, daemon=True, name="test-trip").start()
runpy.run_module("cascade.apps.mcp_server", run_name="__main__", alter_sys=True)
'''


def _env(tmp_path: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
    env.update(CASCADE_CAMERA="mock", CASCADE_ARM="mock", CASCADE_STREAM="0", CASCADE_PREWARM="0",
               CASCADE_RUN_DIR=str(tmp_path / "run"), CASCADE_MCP_TOKEN=TOKEN, PYTHONDONTWRITEBYTECODE="1")
    return env


def _readline(proc, timeout_s: float = 60.0) -> str:
    deadline = time.monotonic() + timeout_s
    buf = b""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 0.1)
        if ready:
            chunk = os.read(proc.stdout.fileno(), 1)
            if not chunk:
                break
            buf += chunk
            if chunk == b"\n":
                return buf.decode()
        elif proc.poll() is not None:
            break
    pytest.fail(f"no protocol line from the server (got {buf!r})")


def _tls_files(tmp_path: Path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1", "-keyout", str(key), "-out", str(cert)],
                   check=True, capture_output=True)
    return cert, key


def _http_initialize(port: int, ctx: ssl.SSLContext | None, *, call=None):
    import http.client

    conn = (http.client.HTTPSConnection("127.0.0.1", port, timeout=60, context=ctx) if ctx is not None
            else http.client.HTTPConnection("127.0.0.1", port, timeout=60))
    frames = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}]
    if call is not None:
        frames.append({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": call, "arguments": {}}})
    session = None
    for frame in frames:
        headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "Accept": "application/json"}
        if session:
            headers["Mcp-Session-Id"] = session
        conn.request("POST", "/mcp", body=json.dumps(frame), headers=headers)
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200, (resp.status, body)
        session = session or resp.getheader("Mcp-Session-Id")
    return conn  # left open, as the CI test left its connection


@pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")
@pytest.mark.parametrize("transport", ["stdio", "stdio-built", "http", "https"])
def test_a_sigterm_the_main_thread_never_woke_for_still_stops_and_tears_down(tmp_path, transport):
    """The real server, idle after initialize (and, `-built`, after a tool call built the runtime): a
    SIGTERM whose wakeup the main thread missed. Main: alive until a second signal. Now: e-stop latched,
    teardown receipt written, exit 143, at once."""
    trigger, sent = tmp_path / "trigger", tmp_path / "sent"
    argv = [sys.executable, "-c", _LAUNCH, str(trigger), str(sent)]
    conn = None
    if transport.startswith("stdio"):
        proc = subprocess.Popen(argv, cwd=REPO, env=_env(tmp_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    else:
        port_file = tmp_path / "port"
        argv += ["--http", "127.0.0.1:0", "--http-port-file", str(port_file)]
        ctx = None
        if transport == "https":
            cert, key = _tls_files(tmp_path)
            argv += ["--tls-cert", str(cert), "--tls-key", str(key)]
            ctx = ssl.create_default_context(cafile=str(cert))
            ctx.check_hostname = False
        proc = subprocess.Popen(argv, cwd=REPO, env=_env(tmp_path), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if transport.startswith("stdio"):
            frames = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}]
            if transport == "stdio-built":
                frames.append({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                               "params": {"name": "move_home", "arguments": {}}})
            for frame in frames:
                proc.stdin.write((json.dumps(frame) + "\n").encode())
                proc.stdin.flush()
                reply = _readline(proc, 120)
                while '"id"' not in reply:  # notifications (tools/list_changed) first
                    reply = _readline(proc, 120)
                assert json.loads(reply)["id"] == frame["id"], reply
        else:
            deadline = time.monotonic() + 60
            while not (port_file.exists() and port_file.read_text().strip()):
                assert proc.poll() is None and time.monotonic() < deadline, "the server never listened"
                time.sleep(0.05)
            conn = _http_initialize(int(port_file.read_text()), ctx)
        trigger.write_text("1")
        deadline = time.monotonic() + 30
        while not sent.exists():
            assert proc.poll() is None and time.monotonic() < deadline, "the helper never sent the signal"
            time.sleep(0.02)
        t0 = time.monotonic()
        try:
            rc = proc.wait(timeout=8)
            took = time.monotonic() - t0
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)  # the second signal rescues it on main
            try:
                proc.wait(timeout=20)
            finally:
                pytest.fail("the SIGTERM was lost: the server ran on until a second signal")
    finally:
        if conn is not None:
            conn.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        err = proc.stderr.read().decode(errors="replace")
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    assert rc == 128 + signal.SIGTERM, (rc, err[-2000:])
    assert took < 5.0, took
    assert "EMERGENCY STOP latched" in err, err[-2000:]
    receipt = json.loads((tmp_path / "run" / "teardown.json").read_text())
    assert receipt["ok"] is True and receipt["complete"] is True, receipt
    assert receipt["runtime_built"] is (transport == "stdio-built"), receipt


#: The server's main() with the stdin reader stuck after EOF: `server.input_closed()` (called by the
#: reader once stdin closes) never returns, so it never queues the worker's EOF and never ends.
_STUCK_READER = r'''
import sys, threading
sys.argv = ["cascade.apps.mcp_server"]
import cascade.apps.mcp_server as m
m.McpSkillServer.input_closed = lambda self: threading.Event().wait()
sys.exit(m.main())
'''


@pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")
def test_a_stdin_reader_that_never_ends_does_not_hold_up_the_teardown(tmp_path):
    """Main joined the stdio reader without a bound: a reader stuck in a frame's handling kept the
    SIGTERM teardown from ever writing its receipt. Now it waits TRANSPORT_STOP_S, says so, and goes on."""
    proc = subprocess.Popen([sys.executable, "-c", _STUCK_READER], cwd=REPO,
                            env=_env(tmp_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}) + "\n").encode())
        proc.stdin.flush()
        assert json.loads(_readline(proc, 120))["id"] == 1
        proc.stdin.close()  # EOF: the reader's input_closed() never returns
        time.sleep(0.5)
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("the teardown waited for the stuck stdin reader")
        took = time.monotonic() - t0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)
        err = proc.stderr.read().decode(errors="replace")
        proc.stdout.close()
        proc.stderr.close()
    assert rc == 128 + signal.SIGTERM and took < 15, (rc, took, err[-1500:])
    assert "the stdin reader did not stop" in err, err[-1500:]
    assert json.loads((tmp_path / "run" / "teardown.json").read_text())["ok"] is True
