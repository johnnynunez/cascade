"""MCP Streamable HTTP transport (B35: run the agent in NemoClaw/OpenShell).

NemoClaw only registers authenticated Streamable HTTP MCP endpoints; it never
launches or wraps a stdio server. These tests run the real server process
over real HTTP and hold it to the same contracts as the stdio transport: one
serial worker, the out-of-band stop channel, cancellation that freezes a
motion. On top of that they check the transport's own guards: a bearer token
is mandatory, a non-loopback listener requires TLS, sessions, body size and
content-type checks, and Origin rejection against DNS rebinding.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import REPO, has_pinocchio, loopback_host

pytestmark = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")

TOKEN = "cascade-test-token-" + "k" * 24
ACCEPT_BOTH = "application/json, text/event-stream"


class HttpMcp:
    """One server subprocess in --http mode plus a minimal MCP HTTP client."""

    #: close()'s wait after SIGTERM (None: demo.TEARDOWN_WAIT_S plus a margin)
    sigterm_bound_s: float | None = None

    def __init__(self, tmp_path: Path, *, token: str | None = TOKEN, bind: str = "127.0.0.1:0",
                 extra_args=(), extra_env: dict | None = None, tls: ssl.SSLContext | None = None):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(REPO / "src"), env.get("PYTHONPATH", "")) if path)
        env["CASCADE_CAMERA"] = "mock"
        env["CASCADE_ARM"] = "mock"
        env["CASCADE_RUN_DIR"] = str(tmp_path / "run")
        env["CASCADE_STREAM"] = "0"
        env.pop("CASCADE_MCP_TOKEN", None)
        if token is not None:
            env["CASCADE_MCP_TOKEN"] = token
        env.update(extra_env or {})
        self.port_file = tmp_path / "mcp.port"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cascade.apps.mcp_server", "--http", bind,
             "--http-port-file", str(self.port_file), *extra_args],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True)
        self.tls = tls
        self.session: str | None = None
        self.port: int | None = None
        self._id = 0

    def wait_ready(self, timeout=60.0) -> "HttpMcp":
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.port_file.exists() and self.port_file.read_text().strip():
                self.port = int(self.port_file.read_text().strip())
                return self
            if self.proc.poll() is not None:
                pytest.fail(f"server exited {self.proc.returncode}: {self.proc.stderr.read()[-2000:]}")
            time.sleep(0.05)
        pytest.fail("server never wrote its port file")

    def exit_status(self, timeout=60.0) -> tuple[int, str]:
        try:
            _, err = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            pytest.fail("server kept running; expected a refusal at start-up")
        return self.proc.returncode, err

    def _conn(self, timeout=120.0):
        host = loopback_host()
        if self.tls is not None:
            return http.client.HTTPSConnection(host, self.port, timeout=timeout, context=self.tls)
        return http.client.HTTPConnection(host, self.port, timeout=timeout)

    def raw(self, method: str, body: bytes | None = None, *, headers: dict | None = None,
            auth: str | None = TOKEN, path="/mcp", timeout=120.0):
        h = {"Accept": ACCEPT_BOTH}
        if body is not None:
            h["Content-Type"] = "application/json"
        if auth is not None:
            h["Authorization"] = f"Bearer {auth}"
        if self.session is not None:
            h["Mcp-Session-Id"] = self.session
        h.update(headers or {})
        conn = self._conn(timeout)
        try:
            conn.request(method, path, body=body, headers=h)
            resp = conn.getresponse()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read().decode()
        finally:
            conn.close()

    def rpc(self, method: str, params: dict | None = None, *, accept=ACCEPT_BOTH, timeout=120.0):
        self._id += 1
        frame = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            frame["params"] = params
        status, headers, body = self.raw("POST", json.dumps(frame).encode(),
                                         headers={"Accept": accept}, timeout=timeout)
        assert status == 200, (status, body)
        if method == "initialize":
            self.session = headers.get("mcp-session-id")
        msg = _decode(headers, body)
        assert msg["id"] == self._id
        return msg

    def notify(self, method: str, params: dict | None = None):
        frame: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        return self.raw("POST", json.dumps(frame).encode())

    def start(self) -> "HttpMcp":
        self.wait_ready()
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "pytest", "version": "0"}})
        status, _, _ = self.notify("notifications/initialized")
        assert status == 202
        return self

    def close(self):
        """SIGTERM, as systemd stops it. A server still running past its teardown's bound
        (demo.TEARDOWN_WAIT_S) plus a margin ignored the signal: that fails the test loudly. A
        silent kill here once hid a lost SIGTERM (#435 review: the server never latched, never
        tore down)."""
        if self.proc.poll() is None:
            from cascade.apps.demo import TEARDOWN_WAIT_S

            bound = TEARDOWN_WAIT_S + 10.0 if self.sigterm_bound_s is None else self.sigterm_bound_s
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=bound)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
                pytest.fail(f"the server ignored SIGTERM for {bound:g} s (its teardown waits at most "
                            f"{TEARDOWN_WAIT_S:g} s); killed it")


def test_close_fails_loudly_when_the_server_ignores_sigterm():
    """A server still up past its teardown's bound ignored SIGTERM: close() kills it and fails the
    test. Main killed it silently, which hid a lost SIGTERM."""
    s = HttpMcp.__new__(HttpMcp)
    s.sigterm_bound_s = 0.5
    s.proc = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, "
                               "signal.SIG_IGN); print('ready', flush=True); time.sleep(120)"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert s.proc.stdout.readline().strip() == "ready"
        with pytest.raises(pytest.fail.Exception, match="ignored SIGTERM"):
            s.close()
        assert s.proc.poll() is not None, "left running"
    finally:
        if s.proc.poll() is None:
            s.proc.kill()
            s.proc.wait(10)


def _decode(headers: dict, body: str) -> dict:
    ctype = headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return json.loads(body)
    assert ctype.startswith("text/event-stream"), ctype
    data = [line[len("data:"):].strip() for line in body.splitlines() if line.startswith("data:")]
    assert len(data) == 1, body
    return json.loads(data[0])


def _payload(msg):
    text = next(b["text"] for b in msg["result"]["content"] if b["type"] == "text")
    return json.loads(text), msg["result"].get("isError", False)


@pytest.fixture
def server(tmp_path):
    s = HttpMcp(tmp_path).start()
    yield s
    s.close()


# ── start-up refusals ──────────────────────────────────────────────────────


def test_http_mode_refuses_to_start_without_a_token(tmp_path):
    s = HttpMcp(tmp_path, token=None)
    code, err = s.exit_status()
    assert code != 0 and "CASCADE_MCP_TOKEN" in err
    assert not s.port_file.exists()


def test_a_short_token_is_refused(tmp_path):
    code, err = HttpMcp(tmp_path, token="short").exit_status()
    assert code != 0 and "CASCADE_MCP_TOKEN" in err


def test_a_non_loopback_listener_requires_tls(tmp_path):
    code, err = HttpMcp(tmp_path, bind="0.0.0.0:0").exit_status()
    assert code != 0 and "TLS" in err


# ── authentication and HTTP hygiene ────────────────────────────────────────


def test_requests_without_the_bearer_token_are_refused(server):
    ping = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}).encode()
    status, headers, _ = server.raw("POST", ping, auth=None)
    assert status == 401 and headers.get("www-authenticate", "").startswith("Bearer")
    status, _, _ = server.raw("POST", ping, auth=TOKEN + "x")
    assert status == 401
    status, headers, body = server.raw("POST", ping)
    assert status == 200 and _decode(headers, body) == {"jsonrpc": "2.0", "id": 99, "result": {}}


def test_unauthenticated_requests_never_reach_the_robot(server):
    frame = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "emergency_stop", "arguments": {}}}
    status, _, _ = server.raw("POST", json.dumps(frame).encode(), auth=None)
    assert status == 401
    # the e-stop was NOT latched by the refused frame: motion still allowed
    payload, is_err = _payload(server.rpc("tools/call", {"name": "move_home", "arguments": {}}))
    assert not is_err, payload


@pytest.mark.parametrize("headers, body, status", [
    ({"Origin": "http://evil.example"}, None, 403),
    ({"Content-Type": "text/plain"}, None, 415),
    ({"MCP-Protocol-Version": "1999-01-01"}, None, 400),
    ({}, b"[" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode() + b"]", 400),
    ({}, b"{not json", 400),
    ({}, "OVERSIZE", 413),
], ids=["origin", "content-type", "protocol-version", "batch", "not-json", "oversize"])
def test_http_guards(server, headers, body, status):
    if body == "OVERSIZE":  # built here: a 140 kB literal in the id overflows the env
        body = b"{" + b" " * 140_000 + b"}"
    body = body if body is not None else json.dumps({"jsonrpc": "2.0", "id": 5, "method": "ping"}).encode()
    got, _, _ = server.raw("POST", body, headers=headers)
    assert got == status


def test_wrong_path_is_404(server):
    ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
    assert server.raw("POST", ping, path="/other")[0] == 404


# ── sessions ───────────────────────────────────────────────────────────────


def test_initialize_issues_a_session_and_delete_ends_it(server):
    assert server.session and len(server.session) >= 16
    assert server.raw("GET", None)[0] == 405
    assert server.raw("DELETE", None)[0] == 200
    ping = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}).encode()
    assert server.raw("POST", ping)[0] == 404  # the client must re-initialize


def test_an_unknown_session_is_404(server):
    server.session = "not-a-session"
    ping = json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"}).encode()
    assert server.raw("POST", ping)[0] == 404


# ── the same tool surface and results as stdio ─────────────────────────────


def test_tools_list_is_the_stdio_catalog(server):
    names = {t["name"] for t in server.rpc("tools/list")["result"]["tools"]}
    from cascade.apps.mcp_server import McpSkillServer
    assert names == {t["name"] for t in McpSkillServer().list_tools()}
    assert {"emergency_stop", "pick_and_place", "get_observation"} <= names


@pytest.mark.parametrize("accept", ["application/json", ACCEPT_BOTH])
def test_tool_calls_answer_as_json_or_sse(server, accept):
    msg = server.rpc("tools/call", {"name": "get_observation", "arguments": {}}, accept=accept)
    payload, is_err = _payload(msg)
    assert not is_err and payload["ok"] and "observation_frame" in payload


def test_a_long_call_keeps_its_sse_stream_alive(tmp_path):
    s = HttpMcp(tmp_path, extra_env={"CASCADE_MCP_SSE_KEEPALIVE_S": "0.1"}).start()
    try:
        s.rpc("tools/call", {"name": "get_observation", "arguments": {}})
        s._id += 1
        frame = {"jsonrpc": "2.0", "id": s._id, "method": "tools/call",
                 "params": {"name": "pick_and_place", "arguments": {"object": "red object"}}}
        status, headers, body = s.raw("POST", json.dumps(frame).encode())
        assert status == 200 and headers["content-type"].startswith("text/event-stream")
        lines = body.splitlines()
        first_data = next(i for i, line in enumerate(lines) if line.startswith("data:"))
        assert any(line.startswith(":") for line in lines[:first_data]), body[:400]
        assert _decode(headers, body)["id"] == s._id
    finally:
        s.close()


# ── the stop channel survives the transport change ─────────────────────────


def test_emergency_stop_preempts_a_running_motion_over_http(server):
    server.rpc("tools/call", {"name": "get_observation", "arguments": {}})
    out: dict = {}

    def pick():
        out["msg"] = server.rpc("tools/call", {"name": "pick_and_place",
                                               "arguments": {"object": "red object"}})
        out["t"] = time.monotonic()

    worker = threading.Thread(target=pick)
    worker.start()
    time.sleep(0.5)
    stop_status, stop_headers, stop_body = server.raw("POST", json.dumps(
        {"jsonrpc": "2.0", "id": 9001, "method": "tools/call",
         "params": {"name": "emergency_stop", "arguments": {}}}).encode())
    t_stop = time.monotonic()
    assert stop_status == 200
    stop = _decode(stop_headers, stop_body)
    assert stop["id"] == 9001 and _payload(stop)[0]["stopped"]
    worker.join(120)
    assert "msg" in out and out["t"] >= t_stop, "the stop answered after the motion finished"
    payload, is_err = _payload(out["msg"])
    assert is_err and "not retrying" in payload["error"], payload
    payload, is_err = _payload(server.rpc("tools/call", {"name": "move_home", "arguments": {}}))
    assert is_err and "e-stop" in payload["error"]


def test_cancelling_a_motion_freezes_the_arm_over_http(server):
    server.rpc("tools/call", {"name": "get_observation", "arguments": {}})
    out: dict = {}
    pick_id = server._id + 1

    def pick():
        out["msg"] = server.rpc("tools/call", {"name": "pick_and_place",
                                               "arguments": {"object": "red object"}})

    worker = threading.Thread(target=pick)
    worker.start()
    time.sleep(0.5)
    assert server.notify("notifications/cancelled", {"requestId": pick_id})[0] == 202
    worker.join(120)
    assert _payload(out["msg"])[1]
    payload, is_err = _payload(server.rpc("tools/call", {"name": "move_home", "arguments": {}}))
    assert is_err and "e-stop" in payload["error"]


def test_a_cancel_from_another_session_does_not_touch_this_motion(server, tmp_path):
    other = HttpMcp.__new__(HttpMcp)
    other.__dict__.update(server.__dict__)
    other.session = None
    other._id = 0
    other.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    assert other.session != server.session
    server.rpc("tools/call", {"name": "get_observation", "arguments": {}})
    out: dict = {}
    pick_id = server._id + 1

    def pick():
        out["msg"] = server.rpc("tools/call", {"name": "pick_and_place",
                                               "arguments": {"object": "red object"}})

    worker = threading.Thread(target=pick)
    worker.start()
    time.sleep(0.5)
    # same JSON-RPC id, different session: ids are unique per session only
    assert other.notify("notifications/cancelled", {"requestId": pick_id})[0] == 202
    worker.join(120)
    payload, is_err = _payload(server.rpc("tools/call", {"name": "move_home", "arguments": {}}))
    assert not is_err, payload


def test_a_connection_timeout_cancel_of_initialize_does_not_touch_a_running_pick(tmp_path):
    """OpenClaw's MCP client aborts its connect signal at connectionTimeoutMs
    (120 s, deploy/openshell/openclaw.json) and sends notifications/cancelled
    for the INITIALIZE request id of the same session, while a ~140 s pick of
    that session is still running (6 of 9 such cancels landed mid-pick in the
    16 Oct kitchen run). Only a cancel naming the running call may stop it.
    The MCP TS SDK numbers a session's requests from 0, so OpenClaw's
    initialize id is 0, as here."""
    server = HttpMcp(tmp_path)
    server._id = -1  # HttpMcp.start(): the session's first request (initialize) carries id 0
    server.start()
    try:
        initialize_id = 0
        server.rpc("tools/call", {"name": "get_observation", "arguments": {}})
        out: dict = {}

        def pick():
            out["msg"] = server.rpc("tools/call", {"name": "pick_and_place",
                                                   "arguments": {"object": "red object"}})

        worker = threading.Thread(target=pick)
        worker.start()
        time.sleep(0.5)
        assert server.notify("notifications/cancelled",
                             {"requestId": initialize_id, "reason": "AbortError: timeout"})[0] == 202
        assert worker.is_alive(), "the pick ended before the cancel arrived; the check proves nothing"
        worker.join(120)
        payload, _ = _payload(out["msg"])
        assert "e-stop" not in json.dumps(payload) and not payload.get("cancelled"), payload
        payload, is_err = _payload(server.rpc("tools/call", {"name": "move_home", "arguments": {}}))
        assert not is_err, payload
    finally:
        server.close()


# ── TLS for a NemoClaw trusted-private endpoint ────────────────────────────


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")
def test_tls_listener_round_trip(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    ctx = ssl.create_default_context(cafile=str(cert))
    s = HttpMcp(tmp_path, extra_args=("--tls-cert", str(cert), "--tls-key", str(key)), tls=ctx)
    try:
        s.start()
        assert s.rpc("ping")["result"] == {}
        plain = http.client.HTTPConnection(loopback_host(), s.port, timeout=10)
        with pytest.raises((http.client.HTTPException, ConnectionError, OSError)):
            plain.request("POST", "/mcp", body=b"{}", headers={"Authorization": f"Bearer {TOKEN}"})
            plain.getresponse().read()
    finally:
        s.close()


def _tls_files(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    return cert, key


def _tls_server(tmp_path) -> HttpMcp:
    cert, key = _tls_files(tmp_path)
    return HttpMcp(tmp_path, extra_args=("--tls-cert", str(cert), "--tls-key", str(key)),
                   tls=ssl.create_default_context(cafile=str(cert)))


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")
def test_a_silent_tls_client_blocks_neither_new_clients_nor_the_stop_channel(tmp_path):
    """#435 review: the listener did the TLS handshake inside accept(), with no timeout. One TCP
    client that never sends its ClientHello stalled it: no new client was served, the stop channel
    included, until that socket closed. Now each handshake runs on its connection's own thread."""
    s = _tls_server(tmp_path)
    silent = None
    try:
        s.start()
        silent = socket.create_connection((loopback_host(), s.port))  # TCP only: no ClientHello, ever
        time.sleep(0.3)
        stop = s.rpc("tools/call", {"name": "emergency_stop", "arguments": {}}, timeout=5)
        assert _payload(stop)[0]["stopped"]
        assert s.rpc("ping", timeout=5)["result"] == {}
        assert s.rpc("tools/list", timeout=5)["result"]["tools"]
    finally:
        if silent is not None:
            silent.close()
        s.close()


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")
def test_sigterm_with_a_silent_tls_client_connected_still_tears_down(tmp_path):
    """#435 review: SIGTERM's listener shutdown waited for a stalled handshake: the server exited
    13 s after the signal, and only because the silent client went away first."""
    s = _tls_server(tmp_path)
    silent = None
    try:
        s.start()
        silent = socket.create_connection((loopback_host(), s.port))
        time.sleep(0.3)
        t0 = time.monotonic()
        s.proc.send_signal(signal.SIGTERM)
        try:
            rc = s.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("SIGTERM did not end the server while a silent TLS client stayed connected")
        took = time.monotonic() - t0
    finally:
        if silent is not None:
            silent.close()
        s.close()
    assert rc == 128 + signal.SIGTERM and took < 8, (rc, took)
    receipt = json.loads((tmp_path / "run" / "teardown.json").read_text())
    assert receipt["ok"] is True and receipt["complete"] is True, receipt


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")
def test_a_tls_handshake_that_never_completes_is_dropped_after_its_bound(tmp_path):
    """The handshake's own timeout: a silent client holds its connection thread at most that long,
    then the server closes the socket. A real client is served meanwhile and afterwards."""
    from http.server import BaseHTTPRequestHandler

    from cascade.apps import mcp_server

    cert, key = _tls_files(tmp_path)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))

    class Ok(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_a):
            pass

    httpd = mcp_server._tls_http_server(("127.0.0.1", 0), Ok, ctx, handshake_s=0.5)
    port = httpd.server_address[1]
    listener = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    listener.start()
    client = ssl.create_default_context(cafile=str(cert))

    def get(pause_s=0.0):
        conn = http.client.HTTPSConnection("127.0.0.1", port, timeout=5, context=client)
        try:
            conn.connect()  # the handshake
            time.sleep(pause_s)  # then a request slower than the handshake's bound: no timeout applies
            conn.request("GET", "/")
            return conn.getresponse().status
        finally:
            conn.close()

    try:
        silent = socket.create_connection(("127.0.0.1", port))
        silent.settimeout(10)
        t0 = time.monotonic()
        assert get() == 200, "served while the silent client holds its handshake"
        try:
            assert silent.recv(1) == b"", "the server closes the silent connection"
        finally:
            silent.close()
        assert 0.4 <= time.monotonic() - t0 < 5
        assert get() == 200
        assert get(pause_s=1.0) == 200, "the handshake's bound must not apply to the request"
    finally:
        httpd.shutdown()
        httpd.server_close()
        listener.join(5)


def test_the_server_logs_its_url_and_writes_the_run_log(server, tmp_path):
    server.rpc("ping")
    log = (tmp_path / "run" / "server.log").read_text()
    assert f"cascade MCP server on http://127.0.0.1:{server.port}/mcp" in log
