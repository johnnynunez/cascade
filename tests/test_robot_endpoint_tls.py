"""B71: opt-in TLS for `cascade.robot-runtime/1`, the split conversation deployment's robot protocol.

B51 (tests/test_conversation_split_service.py) runs the robot runtime in its own
`cascade-robot-service` process behind a loopback-only HTTP/JSON protocol. B71
adds an opt-in TLS listener (server certificate + key) and a client that trusts
ONLY a pinned CA (with the hostname checked) and/or the server certificate's
SHA-256 fingerprint. The bearer token, the lease and the stop semantics are
unchanged; a non-loopback bind without TLS is refused at startup.

Plaintext loopback stays the default. Golden tests pin its server bytes, the
client's request bytes, the robot service's ready.json and banner (captured on
main 38f6d08) and pass there by design, as do the premise tests.

Certificates come from scripts/robot_endpoint_certs.py (the B35 openssl recipe),
so TLS tests skip where the openssl CLI is missing, like B35's. Every server binds
port 0; values in this item's block (47300-47399) are only ever refused, never
bound or dialled. No GPU, physics, speech inference or wall-clock assertion.
"""
import asyncio
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime
from test_conversation_split_service import (
    _Body, _Process, _Provider, _Trace, _read_prefixed, _spoken_reply, _tool_output, _tool_turn, _until, _ws_next)

REPO = Path(__file__).resolve().parents[1]
PORT_BLOCK = range(47300, 47400)
PROTOCOL = "cascade.robot-runtime/1"
TOKEN = "k" * 43
STALE_TOKEN = "s" * 43
TOKEN_ENV = "TEST_B71_ROBOT_ENDPOINT_TOKEN"
CERTS = REPO / "scripts" / "robot_endpoint_certs.py"
BENCH = REPO / "scripts" / "bench_robot_endpoint.py"
EVIDENCE = REPO / "docs" / "evidence" / "b71-robot-runtime-tls-20261010" / "latency.json"
needs_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI not available")

# Golden digests of the DEFAULT (plaintext loopback) wire, captured on main 38f6d08.
GOLDEN_SERVER_WIRE_SHA256 = "7f29ea2a5f275a13a42ff72dd3a2cca1ae489aaf8825c541a34f3d88ec39e4f8"
GOLDEN_CLIENT_WIRE_SHA256 = "0a967127dc2036cf84262c5bcfb9dc3a25e871c81105dc9e3e26fef1d0e8fde0"
GOLDEN_ROBOT_READY_KEYS = ["schema", "protocol", "state", "pid", "published_monotonic_s", "origin", "robot_id",
                           "tools", "catalog_sha256", "lease_ttl_s", "runtime_stopped", "physical_admission",
                           "robot_config_sha256"]
BEARER_LINE = "Bearer token from the configured environment variable; no physical result is claimed."


# --------------------------------------------------------------------------- helpers


def _issue(out, *sans, name="server", force=False):
    """Run the operator script: private CA (reused per directory) + one server certificate."""
    argv = [sys.executable, str(CERTS), "--out", str(out), "--name", name]
    for san in sans:
        argv += ["--san", san]
    if force:
        argv.append("--force")
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    return SimpleNamespace(ca=Path(info["ca"]), cert=Path(info["cert"]), key=Path(info["key"]),
                           sha256=info["certificate_sha256"], info=info)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    root = tmp_path_factory.mktemp("b71-pki")
    good = _issue(root / "a", "IP:127.0.0.1", "DNS:localhost")
    misnamed = _issue(root / "a", "DNS:robot.invalid", name="misnamed")     # same CA, another name
    foreign = _issue(root / "b", "IP:127.0.0.1", "DNS:localhost")           # another CA, right name
    return SimpleNamespace(root=root, good=good, misnamed=misnamed, foreign=foreign)


@contextmanager
def _served(*, tls=None, ttl=5.0, host="127.0.0.1"):
    from cascade.robotics.endpoint import RobotRuntimeEndpoint
    body, trace = _Body(), _Trace()
    runtime = RobotRuntime({"body": body}, trace=trace)
    endpoint = RobotRuntimeEndpoint(runtime, robot_id="fixture", token=TOKEN, lease_ttl_s=ttl)
    kwargs = {} if tls is None else {"tls_cert": str(tls.cert), "tls_key": str(tls.key)}
    origin = endpoint.start(host=host, port=0, **kwargs)
    try:
        yield SimpleNamespace(body=body, trace=trace, runtime=runtime, endpoint=endpoint, origin=origin)
    finally:
        if body.block is not None:
            body.block.set()
        endpoint.close()
        runtime.close()


def _call(origin, method, path, body=None, *, token=TOKEN, protocol=PROTOCOL, ca=None, timeout=20):
    url = urlsplit(origin)
    if url.scheme == "https":
        connection = http.client.HTTPSConnection(url.hostname, url.port, timeout=timeout,
                                                 context=ssl.create_default_context(cafile=str(ca)))
    else:
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    headers = {}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if protocol is not None:
        headers["X-Cascade-Protocol"] = protocol
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    return response.status, json.loads(raw)


ROUTES = (("GET", "/v1/describe", None), ("GET", "/v1/state", None),
          ("POST", "/v1/lease", {"lease_id": None}), ("DELETE", "/v1/lease", {"lease_id": "x"}),
          ("POST", "/v1/stop", {}), ("POST", "/v1/reset", {"lease_id": "x", "expected_generation": 0}),
          ("POST", "/v1/execute", {"lease_id": "x", "tool": "body.read", "arguments": {},
                                   "expected_generation": 0, "deadline_remaining_s": 5}),
          ("POST", "/v1/record", {"lease_id": "x", "tool": "emergency_stop", "arguments": {},
                                  "result": {"ok": True}, "duration_ms": 1}))


def _private_ipv4():
    """A private, non-loopback IPv4 of this host that can be bound, else None (no packet is sent)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        candidate = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    address = ipaddress.ip_address(candidate)
    if not address.is_private or address.is_loopback or address.is_link_local:
        return None
    return candidate


class _RecordingTLS:
    """One-shot TLS listener that records every application byte a client sends after the handshake."""

    def __init__(self, identity):
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(str(identity.cert), str(identity.key))
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.handshakes, self.received, self.done = 0, [], threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        self.listener.settimeout(60)
        data = b""
        try:
            raw, _ = self.listener.accept()
            raw.settimeout(60)
            with self.context.wrap_socket(raw, server_side=True) as tls:
                self.handshakes += 1
                while chunk := tls.recv(65536):
                    data += chunk
        except OSError:
            pass
        finally:
            self.received.append(data)
            self.done.set()

    def close(self):
        self.listener.close()
        self.thread.join(60)


def _env(token=None):
    env = {key: value for key, value in os.environ.items()
           if not (key.startswith("CASCADE_") and key.endswith("_PORT"))}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1")
    env.pop(TOKEN_ENV, None)
    if token is not None:
        env[TOKEN_ENV] = token
    return env


async def _robot(directory, token, *, tls=None, lease_ttl_s=5.0):
    run = directory / "robot"
    argv = [sys.executable, "-m", "cascade.apps.robot_service", "--robot", "microduck_conversation_mock",
            "--port", "0", "--run-dir", str(run), "--token-env", TOKEN_ENV, "--lease-ttl-s", str(lease_ttl_s)]
    if tls is not None:
        argv += ["--tls-cert", str(tls.cert), "--tls-key", str(tls.key)]
    process = await asyncio.create_subprocess_exec(*argv, cwd=REPO, env=_env(token),
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    line = await _read_prefixed(process, b"Robot runtime endpoint ")
    if line is None:
        await process.wait()
        raise AssertionError((await process.stderr.read()).decode())
    service = _Process(process, run, line.split()[3])
    service.banner = [line, (await asyncio.wait_for(process.stdout.readline(), 60)).decode().strip()]
    if tls is not None:
        service.banner.append((await asyncio.wait_for(process.stdout.readline(), 60)).decode().strip())
    ready = service.ready_record()
    assert (ready["origin"], ready["pid"]) == (service.origin, process.pid)   # this child's own port
    ca = None if tls is None else tls.ca
    service.state = lambda: _call(service.origin, "GET", "/v1/state", token=token, ca=ca)[1]
    trace = run / "trace.jsonl"
    service.trace_rows = lambda: ([json.loads(row) for row in trace.read_text().splitlines() if row.strip()]
                                  if trace.exists() else [])
    return service


async def _conversation(directory, values, token, *, expect_ready=True):
    directory.mkdir(parents=True, exist_ok=True)
    run, path = directory / "conversation", directory / "service.json"
    path.write_text(json.dumps({"version": 1, "provider_url": f"ws://127.0.0.1:{PORT_BLOCK[-1]}/v1/realtime",
                                "port": 0, "run_dir": run.name, "intent_timeout_s": 60,
                                "execution_timeout_s": 30, **values}))
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "cascade.apps.conversation", "--config", str(path), cwd=directory,
        env=_env(token), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    line = await _read_prefixed(process, b"Open ")
    if line is None:
        await asyncio.wait_for(process.wait(), 90)
        service = _Process(process, run, None)
        service.returncode, service.stderr = process.returncode, (await process.stderr.read()).decode()
        if expect_ready:
            raise AssertionError(service.stderr)
        return service
    url, credential = line[5:].split("/#")
    service = _Process(process, run, url)
    if not expect_ready:
        await service.terminate()
        raise AssertionError("a conversation that should have been refused started")
    service.url, service.token = url, credential
    ready = service.ready_record()
    assert (ready["origin"], ready["pid"]) == (url, process.pid)
    return service


def _values(robot, provider, **extra):
    return {"robot": "microduck-mock", "provider_url": provider.url, "robot_endpoint": robot.origin,
            "robot_token_env": TOKEN_ENV, "allow_motion": True,
            "allow_tools": ["locomotion.get_base_state", "locomotion.walk_velocity"], **extra}


# --------------------------------------------------------------------- golden / premise


class _StubRuntime:
    """Deterministic runtime: the server golden pins the endpoint's own bytes, not RobotRuntime's."""

    def __init__(self):
        self.resources = (ResourceDescriptor("body/base", "base", "fixture", synthetic=True,
                                             admission="software_only", controller_id="fixture/controller",
                                             writer_id="body"),)
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        self.tool_descriptors = {
            "body.read": ToolDescriptor("body.read", "Synthetic observation", empty, "body", "read"),
            "emergency_stop": ToolDescriptor("emergency_stop", "Latch the stop", empty, "safety", "stop",
                                             effect="stop")}
        self.trace, self.cancellation_token, self.stopped, self.calls = object(), 7, False, []

    def execute(self, name, args, *, expected_generation, deadline_monotonic_s):
        self.calls.append(("execute", name, args, expected_generation))
        return {"ok": expected_generation == self.cancellation_token, "tool": name, "generation": expected_generation}

    def stop(self):
        self.stopped, self.cancellation_token = True, self.cancellation_token + 1
        return {"ok": True, "latched": True, "generation": self.cancellation_token, "physical_stop_verified": False}

    def reset_stop(self, *, expected_generation):
        ok = expected_generation == self.cancellation_token
        if ok:
            self.stopped = False
        return {"ok": ok, "generation": self.cancellation_token}

    def _record_tool_result(self, name, args, result, *, started_monotonic_s):
        self.calls.append(("record", name, args, result))


def _raw_exchange(port, request):
    with socket.create_connection(("127.0.0.1", port), timeout=20) as connection:
        connection.sendall(request)
        reply = b""
        while chunk := connection.recv(65536):
            reply += chunk
    return reply


def _raw_request(method, path, body=None, *, token=TOKEN, protocol=PROTOCOL, length=None):
    data = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    head = f"{method} {path} HTTP/1.1\r\nHost: robot\r\n"
    if token is not None:
        head += f"Authorization: Bearer {token}\r\n"
    if protocol is not None:
        head += f"X-Cascade-Protocol: {protocol}\r\n"
    head += f"Content-Length: {len(data) if length is None else length}\r\n\r\n"
    return head.encode() + data


def test_default_plaintext_server_wire_is_byte_identical():
    """Golden: every default route's response bytes (Date and the random lease id normalized)."""
    from cascade.robotics.endpoint import RobotRuntimeEndpoint
    runtime = _StubRuntime()
    endpoint = RobotRuntimeEndpoint(runtime, robot_id="fixture", token=TOKEN, lease_ttl_s=30.0)
    origin = endpoint.start(port=0)
    port = urlsplit(origin).port
    assert origin == f"http://127.0.0.1:{port}"
    transcript, lease = [], None

    def exchange(request):
        nonlocal lease
        reply = _raw_exchange(port, request)
        found = re.search(rb'"lease_id": "([0-9a-f]{32})"', reply)
        if found and lease is None:
            lease = found.group(1).decode()
        normalized = re.sub(rb"Date: [^\r]*", b"Date: <date>", reply)
        if lease is not None:
            normalized = normalized.replace(lease.encode(), b"<lease>")
            request = request.replace(lease.encode(), b"<lease>")
        transcript.append(request + b"\n=>\n" + normalized)
        return reply

    try:
        exchange(_raw_request("GET", "/v1/health", token=None, protocol=None))
        exchange(_raw_request("GET", "/v1/describe"))
        exchange(_raw_request("GET", "/v1/state"))
        exchange(_raw_request("GET", "/v1/describe", token=None))
        exchange(_raw_request("GET", "/v1/describe", token=STALE_TOKEN))
        exchange(_raw_request("GET", "/v1/describe", protocol="cascade.robot-runtime/2"))
        exchange(_raw_request("GET", "/v1/unknown"))
        exchange(_raw_request("POST", "/v1/lease", {"lease_id": None}))
        exchange(_raw_request("POST", "/v1/lease", {"lease_id": None}))
        exchange(_raw_request("POST", "/v1/lease", {"lease_id": lease}))
        exchange(_raw_request("POST", "/v1/execute", {"lease_id": lease, "tool": "body.read", "arguments": {},
                                                      "expected_generation": 7, "deadline_remaining_s": 5}))
        exchange(_raw_request("POST", "/v1/execute", {"lease_id": lease, "tool": "body.read", "arguments": {},
                                                      "expected_generation": 7}))
        exchange(_raw_request("POST", "/v1/execute", {"lease_id": "0" * 32, "tool": "body.read", "arguments": {},
                                                      "expected_generation": 7, "deadline_remaining_s": 5}))
        exchange(_raw_request("POST", "/v1/stop", {"lease_id": lease}))
        exchange(_raw_request("POST", "/v1/stop", {}))
        exchange(_raw_request("POST", "/v1/reset", {"lease_id": lease, "expected_generation": 7}))
        exchange(_raw_request("POST", "/v1/reset", {"lease_id": lease, "expected_generation": 8}))
        exchange(_raw_request("POST", "/v1/record", {"lease_id": lease, "tool": "emergency_stop", "arguments": {},
                                                     "result": {"ok": True}, "duration_ms": 2.5}))
        exchange(_raw_request("POST", "/v1/record", {"lease_id": lease, "tool": "body.read", "arguments": {},
                                                     "result": {"ok": True}, "duration_ms": 2.5}))
        exchange(_raw_request("POST", "/v1/stop", b"{"))
        exchange(_raw_request("POST", "/v1/stop", length=2_000_000))
        exchange(_raw_request("DELETE", "/v1/lease", {"lease_id": lease}))
        exchange(_raw_request("GET", "/v1/state"))
    finally:
        endpoint.close()
    joined = b"\n----\n".join(transcript)
    assert b"<lease>" in joined and len(transcript) == 23
    assert hashlib.sha256(joined).hexdigest() == GOLDEN_SERVER_WIRE_SHA256, joined.decode()


class _CannedEndpoint:
    """Raw socket peer that records the client's exact request bytes and answers canned JSON."""
    ANSWERS = {("GET", "/v1/describe"): {"robot_id": "fixture", "resources": [], "tools": [],
                                         "trace_recorded": True, "lease_ttl_s": 3000.0},
               ("POST", "/v1/lease"): {"lease_id": "a" * 32, "ttl_s": 3000.0},
               ("GET", "/v1/state"): {"generation": 3, "stopped": False, "lease_held": True,
                                      "lease_expiry_stops": 0, "executing": 0},
               ("POST", "/v1/execute"): {"result": {"ok": True}},
               ("POST", "/v1/stop"): {"result": {"ok": True, "latched": True}},
               ("POST", "/v1/reset"): {"result": {"ok": True}},
               ("POST", "/v1/record"): {"recorded": True},
               ("DELETE", "/v1/lease"): {"released": True}}

    def __init__(self):
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.requests = []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            with connection:
                connection.settimeout(20)
                data = b""
                while b"\r\n\r\n" not in data:
                    data += connection.recv(65536)
                head, _, body = data.partition(b"\r\n\r\n")
                length = int(re.search(rb"Content-Length: (\d+)", head).group(1)) if b"Content-Length" in head else 0
                while len(body) < length:
                    body += connection.recv(65536)
                self.requests.append((head, body))
                method, path = head.split(b" ")[:2]
                payload = json.dumps({"protocol": PROTOCOL, **self.ANSWERS[(method.decode(), path.decode())]})
                connection.sendall(f"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                   f"{len(payload)}\r\n\r\n{payload}".encode())

    def close(self):
        self.listener.close()
        self.thread.join(20)


def test_default_client_requests_are_byte_identical():
    """Golden: the plaintext client's request bytes for every runtime call (port and timing normalized)."""
    from cascade.robotics.endpoint import RemoteRobotRuntime
    peer = _CannedEndpoint()
    try:
        remote = RemoteRobotRuntime(f"http://127.0.0.1:{peer.port}", TOKEN).connect()
        assert remote.cancellation_token == 3 and remote.stopped is False
        assert remote.ready() == {"ready": True, "reason": None}
        assert remote.execute("body.read", {"x": 1}, expected_generation=3,
                              deadline_monotonic_s=time.monotonic() + 5) == {"ok": True}
        assert remote.stop() == {"ok": True, "latched": True}
        assert remote.reset_stop(expected_generation=3) == {"ok": True}
        remote._record_tool_result("emergency_stop", {}, {"ok": True}, started_monotonic_s=time.monotonic())
        assert remote.endpoint_receipt() == {"origin": f"http://127.0.0.1:{peer.port}", "protocol": PROTOCOL,
                                             "catalog_sha256": remote.catalog_sha256, "lease_ttl_s": 3000.0}
        assert remote.close()["lease_released"] is True
    finally:
        peer.close()
    transcript = []
    for head, body in peer.requests:
        length = int(re.search(rb"Content-Length: (\d+)", head).group(1)) if b"Content-Length" in head else 0
        assert length == len(body)
        head = re.sub(rb"Host: 127\.0\.0\.1:\d+", b"Host: 127.0.0.1:<port>", head)
        head = re.sub(rb"Content-Length: \d+", b"Content-Length: <len>", head)
        body = re.sub(rb'"deadline_remaining_s": [-+0-9.e]+', b'"deadline_remaining_s": <s>', body)
        body = re.sub(rb'"duration_ms": [-+0-9.e]+', b'"duration_ms": <ms>', body)
        transcript.append(head + b"\r\n\r\n" + body)
    joined = b"\n----\n".join(transcript)
    assert len(transcript) == 10 and TOKEN.encode() in joined
    assert hashlib.sha256(joined).hexdigest() == GOLDEN_CLIENT_WIRE_SHA256, joined.decode()


def test_default_robot_service_ready_record_and_banner_are_unchanged(tmp_path):
    """Golden through the real CLI: same ready.json keys in the same order, same two stdout lines."""
    token = secrets.token_urlsafe(32)

    async def scenario():
        robot = await _robot(tmp_path, token)
        try:
            ready = robot.ready_record()
            assert list(ready) == GOLDEN_ROBOT_READY_KEYS
            port = urlsplit(robot.origin).port
            assert robot.origin == f"http://127.0.0.1:{port}" and ready["origin"] == robot.origin
            assert (ready["schema"], ready["protocol"], ready["state"]) == (1, PROTOCOL, "listening_at_publication")
            assert (ready["lease_ttl_s"], ready["runtime_stopped"], ready["physical_admission"]) == (5.0, False, False)
            assert robot.banner == [f"Robot runtime endpoint {robot.origin} ({PROTOCOL})", BEARER_LINE]
            assert _call(robot.origin, "GET", "/v1/health", token=None, protocol=None) == (
                200, {"protocol": PROTOCOL, "alive": True})
        finally:
            await robot.terminate()
        assert await robot.process.stdout.read() == b""        # nothing else is printed by default
        assert robot.returncode == 0 and robot.closure()["ok"] is True
    asyncio.run(scenario())


def test_premise_plaintext_endpoint_is_loopback_only_and_speaks_no_tls(pki):
    """Premise (true on main): no off-loopback bind, and a TLS client gets no answer from plaintext."""
    from cascade.robotics.endpoint import RobotRuntimeEndpoint, endpoint_origin
    runtime = RobotRuntime({"body": _Body()})
    try:
        endpoint = RobotRuntimeEndpoint(runtime, robot_id="fixture", token=TOKEN, lease_ttl_s=30)
        for host in ("0.0.0.0", "192.0.2.1"):
            with pytest.raises(ValueError):
                endpoint.start(host=host, port=0)
        endpoint.close()
    finally:
        runtime.close()
    with pytest.raises(ValueError):
        endpoint_origin(f"https://127.0.0.1:{PORT_BLOCK[0]}")
    with pytest.raises(ValueError):
        endpoint_origin(f"http://10.0.0.5:{PORT_BLOCK[0]}")
    with _served() as rig:
        port = urlsplit(rig.origin).port
        with pytest.raises((ssl.SSLError, ConnectionError)):     # no TLS spoken (or reset on the garbage)
            _call(f"https://127.0.0.1:{port}", "GET", "/v1/health", ca=pki.good.ca)
        assert rig.endpoint._state()["lease_held"] is False


# ------------------------------------------------------------------- TLS endpoint (in-process)


def test_tls_endpoint_serves_a_ca_pinned_client_end_to_end(pki):
    from cascade.robotics.endpoint import RemoteRobotRuntime
    with _served(tls=pki.good) as rig:
        port = urlsplit(rig.origin).port
        assert rig.origin == f"https://127.0.0.1:{port}"
        assert rig.endpoint.tls_certificate_sha256 == pki.good.sha256
        assert _call(rig.origin, "GET", "/v1/health", token=None, protocol=None, ca=pki.good.ca) == (
            200, {"protocol": PROTOCOL, "alive": True})
        for origin in (rig.origin, f"https://localhost:{port}"):      # IP and DNS subjectAltName
            remote = RemoteRobotRuntime(origin, TOKEN, tls_ca=pki.good.ca)
            try:
                remote.connect()
                assert remote.robot_id == "fixture" and remote.ready() == {"ready": True, "reason": None}
                receipt = remote.endpoint_receipt()
                assert receipt == {"origin": origin, "protocol": PROTOCOL, "catalog_sha256": remote.catalog_sha256,
                                   "lease_ttl_s": 5.0, "tls": {"certificate_sha256": pki.good.sha256, "trust": "ca"}}
                read = remote.execute("body.read", {}, expected_generation=remote.cancellation_token,
                                      deadline_monotonic_s=time.monotonic() + 5)
                assert read["ok"] is True and rig.trace.rows[-1][0] == "body.read"
                assert remote.stop()["ok"] is True and rig.runtime.stopped and remote.stopped is True
                assert remote.reset_stop(expected_generation=remote.cancellation_token)["ok"] is True
                assert remote.stopped is False
            finally:
                assert remote.close()["lease_released"] is True
        assert [row[0] for row in rig.trace.rows] == ["body.read", "body.read"] and rig.body.stops == 2


def test_wrong_ca_wrong_name_and_plaintext_peers_are_refused_before_any_command(pki, tmp_path):
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    with _served(tls=pki.good) as rig:
        port = urlsplit(rig.origin).port
        stranger = RemoteRobotRuntime(rig.origin, TOKEN, tls_ca=pki.foreign.ca)       # another CA
        with pytest.raises(RobotEndpointError, match="SSLCertVerificationError"):
            stranger.connect()
        assert stranger.stop()["ok"] is False and stranger.ready()["ready"] is False
        stranger.close()
        plain = RemoteRobotRuntime(f"http://127.0.0.1:{port}", TOKEN)              # plaintext client
        with pytest.raises(RobotEndpointError, match="transport failed"):
            plain.connect()
        assert plain.stop()["ok"] is False
        plain.close()
        with pytest.raises((http.client.HTTPException, OSError)):
            _call(f"http://127.0.0.1:{port}", "POST", "/v1/stop", {})            # valid token, in clear
        with pytest.raises(OSError):
            RemoteRobotRuntime(rig.origin, TOKEN, tls_ca=tmp_path / "missing-ca.pem")
        assert rig.body.stops == 0 and not rig.runtime.stopped and rig.trace.rows == []
        assert _call(rig.origin, "GET", "/v1/state", ca=pki.good.ca)[1]["lease_held"] is False
        # Still served after every refused handshake.
        remote = RemoteRobotRuntime(rig.origin, TOKEN, tls_ca=pki.good.ca).connect()
        assert remote.close()["lease_released"] is True
    with _served(tls=pki.misnamed) as rig:                    # right CA, certificate for another host
        remote = RemoteRobotRuntime(rig.origin, TOKEN, tls_ca=pki.misnamed.ca)
        with pytest.raises(RobotEndpointError, match="SSLCertVerificationError"):
            remote.connect()
        remote.close()
        assert rig.endpoint._state()["lease_held"] is False and rig.body.stops == 0
    with _served() as rig:                                    # TLS client, plaintext endpoint
        port = urlsplit(rig.origin).port
        remote = RemoteRobotRuntime(f"https://127.0.0.1:{port}", TOKEN, tls_ca=pki.good.ca)
        with pytest.raises(RobotEndpointError, match="transport failed"):
            remote.connect()
        remote.close()
        assert rig.endpoint._state()["lease_held"] is False


def test_bearer_token_still_gates_every_route_over_tls(pki):
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    with _served(tls=pki.good) as rig:
        for method, path, payload in ROUTES:
            for token in (None, STALE_TOKEN):                  # missing, or rotated away
                status, body = _call(rig.origin, method, path, payload, token=token, ca=pki.good.ca)
                assert (status, body["protocol"]) == (403, PROTOCOL), (method, path, token)
            status, _ = _call(rig.origin, method, path, payload, protocol="cascade.robot-runtime/2", ca=pki.good.ca)
            assert status == 400, (method, path)
        stale = RemoteRobotRuntime(rig.origin, STALE_TOKEN, tls_ca=pki.good.ca)
        with pytest.raises(RobotEndpointError, match="403"):
            stale.connect()
        assert stale.stop()["ok"] is False
        stale.close()
        assert rig.body.stops == 0 and rig.trace.rows == [] and not rig.runtime.stopped
        assert rig.endpoint._state()["lease_held"] is False


def test_fingerprint_pin_is_checked_before_a_single_request_byte(pki):
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    for trust in ({"tls_fingerprint": pki.foreign.sha256},
                  {"tls_ca": pki.good.ca, "tls_fingerprint": pki.foreign.sha256}):
        peer = _RecordingTLS(pki.good)
        try:
            remote = RemoteRobotRuntime(f"https://127.0.0.1:{peer.port}", TOKEN, **trust)
            with pytest.raises(RobotEndpointError, match="fingerprint"):
                remote.connect()
            remote.close()
            assert peer.done.wait(60)
            assert peer.handshakes == 1 and peer.received == [b""], trust    # the token never left
        finally:
            peer.close()
    colon = ":".join(pki.good.sha256[i:i + 2] for i in range(0, 64, 2)).upper()
    with _served(tls=pki.good) as rig:
        for trust, kind in (({"tls_fingerprint": colon}, "fingerprint"),
                            ({"tls_ca": pki.good.ca, "tls_fingerprint": pki.good.sha256}, "ca+fingerprint")):
            remote = RemoteRobotRuntime(rig.origin, TOKEN, **trust).connect()
            try:
                assert remote.endpoint_receipt()["tls"] == {"certificate_sha256": pki.good.sha256, "trust": kind}
                assert remote.stop()["ok"] is True
            finally:
                assert remote.close()["lease_released"] is True
        for bad in ("", "zz" * 32, pki.good.sha256[:-2], pki.good.sha256 + "00", 5):
            with pytest.raises(ValueError):
                RemoteRobotRuntime(rig.origin, TOKEN, tls_fingerprint=bad)
        for origin, trust in ((rig.origin, {}), (rig.origin.replace("https", "http"), {"tls_ca": pki.good.ca})):
            with pytest.raises(ValueError):
                RemoteRobotRuntime(origin, TOKEN, **trust)        # TLS is selected explicitly, both ways


def test_a_stalled_handshake_never_blocks_stop_and_is_dropped(pki, monkeypatch):
    """The handshake runs in the connection's own thread with its own timeout, never in the accept loop;
    after it, the request is read exactly as in clear (no timeout)."""
    from cascade.robotics import endpoint as module
    monkeypatch.setattr(module, "TLS_HANDSHAKE_TIMEOUT_S", .5)
    with _served(tls=pki.good) as rig:
        port = urlsplit(rig.origin).port
        stalled = socket.create_connection(("127.0.0.1", port), timeout=20)   # TCP only: no ClientHello
        try:
            status, body = _call(rig.origin, "POST", "/v1/stop", {}, ca=pki.good.ca, timeout=20)
            assert status == 200 and body["result"]["latched"] is True and rig.runtime.stopped
            try:
                data = stalled.recv(1)
            except ConnectionResetError:
                data = b""
            assert data == b""                       # dropped by the server, not by this test's timeout
        finally:
            stalled.close()
        context = ssl.create_default_context(cafile=str(pki.good.ca))
        with socket.create_connection(("127.0.0.1", port), timeout=20) as raw:
            with context.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
                time.sleep(1.5)                      # past the handshake limit: it bounds the handshake only
                tls.sendall(_raw_request("GET", "/v1/state"))
                reply = b""
                while chunk := tls.recv(65536):
                    reply += chunk
        assert reply.startswith(b"HTTP/1.0 200 OK") and b'"stopped": true' in reply


def test_endpoint_start_fails_closed_off_loopback_without_tls(pki):
    from cascade.robotics.endpoint import RobotRuntimeEndpoint, client_tls_context, server_tls_context
    for context in (server_tls_context(pki.good.cert, pki.good.key), client_tls_context(pki.good.ca),
                    client_tls_context()):
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    runtime = RobotRuntime({"body": _Body()})
    try:
        endpoint = RobotRuntimeEndpoint(runtime, robot_id="fixture", token=TOKEN, lease_ttl_s=30)
        cert, key = str(pki.good.cert), str(pki.good.key)
        for host in ("192.0.2.1", "10.0.0.5", "fd00::5"):
            with pytest.raises(ValueError, match="requires TLS"):
                endpoint.start(host=host, port=0)
        for kwargs in ({"tls_cert": cert}, {"tls_key": key}):
            with pytest.raises(ValueError, match="certificate and its key"):
                endpoint.start(port=0, **kwargs)
        for host in ("0.0.0.0", "::", "224.0.0.1"):
            with pytest.raises(ValueError, match="one explicit unicast IP"):
                endpoint.start(host=host, port=0, tls_cert=cert, tls_key=key)
        with pytest.raises(OSError):
            endpoint.start(port=0, tls_cert=str(pki.root / "missing.pem"), tls_key=key)
        with pytest.raises(ssl.SSLError):
            endpoint.start(port=0, tls_cert=cert, tls_key=str(pki.foreign.key))     # another certificate's key
        assert endpoint.origin is None and endpoint.tls_certificate_sha256 is None
        origin = endpoint.start(port=0, tls_cert=cert, tls_key=key)               # nothing was half-started
        assert origin.startswith("https://127.0.0.1:") and endpoint.tls_certificate_sha256 == pki.good.sha256
        assert endpoint.close()["ok"] is True
    finally:
        runtime.close()


@needs_openssl
def test_tls_endpoint_serves_a_non_loopback_address(tmp_path):
    """Cross-interface on one host (no second host here): a private address, its own certificate."""
    from cascade.robotics.endpoint import RemoteRobotRuntime
    host = _private_ipv4()
    if host is None:
        pytest.skip("no private IPv4 address on this host")
    identity = _issue(tmp_path / "pki", f"IP:{host}")
    with _served(tls=identity, host=host) as rig:
        assert rig.origin == f"https://{host}:{urlsplit(rig.origin).port}"
        remote = RemoteRobotRuntime(rig.origin, TOKEN, tls_ca=identity.ca).connect()
        try:
            assert remote.endpoint_receipt()["tls"]["certificate_sha256"] == identity.sha256
            assert remote.stop()["ok"] is True and rig.runtime.stopped
        finally:
            assert remote.close()["lease_released"] is True


def _robot_cli(tmp_path, *flags):
    """The real entry point in a child; bounded, so a mutant that starts serving fails instead of hanging."""
    argv = [sys.executable, "-m", "cascade.apps.robot_service", "--robot", "conversation_mock",
            "--port", str(PORT_BLOCK[0]), "--run-dir", str(tmp_path / "run"), "--token-env", TOKEN_ENV, *flags]
    return subprocess.run(argv, cwd=REPO, env=_env(TOKEN), capture_output=True, text=True, timeout=120)


def test_robot_service_cli_fails_closed_before_building_anything(tmp_path):
    missing = str(tmp_path / "missing.pem")
    cases = ((["--host", "10.0.0.5"], "requires TLS"), (["--host", "192.0.2.1"], "requires TLS"),
             (["--tls-cert", missing], "go together"), (["--tls-key", missing], "go together"),
             (["--host", "0.0.0.0", "--tls-cert", missing, "--tls-key", missing], "one explicit unicast IP"),
             (["--tls-cert", missing, "--tls-key", missing], "TLS certificate"))
    for flags, reason in cases:
        result = _robot_cli(tmp_path, *flags)
        assert result.returncode == 2 and reason in result.stderr, (flags, result.stderr)
        assert not (tmp_path / "run").exists(), flags      # refused before any runtime or run directory


@needs_openssl
def test_robot_service_cli_refuses_a_key_of_another_certificate(tmp_path, pki):
    result = _robot_cli(tmp_path, "--tls-cert", str(pki.good.cert), "--tls-key", str(pki.foreign.key))
    assert result.returncode == 2 and "TLS certificate" in result.stderr, result.stderr
    assert not (tmp_path / "run").exists()


# --------------------------------------------------------------------- configuration


def test_conversation_configuration_selects_tls_only_explicitly(tmp_path):
    from cascade.apps.conversation import parser
    from cascade.conversation.service import configuration
    from cascade.robotics.endpoint import endpoint_origin
    sha = "ab" * 32
    base = {"version": 1, "provider_url": f"ws://127.0.0.1:{PORT_BLOCK[1]}/v1/realtime", "run_root": "runs",
            "robot_token_env": "ROBOT_TOKEN"}
    path = tmp_path / "service.json"

    def load(**fields):
        path.write_text(json.dumps({**base, **fields}))
        return configuration(parser().parse_args(["--config", str(path)]))

    values = load(robot_endpoint=f"https://Robot.LAN:{PORT_BLOCK[1]}/", robot_tls_ca="pki/ca.pem")
    assert values["robot_endpoint"] == f"https://robot.lan:{PORT_BLOCK[1]}"
    assert values["robot_tls_ca"] == (tmp_path / "pki" / "ca.pem").resolve() and values["robot_tls_fingerprint"] is None
    colon = ":".join(sha[i:i + 2] for i in range(0, 64, 2)).upper()
    values = load(robot_endpoint=f"https://10.0.0.5:{PORT_BLOCK[1]}", robot_tls_fingerprint=colon)
    assert values["robot_tls_fingerprint"] == sha and values["robot_tls_ca"] is None
    values = load(robot_endpoint=f"https://[fd00::5]:{PORT_BLOCK[1]}", robot_tls_ca="/etc/robot/ca.pem",
                  robot_tls_fingerprint=sha)
    assert values["robot_endpoint"] == f"https://[fd00::5]:{PORT_BLOCK[1]}"
    assert values["robot_tls_ca"] == Path("/etc/robot/ca.pem").resolve()
    values = load(robot_endpoint=f"http://127.0.0.1:{PORT_BLOCK[1]}")
    assert values["robot_tls_ca"] is None and values["robot_tls_fingerprint"] is None
    https, plain = f"https://10.0.0.5:{PORT_BLOCK[1]}", f"http://127.0.0.1:{PORT_BLOCK[1]}"
    bad = ({"robot_endpoint": https},                                       # a TLS URL without a trust anchor
           {"robot_endpoint": plain, "robot_tls_ca": "ca.pem"},             # trust without a TLS URL
           {"robot_endpoint": f"http://10.0.0.5:{PORT_BLOCK[1]}", "robot_tls_ca": "ca.pem"},
           {"robot_endpoint": https, "robot_tls_fingerprint": "ab" * 31},
           {"robot_endpoint": https, "robot_tls_fingerprint": "zz" * 32},
           {"robot_endpoint": https, "robot_tls_fingerprint": 5},
           {"robot_endpoint": https, "robot_tls_ca": ""}, {"robot_endpoint": https, "robot_tls_ca": 5},
           {"robot_endpoint": f"https://user@10.0.0.5:{PORT_BLOCK[1]}", "robot_tls_ca": "ca.pem"},
           {"robot_endpoint": https + "/v1", "robot_tls_ca": "ca.pem"},
           {"robot_endpoint": https + "?x=1", "robot_tls_ca": "ca.pem"},
           {"robot_endpoint": "https://10.0.0.5", "robot_tls_ca": "ca.pem"},
           {"robot_endpoint": None, "robot_token_env": None, "robot_tls_ca": "ca.pem"},
           {"robot_endpoint": None, "robot_token_env": None, "robot_tls_fingerprint": sha})
    for change in bad:
        with pytest.raises(ValueError):
            load(**change)
    cli = configuration(parser().parse_args([
        "--provider-url", f"ws://127.0.0.1:{PORT_BLOCK[1]}/v1/realtime", "--run-dir", str(tmp_path / "run"),
        "--robot-endpoint", f"https://robot.lan:{PORT_BLOCK[2]}", "--robot-token-env", "ROBOT_TOKEN",
        "--robot-tls-ca", "ca.pem", "--robot-tls-fingerprint", colon]))
    assert cli["robot_tls_ca"] == Path("ca.pem").resolve() and cli["robot_tls_fingerprint"] == sha
    example = configuration(parser().parse_args(["--config", str(REPO / "configs/conversation/remote_robot_tls.json")]))
    assert example["robot_endpoint"].startswith("https://") and example["robot_tls_ca"] is not None
    assert endpoint_origin(f"https://ROBOT.lan:{PORT_BLOCK[3]}", tls=True) == f"https://robot.lan:{PORT_BLOCK[3]}"
    assert endpoint_origin(f"http://localhost:{PORT_BLOCK[3]}/") == f"http://localhost:{PORT_BLOCK[3]}"
    for url in (f"http://127.0.0.1:{PORT_BLOCK[3]}", "https://robot.lan:0", "https://:47303", 47303, None):
        with pytest.raises(ValueError):
            endpoint_origin(url, tls=True)


# ------------------------------------------------- real processes: robot + conversation over TLS


@needs_openssl
def test_split_deployment_over_tls_completes_a_text_turn_and_an_operator_stop(tmp_path, pki):
    aiohttp = pytest.importorskip("aiohttp")
    token = secrets.token_urlsafe(32)

    async def scenario():
        provider = await _Provider().start()
        robot = await _robot(tmp_path / "robot", token, tls=pki.good)
        conversation = None
        try:
            port = urlsplit(robot.origin).port
            assert robot.origin == f"https://127.0.0.1:{port}"
            robot_ready = robot.ready_record()
            assert list(robot_ready) == GOLDEN_ROBOT_READY_KEYS + ["tls_certificate_sha256"]
            assert robot_ready["tls_certificate_sha256"] == pki.good.sha256
            assert robot.banner == [f"Robot runtime endpoint {robot.origin} ({PROTOCOL})", BEARER_LINE,
                                    f"TLS certificate sha256 {pki.good.sha256}"]
            assert token not in (robot.run / "ready.json").read_text()
            conversation = await _conversation(
                tmp_path / "conversation", _values(robot, provider, robot_tls_ca=str(pki.good.ca),
                                                   start_stopped=True), token)
            ready = conversation.ready_record()
            assert ready["robot_endpoint"] == {"origin": robot.origin, "protocol": PROTOCOL,
                                               "catalog_sha256": robot_ready["catalog_sha256"], "lease_ttl_s": 5.0,
                                               "tls": {"certificate_sha256": pki.good.sha256, "trust": "ca"}}
            assert token not in (conversation.run / "ready.json").read_text()
            assert not (conversation.run / "domains").exists()        # no robot was built here
            async with aiohttp.ClientSession() as anonymous:
                readiness = await anonymous.get(conversation.url + "/readyz")
                assert readiness.status == 200 and (await readiness.json())["ready"] is True
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + conversation.token}) as client:
                status = await (await client.get(conversation.url + "/api/status")).json()
                assert status["stopped"] is True and robot.state()["stopped"] is True
                reset = await client.post(conversation.url + "/api/reset", json={"generation": status["generation"]})
                assert (await reset.json())["ok"] is True and robot.state()["stopped"] is False
                binding = await (await client.post(conversation.url + "/api/session",
                                                   json={"robot_id": "microduck-mock"})).json()
                sid = binding["session_id"]
                async with client.ws_connect(conversation.url + "/api/media?ticket=" + binding["ticket"]) as ws:
                    await _tool_turn(provider, ws, sid, "walk forward a little", "robot_tool_1",
                                     {"vx": .05, "vy": 0, "wz": 0, "duration_s": .2}, rid="r1", cid="c1")
                    walked = await _tool_output(provider, "c1")
                    assert walked["execution_ok"] is True and walked["outcome"] == "unverified"
                    shown = await _ws_next(ws, "tool_result")
                    assert shown["tool"] == "locomotion.walk_velocity" and shown["result"]["outcome"] == "unverified"
                    await _spoken_reply(provider, ws, "r2", "I walked over TLS; the robot reports it unverified.")
                    walks = [r for r in robot.trace_rows() if r["skill"] == "locomotion.walk_velocity"]
                    assert len(walks) == 1 and walks[0]["result"]["outcome"] == "unverified"
                    await _tool_turn(provider, ws, sid, "keep walking", "robot_tool_1",
                                     {"vx": .05, "vy": 0, "wz": 0, "duration_s": 3.0}, rid="r3", cid="c3")
                    await _until(lambda: robot.state()["executing"] == 1)
                    stop = await (await client.post(conversation.url + "/api/stop")).json()
                    assert stop["ok"] is True and robot.state()["stopped"] is True
                    await _until(lambda: robot.state()["executing"] == 0)
                    walks = [r for r in robot.trace_rows() if r["skill"] == "locomotion.walk_velocity"]
                    assert len(walks) == 2 and walks[1]["result"]["error"] == "operation superseded by stop"
                    assert (await _ws_next(ws, "authority_revoked"))["reconnect_required"] is True
                assert (await (await client.delete(conversation.url + "/api/session")).json())["ok"] is True
        finally:
            if conversation is not None:
                await conversation.terminate()
            await robot.terminate()
            await provider.close()
        assert conversation.returncode == 0 and conversation.closure()["ok"] is True
        assert robot.returncode == 0 and robot.closure()["ok"] is True
        assert robot.closure()["lease_expiry_stops"] == []
    asyncio.run(scenario())


@needs_openssl
def test_conversation_refuses_a_robot_it_cannot_verify_or_authenticate(tmp_path, pki):
    """Wrong CA, wrong pin, plaintext to TLS, stale or missing token: no ready.json, no lease, no command."""
    pytest.importorskip("aiohttp")
    token = secrets.token_urlsafe(32)

    async def scenario():
        robot = await _robot(tmp_path / "robot", token, tls=pki.good)
        try:
            port = urlsplit(robot.origin).port
            cases = (("wrong-ca", {"robot_tls_ca": str(pki.foreign.ca)}, token, "RobotEndpointError"),
                     ("wrong-pin", {"robot_tls_fingerprint": pki.foreign.sha256}, token, "RobotEndpointError"),
                     ("plaintext", {"robot_endpoint": f"http://127.0.0.1:{port}"}, token, "RobotEndpointError"),
                     ("stale-token", {"robot_tls_ca": str(pki.good.ca)}, secrets.token_urlsafe(32),
                      "RobotEndpointError"),
                     ("missing-token", {"robot_tls_ca": str(pki.good.ca)}, None, "ValueError"))
            for name, values, credential, error in cases:
                service = await _conversation(tmp_path / name, {
                    "robot": "microduck-mock", "robot_endpoint": robot.origin, "robot_token_env": TOKEN_ENV,
                    "allow_tools": ["locomotion.get_base_state"], **values}, credential, expect_ready=False)
                assert service.returncode == 1, (name, service.stderr)
                closure = service.closure()
                assert closure["ok"] is False and closure["service_error"] == {"type": error}, name
                assert not (service.run / "ready.json").exists(), name
                state = robot.state()
                assert state["lease_held"] is False and state["stopped"] is False, name
                assert robot.trace_rows() == [], name
        finally:
            await robot.terminate()
        assert robot.returncode == 0 and robot.closure()["lease_expiry_stops"] == []
    asyncio.run(scenario())


@needs_openssl
def test_killed_conversation_over_tls_still_latches_the_robot_stop(tmp_path, pki):
    aiohttp = pytest.importorskip("aiohttp")
    token = secrets.token_urlsafe(32)

    async def scenario():
        provider = await _Provider().start()
        robot = await _robot(tmp_path / "robot", token, tls=pki.good, lease_ttl_s=1.0)
        first = None
        try:
            first = await _conversation(tmp_path / "first",
                                        _values(robot, provider, robot_tls_fingerprint=pki.good.sha256), token)
            assert first.ready_record()["robot_endpoint"]["tls"] == {"certificate_sha256": pki.good.sha256,
                                                                      "trust": "fingerprint"}
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + first.token}) as client:
                binding = await (await client.post(first.url + "/api/session",
                                                   json={"robot_id": "microduck-mock"})).json()
                async with client.ws_connect(first.url + "/api/media?ticket=" + binding["ticket"]) as ws:
                    await _tool_turn(provider, ws, binding["session_id"], "walk", "robot_tool_1",
                                     {"vx": .05, "vy": 0, "wz": 0, "duration_s": 3.0}, rid="r1", cid="c1")
                    await _until(lambda: robot.state()["executing"] == 1)
                    assert robot.state()["stopped"] is False and robot.state()["lease_held"] is True
                    first.process.send_signal(signal.SIGKILL)       # no cleanup, no stop request
                    await first.process.wait()
            state = await _until(lambda: (s := robot.state())["lease_expiry_stops"] == 1 and s)
            assert state["lease_held"] is False and state["stopped"] is True
            await _until(lambda: robot.state()["executing"] == 0)
            walk = [r for r in robot.trace_rows() if r["skill"] == "locomotion.walk_velocity"][-1]
            assert walk["result"]["error"] == "operation superseded by stop"
            assert not (first.run / "closure.json").exists()
        finally:
            if first is not None and first.process.returncode is None:
                first.process.kill()
                await first.process.wait()
            await robot.terminate()
            await provider.close()
        assert robot.returncode == 0 and robot.closure()["ok"] is True
        assert [entry["ok"] for entry in robot.closure()["lease_expiry_stops"]] == [True]
    asyncio.run(scenario())


# ----------------------------------------------------------- operator tools and measurement


@needs_openssl
def test_certificate_script_issues_a_private_ca_chain_with_exact_names(tmp_path):
    directory = tmp_path / "pki"
    first = _issue(directory, "IP:127.0.0.1", "DNS:Robot.Lan")
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for key in ("ca.key", "server.key"):
        assert stat.S_IMODE((directory / key).stat().st_mode) == 0o600, key
    assert first.info["san"] == ["IP:127.0.0.1", "DNS:robot.lan"] and first.cert == directory / "server.pem"
    text = subprocess.run(["openssl", "x509", "-noout", "-text", "-in", str(first.cert)],
                          capture_output=True, text=True, check=True).stdout
    assert "IP Address:127.0.0.1" in text and "DNS:robot.lan" in text and "CA:FALSE" in text
    assert "TLS Web Server Authentication" in text
    assert subprocess.run(["openssl", "verify", "-CAfile", str(first.ca), str(first.cert)],
                          capture_output=True).returncode == 0
    printed = subprocess.run(["openssl", "x509", "-noout", "-fingerprint", "-sha256", "-in", str(first.cert)],
                             capture_output=True, text=True, check=True).stdout
    assert printed.split("=", 1)[1].strip().replace(":", "").lower() == first.sha256
    ca = first.ca.read_bytes()
    second = _issue(directory, "IP:10.0.0.5", name="robot-b")
    assert first.ca.read_bytes() == ca and second.cert == directory / "robot-b.pem" and second.sha256 != first.sha256
    assert subprocess.run(["openssl", "verify", "-CAfile", str(first.ca), str(second.cert)],
                          capture_output=True).returncode == 0
    assert not list(directory.glob("*.csr")) and not list(directory.glob("*.ext"))
    _issue(directory, "IP:127.0.0.1", force=True)
    assert first.ca.read_bytes() != ca                          # --force rotates the CA


@needs_openssl
def test_certificate_script_tightens_key_modes_whatever_openssl_writes(tmp_path):
    """Some openssl builds create -keyout files with the umask's mode; the script must not depend on that."""
    shim = tmp_path / "bin" / "openssl"
    shim.parent.mkdir()
    shim.write_text("#!/bin/sh\n"
                    f"'{shutil.which('openssl')}' \"$@\" || exit $?\n"
                    "previous=\n"
                    "for argument in \"$@\"; do\n"
                    "  if [ \"$previous\" = -keyout ]; then chmod 644 \"$argument\"; fi\n"
                    "  previous=$argument\n"
                    "done\n")
    shim.chmod(0o755)
    result = subprocess.run([sys.executable, str(CERTS), "--out", str(tmp_path / "pki"), "--san", "IP:127.0.0.1"],
                            env={**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"},
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for key in ("ca.key", "server.key"):
        assert stat.S_IMODE((tmp_path / "pki" / key).stat().st_mode) == 0o600, key


def test_certificate_script_refuses_ambiguous_names_before_writing(tmp_path):
    out = tmp_path / "pki"
    cases = ((["--san", "IP:0.0.0.0"], "unicast"), (["--san", "IP:224.0.0.1"], "unicast"),
             (["--san", "IP:10.0.0.300"], "invalid IP"), (["--san", "DNS:*.lan"], "subjectAltName must be"),
             (["--san", "DNS:"], "subjectAltName must be"), (["--san", "URI:x"], "subjectAltName must be"),
             (["--san", "10.0.0.5"], "subjectAltName must be"), ([], "--san"),
             (["--san", "IP:127.0.0.1", "--name", "ca"], "--name"),
             (["--san", "IP:127.0.0.1", "--name", "../x"], "--name"),
             (["--san", "IP:127.0.0.1", "--days", "0"], "--days"), (["--san", "IP:127.0.0.1", "--days", "826"], "--days"))
    for flags, reason in cases:
        result = subprocess.run([sys.executable, str(CERTS), "--out", str(out), *flags],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 2 and reason in result.stderr, (flags, result.stderr)
        assert not out.exists(), flags


def _nearest_rank(samples, q):
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


@needs_openssl
def test_latency_benchmark_reports_every_mode_without_judging_speed(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import bench_robot_endpoint as bench
    known = bench.stats([i / 1000 for i in range(1, 21)], ok=20)          # 1..20 ms
    assert (known["n"], known["min_ms"], known["max_ms"], known["ok"]) == (20, 1.0, 20.0, 20)
    assert (known["p50_ms"], known["p95_ms"]) == pytest.approx((10.0, 19.0))
    assert known["mean_ms"] == pytest.approx(10.5) and known["samples_ms"] == pytest.approx(list(range(1, 21)))
    assert [bench.succeeded(r) for r in ({"ok": True}, {"ok": False}, {"ok": False, "execution_ok": True},
                                         {"ok": True, "execution_ok": False}, {}, None)] == [
        True, False, True, False, False, False]
    out = tmp_path / "latency.json"
    result = subprocess.run([sys.executable, str(BENCH), "--calls", "3", "--warmup", "1", "--out", str(out),
                             "--work-dir", str(tmp_path / "work")], cwd=REPO, env=_env(),
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert (report["schema"], report["protocol"], report["calls"], report["warmup"]) == (1, PROTOCOL, 3, 1)
    assert report["robot"] == "microduck_conversation_mock" and report["tool"] == "locomotion.get_base_state"
    assert len(report["tls"]["certificate_sha256"]) == 64 and report["tls"]["server_key"] == "rsa:2048"
    for mode in ("in_process", "plaintext", "tls"):
        for operation in ("state", "execute"):
            stats = report["results"][mode][operation]
            assert stats["n"] == 3 == len(stats["samples_ms"]), (mode, operation)
            values = [stats[key] for key in ("min_ms", "p50_ms", "p95_ms", "max_ms")]
            assert all(math.isfinite(v) and v >= 0 for v in values + [stats["mean_ms"]])
            assert values == pytest.approx([min(stats["samples_ms"]), _nearest_rank(stats["samples_ms"], .5),
                                            _nearest_rank(stats["samples_ms"], .95), max(stats["samples_ms"])],
                                           abs=1e-5)                 # order statistics, not speed
    for mode in ("plaintext", "tls"):
        for operation in ("state", "execute"):
            hop = report["extra_hop_ms"][mode][operation]
            assert hop["p50_ms"] == pytest.approx(report["results"][mode][operation]["p50_ms"]
                                                  - report["results"]["in_process"][operation]["p50_ms"])
    assert report["results"]["plaintext"]["execute"]["ok"] == 3 and report["results"]["tls"]["execute"]["ok"] == 3
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").rglob("*.key"))   # throwaway keys gone


def test_documented_latency_numbers_are_the_recorded_measurement():
    report = json.loads(EVIDENCE.read_text())
    section = (REPO / "docs" / "CONVERSATION.md").read_text().split(
        "## Deploy the conversation service separately", 1)[1].split("\n## ", 1)[0]
    assert f"N = {report['calls']}" in section
    for mode in ("in_process", "plaintext", "tls"):
        for operation in ("state", "execute"):
            stats = report["results"][mode][operation]
            assert stats["n"] == report["calls"] == len(stats["samples_ms"]), (mode, operation)
            assert (stats["p50_ms"], stats["p95_ms"]) == pytest.approx(
                (_nearest_rank(stats["samples_ms"], .5), _nearest_rank(stats["samples_ms"], .95)), abs=1e-5)
            if operation == "execute":
                assert stats["ok"] == report["calls"], mode
            for key in ("p50_ms", "p95_ms"):
                assert f"{stats[key]:.2f}" in section, (mode, operation, key)
