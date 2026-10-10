"""B51: the conversation service deployed apart from the robot runtime.

`cascade.robot-runtime/1` (cascade/robotics/endpoint.py) serves one RobotRuntime
from its own process (`cascade-robot-service`); `cascade-conversation` with
`robot_endpoint` drives it remotely and builds no robot itself. Premise/golden
tests pin the default in-process path, which must stay byte-identical.

Every server here binds a loopback port the OS assigns (port 0, as B64's
tests/owned_server.py): the in-process ones report the bound port, the child
processes announce it on stdout and in their own ready.json under this test's
run directory, and a per-test bearer token makes any foreign listener refuse.
Block ports (46400-46499) appear only in values nothing binds or dials.
Mock ASR/LLM/TTS = a loopback Realtime provider stub; the robot is the mock
MicroDuck kinematic base behind its real SafeBase/RobotRuntime. No speech
inference, microphone, GPU or physics: transport and authority only.
"""
import asyncio
import base64
import copy
import hashlib
import http.client
import json
import math
import os
import secrets
import signal
import struct
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from cascade.robotics.contracts import ResourceDescriptor, ToolDescriptor
from cascade.robotics.runtime import RobotRuntime

REPO = Path(__file__).resolve().parents[1]
PORT_BLOCK = range(46400, 46500)
PROTOCOL = "cascade.robot-runtime/1"
TOKEN = "t" * 43
TOKEN_ENV = "TEST_B51_ROBOT_ENDPOINT_TOKEN"

# The in-process gateway's route table on be57535 (default deployment).
GOLDEN_ROUTES = {
    ("DELETE", "/api/session"), ("GET", "/"), ("GET", "/api/media"), ("GET", "/api/status"),
    ("GET", "/app.css"), ("GET", "/app.js"), ("GET", "/capture.js"), ("GET", "/playback.js"),
    ("HEAD", "/"), ("HEAD", "/api/media"), ("HEAD", "/api/status"), ("HEAD", "/app.css"),
    ("HEAD", "/app.js"), ("HEAD", "/capture.js"), ("HEAD", "/playback.js"), ("POST", "/api/reset"),
    ("POST", "/api/robot/start"), ("POST", "/api/session"), ("POST", "/api/stop")}
# ready.json keys published by the in-process service on be57535.
GOLDEN_READY_KEYS = {
    "schema", "state", "pid", "published_monotonic_s", "origin", "robot_id", "tools", "runtime_stopped",
    "provider_connected", "physical_admission", "intent_timeout_s", "execution_timeout_s",
    "robot_lifecycle", "service_config_sha256", "robot_config_sha256"}


# --------------------------------------------------------------------------- ports


def _loopback_port(origin):
    """The port of an announced http(s)/ws loopback origin; never 0."""
    parts = urlsplit(origin)
    assert parts.hostname == "127.0.0.1" and parts.port, origin
    return parts.port


# ------------------------------------------------------------- fixture robot (in-process)


class _Body:
    """A synthetic domain whose motion can block, like test_conversation_protocol.Robot."""
    domain_id = "body"

    def __init__(self):
        self.resources = (ResourceDescriptor("body/base", "base", "fixture", synthetic=True,
                                             admission="software_only", controller_id="fixture/controller",
                                             writer_id="body"),)
        walk = {"type": "object", "properties": {"vx": {"type": "number", "minimum": -.3, "maximum": .3}},
                "required": ["vx"], "additionalProperties": False}
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        self.tool_descriptors = (
            ToolDescriptor("body.walk", "Synthetic bounded walk", walk, "body", "walk",
                           effect="motion", writes=("body/base",)),
            ToolDescriptor("body.read", "Synthetic observation", empty, "body", "read"))
        self.calls, self.stops, self.resets = [], 0, 0
        self.entered, self.block = threading.Event(), None

    def execute(self, name, args):
        self.calls.append((name, args))
        self.entered.set()
        if self.block is not None:
            assert self.block.wait(20), "test did not release its blocked action"
        return {"ok": True, "synthetic": True, "postcondition": {"status": "unverified"}}

    def stop(self):
        self.stops += 1
        return {"ok": True}

    def reset_stop(self):
        self.resets += 1
        return {"ok": True}

    def close(self):
        return {"ok": True}


class _Trace:
    def __init__(self):
        self.rows = []

    def record(self, name, args, result, duration_ms, **_context):
        self.rows.append((name, copy.deepcopy(args), copy.deepcopy(result)))


@contextmanager
def _served(*, ttl=5.0):
    from cascade.robotics.endpoint import RobotRuntimeEndpoint
    body, trace = _Body(), _Trace()
    runtime = RobotRuntime({"body": body}, trace=trace)
    endpoint = RobotRuntimeEndpoint(runtime, robot_id="fixture", token=TOKEN, lease_ttl_s=ttl)
    origin = endpoint.start(port=0)
    _loopback_port(origin)
    try:
        yield SimpleNamespace(body=body, trace=trace, runtime=runtime, endpoint=endpoint, origin=origin)
    finally:
        if body.block is not None:
            body.block.set()
        endpoint.close()
        runtime.close()


def _http(origin, method, path, body=None, *, token=TOKEN, protocol=PROTOCOL, timeout=20):
    url = urlsplit(origin)
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


def _state(rig):
    status, body = _http(rig.origin, "GET", "/v1/state")
    assert status == 200 and body["protocol"] == PROTOCOL
    return body


def _lease(rig):
    status, body = _http(rig.origin, "POST", "/v1/lease", {"lease_id": None})
    assert status == 200 and body["ttl_s"] == rig.endpoint.lease_ttl_s
    return body["lease_id"]


def _wait(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError("condition not reached before the test deadline")


# --------------------------------------------------------------------- golden / premise


def test_default_configuration_keeps_old_values_and_needs_no_robot_endpoint(tmp_path):
    """Premise: the default service still selects the in-process robot runtime."""
    from cascade.apps.conversation import parser
    from cascade.conversation.service import configuration
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"version": 1, "provider_url": "ws://127.0.0.1:8765/v1/realtime",
                                "run_root": "runs"}))
    values = configuration(parser().parse_args(["--config", str(path)]))
    assert values.get("robot_endpoint") is None and values.get("robot_token_env") is None
    old = {"robot": "conversation_mock", "provider_url": "ws://127.0.0.1:8765/v1/realtime", "token_env": None,
           "allow_tool": [], "allow_motion": False, "barge_in": "stop_robot", "port": 8780, "config_dir": None,
           "run_dir": None, "run_root": tmp_path / "runs", "start_stopped": False, "intent_timeout_s": 10,
           "execution_timeout_s": 30, "provider_release_contract": None, "robot_lifecycle": None,
           "service_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert {key: values[key] for key in old} == old


def test_default_gateway_route_table_is_unchanged():
    """Golden: probe routes exist only in split deployments."""
    pytest.importorskip("aiohttp")
    from cascade.conversation.domain import ConversationDomain
    from cascade.conversation.gateway import ConversationGateway
    runtime = RobotRuntime({"body": _Body()})
    try:
        domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=("body.read",))
        gate = ConversationGateway(domain, lambda: None)
        assert {(r.method, r.resource.canonical) for r in gate.application().router.routes()} == GOLDEN_ROUTES
    finally:
        runtime.close()


def test_split_gateway_probes_report_robot_readiness_and_never_ready_while_closing():
    pytest.importorskip("aiohttp")
    from cascade.conversation.domain import ConversationDomain
    from cascade.conversation.gateway import ConversationGateway
    runtime = RobotRuntime({"body": _Body()})
    robot = {"ready": True, "reason": None}
    try:
        domain = ConversationDomain(runtime, robot_id="fixture", allow_tools=("body.read",))
        gate = ConversationGateway(domain, lambda: None, probes=lambda: dict(robot))
        routes = {(r.method, r.resource.canonical) for r in gate.application().router.routes()}
        assert routes - GOLDEN_ROUTES == {("GET", "/healthz"), ("HEAD", "/healthz"),
                                          ("GET", "/readyz"), ("HEAD", "/readyz")}

        async def probe(handler):
            response = await handler(None)
            return response.status, json.loads(response.body)

        assert asyncio.run(probe(gate._healthz)) == (200, {"alive": True})
        assert asyncio.run(probe(gate._readyz)) == (200, {"ready": True, "robot": robot})
        robot.update(ready=False, reason="lease lost")
        assert asyncio.run(probe(gate._readyz)) == (503, {"ready": False, "robot": robot})
        robot.update(ready=True, reason=None)
        gate._closing = True          # the first thing close() does
        assert asyncio.run(probe(gate._readyz))[0] == 503
    finally:
        runtime.close()


def test_default_in_process_service_publishes_the_old_ready_record(tmp_path):
    """Golden through the real CLI: same ready.json keys, no probe routes, robot built in-process."""
    aiohttp = pytest.importorskip("aiohttp")

    async def scenario():
        service = await _conversation_service(tmp_path, {"robot": "conversation_mock",
                                                         "allow_tools": ["sensing.read_sensor"]})
        try:
            ready = service.ready_record()
            assert set(ready) == GOLDEN_READY_KEYS
            assert ready["robot_id"] == "conversation_mock" and len(ready["robot_config_sha256"]) == 64
            assert (service.run / "domains").is_dir()      # build_robot_runtime ran in this process
            async with aiohttp.ClientSession() as anonymous:
                for path in ("/healthz", "/readyz"):
                    assert (await anonymous.get(service.url + path)).status == 404
        finally:
            await service.terminate()
        assert service.returncode == 0 and service.closure()["ok"]
    asyncio.run(scenario())


# ------------------------------------------------------------------ protocol endpoint


def test_endpoint_probes_are_open_and_every_other_route_needs_token_and_protocol():
    with _served() as rig:
        assert _http(rig.origin, "GET", "/v1/health", token=None, protocol=None) == (
            200, {"protocol": PROTOCOL, "alive": True})
        routes = (("GET", "/v1/describe", None), ("GET", "/v1/state", None),
                  ("POST", "/v1/lease", {"lease_id": None}), ("DELETE", "/v1/lease", {"lease_id": "x"}),
                  ("POST", "/v1/stop", {}), ("POST", "/v1/reset", {"lease_id": "x", "expected_generation": 0}),
                  ("POST", "/v1/execute", {"lease_id": "x", "tool": "body.read", "arguments": {},
                                           "expected_generation": 0, "deadline_remaining_s": 5}),
                  ("POST", "/v1/record", {"lease_id": "x", "tool": "emergency_stop", "arguments": {},
                                          "result": {"ok": True}, "duration_ms": 1}))
        for method, path, payload in routes:
            for token in (None, "x" * 43):
                status, body = _http(rig.origin, method, path, payload, token=token)
                assert (status, body["protocol"]) == (403, PROTOCOL), (method, path)
            status, body = _http(rig.origin, method, path, payload, protocol="cascade.robot-runtime/2")
            assert (status, body["protocol"]) == (400, PROTOCOL), (method, path)
        # Nothing from an unauthenticated or mismatched peer reached the robot.
        assert rig.body.stops == 0 and rig.trace.rows == [] and not rig.runtime.stopped
        assert _state(rig)["lease_held"] is False
        status, body = _http(rig.origin, "GET", "/v1/describe")
        assert status == 200 and body["robot_id"] == "fixture"
        assert body["tools"] == [d.as_dict() for d in rig.runtime.tool_descriptors.values()]
        assert body["resources"] == [r.as_dict() for r in rig.runtime.resources]
        assert body["trace_recorded"] is True and body["lease_ttl_s"] == 5.0
        assert _http(rig.origin, "GET", "/v1/unknown") == (404, {"protocol": PROTOCOL, "error": "unknown route"})
        rig.endpoint.close()
        with pytest.raises(OSError):
            _http(rig.origin, "GET", "/v1/health", token=None, protocol=None, timeout=3)


def test_endpoint_refuses_weak_token_foreign_identity_bad_lease_and_public_bind():
    from cascade.robotics.endpoint import RobotRuntimeEndpoint
    runtime = RobotRuntime({"body": _Body()})
    try:
        for kwargs in ({"token": "x" * 31}, {"token": 7}, {"robot_id": "someone-else"},
                       {"lease_ttl_s": .2}, {"lease_ttl_s": 30.5}, {"lease_ttl_s": float("nan")},
                       {"lease_ttl_s": True}):
            with pytest.raises(ValueError):
                RobotRuntimeEndpoint(runtime, **{"robot_id": "fixture", "token": TOKEN, **kwargs})
        endpoint = RobotRuntimeEndpoint(runtime, robot_id="fixture", token=TOKEN, lease_ttl_s=30)
        for host in ("0.0.0.0", "192.0.2.1"):
            with pytest.raises(ValueError):
                endpoint.start(host=host, port=0)
        assert endpoint.close()["ok"] is True
    finally:
        runtime.close()


def test_execute_needs_the_live_lease_an_episode_generation_and_a_deadline():
    with _served() as rig:
        generation = _state(rig)["generation"]
        good = {"tool": "body.read", "arguments": {}, "expected_generation": generation,
                "deadline_remaining_s": 5.0}
        assert _http(rig.origin, "POST", "/v1/execute", {"lease_id": "0" * 32, **good})[0] == 410
        lease = _lease(rig)
        broken = ({"expected_generation": None}, {"expected_generation": -1}, {"expected_generation": True},
                  {"expected_generation": 1.0}, {"deadline_remaining_s": 0}, {"deadline_remaining_s": -1},
                  {"deadline_remaining_s": 300.5}, {"deadline_remaining_s": "5"},
                  {"deadline_remaining_s": float("nan")}, {"deadline_remaining_s": True},
                  {"tool": 3}, {"arguments": []}, {"extra": 1})
        for change in broken:
            status, body = _http(rig.origin, "POST", "/v1/execute", {"lease_id": lease, **good, **change})
            assert status == 400, change
        missing = {key: value for key, value in good.items() if key != "deadline_remaining_s"}
        assert _http(rig.origin, "POST", "/v1/execute", {"lease_id": lease, **missing})[0] == 400
        assert rig.trace.rows == [] and rig.body.calls == []   # nothing malformed reached the runtime
        status, body = _http(rig.origin, "POST", "/v1/execute", {"lease_id": lease, **good})
        assert status == 200 and body["result"]["ok"] is True and body["protocol"] == PROTOCOL
        assert rig.trace.rows[-1][0] == "body.read" and rig.body.calls == [("read", {})]
        # The runtime's own generation and deadline fences still decide.
        stale = _http(rig.origin, "POST", "/v1/execute", {"lease_id": lease, **good,
                                                           "expected_generation": generation + 1})[1]
        assert "stale execution generation" in stale["result"]["error"]
        late = _http(rig.origin, "POST", "/v1/execute", {"lease_id": lease, **good, "tool": "body.walk",
                                                          "arguments": {"vx": .1}, "deadline_remaining_s": 1e-6})[1]
        assert "deadline expired" in late["result"]["error"]
        assert ("walk", {"vx": .1}) not in rig.body.calls
        assert _state(rig)["executing"] == 0


def test_one_supervisor_holds_the_lease_and_its_expiry_latches_the_robot_stop():
    with _served(ttl=1.0) as rig:
        first = _lease(rig)
        assert _http(rig.origin, "POST", "/v1/lease", {"lease_id": None})[0] == 409
        assert _http(rig.origin, "POST", "/v1/lease", {"lease_id": "f" * 32})[0] == 410
        renewed = _http(rig.origin, "POST", "/v1/lease", {"lease_id": first})
        assert renewed == (200, {"protocol": PROTOCOL, "lease_id": first, "ttl_s": 1.0})
        before = _state(rig)
        assert before["lease_held"] is True and before["stopped"] is False and rig.body.stops == 0
        # The expiry is counted after runtime.stop() returned its receipt (the latch flips earlier).
        expired = _wait(lambda: (state := _state(rig))["lease_expiry_stops"] == 1 and state)
        assert expired["lease_held"] is False and expired["stopped"] is True
        assert expired["generation"] > before["generation"] and rig.body.stops == 1
        assert _http(rig.origin, "POST", "/v1/lease", {"lease_id": first})[0] == 410
        assert _http(rig.origin, "POST", "/v1/execute", {
            "lease_id": first, "tool": "body.read", "arguments": {},
            "expected_generation": expired["generation"], "deadline_remaining_s": 5})[0] == 410
        second = _lease(rig)
        assert second != first
        assert _http(rig.origin, "DELETE", "/v1/lease", {"lease_id": first})[0] == 410
        assert _http(rig.origin, "DELETE", "/v1/lease", {"lease_id": second}) == (
            200, {"protocol": PROTOCOL, "released": True})
        time.sleep(1.5)   # a released lease never fires the expiry stop
        assert _state(rig)["lease_expiry_stops"] == 1 and rig.body.stops == 1
        receipt = rig.endpoint.close()
        assert receipt["ok"] and receipt["lease_expiry_stops"] == [{"ok": True, "generation": expired["generation"]}]


def test_operator_stop_needs_no_lease_and_wins_over_a_blocked_motion():
    with _served() as rig:
        lease = _lease(rig)
        rig.body.block = threading.Event()
        generation, outcome = _state(rig)["generation"], {}
        worker = threading.Thread(target=lambda: outcome.update(reply=_http(
            rig.origin, "POST", "/v1/execute", {"lease_id": lease, "tool": "body.walk", "arguments": {"vx": .1},
                                                "expected_generation": generation, "deadline_remaining_s": 30.0},
            timeout=60)))
        worker.start()
        assert rig.body.entered.wait(20)
        assert _state(rig)["executing"] == 1
        assert _http(rig.origin, "POST", "/v1/stop", {"lease_id": lease})[0] == 400   # exact empty body
        status, body = _http(rig.origin, "POST", "/v1/stop", {})               # no lease at all
        assert status == 200 and body["result"]["ok"] is True and body["result"]["latched"] is True
        assert body["result"]["physical_stop_verified"] is False
        assert worker.is_alive() and _state(rig)["stopped"] is True   # stop did not wait for the motion
        rig.body.block.set()
        worker.join(20)
        status, body = outcome["reply"]
        assert status == 200 and body["result"]["error"] == "operation superseded by stop"
        assert _state(rig)["executing"] == 0


def test_reset_needs_the_lease_and_the_observed_generation_and_execute_cannot_reset():
    with _served() as rig:
        lease = _lease(rig)
        assert _http(rig.origin, "POST", "/v1/stop", {})[0] == 200
        state = _state(rig)
        assert state["stopped"] is True
        via_execute = _http(rig.origin, "POST", "/v1/execute", {
            "lease_id": lease, "tool": "reset_stop", "arguments": {},
            "expected_generation": state["generation"], "deadline_remaining_s": 5.0})[1]
        assert via_execute["result"]["ok"] is False and _state(rig)["stopped"] is True
        assert _http(rig.origin, "POST", "/v1/reset", {"lease_id": "0" * 32,
                                                        "expected_generation": state["generation"]})[0] == 410
        for broken in ({"lease_id": lease}, {"lease_id": lease, "expected_generation": None},
                       {"lease_id": lease, "expected_generation": True}):
            assert _http(rig.origin, "POST", "/v1/reset", broken)[0] == 400
        stale = _http(rig.origin, "POST", "/v1/reset", {"lease_id": lease,
                                                         "expected_generation": state["generation"] - 1})[1]
        assert stale["result"]["ok"] is False and _state(rig)["stopped"] is True and rig.body.resets == 0
        reset = _http(rig.origin, "POST", "/v1/reset", {"lease_id": lease,
                                                         "expected_generation": state["generation"]})[1]
        assert reset["result"]["ok"] is True and _state(rig)["stopped"] is False and rig.body.resets == 1


def test_record_writes_only_stop_tool_rows_into_the_robot_trace():
    with _served() as rig:
        lease = _lease(rig)
        row = {"tool": "emergency_stop", "arguments": {}, "result": {"ok": True, "latched": True},
               "duration_ms": 2.5}
        assert _http(rig.origin, "POST", "/v1/record", {"lease_id": "0" * 32, **row})[0] == 410
        for change in ({"tool": "body.read"}, {"tool": "body.walk"}, {"tool": "nope"},
                       {"duration_ms": -1}, {"duration_ms": float("inf")}, {"result": []}, {"arguments": 3}):
            assert _http(rig.origin, "POST", "/v1/record", {"lease_id": lease, **row, **change})[0] == 400, change
        assert rig.trace.rows == []
        assert _http(rig.origin, "POST", "/v1/record", {"lease_id": lease, **row}) == (
            200, {"protocol": PROTOCOL, "recorded": True})
        assert rig.trace.rows == [("emergency_stop", {}, {"ok": True, "latched": True})]
        assert rig.body.stops == 0     # recording never delivers a stop


# ------------------------------------------------------------------- remote runtime client


def test_remote_runtime_mirrors_the_catalog_and_fails_closed_without_its_endpoint():
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    with _served() as rig:
        remote = RemoteRobotRuntime(rig.origin, TOKEN)
        try:
            remote.connect()
            assert remote.robot_id == "fixture" and remote.lease_ttl_s == 5.0 and remote.lease_id
            assert [r.as_dict() for r in remote.resources] == [r.as_dict() for r in rig.runtime.resources]
            assert ({name: d.as_dict() for name, d in remote.tool_descriptors.items()} ==
                    {name: d.as_dict() for name, d in rig.runtime.tool_descriptors.items()})
            assert remote.trace is not None and remote.ready() == {"ready": True, "reason": None}
            assert remote.cancellation_token == rig.runtime.cancellation_token and remote.stopped is False
            read = remote.execute("body.read", {}, expected_generation=remote.cancellation_token,
                                  deadline_monotonic_s=time.monotonic() + 5)
            assert read["ok"] is True and rig.trace.rows[-1][0] == "body.read"   # the robot process traced it
            rows = len(rig.trace.rows)
            expired = remote.execute("body.read", {}, expected_generation=remote.cancellation_token,
                                     deadline_monotonic_s=time.monotonic() - .001)
            untokened = remote.execute("body.read", {})
            assert expired == {"ok": False, "error": "ValueError: execution deadline expired"}
            assert untokened == {"ok": False, "error": "ValueError: remote execution requires an episode "
                                                       "generation and a local deadline"}
            assert len(rig.trace.rows) == rows        # refused locally: nothing was sent
            stop = remote.stop()
            assert stop["ok"] is True and rig.runtime.stopped and remote.stopped is True
            assert remote.reset_stop(expected_generation=remote.cancellation_token)["ok"] is True
            assert remote.stopped is False
            rig.endpoint.close()
            assert remote.stopped is True                      # unknown is reported as stopped
            with pytest.raises(RobotEndpointError):
                remote.cancellation_token
            lost = remote.execute("body.walk", {"vx": .1}, expected_generation=0,
                                  deadline_monotonic_s=time.monotonic() + 5)
            assert lost["ok"] is False and lost["execution_ok"] is False and lost["delivery_uncertain"] is True
            assert remote.stop()["ok"] is False and remote.stop()["physical_stop_verified"] is False
            assert remote.reset_stop(expected_generation=0)["ok"] is False
            assert remote.ready()["ready"] is False
        finally:
            receipt = remote.close()
        assert receipt["robot_runtime_closed"] is False and receipt["lease_released"] is False
        assert receipt["ok"] is False


def test_remote_runtime_refuses_another_protocol_version_before_taking_a_lease():
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    seen = []

    class Foreign(BaseHTTPRequestHandler):
        def _answer(self):
            seen.append((self.command, self.path))
            payload = json.dumps({"protocol": "cascade.robot-runtime/2", "robot_id": "fixture",
                                  "resources": [], "tools": [], "trace_recorded": False,
                                  "lease_ttl_s": 3.0, "lease_id": "a" * 32, "ttl_s": 3.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = _answer

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Foreign)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        remote = RemoteRobotRuntime(f"http://127.0.0.1:{port}", TOKEN)
        with pytest.raises(RobotEndpointError, match="protocol"):
            remote.connect()
        assert ("POST", "/v1/lease") not in seen and remote.lease_id is None
        remote.close()
    finally:
        server.shutdown()
        server.server_close()


def test_remote_origin_and_token_validation():
    from cascade.robotics.endpoint import RemoteRobotRuntime, endpoint_origin, endpoint_token
    assert endpoint_origin("http://127.0.0.1:46400") == "http://127.0.0.1:46400"
    assert endpoint_origin("http://localhost:46400/") == "http://localhost:46400"
    assert endpoint_origin("http://[::1]:46400") == "http://[::1]:46400"
    for url in ("https://127.0.0.1:46400", "http://10.0.0.5:46400", "http://127.0.0.1",
                "http://user:pw@127.0.0.1:46400", "http://127.0.0.1:46400/v1", "http://127.0.0.1:46400?x=1",
                "http://127.0.0.1:46400#f", "ws://127.0.0.1:46400", 46400, None, "http://127.0.0.1:0"):
        with pytest.raises(ValueError):
            endpoint_origin(url)
    with pytest.raises(ValueError):
        RemoteRobotRuntime("http://127.0.0.1:46400", "short")
    os.environ.pop(TOKEN_ENV, None)
    for name, value in ((TOKEN_ENV, None), (TOKEN_ENV, "x" * 31), ("not an identifier", TOKEN)):
        if value is not None:
            os.environ[TOKEN_ENV] = value
        try:
            with pytest.raises(ValueError):
                endpoint_token(name)
        finally:
            os.environ.pop(TOKEN_ENV, None)
    os.environ[TOKEN_ENV] = TOKEN
    try:
        assert endpoint_token(TOKEN_ENV) == TOKEN
    finally:
        os.environ.pop(TOKEN_ENV, None)


def test_client_renews_its_lease_and_treats_a_lost_lease_as_terminal():
    from cascade.robotics.endpoint import RemoteRobotRuntime
    with _served(ttl=1.0) as rig:
        remote = RemoteRobotRuntime(rig.origin, TOKEN)
        try:
            remote.connect()
            time.sleep(2.5)          # > 2 TTLs: only the client's renewals keep the lease
            assert not rig.runtime.stopped and _state(rig)["lease_held"] is True
            assert _state(rig)["lease_expiry_stops"] == 0 and remote.ready()["ready"] is True
            # The robot side drops the lease (operator, restart): the client must stop acting.
            assert _http(rig.origin, "DELETE", "/v1/lease", {"lease_id": remote.lease_id})[0] == 200
            assert _wait(lambda: remote.ready() == {"ready": False, "reason": "lease lost"})
            generation = remote.cancellation_token
            lost = {"ok": False, "error": "robot endpoint lease lost; restart this supervisor"}
            assert remote.execute("body.read", {}, expected_generation=generation,
                                  deadline_monotonic_s=time.monotonic() + 5) == lost    # refused locally
            assert remote.reset_stop(expected_generation=generation) == lost
            assert rig.trace.rows == [] and rig.body.resets == 0
            assert remote.stop()["ok"] is True and rig.runtime.stopped   # stop never needs the lease
        finally:
            receipt = remote.close()
        assert receipt["lease_released"] is False


def test_conversation_domain_drives_the_remote_runtime_and_records_its_stop_tool_remotely():
    from cascade.conversation.domain import ConversationDomain, ToolIntent
    from cascade.robotics.endpoint import RemoteRobotRuntime
    with _served() as rig:
        remote = RemoteRobotRuntime(rig.origin, TOKEN)
        try:
            remote.connect()
            domain = ConversationDomain(remote, robot_id="fixture", allow_tools=("body.read", "emergency_stop"))

            async def scenario():
                domain.claim("s1")
                generation, deadline = remote.cancellation_token, time.monotonic() + 20
                read = await domain.dispatch(ToolIntent("s1", "fixture", "r1", "c1", generation, deadline,
                                                        "body.read", {}))
                stop = await domain.dispatch(ToolIntent("s1", "fixture", "r1", "c2", generation, deadline,
                                                        "emergency_stop", {}))
                return read, stop, await domain.close()

            read, stop, closure = asyncio.run(scenario())
            assert read["ok"] is True and stop["ok"] is True and stop["latched"] is True
            assert closure["ok"] is True and rig.runtime.stopped
            assert [row[0] for row in rig.trace.rows] == ["body.read", "emergency_stop"]
        finally:
            receipt = remote.close()
        assert receipt == {"ok": True, "complete": True, "lease_released": True, "robot_runtime_closed": False}
        assert _state(rig)["lease_held"] is False


def test_lease_time_is_checked_at_every_request_not_only_by_the_watchdog(monkeypatch):
    """An expired lease grants nothing even before the expiry watchdog has run."""
    from cascade.robotics import endpoint as module
    with _served(ttl=30.0) as rig:
        lease = _lease(rig)
        generation = _state(rig)["generation"]
        real = time.monotonic
        monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: real() + 31.0))
        assert _http(rig.origin, "POST", "/v1/execute", {
            "lease_id": lease, "tool": "body.read", "arguments": {}, "expected_generation": generation,
            "deadline_remaining_s": 5})[0] == 410
        assert _http(rig.origin, "POST", "/v1/reset", {"lease_id": lease, "expected_generation": generation})[0] == 410
        assert _state(rig)["lease_held"] is False
        assert _http(rig.origin, "POST", "/v1/lease", {"lease_id": None})[0] == 200   # no 409 for a lapsed lease
        assert rig.trace.rows == [] and rig.body.calls == []


def test_endpoint_rejects_malformed_lease_requests_and_bodies():
    with _served() as rig:
        lease = _lease(rig)
        generation = _state(rig)["generation"]
        for body in ({"lease_id": None, "x": 1}, {"lease_id": 5}, {}, 5, "lease_id", []):
            assert _http(rig.origin, "POST", "/v1/lease", body)[0] == 400, body
        assert _http(rig.origin, "DELETE", "/v1/lease", {"lease_id": lease, "x": 1})[0] == 400
        for bad in (5, "é" * 32, None):
            assert _http(rig.origin, "POST", "/v1/execute", {
                "lease_id": bad, "tool": "body.read", "arguments": {}, "expected_generation": generation,
                "deadline_remaining_s": 5})[0] == 410, bad
        url = urlsplit(rig.origin)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
        try:
            connection.request("POST", "/v1/stop", body=b"{", headers={
                "Authorization": "Bearer " + TOKEN, "X-Cascade-Protocol": PROTOCOL, "Content-Length": "1"})
            response = connection.getresponse()
            assert response.status == 400 and json.loads(response.read())["protocol"] == PROTOCOL
        finally:
            connection.close()
        assert rig.trace.rows == [] and rig.body.stops == 0 and not rig.runtime.stopped
        with pytest.raises(ValueError):
            rig.endpoint.start(port=0)          # one listener per endpoint
    with pytest.raises(ValueError):
        rig.endpoint.start(port=0)              # never restarted after close


def test_only_one_supervisor_connects_and_a_wrong_token_never_connects():
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    with _served() as rig:
        stranger = RemoteRobotRuntime(rig.origin, "w" * 43)
        with pytest.raises(RobotEndpointError, match="403"):
            stranger.connect()
        refused_stop = stranger.stop()
        assert refused_stop["ok"] is False and not rig.runtime.stopped and rig.body.stops == 0
        first = RemoteRobotRuntime(rig.origin, TOKEN).connect()
        second = RemoteRobotRuntime(rig.origin, TOKEN)
        try:
            with pytest.raises(RobotEndpointError, match="409"):
                second.connect()
            assert second.lease_id is None and second.ready()["ready"] is False
            assert first.ready()["ready"] is True
        finally:
            second.close()
            assert first.close()["lease_released"] is True
        stranger.close()


def test_client_notices_a_revoked_lease_on_its_next_request_and_clamps_long_deadlines():
    from cascade.robotics.endpoint import RemoteRobotRuntime, RobotEndpointError
    with _served(ttl=30.0) as rig:
        remote = RemoteRobotRuntime(rig.origin, TOKEN).connect()
        try:
            generation = remote.cancellation_token
            far = remote.execute("body.read", {}, expected_generation=generation,
                                 deadline_monotonic_s=time.monotonic() + 1000)
            assert far["ok"] is True                      # sent as the protocol maximum, not refused
            malformed = remote.execute("body.read", [], expected_generation=generation,
                                       deadline_monotonic_s=time.monotonic() + 5)
            assert malformed["ok"] is False and malformed["delivery_uncertain"] is False
            assert "400" in malformed["error"]
            for kwargs in ({"expected_generation": generation, "domain": "body"},
                           {"expected_generation": generation, "deadline_monotonic_s": time.monotonic() + 5},
                           {"expected_generation": None}):
                assert remote.reset_stop(**kwargs)["ok"] is False
            # Revoked behind the client's back (renewal is 10 s away): the next requests are refused.
            assert _http(rig.origin, "DELETE", "/v1/lease", {"lease_id": remote.lease_id})[0] == 200
            assert remote.ready()["ready"] is True        # not noticed yet
            assert "refused reset (410)" in remote.reset_stop(expected_generation=generation)["error"]
            with pytest.raises(RobotEndpointError, match="410"):
                remote._record_tool_result("emergency_stop", {}, {"ok": True}, started_monotonic_s=time.monotonic())
            refused = remote.execute("body.read", {}, expected_generation=generation,
                                     deadline_monotonic_s=time.monotonic() + 5)
            assert refused == {"ok": False, "error": "robot endpoint revoked the lease (410); restart this supervisor"}
            assert remote.ready() == {"ready": False, "reason": "lease lost"}
            assert [row[0] for row in rig.trace.rows] == ["body.read"]
        finally:
            remote.close()


def test_client_gives_up_a_lease_it_could_not_renew_for_a_whole_ttl():
    from cascade.robotics.endpoint import RemoteRobotRuntime
    with _served(ttl=.5) as rig:
        remote = RemoteRobotRuntime(rig.origin, TOKEN).connect()
        try:
            rig.endpoint.close()
            assert _wait(lambda: remote.ready()["reason"] == "lease lost")
        finally:
            remote.close()


def test_non_finite_numbers_never_reach_the_runtime():
    """Strict JSON at the boundary: a NaN velocity is refused even if a schema bound would not catch it."""
    with _served() as rig:
        lease = _lease(rig)
        generation = _state(rig)["generation"]
        status, body = _http(rig.origin, "POST", "/v1/execute", {
            "lease_id": lease, "tool": "body.walk", "arguments": {"vx": float("nan")},
            "expected_generation": generation, "deadline_remaining_s": 5})
        assert (status, body["error"]) == (400, "request body must be strict JSON")
        status, body = _http(rig.origin, "POST", "/v1/record", {
            "lease_id": lease, "tool": "emergency_stop", "arguments": {}, "result": {"ok": True, "x": float("inf")},
            "duration_ms": 1})
        assert (status, body["error"]) == (400, "request body must be strict JSON")
        assert rig.body.calls == [] and rig.trace.rows == []


def test_one_failed_renewal_inside_the_ttl_does_not_lose_the_lease(monkeypatch):
    from cascade.robotics.endpoint import RemoteRobotRuntime
    with _served(ttl=1.5) as rig:
        remote = RemoteRobotRuntime(rig.origin, TOKEN).connect()
        try:
            time.sleep(2.0)            # renewals succeeded for longer than one TTL
            original, calls = rig.endpoint._lease_route, []

            def flaky(body):
                calls.append(body)
                if len(calls) == 1:
                    raise RuntimeError("transient")      # one renewal answered with HTTP 500
                return original(body)

            monkeypatch.setattr(rig.endpoint, "_lease_route", flaky)
            assert _wait(lambda: len(calls) >= 3)          # the failure and two later renewals
            assert remote.ready() == {"ready": True, "reason": None}
            assert _state(rig)["lease_held"] is True and not rig.runtime.stopped
        finally:
            remote.close()


def test_catalog_digest_binds_identity_resources_and_tools():
    from cascade.robotics.endpoint import catalog_digest
    with _served() as rig:
        description = rig.endpoint.describe()
        canonical = {key: description[key] for key in ("robot_id", "resources", "tools")}
        expected = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert catalog_digest(description) == expected
        for key, value in (("robot_id", "other"), ("resources", []), ("tools", description["tools"][:-1])):
            assert catalog_digest({**description, key: value}) != expected
        assert catalog_digest({**description, "lease_ttl_s": 1.0, "trace_recorded": False}) == expected


def test_robot_side_imports_no_conversation_or_model_dependency():
    """cascade-robot-service deploys without the `conversation` extra."""
    import subprocess
    script = ("import sys\nimport cascade.robotics.endpoint, cascade.apps.robot_service\n"
              "assert all(name not in sys.modules for name in "
              "('aiohttp', 'openai', 'torch', 'websockets', 'cascade.conversation'))\n")
    subprocess.run([sys.executable, "-c", script], env=_env(), check=True, timeout=60)


def test_conversation_service_refuses_a_remote_robot_with_another_identity(tmp_path):
    """The configured robot id binds the remote catalog; a mismatch starts nothing."""
    from cascade.apps.conversation import parser, serve
    from cascade.conversation.service import configuration
    with _served() as rig:
        os.environ[TOKEN_ENV] = TOKEN
        try:
            args = configuration(parser().parse_args([
                "--provider-url", "ws://127.0.0.1:46499/v1/realtime", "--run-dir", str(tmp_path / "run"),
                "--robot", "someone-else", "--robot-endpoint", rig.origin, "--robot-token-env", TOKEN_ENV,
                "--port", "0"]))
            # Bounded: a service that wrongly started would otherwise wait for a signal.
            assert asyncio.run(asyncio.wait_for(serve(SimpleNamespace(**args)), 60)) == 1
        finally:
            os.environ.pop(TOKEN_ENV, None)
        closure = json.loads((tmp_path / "run" / "closure.json").read_text())
        assert closure["service_error"] == {"type": "ValueError"} and not (tmp_path / "run" / "ready.json").exists()
        assert closure["runtime"]["lease_released"] is True and _state(rig)["lease_held"] is False
        assert rig.trace.rows == [] and not rig.runtime.stopped


# --------------------------------------------------------------------- configuration


def test_split_configuration_is_validated_and_the_shipped_example_parses(tmp_path):
    from cascade.apps.conversation import parser
    from cascade.conversation.service import configuration
    example = REPO / "configs" / "conversation" / "remote_robot.json"
    values = configuration(parser().parse_args(["--config", str(example)]))
    assert values["robot_endpoint"] == "http://127.0.0.1:8781"
    assert values["robot_token_env"] == "CASCADE_ROBOT_ENDPOINT_TOKEN" and values["robot"] == "conversation_mock"
    cli = configuration(parser().parse_args([
        "--provider-url", "ws://127.0.0.1:8765/v1/realtime", "--run-dir", str(tmp_path / "run"),
        "--robot-endpoint", "http://localhost:46401/", "--robot-token-env", "ROBOT_TOKEN"]))
    assert (cli["robot_endpoint"], cli["robot_token_env"]) == ("http://localhost:46401", "ROBOT_TOKEN")
    base = {"version": 1, "provider_url": "ws://127.0.0.1:8765/v1/realtime", "run_root": "runs"}
    remote = {"robot_endpoint": "http://127.0.0.1:46401", "robot_token_env": "ROBOT_TOKEN"}
    for change in ({"robot_endpoint": "http://10.0.0.5:46401"}, {"robot_endpoint": "https://127.0.0.1:46401"},
                   {"robot_endpoint": 46401}, {"robot_token_env": None}, {"robot_token_env": "no such"},
                   {"robot_token_env": 5}, {"robot_lifecycle": "bounded_hand"}, {"config_dir": "profiles"},
                   {"robot_endpoint": None}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({**base, **remote, **change}))
        with pytest.raises(ValueError):
            configuration(parser().parse_args(["--config", str(path)]))


def test_robot_service_cli_refuses_missing_or_weak_credentials(tmp_path, monkeypatch, capsys):
    from cascade.apps import robot_service
    for value in (None, "x" * 31):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        if value is not None:
            monkeypatch.setenv(TOKEN_ENV, value)
        monkeypatch.setattr(sys, "argv", ["cascade-robot-service", "--robot", "conversation_mock",
                                          "--port", str(PORT_BLOCK[-1]), "--run-dir", str(tmp_path / "run"),
                                          "--token-env", TOKEN_ENV])
        with pytest.raises(SystemExit) as exit_info:
            robot_service.main()
        assert exit_info.value.code == 2 and "token" in capsys.readouterr().err
        assert not (tmp_path / "run").exists()
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    for flags in (["--lease-ttl-s", "0.2"], ["--lease-ttl-s", "nan"], ["--lease-ttl-s", "31"], ["--port", "70000"]):
        argv = ["cascade-robot-service", "--robot", "conversation_mock", "--port", str(PORT_BLOCK[-1]),
                "--run-dir", str(tmp_path / "run"), "--token-env", TOKEN_ENV, *flags]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exit_info:
            robot_service.main()
        assert exit_info.value.code == 2, flags
        assert not (tmp_path / "run").exists()


# ------------------------------------------------- real processes: robot + conversation


def _env(token=None):
    env = {key: value for key, value in os.environ.items()
           if not (key.startswith("CASCADE_") and key.endswith("_PORT"))}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1")
    env.pop(TOKEN_ENV, None)
    if token is not None:
        env[TOKEN_ENV] = token
    return env


class _Process:
    def __init__(self, process, run, origin):
        self.process, self.run, self.origin = process, run, origin
        self.returncode = None

    def ready_record(self):
        return json.loads((self.run / "ready.json").read_text())

    def closure(self):
        return json.loads((self.run / "closure.json").read_text())

    async def terminate(self):
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
                raise
        self.returncode = self.process.returncode


async def _read_prefixed(process, prefix):
    for _ in range(64):
        line = await asyncio.wait_for(process.stdout.readline(), 90)
        if not line:
            return None
        if line.startswith(prefix):
            return line.decode().strip()
    return None


async def _robot_service(directory, token, *, lease_ttl_s):
    run = directory / "robot"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "cascade.apps.robot_service", "--robot", "microduck_conversation_mock",
        "--port", "0", "--run-dir", str(run), "--token-env", TOKEN_ENV,
        "--lease-ttl-s", str(lease_ttl_s), cwd=REPO, env=_env(token),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    line = await _read_prefixed(process, b"Robot runtime endpoint ")
    if line is None:
        await process.wait()
        raise AssertionError((await process.stderr.read()).decode())
    service = _Process(process, run, line.split()[3])
    # The port the OS gave THIS child: its stdout line and its own ready record agree.
    ready = service.ready_record()
    assert _loopback_port(service.origin) and (ready["origin"], ready["pid"]) == (service.origin, process.pid)
    service.state = lambda: _http(service.origin, "GET", "/v1/state", token=token)[1]
    service.trace_rows = lambda: [json.loads(row) for row in
                                  (run / "trace.jsonl").read_text().splitlines() if row.strip()]
    return service


async def _conversation_service(directory, values, *, token=None):
    directory.mkdir(parents=True, exist_ok=True)
    run, path = directory / "conversation", directory / "service.json"
    path.write_text(json.dumps({"version": 1, "provider_url": "ws://127.0.0.1:46499/v1/realtime",
                                "port": 0, "run_dir": run.name, "intent_timeout_s": 60,
                                "execution_timeout_s": 30, **values}))
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "cascade.apps.conversation", "--config", str(path), cwd=directory,
        env=_env(token), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    line = await _read_prefixed(process, b"Open ")
    if line is None:
        await process.wait()
        closure = json.loads((run / "closure.json").read_text()) if (run / "closure.json").exists() else {}
        raise AssertionError((await process.stderr.read()).decode() or closure)
    url, credential = line[5:].split("/#")
    service = _Process(process, run, url)
    service.url, service.token = url, credential
    ready = service.ready_record()
    assert _loopback_port(url) and (ready["origin"], ready["pid"]) == (url, process.pid)
    return service


class _Provider:
    """Loopback Realtime stub standing in for ASR + LLM + TTS (no inference)."""

    def __init__(self):
        self.events, self.ws, self.connections = None, None, 0

    async def start(self):
        from aiohttp import web
        self.events = asyncio.Queue()
        app = web.Application()
        app.router.add_get("/v1/realtime", self._handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.url = f"ws://127.0.0.1:{self.runner.addresses[0][1]}/v1/realtime"
        _loopback_port(self.url)
        return self

    async def _handle(self, request):
        import aiohttp
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws = ws
        self.connections += 1
        await ws.send_json({"type": "session.created", "session": {"id": f"fixture-{self.connections}"}})
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                break
            event = json.loads(message.data)
            if event["type"] == "session.update":
                await ws.send_json({"type": "session.updated", "session": event["session"]})
            else:
                await self.events.put(event)
        return ws

    async def emit(self, event):
        await self.ws.send_json(event)

    async def next(self, kind, timeout=60):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            event = await asyncio.wait_for(self.events.get(), max(.01, deadline - loop.time()))
            if event["type"] == kind:
                return event

    async def close(self):
        await self.runner.cleanup()


def _pcm():
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 24000))) for i in range(960))


async def _ws_next(ws, kind, timeout=60):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        event = await ws.receive_json(timeout=max(.01, deadline - loop.time()))
        if event["type"] == kind:
            return event


async def _until(predicate, timeout=60):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(.05)
    raise AssertionError("condition not reached before the test deadline")


async def _tool_turn(provider, ws, session_id, text, alias, args, *, rid, cid):
    """Typed text -> provider decides one tool call (the mock LLM) -> completed response."""
    await ws.send_json({"type": "text", "session_id": session_id, "text": text})
    item = await provider.next("conversation.item.create")
    assert item["item"]["content"] == [{"type": "input_text", "text": text}]
    request = await provider.next("response.create")
    await provider.emit({"type": "response.created", "response": {
        "id": rid, "status": "in_progress", "metadata": request["response"]["metadata"]}})
    await provider.emit({"type": "response.function_call_arguments.done", "response_id": rid, "call_id": cid,
                         "name": alias, "output_index": 0, "arguments": json.dumps(args)})
    await provider.emit({"type": "response.done", "response": {
        "id": rid, "status": "completed", "output": [{"type": "function_call", "call_id": cid}]}})


async def _tool_output(provider, cid):
    output = await provider.next("conversation.item.create")
    assert output["item"]["type"] == "function_call_output" and output["item"]["call_id"] == cid
    return json.loads(output["item"]["output"])


async def _spoken_reply(provider, ws, rid, words):
    """The model's continuation (mock TTS PCM + transcript) reaches the browser."""
    continuation = await provider.next("response.create")
    pcm = _pcm()
    await provider.emit({"type": "response.created", "response": {
        "id": rid, "status": "in_progress", "metadata": continuation["response"]["metadata"]}})
    await provider.emit({"type": "response.output_audio.delta", "response_id": rid,
                         "delta": base64.b64encode(pcm).decode()})
    await provider.emit({"type": "response.output_audio_transcript.done", "response_id": rid, "transcript": words})
    await provider.emit({"type": "response.done", "response": {"id": rid, "status": "completed", "output": []}})
    audio = await _ws_next(ws, "audio")
    assert base64.b64decode(audio["audio"]) == pcm and audio["response_id"] == rid
    assert (await _ws_next(ws, "transcript"))["text"] == words


def _split_values(robot, provider, **values):
    return {"robot": "microduck-mock", "provider_url": provider.url, "robot_endpoint": robot.origin,
            "robot_token_env": TOKEN_ENV, "allow_motion": True,
            "allow_tools": ["locomotion.get_base_state", "locomotion.walk_velocity"], **values}


def test_split_deployment_completes_a_text_turn_and_keeps_stop_and_reconnect(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    token = secrets.token_urlsafe(32)

    async def scenario():
        provider = await _Provider().start()
        robot = await _robot_service(tmp_path / "robot", token, lease_ttl_s=5.0)
        conversation = None
        try:
            assert _http(robot.origin, "GET", "/v1/health", token=None, protocol=None)[1]["alive"] is True
            robot_ready = robot.ready_record()
            assert robot_ready["protocol"] == PROTOCOL and robot_ready["robot_id"] == "microduck-mock"
            assert robot_ready["origin"] == robot.origin and robot_ready["physical_admission"] is False
            assert token not in (robot.run / "ready.json").read_text()
            conversation = await _conversation_service(
                tmp_path / "conversation", _split_values(robot, provider, start_stopped=True), token=token)
            ready = conversation.ready_record()
            assert ready["robot_id"] == "microduck-mock" and ready["robot_config_sha256"] is None
            assert ready["robot_endpoint"] == {"origin": robot.origin, "protocol": PROTOCOL,
                                               "catalog_sha256": robot_ready["catalog_sha256"], "lease_ttl_s": 5.0}
            assert ready["tools"] == ["locomotion.get_base_state", "locomotion.walk_velocity"]
            assert token not in (conversation.run / "ready.json").read_text()
            assert not (conversation.run / "domains").exists()        # no robot was built here
            async with aiohttp.ClientSession() as anonymous:
                assert await (await anonymous.get(conversation.url + "/healthz")).json() == {"alive": True}
                readiness = await anonymous.get(conversation.url + "/readyz")
                assert readiness.status == 200 and (await readiness.json())["ready"] is True
                assert (await anonymous.get(conversation.url + "/api/status")).status == 403
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + conversation.token}) as client:
                status = await (await client.get(conversation.url + "/api/status")).json()
                assert status["stopped"] is True and robot.state()["stopped"] is True   # start_stopped crossed over
                assert status["generation"] == robot.state()["generation"]
                reset = await client.post(conversation.url + "/api/reset", json={"generation": status["generation"]})
                assert (await reset.json())["ok"] is True and robot.state()["stopped"] is False
                binding = await (await client.post(conversation.url + "/api/session",
                                                   json={"robot_id": "microduck-mock"})).json()
                sid = binding["session_id"]
                async with client.ws_connect(conversation.url + "/api/media?ticket=" + binding["ticket"]) as ws:
                    # One complete text turn: text -> tool in the robot process -> result -> spoken reply.
                    await _tool_turn(provider, ws, sid, "walk forward a little", "robot_tool_1",
                                     {"vx": .05, "vy": 0, "wz": 0, "duration_s": .2}, rid="r1", cid="c1")
                    walked = await _tool_output(provider, "c1")
                    assert walked["execution_ok"] is True and walked["outcome"] == "unverified"
                    assert walked["ok"] is False          # mock kinematics never become physical evidence
                    shown = await _ws_next(ws, "tool_result")
                    assert shown["tool"] == "locomotion.walk_velocity" and shown["result"]["outcome"] == "unverified"
                    await _spoken_reply(provider, ws, "r2", "I walked; the robot reports it unverified.")
                    walks = [r for r in robot.trace_rows() if r["skill"] == "locomotion.walk_velocity"]
                    assert len(walks) == 1 and walks[0]["result"]["outcome"] == "unverified"
                    # Operator stop wins over a motion running in the other process.
                    await _tool_turn(provider, ws, sid, "keep walking", "robot_tool_1",
                                     {"vx": .05, "vy": 0, "wz": 0, "duration_s": 3.0}, rid="r3", cid="c3")
                    await _until(lambda: robot.state()["executing"] == 1)
                    stop = await (await client.post(conversation.url + "/api/stop")).json()
                    assert stop["ok"] is True and stop["physical_stop_verified"] is False
                    assert robot.state()["stopped"] is True
                    await _until(lambda: robot.state()["executing"] == 0)
                    walks = [r for r in robot.trace_rows() if r["skill"] == "locomotion.walk_velocity"]
                    assert len(walks) == 2 and walks[1]["result"]["error"] == "operation superseded by stop"
                    assert (await _ws_next(ws, "authority_revoked"))["reconnect_required"] is True
                # Reconnect: disconnect, explicit reset with the observed generation, new session.
                closed = await (await client.delete(conversation.url + "/api/session")).json()
                assert closed["ok"] is True
                status = await (await client.get(conversation.url + "/api/status")).json()
                assert status["stopped"] is True and status["generation"] == robot.state()["generation"]
                stale = await client.post(conversation.url + "/api/reset", json={"generation": status["generation"] - 1})
                assert (await stale.json())["ok"] is False and robot.state()["stopped"] is True
                reset = await client.post(conversation.url + "/api/reset", json={"generation": status["generation"]})
                assert (await reset.json())["ok"] is True and robot.state()["stopped"] is False
                binding = await (await client.post(conversation.url + "/api/session",
                                                   json={"robot_id": "microduck-mock"})).json()
                async with client.ws_connect(conversation.url + "/api/media?ticket=" + binding["ticket"]) as ws:
                    await _tool_turn(provider, ws, binding["session_id"], "where are you", "robot_tool_0", {},
                                     rid="r4", cid="c4")
                    state = await _tool_output(provider, "c4")
                    assert state["ok"] is True and state["state"]["measurement_kind"] == "kinematic_mock"
                    assert state["state"]["position_world"][0] > 0      # the robot process kept its state
                    assert state["state"]["epoch"] == walks[0]["result"]["measured"]["before"]["epoch"]
        finally:
            if conversation is not None:
                await conversation.terminate()
            await robot.terminate()
            await provider.close()
        assert conversation.returncode == 0 and conversation.closure()["ok"] is True
        assert robot.returncode == 0 and robot.closure()["ok"] is True
        assert robot.closure()["lease_expiry_stops"] == [] and robot.closure()["signals"] == ["SIGTERM"]
    asyncio.run(scenario())


def test_killed_conversation_service_stops_the_robot_and_a_new_one_reattaches(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    token = secrets.token_urlsafe(32)

    async def scenario():
        provider = await _Provider().start()
        robot = await _robot_service(tmp_path / "robot", token, lease_ttl_s=1.0)
        first = second = None
        try:
            first = await _conversation_service(tmp_path / "first", _split_values(robot, provider), token=token)
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + first.token}) as client:
                binding = await (await client.post(first.url + "/api/session",
                                                   json={"robot_id": "microduck-mock"})).json()
                async with client.ws_connect(first.url + "/api/media?ticket=" + binding["ticket"]) as ws:
                    await _tool_turn(provider, ws, binding["session_id"], "walk", "robot_tool_1",
                                     {"vx": .05, "vy": 0, "wz": 0, "duration_s": 3.0}, rid="r1", cid="c1")
                    await _until(lambda: robot.state()["executing"] == 1)
                    assert robot.state()["stopped"] is False
                    first.process.send_signal(signal.SIGKILL)     # no cleanup, no stop request
                    await first.process.wait()
            # Dead-man: the robot stops itself once the supervisor's lease lapses.
            state = await _until(lambda: (s := robot.state())["lease_expiry_stops"] == 1 and s)
            assert state["lease_held"] is False and state["stopped"] is True
            await _until(lambda: robot.state()["executing"] == 0)
            walk = [r for r in robot.trace_rows() if r["skill"] == "locomotion.walk_velocity"][-1]
            assert walk["result"]["error"] == "operation superseded by stop"
            assert not (first.run / "closure.json").exists()
            # A new conversation process attaches to the SAME robot process; the latch survives.
            second = await _conversation_service(tmp_path / "second", _split_values(robot, provider), token=token)
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + second.token}) as client:
                status = await (await client.get(second.url + "/api/status")).json()
                assert status["stopped"] is True
                reset = await client.post(second.url + "/api/reset", json={"generation": status["generation"]})
                assert (await reset.json())["ok"] is True
                binding = await (await client.post(second.url + "/api/session",
                                                   json={"robot_id": "microduck-mock"})).json()
                async with client.ws_connect(second.url + "/api/media?ticket=" + binding["ticket"]) as ws:
                    await _tool_turn(provider, ws, binding["session_id"], "where are you", "robot_tool_0", {},
                                     rid="r2", cid="c2")
                    observed = await _tool_output(provider, "c2")
                    assert observed["state"]["epoch"] == walk["result"]["domain_result"]["measured"]["before"]["epoch"]
                closed = await (await client.delete(second.url + "/api/session")).json()
                assert closed["ok"] is True
            # Readiness follows the robot: the conversation stays alive but is no longer ready.
            await robot.terminate()
            assert robot.returncode == 0 and robot.closure()["ok"] is True
            assert [entry["ok"] for entry in robot.closure()["lease_expiry_stops"]] == [True]
            async with aiohttp.ClientSession() as anonymous:
                assert (await anonymous.get(second.url + "/healthz")).status == 200
                readiness = await anonymous.get(second.url + "/readyz")
                assert readiness.status == 503 and (await readiness.json())["ready"] is False
            async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + second.token}) as client:
                # No fabricated state: without the robot there is no generation to report.
                assert (await client.get(second.url + "/api/status")).status == 500
        finally:
            for service in (second, first):
                if service is not None:
                    await service.terminate()
            await robot.terminate()
            await provider.close()
        # Shutdown could not confirm a robot stop: visible failure, never a clean exit.
        assert second.returncode == 1 and second.closure()["ok"] is False
    asyncio.run(scenario())
