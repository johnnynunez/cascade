"""`cascade.robot-runtime/1`: one robot runtime served to a separately deployed supervisor.

The conversation service (``cascade-conversation``) normally builds its robot
runtime in its own process. This module lets the two run as separate processes
on one host: ``RobotRuntimeEndpoint`` serves an existing ``RobotRuntime`` over
loopback HTTP/JSON (stdlib only, no extra dependency), and ``RemoteRobotRuntime``
is the client with the surface ``ConversationDomain`` and its gateway use.

Safety properties carried across the process boundary (B51):

* The robot process keeps every authority it had: the runtime's generation and
  deadline fences, each domain's harness/verifier and its own trace decide.
  The endpoint adds checks; it never relaxes or replaces one.
* Stop needs only the bearer token -- no lease, no generation -- and is served
  on its own request thread while a motion request is still in flight.
* One supervisor holds a renewable lease. Execute, reset and stop-tool
  recording need it. If the lease lapses (supervisor killed, hung or
  partitioned) the endpoint latches ``runtime.stop()``: a dead-man, not a
  physical rest proof. Releasing a lease is not a stop and not a reset.
* Every remote execute carries the episode generation and the remaining local
  deadline; the endpoint re-anchors that budget on its own monotonic clock at
  receipt (late by the one-way loopback latency). Reset requires the lease and
  the exact observed generation; execute can never clear a stop.
* The client reports unknown state as stopped, a lost transport as
  ``delivery_uncertain`` and a lost lease as terminal for that supervisor.

Loopback only, like the conversation gateway: a cross-host deployment needs an
authenticated encrypted transport this protocol does not provide.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .contracts import ResourceDescriptor, ToolDescriptor

PROTOCOL = "cascade.robot-runtime/1"
PROTOCOL_HEADER = "X-Cascade-Protocol"
MAX_REQUEST_BYTES = 1 << 20
MAX_RESPONSE_BYTES = 16 << 20
MAX_DEADLINE_S = 300.0
_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}


class RobotEndpointError(RuntimeError):
    """The robot endpoint was unreachable, refused us, or spoke another protocol."""


def _check_token(token):
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("robot endpoint token must contain at least 32 characters")
    return token


def endpoint_origin(url):
    """Normalize ``http://<loopback>:<port>``; anything else is refused."""
    if not isinstance(url, str):
        raise ValueError("robot endpoint must be an http://<loopback>:<port> URL")
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("robot endpoint port is invalid") from exc
    if (parts.scheme != "http" or parts.hostname not in _LOOPBACK_NAMES or not port
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in {"", "/"}):
        raise ValueError("robot endpoint must be an http://<loopback>:<port> URL without path or credentials")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"http://{host}:{port}"


def endpoint_token(env_name):
    """Read the shared bearer token from an operator-owned environment variable."""
    if not isinstance(env_name, str) or not env_name.isidentifier():
        raise ValueError("robot endpoint token variable must be an identifier")
    value = os.environ.get(env_name)
    if not value or len(value) < 32:
        raise ValueError(f"robot endpoint token variable {env_name} is unset or shorter than 32 characters")
    return value


def catalog_digest(description):
    """SHA256 of the robot identity, resources and tools a supervisor is bound to."""
    canonical = {key: description[key] for key in ("robot_id", "resources", "tools")}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _strict_json(raw):
    return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _generation(value):
    return type(value) is int and value >= 0


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


class _Refused(Exception):
    def __init__(self, status, error):
        super().__init__(error)
        self.status, self.error = status, error


class RobotRuntimeEndpoint:
    """Serve one RobotRuntime; never closes it (its process owner does)."""

    def __init__(self, runtime, *, robot_id, token, lease_ttl_s=3.0):
        self.token = _check_token(token)
        if not isinstance(robot_id, str) or not 0 < len(robot_id) <= 128:
            raise ValueError("robot_id is required")
        if any(resource.robot_id != robot_id for resource in runtime.resources):
            raise ValueError("endpoint robot identity differs from the resource catalog")
        if not _finite_number(lease_ttl_s) or not .2 < lease_ttl_s <= 30:
            raise ValueError("lease_ttl_s must be finite and in (0.2, 30]")
        self.runtime, self.robot_id, self.lease_ttl_s = runtime, robot_id, float(lease_ttl_s)
        self._lock = threading.Condition(threading.Lock())
        self._lease = None            # (lease_id, expires_monotonic_s)
        self._executing = 0
        self._closing = False
        self.lease_expiry_stops = []
        self._server = self._thread = self._watchdog = None
        self.origin = None
        self._description = {
            "robot_id": robot_id,
            "resources": [resource.as_dict() for resource in runtime.resources],
            "tools": [descriptor.as_dict() for descriptor in runtime.tool_descriptors.values()],
        }

    # ------------------------------------------------------------------ lifecycle

    def start(self, *, host="127.0.0.1", port=0):
        if self._server is not None or self._closing:
            raise ValueError("endpoint already started or closed")
        if not ipaddress.ip_address(host).is_loopback or type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("robot endpoint binds an explicit loopback IP only")
        server = ThreadingHTTPServer((host, port), self._handler())
        server.daemon_threads = True
        self._server = server
        rendered = f"[{host}]" if ":" in host else host
        self.origin = f"http://{rendered}:{server.server_address[1]}"
        self._thread = threading.Thread(target=server.serve_forever, name="robot-endpoint", daemon=True)
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watch_lease, name="robot-endpoint-lease", daemon=True)
        self._watchdog.start()
        return self.origin

    def close(self):
        with self._lock:
            self._closing = True
            self._lock.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._watchdog.join()
            self._server = None
        with self._lock:
            return {"ok": True, "complete": True, "lease_expiry_stops": copy.deepcopy(self.lease_expiry_stops)}

    def _watch_lease(self):
        """Dead-man: an expired lease latches the robot stop, exactly once."""
        while True:
            with self._lock:
                while not self._closing and (self._lease is None or time.monotonic() < self._lease[1]):
                    timeout = None if self._lease is None else max(0., self._lease[1] - time.monotonic())
                    self._lock.wait(timeout)
                if self._closing:
                    return
                self._lease = None
            receipt = self.runtime.stop()
            with self._lock:
                self.lease_expiry_stops.append({"ok": receipt.get("ok") is True,
                                                "generation": receipt.get("generation")})

    # ------------------------------------------------------------------ routes

    def _live_lease(self, lease_id):
        """Called with the lock held."""
        lease = self._lease
        if (lease is None or not isinstance(lease_id, str) or not lease_id.isascii()
                or not hmac.compare_digest(lease_id, lease[0]) or time.monotonic() >= lease[1]):
            raise _Refused(410, "lease expired or unknown")

    def _lease_route(self, body):
        if set(body) != {"lease_id"} or not (body["lease_id"] is None or isinstance(body["lease_id"], str)):
            raise _Refused(400, "lease requires an exact lease_id (null to acquire)")
        with self._lock:
            now = time.monotonic()
            if body["lease_id"] is None:
                if self._lease is not None and now < self._lease[1]:
                    raise _Refused(409, "lease held by another supervisor")
                lease_id = secrets.token_hex(16)
            else:
                self._live_lease(body["lease_id"])
                lease_id = self._lease[0]
            self._lease = (lease_id, now + self.lease_ttl_s)
            self._lock.notify_all()
        return {"lease_id": lease_id, "ttl_s": self.lease_ttl_s}

    def _release_route(self, body):
        if set(body) != {"lease_id"}:
            raise _Refused(400, "release requires an exact lease_id")
        with self._lock:
            self._live_lease(body["lease_id"])
            self._lease = None
            self._lock.notify_all()
        return {"released": True}

    def _execute_route(self, body):
        if (set(body) != {"lease_id", "tool", "arguments", "expected_generation", "deadline_remaining_s"}
                or not isinstance(body["tool"], str) or not isinstance(body["arguments"], dict)
                or not _generation(body["expected_generation"])
                or not _finite_number(body["deadline_remaining_s"])
                or not 0 < body["deadline_remaining_s"] <= MAX_DEADLINE_S):
            raise _Refused(400, "execute requires lease, tool, object arguments, generation and deadline")
        with self._lock:
            self._live_lease(body["lease_id"])
            self._executing += 1
        try:
            # Re-anchor the caller's remaining budget on this process's clock.
            deadline = time.monotonic() + body["deadline_remaining_s"]
            result = self.runtime.execute(body["tool"], body["arguments"],
                                          expected_generation=body["expected_generation"],
                                          deadline_monotonic_s=deadline)
        finally:
            with self._lock:
                self._executing -= 1
        return {"result": result}

    def _stop_route(self, body):
        if body != {}:
            raise _Refused(400, "stop takes an empty object")
        return {"result": self.runtime.stop()}

    def _reset_route(self, body):
        if set(body) != {"lease_id", "expected_generation"} or not _generation(body["expected_generation"]):
            raise _Refused(400, "reset requires the lease and the exact observed generation")
        with self._lock:
            self._live_lease(body["lease_id"])
        return {"result": self.runtime.reset_stop(expected_generation=body["expected_generation"])}

    def _record_route(self, body):
        if (set(body) != {"lease_id", "tool", "arguments", "result", "duration_ms"}
                or not isinstance(body["arguments"], dict) or not isinstance(body["result"], dict)
                or not _finite_number(body["duration_ms"]) or body["duration_ms"] < 0):
            raise _Refused(400, "record requires lease, tool, arguments, result and duration")
        descriptor = self.runtime.tool_descriptors.get(body["tool"])
        if descriptor is None or descriptor.effect != "stop":
            raise _Refused(400, "only an already delivered stop tool is recorded remotely")
        with self._lock:
            self._live_lease(body["lease_id"])
        self.runtime._record_tool_result(body["tool"], body["arguments"], body["result"],
                                         started_monotonic_s=time.monotonic() - body["duration_ms"] / 1000)
        return {"recorded": True}

    def _state(self):
        with self._lock:
            held = self._lease is not None and time.monotonic() < self._lease[1]
            executing, expiries = self._executing, len(self.lease_expiry_stops)
        return {"generation": self.runtime.cancellation_token, "stopped": self.runtime.stopped,
                "lease_held": held, "lease_expiry_stops": expiries, "executing": executing}

    def describe(self):
        return {**copy.deepcopy(self._description), "trace_recorded": self.runtime.trace is not None,
                "lease_ttl_s": self.lease_ttl_s}

    def _dispatch(self, method, path, headers, read_body):
        if (method, path) == ("GET", "/v1/health"):
            return 200, {"alive": True}
        if not hmac.compare_digest(headers.get("Authorization", ""), "Bearer " + self.token):
            raise _Refused(403, "authorization required")
        if headers.get(PROTOCOL_HEADER) != PROTOCOL:
            raise _Refused(400, "protocol mismatch")
        routes = {("GET", "/v1/describe"): None, ("GET", "/v1/state"): None,
                  ("POST", "/v1/lease"): self._lease_route, ("DELETE", "/v1/lease"): self._release_route,
                  ("POST", "/v1/execute"): self._execute_route, ("POST", "/v1/stop"): self._stop_route,
                  ("POST", "/v1/reset"): self._reset_route, ("POST", "/v1/record"): self._record_route}
        if (method, path) not in routes:
            raise _Refused(404, "unknown route")
        if path == "/v1/describe":
            return 200, self.describe()
        if path == "/v1/state":
            return 200, self._state()
        body = read_body()
        if not isinstance(body, dict):
            raise _Refused(400, "request body must be a JSON object")
        return 200, routes[(method, path)](body)

    def _handler(self):
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "cascade-robot-runtime"
            sys_version = ""

            def _serve(self):
                def read_body():
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError:
                        raise _Refused(400, "invalid Content-Length") from None
                    if not 0 <= length <= MAX_REQUEST_BYTES:
                        raise _Refused(413, "request too large")
                    try:
                        return _strict_json(self.rfile.read(length) or b"null")
                    except ValueError:
                        raise _Refused(400, "request body must be strict JSON") from None

                try:
                    status, payload = endpoint._dispatch(self.command, self.path, self.headers, read_body)
                except _Refused as refused:
                    status, payload = refused.status, {"error": refused.error}
                except Exception as exc:  # noqa: BLE001 -- never leak a traceback to the peer
                    status, payload = 500, {"error": type(exc).__name__}
                data = json.dumps({"protocol": PROTOCOL, **payload}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_DELETE = _serve

            def log_message(self, *_args):  # no access log: requests carry credentials
                pass

        return Handler


class _RemoteTrace:
    """Marker: the robot process records every remote execute in its own trace."""

    def __repr__(self):
        return "<remote robot trace>"


class RemoteRobotRuntime:
    """Client-side stand-in for RobotRuntime over ``cascade.robot-runtime/1``.

    It owns a supervision lease, never the robot: ``close()`` releases the lease
    and leaves the remote runtime running (stopped, if the supervisor stopped it).
    """
    robot_mode = "remote"

    def __init__(self, origin, token, *, io_timeout_s=2.0, execute_timeout_s=330.0):
        self.origin = endpoint_origin(origin)
        self._token = _check_token(token)
        self.io_timeout_s, self.execute_timeout_s = io_timeout_s, execute_timeout_s
        self.robot_id = None
        self.resources, self.tool_descriptors, self.trace = (), {}, None
        self.lease_id = self.lease_ttl_s = self.catalog_sha256 = None
        self._lock = threading.Lock()
        self._lost = False
        self._closed = threading.Event()
        self._renewer = None

    # ------------------------------------------------------------------ transport

    def _request(self, method, path, body=None, *, timeout=None):
        parts = urlsplit(self.origin)
        connection = http.client.HTTPConnection(parts.hostname, parts.port,
                                                timeout=self.io_timeout_s if timeout is None else timeout)
        headers = {"Authorization": "Bearer " + self._token, PROTOCOL_HEADER: PROTOCOL}
        data = None
        if body is not None:
            data = json.dumps(body, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise RobotEndpointError(f"robot endpoint transport failed: {type(exc).__name__}") from None
        finally:
            connection.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RobotEndpointError("robot endpoint response too large")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise RobotEndpointError("robot endpoint answered without JSON") from None
        if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
            raise RobotEndpointError("robot endpoint protocol mismatch")
        return response.status, payload

    def connect(self):
        status, description = self._request("GET", "/v1/describe")
        if status != 200:
            raise RobotEndpointError(f"robot endpoint refused describe ({status})")
        self.robot_id = description["robot_id"]
        self.resources = tuple(ResourceDescriptor(**{**r, "capabilities": tuple(r["capabilities"])})
                               for r in description["resources"])
        self.tool_descriptors = {t["name"]: ToolDescriptor(**{**t, "requires": tuple(t["requires"]),
                                                               "writes": tuple(t["writes"])})
                                 for t in description["tools"]}
        self.trace = _RemoteTrace() if description["trace_recorded"] is True else None
        self.catalog_sha256 = catalog_digest(description)
        status, lease = self._request("POST", "/v1/lease", {"lease_id": None})
        if status != 200:
            raise RobotEndpointError(f"robot endpoint lease refused ({status})")
        self.lease_id, self.lease_ttl_s = lease["lease_id"], lease["ttl_s"]
        self._renewer = threading.Thread(target=self._renew, name="robot-endpoint-renew", daemon=True)
        self._renewer.start()
        return self

    def _renew(self):
        last_ok = time.monotonic()
        while not self._closed.wait(self.lease_ttl_s / 3):
            try:
                status, _ = self._request("POST", "/v1/lease", {"lease_id": self.lease_id})
            except RobotEndpointError:
                status = None
            if status == 200:
                last_ok = time.monotonic()
            elif status == 410 or time.monotonic() - last_ok >= self.lease_ttl_s:
                # Gone, or certainly lapsed: the endpoint has latched (or will
                # latch) its stop. This supervisor never acts on the robot again.
                with self._lock:
                    self._lost = True
                return

    def _lease_lost(self):
        with self._lock:
            return self._lost or self.lease_id is None

    def endpoint_receipt(self):
        return {"origin": self.origin, "protocol": PROTOCOL, "catalog_sha256": self.catalog_sha256,
                "lease_ttl_s": self.lease_ttl_s}

    # ------------------------------------------------------------------ runtime surface

    def _state(self):
        status, state = self._request("GET", "/v1/state")
        if status != 200:
            raise RobotEndpointError(f"robot endpoint refused state ({status})")
        return state

    @property
    def cancellation_token(self):
        return self._state()["generation"]

    @property
    def stopped(self):
        try:
            return self._state()["stopped"] is not False
        except (RobotEndpointError, KeyError):
            return True        # unknown is never reported as running

    def ready(self):
        if self._lease_lost():
            return {"ready": False, "reason": "lease lost"}
        try:
            self._state()
        except RobotEndpointError:
            return {"ready": False, "reason": "robot endpoint unreachable"}
        return {"ready": True, "reason": None}

    def execute(self, name, args=None, *, expected_generation=None, deadline_monotonic_s=None):
        if not _generation(expected_generation) or not _finite_number(deadline_monotonic_s):
            return {"ok": False, "error": "ValueError: remote execution requires an episode generation "
                                          "and a local deadline"}
        if self._lease_lost():
            return {"ok": False, "error": "robot endpoint lease lost; restart this supervisor"}
        remaining = deadline_monotonic_s - time.monotonic()
        if remaining <= 0:
            return {"ok": False, "error": "ValueError: execution deadline expired"}
        try:
            status, reply = self._request("POST", "/v1/execute", {
                "lease_id": self.lease_id, "tool": name, "arguments": {} if args is None else args,
                "expected_generation": expected_generation, "deadline_remaining_s": min(remaining, MAX_DEADLINE_S)},
                timeout=self.execute_timeout_s)
        except (RobotEndpointError, TypeError, ValueError) as exc:
            return {"ok": False, "execution_ok": False, "delivery_uncertain": True, "error": str(exc)}
        if status == 410:
            with self._lock:
                self._lost = True
            return {"ok": False, "error": "robot endpoint revoked the lease (410); restart this supervisor"}
        if status != 200:
            return {"ok": False, "execution_ok": False, "delivery_uncertain": status >= 500,
                    "error": f"robot endpoint refused execute ({status}): {reply.get('error')}"}
        return reply["result"]

    def stop(self, **_kwargs):
        try:
            status, reply = self._request("POST", "/v1/stop", {})
        except RobotEndpointError as exc:
            return {"ok": False, "latched": None, "error": str(exc), "physical_stop_verified": False}
        if status != 200:
            return {"ok": False, "latched": None, "error": f"robot endpoint refused stop ({status})",
                    "physical_stop_verified": False}
        return reply["result"]

    def reset_stop(self, *, expected_generation=None, deadline_monotonic_s=None, domain=None):
        if domain is not None or deadline_monotonic_s is not None or not _generation(expected_generation):
            return {"ok": False, "error": "remote reset requires only the exact observed generation"}
        if self._lease_lost():
            return {"ok": False, "error": "robot endpoint lease lost; restart this supervisor"}
        try:
            status, reply = self._request("POST", "/v1/reset", {"lease_id": self.lease_id,
                                                                "expected_generation": expected_generation})
        except RobotEndpointError as exc:
            return {"ok": False, "error": str(exc)}
        if status != 200:
            return {"ok": False, "error": f"robot endpoint refused reset ({status})"}
        return reply["result"]

    def _record_tool_result(self, name, args, result, *, started_monotonic_s):
        status, reply = self._request("POST", "/v1/record", {
            "lease_id": self.lease_id, "tool": name, "arguments": args or {}, "result": result,
            "duration_ms": max(0., (time.monotonic() - started_monotonic_s) * 1000)})
        if status != 200:
            raise RobotEndpointError(f"robot endpoint refused the stop record ({status}): {reply.get('error')}")

    def close(self):
        self._closed.set()
        if self._renewer is not None:
            self._renewer.join()
        released = False
        if self.lease_id is not None and not self._lease_lost():
            try:
                released = self._request("DELETE", "/v1/lease", {"lease_id": self.lease_id})[0] == 200
            except RobotEndpointError:
                released = False
        with self._lock:
            self._lost = True
        return {"ok": released, "complete": True, "lease_released": released, "robot_runtime_closed": False}
