"""Pure admission/lease state for a single simulator-owned mobile controller.

This class never steps physics, writes a body pose, or evaluates a policy. Kit
owns those operations. A control tick consumes a bounded body-frame command;
state readers only receive copies of the last completed physics observation.
"""
from __future__ import annotations

import copy
import math
import ipaddress
import json
import socket
import threading
import time
import uuid

from cascade.control.mobile_base import BaseState


def _number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not a boolean/string")
    return float(value)


def _positive(value, name: str) -> float:
    value = _number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _vector(value, count: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{name} must contain {count} numbers")
    return [_number(x, name) for x in value]


def _token(value, name: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 256
            or value != value.strip() or any(ord(c) < 32 for c in value)):
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


class MobileBridgeController:
    """Admission and a dual-clock deadman, independent of the RPC worker.

    Admission consumes a generation. Stop, fault and reset invalidate it;
    normal completion does not. Resetting the stop latch
    does not restore the old command; a fault requires an explicit new epoch
    after the simulator owner has repaired/reset the scene outside an episode.
    """

    def __init__(
        self, *, robot_id: str, source: str, engine: str, device: str,
        asset_sha256: str, policy_sha256: str,
        max_linear_speed: float, max_angular_speed: float, max_duration_s: float,
        lease_s: float, max_state_age_s: float, physics_dt: float = 0.005,
        policy_dt: float = 0.020, max_action_wall_s: float = 120.0,
        clock=time.monotonic,
    ):
        if engine not in {"physx", "newton"}:
            raise ValueError("engine must be physx or newton")
        for label, digest in (("asset", asset_sha256), ("policy", policy_sha256)):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
        self._identity = {
            "protocol": 1, "kind": "microduck", "robot_id": _token(robot_id, "robot_id"),
            "source": _token(source, "source"), "engine": engine,
            "device": _token(device, "device"), "asset_sha256": asset_sha256,
            "policy_sha256": policy_sha256, "physics_dt": _positive(physics_dt, "physics_dt"),
            "policy_dt": _positive(policy_dt, "policy_dt"),
            "capabilities": ["state", "velocity", "stop", "reset_stop"],
            "measurement_kind": "physics",
        }
        self.max_linear_speed = _positive(max_linear_speed, "max_linear_speed")
        self.max_angular_speed = _positive(max_angular_speed, "max_angular_speed")
        self.max_duration_s = _positive(max_duration_s, "max_duration_s")
        self.lease_s = _positive(lease_s, "lease_s")
        self.max_state_age_s = _positive(max_state_age_s, "max_state_age_s")
        self.max_action_wall_s = _positive(max_action_wall_s, "max_action_wall_s")
        self._clock = clock
        self._lock = threading.RLock()  # only short in-memory operations; never I/O/Kit
        self._generation = 0
        self._epoch = uuid.uuid4().hex
        self._state: dict | None = None
        self._state_wall: float | None = None
        self._active: dict | None = None
        self._latched = False
        self._fault = ""
        self._last_control_time: float | None = None
        self._last_completed: str | None = None
        self._completed_owner: str | None = None

    def hello(self) -> dict:
        with self._lock:
            return {"ok": True, **copy.deepcopy(self._identity), "epoch": self._epoch,
                    "generation": self._generation, "lease_s": self.lease_s,
                    "max_linear_speed": self.max_linear_speed,
                    "max_angular_speed": self.max_angular_speed,
                    "max_duration_s": self.max_duration_s,
                    "max_action_wall_s": self.max_action_wall_s}

    def _status(self) -> str:
        if self._fault:
            return "fault"
        if self._latched:
            return "stopped"
        if self._state is None:
            return "disarmed"
        return "walking" if self._active is not None else "balancing"

    def _mobile_status(self) -> str:
        if self._fault:
            return "fault"
        if self._state is None:
            return "disabled"
        # A command-permission latch is not torque-off. Only an explicit
        # producer attestation may preserve live balance after that latch;
        # missing attestation remains conservative. It proves neither rest
        # nor upright stability: the independent physical windows still do.
        balance = self._state.get("balance_active")
        if balance is False or (self._latched and balance is not True):
            return "disabled"
        return "active" if self._active else "ready"

    def _ack(self) -> dict:
        return {"ok": True, "robot_id": self._identity["robot_id"],
                "source": self._identity["source"], "epoch": self._epoch,
                "generation": self._generation, "latched": self._latched}

    def _snapshot(self, state: dict, age: float) -> dict:
        return BaseState(
            robot_id=self._identity["robot_id"], source=self._identity["source"],
            epoch=self._epoch, step=state["step"], sim_time_s=state["sim_time"],
            received_monotonic_s=self._state_wall or 0., producer_age_s=age,
            position_world=state["position"], orientation_wxyz=state["orientation_wxyz"],
            linear_velocity_world=state["linear_velocity"],
            angular_velocity_body=state["angular_velocity"],
            joint_names=state["joint_names"], joint_positions=state["q"],
            joint_velocities=state["dq"], controller_status=self._mobile_status(),
            generation=self._generation, contacts=state["contacts"], fallen=state["fallen"],
            latched=self._latched, measurement_kind="physics",
        ).as_dict()

    def state(self) -> dict:
        """Return history, not a refresh or an assertion that physics advanced."""
        with self._lock:
            age = None if self._state_wall is None else max(0.0, self._clock() - self._state_wall)
            return {
                **copy.deepcopy(self._state or {}), **self.hello(),
                "controller": self._status(), "latched": self._latched,
                "fault": self._fault, "state_age_s": age,
                "command_id": self._active["command_id"] if self._active else None,
                "last_completed_command_id": self._last_completed,
                "feedback_available": self._state is not None,
                "state": self._snapshot(self._state, age) if self._state is not None else None,
            }

    def publish(self, state: dict) -> None:
        """Called only by the simulator after a completed physical step.

        Bad/repeated feedback is terminal. It must not refresh a producer age
        or allow a dead client to keep walking on a frozen observation.
        """
        with self._lock:
            try:
                if not isinstance(state, dict):
                    raise ValueError("physics state must be an object")
                step = state.get("step")
                if type(step) is not int or step < 0:
                    raise ValueError("physics step must be a nonnegative integer")
                sim_time = _number(state.get("sim_time"), "sim_time")
                if sim_time < 0:
                    raise ValueError("simulation time must not be negative")
                if self._state is not None and (
                    step <= self._state["step"] or sim_time <= self._state["sim_time"]
                ):
                    raise ValueError("physics step and time must advance within an epoch")
                validated = copy.deepcopy(state)
                for key, size in (("position", 3), ("orientation_wxyz", 4),
                                  ("linear_velocity", 3), ("angular_velocity", 3),
                                  ("q", 14), ("dq", 14)):
                    validated[key] = _vector(state.get(key), size, key)
                norm = math.sqrt(sum(x * x for x in validated["orientation_wxyz"]))
                if not math.isclose(norm, 1.0, abs_tol=1e-4):
                    raise ValueError("physics orientation must be a unit wxyz quaternion")
                if type(state.get("fallen")) is not bool:
                    raise ValueError("physics state has no boolean fall observation")
                if "balance_active" in state and type(state["balance_active"]) is not bool:
                    raise ValueError("balance_active must be an explicit boolean")
                for key in ("robot_id", "source", "engine"):
                    if key in state and state[key] != self._identity[key]:
                        raise ValueError(f"physics {key} changed")
                if "epoch" in state and state["epoch"] != self._epoch:
                    raise ValueError("physics epoch changed without reset")
                validated["step"], validated["sim_time"] = step, sim_time
                if len(state.get("joint_names", ())) != 14:
                    raise ValueError("joint_names must name all 14 observed joints")
                self._snapshot(validated, 0.)  # same strict schema as all MOBILE readers
            except (ValueError, TypeError, KeyError) as exc:
                self.fault(str(exc))
                raise ValueError(str(exc)) from exc
            self._state = validated
            self._state_wall = self._clock()
            if validated["fallen"]:
                self.fault("physics state is fallen")
                raise ValueError("physics state is fallen")

    def _binding(self, request: dict, *, robot: bool = False) -> None:
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        if request.get("epoch") != self._epoch:
            raise ValueError("command epoch mismatch")
        generation = request.get("generation")
        if type(generation) is not int or generation != self._generation:
            raise ValueError("command generation mismatch")
        if robot:
            for key in ("robot_id", "source"):
                if request.get(key) != self._identity[key]:
                    raise ValueError(f"command {key} mismatch")

    def _fresh(self, now: float) -> None:
        if self._state is None or self._state_wall is None:
            raise ValueError("no completed physics state")
        if now < self._state_wall or now - self._state_wall > self.max_state_age_s:
            raise ValueError("physics state is stale or its wall clock regressed")
        if self._state.get("balance_active") is False:
            raise ValueError("balance controller is not active")

    def command_velocity(self, request: dict) -> dict:
        with self._lock:
            self._binding(request, robot=True)
            if self._latched or self._fault:
                raise ValueError("motion is latched; operator reset required")
            now = self._clock()
            self._fresh(now)
            received = _number(request.get("_received_wall", now), "request receipt")
            if received > now or now - received >= min(self.lease_s, self.max_action_wall_s):
                raise ValueError("command expired in admission queue")
            owner = _token(request.get("owner"), "owner")
            command_id = _token(request.get("command_id"), "command_id")
            vx, vy, wz = (_number(request.get(k), k) for k in ("vx", "vy", "wz"))
            duration = _positive(request.get("duration_s"), "duration_s")
            if math.hypot(vx, vy) > self.max_linear_speed:
                raise ValueError("linear velocity exceeds admitted limit")
            if abs(wz) > self.max_angular_speed:
                raise ValueError("angular velocity exceeds admitted limit")
            if duration > self.max_duration_s:
                raise ValueError("duration exceeds admitted limit")
            if self._active is not None:
                raise ValueError("an active command already owns the controller")
            if command_id == self._last_completed:
                raise ValueError("command_id was already completed; no motion replay")
            self._generation += 1
            self._active = {
                "owner": owner, "command_id": command_id, "twist": (vx, vy, wz),
                "until_sim": self._state["sim_time"] + duration,
                "lease_until_wall": received + self.lease_s,
                "until_wall": received + self.max_action_wall_s,
            }
            return {**self._ack(), "accepted": True, "completed": False,
                    "command_id": command_id, "start_sim_time_s": self._state["sim_time"],
                    "end_sim_time_s": self._active["until_sim"]}

    def renew(self, request: dict) -> dict:
        with self._lock:
            self._binding(request)
            _token(request.get("owner"), "owner")
            _token(request.get("command_id"), "command_id")
            if self._active is None:
                if (request.get("command_id") == self._last_completed
                        and request.get("owner") == self._completed_owner):
                    return {**self._ack(), "active": False}
                raise ValueError("no active command to renew")
            if request.get("owner") != self._active["owner"]:
                raise ValueError("lease owner mismatch")
            if request.get("command_id") != self._active["command_id"]:
                raise ValueError("lease command_id mismatch")
            now = self._clock()
            if now >= self._active["lease_until_wall"] or now >= self._active["until_wall"]:
                self.fault("command lease expired")
                raise ValueError("expired lease cannot be renewed")
            self._fresh(now)
            self._active["lease_until_wall"] = min(now + self.lease_s, self._active["until_wall"])
            return {**self._ack(), "active": True}

    def watchdog(self) -> None:
        """Server wall-clock thread: runs even when Kit is paused or blocked."""
        with self._lock:
            if self._active is None:
                return
            now = self._clock()
            if now >= self._active["lease_until_wall"] or now >= self._active["until_wall"]:
                self.fault("command lease expired")
                return
            try:
                self._fresh(now)
            except ValueError as exc:
                self.fault(str(exc))

    def owner_disconnected(self, owner: str) -> None:
        with self._lock:
            if self._active is not None and self._active["owner"] == owner:
                self.stop(latch=True)

    def control_at(self, sim_time: float) -> tuple[float, float, float]:
        """Called before physics; zero twist does not mean cut motor torque."""
        with self._lock:
            sim_time = _number(sim_time, "sim_time")
            if (sim_time < 0 or (self._last_control_time is not None
                                and sim_time < self._last_control_time)):
                self.fault("control simulation clock regressed")
            self._last_control_time = sim_time
            if self._latched or self._fault or self._active is None:
                return (0.0, 0.0, 0.0)
            now = self._clock()
            if now >= self._active["lease_until_wall"] or now >= self._active["until_wall"]:
                self.fault("command lease expired")
                return (0.0, 0.0, 0.0)
            try:
                self._fresh(now)
            except ValueError as exc:
                self.fault(str(exc))
                return (0.0, 0.0, 0.0)
            if sim_time >= self._active["until_sim"]:
                self._last_completed = self._active["command_id"]
                self._completed_owner = self._active["owner"]
                self._active = None
                return (0.0, 0.0, 0.0)
            return self._active["twist"]

    def stop(self, *, latch: bool = True) -> dict:
        if type(latch) is not bool:
            raise ValueError("latch must be a boolean")
        with self._lock:
            self._active = None
            self._generation += 1
            self._latched = self._latched or latch
            return {**self._ack(), "cancelled": True,
                    "physical_stop_verified": False}

    def reset_stop(self, request: dict | None = None) -> dict:
        with self._lock:
            if request is not None:
                self._binding(request)
            if self._active is not None:
                raise ValueError("active command must be stopped before reset")
            if self._fault:
                raise ValueError("controller fault requires a new simulator epoch")
            self._active = None
            self._generation += 1
            self._latched = False
            return {**self._ack(), "resumed_motion": False}

    def fault(self, reason: str) -> None:
        with self._lock:
            if not self._fault:
                self._fault = _token(reason, "fault reason")
                self._generation += 1
            self._active = None
            self._latched = True

    def begin_epoch(self) -> dict:
        """Simulator-owner fixture only; never exposed as an agent motion tool."""
        with self._lock:
            self._epoch = uuid.uuid4().hex
            self._generation += 1
            self._active = None
            self._state = None
            self._state_wall = None
            self._last_control_time = None
            self._last_completed = None
            self._completed_owner = None
            self._fault = ""
            self._latched = True  # reset is not an authorization to walk
            return self.hello()


def loopback_address(host: str) -> str:
    """No DNS resolution or remote interfaces, including at construction time."""
    if host == "localhost":
        return "127.0.0.1"
    try:
        addr = ipaddress.ip_address(host)
    except (ValueError, TypeError) as exc:
        raise ValueError("bridge host must be a loopback IP or localhost") from exc
    if not addr.is_loopback:
        raise ValueError("bridge host must be loopback")
    return str(addr)


class MobileBridgeServer:
    """Bounded newline JSON RPC; each connection has its own worker.

    Only cached Python snapshots are legal callbacks here, never Kit calls.
    A frame_callback(request) must be thread-safe and bounded; arbitrary native
    functions cannot be forcibly cancelled by a Python transport. It is never
    invoked under a controller, ownership or stop lock.
    """

    MAX_REQUEST_BYTES = 16 * 1024
    MAX_RESPONSE_BYTES = 8 * 1024 * 1024
    MAX_CONNECTIONS = 32
    IO_TIMEOUT_S = 1.0

    def __init__(self, controller, *, host="127.0.0.1", port: int, frame_callback=None):
        self.controller = controller
        self._host = loopback_address(host)
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("explicit bridge port must be an integer in 0..65535")
        self._address = (self._host, port)
        self._frame = frame_callback
        self._guard = threading.Lock()  # registry only, never controller calls/I/O
        self._halt = threading.Event()
        self._slots = threading.BoundedSemaphore(self.MAX_CONNECTIONS)
        self._clients = set()
        self._workers = set()
        self._owners = {}
        self._listener = None
        self._accept_thread = None
        self._watchdog_thread = None

    @property
    def address(self):
        return self._address

    def start(self) -> None:
        if self._listener is not None:
            return
        if self._halt.is_set():
            raise RuntimeError("closed bridge server cannot be restarted")
        sock = socket.socket(socket.AF_INET6 if ":" in self._host else socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(self._address)
            sock.listen(self.MAX_CONNECTIONS)
            sock.settimeout(0.05)
        except BaseException:
            sock.close()
            raise
        self._address = sock.getsockname()[:2]
        self._listener = sock
        self._accept_thread = threading.Thread(target=self._accept, name="mobile-rpc-accept", daemon=True)
        self._watchdog_thread = threading.Thread(target=self._watchdog, name="mobile-rpc-watchdog", daemon=True)
        self._accept_thread.start()
        self._watchdog_thread.start()

    def _watchdog(self):
        interval = max(0.001, min(0.05, self.controller.lease_s / 4))
        while not self._halt.wait(interval):
            self.controller.watchdog()

    def dispatch(self, request: dict) -> dict:
        """Pure routing entry point; errors are explicit protocol responses."""
        try:
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            op = request.get("op")
            if op == "hello":
                hello = self.controller.hello()
                if self._frame is not None:
                    hello["capabilities"].append("frame")
                return hello
            if op == "state":
                return self.controller.state()
            if op == "command_velocity":
                request = {**request, "_received_wall": request.get("_received_wall", self.controller._clock())}
                return self.controller.command_velocity(request)
            if op == "renew":
                return self.controller.renew(request)
            if op == "stop":
                return self.controller.stop(latch=request.get("latch", True))
            if op == "reset_stop":
                return self.controller.reset_stop(request)
            if op == "frame" and self._frame is not None:
                return {**copy.deepcopy(self._frame(request)), "ok": True}
            raise ValueError("unsupported mobile bridge operation")
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}

    def _accept(self):
        while not self._halt.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                conn.close()
                continue
            with self._guard:
                worker = threading.Thread(target=self._serve, args=(conn,), name="mobile-rpc-client", daemon=True)
                self._clients.add(conn)
                self._workers.add(worker)
            worker.start()

    @staticmethod
    def _decode(line):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON field")
                result[key] = value
            return result
        def bad_constant(value):
            raise ValueError(f"non-finite JSON number: {value}")
        return json.loads(line, object_pairs_hook=pairs, parse_constant=bad_constant)

    def _send(self, conn, result):
        wire = json.dumps(result, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(wire) > self.MAX_RESPONSE_BYTES:
            raise ValueError("response too large")
        conn.settimeout(self.IO_TIMEOUT_S)
        conn.sendall(wire)

    def _serve(self, conn):
        role, owner, owner_conn = "reader", None, None
        handshaken = False
        buffer = bytearray()
        deadline = None
        received_wall = None
        try:
            while not self._halt.is_set():
                left = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
                if left <= 0:
                    raise ValueError("request wall-time deadline expired")
                conn.settimeout(left)
                try:
                    part = conn.recv(4096)
                except socket.timeout:
                    continue
                if not part:
                    return
                if deadline is None:
                    deadline = time.monotonic() + self.IO_TIMEOUT_S
                    received_wall = self.controller._clock()
                buffer.extend(part)
                if len(buffer) > self.MAX_REQUEST_BYTES:
                    raise ValueError("request too large")
                if b"\n" not in buffer:
                    continue
                line, extra = bytes(buffer).split(b"\n", 1)
                if extra:
                    raise ValueError("pipelined requests are not supported")
                request = self._decode(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                request["_received_wall"] = received_wall
                op = request.get("op")
                if op == "hello":
                    if handshaken:
                        raise ValueError("channel already handshaken")
                    role = request.get("role", "reader")
                    if role not in {"reader", "control", "stop", "renew"}:
                        raise ValueError("unknown channel role")
                    if role != "reader":
                        owner = _token(request.get("owner"), "owner")
                        with self._guard:
                            if role == "control":
                                if owner in self._owners:
                                    raise ValueError("owner channel already connected")
                                self._owners[owner] = conn
                            elif owner not in self._owners:
                                raise ValueError("owner control channel is not connected")
                            owner_conn = self._owners[owner]
                    handshaken = True
                    result = self.dispatch(request)
                else:
                    allowed = {"reader": {"state", "frame"}, "control": {"command_velocity", "reset_stop"},
                               "stop": {"stop"}, "renew": {"renew"}}
                    with self._guard:
                        owner_live = owner_conn is not None and self._owners.get(owner) is owner_conn
                    if role != "reader" and not owner_live:
                        result = {"ok": False, "error": "owner channel disconnected"}
                    elif not handshaken or op not in allowed[role]:
                        result = {"ok": False, "error": "operation forbidden on channel role"}
                    elif op in {"command_velocity", "renew"} and request.get("owner") != owner:
                        result = {"ok": False, "error": "request owner differs from channel owner"}
                    else:
                        result = self.dispatch(request)
                self._send(conn, result)
                buffer.clear()
                deadline = None
        except (OSError, ValueError, TypeError, RecursionError):
            # Invalid framing poisons the stream: no retry or late response reuse.
            pass
        finally:
            if role == "control" and owner is not None:
                with self._guard:
                    registered = self._owners.get(owner) is conn
                    if registered:
                        del self._owners[owner]
                if registered:
                    self.controller.owner_disconnected(owner)
            conn.close()
            with self._guard:
                self._clients.discard(conn)
                self._workers.discard(threading.current_thread())
            self._slots.release()

    def close(self) -> None:
        if self._halt.is_set():
            return
        self._halt.set()
        if self._listener is not None:
            self.controller.stop(latch=True)
            self._listener.close()
        with self._guard:
            clients, workers = list(self._clients), list(self._workers)
        for conn in clients:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
        deadline = time.monotonic() + self.IO_TIMEOUT_S + 0.2
        for worker in [self._accept_thread, self._watchdog_thread, *workers]:
            if worker is not None and worker is not threading.current_thread():
                worker.join(max(0., deadline - time.monotonic()))
