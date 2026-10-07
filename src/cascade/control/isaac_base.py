"""MicroDuck-only mobile transport; no simulator, policy, or arm construction.

Admission receipts are not physical outcomes. All calls use bounded separate
BridgeClient channels. Epoch/fences are captured before any transport wait;
uncertain delivery is never retried or converted into a successful receipt.
"""
from __future__ import annotations

import math
import threading
import time
import uuid

from .mobile_base import BaseState, MobileBase, VelocityCommand, finite_real, identifier, nonnegative_int
from ..sim.bridge_client import BridgeClient, BridgeError
from ..sim.mobile_bridge import loopback_address


def _positive(value, name):
    result = finite_real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def state_from_wire(payload, *, received_monotonic_s, round_trip_s) -> BaseState:
    """Pure decode: remote monotonic is NEVER treated as a local timestamp."""
    received = finite_real(received_monotonic_s, "received_monotonic_s")
    rtt = finite_real(round_trip_s, "round_trip_s")
    if received < 0 or rtt < 0:
        raise ValueError("receipt and round_trip_s must be nonnegative")
    if not isinstance(payload, dict):
        raise ValueError("state payload must be an object")
    if "ok" in payload:
        if payload["ok"] is not True or not isinstance(payload.get("state"), dict):
            raise ValueError("no completed physics state feedback")
        payload = payload["state"]
    state = BaseState.from_dict(payload)
    data = state.as_dict()
    data["received_monotonic_s"] = received
    data["producer_age_s"] += rtt  # entire RTT is a conservative transport bound
    return BaseState.from_dict(data)


class IsaacBase(MobileBase):
    REQUIRED = frozenset({"robot_id", "source", "engine", "device", "asset_sha256",
                          "policy_sha256", "model_identity_sha256", "support_contract",
                          "bridge_host", "bridge_port", "timeout_s"})

    def __init__(self, profile: dict):
        if not isinstance(profile, dict) or not self.REQUIRED <= profile.keys():
            missing = sorted(self.REQUIRED - profile.keys()) if isinstance(profile, dict) else sorted(self.REQUIRED)
            raise ValueError(f"IsaacBase requires explicit fields: {missing}")
        self._expected = {key: identifier(profile[key], key) for key in
                          ("robot_id", "source", "engine", "device", "asset_sha256", "policy_sha256", "model_identity_sha256")}
        if self._expected["engine"] not in {"physx", "newton"}:
            raise ValueError("engine must be physx or newton")
        for key in ("asset_sha256", "policy_sha256", "model_identity_sha256"):
            digest = self._expected[key]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{key} must be 64 lowercase hexadecimal characters")
        from ..sim.mobile_bridge import KINDS
        kind = profile.get("kind", "microduck")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        self._expected.update(protocol=1, kind=kind, measurement_kind="physics")
        from ..sim.mobile_identity import support_contract_digest
        self._expected['support_contract_sha256'] = support_contract_digest(
            profile['support_contract'], self._expected['model_identity_sha256'])
        for key, default in (("physics_dt", .005), ("policy_dt", .020)):
            self._expected[key] = _positive(profile.get(key, default), key)
        host = loopback_address(profile["bridge_host"])
        port = profile["bridge_port"]
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("bridge_port must be an explicit integer in 1..65535")
        self._address = (host, port)
        self._timeout = _positive(profile["timeout_s"], "timeout_s")
        self._gate = threading.Lock()  # memory only, never I/O
        self._lifecycle = threading.Lock()
        self._motion = threading.Lock()
        self._clients = {}
        self._connected = False
        self._epoch = None
        self._generation = None
        self._serial = 0
        self._session = 0
        self._owner = None
        self._latched = False
        self._pending_stop = False
        self._active = None
        self._renew_halt = threading.Event()
        self._renew_thread = None
        self._renew_interval = None

    @property
    def metadata(self):
        return {key: self._expected[key] for key in ("robot_id", "source", "measurement_kind", "model_identity_sha256")}

    @property
    def capabilities(self):
        return frozenset({"walk_velocity", "turn", "stop_navigation"})

    @property
    def connected(self):
        with self._gate:
            return self._connected

    def _hello(self, hello, epoch=None):
        if hello.get("ok") is not True:
            raise BridgeError("hello failed")
        for key, value in self._expected.items():
            actual = hello.get(key)
            if actual != value or (type(value) is int and type(actual) is not int):
                raise BridgeError(f"hello {key} mismatch")
        identifier(hello.get("epoch"), "hello epoch")
        nonnegative_int(hello.get("generation"), "hello generation")
        if epoch is not None and hello["epoch"] != epoch:
            raise BridgeError("hello epoch mismatch across channels")
        caps = hello.get("capabilities")
        if not isinstance(caps, list) or not {"state", "velocity", "stop", "reset_stop"} <= set(caps):
            raise BridgeError("hello capabilities missing")
        for field in ("lease_s", "max_action_wall_s"):
            _positive(hello.get(field), field)
        return hello

    def connect(self):
        if not self._lifecycle.acquire(blocking=False):
            raise BridgeError("connection lifecycle already in progress")
        clients = {}
        try:
            with self._gate:
                if self._connected:
                    return
                session = self._session
            owner, epoch = uuid.uuid4().hex, None
            for role in ("control", "stop", "reader", "renew"):
                client = BridgeClient(*self._address, timeout_s=self._timeout)
                clients[role] = client
                client.connect()
                hello = self._hello(client.request({"op": "hello", "role": role, "owner": owner}), epoch)
                epoch = hello["epoch"]
            with self._gate:
                if self._session != session:
                    raise BridgeError("connect cancelled by disconnect")
                self._clients = clients
                self._owner = owner
                self._epoch, self._generation = epoch, hello["generation"]
                self._connected = True
                pending = self._pending_stop
                self._renew_interval = min(hello["lease_s"] / 3., .1)
                self._renew_halt = threading.Event()
                self._renew_thread = threading.Thread(target=self._renew_loop,
                                                      name="mobile-client-renew", daemon=True)
                self._renew_thread.start()
            if pending:
                ack = self.stop(latch=True)
                if not ack["ok"]:
                    raise BridgeError("pending stop could not be acknowledged")
        except BaseException:
            with self._gate:
                self._connected = False
                self._clients = {}
                self._renew_halt.set()
            for client in clients.values():
                client.close()
            raise
        finally:
            self._lifecycle.release()

    def _channel(self, role):
        with self._gate:
            if not self._connected:
                raise BridgeError("mobile bridge not connected")
            return self._clients[role]

    def get_state(self):
        channel = self._channel("reader")
        begin = time.monotonic()
        payload = channel.request({"op": "state"})
        received = time.monotonic()
        try:
            state = state_from_wire(payload, received_monotonic_s=received, round_trip_s=received - begin)
            if any(getattr(state, key) != value for key, value in self.metadata.items()):
                raise ValueError("state identity/source/measurement_kind mismatch")
            with self._gate:
                if state.epoch != self._epoch:
                    raise ValueError("state epoch changed; explicit reconnect required")
                self._generation = max(self._generation, state.generation)
            return state
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc

    def _ack(self, response, *, epoch, generation=None):
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise BridgeError("invalid mobile ACK")
        for key in ("robot_id", "source", "model_identity_sha256"):
            if response.get(key) != self._expected[key]:
                raise BridgeError(f"ACK {key} mismatch")
        if response.get("epoch") != epoch:
            raise BridgeError("ACK epoch mismatch")
        actual = nonnegative_int(response.get("generation"), "ACK generation")
        if generation is not None and actual != generation:
            raise BridgeError("ACK generation mismatch")
        if type(response.get("latched")) is not bool:
            raise BridgeError("ACK missing boolean latched")
        return dict(response)

    def command_velocity(self, command, *, generation):
        if not isinstance(command, VelocityCommand):
            raise ValueError("VelocityCommand required")
        generation = nonnegative_int(generation, "generation")
        with self._gate:
            epoch, serial, owner = self._epoch, self._serial, self._owner
            if not self._connected or self._latched:
                return {"ok": False, "error": "not connected or stop latched"}
        request = {"op": "command_velocity", **command.as_dict(), "generation": generation,
                   "epoch": epoch, "robot_id": self._expected["robot_id"], "source": self._expected["source"],
                   "owner": owner, "command_id": uuid.uuid4().hex}
        if not self._motion.acquire(blocking=False):
            return {"ok": False, "error": "command or reset already in progress"}
        try:
            response = self._channel("control").request(request)
            ack = self._ack(response, epoch=epoch, generation=generation + 1)
            start = finite_real(ack.get("start_sim_time_s"), "start_sim_time_s")
            end = finite_real(ack.get("end_sim_time_s"), "end_sim_time_s")
            if (ack.get("accepted") is not True or ack["latched"] or start < 0 or end <= start
                    or not math.isclose(end - start, command.duration_s, rel_tol=1e-9, abs_tol=1e-12)):
                raise BridgeError("invalid admission/duration ACK")
            with self._gate:
                if self._serial != serial or not self._connected:
                    raise BridgeError("stop/disconnect crossed command delivery")
                self._generation = ack["generation"]
                self._active = {"op": "renew", "epoch": epoch, "generation": ack["generation"],
                                "owner": owner, "command_id": request["command_id"]}
            return ack
        except Exception as exc:
            self.stop(latch=True)
            return {"ok": False, "error": str(exc), "delivery_uncertain": True}
        except BaseException:
            self.stop(latch=True)
            raise
        finally:
            self._motion.release()

    def _renew_loop(self):
        while not self._renew_halt.wait(self._renew_interval):
            with self._gate:
                active = self._active
            if active is None:
                continue
            try:
                ack = self._ack(self._channel("renew").request(active), epoch=active["epoch"],
                                generation=active["generation"])
                if type(ack.get("active")) is not bool or ack["latched"]:
                    raise BridgeError("invalid renewal ACK")
                if not ack["active"]:
                    with self._gate:
                        if self._active is active:
                            self._active = None
            except Exception:
                # An old renewal crossing an intentional stop is not a new fault.
                with self._gate:
                    current = self._active is active
                if current:
                    self.stop(latch=True)

    def stop(self, *, latch=True):
        if type(latch) is not bool:
            raise ValueError("latch must be boolean")
        with self._gate:
            self._serial += 1
            self._active = None
            self._latched = self._latched or latch
            self._pending_stop = self._pending_stop or latch
            epoch, previous_generation = self._epoch, self._generation
        try:
            ack = self._ack(self._channel("stop").request({"op": "stop", "latch": latch}), epoch=epoch)
            if previous_generation is not None and ack["generation"] <= previous_generation:
                raise BridgeError("stop ACK did not invalidate generation")
            if latch and not ack["latched"]:
                raise BridgeError("stop ACK failed to latch")
            with self._gate:
                self._generation = max(self._generation, ack["generation"])
                self._latched = self._latched or ack["latched"]
                ack["latched"] = self._latched
            return ack
        except Exception as exc:
            with self._gate:
                self._latched = self._pending_stop = True
            return {"ok": False, "error": str(exc), "delivery_uncertain": True, "latched": True}

    def reset_stop(self):
        with self._gate:
            if self._active is not None:
                return {"ok": False, "error": "active motion must be stopped before reset"}
            epoch, generation, serial = self._epoch, self._generation, self._serial
        request = {"op": "reset_stop", "epoch": epoch, "generation": generation}
        if not self._motion.acquire(blocking=False):
            return {"ok": False, "error": "command or reset still in progress"}
        try:
            ack = self._ack(self._channel("control").request(request), epoch=epoch, generation=generation + 1)
            with self._gate:
                if self._serial != serial or ack["latched"]:
                    raise BridgeError("stop crossed reset; permission remains latched")
                self._latched = self._pending_stop = False
                self._generation = ack["generation"]
            return ack
        except Exception as exc:
            self.stop(latch=True)
            return {"ok": False, "error": str(exc), "latched": True}
        finally:
            self._motion.release()

    def disconnect(self):
        with self._gate:
            self._session += 1
            connected = self._connected
            self._renew_halt.set()
        if connected:
            self.stop(latch=True)
        with self._gate:
            self._serial += 1
            self._connected = False
            self._active = None
            clients, self._clients = self._clients, {}
        for client in clients.values():
            client.close()
        worker = self._renew_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(self._timeout + .2)
