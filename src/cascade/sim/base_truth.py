"""Read-only MicroDuck truth over a separately owned, bounded TCP connection.

This reader can send only hello(role=reader), state and (opt-in, B72) the
reader op state_history. It never constructs an
IsaacBase, requests control ownership, steps physics, or modifies any target.
State parsing is shared with the transport, not with an actuator's result.
"""
from __future__ import annotations

from copy import deepcopy
import ipaddress
import re
import threading
import time

from ..control.mobile_base import BaseState, finite_real, identifier, nonnegative_int
from .bridge_client import BridgeClient


class BaseTruthReader:
    """Independent observation channel, pinned to its first valid hello epoch.

    The explicit profile uses the IsaacBase identity/endpoint fields. Construction
    has no I/O. ``timeout_s`` bounds a complete call, including initial connect,
    hello and state; configure checker.read_timeout_s at least this large.
    A reset requires a NEW reader: reconnection never silently rebinds epoch.
    """

    def __init__(self, profile: dict):
        required = {"robot_id", "source", "engine", "device", "asset_sha256",
                    "policy_sha256", "model_identity_sha256", "support_contract",
                    "bridge_host", "bridge_port", "timeout_s"}
        if not isinstance(profile, dict) or not required <= profile.keys():
            raise ValueError("truth reader requires explicit identity, hashes and endpoint")
        profile = deepcopy(profile)
        for key in ("robot_id", "source", "device", "bridge_host"):
            identifier(profile[key], key)
        if profile["engine"] not in ("physx", "newton"):
            raise ValueError("unsupported engine")
        for key in ("asset_sha256", "policy_sha256", "model_identity_sha256"):
            if not isinstance(profile[key], str) or not re.fullmatch(r"[0-9a-f]{64}", profile[key]):
                raise ValueError(f"{key} must be an exact lowercase sha256")
        try:
            loopback = profile["bridge_host"] == "localhost" or ipaddress.ip_address(profile["bridge_host"]).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("truth endpoint must be explicit loopback")
        port = nonnegative_int(profile["bridge_port"], "bridge_port")
        if not 1 <= port <= 65535:
            raise ValueError("bridge_port must be 1..65535; no implicit default")
        timeout = finite_real(profile["timeout_s"], "timeout_s")
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        for key, default in (("physics_dt", 0.005), ("policy_dt", 0.020)):
            profile[key] = finite_real(profile.get(key, default), key)
            if profile[key] <= 0:
                raise ValueError(f"{key} must be positive")
        from .mobile_identity import support_contract_digest
        self._support_contract_sha256 = support_contract_digest(profile['support_contract'], profile['model_identity_sha256'])
        self._profile = profile
        self._timeout_s = timeout
        self._client = BridgeClient(host=profile["bridge_host"], port=port, timeout_s=timeout)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._ready = False
        self._history_size = None  # advertised `state_history` bound (B72), from hello
        self._epoch = identifier(profile["epoch"], "epoch") if "epoch" in profile else None
        from .mobile_bridge import KINDS
        self._kind = profile.get("kind", "microduck")
        if self._kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        self.last_error: str | None = None

    def _hello(self, response):
        if response.get("ok") is not True:
            raise ValueError("truth hello requires boolean ok=true")
        if response.get('support_contract_sha256') != self._support_contract_sha256:
            raise ValueError('truth hello support_contract mismatch')
        if type(response.get("protocol")) is not int or response["protocol"] != 1:
            raise ValueError("truth protocol mismatch")
        if response.get("kind") != self._kind or response.get("measurement_kind") != "physics":
            raise ValueError("truth requires the pinned embodiment's physics, not another robot or mock")
        for key in ("robot_id", "source", "engine", "device", "asset_sha256", "policy_sha256", "model_identity_sha256"):
            if response.get(key) != self._profile[key]:
                raise ValueError(f"truth hello {key} mismatch")
        for key in ("physics_dt", "policy_dt"):
            if finite_real(response.get(key), key) != self._profile[key]:
                raise ValueError(f"truth hello {key} mismatch")
        capabilities = response.get("capabilities")
        if not isinstance(capabilities, list) or "state" not in capabilities:
            raise ValueError("truth endpoint lacks state capability")
        epoch = identifier(response.get("epoch"), "epoch")
        if self._epoch is not None and epoch != self._epoch:
            raise ValueError("truth hello epoch changed; construct a new reader after reset")
        self._epoch = epoch
        # Opt-in producer history (B72): its advertised bound, checked on use.
        self._history_size = response.get("state_history") if "state_history" in capabilities else None

    def __call__(self) -> BaseState | None:
        return self._exchange(self._state)

    def history(self, after_step=None):
        """The completed states the bridge recorded as it published them (its
        opt-in `state_history`, B72) with a step after `after_step`, oldest
        first. None on any failure (`last_error`), exactly like a state read;
        a bridge that does not advertise the capability is refused."""
        return self._exchange(lambda remaining: self._history(after_step, remaining))

    def _exchange(self, request):
        deadline = time.monotonic() + self._timeout_s
        if not self._lock.acquire(timeout=self._timeout_s):
            self.last_error = "truth reader lock timeout"
            return None
        try:
            if self._closed.is_set():
                self.last_error = "truth reader is closed"
                return None

            def remaining():
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("truth reader wall deadline")
                return budget

            if not self._ready:
                self._client.connect()
                self._hello(self._client.request({"op": "hello", "role": "reader"}, timeout_s=remaining()))
                self._ready = True
            value = request(remaining)
            if self._closed.is_set():
                raise ValueError("truth reader closed during read")
            remaining()
            self.last_error = None
            return value
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:400]
            self._ready = False
            self._client.close()
            return None
        finally:
            self._lock.release()

    def _checked(self, value):
        if (value.robot_id != self._profile["robot_id"] or value.source != self._profile["source"]
                or value.epoch != self._epoch or value.measurement_kind != "physics"
                or value.model_identity_sha256 != self._profile["model_identity_sha256"]):
            raise ValueError("truth state identity/source/epoch/measurement_kind mismatch")
        return value

    def _state(self, remaining):
        # No import of an actuator, policy, Isaac or Kit. This pure helper
        # validates BaseState and stamps the receipt clock on THIS client.
        from ..control.isaac_base import state_from_wire
        started = time.monotonic()
        payload = self._client.request({"op": "state"}, timeout_s=remaining())
        received = time.monotonic()
        return self._checked(state_from_wire(payload, received_monotonic_s=received,
                                             round_trip_s=received - started))

    def _history(self, after_step, remaining):
        from ..control.isaac_base import state_from_wire
        from .mobile_bridge import MAX_STATE_HISTORY
        size = self._history_size
        if type(size) is not int or not 1 <= size <= MAX_STATE_HISTORY:
            raise ValueError("truth endpoint does not serve a bounded state_history")
        if after_step is not None and (type(after_step) is not int or after_step < 0):
            raise ValueError("after_step must be a nonnegative integer or None")
        started = time.monotonic()
        payload = self._client.request({"op": "state_history", "after_step": after_step}, timeout_s=remaining())
        received = time.monotonic()
        if payload.get("epoch") != self._epoch:
            raise ValueError("truth state history epoch mismatch")
        states = payload.get("states")
        if not isinstance(states, list) or len(states) > size:
            raise ValueError("truth state history must be a bounded list")
        values, last = [], None
        for item in states:
            # One receipt for the whole reply; every age includes its round trip.
            value = self._checked(state_from_wire(item, received_monotonic_s=received,
                                                  round_trip_s=received - started))
            if ((after_step is not None and value.step <= after_step)
                    or (last is not None and (value.step <= last.step or value.sim_time_s <= last.sim_time_s))):
                raise ValueError("truth state history must advance")
            values.append(last := value)
        return tuple(values)

    def close(self):
        self._closed.set()
        # Each request is bounded by the same deadline, so waiting for its
        # transport lock cannot strand a live observation worker indefinitely.
        with self._lock:
            self._client.close()
            self._ready = False
