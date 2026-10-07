"""Read-only MicroDuck truth over a separately owned, bounded TCP connection.

This reader can send only hello(role=reader) and state. It never constructs an
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

    def __call__(self) -> BaseState | None:
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
            # No import of an actuator, policy, Isaac or Kit. This pure helper
            # validates BaseState and stamps the receipt clock on THIS client.
            from ..control.isaac_base import state_from_wire
            started = time.monotonic()
            payload = self._client.request({"op": "state"}, timeout_s=remaining())
            received = time.monotonic()
            value = state_from_wire(payload, received_monotonic_s=received,
                                    round_trip_s=received - started)
            if (value.robot_id != self._profile["robot_id"] or value.source != self._profile["source"]
                    or value.epoch != self._epoch or value.measurement_kind != "physics"
                    or value.model_identity_sha256 != self._profile["model_identity_sha256"]):
                raise ValueError("truth state identity/source/epoch/measurement_kind mismatch")
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

    def close(self):
        self._closed.set()
        # Each request is bounded by the same deadline, so waiting for its
        # transport lock cannot strand a live observation worker indefinitely.
        with self._lock:
            self._client.close()
            self._ready = False
