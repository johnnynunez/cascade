"""Bounded passive observation registry. No provider may own actuator access.

A timed-out provider is quarantined, not retried with another worker. Python
cannot cancel an arbitrary blocked reader; close reports that worker honestly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import queue
import threading
import time
from typing import Protocol

from ..robotics.contracts import identifier
from .models import ObservationEnvelope, PAYLOAD_TYPES, digest, integer, number, token, wire


class SensorError(RuntimeError):
    pass


@dataclass(frozen=True)
class SensorDescriptor:
    sensor_id: str
    robot_id: str
    source: str
    modality: str
    frame_id: str
    clock_domain: str
    measurement_kind: str
    model_identity_sha256: str | None = None
    epoch: str | None = None
    calibration_id: str | None = None
    max_age_s: float = 0.5
    read_timeout_s: float = 0.25

    def __post_init__(self):
        identifier(self.sensor_id, "sensor_id")
        identifier(self.robot_id, "robot_id")
        for key in ("source", "frame_id", "clock_domain"):
            token(getattr(self, key), key)
        for key in ("epoch", "calibration_id"):
            if getattr(self, key) is not None:
                token(getattr(self, key), key)
        if self.modality not in {kind.modality for kind in PAYLOAD_TYPES}:
            raise ValueError("unknown sensor modality")
        if self.measurement_kind not in ("synthetic", "physics", "hardware"):
            raise ValueError("unknown measurement kind")
        digest(self.model_identity_sha256)
        if self.measurement_kind == "physics" and self.model_identity_sha256 is None:
            raise ValueError("physics sensors require model identity")
        for key, maximum in (("max_age_s", 60), ("read_timeout_s", 5)):
            value = number(getattr(self, key), key)
            if not 0 < value <= maximum:
                raise ValueError(f"{key} outside bounded range")
            object.__setattr__(self, key, value)

    def as_dict(self):
        return wire(self)


class SensorProvider(Protocol):
    """Construct/read/close must never step physics or acquire actuator ownership."""
    descriptor: SensorDescriptor

    def read(self) -> ObservationEnvelope: ...
    def close(self) -> None: ...


class _Slot:
    def __init__(self, provider):
        self.provider = provider
        self.descriptor = provider.descriptor
        self.requests = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.thread = None
        self.busy = False
        self.quarantined = False
        self.close_error = None
        self.watermark = None  # clocks/identity only: no image retained here

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, daemon=True,
                                           name=f"sensor:{self.descriptor.sensor_id}")
            self.thread.start()

    def run(self):
        try:
            while not self.closed.is_set():
                request = self.requests.get()
                if request is None or self.closed.is_set():
                    break
                done, result = request
                try:
                    result["value"] = self.provider.read()
                except BaseException as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"[:400]
                finally:
                    done.set()
                # An idle worker must not retain its previous payload outside
                # the byte-bounded hub history.
                del request, done, result
        finally:
            try:
                self.provider.close()
            except BaseException as exc:
                self.close_error = f"{type(exc).__name__}: {exc}"[:400]


class SensorHub:
    """A bounded history and one lazy reader worker per registered provider.

    Epoch changes require a new hub/provider. Repeated captures are refusals,
    including a repeated capture carrying a newly stamped receipt time.
    """
    def __init__(self, *, max_providers=16, max_history=32,
                 max_history_bytes=16 * 1024 * 1024, max_packet_bytes=8 * 1024 * 1024,
                 clock=time.monotonic):
        for value, label, cap in ((max_providers, "max_providers", 64),
                                  (max_history, "max_history", 256),
                                  (max_history_bytes, "max_history_bytes", 64 * 1024 * 1024),
                                  (max_packet_bytes, "max_packet_bytes", 16 * 1024 * 1024)):
            if integer(value, label, maximum=cap) == 0:
                raise ValueError(f"{label} must be positive")
        if max_packet_bytes > max_history_bytes:
            raise ValueError("packet bound exceeds history byte bound")
        self._max_providers, self._max_history = max_providers, max_history
        self._max_bytes, self._max_packet = max_history_bytes, max_packet_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._slots = {}
        self._history = deque()
        self._history_bytes = 0
        self._closed = False
        self._sealed = False

    def register(self, provider: SensorProvider):
        if not isinstance(provider.descriptor, SensorDescriptor):
            raise ValueError("typed sensor descriptor required")
        if not callable(getattr(provider, "read", None)) or not callable(getattr(provider, "close", None)):
            raise ValueError("passive read and close are required")
        with self._lock:
            if self._closed or self._sealed:
                raise SensorError("sensor hub closed or registry sealed")
            name = provider.descriptor.sensor_id
            if name in self._slots or len(self._slots) >= self._max_providers:
                raise ValueError("duplicate sensor or provider capacity exceeded")
            self._slots[name] = _Slot(provider)

    def seal(self):
        """Freeze the advertised registry before composing resource requirements."""
        with self._lock:
            self._sealed = True

    @property
    def descriptors(self):
        with self._lock:
            return tuple(slot.descriptor for slot in self._slots.values())

    def _admit(self, slot, observation, size):
        if type(observation) is not ObservationEnvelope:
            raise SensorError("provider did not return an immutable observation")
        descriptor = slot.descriptor
        for key in ("sensor_id", "source", "clock_domain", "measurement_kind", "model_identity_sha256"):
            if getattr(observation, key) != getattr(descriptor, key):
                raise SensorError(f"sensor {key} mismatch")
        if (observation.payload.modality != descriptor.modality
                or observation.payload.metadata.frame_id != descriptor.frame_id
                or observation.payload.metadata.calibration_id != descriptor.calibration_id):
            raise SensorError("sensor modality/frame/calibration mismatch")
        epoch = descriptor.epoch if slot.watermark is None else slot.watermark[0]
        if epoch is not None and observation.epoch != epoch:
            raise SensorError("sensor epoch mismatch; rebuild after reset")
        if slot.watermark is not None:
            if observation.sequence <= slot.watermark[1] or observation.capture_time_s <= slot.watermark[2]:
                raise SensorError("sensor replay or capture clock regression")
            if observation.received_monotonic_s < slot.watermark[3]:
                raise SensorError("sensor local receipt clock regressed")
        # Even a stale, future-stamped or oversized capture has been seen. It
        # cannot later be rejuvenated by presenting the same identity again.
        slot.watermark = (observation.epoch, observation.sequence, observation.capture_time_s,
                          observation.received_monotonic_s)
        try:
            age = observation.age_s(self._clock())
        except ValueError as exc:
            raise SensorError(str(exc)) from exc
        if age > descriptor.max_age_s:
            raise SensorError("stale sensor observation")
        if size > self._max_packet:
            raise SensorError("sensor packet exceeds byte bound")
        self._history.append((observation, size))
        self._history_bytes += size
        while len(self._history) > self._max_history or self._history_bytes > self._max_bytes:
            self._history_bytes -= self._history.popleft()[1]

    def read(self, sensor_id):
        with self._lock:
            if self._closed:
                raise SensorError("sensor hub closed")
            slot = self._slots.get(sensor_id)
            if slot is None:
                raise SensorError("unknown sensor")
            if slot.quarantined or slot.busy:
                raise SensorError("sensor quarantined or read already in flight")
            slot.busy = True
            done, result = threading.Event(), {}
            deadline = time.monotonic() + slot.descriptor.read_timeout_s
            slot.start()
            slot.requests.put_nowait((done, result))
        try:
            if not done.wait(max(0, deadline - time.monotonic())):
                with self._lock:
                    slot.quarantined = True
                raise SensorError("sensor read deadline expired; provider quarantined")
            if "error" in result:
                raise SensorError(result["error"])
            observation = result["value"]
            if type(observation) is not ObservationEnvelope:
                raise SensorError("provider did not return an immutable observation")
            size = len(json.dumps(observation.as_dict(), separators=(",", ":"), allow_nan=False).encode())
            with self._lock:
                if self._closed:
                    raise SensorError("sensor hub closed during read")
                if time.monotonic() > deadline:
                    raise SensorError("sensor read deadline expired during validation")
                self._admit(slot, observation, size)
            return observation
        finally:
            with self._lock:
                slot.busy = False

    def history(self, sensor_id=None):
        """Historical captures preserve their timestamps; this is not a fresh read."""
        with self._lock:
            return tuple(value for value, _ in self._history
                         if sensor_id is None or value.sensor_id == sensor_id)

    def close(self, timeout_s=0.25):
        timeout = number(timeout_s, "close timeout", minimum=0)
        if timeout > 5:
            raise ValueError("close timeout exceeds 5 seconds")
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closed = True
            slots = tuple(self._slots.values())
            for slot in slots:
                slot.closed.set()
                slot.start()
                try:
                    slot.requests.put_nowait(None)
                except queue.Full:
                    pass  # The queued read observes closed before doing I/O.
        for slot in slots:
            slot.thread.join(max(0, deadline - time.monotonic()))
        pending = [slot.descriptor.sensor_id for slot in slots if slot.thread.is_alive()]
        errors = {slot.descriptor.sensor_id: slot.close_error for slot in slots if slot.close_error}
        return {"ok": not pending and not errors, "pending_providers": pending, "errors": errors}
