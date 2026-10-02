"""Finite mobile commands with fail-closed feedback and priority cancellation.

No physics is implemented here. ``execution_ok`` records transport/control-loop
completion only; ``ok`` remains false and ``outcome`` unverified until a separate
physical postcondition checker judges the measured window. A kinematic mock
can never supply that evidence. Backend wall leases are mandatory: this client
watchdog cannot rescue a dead process or an unbounded/broken stop transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from types import MappingProxyType

from ..control.mobile_base import (
    BaseState, MobileBase, VelocityCommand, finite_real, identifier, nonnegative_int,
)


class BaseSafetyHarness:
    """Pure profile checks; no global configuration, arm or simulator imports."""

    REQUIRED = frozenset({
        "max_vx", "max_vy", "max_wz", "max_duration_s", "max_state_age_s",
        "max_no_progress_s", "max_wall_duration_s", "poll_interval_s",
        "turn_speed_rad_s", "turn_tolerance_rad", "max_turn_angle_rad",
    })

    def __init__(self, limits: dict):
        if not isinstance(limits, dict) or set(limits) != self.REQUIRED:
            raise ValueError(f"limits require exactly {sorted(self.REQUIRED)}")
        values = {key: finite_real(value, key) for key, value in limits.items()}
        for key, value in values.items():
            if value < 0 or (value == 0 and key != "max_vy"):
                raise ValueError(f"{key} must be positive (max_vy may be zero)")
        if not (values["poll_interval_s"] <= values["max_no_progress_s"] <= values["max_wall_duration_s"]):
            raise ValueError("require poll_interval_s <= max_no_progress_s <= max_wall_duration_s")
        if values["turn_speed_rad_s"] > values["max_wz"]:
            raise ValueError("turn_speed_rad_s exceeds max_wz")
        if not (values["turn_tolerance_rad"] < values["max_turn_angle_rad"] <= math.pi):
            raise ValueError("require turn_tolerance_rad < max_turn_angle_rad <= pi")
        self.limits = MappingProxyType(values)

    def validate_command(self, command: VelocityCommand) -> None:
        if not isinstance(command, VelocityCommand):
            raise ValueError("VelocityCommand required")
        for axis in ("vx", "vy", "wz"):
            if abs(getattr(command, axis)) > self.limits[f"max_{axis}"]:
                raise ValueError(f"{axis} exceeds explicit profile limit")
        if command.duration_s > self.limits["max_duration_s"]:
            raise ValueError("duration_s exceeds explicit profile limit")

    def validate_state(self, state: BaseState, *, robot_id: str, source: str,
                       measurement_kind: str, now: float, epoch=None, generation=None) -> None:
        if not isinstance(state, BaseState):
            raise ValueError("missing BaseState feedback")
        # Revalidate even subclasses/adapters which circumvent frozen dataclasses.
        BaseState.from_dict(state.as_dict())
        if (state.robot_id, state.source, state.measurement_kind) != (robot_id, source, measurement_kind):
            raise ValueError("feedback identity/source/measurement_kind mismatch")
        if epoch is not None and state.epoch != epoch:
            raise ValueError("feedback epoch changed; explicit reset required")
        if generation is not None and state.generation != generation:
            raise ValueError("feedback generation changed")
        age = now - state.received_monotonic_s
        if age < 0 or age + state.producer_age_s > self.limits["max_state_age_s"]:
            raise ValueError("stale or future-dated feedback")
        if state.fallen or state.controller_status not in ("ready", "active"):
            raise ValueError("fallen or unhealthy controller feedback")
        if state.latched:
            raise ValueError("backend stop is latched")

    def validate_transition(self, before: BaseState, after: BaseState) -> bool:
        """True only for a newer completed step AND increasing simulation time."""
        if (before.robot_id, before.source, before.epoch, before.measurement_kind) != (
                after.robot_id, after.source, after.epoch, after.measurement_kind):
            raise ValueError("feedback identity/epoch changed")
        ds, dt = after.step - before.step, after.sim_time_s - before.sim_time_s
        if ds < 0 or dt < 0 or (ds == 0) != (dt == 0):
            raise ValueError("inconsistent or regressing physics clock")
        if after.received_monotonic_s < before.received_monotonic_s:
            raise ValueError("client receipt clock regressed")
        return ds > 0


class _Cancelled(RuntimeError):
    pass


@dataclass
class _Operation:
    serial: int
    deadline: float
    cancel: threading.Event = field(default_factory=threading.Event)
    reason: str = "cancelled"


class SafeBase:
    """One active motion; stop never takes the execution or observation lock.

    The underlying backend MUST atomically fence command admission against
    generation+epoch, expose bounded independent transports, and own a wall
    lease. Client cancellation alone cannot prevent a late server delivery.
    """

    def __init__(self, raw: MobileBase, limits: dict):
        self.harness = BaseSafetyHarness(limits)
        self.raw = raw
        metadata = raw.metadata
        self._identity = {key: identifier(metadata[key], key)
                          for key in ("robot_id", "source", "measurement_kind")}
        if self._identity["measurement_kind"] not in ("physics", "hardware", "kinematic_mock"):
            raise ValueError("unknown metadata measurement_kind")
        self._gate = threading.Lock()
        self._motion = threading.Lock()
        self._serial = 0
        self._latched = False
        self._control_ops = 0
        self._active = None
        self._epoch = None

    @property
    def metadata(self):
        return dict(self._identity)

    @property
    def capabilities(self):
        return self.raw.capabilities

    @property
    def latched(self):
        with self._gate:
            return self._latched

    def connect(self) -> None:
        self.raw.connect()
        if self.latched:
            self.stop(latch=True)

    def disconnect(self) -> None:
        self.stop(latch=True)
        self.raw.disconnect()

    def get_state(self) -> BaseState:
        """Passive read, including fault states, not a motion authorization."""
        state = self.raw.get_state()
        if not isinstance(state, BaseState):
            raise ValueError("missing BaseState feedback")
        BaseState.from_dict(state.as_dict())
        if any(getattr(state, key) != value for key, value in self._identity.items()):
            raise ValueError("feedback identity mismatch")
        return state

    def _ack(self, result, *, epoch=None, generation=None) -> dict:
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError(f"backend refused or failed: {result!r}")
        for key in ("robot_id", "source"):
            if result.get(key) != self._identity[key]:
                raise ValueError(f"ACK {key} mismatch")
        identifier(result.get("epoch"), "ACK epoch")
        if epoch is not None and result["epoch"] != epoch:
            raise ValueError("ACK epoch mismatch")
        actual = nonnegative_int(result.get("generation"), "ACK generation")
        if generation is not None and actual != generation:
            raise ValueError("ACK generation mismatch")
        if type(result.get("latched")) is not bool:
            raise ValueError("ACK missing boolean latched")
        return dict(result)

    def _priority_stop(self, *, latch, reason, only_if=None) -> dict:
        if type(latch) is not bool:
            raise ValueError("latch must be boolean")
        with self._gate:
            if only_if is not None and self._active is not only_if:
                return {"ok": False, "error": "operation already ended"}
            self._serial += 1
            self._latched = self._latched or latch
            self._control_ops += 1
            if self._active is not None:
                self._active.reason = reason
                self._active.cancel.set()
        try:
            ack = self._ack(self.raw.stop(latch=latch))
            with self._gate:
                self._latched = self._latched or ack["latched"]
                effective_latch = self._latched
            if latch and not ack["latched"]:
                raise ValueError("backend failed to latch stop")
            return {**ack, "latched": effective_latch, "outcome": "unverified",
                    "reason": "stop acknowledged, physical rest unverified"}
        except Exception as error:
            with self._gate:
                self._latched = True
            return {"ok": False, "latched": True, "outcome": "unverified",
                    "error": str(error), "delivery_uncertain": True}
        finally:
            with self._gate:
                self._control_ops -= 1

    def stop(self, latch=True) -> dict:
        return self._priority_stop(latch=latch, reason="cancelled by stop")

    def reset_stop(self) -> dict:
        # Refuse, rather than queue, while any old motion/delivery is alive.
        with self._gate:
            if self._active is not None or self._control_ops:
                return {"ok": False, "error": "motion or control operation still active", "latched": self._latched}
            self._serial += 1
            serial = self._serial
            self._control_ops += 1
        try:
            ack = self._ack(self.raw.reset_stop())
            if ack["latched"]:
                raise ValueError("backend remains latched")
            with self._gate:
                raced = self._serial != serial
                if not raced:
                    self._latched = False
                    self._epoch = ack["epoch"]
            if raced:
                # A slow reset may have reached the server AFTER a priority stop.
                self.stop(latch=True)
                raise _Cancelled("stop raced reset; permission remains latched")
            return {**ack, "reason": "permission reset; no motion resumed"}
        except Exception as error:
            with self._gate:
                self._latched = True
            return {"ok": False, "latched": True, "error": str(error)}
        finally:
            with self._gate:
                self._control_ops -= 1

    def _check(self, op):
        if op.cancel.is_set():
            raise _Cancelled(op.reason)
        if time.monotonic() >= op.deadline:
            self._priority_stop(latch=True, reason="wall deadline expired", only_if=op)
            raise _Cancelled("wall deadline expired")

    def _read(self, op, *, epoch=None, generation=None):
        self._check(op)
        state = self.get_state()
        self._check(op)
        self.harness.validate_state(state, **self._identity, now=time.monotonic(),
                                    epoch=epoch, generation=generation)
        return state

    def _next(self, op, previous, *, generation):
        deadline = time.monotonic() + self.harness.limits["max_no_progress_s"]
        while True:
            self._check(op)
            op.cancel.wait(self.harness.limits["poll_interval_s"])
            state = self._read(op, epoch=previous.epoch, generation=generation)
            if self.harness.validate_transition(previous, state):
                return state
            if time.monotonic() >= deadline:
                raise ValueError("feedback clock did not advance")

    @staticmethod
    def _result(*, execution_ok=False, samples=(), command=None, error=None, **extra):
        data = [state.as_dict() for state in samples]
        result = {"ok": False, "execution_ok": execution_ok, "outcome": "unverified",
                  "reason": "independent physical postcondition required",
                  "command": command,
                  "measured": {"before": data[0] if data else None,
                               "after": data[-1] if data else None, "samples": data}, **extra}
        if error is not None:
            result["error"] = str(error)
        return result

    def walk_velocity(self, vx, vy, wz, duration_s) -> dict:
        try:
            command = VelocityCommand(vx, vy, wz, duration_s)
            self.harness.validate_command(command)
        except (TypeError, ValueError) as error:
            return self._result(error=error)
        return self._execute(command)

    def turn(self, angle_rad) -> dict:
        try:
            angle = finite_real(angle_rad, "angle_rad")
            limits = self.harness.limits
            if not limits["turn_tolerance_rad"] < abs(angle) <= limits["max_turn_angle_rad"]:
                raise ValueError("turn angle must exceed tolerance and stay within the profile limit")
            command = VelocityCommand(0., 0., math.copysign(limits["turn_speed_rad_s"], angle),
                                      limits["max_duration_s"])
            self.harness.validate_command(command)
        except (TypeError, ValueError) as error:
            return self._result(error=error)
        result = self._execute(command, angle=angle)
        result["requested_angle_rad"] = angle
        result.setdefault("measured_angle_rad", None)  # No feedback is not zero rotation.
        return result

    @staticmethod
    def _yaw(state):
        w, x, y, z = state.orientation_wxyz
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _execute(self, command, *, angle=None) -> dict:
        if not self._motion.acquire(blocking=False):
            return self._result(error="concurrent motion refused", command=command.as_dict())
        op = None
        timer = None
        samples = []
        dispatched = False
        ack_validated = False
        ack = None
        measured_angle = None
        try:
            with self._gate:
                if self._latched or self._control_ops:
                    return self._result(error="stop latched or control operation active", command=command.as_dict())
                op = _Operation(self._serial, time.monotonic() + self.harness.limits["max_wall_duration_s"])
                self._active = op
            timer = threading.Timer(self.harness.limits["max_wall_duration_s"],
                                    lambda: self._priority_stop(latch=True, reason="wall deadline expired", only_if=op))
            timer.daemon = True
            timer.name = "mobile-wall-watchdog"
            timer.start()
            first = self._read(op, epoch=self._epoch)
            samples.append(first)
            start = self._next(op, first, generation=first.generation)
            samples.append(start)
            if start.controller_status != "ready":
                raise ValueError("another command/controller is already active")
            with self._gate:
                self._epoch = start.epoch
            self._check(op)
            dispatched = True
            ack = self.raw.command_velocity(command, generation=start.generation)
            self._check(op)
            ack = self._ack(ack, epoch=start.epoch, generation=start.generation + 1)
            if ack.get("accepted") is not True or ack["latched"]:
                raise ValueError("command was not admitted")
            begin = finite_real(ack.get("start_sim_time_s"), "start_sim_time_s")
            end = finite_real(ack.get("end_sim_time_s"), "end_sim_time_s")
            if (begin < start.sim_time_s or end <= begin
                    or not math.isclose(end - begin, command.duration_s, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("command ACK has inconsistent simulation duration")
            ack_validated = True
            state = start
            measured_angle = 0.
            while state.sim_time_s < end:
                previous = state
                state = self._next(op, state, generation=ack["generation"])
                samples.append(state)
                if angle is not None:
                    delta = self._yaw(state) - self._yaw(previous)
                    # Unwrap MEASURED yaw, not command.wz * elapsed time.
                    measured_angle += math.atan2(math.sin(delta), math.cos(delta))
                    error = angle - measured_angle
                    if abs(error) <= self.harness.limits["turn_tolerance_rad"]:
                        break
                    if error * angle < 0:
                        raise ValueError("measured yaw overshot target tolerance")
            if angle is not None and abs(angle - measured_angle) > self.harness.limits["turn_tolerance_rad"]:
                raise ValueError("measured yaw did not reach target before simulation deadline")
            self._check(op)
            stop_ack = self._ack(self.raw.stop(latch=False), epoch=start.epoch,
                                 generation=ack["generation"] + 1)
            self._check(op)
            if stop_ack["latched"]:
                raise ValueError("backend stop unexpectedly latched")
            return self._result(execution_ok=True, samples=samples, command=command.as_dict(),
                                ack=ack, stop_ack=stop_ack,
                                **({"requested_angle_rad": angle, "measured_angle_rad": measured_angle}
                                   if angle is not None else {}))
        except _Cancelled as error:
            # Backend generation handles a command crossing the stop boundary.
            return self._result(samples=samples, command=command.as_dict(), error=error,
                                delivery_uncertain=dispatched and not ack_validated,
                                **({"measured_angle_rad": measured_angle} if angle is not None else {}))
        except Exception as error:
            stop_ack = self.stop(latch=True)
            return self._result(samples=samples, command=command.as_dict(), error=error,
                                delivery_uncertain=dispatched and not ack_validated, stop_ack=stop_ack,
                                **({"measured_angle_rad": measured_angle} if angle is not None else {}))
        except BaseException:
            # SIGINT/SystemExit must not leave an admitted command running;
            # invalidate it first, but never swallow process cancellation.
            self.stop(latch=True)
            raise
        finally:
            if timer is not None:
                timer.cancel()
            with self._gate:
                if self._active is op:
                    self._active = None
            if timer is not None:
                timer.join(timeout=self.harness.limits["max_wall_duration_s"])
            self._motion.release()
