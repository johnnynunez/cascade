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
    admission_check: object = None


class SafeBase:
    """One active motion; stop never takes the execution or observation lock.

    The underlying backend MUST atomically fence command admission against
    generation+epoch, expose bounded independent transports, and own a wall
    lease. Client cancellation alone cannot prevent a late server delivery.
    """

    def __init__(self, raw: MobileBase, limits: dict, *, distance_control=None,
                 turn_control=None, support_contract=None):
        self.harness = BaseSafetyHarness(limits)
        self.raw = raw
        self.distance_control = self._distance_config(distance_control)
        self.turn_control = self._turn_config(turn_control)
        self._support_contract = None
        if (self.distance_control is not None or self.turn_control is not None) and support_contract is not None:
            from ..control.mobile_support import support_contract as validate_support_contract
            self._support_contract = validate_support_contract(support_contract)
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
        caps = self.raw.capabilities
        if self.distance_control is not None and "walk_velocity" in caps:
            caps = caps | {"walk_distance"}
        return frozenset(caps)

    def _distance_config(self, value):
        if value is None:
            return None
        keys = {"speed_m_s", "max_distance_m", "tolerance_m", "max_lateral_drift_m",
                "max_heading_drift_rad", "min_height_m", "max_tilt_rad"}
        # The admission pair (B74) is the one optional extension, both or neither.
        optional = set(self._ADMISSION_KEYS)
        if not isinstance(value, dict) or not keys <= set(value) <= keys | optional:
            raise ValueError("distance_control requires the exact explicit contract")
        admission = {k: value[k] for k in self._ADMISSION_KEYS if k in value}
        result = {k: finite_real(v, k) for k, v in value.items() if k in keys}
        if any(v <= 0 for v in result.values()):
            raise ValueError("distance control limits must be positive")
        if result["speed_m_s"] > self.harness.limits["max_vx"]:
            raise ValueError("distance control speed exceeds max_vx")
        if not result["tolerance_m"] < result["max_distance_m"]:
            raise ValueError("distance tolerance must be below the maximum distance")
        if result["max_heading_drift_rad"] >= math.pi or result["max_tilt_rad"] >= math.pi / 2:
            raise ValueError("distance control attitude bounds must be unambiguous and upright")
        if admission:
            result.update(self._admission_config(admission, result))
        return MappingProxyType(result)

    _ADMISSION_KEYS = ("max_admission_lateral_m", "max_admission_heading_rad")

    def _admission_config(self, value, bounds):
        """Opt-in (B74) bound on the change from the last pre-ACK sample to the baseline.

        Both keys or neither; an explicit null pair keeps it OFF, so an
        ``extends:`` child can A/B it away. The segment precedes the walk, so its
        bound may not exceed the walk's own lateral/heading drift bound.
        """
        if set(value) != set(self._ADMISSION_KEYS):
            raise ValueError("distance_control admission bounds require both " + " and ".join(self._ADMISSION_KEYS))
        if all(v is None for v in value.values()):
            return {}
        result = {k: finite_real(v, k) for k, v in value.items()}
        if not (0 < result["max_admission_lateral_m"] <= bounds["max_lateral_drift_m"]
                and 0 < result["max_admission_heading_rad"] <= bounds["max_heading_drift_rad"]):
            raise ValueError("distance control admission bounds must be positive and within the drift bounds")
        return result

    def _turn_config(self, value):
        if value is None:
            return None
        keys = {"max_translation_path_m", "min_height_m", "max_tilt_rad"}
        # ``goal_ramp`` is the one optional key (opt-in, profile-scoped); every
        # other field remains the exact explicit contract.
        if not isinstance(value, dict) or not keys <= set(value) <= keys | {"goal_ramp"}:
            raise ValueError("turn_control requires the exact explicit contract")
        result: dict = {k: finite_real(v, k) for k, v in value.items() if k != "goal_ramp"}
        if any(v <= 0 for v in result.values()) or result["max_tilt_rad"] >= math.pi / 2:
            raise ValueError("turn control bounds must be positive and upright")
        if "goal_ramp" in value:
            # An explicit null keeps the ramp OFF, so an ``extends:`` child can A/B it away.
            result["goal_ramp"] = self._goal_ramp_config(value["goal_ramp"])
        return MappingProxyType(result)

    def _goal_ramp_config(self, value):
        """Deceleration of the ADMITTED turn rate before the goal; never above it."""
        if value is None:
            return None
        keys = {"decel_rad_s2", "min_rate_rad_s", "rate_step_rad_s"}
        # ``time_budget`` (B29b) is the one optional key; an explicit null keeps it off.
        if not isinstance(value, dict) or not keys <= set(value) <= keys | {"time_budget"}:
            raise ValueError("turn_control.goal_ramp requires exactly " + ", ".join(sorted(keys))
                             + " (and optionally time_budget)")
        ramp: dict = {k: finite_real(v, k) for k, v in value.items() if k != "time_budget"}
        if any(v <= 0 for v in ramp.values()):
            raise ValueError("goal ramp values must be positive")
        top = self.harness.limits["turn_speed_rad_s"]
        if not ramp["min_rate_rad_s"] < top:
            raise ValueError("goal ramp floor must stay below the admitted turn speed")
        if ramp["rate_step_rad_s"] > top - ramp["min_rate_rad_s"]:
            raise ValueError("goal ramp rate step exceeds the whole ramp")
        if "time_budget" in value:
            ramp["time_budget"] = self._time_budget_config(value["time_budget"])
        return MappingProxyType(ramp)

    def _time_budget_config(self, value):
        """Never decelerate below the rate the remaining yaw needs in the admitted time left."""
        if value is None:
            return None
        keys = {"reserve_s", "tracking"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("turn_control.goal_ramp.time_budget requires exactly reserve_s, tracking")
        budget = {k: finite_real(v, k) for k, v in value.items()}
        if not 0 < budget["reserve_s"] < self.harness.limits["max_duration_s"]:
            raise ValueError("time budget reserve must be positive and leave part of the command")
        # Above 1 would assume the robot outruns its command, i.e. permit LATER deceleration.
        if not 0 < budget["tracking"] <= 1:
            raise ValueError("time budget tracking must be in (0, 1]")
        return MappingProxyType(budget)

    def _goal_ramp(self):
        return None if self.turn_control is None else self.turn_control.get("goal_ramp")

    def _distance_state(self, state, *, require_load=False):
        self._geometric_state(state, self.distance_control, "distance", require_load=require_load)

    def _geometric_state(self, state, limits, label, *, require_load=False):
        """Control-side veto only; independent observation still judges success."""
        _, x, y, _ = state.orientation_wxyz
        tilt = math.acos(max(-1., min(1., 1 - 2 * (x*x + y*y))))
        if state.position_world[2] < limits["min_height_m"] or tilt > limits["max_tilt_rad"]:
            raise ValueError(f"{label} control posture bound exceeded")
        if state.measurement_kind == "kinematic_mock":
            return  # Software fixture, never independently confirmed physics.
        contract, support = self._support_contract, state.support
        if contract is None or support is None or support.status != "known":
            raise ValueError(f"{label} control requires known solved contact evidence")
        if state.model_identity_sha256 != contract["model_identity_sha256"]:
            raise ValueError(f"{label} control support identity mismatch")
        robot = set(contract["robot_shapes"])
        feet, ground = set(contract["foot_shapes"]), set(contract["ground_shapes"])
        gravity = contract["gravity_world_m_s2"]
        gravity_norm = math.hypot(*gravity)
        up = [-v / gravity_norm for v in gravity]
        loaded = False
        for contact in support.contacts:
            a, b = contact.shape_a, contact.shape_b
            if (a in robot) == (b in robot):
                continue
            body, external = (a, b) if a in robot else (b, a)
            if body not in feet or external not in ground:
                raise ValueError(f"{label} control forbidden external robot contact")
            sign = -1 if a in robot else 1
            upward = sign * sum(f * u for f, u in zip(contact.force_on_b_world_n, up))
            normal_up = sign * sum(n * u for n, u in zip(contact.normal_a_to_b_world, up))
            loaded |= contact.normal_force_n > 0 and upward > 0 and normal_up > 0
        if require_load and not loaded:
            raise ValueError(f"{label} control preflight requires positive solved sole support")

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
        # A newly revoked external authority has not necessarily delivered stop.
        # Retain the existing generation fence and stop before retiring this op.
        if op.admission_check is not None and op.admission_check() is not True:
            self._priority_stop(latch=True, reason="navigation authority expired or cancelled", only_if=op)
            raise _Cancelled("navigation authority expired or cancelled")
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

    def walk_velocity(self, vx, vy, wz, duration_s, *, admission_check=None) -> dict:
        try:
            command = VelocityCommand(vx, vy, wz, duration_s)
            self.harness.validate_command(command)
        except (TypeError, ValueError) as error:
            return self._result(error=error)
        return self._execute(command, admission_check=admission_check)

    def walk_distance(self, distance_m, *, admission_check=None) -> dict:
        """Signed body-frame travel, terminated by feedback, never vx*time.

        The profile's internal policy command is reported as ``command``. It
        is not an assertion of actual walking speed or a remapping of a caller's
        velocity request. The original velocity skill retains its own meaning.
        """
        try:
            if self.distance_control is None or "walk_distance" not in self.capabilities:
                raise ValueError("walk_distance is not configured for this base")
            distance = finite_real(distance_m, "distance_m")
            bounds = self.distance_control
            if not bounds["tolerance_m"] < abs(distance) <= bounds["max_distance_m"]:
                raise ValueError("distance must exceed tolerance and stay within the profile limit")
            command = VelocityCommand(math.copysign(bounds["speed_m_s"], distance), 0., 0.,
                                      self.harness.limits["max_duration_s"])
            self.harness.validate_command(command)
        except (TypeError, ValueError) as error:
            return self._result(error=error)
        result = self._execute(command, distance=distance, admission_check=admission_check)
        result["requested_distance_m"] = distance
        result.setdefault("measured_distance_m", None)
        return result

    def turn(self, angle_rad, *, admission_check=None) -> dict:
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
        if self._goal_ramp() is not None and getattr(self.raw, "velocity_scaling", False) is not True:
            # Refused before any read or command: this profile's turn decelerates
            # inside its admission, and a backend that cannot do that would cut
            # the full rate to zero at the goal instead (the measured settle fault).
            result = self._result(error="turn_control.goal_ramp requires backend velocity scaling, "
                                        "which this backend does not advertise", command=command.as_dict())
        else:
            result = self._execute(command, angle=angle, admission_check=admission_check)
        result["requested_angle_rad"] = angle
        result.setdefault("measured_angle_rad", None)  # No feedback is not zero rotation.
        return result

    @staticmethod
    def _yaw(state):
        w, x, y, z = state.orientation_wxyz
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @classmethod
    def _body_increment(cls, previous, state):
        """Measured (forward, lateral, yaw) step at the midpoint heading, wrap-safe across +/-pi."""
        yaw = cls._yaw(previous)
        delta = cls._yaw(state) - yaw
        dyaw = math.atan2(math.sin(delta), math.cos(delta))
        heading = yaw + dyaw / 2
        dx, dy = (state.position_world[i] - previous.position_world[i] for i in (0, 1))
        return math.cos(heading)*dx + math.sin(heading)*dy, -math.sin(heading)*dx + math.cos(heading)*dy, dyaw

    def _admission_drift(self, previous, state, record):
        """Opt-in (B74) veto on the segment the distance baseline would otherwise absorb.

        Integrates the lateral/heading change from the last pre-ACK sample through
        every delivery sample to the baseline with the walk's own increments, and
        checks it at each sample like the walk's drift veto. Forward delivery
        motion is neither bounded here nor ever credited to ``measured_distance``.
        """
        _, lateral, dyaw = self._body_increment(previous, state)
        record["measured_admission_lateral_m"] += lateral
        record["measured_admission_heading_rad"] += dyaw
        bounds = self.distance_control
        if (abs(record["measured_admission_lateral_m"]) > bounds["max_admission_lateral_m"]
                or abs(record["measured_admission_heading_rad"]) > bounds["max_admission_heading_rad"]):
            raise ValueError("distance control admission change exceeded before the distance baseline")

    def _ramp_turn(self, op, ramp, state, remaining, rate, ack, updates, end, budget_record):
        """Lower the ADMITTED turn rate on the measured remaining yaw; never raise it.

        ``|wz| = max(min_rate, min(previous, sqrt(2 * decel * (|remaining| - tolerance))))``,
        sign kept by the admitted command: the deceleration aims at the tolerance
        boundary where the unchanged zero-twist stop fires, and the floor keeps the
        robot progressing until then. Only the freshly validated advancing sample in
        hand is used; nothing is extrapolated between samples, and a stale or missing
        state has already failed closed in ``_read`` before this runs. A new rate is
        sent only when it is ``rate_step`` lower (or reaches the floor), inside the same
        admission: the backend scales the admitted twist, never re-admits or extends it.

        Optional ``time_budget`` (B29b): the distance term may not go below
        ``(|remaining| - tolerance) / (tracking * (end - sim_time - reserve))``, the
        constant rate that still covers the remaining measured yaw in the ADMITTED
        command time left (ACK ``end_sim_time_s`` minus this sample's sim time), at the
        configured tracking, keeping ``reserve_s``; inside the reserve the rate is held.
        It reads no yaw RATE, so a transient dip moves it only by the yaw it really
        cost. It can only withhold a deceleration (``min(previous, ...)`` still caps
        it), never raise the rate, extend the command or skip the zero-twist stop.
        """
        limits = self.harness.limits
        distance = max(0., abs(remaining) - limits["turn_tolerance_rad"])
        desired = math.sqrt(2 * ramp["decel_rad_s2"] * distance)
        budget = ramp.get("time_budget")
        left = need = None
        if budget is not None:
            left = end - state.sim_time_s
            usable = left - budget["reserve_s"]
            need = distance / (budget["tracking"] * usable) if usable > 0 else math.inf
            if max(ramp["min_rate_rad_s"], min(rate, need)) > max(ramp["min_rate_rad_s"], min(rate, desired)):
                # The time left, not the distance left, sets this sample's rate.
                budget_record["held_samples"] += 1
                if budget_record["first_held"] is None:
                    budget_record["first_held"] = {
                        "step": state.step, "sim_time_s": state.sim_time_s, "remaining_rad": remaining,
                        "command_time_left_s": left, "distance_rate_rad_s": desired,
                        "budget_rate_rad_s": need if math.isfinite(need) else None, "rate_rad_s": rate}
            desired = max(desired, need)
        target = max(ramp["min_rate_rad_s"], min(rate, desired))
        if rate - target < ramp["rate_step_rad_s"] and not target == ramp["min_rate_rad_s"] < rate:
            return rate
        self._check(op)
        reply = self.raw.scale_velocity(target / limits["turn_speed_rad_s"], generation=ack["generation"])
        reply = self._ack(reply, epoch=ack["epoch"], generation=ack["generation"])
        if reply.get("accepted") is not True or reply["latched"]:
            raise ValueError("turn rate scaling was not accepted inside the admitted command")
        update = {"step": state.step, "sim_time_s": state.sim_time_s, "remaining_rad": remaining, "rate_rad_s": target}
        if budget is not None:  # an update is only ever sent outside the reserve, so ``need`` is finite here
            update.update(command_time_left_s=left, budget_rate_rad_s=need)
        updates.append(update)
        self._check(op)
        return target

    def _execute(self, command, *, angle=None, distance=None, admission_check=None) -> dict:
        if not self._motion.acquire(blocking=False):
            return self._result(error="concurrent motion refused", command=command.as_dict())
        op = None
        timer = None
        samples = []
        dispatched = False
        ack_validated = False
        ack = None
        measured_angle = None
        measured_distance = None
        lateral = 0.
        distance_baseline = None
        turn_baseline = None
        translation_path = 0.
        # Opt-in (distance_control.max_admission_*): change from the last pre-ACK
        # sample to the distance baseline; None until a command is admitted.
        admission = ({"measured_admission_lateral_m": None, "measured_admission_heading_rad": None}
                     if distance is not None and "max_admission_lateral_m" in self.distance_control else {})
        # Opt-in (turn_control.goal_ramp): the admitted |wz| only ever decreases.
        ramp = self._goal_ramp() if angle is not None else None
        rate = abs(command.wz)
        rate_updates = []
        ramped: dict = {"turn_rate_updates": rate_updates} if ramp is not None else {}
        # Opt-in (goal_ramp.time_budget): how often the command time left withheld a deceleration.
        budget_record = {"held_samples": 0, "first_held": None}
        if ramp is not None and ramp.get("time_budget") is not None:
            ramped["turn_rate_budget"] = budget_record
        try:
            with self._gate:
                if self._latched or self._control_ops:
                    return self._result(error="stop latched or control operation active", command=command.as_dict())
                op = _Operation(self._serial, time.monotonic() + self.harness.limits["max_wall_duration_s"],
                                admission_check=admission_check)
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
            if distance is not None:
                self._distance_state(start, require_load=True)
            if angle is not None and self.turn_control is not None:
                self._geometric_state(start, self.turn_control, "turn", require_load=True)
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
            if admission:
                admission.update(measured_admission_lateral_m=0., measured_admission_heading_rad=0.)
            while state.sim_time_s < end:
                previous = state
                state = self._next(op, state, generation=ack["generation"])
                samples.append(state)
                if distance is not None:
                    self._distance_state(state)
                    # Inspect late samples for safety, but never count motion
                    # first observed after the admitted command expired.
                    if state.sim_time_s > end:
                        break
                    # Exclude all preflight/ACK delivery drift. The first
                    # completed admitted state is an observed baseline, never
                    # positive travel credit. Subsequent increments use measured
                    # midpoint heading, including wrap across +/-pi.
                    if distance_baseline is None:
                        if admission:
                            # Opt-in: the lateral/heading part of that drift is
                            # bounded; forward delivery still earns nothing.
                            self._admission_drift(previous, state, admission)
                        if state.sim_time_s <= begin:
                            continue
                        distance_baseline = state
                        measured_distance = measured_angle = 0.
                        continue
                    forward, side, dyaw = self._body_increment(previous, state)
                    measured_distance += forward
                    lateral += side
                    measured_angle += dyaw
                    bounds = self.distance_control
                    if abs(lateral) > bounds["max_lateral_drift_m"] or abs(measured_angle) > bounds["max_heading_drift_rad"]:
                        raise ValueError("distance control lateral/heading drift exceeded")
                    error = distance - measured_distance
                    if abs(error) <= bounds["tolerance_m"]:
                        break
                    if error * distance < 0:
                        raise ValueError("measured distance overshot target tolerance")
                if angle is not None:
                    # Retain every observed 3D segment, including delivery and
                    # late motion. Returning to the origin cannot erase travel.
                    translation_path += math.dist(previous.position_world, state.position_world)
                    if self.turn_control is not None:
                        self._geometric_state(state, self.turn_control, "turn")
                        if translation_path > self.turn_control["max_translation_path_m"]:
                            raise ValueError("turn control translation path exceeded")
                    # Match distance admission: delivery/preflight rotation is
                    # not task progress, and a late sample cannot complete it.
                    if state.sim_time_s > end:
                        break
                    if turn_baseline is None:
                        if state.sim_time_s <= begin:
                            continue
                        turn_baseline = state
                        continue
                    delta = self._yaw(state) - self._yaw(previous)
                    # Unwrap MEASURED yaw, not command.wz * elapsed time.
                    measured_angle += math.atan2(math.sin(delta), math.cos(delta))
                    error = angle - measured_angle
                    if abs(error) <= self.harness.limits["turn_tolerance_rad"]:
                        break
                    if error * angle < 0:
                        raise ValueError("measured yaw overshot target tolerance")
                    if ramp is not None:
                        rate = self._ramp_turn(op, ramp, state, error, rate, ack, rate_updates, end, budget_record)
            if angle is not None and abs(angle - measured_angle) > self.harness.limits["turn_tolerance_rad"]:
                raise ValueError("measured yaw did not reach target before simulation deadline")
            if distance is not None and (measured_distance is None or
                    abs(distance - measured_distance) > self.distance_control["tolerance_m"]):
                raise ValueError("measured distance did not reach target before simulation deadline")
            self._check(op)
            stop_ack = self._ack(self.raw.stop(latch=False), epoch=start.epoch,
                                 generation=ack["generation"] + 1)
            self._check(op)
            if stop_ack["latched"]:
                raise ValueError("backend stop unexpectedly latched")
            return self._result(execution_ok=True, samples=samples, command=command.as_dict(),
                                ack=ack, stop_ack=stop_ack,
                                **({"requested_distance_m": distance, "measured_distance_m": measured_distance,
                                    "distance_baseline": distance_baseline.as_dict(),
                                    "measured_lateral_m": lateral, "measured_heading_rad": measured_angle}
                                   if distance is not None else {}),
                                **admission,
                                **({"requested_angle_rad": angle, "measured_angle_rad": measured_angle,
                                    "measured_translation_path_m": translation_path}
                                   if angle is not None else {}),
                                **({**ramped, "commanded_rate_at_stop_rad_s": rate} if ramped else {}))
        except _Cancelled as error:
            # Backend generation handles a command crossing the stop boundary.
            return self._result(samples=samples, command=command.as_dict(), error=error,
                                delivery_uncertain=dispatched and not ack_validated,
                                **({"ack": ack} if ack_validated else {}),
                                **({"measured_distance_m": measured_distance,
                                    "distance_baseline": distance_baseline.as_dict() if distance_baseline else None}
                                   if distance is not None else {}),
                                **admission,
                                **({"measured_angle_rad": measured_angle,
                                    "measured_translation_path_m": translation_path} if angle is not None else {}),
                                **ramped)
        except Exception as error:
            stop_ack = self.stop(latch=True)
            return self._result(samples=samples, command=command.as_dict(), error=error,
                                delivery_uncertain=dispatched and not ack_validated, stop_ack=stop_ack,
                                **({"ack": ack} if ack_validated else {}),
                                **({"measured_distance_m": measured_distance,
                                    "distance_baseline": distance_baseline.as_dict() if distance_baseline else None}
                                   if distance is not None else {}),
                                **admission,
                                **({"measured_angle_rad": measured_angle,
                                    "measured_translation_path_m": translation_path} if angle is not None else {}),
                                **ramped)
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
