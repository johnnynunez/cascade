"""Per-solve contracts and write fencing for a mounted fastening controller.

These contracts do not establish sensor fidelity. An admitted native adapter
must supply the declared complete contact registry, geometry and solved state.
The journal is passive: reading it never steps physics or refreshes capture time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import struct
import threading
import time

from ..sim.threading_verification import ThreadSample


class FasteningFault(RuntimeError):
    pass


class FasteningObservationAgeFault(FasteningFault):
    """Bounded capture/check evidence survives existing string-only fault paths.

    Nonfinite clock readings are represented as null, never as JSON NaN/Infinity.
    The payload is diagnostic only: it neither renews capture time nor supplies
    an independent observation. Returning a copy preserves the retained record.
    """

    def __init__(self, row, binding, limits, now, age, *, stage):
        if not isinstance(stage, str) or not stage or len(stage) > 64:
            raise ValueError("invalid observation check stage")
        reason = ("nonfinite_check_clock" if not math.isfinite(now) else
                  "nonfinite_age" if not math.isfinite(age) else
                  "future" if age < 0 else "stale")
        evidence = {
            "stage": stage, "reason": reason,
            "captured_monotonic_s": row.captured_monotonic_s,
            "checked_monotonic_s": now if math.isfinite(now) else None,
            "age_s": age if math.isfinite(age) else None,
            "max_observation_age_s": limits.max_observation_age_s,
            "step": row.step, "simulation_time_s": row.simulation_time_s,
            "generation": row.generation, "epoch": row.epoch,
            "binding_sha256": row.binding_sha256,
            "model_identity_sha256": binding.model_sha256,
        }
        self._observation_age_json = json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
        super().__init__("stale or future solved state; observation_age=" + self._observation_age_json)

    @property
    def observation_age(self):
        return json.loads(self._observation_age_json)


class FasteningRevoked(FasteningFault):
    """Normal priority cancellation reached an old proposed write."""


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"invalid {name}")
    return value


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"invalid {name}")
    return float(value)


def observed_effort_ceiling(cap):
    """Exact declared cap or its native float32 representation, whichever is larger.

    This fixed Factory contract requires the adapter's strict float32 channel
    checks. It is not an epsilon or permission to round/clamp observed values.
    Requested commands remain bounded by the original cap in Python precision.
    """
    return max(cap, struct.unpack("f", struct.pack("f", cap))[0])


def _vector(value, count, name):
    if isinstance(value, (str, bytes)) or len(value) != count:
        raise ValueError(f"invalid {name} dimension")
    return tuple(_number(v, name) for v in value)


def _index(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _quaternion(value):
    q = _vector(value, 4, "quaternion_xyzw")
    if abs(sum(v*v for v in q) - 1.) > .002:
        raise ValueError("quaternion must already be normalized")
    return q


@dataclass(frozen=True)
class FasteningBinding:
    """Full recipe digest includes physics, assets, controller and observer code."""

    model_sha256: str
    limits_sha256: str
    robot_id: str
    fixture_id: str
    fastener_id: str
    tool_id: str
    joint_names: tuple[str, ...]
    collider_names: tuple[str, ...]
    allowed_contact_pairs: tuple[tuple[str, str], ...]
    thread_contact_pair: tuple[str, str]
    tool_contact_pairs: tuple[tuple[str, str], ...]
    dt_s: float
    fixture_origin_m: tuple[float, float, float] = (.24, 0., 0.)
    fixture_recipe: str = "factory_m20_fixed_axis_v1"
    thread_pitch_m: float = .0025
    seat_contact_pair: tuple[str, str] | None = None

    def __post_init__(self):
        for digest in (self.model_sha256, self.limits_sha256):
            if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("invalid model/limits SHA-256")
        for name in ("robot_id", "fixture_id", "fastener_id", "tool_id"):
            _text(getattr(self, name), name)
        if len({self.fixture_id, self.fastener_id, self.tool_id}) != 3:
            raise ValueError("fixture, fastener and tool identities must differ")
        for name in ("joint_names", "collider_names"):
            values = tuple(_text(x, name) for x in getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ValueError(f"empty or duplicate {name}")
            object.__setattr__(self, name, values)
        pairs = []
        for pair in self.allowed_contact_pairs:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(self.collider_names):
                raise ValueError("invalid allowed contact pair")
            pairs.append(tuple(sorted(pair)))
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate allowed contact pair")
        object.__setattr__(self, "allowed_contact_pairs", tuple(sorted(pairs)))
        thread = tuple(sorted(self.thread_contact_pair))
        tools = tuple(sorted(tuple(sorted(pair)) for pair in self.tool_contact_pairs))
        if thread not in pairs or not tools or len(set(tools)) != len(tools) or thread in tools or any(pair not in pairs for pair in tools):
            raise ValueError("thread/tool witnesses must name disjoint admitted collider pairs")
        object.__setattr__(self, "thread_contact_pair", thread)
        object.__setattr__(self, "tool_contact_pairs", tools)
        if self.seat_contact_pair is not None:
            seat = tuple(sorted(self.seat_contact_pair))
            if (seat not in pairs or seat == thread or seat in tools
                    or len(set(seat).intersection(thread)) != 1
                    or any(set(seat).intersection(pair) != set(seat).intersection(thread) for pair in tools)):
                raise ValueError("seat witness must be a distinct admitted collider pair")
            object.__setattr__(self, "seat_contact_pair", seat)
        _number(self.dt_s, "dt_s", positive=True)
        object.__setattr__(self, "fixture_origin_m", _vector(self.fixture_origin_m, 3, "fixture origin"))
        if self.fixture_recipe not in {"factory_m20_fixed_axis_v1", "factory_m20_fixed_axis_margin_v2"} or self.thread_pitch_m != .0025:
            raise ValueError("only the fixed +Z Factory M20 2.5 mm recipe is implemented")
        if self.fixture_recipe == "factory_m20_fixed_axis_margin_v2" and self.fixture_origin_m != (.23, 0., 0.):
            raise ValueError("mounted margin recipe requires its declared fixture origin")

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FasteningLimits:
    """All arrays are ordered by binding.joint_names (arm and fixed-open jaw).

    Joint/effort limits must come from the actual imported model. Workspace and
    bottom height concern the complete declared arm/tool geometry, not its origin.
    The initial candidate uses the existing SO-101 0.8 rad/s and 0.02 rad margin.
    Further numerical values are explicit task budgets, not hardware certification.
    """

    joint_lower_rad: tuple[float, ...]
    joint_upper_rad: tuple[float, ...]
    joint_effort_nm: tuple[float, ...]
    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    table_z_m: float
    max_joint_speed_rad_s: float = .8
    joint_margin_rad: float = .02
    max_tracking_error_rad: float = .1
    spindle_effort_nm: float = .05
    max_rotational_speed_rad_s: float = 10.
    max_command_sim_s: float = 4.
    max_command_wall_s: float = 40.
    max_observation_age_s: float = .2
    rest_window_sim_s: float = .5
    rest_timeout_sim_s: float = 3.
    rest_timeout_wall_s: float = 30.
    rest_linear_speed_m_s: float = .001
    rest_angular_speed_rad_s: float = .02
    contact_load_threshold_n: float = 1e-6
    seating: object | None = None

    def __post_init__(self):
        n = len(self.joint_lower_rad)
        if not n:
            raise ValueError("joint limits cannot be empty")
        for key in ("joint_lower_rad", "joint_upper_rad", "joint_effort_nm"):
            object.__setattr__(self, key, _vector(getattr(self, key), n, key))
        for key in ("workspace_min_m", "workspace_max_m"):
            object.__setattr__(self, key, _vector(getattr(self, key), 3, key))
        _number(self.table_z_m, "table_z_m")
        for key, value in asdict(self).items():
            if key not in {"joint_lower_rad", "joint_upper_rad", "joint_effort_nm",
                           "workspace_min_m", "workspace_max_m", "table_z_m", "seating"}:
                _number(value, key, positive=True)
        if self.seating is not None:
            from .fastening_seat import SeatingLimits
            if (type(self.seating) is not SeatingLimits
                    or self.seating.minimum_loaded_effort_nm > self.spindle_effort_nm
                    or self.seating.motor_off_window_sim_s > self.rest_timeout_sim_s):
                raise ValueError("invalid distinct seating task limits")
        if any(lo + 2*self.joint_margin_rad >= hi for lo, hi in zip(
                self.joint_lower_rad, self.joint_upper_rad, strict=True)):
            raise ValueError("joint limits have no interior")
        if any(x <= 0 for x in self.joint_effort_nm):
            raise ValueError("joint effort caps must be positive")
        if any(lo >= hi for lo, hi in zip(self.workspace_min_m, self.workspace_max_m, strict=True)):
            raise ValueError("empty workspace")
        if self.rest_window_sim_s > self.rest_timeout_sim_s:
            raise ValueError("rest observation budget shorter than required window")

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SolvedPair:
    collider_a: str
    collider_b: str
    normal_force_n: float

    def __post_init__(self):
        _text(self.collider_a, "collider_a")
        _text(self.collider_b, "collider_b")
        if self.collider_a == self.collider_b or _number(self.normal_force_n, "normal_force_n") < 0:
            raise ValueError("invalid solved contact")


@dataclass(frozen=True)
class FasteningSolve:
    binding_sha256: str
    epoch: str
    generation: int
    step: int
    simulation_time_s: float
    captured_monotonic_s: float
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_effort_nm: tuple[float, ...]
    # Conservative world AABB of all moving arm/tool collision geometry.
    geometry_min_m: tuple[float, float, float]
    geometry_max_m: tuple[float, float, float]
    fastener_position_m: tuple[float, float, float]
    fastener_quaternion_xyzw: tuple[float, float, float, float]
    fixture_position_m: tuple[float, float, float]
    fixture_quaternion_xyzw: tuple[float, float, float, float]
    tool_position_m: tuple[float, float, float]
    tool_quaternion_xyzw: tuple[float, float, float, float]
    fastener_linear_speed_m_s: float
    fastener_angular_speed_rad_s: float
    tool_linear_speed_m_s: float
    tool_angular_speed_rad_s: float
    spindle_speed_rad_s: float
    spindle_effort_nm: float
    commanded_spindle_effort_nm: float
    contacts: tuple[SolvedPair, ...]
    thread_contacts: int
    tool_contacts: int
    # Adapter must have checked BOTH collision and solver buffers. Unknown
    # coverage has no FasteningSolve representation: publish a journal fault.
    collision_capacity: int
    collision_count: int
    solver_capacity: int
    solver_count: int

    def __post_init__(self):
        if not isinstance(self.binding_sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.binding_sha256):
            raise ValueError("invalid solve binding")
        _text(self.epoch, "epoch")
        for name in ("generation", "step", "thread_contacts", "tool_contacts",
                     "collision_capacity", "collision_count", "solver_capacity", "solver_count"):
            _index(getattr(self, name), name)
        if (not self.collision_capacity or not self.solver_capacity or
                self.collision_count >= self.collision_capacity or self.solver_count >= self.solver_capacity):
            raise ValueError("contact capacity reached or unknown")
        n = len(self.joint_position_rad)
        if not n:
            raise ValueError("missing joint readback")
        for name in ("joint_position_rad", "joint_velocity_rad_s", "joint_effort_nm"):
            object.__setattr__(self, name, _vector(getattr(self, name), n, name))
        for name in ("geometry_min_m", "geometry_max_m", "fastener_position_m",
                     "fixture_position_m", "tool_position_m"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        for name in ("fastener_quaternion_xyzw", "fixture_quaternion_xyzw", "tool_quaternion_xyzw"):
            object.__setattr__(self, name, _quaternion(getattr(self, name)))
        for name in ("simulation_time_s", "captured_monotonic_s", "fastener_linear_speed_m_s",
                     "fastener_angular_speed_rad_s", "tool_linear_speed_m_s", "tool_angular_speed_rad_s",
                     "spindle_speed_rad_s", "spindle_effort_nm", "commanded_spindle_effort_nm"):
            _number(getattr(self, name), name)
        for name in ("fastener_linear_speed_m_s", "fastener_angular_speed_rad_s",
                     "tool_linear_speed_m_s", "tool_angular_speed_rad_s"):
            if getattr(self, name) < 0:
                raise ValueError("speed magnitude cannot be negative")
        pairs = tuple(self.contacts)
        if any(not isinstance(x, SolvedPair) for x in pairs) or len(pairs) != self.solver_count:
            raise ValueError("solved contact records do not cover the full solver count")
        if self.thread_contacts + self.tool_contacts > len(pairs):
            raise ValueError("thread/tool contacts exceed contact records")
        object.__setattr__(self, "contacts", pairs)

    def thread_sample(self, binding):
        return ThreadSample(epoch=self.epoch, step=self.step, time_s=self.simulation_time_s,
            fastener_id=binding.fastener_id, fixture_id=binding.fixture_id,
            fastener_position_m=self.fastener_position_m,
            fastener_quaternion_xyzw=self.fastener_quaternion_xyzw,
            fixture_position_m=self.fixture_position_m,
            fixture_quaternion_xyzw=self.fixture_quaternion_xyzw,
            thread_contacts=self.thread_contacts, tool_contacts=self.tool_contacts)


def check_solve(row, binding, limits, now, *, epoch=None, previous=None, stage="check_solve"):
    """Shared safety/freshness checks; never fabricate coverage from empty data."""
    if not isinstance(row, FasteningSolve) or row.binding_sha256 != binding.sha256:
        raise FasteningFault("solve identity differs from admitted binding")
    if epoch is not None and row.epoch != epoch:
        raise FasteningFault("physics epoch changed")
    if (any(abs(a-b) > .0001 for a, b in zip(row.fixture_position_m, binding.fixture_origin_m, strict=True)) or
            any(abs(v) > 1e-6 for v in row.fixture_quaternion_xyzw[:3]) or
            abs(abs(row.fixture_quaternion_xyzw[3])-1.) > 1e-6):
        raise FasteningFault("fixture differs from declared fixed world +Z frame")
    age = now - row.captured_monotonic_s
    if not 0 <= age <= limits.max_observation_age_s:
        raise FasteningObservationAgeFault(row, binding, limits, now, age, stage=stage)
    if len(row.joint_position_rad) != len(binding.joint_names):
        raise FasteningFault("joint readback does not match binding")
    if previous is not None:
        if row.step != previous.step + 1 or not math.isclose(
                row.simulation_time_s - previous.simulation_time_s, binding.dt_s, rel_tol=1e-6, abs_tol=1e-9):
            raise FasteningFault("missing solve or discontinuous physical clock")
        if row.captured_monotonic_s < previous.captured_monotonic_s:
            raise FasteningFault("capture clock reversed")
    for q, lo, hi in zip(row.joint_position_rad, limits.joint_lower_rad, limits.joint_upper_rad, strict=True):
        if not lo + limits.joint_margin_rad <= q <= hi - limits.joint_margin_rad:
            raise FasteningFault("joint position outside model margin")
    if any(abs(v) > limits.max_joint_speed_rad_s for v in row.joint_velocity_rad_s):
        raise FasteningFault("measured joint speed exceeded limit")
    if any(abs(v) > observed_effort_ceiling(cap) for v, cap in zip(row.joint_effort_nm, limits.joint_effort_nm, strict=True)):
        raise FasteningFault("measured joint effort exceeded model cap")
    check_geometry(row.geometry_min_m, row.geometry_max_m, limits)
    if max(abs(row.spindle_speed_rad_s), row.fastener_angular_speed_rad_s) > limits.max_rotational_speed_rad_s:
        raise FasteningFault("measured spindle/fastener speed exceeded bound")
    if (abs(row.spindle_effort_nm) > observed_effort_ceiling(limits.spindle_effort_nm)
            or abs(row.commanded_spindle_effort_nm) > limits.spindle_effort_nm):
        raise FasteningFault("spindle effort exceeded limit")
    registry, permitted = set(binding.collider_names), set(binding.allowed_contact_pairs)
    thread_count = tool_count = 0
    for contact in row.contacts:
        pair = tuple(sorted((contact.collider_a, contact.collider_b)))
        if not set(pair) <= registry:
            raise FasteningFault("unknown collider in solved contact stream")
        if contact.normal_force_n > 0 and pair not in permitted:
            raise FasteningFault("forbidden solved contact: " + "/".join(pair))
        if contact.normal_force_n > limits.contact_load_threshold_n:
            thread_count += pair == binding.thread_contact_pair
            tool_count += pair in binding.tool_contact_pairs
    if (row.thread_contacts, row.tool_contacts) != (thread_count, tool_count):
        raise FasteningFault("thread/tool witnesses differ from solved collider pairs")
    if limits.seating is not None:
        from .fastening_seat import check_seating_solve
        check_seating_solve(row, binding)


def check_geometry(minimum, maximum, limits):
    for lo, hi, lower, upper in zip(minimum, maximum, limits.workspace_min_m,
                                   limits.workspace_max_m, strict=True):
        if not lower <= lo <= hi <= upper:
            raise FasteningFault("arm/tool collision geometry outside workspace")
    if minimum[2] < limits.table_z_m:
        raise FasteningFault("arm/tool collision geometry below table")


class SolveJournal:
    """One producer, passive readers, immutable records and sticky errors.

    Native and synthetic producers are explicitly distinguished by their owner;
    this journal alone is not a claim that a sensor contract has been validated.
    """

    def __init__(self, binding, *, capacity=20000):
        if type(capacity) is not int or capacity < 3:
            raise ValueError("invalid solve journal capacity")
        self.binding, self.capacity = binding, capacity
        self._rows = deque(maxlen=capacity)
        self._condition = threading.Condition()
        self._error = None

    def fail(self, reason):
        with self._condition:
            self._error = self._error or str(reason) or "producer failed without an error message"
            self._condition.notify_all()

    def publish(self, row):
        with self._condition:
            if self._error:
                raise FasteningFault(self._error)
            previous = self._rows[-1] if self._rows else None
            if (not isinstance(row, FasteningSolve) or row.binding_sha256 != self.binding.sha256 or
                    previous is not None and (row.epoch != previous.epoch or row.step != previous.step + 1 or
                    not math.isclose(row.simulation_time_s - previous.simulation_time_s,
                                     self.binding.dt_s, rel_tol=1e-6, abs_tol=1e-9))):
                self.fail("invalid identity/epoch/clock in producer stream")
                raise FasteningFault(self._error)
            self._rows.append(row)
            self._condition.notify_all()

    def read(self, after_step=None, *, timeout_s=0.):
        _number(timeout_s, "read timeout")
        if timeout_s < 0:
            raise ValueError("negative read timeout")
        with self._condition:
            self._condition.wait_for(lambda: self._error or bool(self._rows) and
                (after_step is None or self._rows[-1].step > after_step), timeout=timeout_s)
            if self._error:
                raise FasteningFault(self._error)
            if not self._rows:
                return ()
            if after_step is None:
                return (self._rows[-1],)
            _index(after_step, "reader cursor")
            if after_step < self._rows[0].step - 1:
                raise FasteningFault("reader lost solves to journal capacity")
            return tuple(row for row in self._rows if row.step > after_step)


@dataclass(frozen=True)
class FasteningPermit:
    binding_sha256: str
    epoch: str
    generation: int
    admission_step: int
    admission_time_s: float
    admitted_monotonic_s: float
    deadline_monotonic_s: float
    end_simulation_time_s: float
    operation: str = "turn"

    def __post_init__(self):
        if not isinstance(self.binding_sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.binding_sha256):
            raise ValueError("invalid permit binding")
        _text(self.epoch, "permit epoch")
        _index(self.generation, "permit generation")
        _index(self.admission_step, "admission step")
        for name in ("admission_time_s", "admitted_monotonic_s", "deadline_monotonic_s", "end_simulation_time_s"):
            if _number(getattr(self, name), name) < 0:
                raise ValueError("negative permit clock")
        if self.deadline_monotonic_s <= self.admitted_monotonic_s or self.end_simulation_time_s <= self.admission_time_s:
            raise ValueError("permit has no remaining time")
        if self.operation not in ("turn", "seat"):
            raise ValueError("unknown fastening permit operation")


@dataclass(frozen=True)
class FasteningUpload:
    """Authority actually uploaded BEFORE an interval, never a later latch."""

    generation: int
    before_step: int
    effort_nm: float
    started_monotonic_s: float
    completed_monotonic_s: float

    def __post_init__(self):
        _index(self.generation, "upload generation")
        _index(self.before_step, "upload solve")
        _number(self.effort_nm, "upload effort")
        for t in (self.started_monotonic_s, self.completed_monotonic_s):
            if _number(t, "upload clock") < 0:
                raise ValueError("negative upload clock")
        if self.completed_monotonic_s < self.started_monotonic_s:
            raise ValueError("reversed upload clock")


class FasteningWriteGuard:
    """Single-writer, per-control-write authority; stop never waits for a solve.

    apply() holds only the authorization/write lock across the final check and
    backend upload. Solver execution and observation/logging happen outside it.
    The backend writer must be bounded. Rejection latches stop; it does not
    invent a safe torque-off pose or claim that the body is at rest.
    """

    def __init__(self, binding, limits, *, clock=time.monotonic):
        if len(binding.joint_names) != len(limits.joint_lower_rad) or binding.limits_sha256 != limits.sha256:
            raise ValueError("joint or limits binding mismatch")
        if (limits.seating is None) != (binding.seat_contact_pair is None):
            raise ValueError("seating task and shoulder registry must be bound together")
        self.binding, self.limits, self.clock = binding, limits, clock
        self._lock = threading.RLock()
        self._generation, self._latched, self._closed = 0, True, False
        self._permit = None
        self._last_write_step = None
        self._previous_target = None

    @property
    def generation(self):
        with self._lock:
            return self._generation

    @property
    def current_permit(self):
        with self._lock:
            return self._permit

    def zero_hold(self, before_step, writer):
        """Owner emergency path: exact spindle zero, existing arm hold unchanged.

        This is permitted while latched/closed or without a solved observation.
        It cannot authorize a new arm target or claim physical rest. The owner
        must supply the bounded spindle-only writer, and report upload failure.
        """
        with self._lock:
            _index(before_step, "upload step")
            start = self.clock()
            try:
                writer(0.)
            except BaseException:
                if not self._latched:
                    self.stop()
                raise
            return FasteningUpload(self._generation, before_step, 0., start, self.clock())

    def reset_stop(self, row):
        with self._lock:
            if self._closed:
                raise FasteningFault("controller is closed")
            check_solve(row, self.binding, self.limits, self.clock(), stage="reset_stop")
            if (row.generation != self._generation or row.commanded_spindle_effort_nm != 0 or
                    max(abs(v) for v in row.joint_velocity_rad_s) > self.limits.rest_angular_speed_rad_s or
                    row.fastener_angular_speed_rad_s > self.limits.rest_angular_speed_rad_s or
                    row.tool_angular_speed_rad_s > self.limits.rest_angular_speed_rad_s or
                    abs(row.spindle_speed_rad_s) > self.limits.rest_angular_speed_rad_s or
                    row.spindle_effort_nm != 0. or
                    row.fastener_linear_speed_m_s > self.limits.rest_linear_speed_m_s or
                    row.tool_linear_speed_m_s > self.limits.rest_linear_speed_m_s):
                raise FasteningFault("reset needs fresh current-generation stopped observations")
            self._generation += 1
            self._latched = False
            self._permit = None
            return self._generation

    def admit(self, row, *, expected_generation, turns=1., direction="tighten"):
        with self._lock:
            if self._closed or self._latched or expected_generation != self._generation:
                raise FasteningFault("controller stopped or generation invalidated")
            if self._permit is not None:
                raise FasteningFault("another fastening command owns the controller")
            if type(turns) not in (float, int) or turns != 1. or direction != "tighten":
                raise FasteningFault("this mounted fixture admits exactly one tightening turn")
            return self._admit_interval(row, self.limits, "turn")

    def admit_seating(self, row, *, expected_generation):
        with self._lock:
            if self._closed or self._latched or expected_generation != self._generation:
                raise FasteningFault("controller stopped or generation invalidated")
            if self._permit is not None:
                raise FasteningFault("another fastening command owns the controller")
            if self.limits.seating is None:
                raise FasteningFault("shoulder seating is not configured")
            task = self.limits.seating
            available = (row.fastener_position_m[2]-row.fixture_position_m[2]
                -task.nut_half_height_m-task.shoulder_height_m)
            if available < task.minimum_turns*self.binding.thread_pitch_m:
                raise FasteningFault("fixed seating approach requires the initial pre-engaged nut height")
            return self._admit_interval(row, self.limits.seating, "seat")

    def _admit_interval(self, row, task_limits, operation):
        # Caller holds the same single-writer lock for the complete admission.
        now = self.clock()
        check_solve(row, self.binding, self.limits, now, stage=operation+"_admission")
        if row.generation != self._generation or not row.thread_contacts or not row.tool_contacts:
            raise FasteningFault("fresh same-generation pre-engaged contacts required")
        self._generation += 1
        self._permit = FasteningPermit(self.binding.sha256, row.epoch, self._generation,
            row.step, row.simulation_time_s, now, now + task_limits.max_command_wall_s,
            row.simulation_time_s + task_limits.max_command_sim_s, operation)
        self._previous_target = row.joint_position_rad
        self._last_write_step = None
        return self._permit

    def stop(self):
        with self._lock:
            self._generation += 1
            self._latched = True
            self._permit = None
            return {"ok": True, "generation": self._generation, "latched": True,
                    "physical_stop_verified": False, "accepted_monotonic_s": self.clock()}

    def close(self):
        with self._lock:
            receipt = self.stop()
            self._closed = True
            return receipt

    def apply(self, row, target_q, target_bounds, spindle_effort_nm, writer, *, permit):
        """Backend passes FK collision bounds for *this exact* target_q.

        Geometry computation must be from the identity-bound model, outside this
        critical section; the native adapter is responsible for that binding.
        No boolean 'safe' argument can bypass joint/geometry/contact checks.
        """
        with self._lock:
            try:
                if (self._closed or self._latched or permit is not self._permit or
                        permit is None or permit.generation != self._generation):
                    raise FasteningRevoked("write permission revoked")
                now = self.clock()
                if now >= permit.deadline_monotonic_s or row.simulation_time_s >= permit.end_simulation_time_s:
                    raise FasteningFault("command lease expired")
                check_solve(row, self.binding, self.limits, now, epoch=permit.epoch,
                            stage="control_upload")
                if row.step < permit.admission_step or not math.isclose(row.simulation_time_s,
                        permit.admission_time_s+(row.step-permit.admission_step)*self.binding.dt_s,
                        rel_tol=1e-6, abs_tol=1e-9):
                    raise FasteningFault("control clock differs from admission")
                # The admission row predates the first control write. Later rows
                # must name this exact command generation and every solve.
                if row.step != permit.admission_step and row.generation != permit.generation:
                    raise FasteningFault("solved generation mismatch")
                if self._last_write_step is not None and row.step != self._last_write_step + 1:
                    raise FasteningFault("control write skipped or repeated a solve")
                target = _vector(target_q, len(self.binding.joint_names), "joint target")
                for q, current, prior, lo, hi in zip(target, row.joint_position_rad,
                        self._previous_target, self.limits.joint_lower_rad,
                        self.limits.joint_upper_rad, strict=True):
                    if not lo + self.limits.joint_margin_rad <= q <= hi - self.limits.joint_margin_rad:
                        raise FasteningFault("target outside joint margin")
                    if abs(q-current) > self.limits.max_tracking_error_rad:
                        raise FasteningFault("joint following error exceeds bound")
                    if abs(q-prior) > self.limits.max_joint_speed_rad_s*self.binding.dt_s + 1e-9:
                        raise FasteningFault("joint target step exceeds velocity limit")
                check_geometry(_vector(target_bounds[0], 3, "target bounds"),
                               _vector(target_bounds[1], 3, "target bounds"), self.limits)
                effort = _number(spindle_effort_nm, "spindle command")
                if abs(effort) > self.limits.spindle_effort_nm:
                    raise FasteningFault("requested spindle effort exceeds cap")
                writer(target, effort)
                self._last_write_step = row.step
                self._previous_target = target
                return FasteningUpload(self._generation, row.step, effort, now, self.clock())
            except BaseException:
                # A concurrent stop already issued its causal ACK. Preserve
                # that generation when the revoked in-flight write reaches us.
                if not self._latched:
                    self.stop()
                raise


class FasteningController:
    """Command endpoint shared by a domain and its single native solve owner.

    Construction, requests and passive reads never advance physics. The owner
    calls accept_solve after every solve and guard.apply around every active
    upload. A separate shutdown callback must join that owner and preserve its
    final receipt; no controller destructor stops somebody else's process.
    """

    def __init__(self, binding, limits, journal, *, synthetic, close_owner, clock=time.monotonic):
        if type(synthetic) is not bool or journal.binding != binding or not callable(close_owner):
            raise ValueError("explicit producer identity, synthetic flag and owner closure required")
        self.binding, self.limits, self.journal = binding, limits, journal
        self.synthetic, self.close_owner, self.clock = synthetic, close_owner, clock
        self.guard = FasteningWriteGuard(binding, limits, clock=clock)
        self._previous = None

    @property
    def generation(self):
        return self.guard.generation

    def accept_solve(self, row):
        try:
            check_solve(row, self.binding, self.limits, self.clock(),
                        epoch=None if self._previous is None else self._previous.epoch,
                        previous=self._previous, stage="controller_accept_solve")
            self.journal.publish(row)
            self._previous = row
        except Exception as exc:
            self.guard.stop()
            self.journal.fail(str(exc))
            raise

    def _latest(self):
        rows = self.journal.read()
        if not rows:
            raise FasteningFault("no solved controller observation")
        return rows[-1]

    def request_turn(self, **arguments):
        return self.guard.admit(self._latest(), **arguments)

    def request_seating(self, **arguments):
        return self.guard.admit_seating(self._latest(), **arguments)

    def stop(self):
        return self.guard.stop()

    def reset_stop(self):
        return {"ok": True, "generation": self.guard.reset_stop(self._latest()), "latched": False}

    def close(self):
        stop = self.guard.close()
        closure = self.close_owner()
        return {"ok": isinstance(closure, dict) and closure.get("ok") is True,
                "stop": stop, "owner": closure, "physical_stop_verified": False}
