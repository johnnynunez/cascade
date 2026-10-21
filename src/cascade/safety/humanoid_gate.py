"""Humanoid safety gate: every control step of a learned whole-body policy is vetted.

A learned loco-manipulation policy (the GR00T executor of ``humanoid_locomanip``) proposes one
action per control step. The simulator owner (the Arena bridge) sends the MEASURED state the step
starts from and the PROPOSED action; this gate answers PASS (the action may be applied) or STOP
(it is not applied; the owner switches to its hold command and ends the segment). A STOP latches
until an explicit ``reset()``; nothing here retries.

The only edit the gate makes to an action is the finger rule, defined from the articulation and
not from observed policy output: finger joints (``finger_joint_pattern``) are position-limited by
the articulation itself, so a finger target beyond a joint stop cannot move the finger past that
stop; its only effect is extra drive force pressing the finger into its own stop. Finger targets
are therefore CLAMPED to the joint range (each clamp is returned in ``GateDecision.clamps`` and the
owner must apply the clamped values), and fingers are judged on what they physically do: their
MEASURED positions and velocities go through the same ``joint_position_rad`` and
``joint_velocity_frac`` gates as every other joint, so a finger that actually leaves its range or
moves too fast still STOPs. Arm and waist targets are never edited: one beyond its range by more
than ``command_joint_tolerance_rad`` (0.02 rad in the shipped profile) is a STOP.

The gate never forwards an unchecked channel: every commanded channel is limited (the proposed
action must carry exactly ``joint_targets``, ``nav``, ``base_height`` and ``torso_rpy``; an
action with another channel, or a control step without an action, is malformed and STOPs).
Every value it compares is a number: a string (``"nan"``, ``"0.1"``), a bool or None where a
number belongs is malformed and STOPs, never parsed; and a NaN never compares as inside a limit.

Checks, every one reported as ``value`` against ``limit``:

1. ``state_finite``: every measured and commanded number is finite, and the step is well formed;
2. ``pelvis_tilt_deg``: angle between the pelvis z axis and world up (fall criterion; the
   humanoid vertical uses 30 deg, docs/HUMANOID_H2.md);
3. ``pelvis_height_m``: pelvis height above the floor (fall criterion);
4. ``joint_position_rad``: worst excursion of a measured joint beyond its articulation limit;
5. ``joint_velocity_frac``: worst ``|qd| / velocity limit`` over measured joints;
6. ``workspace_m``: distance of the pelvis outside the declared xy box (0 inside);
7. ``command_joint_rad``: worst excursion of a policy-commanded arm/waist target beyond its limit
   (finger targets are clamped, see above);
8. ``command_wbc_joint_rad``: the same rule for the targets of the joints the whole-body
   controller owns (``policy_joint_exclude``: legs). They are not policy targets, but they are
   in the action the owner applies, so they are limited too (fail closed);
9. ``command_nav``: worst ``command / limit`` over the navigation command (vx, vy, wz), per
   direction (``max_nav_command`` entries: ``m`` = [-m, m], or an asymmetric ``[min, max]``);
10. ``command_base_height_m``: distance of the base-height command outside its range;
11. ``command_torso_rpy``: worst ``command / limit`` over the torso orientation command
    (roll, pitch, yaw; ``torso_rpy_command_rad``, same entry forms).

Pre-motion admission (:meth:`HumanoidSafetyGate.admit`) vets the measured state alone; a control
step (:meth:`HumanoidSafetyGate.check`) always carries the proposed action.

The gate is pure (no simulator import, no IO); the domain owns transport and timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import threading

GATES = ("state_finite", "pelvis_tilt_deg", "pelvis_height_m", "joint_position_rad", "joint_velocity_frac",
         "workspace_m", "command_joint_rad", "command_wbc_joint_rad", "command_nav", "command_base_height_m",
         "command_torso_rpy")

#: the channels of a proposed action; the gate checks each one and refuses any other
ACTION_CHANNELS = frozenset({"joint_targets", "nav", "base_height", "torso_rpy"})

_LIMIT_KEYS = {"max_pelvis_tilt_deg", "min_pelvis_height_m", "joint_position_tolerance_rad",
               "max_joint_velocity_fraction", "workspace_xy_m", "max_nav_command",
               "base_height_command_m", "command_joint_tolerance_rad", "policy_joint_exclude",
               "finger_joint_pattern", "torso_rpy_command_rad"}


def _finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _ranges(value, name, axes) -> tuple[tuple[float, float], ...]:
    """Per-axis command ranges. An entry ``m`` is the symmetric range [-m, m]; ``[min, max]`` is an
    asymmetric one. Every range holds zero strictly inside (the hold command, zero, must pass)."""
    if not isinstance(value, list) or len(value) != len(axes):
        raise ValueError(f"{name} must be [{', '.join(axes)}]")
    out = []
    for entry in value:
        if isinstance(entry, list):
            if len(entry) != 2:
                raise ValueError(f"{name} entries must be a number or [min, max]")
            lo, hi = _finite(entry[0], name), _finite(entry[1], name)
        else:
            m = _finite(entry, name, positive=True)
            lo, hi = -m, m
        if not lo < 0.0 < hi:
            raise ValueError(f"{name} ranges must hold zero strictly inside (min < 0 < max)")
        out.append((lo, hi))
    return tuple(out)


def _range_ratio(values, ranges) -> float:
    """Worst ``command / limit`` in the command's own direction (1.0 = on a limit); NaN if any is NaN."""
    ratios = [v / hi if v >= 0 else v / lo for v, (lo, hi) in zip(values, ranges)]
    return math.nan if any(math.isnan(r) for r in ratios) else max(ratios)


def _excess(value: float, lo: float, hi: float) -> float:
    """How far ``value`` lies outside [lo, hi] (0 inside); NaN for a NaN (``max(0.0, ...)`` drops it)."""
    return math.nan if math.isnan(value) else max(0.0, lo - value, value - hi)


def _worse(value: float, worst: float) -> bool:
    """True when ``value`` replaces ``worst`` in a worst-of scan. A NaN replaces any worst and is never
    replaced (``x > NaN`` is False), so it reaches its row and fails the ``value <= limit`` check; a
    plain ``>`` scan would skip it (``NaN > worst`` is False) and read it as inside the limit."""
    return math.isnan(value) or value > worst


@dataclass(frozen=True)
class HumanoidGateLimits:
    max_pelvis_tilt_deg: float
    min_pelvis_height_m: float
    joint_position_tolerance_rad: float
    max_joint_velocity_fraction: float
    workspace_xy_m: tuple[tuple[float, float], tuple[float, float]]
    #: per axis (vx, vy, wz): (min, max)
    max_nav_command: tuple[tuple[float, float], ...]
    base_height_command_m: tuple[float, float]
    command_joint_tolerance_rad: float
    policy_joint_exclude: tuple[str, ...] = field(default_factory=tuple)
    finger_joint_pattern: str = ".*_hand_.*"
    #: per axis (roll, pitch, yaw): (min, max)
    torso_rpy_command_rad: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_profile(cls, data) -> "HumanoidGateLimits":
        if not isinstance(data, dict) or set(data) != _LIMIT_KEYS:
            raise ValueError(f"humanoid safety requires exactly {sorted(_LIMIT_KEYS)}")
        tilt = _finite(data["max_pelvis_tilt_deg"], "max_pelvis_tilt_deg", positive=True)
        if tilt >= 90:
            raise ValueError("max_pelvis_tilt_deg must be below 90")
        box = data["workspace_xy_m"]
        if (not isinstance(box, list) or len(box) != 2 or any(not isinstance(p, list) or len(p) != 2 for p in box)):
            raise ValueError("workspace_xy_m must be [[x0, y0], [x1, y1]]")
        lo = tuple(_finite(v, "workspace_xy_m") for v in box[0])
        hi = tuple(_finite(v, "workspace_xy_m") for v in box[1])
        if not (lo[0] < hi[0] and lo[1] < hi[1]):
            raise ValueError("workspace_xy_m must be [[min x, min y], [max x, max y]]")
        nav = _ranges(data["max_nav_command"], "max_nav_command", ("vx", "vy", "wz"))
        torso = _ranges(data["torso_rpy_command_rad"], "torso_rpy_command_rad", ("roll", "pitch", "yaw"))
        height = data["base_height_command_m"]
        if not isinstance(height, list) or len(height) != 2:
            raise ValueError("base_height_command_m must be [min, max]")
        h = (_finite(height[0], "base_height_command_m"), _finite(height[1], "base_height_command_m"))
        if not 0 < h[0] < h[1]:
            raise ValueError("base_height_command_m must be an increasing positive range")
        exclude = data["policy_joint_exclude"]
        if not isinstance(exclude, list) or not all(isinstance(p, str) and p for p in exclude):
            raise ValueError("policy_joint_exclude must be a list of joint-name patterns")
        for pattern in [*exclude, data["finger_joint_pattern"]]:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError("joint-name patterns must be nonempty strings")
            re.compile(pattern)
        return cls(tilt, _finite(data["min_pelvis_height_m"], "min_pelvis_height_m", positive=True),
                   _finite(data["joint_position_tolerance_rad"], "joint_position_tolerance_rad"),
                   _finite(data["max_joint_velocity_fraction"], "max_joint_velocity_fraction", positive=True),
                   (lo, hi), nav, h,
                   _finite(data["command_joint_tolerance_rad"], "command_joint_tolerance_rad"), tuple(exclude),
                   data["finger_joint_pattern"], torso)

    def as_dict(self) -> dict:
        return {"max_pelvis_tilt_deg": self.max_pelvis_tilt_deg, "min_pelvis_height_m": self.min_pelvis_height_m,
                "joint_position_tolerance_rad": self.joint_position_tolerance_rad,
                "max_joint_velocity_fraction": self.max_joint_velocity_fraction,
                "workspace_xy_m": [list(self.workspace_xy_m[0]), list(self.workspace_xy_m[1])],
                "max_nav_command": [list(r) for r in self.max_nav_command],
                "base_height_command_m": list(self.base_height_command_m),
                "command_joint_tolerance_rad": self.command_joint_tolerance_rad,
                "policy_joint_exclude": list(self.policy_joint_exclude),
                "finger_joint_pattern": self.finger_joint_pattern,
                "torso_rpy_command_rad": [list(r) for r in self.torso_rpy_command_rad]}


@dataclass(frozen=True)
class JointModel:
    """Articulation joint order and limits, as the simulator owner reports them."""

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    velocity: tuple[float, ...]

    @classmethod
    def from_dict(cls, data) -> "JointModel":
        if not isinstance(data, dict) or set(data) != {"names", "lower", "upper", "velocity"}:
            raise ValueError("joint model requires names, lower, upper and velocity")
        names = tuple(str(n) for n in data["names"])
        cols = [tuple(_finite(float(v), key) for v in data[key]) for key in ("lower", "upper", "velocity")]
        if not names or len(set(names)) != len(names) or any(len(c) != len(names) for c in cols):
            raise ValueError("joint model columns must match the unique joint names")
        if any(lo > hi for lo, hi in zip(cols[0], cols[1])) or any(v <= 0 for v in cols[2]):
            raise ValueError("joint limits must be ordered and velocity limits positive")
        return cls(names, *cols)


@dataclass(frozen=True)
class GateRow:
    name: str
    value: float | None
    limit: float
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict:
        out = {"name": self.name, "value": None if self.value is None else round(float(self.value), 5),
               "limit": round(float(self.limit), 5), "ok": self.ok}
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass(frozen=True)
class FingerClamp:
    """One finger target moved onto its joint range: ``commanded`` -> ``applied`` (a stop)."""

    joint: str
    index: int
    commanded: float
    applied: float

    @property
    def excess_rad(self) -> float:
        return abs(self.commanded - self.applied)

    def as_dict(self) -> dict:
        return {"joint": self.joint, "index": self.index, "commanded": round(self.commanded, 5),
                "applied": round(self.applied, 5)}


@dataclass(frozen=True)
class GateDecision:
    step: int
    verdict: str  # "PASS" | "STOP"
    rows: tuple[GateRow, ...]
    reason: str = ""
    #: finger targets the owner must replace before applying the action (empty: apply as proposed)
    clamps: tuple[FingerClamp, ...] = ()

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def as_dict(self) -> dict:
        out = {"step": self.step, "verdict": self.verdict, "reason": self.reason,
               "rows": [r.as_dict() for r in self.rows]}
        if self.clamps:
            out["clamps"] = [c.as_dict() for c in self.clamps]
        return out


def _vec(data, key, n):
    value = data.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{key} must have {n} values")
    return [_num(v, key) for v in value]


def tilt_deg(quat_wxyz) -> float:
    """Angle between the body z axis and world +z, for a unit quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("degenerate quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    up_z = 1.0 - 2.0 * (x * x + y * y)  # R[2, 2]: world-z component of the body z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


class HumanoidSafetyGate:
    """Per-control-step gate with a latched hard stop. Thread-safe: ``estop`` may come from any
    thread (the agent's emergency stop) while a segment streams steps."""

    def __init__(self, limits: HumanoidGateLimits, joints: JointModel | None = None):
        self.limits = limits
        self.joints = joints
        self._lock = threading.Lock()
        self._latched: str | None = None
        self._policy_mask: tuple[bool, ...] | None = None
        self._finger_mask: tuple[bool, ...] | None = None
        if joints is not None:
            self.bind(joints)

    def bind(self, joints: JointModel) -> None:
        patterns = [re.compile(p) for p in self.limits.policy_joint_exclude]
        finger = re.compile(self.limits.finger_joint_pattern)
        with self._lock:
            self.joints = joints
            self._policy_mask = tuple(not any(p.fullmatch(n) for p in patterns) for n in joints.names)
            self._finger_mask = tuple(bool(finger.fullmatch(n)) for n in joints.names)

    # -- latch ---------------------------------------------------------------------------
    @property
    def latched(self) -> str | None:
        with self._lock:
            return self._latched

    def estop(self, reason: str = "emergency stop") -> None:
        with self._lock:
            self._latched = self._latched or str(reason)

    def reset(self) -> str | None:
        """Clear the latch; returns the stop it cleared (None: nothing was latched)."""
        with self._lock:
            cleared, self._latched = self._latched, None
        return cleared

    # -- checks ---------------------------------------------------------------------------
    def _state_rows(self, state: dict, rows: list[GateRow]) -> None:
        lim = self.limits
        quat = _vec(state, "pelvis_quat_wxyz", 4)
        pos = _vec(state, "pelvis_pos", 3)
        floor = _num(state.get("floor_z", 0.0), "floor_z")
        tilt = tilt_deg(quat)
        rows.append(GateRow("pelvis_tilt_deg", tilt, lim.max_pelvis_tilt_deg, tilt <= lim.max_pelvis_tilt_deg))
        height = pos[2] - floor
        rows.append(GateRow("pelvis_height_m", height, lim.min_pelvis_height_m, height >= lim.min_pelvis_height_m))
        jm = self.joints
        q = [_num(v, "joint_pos") for v in state["joint_pos"]]
        qd = [_num(v, "joint_vel") for v in state["joint_vel"]]
        if jm is None or len(q) != len(jm.names) or len(qd) != len(jm.names):
            raise ValueError("measured joints do not match the bound joint model")
        worst, worst_name = 0.0, ""
        for name, v, lo, hi in zip(jm.names, q, jm.lower, jm.upper):
            excess = _excess(v, lo, hi)
            if _worse(excess, worst):
                worst, worst_name = excess, name
        rows.append(GateRow("joint_position_rad", worst, lim.joint_position_tolerance_rad,
                            worst <= lim.joint_position_tolerance_rad, worst_name))
        frac, frac_name = 0.0, ""
        for name, v, vmax in zip(jm.names, qd, jm.velocity):
            f = abs(v) / vmax
            if _worse(f, frac):
                frac, frac_name = f, name
        rows.append(GateRow("joint_velocity_frac", frac, lim.max_joint_velocity_fraction,
                            frac <= lim.max_joint_velocity_fraction, frac_name))
        (x0, y0), (x1, y1) = lim.workspace_xy_m
        out = math.hypot(_excess(pos[0], x0, x1), _excess(pos[1], y0, y1))
        rows.append(GateRow("workspace_m", out, 0.0, out <= 0.0))

    def _action_rows(self, action: dict, rows: list[GateRow], clamps: list[FingerClamp]) -> None:
        lim, jm = self.limits, self.joints
        targets = [_num(v, "joint_targets") for v in action["joint_targets"]]
        if jm is None or len(targets) != len(jm.names):
            raise ValueError("commanded joint targets do not match the bound joint model")
        mask = self._policy_mask or (True,) * len(jm.names)
        fingers = self._finger_mask or (False,) * len(jm.names)
        worst, worst_name = 0.0, ""
        wbc, wbc_name = 0.0, ""
        for i, (name, v, lo, hi, policy, finger) in enumerate(zip(jm.names, targets, jm.lower, jm.upper, mask,
                                                                  fingers)):
            excess = _excess(v, lo, hi)
            if not policy:
                # whole-body-controller joints (legs): not policy targets, yet in the applied action
                if _worse(excess, wbc):
                    wbc, wbc_name = excess, name
                continue
            if finger and not math.isnan(v):  # a NaN finger target is never clamped into a number
                applied = min(max(v, lo), hi)
                if applied != v:
                    clamps.append(FingerClamp(name, i, v, applied))
                continue
            if _worse(excess, worst):
                worst, worst_name = excess, name
        rows.append(GateRow("command_joint_rad", worst, lim.command_joint_tolerance_rad,
                            worst <= lim.command_joint_tolerance_rad, worst_name))
        rows.append(GateRow("command_wbc_joint_rad", wbc, lim.command_joint_tolerance_rad,
                            wbc <= lim.command_joint_tolerance_rad, wbc_name))
        nav = _vec(action, "nav", 3)
        ratio = _range_ratio(nav, lim.max_nav_command)
        rows.append(GateRow("command_nav", ratio, 1.0, ratio <= 1.0,
                            "vx, vy, wz = " + ", ".join(f"{v:.3f}" for v in nav)))
        h = _num(action["base_height"], "base_height")
        lo, hi = lim.base_height_command_m
        out = _excess(h, lo, hi)
        rows.append(GateRow("command_base_height_m", out, 0.0, out <= 0.0, f"{h:.3f} m"))
        torso = _vec(action, "torso_rpy", 3)
        ratio = _range_ratio(torso, lim.torso_rpy_command_rad)
        rows.append(GateRow("command_torso_rpy", ratio, 1.0, ratio <= 1.0,
                            "roll, pitch, yaw = " + ", ".join(f"{v:.3f}" for v in torso)))

    def admit(self, step, state: dict) -> GateDecision:
        """Pre-motion admission: vet the measured start state (nothing is commanded yet)."""
        return self._decide(step, state, None, acting=False)

    def check(self, step, state: dict, action: dict) -> GateDecision:
        """Vet one control step: the measured ``state`` and the proposed ``action``.

        A control step always carries its action: one without it (or with a malformed step
        index, state or action) is malformed and STOPs, so the owner never applies an action
        the gate did not see. On PASS, ``clamps`` lists the finger targets to replace before the
        action is applied."""
        return self._decide(step, state, action, acting=True)

    def _decide(self, step, state, action, *, acting: bool) -> GateDecision:
        rows: list[GateRow] = []
        clamps: list[FingerClamp] = []
        index = step if type(step) is int and step >= 0 else -1
        try:
            if index < 0:
                raise ValueError(f"step index {step!r} is not a nonnegative integer")
            if not isinstance(state, dict):
                raise TypeError("the measured state is not an object")
            if acting and not isinstance(action, dict):
                raise ValueError("a control step without a proposed action is refused")
            unchecked = sorted(map(str, set(action) - ACTION_CHANNELS)) if acting else []
            if unchecked:
                # a channel the gate does not check is never passed on to be applied
                raise ValueError(f"action channels the gate does not check: {unchecked}")
            # every leaf of the action is a number (a string, bool or None is malformed, never skipped)
            values = [*_flat(state), *(_flat(action, strict=True) if acting else [])]
            finite = all(math.isfinite(v) for v in values)
            rows.append(GateRow("state_finite", 0.0 if finite else 1.0, 0.0, finite))
            if finite:
                self._state_rows(state, rows)
                if acting:
                    self._action_rows(action, rows, clamps)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:  # Overflow: a JSON int too big for a float
            rows.append(GateRow("state_finite", None, 0.0, False, f"malformed step: {exc}"))
        failed = [r for r in rows if not r.ok]
        with self._lock:
            if failed and self._latched is None:
                r = failed[0]
                self._latched = f"{r.name} {r.value if r.value is None else round(r.value, 4)} vs limit {r.limit}" + (
                    f" ({r.detail})" if r.detail else "")
            latched = self._latched
        if latched is not None:
            return GateDecision(index, "STOP", tuple(rows), latched)
        return GateDecision(index, "PASS", tuple(rows), clamps=tuple(clamps))


def _flat(data, *, strict: bool = False) -> list[float]:
    """Every number in ``data``. ``strict`` (a proposed action): any other leaf is malformed. The
    measured state may carry other fields; every state value the gate reads goes through ``_num``."""
    out: list[float] = []
    if isinstance(data, dict):
        for value in data.values():
            out.extend(_flat(value, strict=strict))
    elif isinstance(data, (list, tuple)):
        for value in data:
            out.extend(_flat(value, strict=strict))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        out.append(float(data))
    elif strict:
        raise ValueError(f"non-numeric command value {data!r}")
    return out


def _num(value, name: str) -> float:
    """A number the gate compares: never parsed from a string ("nan", "0.1"), a bool or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} value {value!r} is not a number")
    return float(value)
