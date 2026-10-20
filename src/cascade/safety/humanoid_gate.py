"""Humanoid safety gate: every control step of a learned whole-body policy is vetted.

A learned loco-manipulation policy (the GR00T executor of ``humanoid_locomanip``) proposes one
action per control step. The simulator owner (the Arena bridge) sends the MEASURED state the step
starts from and the PROPOSED action; this gate answers PASS (the action may be applied) or STOP
(it is not applied; the owner switches to its hold command and ends the segment). A STOP latches
until an explicit ``reset()``; nothing here retries, clips or edits an action.

Checks, every one reported as ``value`` against ``limit``:

1. ``state_finite``: every measured and commanded number is finite;
2. ``pelvis_tilt_deg``: angle between the pelvis z axis and world up (fall criterion; the
   humanoid vertical uses 30 deg, docs/HUMANOID_H2.md);
3. ``pelvis_height_m``: pelvis height above the floor (fall criterion);
4. ``joint_position_rad``: worst excursion of a measured joint beyond its articulation limit;
5. ``joint_velocity_frac``: worst ``|qd| / velocity limit`` over measured joints;
6. ``workspace_m``: distance of the pelvis outside the declared xy box (0 inside);
7. ``command_joint_rad``: worst excursion of a policy-commanded joint target beyond its limit
   (leg joints belong to the whole-body controller and are not policy targets);
8. ``command_nav``: worst ``|command| / limit`` over the navigation command (vx, vy, wz);
9. ``command_base_height_m``: distance of the base-height command outside its range.

The gate is pure (no simulator import, no IO); the domain owns transport and timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import threading

GATES = ("state_finite", "pelvis_tilt_deg", "pelvis_height_m", "joint_position_rad", "joint_velocity_frac",
         "workspace_m", "command_joint_rad", "command_nav", "command_base_height_m")

_LIMIT_KEYS = {"max_pelvis_tilt_deg", "min_pelvis_height_m", "joint_position_tolerance_rad",
               "max_joint_velocity_fraction", "workspace_xy_m", "max_nav_command",
               "base_height_command_m", "command_joint_tolerance_rad", "policy_joint_exclude"}


def _finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class HumanoidGateLimits:
    max_pelvis_tilt_deg: float
    min_pelvis_height_m: float
    joint_position_tolerance_rad: float
    max_joint_velocity_fraction: float
    workspace_xy_m: tuple[tuple[float, float], tuple[float, float]]
    max_nav_command: tuple[float, float, float]
    base_height_command_m: tuple[float, float]
    command_joint_tolerance_rad: float
    policy_joint_exclude: tuple[str, ...] = field(default_factory=tuple)

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
        nav = data["max_nav_command"]
        if not isinstance(nav, list) or len(nav) != 3:
            raise ValueError("max_nav_command must be [vx, vy, wz]")
        height = data["base_height_command_m"]
        if not isinstance(height, list) or len(height) != 2:
            raise ValueError("base_height_command_m must be [min, max]")
        h = (_finite(height[0], "base_height_command_m"), _finite(height[1], "base_height_command_m"))
        if not 0 < h[0] < h[1]:
            raise ValueError("base_height_command_m must be an increasing positive range")
        exclude = data["policy_joint_exclude"]
        if not isinstance(exclude, list) or not all(isinstance(p, str) and p for p in exclude):
            raise ValueError("policy_joint_exclude must be a list of joint-name patterns")
        for pattern in exclude:
            re.compile(pattern)
        return cls(tilt, _finite(data["min_pelvis_height_m"], "min_pelvis_height_m", positive=True),
                   _finite(data["joint_position_tolerance_rad"], "joint_position_tolerance_rad"),
                   _finite(data["max_joint_velocity_fraction"], "max_joint_velocity_fraction", positive=True),
                   (lo, hi), tuple(_finite(v, "max_nav_command", positive=True) for v in nav), h,
                   _finite(data["command_joint_tolerance_rad"], "command_joint_tolerance_rad"), tuple(exclude))

    def as_dict(self) -> dict:
        return {"max_pelvis_tilt_deg": self.max_pelvis_tilt_deg, "min_pelvis_height_m": self.min_pelvis_height_m,
                "joint_position_tolerance_rad": self.joint_position_tolerance_rad,
                "max_joint_velocity_fraction": self.max_joint_velocity_fraction,
                "workspace_xy_m": [list(self.workspace_xy_m[0]), list(self.workspace_xy_m[1])],
                "max_nav_command": list(self.max_nav_command),
                "base_height_command_m": list(self.base_height_command_m),
                "command_joint_tolerance_rad": self.command_joint_tolerance_rad,
                "policy_joint_exclude": list(self.policy_joint_exclude)}


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
class GateDecision:
    step: int
    verdict: str  # "PASS" | "STOP"
    rows: tuple[GateRow, ...]
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def as_dict(self) -> dict:
        return {"step": self.step, "verdict": self.verdict, "reason": self.reason,
                "rows": [r.as_dict() for r in self.rows]}


def _vec(data, key, n):
    value = data.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{key} must have {n} values")
    return [float(v) for v in value]


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
        if joints is not None:
            self.bind(joints)

    def bind(self, joints: JointModel) -> None:
        patterns = [re.compile(p) for p in self.limits.policy_joint_exclude]
        with self._lock:
            self.joints = joints
            self._policy_mask = tuple(not any(p.fullmatch(n) for p in patterns) for n in joints.names)

    # -- latch ---------------------------------------------------------------------------
    @property
    def latched(self) -> str | None:
        with self._lock:
            return self._latched

    def estop(self, reason: str = "emergency stop") -> None:
        with self._lock:
            self._latched = self._latched or str(reason)

    def reset(self) -> None:
        with self._lock:
            self._latched = None

    # -- checks ---------------------------------------------------------------------------
    def _state_rows(self, state: dict, rows: list[GateRow]) -> None:
        lim = self.limits
        quat = _vec(state, "pelvis_quat_wxyz", 4)
        pos = _vec(state, "pelvis_pos", 3)
        floor = float(state.get("floor_z", 0.0))
        tilt = tilt_deg(quat)
        rows.append(GateRow("pelvis_tilt_deg", tilt, lim.max_pelvis_tilt_deg, tilt <= lim.max_pelvis_tilt_deg))
        height = pos[2] - floor
        rows.append(GateRow("pelvis_height_m", height, lim.min_pelvis_height_m, height >= lim.min_pelvis_height_m))
        jm = self.joints
        q = [float(v) for v in state["joint_pos"]]
        qd = [float(v) for v in state["joint_vel"]]
        if jm is None or len(q) != len(jm.names) or len(qd) != len(jm.names):
            raise ValueError("measured joints do not match the bound joint model")
        worst, worst_name = 0.0, ""
        for name, v, lo, hi in zip(jm.names, q, jm.lower, jm.upper):
            excess = max(lo - v, v - hi, 0.0)
            if excess > worst:
                worst, worst_name = excess, name
        rows.append(GateRow("joint_position_rad", worst, lim.joint_position_tolerance_rad,
                            worst <= lim.joint_position_tolerance_rad, worst_name))
        frac, frac_name = 0.0, ""
        for name, v, vmax in zip(jm.names, qd, jm.velocity):
            f = abs(v) / vmax
            if f > frac:
                frac, frac_name = f, name
        rows.append(GateRow("joint_velocity_frac", frac, lim.max_joint_velocity_fraction,
                            frac <= lim.max_joint_velocity_fraction, frac_name))
        (x0, y0), (x1, y1) = lim.workspace_xy_m
        out = math.hypot(max(x0 - pos[0], 0.0, pos[0] - x1), max(y0 - pos[1], 0.0, pos[1] - y1))
        rows.append(GateRow("workspace_m", out, 0.0, out <= 0.0))

    def _action_rows(self, action: dict, rows: list[GateRow]) -> None:
        lim, jm = self.limits, self.joints
        targets = [float(v) for v in action["joint_targets"]]
        if jm is None or len(targets) != len(jm.names):
            raise ValueError("commanded joint targets do not match the bound joint model")
        worst, worst_name = 0.0, ""
        mask = self._policy_mask or (True,) * len(jm.names)
        for name, v, lo, hi, policy in zip(jm.names, targets, jm.lower, jm.upper, mask):
            if not policy:
                continue
            excess = max(lo - v, v - hi, 0.0)
            if excess > worst:
                worst, worst_name = excess, name
        rows.append(GateRow("command_joint_rad", worst, lim.command_joint_tolerance_rad,
                            worst <= lim.command_joint_tolerance_rad, worst_name))
        nav = _vec(action, "nav", 3)
        ratio = max(abs(v) / m for v, m in zip(nav, lim.max_nav_command))
        rows.append(GateRow("command_nav", ratio, 1.0, ratio <= 1.0,
                            "vx, vy, wz = " + ", ".join(f"{v:.3f}" for v in nav)))
        h = float(action["base_height"])
        lo, hi = lim.base_height_command_m
        out = max(lo - h, 0.0, h - hi)
        rows.append(GateRow("command_base_height_m", out, 0.0, out <= 0.0, f"{h:.3f} m"))

    def check(self, step: int, state: dict, action: dict | None = None) -> GateDecision:
        """Vet one control step: the measured ``state`` and (when acting) the proposed ``action``."""
        rows: list[GateRow] = []
        try:
            values = [*_flat(state), *(_flat(action) if action is not None else [])]
            finite = all(math.isfinite(v) for v in values)
            rows.append(GateRow("state_finite", 0.0 if finite else 1.0, 0.0, finite))
            if finite:
                self._state_rows(state, rows)
                if action is not None:
                    self._action_rows(action, rows)
        except (KeyError, TypeError, ValueError) as exc:
            rows.append(GateRow("state_finite", None, 0.0, False, f"malformed step: {exc}"))
        failed = [r for r in rows if not r.ok]
        with self._lock:
            if failed and self._latched is None:
                r = failed[0]
                self._latched = f"{r.name} {r.value if r.value is None else round(r.value, 4)} vs limit {r.limit}" + (
                    f" ({r.detail})" if r.detail else "")
            latched = self._latched
        if latched is not None:
            return GateDecision(int(step), "STOP", tuple(rows), latched)
        return GateDecision(int(step), "PASS", tuple(rows))


def _flat(data) -> list[float]:
    out: list[float] = []
    if isinstance(data, dict):
        for value in data.values():
            out.extend(_flat(value))
    elif isinstance(data, (list, tuple)):
        for value in data:
            out.extend(_flat(value))
    elif isinstance(data, bool) or data is None or isinstance(data, str):
        pass
    elif isinstance(data, (int, float)):
        out.append(float(data))
    return out
