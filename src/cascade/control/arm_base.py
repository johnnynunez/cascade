"""Arm interface. All motion is streamed as min-jerk joint waypoints so the
safety harness can vet every intermediate configuration, and completion is
based on feedback (mock: kinematic state; RobStride: mechPos param reads;
Feetech: Present_Position reads; MuJoCo: qpos), never on sleep(duration) like
the baseline.

A backend implements six methods and declares its own DOF. Nothing above this
layer knows which robot is attached: skills hold a SafeArm, and the arm's joint
count, home poses, tool-frame convention and reach come from its profile in
configs/arms/ (see cascade/config.py's note on profile `overrides`).
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable

import numpy as np

from ..config import Cfg
from ..types import ContactDetected, RobotState, SafetyViolation
from .motion_profile import min_jerk

#: same plausibility bound the contact monitor applies to each reading
#: (cascade.safety.contact); kept here so control/ does not import safety/
READ_MAX_JOINT_SPEED = 6.0     # rad/s
READ_JUMP_TOL_RAD = 0.15


PREFLIGHT_MAX_DRIFT_RAD = 1e-3
PREFLIGHT_REBIND_BUDGET_S = 5.0
PREFLIGHT_MAX_CHECKS = 8


def path_length(q_start, waypoints) -> float:
    """Chebyshev arc length of the joint polyline start -> waypoints: the sum
    over segments of the LARGEST single-joint change. Every joint's travel is
    bounded by it, so a min-jerk profile over this length peaks at
    1.875 * L / T on every joint -- the same bound move_joints stretches by."""
    pts = np.vstack([np.asarray(q_start, dtype=float).reshape(1, -1)]
                    + [np.asarray(w, dtype=float).reshape(1, -1) for w in waypoints])
    return float(np.sum(np.max(np.abs(np.diff(pts, axis=0)), axis=1)))


def path_ticks(q_start, waypoints, duration_s, rate_hz=50.0):
    """Streamed ticks along the joint polyline start -> waypoints, min-jerk in
    (Chebyshev) arc length. Returns (ticks, dt); the last tick is the last
    waypoint. Deterministic, so a preflight can vet exactly what streams."""
    pts = np.vstack([np.asarray(q_start, dtype=float).reshape(1, -1)]
                    + [np.asarray(w, dtype=float).reshape(1, -1) for w in waypoints])
    seg = np.max(np.abs(np.diff(pts, axis=0)), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    steps = max(2, int(duration_s * rate_hz))
    dt = duration_s / steps
    ticks = []
    for i in range(1, steps + 1):
        if total <= 0.0:
            ticks.append(pts[-1].copy())
            continue
        s = min_jerk(i / steps) * total
        k = int(np.searchsorted(cum, s, side="right")) - 1
        k = min(max(k, 0), len(seg) - 1)
        f = 0.0 if seg[k] <= 0.0 else min(max((s - cum[k]) / seg[k], 0.0), 1.0)
        ticks.append(pts[k] + (pts[k + 1] - pts[k]) * f)
    ticks[-1] = pts[-1].copy()
    return ticks, dt


def prepare_stream(arm, q_target, duration_s, preflight=None, before_stream=None):
    """Bind preflight to feedback before starting the streaming clock."""
    start = np.asarray(arm.get_state().q, dtype=float).copy()
    target = np.asarray(q_target, dtype=float).reshape(-1)
    if preflight is not None:
        deadline = time.monotonic() + PREFLIGHT_REBIND_BUDGET_S
        for attempt in range(PREFLIGHT_MAX_CHECKS):
            if before_stream is not None:
                before_stream()
            if time.monotonic() >= deadline:
                raise SafetyViolation("route preflight exhausted its feedback budget; no motion sent")
            preflight(start, duration_s)
            observed = np.asarray(arm.get_state().q, dtype=float)
            if observed.shape != start.shape or not np.isfinite(observed).all():
                raise SafetyViolation("invalid joint feedback during route preflight; no motion sent")
            if before_stream is not None:
                before_stream()
            if time.monotonic() >= deadline:
                raise SafetyViolation("route preflight exhausted its feedback budget; no motion sent")
            if np.max(np.abs(observed - start)) <= PREFLIGHT_MAX_DRIFT_RAD:
                break
            if attempt + 1 == PREFLIGHT_MAX_CHECKS:
                raise SafetyViolation("joint feedback changed during route preflight; no motion sent")
            # A preceding move may still be settling within its ordinary
            # endpoint tolerance. Re-vet from this new measurement; never
            # authorize the earlier path by widening the 1 mrad drift limit.
            start = observed.copy()
    if before_stream is not None:
        before_stream()
    return start, target


class ArmBase(abc.ABC):
    """An N-joint arm + 1 gripper motor behind a uniform, feedback-based API."""

    #: Controlled arm joints, EXCLUDING the gripper. Shipped arms run 5
    #: (SO-101), 6 (reBot B601) and 7 (Panda in LIBERO); the class default is
    #: only a fallback and every backend should take it from its profile's
    #: `n_joints`, because a wrong value silently truncates or broadcasts each
    #: commanded pose instead of failing.
    n_joints = 6
    # Planner-visible nominal rate. Hardware streaming keeps its 50 Hz default.
    motion_rate_hz = 50.0
    #: settle tolerance (rad). Real arms hold with pure PD (no gravity
    #: feedforward), so steady-state droop under payload needs headroom;
    #: backends override from config.
    settle_tol = 0.03
    #: how long wait_settled() waits after the streamed profile ends (s).
    #: Measured on the Isaac rig: PhysX converges inside ~1 s, but Newton's
    #: solver takes noticeably longer to bleed off the last of the error --
    #: a 0.17 rad step needed 3.6 s to come inside settle_tol, so the old
    #: hard-coded 2.0 s made every pregrasp report "did not settle" even
    #: though the arm was on its way to the right pose. Backends override
    #: from config (`arm.settle_timeout_s`).
    settle_timeout_s = 2.0

    @property
    def disconnect_preserves_drive_state(self) -> bool:
        """Whether disconnect only releases transport, without drive commands.

        False keeps the existing park-before-disconnect policy. Only the
        Isaac backend currently declares this capability; it is not inferred
        from another driver's lack of explicit torque-off, nor does it prove
        that a payload remains held.
        """
        return False

    @property
    def disconnect_cuts_torque(self) -> bool:
        """Whether disconnect() may leave the arm without torque, so a raised
        arm drops: a serial bus's torque-off (so101), an SDK's disable
        (rebot_rs). Teardown then disconnects such an arm only after its park
        brought it to rest, or on the operator's explicit confirmation
        (`apps.demo.TORQUE_OFF_CONFIRM_ENV`); otherwise it stays powered and
        holding. True unless the backend states otherwise: one whose
        disconnect only releases transport (`disconnect_preserves_drive_state`),
        leaves a controller holding the arm, or drives no hardware.
        """
        return not self.disconnect_preserves_drive_state

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def get_state(self) -> RobotState: ...

    @abc.abstractmethod
    def send_joint_target(self, q: np.ndarray) -> None:
        """Stream one joint-space setpoint (already safety-approved)."""

    @abc.abstractmethod
    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        """Command gripper motor position with an effort/torque scale 0..1."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Best-effort immediate halt (software e-stop)."""

    # ── contact monitoring hooks (safety.contact_monitor) ────────────────

    def contact_joint_reader(self) -> Callable[[int, float], float] | None:
        """`read(j, timeout_s) -> float`: ONE joint's measured position for
        the contact monitor, raising when it cannot be read -- or None when
        this backend only reads its full state (then each mid-stream sample
        is one `get_state()`). Backends whose full read is slow (the RS
        drivers poll six motors one after another) return a reader, so the
        50 Hz stream reads as many joints as fit in each cycle's slack."""
        return None

    def read_positions_for_monitor(self) -> np.ndarray:
        """A FRESH full measured pose, raising when it cannot be read (never
        a cached last-known value): the contact monitor's settle samples and
        the pose the arm holds after a contact."""
        return np.asarray(self.get_state().q, dtype=float)

    # ── shared streaming executor ────────────────────────────────────────

    def stream_to(
        self,
        q_target: np.ndarray,
        duration_s: float,
        rate_hz: float = 50.0,
        approve=None,
        settle_tol: float | None = None,
        settle_timeout_s: float | None = None,
        preflight=None,
        before_stream=None,
        contact_monitor=None,
    ) -> bool:
        """Min-jerk interpolate current->target, vetting each waypoint.

        `approve(q_prev, q_next, dt)` raises SafetyViolation to abort.
        Returns True when the arm settles within `settle_tol` rad of target.
        `contact_monitor` (cascade.safety.contact.ContactMonitor) compares
        measured joints with the stream every cycle and through the settle
        phase; on a trip the arm holds its measured pose and ContactDetected
        propagates (see `_stream_ticks`).
        """
        if settle_tol is None:
            settle_tol = self.settle_tol
        if settle_timeout_s is None:
            settle_timeout_s = self.settle_timeout_s
        q_start, q_target = prepare_stream(self, q_target, duration_s, preflight, before_stream)
        steps = max(2, int(duration_s * rate_hz))
        dt = duration_s / steps
        ticks = (q_start + (q_target - q_start) * min_jerk(i / steps) for i in range(1, steps + 1))
        feedback = self._stream_ticks(q_start, ticks, dt, approve, contact_monitor, paced=True)
        return self._settle(q_target, settle_tol, settle_timeout_s, feedback)

    def stream_path(
        self,
        waypoints,
        duration_s: float,
        rate_hz: float = 50.0,
        approve=None,
        settle_tol: float | None = None,
        settle_timeout_s: float | None = None,
        preflight=None,
        before_stream=None,
        contact_monitor=None,
    ) -> bool:
        """Stream a dense joint path (e.g. a Cartesian line) as ONE motion.

        Same contract as `stream_to` -- bound start feedback via
        `prepare_stream`, `approve(q_prev, q_next, dt)` on every tick,
        deadline pacing, feedback-based settle, optional contact monitor --
        but the ticks follow the polyline start -> waypoints (min-jerk in arc
        length, `path_ticks`) instead of a straight joint-space line, and the
        arm does not stop at intermediate waypoints. Backends whose motion has
        its own clock (Isaac) must refuse rather than inherit this.
        """
        if settle_tol is None:
            settle_tol = self.settle_tol
        if settle_timeout_s is None:
            settle_timeout_s = self.settle_timeout_s
        waypoints = [np.asarray(w, dtype=float).reshape(-1) for w in waypoints]
        if not waypoints:
            raise SafetyViolation("empty joint path; no motion sent")
        q_start, _ = prepare_stream(self, waypoints[-1], duration_s, preflight, before_stream)
        ticks, dt = path_ticks(q_start, waypoints, duration_s, rate_hz)
        feedback = self._stream_ticks(q_start, ticks, dt, approve, contact_monitor, paced=True)
        return self._settle(waypoints[-1], settle_tol, settle_timeout_s, feedback)

    def _stream_ticks(self, q_start, ticks, dt, approve, contact_monitor, *, paced):
        """Send each tick (approved first). Paced: deadline-based, so per-step
        overhead (approve FK + CAN sends + feedback reads) does not stretch
        the total duration. Returns the contact feedback (None without a
        monitor) for the settle phase."""
        feedback = (None if contact_monitor is None
                    else ContactFeedback(self, contact_monitor, q_start, dt=dt, paced=paced))
        q_prev = q_start
        t0 = time.monotonic()
        if feedback is not None:
            feedback.start(t0)
        try:
            for i, q_i in enumerate(ticks, start=1):
                if approve is not None:
                    approve(q_prev, q_i, dt)
                self.send_joint_target(q_i)
                q_prev = q_i
                if feedback is not None:
                    feedback.after_send(i, q_i, deadline=t0 + i * dt)
                if paced:
                    sleep_s = t0 + i * dt - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        except SafetyViolation as exc:
            if feedback is not None:
                feedback.stopped(exc)
            raise
        return feedback

    def _settle(self, q_target, tol, timeout_s, feedback) -> bool:
        if feedback is None:
            return self.wait_settled(q_target, tol, timeout_s)
        try:
            return feedback.settle(np.asarray(q_target, dtype=float), tol, timeout_s)
        except SafetyViolation as exc:
            feedback.stopped(exc)
            raise

    def wait_settled(self, q_target: np.ndarray, tol: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            err = np.abs(self.get_state().q - q_target).max()
            if err < tol:
                return True
            time.sleep(0.05)
        return False


class ContactFeedback:
    """Mid-stream feedback for one motion's ContactMonitor.

    After every streamed target: record it, then read joints -- one motor at a
    time when the backend has a `contact_joint_reader`, as many as fit before
    the next target is due (at least `min_reads_per_cycle`), round-robin; else
    one full `get_state()` -- and hand each reading to the monitor. Through the
    settle phase: one fresh full read per 50 ms poll. Any trip or lost
    feedback raises out of the stream before another target is sent.

    On ContactDetected the arm is commanded to hold its CURRENT MEASURED pose
    (a fresh full read, checked against the monitor's own latest samples) --
    never the command it could not reach, never torque off. If that read
    fails or disagrees, nothing is sent: the motors keep the last streamed
    target, like every other abort. Paced streams use wall time since the
    first target; the unpaced mock uses logical tick time.
    """

    #: slack kept free before the next target is due, beyond one read's cost
    GUARD_S = 0.001
    #: initial single-read cost estimate (s), refined per read (EMA)
    INITIAL_READ_COST_S = 0.010
    SETTLE_POLL_S = 0.05
    #: consecutive failed settle reads tolerated in measure mode (enforce
    #: mode stops on the monitor's feedback-gap bound first); the same count
    #: the RS drivers' get_state tolerates before "CAN feedback lost"
    SETTLE_MAX_READ_FAILURES = 3

    def __init__(self, arm, monitor, q_start, *, dt, paced):
        self.arm, self.m, self.dt, self.paced = arm, monitor, float(dt), bool(paced)
        monitor.begin(q_start)
        self.reader = arm.contact_joint_reader()
        self.cost = self.INITIAL_READ_COST_S
        self.t0 = None
        self.tick = 0

    def start(self, t0):
        self.t0 = t0

    def _now(self):
        if self.paced:
            return time.monotonic() - self.t0
        return (self.tick - 0.5) * self.dt   # between this target and the next

    def after_send(self, i, q_i, *, deadline):
        m = self.m
        self.tick = i
        m.commanded(time.monotonic() - self.t0 if self.paced else (i - 1) * self.dt, q_i)
        costs = []
        if self.reader is None:
            self._read_full(costs)
        else:
            cfg = m.cfg
            while len(costs) < m.n:
                if (self.paced and len(costs) >= cfg.min_reads_per_cycle
                        and deadline - time.monotonic() < self.cost + self.GUARD_S):
                    break
                j = m.next_joint()
                ta = time.monotonic()
                try:
                    x = self.reader(j, cfg.read_timeout_s)
                except Exception as exc:  # noqa: BLE001 -- any failed read is lost feedback
                    tb = time.monotonic()
                    self._cost(tb - ta, costs)
                    m.read_failed(self._t(ta, tb), j, f"{type(exc).__name__}: {exc}")
                    continue
                tb = time.monotonic()
                self._cost(tb - ta, costs)
                m.observe(self._t(ta, tb), j, x)
        m.check_feedback(self._now())
        m.note_cycle(len(costs), costs, overrun=self.paced and time.monotonic() > deadline)

    def _t(self, ta, tb):
        return (0.5 * (ta + tb) - self.t0) if self.paced else self._now()

    def _cost(self, c, costs):
        costs.append(c)
        self.cost = 0.7 * self.cost + 0.3 * c

    def _read_full(self, costs):
        ta = time.monotonic()
        try:
            q = np.asarray(self.arm.get_state().q, dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            tb = time.monotonic()
            costs.append(tb - ta)
            for j in range(self.m.n):
                self.m.read_failed(self._t(ta, tb), j, f"{type(exc).__name__}: {exc}")
            return
        tb = time.monotonic()
        costs.append(tb - ta)
        self.m.observe_all(self._t(ta, tb), q)

    def settle(self, q_target, tol, timeout_s) -> bool:
        """wait_settled, with every poll handed to the monitor."""
        m = self.m
        m.phase = "settle"
        deadline = time.monotonic() + timeout_s
        failures = 0
        while time.monotonic() < deadline:
            ta = time.monotonic()
            try:
                q = np.asarray(self.arm.read_positions_for_monitor(), dtype=float).reshape(-1)
            except Exception as exc:
                failures += 1
                tb = time.monotonic()
                for j in range(m.n):
                    m.read_failed(self._t(ta, tb), j, f"{type(exc).__name__}: {exc}")
                if failures >= self.SETTLE_MAX_READ_FAILURES:
                    raise RuntimeError(f"feedback lost while settling ({failures} consecutive "
                                       f"failed reads): {exc}") from exc
                time.sleep(self.SETTLE_POLL_S)
                continue
            failures = 0
            tb = time.monotonic()
            m.observe_all(self._t(ta, tb), q)
            m.check_feedback(self._now())
            if q.shape == q_target.shape and float(np.abs(q - q_target).max()) < tol:
                return True
            time.sleep(self.SETTLE_POLL_S)
        return False

    def stopped(self, exc):
        """The stream is ending on `exc`. After a contact, hold the measured
        pose; after anything else, the last streamed target stands."""
        m = self.m
        if not isinstance(exc, ContactDetected):
            if m.hold is None:
                m.hold = f"last command ({'feedback lost' if m.feedback_lost else 'stream aborted'})"
            return
        why = None
        q = None
        try:
            q = np.asarray(self.arm.read_positions_for_monitor(), dtype=float).reshape(-1)
            if q.shape != (m.n,) or not np.all(np.isfinite(q)):
                why, q = f"bad reading {q.tolist()}", None
            else:
                elapsed = np.maximum(self._now() - m.latest_t, 0.0)
                allowed = READ_MAX_JOINT_SPEED * elapsed + READ_JUMP_TOL_RAD
                if np.any(np.abs(q - m.latest) > allowed):
                    why, q = "reading disagrees with the monitor's samples", None
        except Exception as e:  # noqa: BLE001
            why = f"{type(e).__name__}: {e}"
        if q is None:
            m.hold = f"last command (measured pose unavailable: {why})"
            return
        try:
            self.arm.send_joint_target(q)
        except Exception as e:  # noqa: BLE001 -- the contact itself must still propagate
            m.hold = f"last command (hold command failed: {type(e).__name__}: {e})"
            return
        m.hold = "measured pose [" + " ".join(f"{v:+.3f}" for v in q) + "]"
        if isinstance(exc.event, dict):
            exc.event["hold_q"] = [float(v) for v in q]


def make_arm(cfg: Cfg, kinematics=None) -> ArmBase:
    if cfg.get("template") is True:
        # configs/arms/ros2_generic.yaml and friends: documentation you copy,
        # not a robot you can run. Building one would drive a real transport
        # with placeholder kinematics (empty ros_joints, no home_q).
        raise ValueError(
            "this arm profile is a TEMPLATE (template: true): copy the file, "
            "fill in the robot's own numbers, and drop the flag"
        )
    kind = cfg.type
    if kind == "mock":
        from .mock_arm import MockArm

        return MockArm(cfg, kinematics)
    if kind == "rebot_rs":
        from .rebot_rs_arm import RebotRSArm

        return RebotRSArm(cfg)
    if kind == "rebot_rs_mb":
        from .rebot_rs_mb_arm import RebotRSMotorBridgeArm

        return RebotRSMotorBridgeArm(cfg)
    if kind == "isaac":
        from .isaac_arm import IsaacArm

        return IsaacArm(cfg, kinematics)
    if kind == "mujoco":
        from .mujoco_arm import MujocoArm

        return MujocoArm(cfg, kinematics)
    if kind == "so101":
        from .feetech_arm import FeetechArm

        return FeetechArm(cfg)
    if kind == "ros2":
        from .ros2_arm import Ros2Arm

        return Ros2Arm(cfg)
    if kind == "unitree_arm":
        from .unitree_arm import UnitreeArm

        return UnitreeArm(cfg)
    raise ValueError(
        f"unknown arm type {kind!r} "
        f"(mock|rebot_rs|rebot_rs_mb|isaac|mujoco|so101|ros2|unitree_arm)"
    )
