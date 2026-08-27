"""Safety harness: every joint waypoint is vetted before it reaches a motor.

Checks, in order (cheapest first):
1. e-stop latch and perception watchdog (stale observations halt motion);
2. joint limits with margin;
3. per-joint velocity cap;
4. workspace AABB on the TCP (FK);
5. table-plane clearance for TCP and intermediate joint origins, with a
   grasp-exemption cylinder so the tool may descend onto the active target;
6. optional keep-out AABBs (e.g. the camera tripod).

The arm SDK enforces none of this (joint limits only inside its IK), so this
layer is the difference between "the LLM suggested a pose" and "the arm is
allowed to go there". It fails closed: any violation raises SafetyViolation,
which aborts the streamed motion mid-flight; the e-stop latch is separate
(explicit estop()/SafeArm.stop(), wired to SIGINT in the demo CLI).

Watchdog semantics: perception freshness is checked when a motion BEGINS
(begin_motion()); it is not re-checked per waypoint, because a legitimate
grasp sequence intentionally acts for several seconds without re-observing.

Recovery rule: if the TCP is already below the table clearance when an
exemption disappears (e.g. a failed grasp aborted mid-descent), waypoints
that move the TCP upward are still allowed -- the arm must always be able to
escape vertically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..types import MotionHalted, SafetyViolation


@dataclass
class SafetyLimits:
    workspace_min: np.ndarray  # (3,) base frame, meters
    workspace_max: np.ndarray
    table_z: float = 0.0
    table_clearance: float = 0.02
    max_joint_vel: float = 1.2  # rad/s
    joint_margin: float = 0.02  # rad inside URDF limits
    watchdog_s: float = 5.0  # halt if perception heartbeat older than this
    keep_out: list = field(default_factory=list)  # list of (min(3,), max(3,))
    min_clearance_m: float = 0.03  # TCP/link distance to occupancy obstacles

    @classmethod
    def from_config(cls, cfg) -> "SafetyLimits":
        ws = cfg.workspace
        return cls(
            workspace_min=np.asarray(ws.min, dtype=float),
            workspace_max=np.asarray(ws.max, dtype=float),
            table_z=float(cfg.get("table_z", 0.0)),
            table_clearance=float(cfg.get("table_clearance", 0.02)),
            max_joint_vel=float(cfg.get("max_joint_vel", 1.2)),
            joint_margin=float(cfg.get("joint_margin", 0.02)),
            watchdog_s=float(cfg.get("watchdog_s", 5.0)),
            keep_out=[
                (np.asarray(k["min"], dtype=float), np.asarray(k["max"], dtype=float))
                for k in cfg.get("keep_out", [])
            ],
            min_clearance_m=float(cfg.get("min_clearance_m", 0.03)),
        )


class SafetyHarness:
    def __init__(self, limits: SafetyLimits, kinematics=None, occupancy=None):
        self.limits = limits
        self.kin = kinematics
        # OccupancyMap | None (see perception/occupancy.py). Optional and
        # None by default: no nvblox bridge runs unless configured, and an
        # unconfigured/stale map must never gate motion (booth rule).
        self.occupancy = occupancy
        self._estopped = False
        self._halt: str | None = None
        self._grasp_exempt: tuple[np.ndarray, float, float] | None = None
        self._last_heartbeat = time.monotonic()
        self._motion_active = False
        self.violations: list[str] = []

    # ── state management ─────────────────────────────────────────────────

    def heartbeat(self) -> None:
        """Call whenever fresh perception arrives."""
        self._last_heartbeat = time.monotonic()

    def estop(self, reason: str = "manual") -> None:
        self._estopped = True
        self.violations.append(f"ESTOP: {reason}")

    def reset_estop(self) -> None:
        self._estopped = False

    # ── halt / redirect (VoLo's monitor-halt-redirect) ───────────────────

    def halt(self, reason: str) -> None:
        """Ask the in-flight motion to stop at the next waypoint.

        VoLo (chicychen.github.io/VoLo) names the core requirement of a
        physical agent `monitor - halt - redirect`: the world does not pause
        while the agent thinks, so it must be able to stop an action that has
        become wrong and reissue a different one. HumanCLAW puts the same idea
        in a verifier that rejects a decision before the body executes it.

        This is NOT an e-stop. An e-stop latches and means "the rig is unsafe,
        stop everything until a human clears it". A halt means "this particular
        motion is no longer the right thing to do", clears itself when the next
        motion begins, and leaves the arm powered and controllable. Conflating
        the two would either make halting dangerous to recover from or make the
        e-stop too easy to clear.

        Checked inside `approve()`, so it takes effect within one 50 Hz
        waypoint (20 ms) rather than at the end of the trajectory.
        """
        self._halt = reason
        self.violations.append(f"HALT: {reason}")

    def clear_halt(self) -> None:
        self._halt = None

    @property
    def halted(self) -> str | None:
        return self._halt

    @property
    def estopped(self) -> bool:
        return self._estopped

    def allow_grasp_descent(
        self, center_xy: np.ndarray, radius_m: float = 0.07, z_min: float | None = None
    ) -> None:
        """Open a cylinder over the grasp target where the TCP may go low."""
        z = self.limits.table_z if z_min is None else z_min
        self._grasp_exempt = (np.asarray(center_xy, dtype=float)[:2], radius_m, z)

    def clear_grasp_exemption(self) -> None:
        self._grasp_exempt = None

    def begin_motion(self) -> None:
        """Check perception freshness once, then suspend the watchdog for the
        duration of this motion (grasp sequences legitimately run > watchdog_s
        without a new observation)."""
        if self._estopped:
            raise SafetyViolation("e-stop latched")
        # A halt applies to the motion that was in flight when it was raised,
        # not to every future one. Clearing here (rather than making the caller
        # remember) is what keeps halt recoverable and distinct from e-stop:
        # forget this and the first halt of the session bricks the arm.
        self._halt = None
        age = time.monotonic() - self._last_heartbeat
        if age > self.limits.watchdog_s:
            self._reject(f"perception watchdog: last observation {age:.1f}s old")
        self._motion_active = True

    def end_motion(self) -> None:
        self._motion_active = False

    # ── the gate ─────────────────────────────────────────────────────────

    def approve(self, q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> None:
        """Raise SafetyViolation if the waypoint must not be executed."""
        if self._estopped:
            raise SafetyViolation("e-stop latched")
        if self._halt is not None:
            # MotionHalted subclasses SafetyViolation, so every existing
            # abort path still stops the stream; callers that want to tell
            # "unsafe" from "changed my mind" can catch the subclass.
            raise MotionHalted(f"halted: {self._halt}")
        if not self._motion_active:
            age = time.monotonic() - self._last_heartbeat
            if age > self.limits.watchdog_s:
                self._reject(f"perception watchdog: last observation {age:.1f}s old")

        q_prev = np.asarray(q_prev, dtype=float)
        q_next = np.asarray(q_next, dtype=float)

        if self.kin is not None:
            lo, hi = self.kin.joint_limits
            m = self.limits.joint_margin
            low_bad = q_next < lo + m - 1e-9
            high_bad = q_next > hi - m + 1e-9
            if np.any(low_bad) or np.any(high_bad):
                # Escape rule (mirrors the below-table rule): a joint already
                # at/outside the margin may move STRICTLY back toward the
                # valid band -- otherwise an arm parked exactly on a limit
                # (sim spawn at q=0, drift on the real rig) can never move
                # again. Holds at a violation stay rejected: re-commanding
                # the violated pose drives the motor INTO the limit.
                escaping = True
                for j in np.nonzero(low_bad | high_bad)[0]:
                    if low_bad[j] and q_next[j] > q_prev[j] + 1e-12:
                        continue
                    if high_bad[j] and q_next[j] < q_prev[j] - 1e-12:
                        continue
                    escaping = False
                    break
                if not escaping:
                    bad = int(np.argmax(low_bad | high_bad))
                    self._reject(
                        f"joint {bad + 1} target {q_next[bad]:.3f} rad outside "
                        f"[{lo[bad] + m:.3f}, {hi[bad] - m:.3f}]"
                    )

        if dt > 0:
            vel = np.abs(q_next - q_prev) / dt
            if np.any(vel > self.limits.max_joint_vel):
                bad = int(np.argmax(vel))
                self._reject(
                    f"joint {bad + 1} velocity {vel[bad]:.2f} rad/s exceeds "
                    f"{self.limits.max_joint_vel:.2f}"
                )

        if self.kin is None:
            return

        T = self.kin.fk(q_next)
        tcp = T[:3, 3]
        lo_w, hi_w = self.limits.workspace_min, self.limits.workspace_max
        if np.any(tcp < lo_w) or np.any(tcp > hi_w):
            # Escape rule (mirrors the joint-limit and below-table rules):
            # a TCP already OUTSIDE the workspace box may move strictly
            # toward it -- an arm that spawns/drifts outside (e.g. the
            # straight-up presentation pose has x~0) must always be able to
            # come home, but never wander further out.
            def _dist_to_box(p):
                d = np.maximum(np.maximum(lo_w - p, 0.0), p - hi_w)
                return float(np.linalg.norm(d))

            tcp_prev = self.kin.fk(np.asarray(q_prev, dtype=float))[:3, 3]
            prev_out = np.any(tcp_prev < lo_w) or np.any(tcp_prev > hi_w)
            if not (prev_out and _dist_to_box(tcp) <= _dist_to_box(tcp_prev) + 1e-3):
                self._reject(
                    f"TCP {np.round(tcp, 3).tolist()} outside workspace "
                    f"[{lo_w.tolist()} .. {hi_w.tolist()}]"
                )

        floor = self.limits.table_z + self.limits.table_clearance
        if tcp[2] < floor and not self._in_grasp_cylinder(tcp):
            # Escape rule: if the TCP is ALREADY below the floor (aborted
            # descent, exemption since cleared), allow ascending moves and
            # in-place holds so the arm can always recover upward -- but no
            # lateral sliding below the clearance plane.
            T_prev = self.kin.fk(q_prev)
            prev_z = float(T_prev[2, 3])
            dxy = float(np.linalg.norm(tcp[:2] - T_prev[:2, 3]))
            ascending = tcp[2] > prev_z + 1e-9 or (dxy < 1e-6 and tcp[2] >= prev_z - 1e-9)
            if not (prev_z < floor and ascending):
                self._reject(
                    f"TCP z={tcp[2]:.3f} below table clearance {floor:.3f} "
                    "outside the grasp exemption zone"
                )

        for kmin, kmax in self.limits.keep_out:
            if np.all(tcp >= kmin) and np.all(tcp <= kmax):
                self._reject(f"TCP inside keep-out zone {kmin.tolist()}..{kmax.tolist()}")

        # Coarse link check: joint origins must stay above the table too
        # (elbow scooping the table is the classic failure). The LAST link is
        # the gripper_end/TCP itself -- it legitimately descends to the
        # object during a grasp and is already governed by the TCP clearance
        # + grasp-exemption check above, so excluding it here avoids a false
        # "link would hit the table" abort when the jaws close on a low
        # object inside the exemption cylinder.
        links = self.kin.link_positions(q_next)
        elbow_links = links[1:-1] if len(links) > 2 else links[1:]
        for i, p in enumerate(elbow_links, start=2):
            if p[2] < self.limits.table_z + 0.01 and not self._in_grasp_cylinder(p):
                self._reject(f"link/joint {i} at z={p[2]:.3f} would hit the table")

        if self.occupancy is not None:
            reason = self._occupancy_violation(
                np.vstack([tcp[None, :], links[1:]]), self._grasp_exempt
            )
            if reason is not None:
                self._reject(reason)

    def vet_pose(
        self,
        q: np.ndarray,
        exempt_xy: np.ndarray | None = None,
        exempt_radius_m: float = 0.07,
        exempt_z_min: float | None = None,
    ) -> str | None:
        """Statically vet a candidate joint pose BEFORE any motion exists.

        Runs the same geometric gates approve() applies per waypoint (joint
        limits, workspace AABB, table clearance, keep-outs) under the grasp
        exemption cylinder that WILL be opened during the descent, and
        returns the rejection reason instead of raising. Grasp selection
        uses this to discard doomed candidates up front -- the harness
        stays the runtime backstop, but a candidate that would abort
        mid-descent should never win the ranking in the first place.
        Escape rules do not apply here: this vets a chosen target, not a
        recovery move.
        """
        if self.kin is None:
            return None
        q = np.asarray(q, dtype=float)
        exempt = None
        if exempt_xy is not None:
            z_min = self.limits.table_z if exempt_z_min is None else float(exempt_z_min)
            exempt = (np.asarray(exempt_xy, dtype=float)[:2], float(exempt_radius_m), z_min)

        lo, hi = self.kin.joint_limits
        m = self.limits.joint_margin
        low_bad = q < lo + m - 1e-9
        high_bad = q > hi - m + 1e-9
        if np.any(low_bad) or np.any(high_bad):
            bad = int(np.argmax(low_bad | high_bad))
            return (
                f"joint {bad + 1} target {q[bad]:.3f} rad outside "
                f"[{lo[bad] + m:.3f}, {hi[bad] - m:.3f}]"
            )

        tcp = self.kin.fk(q)[:3, 3]
        lo_w, hi_w = self.limits.workspace_min, self.limits.workspace_max
        if np.any(tcp < lo_w) or np.any(tcp > hi_w):
            return (
                f"TCP {np.round(tcp, 3).tolist()} outside workspace "
                f"[{lo_w.tolist()} .. {hi_w.tolist()}]"
            )

        floor = self.limits.table_z + self.limits.table_clearance
        if tcp[2] < floor and not self._in_cylinder(tcp, exempt):
            return f"TCP z={tcp[2]:.3f} below table clearance {floor:.3f}"

        for kmin, kmax in self.limits.keep_out:
            if np.all(tcp >= kmin) and np.all(tcp <= kmax):
                return f"TCP inside keep-out zone {kmin.tolist()}..{kmax.tolist()}"

        links = self.kin.link_positions(q)
        for i, p in enumerate(links[1:], start=2):
            if p[2] < self.limits.table_z + 0.01 and not self._in_cylinder(p, exempt):
                return f"link/joint {i} at z={p[2]:.3f} would hit the table"

        if self.occupancy is not None:
            reason = self._occupancy_violation(np.vstack([tcp[None, :], links[1:]]), exempt)
            if reason is not None:
                return reason
        return None

    def _occupancy_violation(self, points: np.ndarray, exempt: tuple | None) -> str | None:
        """nvblox-backed check: any query point closer than min_clearance_m
        to a cached obstacle, outside the active grasp exemption.

        `points.clearance()` returns None when the map has no fresh data
        (never refreshed, stale, or the bridge is down) -- that is
        indistinguishable from "not configured" on purpose: a stalled
        nvblox bridge must degrade the same way a missing one does, never
        freeze the arm.
        """
        dist = self.occupancy.clearance(points)
        if dist is None:
            return None
        min_c = self.limits.min_clearance_m
        for i, (p, d) in enumerate(zip(points, dist)):
            if d < min_c and not self._in_cylinder(p, exempt):
                return f"point {i} clearance {d:.3f} m below {min_c:.3f} m (nvblox occupancy)"
        return None

    def _in_grasp_cylinder(self, p: np.ndarray) -> bool:
        return self._in_cylinder(p, self._grasp_exempt)

    @staticmethod
    def _in_cylinder(p: np.ndarray, exempt: tuple | None) -> bool:
        if exempt is None:
            return False
        center_xy, radius, z_min = exempt
        return (
            np.linalg.norm(np.asarray(p[:2]) - center_xy) <= radius
            and p[2] >= z_min - 1e-6
        )

    def _reject(self, reason: str) -> None:
        self.violations.append(reason)
        raise SafetyViolation(reason)


class SafeArm:
    """Wrap an ArmBase so every motion passes through the harness.

    This is the only handle the skill runtime gets: raw arm objects never
    leak upward to the agent layer.
    """

    def __init__(self, arm, harness: SafetyHarness):
        self._arm = arm
        self.harness = harness

    @property
    def n_joints(self) -> int:
        return self._arm.n_joints

    def get_state(self):
        return self._arm.get_state()

    def move_joints(self, q_target: np.ndarray, duration_s: float = 2.0,
                    **backend_kw) -> bool:
        # Min-jerk peak velocity is 1.875 * dq / T; stretch the duration so
        # the planned profile stays safely under the cap (harness remains the
        # backstop for anything else).
        dq_max = float(np.max(np.abs(np.asarray(q_target, dtype=float) - self._arm.get_state().q)))
        needed = 1.875 * dq_max / (0.9 * self.harness.limits.max_joint_vel)
        duration_s = max(duration_s, needed)
        self.harness.begin_motion()  # perception-freshness check happens here
        try:
            # `backend_kw` forwards backend-specific hints (e.g. a measured
            # descend-bias compensation) without this layer knowing what they
            # mean. Silently dropped by backends that do not accept them, so a
            # hint never becomes a hard dependency.
            try:
                return self._arm.stream_to(q_target, duration_s,
                                           approve=self.harness.approve,
                                           **backend_kw)
            except TypeError:
                if not backend_kw:
                    raise
                return self._arm.stream_to(q_target, duration_s,
                                           approve=self.harness.approve)
        finally:
            self.harness.end_motion()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        if self.harness.estopped:
            raise SafetyViolation("e-stop latched")
        self._arm.set_gripper(pos, effort)

    def stop(self) -> None:
        self.harness.estop("stop requested")
        self._arm.stop()

    @property
    def raw(self):
        """Escape hatch for backend-specific ops (e.g. two-stage close)."""
        return self._arm
