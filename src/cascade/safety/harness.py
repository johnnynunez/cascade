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
import threading
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
    #: minimum clearance between THIS arm's links and a neighbour arm's links,
    #: in the shared table frame. Only used when neighbours are registered
    #: (see SafetyHarness.add_neighbor); a single-arm rig never pays for it.
    #:
    #: With `link_radii_m` declared on both arms this is a margin between link
    #: SURFACES; without radii it is a centreline distance and has to absorb
    #: the link thickness itself. MEASURED against MuJoCo on the dual-SO-101
    #: profiles (tests/test_multi_arm_physics.py), a 0.05 m centreline gate
    #: approves pose pairs whose collision meshes already overlap -- the
    #: links are up to 9 cm thick -- while the surface gate keeps every
    #: approved pair at least the margin apart in physics (the dual profiles
    #: use 0.02 m: two 50 Hz ticks at the velocity cap; 87% of random joint
    #: space approved, centreline 0.05 m approved 99% including overlaps).
    neighbor_clearance_m: float = 0.05
    #: per-SEGMENT link thickness, in kinematic order (one entry per segment
    #: of `SafetyHarness.link_points_table_frame`: joint origins then TCP).
    #: Each radius must envelope every collision surface of that segment at
    #: every pose. None = zero-radius lines (the pre-2026-10 behaviour). A
    #: list of the wrong length is a config error and fails closed.
    link_radii_m: tuple[float, ...] | None = None

    @classmethod
    def from_config(cls, cfg) -> "SafetyLimits":
        ws = cfg.workspace
        radii = cfg.get("link_radii_m")
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
            neighbor_clearance_m=float(cfg.get("neighbor_clearance_m", 0.10)),
            link_radii_m=None if radii is None else tuple(float(r) for r in radii),
        )


class SafetyHarness:
    def __init__(self, limits: SafetyLimits, kinematics=None, occupancy=None,
                 base_pose=None):
        self.limits = limits
        self.kin = kinematics
        # OccupancyMap | None (see perception/occupancy.py). Optional and
        # None by default: no occupancy bridge runs unless configured, and an
        # unconfigured/stale map must never gate motion (booth rule).
        self.occupancy = occupancy
        # Where this arm is bolted, as a 4x4 base->table transform. None means
        # "the base frame IS the table frame", which is exactly true for a
        # single-arm rig and is why nothing here changes for one arm.
        #
        # This is the piece that makes an inter-arm check possible at all:
        # every position in this codebase is in the ROBOT BASE frame, so two
        # arms' link positions are not comparable until both are lifted into
        # one shared frame. RPent solves the same problem by declaring one
        # arm's base the canonical frame; a table frame is the same idea
        # without privileging a robot.
        self.base_pose = None if base_pose is None else np.asarray(base_pose, dtype=float)
        #: name -> callable returning that arm's link points in the TABLE
        #: frame, or None when it cannot be read cheaply/safely.
        self._neighbors: dict = {}
        #: name -> that arm's per-segment link radii (or None = centrelines)
        self._neighbor_radii: dict = {}
        self._check_own_link_radii()
        self._estopped = False
        self._halt: str | None = None
        self._halt_generation = 0
        # A passive goal ledger can commit under this short lock after doing
        # all geometry, SDK reads and I/O outside it. Lock order: world then
        # this lock; stop/reset never acquire a world lock. This token also
        # remembers estop -> reset without changing legacy halt generations.
        self._observation_lock = threading.Lock()
        self._observation_cancel_generation = 0
        self._pending_contact_episode = None
        self._contact_scope = threading.local()
        self._pending_release_episode = None
        self._release_scope = threading.local()
        # Optional model-backed empty-tool recovery. Its adapter owns the
        # geometry; this common gate remains SDK-independent.
        self._pending_model_withdrawal = None
        self._model_withdrawal_scope = threading.local()
        self._grasp_exempt: tuple[np.ndarray, float, float] | None = None
        self._last_heartbeat = time.monotonic()
        self._motion_active = False
        self.violations: list[str] = []

    # ── state management ─────────────────────────────────────────────────

    def heartbeat(self) -> None:
        """Call whenever fresh perception arrives."""
        self._last_heartbeat = time.monotonic()

    def estop(self, reason: str = "manual") -> None:
        with self._observation_lock:
            self._estopped = True
            self._observation_cancel_generation += 1
        self.violations.append(f"ESTOP: {reason}")

    def reset_estop(self) -> None:
        with self._observation_lock:
            self._estopped = False

    def check_motion_cancellation(self, token: int | None) -> None:
        """Keep an observed command context invalid after stop is cleared.

        Only inspect the monotonic token/latch under this short lock. No
        callback, geometry, device read or command may run while holding it.
        """
        if token is None:
            return
        if type(token) is not int or token < 0:
            raise SafetyViolation('motion cancellation token must be a nonnegative integer')
        with self._observation_lock:
            if token != self._observation_cancel_generation or self._estopped:
                raise SafetyViolation('observed motion context was cancelled')

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
        with self._observation_lock:
            self._halt = reason
            self._halt_generation += 1
            self._observation_cancel_generation += 1
        self.violations.append(f"HALT: {reason}")

    def clear_halt(self) -> None:
        with self._observation_lock:
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
        candidate = (np.asarray(center_xy, dtype=float)[:2].copy(), radius_m, z)
        if self._pending_model_withdrawal is not None:
            raise SafetyViolation("retained model withdrawal cannot change its grasp cylinder")
        pending = self._pending_release_episode
        if pending is not None:
            scope = getattr(self._release_scope, "value", None)
            original = pending["cylinder"]
            if (scope is None or scope[0] is not pending or scope[1] != "retreat"
                    or not np.array_equal(candidate[0], original[0]) or candidate[1:] != original[1:]):
                raise SafetyViolation("retained release only permits its original scoped cylinder")
        self._grasp_exempt = candidate

    def clear_grasp_exemption(self) -> None:
        if self._pending_model_withdrawal is not None:
            self.check_model_withdrawal(cleanup=True)
        pending = self._pending_release_episode
        if pending is not None:
            scope = getattr(self._release_scope, "value", None)
            if scope is None or scope[0] is not pending or scope[1] != "cleanup":
                raise SafetyViolation("retained release cylinder cleanup requires its episode scope")
        self._grasp_exempt = None

    # ── inter-arm awareness ──────────────────────────────────────────────

    def _check_own_link_radii(self) -> None:
        """Fail at construction, not mid-motion, when `link_radii_m` does not
        describe this arm's chain (one radius per segment of
        `link_points_table_frame`). Without kinematics there is no chain and
        no gate, so nothing to compare; a kinematics stub that cannot produce
        a chain here is checked at the first gate instead (it blocks there)."""
        radii = self.limits.link_radii_m
        if radii is None or self.kin is None:
            return
        n = getattr(self.kin, "n", None)
        if n is None:
            limits = getattr(self.kin, "joint_limits", None)
            n = None if limits is None else len(limits[0])
        if n is None:
            return
        try:
            pts = self.link_points_table_frame(np.zeros(int(n)))
        except Exception:
            return
        if pts is None:
            return
        n_segments = max(int(pts.shape[0]) - 1, 1)
        if len(radii) != n_segments:
            raise ValueError(
                f"safety.link_radii_m has {len(radii)} entries but this arm's chain has "
                f"{n_segments} segments ({pts.shape[0]} points: joint origins then TCP)"
            )

    def add_neighbor(self, name: str, points_in_table_frame, link_radii_m=None) -> None:
        """Register another arm whose links this one must not hit.

        `points_in_table_frame` is a zero-argument callable returning that
        arm's link points as (N, 3) in the SHARED TABLE frame, or None when
        the pose cannot be read. Passing a callable rather than the arm keeps
        this layer ignorant of arm objects (and lets the caller decide what is
        safe to touch -- notably, never materializing a standby LazyArm).

        `link_radii_m` is that arm's per-segment thickness (its own
        `SafetyLimits.link_radii_m`); the gate subtracts both arms' radii so
        the clearance it enforces is between link surfaces. None means the
        neighbour is measured as centrelines.

        Returning None means "unknown", and unknown means SKIP, never
        "blocked": same booth rule as the occupancy map. An arm that cannot
        see its neighbour must not freeze mid-demo -- it falls back to the
        static workspace/keep-out partition, which is still enforced. A radii
        list that does not match the neighbour's chain is NOT unknown, it is
        a config error, and the gate refuses the motion naming it.
        """
        self._neighbors[str(name)] = points_in_table_frame
        self._neighbor_radii[str(name)] = (
            None if link_radii_m is None else tuple(float(r) for r in link_radii_m))

    def link_points_table_frame(self, q: np.ndarray) -> np.ndarray | None:
        """This arm's link chain at pose q, in the TABLE frame.

        Returned in KINEMATIC ORDER (base joint first, TCP last) so that
        consecutive points bound one physical link and the chain can be read
        as a polyline. That ordering is load-bearing for the segment check:
        putting the TCP first (as an earlier revision did) would invent a
        segment from the tool back to the base joint and measure a link that
        does not exist.

        Returns None without kinematics (nothing to compute from). With no
        `base_pose` the base frame IS the table frame, so points pass through
        unchanged -- which is why a single-arm rig is unaffected.
        """
        if self.kin is None:
            return None
        q = np.asarray(q, dtype=float)
        pts = np.vstack([self.kin.link_positions(q), self.kin.fk(q)[:3, 3][None, :]])
        if self.base_pose is None:
            return pts
        from ..types import transform_points

        return transform_points(self.base_pose, pts)

    def _neighbor_clearances(self, q: np.ndarray):
        """Per readable neighbour: `(name, clearance, my_link, their_link,
        what)` at pose q -- surface clearance when either arm declares
        `link_radii_m`, centreline distance otherwise -- or a reason string
        when a neighbour's radii do not describe its chain (a config error,
        refused rather than skipped). Unreadable neighbours are omitted."""
        mine = self.link_points_table_frame(q)
        if mine is None:
            return []
        from .geometry import chain_distance

        my_radii = self.limits.link_radii_m
        out = []
        for name, source in self._neighbors.items():
            try:
                theirs = source()
            except Exception:
                continue  # unreadable neighbour = unknown = skip
            if theirs is None:
                continue
            theirs = np.asarray(theirs, dtype=float).reshape(-1, 3)
            if theirs.size == 0:
                continue
            their_radii = self._neighbor_radii.get(name)
            try:
                worst, i, j = chain_distance(mine, theirs, my_radii, their_radii)
            except ValueError as exc:
                return (
                    f"inter-arm clearance to {name!r} cannot be measured: link_radii_m "
                    f"does not describe the chain ({exc})"
                )
            what = "link surfaces" if (my_radii is not None or their_radii is not None) else "link centrelines"
            out.append((name, float(worst), int(i), int(j), what))
        return out

    def _neighbor_violation(self, q_next: np.ndarray) -> str | None:
        """Closest approach between this arm's LINKS and each neighbour's.

        Segment-to-segment, not point-to-point. Sampling only joint origins
        leaves a real blind spot: a link can pass through the gap between two
        of a neighbour's origins with every origin far from every other one,
        so the point check reports "clear" while the links nearly touch.
        MEASURED over 4000 random pose pairs on the shipped dual-SO-101
        profiles, point-to-point overestimates clearance by 0.1 cm on average
        but by up to 3.3 cm in the worst case -- and that worst case is a pose
        whose links are 3.3 cm apart being reported as 6.6 cm, i.e. accepted
        by a 5 cm gate that should have rejected it.

        Segments are still not the real geometry: a link has thickness. With
        `link_radii_m` on both arms the distance compared against
        `neighbor_clearance_m` is between link SURFACES (centreline distance
        minus both radii), which MEASURED against MuJoCo is a lower bound on
        the physical clearance (tests/test_multi_arm_physics.py). Without
        radii the gate is a centreline gate and the margin must absorb the
        thickness itself -- on the SO-101 it cannot: 0.05 m approves overlaps.

        This is the STRICT gate (one pose, no history): `vet_pose` uses it
        for a candidate, and approve() calls it first for every waypoint
        before `_neighbor_gate` applies the retreat rule.
        """
        if not self._neighbors:
            return None
        tol = float(self.limits.neighbor_clearance_m)
        if tol <= 0:
            return None
        clearances = self._neighbor_clearances(q_next)
        if isinstance(clearances, str):
            return clearances
        for name, worst, i, j, what in clearances:
            if worst < tol:
                return (
                    f"inter-arm clearance {worst:.3f} m to {name!r} "
                    f"(my link {i} vs their link {j}, {what}) below {tol:.3f} m"
                )
        return None

    def _neighbor_gate(self, q_prev: np.ndarray, q_next: np.ndarray) -> str | None:
        """approve()'s inter-arm check: the strict gate, then RETREAT IS OPEN.

        An arm that is ALREADY inside the margin at `q_prev` (the stream's
        current pose) may take any waypoint that does not bring it closer to
        that neighbour -- never one that does, and never a waypoint that
        enters the margin from outside. Without this a gate that cannot be
        escaped strands both robots. MEASURED on physics: the right SO-101
        reached its approved inward pose with 0.032 m of surface clearance,
        settled 0.1 mm closer under gravity, and was then refused its own
        park home. `vet_pose` (a candidate, no current pose) stays strict.
        """
        reason = self._neighbor_violation(q_next)
        if reason is None or q_prev is None:
            return reason
        tol = float(self.limits.neighbor_clearance_m)
        before = self._neighbor_clearances(q_prev)
        after = self._neighbor_clearances(q_next)
        if isinstance(before, str) or isinstance(after, str):
            return reason
        now_by_name = {name: worst for name, worst, *_ in before}
        for name, worst, i, j, what in after:
            if worst >= tol:
                continue
            now = now_by_name.get(name)
            if now is None or now >= tol:
                return reason  # entering the margin from outside: refused
            if worst < now - 1e-6:
                return (
                    f"inter-arm clearance {worst:.3f} m to {name!r} "
                    f"(my link {i} vs their link {j}, {what}) below {tol:.3f} m and "
                    f"closing from {now:.3f} m: only a retreat is allowed from here"
                )
        return None  # already inside, not closing in on anyone: retreat

    def _check_halt_generation(self, expected: int | None) -> None:
        if expected is not None and expected != self._halt_generation:
            raise MotionHalted("halt received during route planning or execution")

    def check_contact_episode(self, *, gripper=False) -> None:
        """A failed close remains stationary until its explicit reset retreat.

        Authority belongs to the initiating thread and the exact retained
        episode. A concurrent skill cannot borrow the recovery exemption.
        """
        self.check_release_episode(gripper=gripper)
        pending = self._pending_contact_episode
        scope = getattr(self._contact_scope, "value", None)
        if pending is not None and (scope is None or scope[0] is not pending
                                    or (gripper and not scope[1])):
            raise SafetyViolation("unfinished contact episode; explicit reset_scene recovery required")
        if pending is not None:
            self._check_halt_generation(pending["halt_generation"])

    def check_release_episode(self, *, gripper=False, target=None, duration=None, grip=None):
        """Only the exact original release/withdrawal may use a retained scope."""
        self.check_model_withdrawal(gripper=gripper)
        pending = self._pending_release_episode
        if pending is None:
            return
        scope = getattr(self._release_scope, "value", None)
        expected = "open" if gripper else "retreat"
        if scope is None or scope[0] is not pending or scope[1] != expected:
            raise SafetyViolation("unfinished release episode; explicit reset_scene recovery required")
        self._check_halt_generation(pending["halt_generation"])
        if self.occupancy.scene_reset_generation != pending["scene_generation"]:
            raise SafetyViolation("release episode scene identity changed")
        if not gripper and (self.occupancy._contact_paths != ()
                            or self.occupancy._payload_epoch != pending["clock"]["epoch"]
                            or self.occupancy.is_stale()):
            raise SafetyViolation("released withdrawal contact or producer identity changed")
        original, active = pending["cylinder"], self._grasp_exempt
        if (active is None or not np.array_equal(active[0], original[0]) or active[1:] != original[1:]):
            raise SafetyViolation("release contact cylinder changed")
        if gripper and grip is not None and grip != pending["open_position"]:
            raise SafetyViolation("release episode only authorizes the original opening")
        if target is not None and (not np.array_equal(np.asarray(target), pending["q_retreat"])
                                   or duration != pending["duration_s"]):
            raise SafetyViolation("release episode only authorizes its exact original retreat")

    def check_model_withdrawal(self, **operation):
        pending = self._pending_model_withdrawal
        if pending is not None:
            pending.check_actuation(getattr(self._model_withdrawal_scope, "value", None),
                                    **operation)

    def begin_motion(self, *, halt_generation: int | None = None) -> None:
        """Check perception freshness once, then suspend the watchdog for the
        duration of this motion (grasp sequences legitimately run > watchdog_s
        without a new observation)."""
        self.check_contact_episode()
        if self._estopped:
            raise SafetyViolation("e-stop latched")
        # A halt before the caller started is recoverable; a new halt during
        # route planning or between its segments must cancel that same route.
        # A halt applies to the motion that was in flight when it was raised,
        # not to every future one. Clearing here (rather than making the caller
        # remember) is what keeps halt recoverable and distinct from e-stop:
        # forget this and the first halt of the session bricks the arm.
        with self._observation_lock:
            self._check_halt_generation(halt_generation)
            self._halt = None
        age = time.monotonic() - self._last_heartbeat
        if age > self.limits.watchdog_s:
            self._reject(f"perception watchdog: last observation {age:.1f}s old")
        self._motion_active = True

    def end_motion(self) -> None:
        self._motion_active = False

    @property
    def motion_active(self) -> bool:
        """True between begin_motion() and end_motion() (read-only; the
        extrinsic drift monitor checks only a static arm)."""
        return bool(self._motion_active)

    def check_stream_start(self, *, halt_generation: int | None = None) -> None:
        """Recheck live guards after planning without clearing a new halt."""
        self.check_contact_episode()
        if self._estopped:
            raise SafetyViolation("e-stop latched")
        self._check_halt_generation(halt_generation)
        if self._halt is not None:
            raise MotionHalted(f"halted: {self._halt}")
        age = time.monotonic() - self._last_heartbeat
        if age > self.limits.watchdog_s:
            self._reject(f"perception watchdog: last observation {age:.1f}s old")

    # ── the gate ─────────────────────────────────────────────────────────

    def approve(self, q_prev: np.ndarray, q_next: np.ndarray, dt: float,
                joint_margin: float | None = None) -> None:
        """Raise SafetyViolation if the waypoint must not be executed.

        `joint_margin` overrides the configured safety margin for this one
        move -- the park drives a joint onto its mechanical stop (exact zero),
        which the normal margin would reject. None keeps `limits.joint_margin`.
        Every other gate (workspace, table clearance, keep-outs, velocity,
        neighbours) still runs at full strength.
        """
        self._approve_step(q_prev, q_next, dt, joint_margin, self._reject, True)

    def vet_step(self, q_prev: np.ndarray, q_next: np.ndarray, dt: float,
                 joint_margin: float | None = None) -> str | None:
        """Preflight the same geometry/escape gates without recording a violation.

        This cannot authorize motion: begin_motion still checks the live watchdog
        and clears a previous halt, and approve still checks every streamed step.
        """
        def reject(reason):
            raise SafetyViolation(reason)

        try:
            self._approve_step(q_prev, q_next, dt, joint_margin, reject, False)
        except SafetyViolation as exc:
            return str(exc)
        return None

    def _approve_step(self, q_prev, q_next, dt, joint_margin, reject, live_checks):
        self.check_contact_episode()
        if self._estopped:
            raise SafetyViolation("e-stop latched")
        if live_checks and self._halt is not None:
            # MotionHalted subclasses SafetyViolation, so every existing
            # abort path still stops the stream; callers that want to tell
            # "unsafe" from "changed my mind" can catch the subclass.
            raise MotionHalted(f"halted: {self._halt}")
        if live_checks and not self._motion_active:
            age = time.monotonic() - self._last_heartbeat
            if age > self.limits.watchdog_s:
                reject(f"perception watchdog: last observation {age:.1f}s old")

        q_prev = np.asarray(q_prev, dtype=float)
        q_next = np.asarray(q_next, dtype=float)

        if self.kin is not None:
            lo, hi = self.kin.joint_limits
            m = self.limits.joint_margin if joint_margin is None else float(joint_margin)
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
                    reject(
                        f"joint {bad + 1} target {q_next[bad]:.3f} rad outside "
                        f"[{lo[bad] + m:.3f}, {hi[bad] - m:.3f}]"
                    )

        if dt > 0:
            vel = np.abs(q_next - q_prev) / dt
            if np.any(vel > self.limits.max_joint_vel):
                bad = int(np.argmax(vel))
                reject(
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
                reject(
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
                reject(
                    f"TCP z={tcp[2]:.3f} below table clearance {floor:.3f} "
                    "outside the grasp exemption zone"
                )

        for kmin, kmax in self.limits.keep_out:
            if np.all(tcp >= kmin) and np.all(tcp <= kmax):
                reject(f"TCP inside keep-out zone {kmin.tolist()}..{kmax.tolist()}")

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
                reject(f"link/joint {i} at z={p[2]:.3f} would hit the table")

        if self.occupancy is not None:
            reason = self._occupancy_violation(
                np.vstack([tcp[None, :], links[1:]]), self._grasp_exempt
            )
            if reason is not None:
                reject(reason)

            reason = self._payload_occupancy_violation(q_next, self._grasp_exempt)
            if reason is not None:
                reject(reason)

        # Inter-arm proximity, LAST because it is the only check that reads
        # another robot's live state. No neighbours registered = no cost, so
        # a single-arm rig runs the identical code path it always did.
        reason = self._neighbor_gate(q_prev, q_next)
        if reason is not None:
            reject(reason)

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
            reason = self._payload_occupancy_violation(q, exempt)
            if reason is not None:
                return reason
        # Same inter-arm gate approve() applies per waypoint, so a grasp
        # candidate that would abort against the neighbour is discarded at
        # ranking time instead of failing mid-descent.
        return self._neighbor_violation(q)

    def _payload_occupancy_violation(self, q, exempt):
        points_fn = getattr(self.occupancy, "payload_points", None)
        if points_fn is None:
            return None
        pose = self.kin.fk(q)
        if self.base_pose is not None:
            pose = self.base_pose @ pose
        points = points_fn(pose)
        if not len(points):
            return None
        reason = self._occupancy_violation(points, exempt, attached=True)
        return f"attached object {reason}" if reason else None

    def _occupancy_violation(self, points: np.ndarray, exempt: tuple | None, *, attached=False) -> str | None:
        """Occupancy-map check: any query point closer than min_clearance_m
        to a cached obstacle, outside the active grasp exemption.

        `points.clearance()` returns None when the map has no fresh data
        (never refreshed, stale, or the bridge is down) -- that is
        unavailable optional map skips this check. Required maps and carried
        surfaces instead need observed distance-grid support; their missing
        or unknown observations must not admit motion.
        """
        query = self.occupancy.payload_clearance if attached else self.occupancy.clearance
        observed_required = attached or getattr(self.occupancy, "required", False)
        dist = query(points)
        if dist is None:
            return "clearance unavailable (fresh observed distance grid required)" if observed_required else None
        min_c = self.limits.min_clearance_m
        for i, (p, d) in enumerate(zip(points, dist)):
            if not np.isfinite(d) and observed_required and not self._in_cylinder(p, exempt):
                return (f"point {i} has unobserved clearance (occupancy map)"
                        f" at [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
            if d < min_c and not self._in_cylinder(p, exempt):
                return (f"point {i} clearance {d:.3f} m below {min_c:.3f} m (occupancy map)"
                        f" at [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
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

    def __init__(self, arm, harness: SafetyHarness, motion_planner=None):
        self._arm = arm
        self.harness = harness
        self.motion_planner = motion_planner

    @property
    def n_joints(self) -> int:
        return self._arm.n_joints

    def connect(self) -> None:
        """Bring the backend up.

        Lifecycle is part of the arm surface, not a backend extra: the arm rig
        holds SafeArms and must be able to connect/disconnect every member
        without reaching through `.raw`. Both calls are on LazyArm's own
        surface (connect is a deliberate no-op there, disconnect only acts on
        an already-materialized arm), so neither powers motors as a side
        effect of lifecycle management.
        """
        self._arm.connect()

    def disconnect(self) -> None:
        """Release the backend. On a real arm this disables torque, so the
        caller must have parked the arm first (see move_home)."""
        try:
            self._arm.disconnect()
        finally:
            if self.motion_planner is not None:
                self.motion_planner.close()

    def get_state(self, **kwargs):
        return self._arm.get_state(**kwargs)

    def move_joints(self, q_target: np.ndarray, duration_s: float = 2.0,
                    joint_margin: float | None = None, _preflight=None,
                    _halt_generation: int | None = None,
                    _trajectory_preflight=None,
                    _linear_tool_path=False,
                    _cancellation_token: int | None = None,
                    **backend_kw) -> bool:
        if _cancellation_token is not None:
            self.harness.check_motion_cancellation(_cancellation_token)
        self.harness.check_model_withdrawal(command=True, target=q_target,
                                           duration=duration_s, joint_margin=joint_margin)
        if self.motion_planner is not None:
            if _cancellation_token is not None:
                raise SafetyViolation('observed cancellation context requires the bound joint streamer')
            from ..planning.runtime import execute
            try:
                return execute(self, q_target, duration_s, joint_margin=joint_margin,
                               legacy_preflight=_preflight, trajectory_preflight=_trajectory_preflight,
                               halt_generation=_halt_generation, linear_tool_path=_linear_tool_path,
                               **backend_kw)
            finally:
                self.harness.end_motion()
        if _preflight is not None and any(key in backend_kw for key in ("preflight", "before_stream")):
            # Route validation and observed-scene callbacks have distinct
            # targets. Never replace one safety callback with the other.
            raise SafetyViolation("ambiguous motion preflight callbacks; no motion sent")
        # Min-jerk peak velocity is 1.875 * dq / T; stretch the duration so
        # the planned profile stays safely under the cap (harness remains the
        # backstop for anything else).
        self.harness.check_release_episode(target=q_target, duration=duration_s)
        start_state = self._arm.get_state()
        if _cancellation_token is not None:
            self.harness.check_motion_cancellation(_cancellation_token)
        feedback_guard = backend_kw.get("feedback_guard")
        if feedback_guard is not None:
            # Reuse the stretching read. A lost attachment or cancelled
            # observed-finger state must not disappear behind a later sample.
            try:
                feedback_guard(start_state)
            except TypeError as exc:
                raise SafetyViolation("motion safety callback failed; no unguarded retry") from exc
        dq_max = float(np.max(np.abs(np.asarray(q_target, dtype=float) - start_state.q)))
        needed = 1.875 * dq_max / (0.9 * self.harness.limits.max_joint_vel)
        duration_s = max(duration_s, needed)
        if _halt_generation is None:
            self.harness.begin_motion()
        else:
            self.harness.begin_motion(halt_generation=_halt_generation)
        approve = self.harness.approve
        if joint_margin is not None:
            # The park drives onto the mechanical stop; relax only the joint
            # margin for THIS move (see SafetyHarness.approve). The closure
            # keeps the stream's per-waypoint approve() call signature intact.
            h = self.harness
            jm = float(joint_margin)

            def approve(q_prev, q_next, dt):
                h.approve(q_prev, q_next, dt, joint_margin=jm)

        if _preflight is not None:
            # Backends run this AFTER reading their start state but BEFORE
            # their pacing clock. A slow preflight cannot create a burst of
            # targets trying to catch up with expired streaming deadlines.
            backend_kw = {**backend_kw, "preflight": _preflight,
                          "before_stream": lambda: self.harness.check_stream_start(
                              halt_generation=_halt_generation)}

        if _cancellation_token is not None:
            original_approve, original_before = approve, backend_kw.get('before_stream')
            def approve(q_prev, q_next, dt):
                self.harness.check_motion_cancellation(_cancellation_token)
                original_approve(q_prev, q_next, dt)
                self.harness.check_motion_cancellation(_cancellation_token)
            def before_stream():
                self.harness.check_motion_cancellation(_cancellation_token)
                if original_before is not None:
                    original_before()
                self.harness.check_motion_cancellation(_cancellation_token)
            backend_kw = {**backend_kw, 'before_stream': before_stream}

        try:
            # `backend_kw` forwards backend-specific hints (e.g. a measured
            # descend-bias compensation) without this layer knowing what they
            # mean. Silently dropped by backends that do not accept them, so a
            # hint never becomes a hard dependency.
            try:
                result = self._arm.stream_to(q_target, duration_s,
                                             approve=approve,
                                             **backend_kw)
                if _cancellation_token is not None:
                    self.harness.check_motion_cancellation(_cancellation_token)
                return result
            except TypeError as exc:
                if any(key in backend_kw for key in ("preflight", "before_stream", "feedback_guard")):
                    # Safety callbacks are mandatory, never backend hints. A
                    # TypeError after a target must not retry without them.
                    raise SafetyViolation("motion safety callback failed; no unguarded retry") from exc
                if not backend_kw:
                    raise
                return self._arm.stream_to(q_target, duration_s,
                                           approve=approve)
        finally:
            self.harness.end_motion()

    def move_planned(self, q_target: np.ndarray, duration_s: float = 3.0, *,
                     _halt_generation=None, rate_hz=None) -> bool:
        """Execute a fully vetted deterministic route, retaining live gates."""
        # A retained adapter executes its individually revalidated segments;
        # an unrelated caller cannot borrow that scope via this route API.
        self.harness.check_model_withdrawal(command=True, planned=True)
        if self.motion_planner is not None:
            return self.move_joints(q_target, duration_s, _halt_generation=_halt_generation,
                                    rate_hz=rate_hz)
        from .trajectory import (PLAN_BUDGET_S, geometry_guard, plan_route,
                                 vet_route, vet_segment)
        from ..control.motion_profile import resolve_motion_rate

        self.harness.check_release_episode(target=q_target, duration=duration_s)
        halt_generation = self.harness._halt_generation if _halt_generation is None else _halt_generation
        self.harness._check_halt_generation(halt_generation)
        start = self.get_state().q  # Materialize LazyArm before inspecting its rate.
        self.harness._check_halt_generation(halt_generation)
        rate = resolve_motion_rate(self._arm, rate_hz)
        route = plan_route(self.harness, start, q_target, duration_s, rate_hz=rate)
        for index, goal in enumerate(route):
            def revalidate(start, actual_duration):
                deadline = time.monotonic() + PLAN_BUDGET_S
                # Never reuse an approval after feedback drift or an occupancy
                # refresh. Vet this stream and all remaining segments afresh.
                with geometry_guard(self.harness, deadline=deadline):
                    reason = vet_segment(self.harness, start, goal, actual_duration,
                                         deadline=deadline, stretch=False, rate_hz=rate)
                    if reason is None:
                        reason = vet_route(self.harness, goal, route[index + 1:],
                                           duration_s, deadline=deadline, rate_hz=rate)
                if reason:
                    raise SafetyViolation(f"planned route became unsafe: {reason}")

            if not self.move_joints(goal, duration_s=duration_s, _preflight=revalidate,
                                    _halt_generation=halt_generation, rate_hz=rate):
                return False
        return True

    def move_cartesian(self, T_goal: np.ndarray, duration_s: float = 2.0, *,
                       rate_hz: float = 50.0) -> bool:
        """Move the TCP along a straight line to `T_goal` (Seeed WRC e97998c).

        Plans from MEASURED feedback with `planning.cartesian` (every sample
        solved, continuous, no branch flips -- else SkillError), then executes
        through `move_joint_path`: the whole dense path is preflighted with
        the harness before the first command and every tick is approved.
        """
        self._refuse_path_with_planner()
        if self.harness.estopped:
            raise SafetyViolation("e-stop latched")
        kin = self.harness.kin
        if kin is None:
            from ..types import SkillError

            raise SkillError("cartesian motion needs the arm's kinematics")
        self.harness.check_model_withdrawal(command=True, planned=True)
        self.harness.check_release_episode(target=T_goal, duration=duration_s)
        from ..planning.cartesian import plan_cartesian_path

        start = np.asarray(self._arm.get_state().q, dtype=float)
        path = plan_cartesian_path(kin, start, T_goal)
        return self.move_joint_path(path, duration_s=duration_s, rate_hz=rate_hz)

    def move_joint_path(self, waypoints, duration_s: float = 2.0, *,
                        rate_hz: float = 50.0) -> bool:
        """Stream a dense joint path as one harness-gated motion.

        Same stretch rule as move_joints (min-jerk peak 1.875 * L / T kept
        under 0.9 * the velocity cap, L the path's Chebyshev arc length), a
        full `vet_step` preflight of the exact ticks from the bound start
        feedback (a doomed path never starts), then `approve()` per tick.
        """
        from ..control.arm_base import path_length, path_ticks

        self._refuse_path_with_planner()
        waypoints = [np.asarray(w, dtype=float).reshape(-1) for w in waypoints]
        if not waypoints:
            raise SafetyViolation("empty joint path; no motion sent")
        final = waypoints[-1]
        self.harness.check_model_withdrawal(command=True, target=final,
                                           duration=duration_s, joint_margin=None)
        self.harness.check_release_episode(target=final, duration=duration_s)
        start = np.asarray(self._arm.get_state().q, dtype=float)
        needed = 1.875 * path_length(start, waypoints) / (0.9 * self.harness.limits.max_joint_vel)
        duration_s = max(float(duration_s), needed)
        steps = max(2, int(duration_s * rate_hz))
        #: inspection hook (tests): the tick period of the last streamed path
        self.last_path_dt = duration_s / steps
        h = self.harness

        def preflight(actual_start, dur):
            ticks, dt = path_ticks(actual_start, waypoints, dur, rate_hz)
            prev = actual_start
            for q in ticks:
                reason = h.vet_step(prev, q, dt)
                if reason:
                    raise SafetyViolation(f"joint path is unsafe, no motion sent: {reason}")
                prev = q

        h.begin_motion()
        try:
            return self._arm.stream_path(waypoints, duration_s, rate_hz=rate_hz,
                                         approve=h.approve, preflight=preflight,
                                         before_stream=h.check_stream_start)
        finally:
            h.end_motion()

    def _refuse_path_with_planner(self) -> None:
        if self.motion_planner is not None:
            # A planner-bound arm owns its curves (and their evidence); the
            # wall-clock joint-path streamer must not route around it.
            raise SafetyViolation(
                "joint-path streaming would bypass this arm's motion planner; no motion sent")

    def set_gripper(self, pos: float, effort: float = 1.0, *, _halt_generation=None) -> None:
        self.harness.check_model_withdrawal(command=True, gripper=True, grip=pos, effort=effort)
        self.harness.check_release_episode(gripper=True, grip=pos)
        self.harness.check_contact_episode(gripper=True)
        self.harness._check_halt_generation(_halt_generation)
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
