"""reBot RS gripper: a turn reference at power-on, and opening that never pushes.

Rig incident, 2026-10-12 00:21. The arm was powered on with the jaws fully
open. The first `grasp_object` opened the gripper "to 6.2 rad" and drove the
jaws far past their open end, breaking the plastic stops.

Why. A RobStride motor knows its position only within one turn after power-on.
mechPos starts in [0, 2*pi) (or [-pi, pi) with zero_sta = 1), counted from its
stored zero. The reBot jaw travels 0 (closed) -> +6.39 rad (open hard end),
which is MORE than one turn (2*pi = 6.283). Powered on fully open (6.3-6.39
rad), the motor read ~0.05: "closed". Every target is in the motor's
coordinates, so `open_pos` 6.2 meant 6.2 rad further open than the jaw really
was. The travel clamp checked the target against the profile travel, but in
the wrong frame, so it did not help.

What this module adds, for both RS transports (`RebotRSArm`,
`RebotRSMotorBridgeArm`):

* **Reference before the first gripper command.** After every connect, and
  after the bus loses the arm (feedback lost = the arm may have been powered
  off), the gripper is unreferenced. Before the first gripper command, the jaws
  close slowly at the normal grip stiffness until they stall. Closing is the
  safe direction: the closed stop is the jaws meeting, which every close
  already presses at full grip torque. The stall must be the closed stop, so
  the reading there is `closed_pos` plus a whole number of turns. That number
  becomes the offset between the motor's frame and the jaw's. A stall away
  from the closed stop (more than `REF_CLOSED_TOL` short: something between
  the jaws) leaves the gripper unreferenced and raises `GripperFault`. The arm
  itself can still move.
* **Opening never pushes.** Opening needs no force. Every opening command
  ramps at `OPEN_SPEED` while the jaw position is watched. If the jaw stops
  advancing short of the target, it is blocked: it holds where it is and
  `GripperFault` is raised. It does not keep pressing until something breaks.
* **Readings stay honest.** An unreferenced gripper reads as unknown (None),
  never as a guessed position. The reference is dropped when the motor may
  have been power-cycled: a referenced reading outside the jaw's travel,
  `MAX_READ_FAILURES` failed gripper reads in a row, or (in the drivers) the
  arm's own feedback lost or a reconnect. A power cycle shorter than that,
  whose new reading happens to land inside the travel, is not detectable
  from mechPos alone.
* No target ever leaves the MIT position range (+-4*pi).

All timing goes through this module's `time`, so tests run on a fake clock.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable

from ..types import GripperFault
from .robstride import MIT_P_MAX, clamp_to_travel

TWO_PI = 2.0 * math.pi

#: closing speed while referencing (rad/s). Full travel takes ~3.3 s.
REF_SPEED = 2.0
#: command period while referencing or opening (s)
STEP_S = 0.05
#: a stall is judged over this look-back (s) ...
STALL_WINDOW_S = 0.3
#: ... and needs at least this much commanded travel before it counts (rad)
REF_GRACE_RAD = 0.2
#: referencing: moved less than this toward closed over the window = stalled
REF_STALL_RAD = 0.03
#: the stall must be within this of `closed_pos` (mod one turn) to be the
#: closed stop. Farther = the jaws stopped on something.
REF_CLOSED_TOL = 0.35
#: referencing ramps at most the travel plus this before giving up (rad)
REF_EXTRA_RAD = 0.6
#: opening speed (rad/s). Full open takes ~3.1 s. A blocked jaw is caught
#: within ~STALL_WINDOW_S, pressing at about kp * (lag + 0.6 rad).
OPEN_SPEED = 2.0
#: opening stalls when it advanced less than this fraction of the commanded
#: advance over the window, once the ramp is past the grace period
OPEN_STALL_FRAC = 0.25
OPEN_GRACE_S = 0.4
#: after the ramp ends, this long to settle (s) ...
OPEN_SETTLE_S = 1.5
#: ... and a jaw that stops more than this short of its target is blocked
OPEN_SHORT_RAD = 0.4
#: commands that open by less than this go out directly (holding, nudges)
OPEN_RAMP_MIN_RAD = 0.15
#: a referenced reading this far outside the travel means the turn count is
#: gone (motor power-cycled): the reference is dropped
TRAVEL_TOL_RAD = 0.5
#: consecutive failed reads that abort a reference or an opening
MAX_READ_FAILURES = 3
#: keep raw targets this far inside the MIT position range
MIT_MARGIN_RAD = 0.05


class RSGripperGuard:
    """Turn reference and guarded opening for one RS gripper motor.

    `read_raw()` returns the motor's mechPos (raises on a failed or
    implausible read). `send_raw(pos, kp)` sends one MIT position command in
    the motor's own frame. `stopped()` is the driver's soft-stop flag. The
    guard works in the JAW frame (`closed_pos` .. `open_pos` of the profile)
    and converts at the boundary: raw = jaw + offset.
    """

    def __init__(self, *, open_pos: float, closed_pos: float, ref_kp: float,
                 read_raw: Callable[[], float], send_raw: Callable[[float, float], None],
                 stopped: Callable[[], bool], log: Callable[[str], None] = print):
        self.open_pos = float(open_pos)
        #: stiffness of the referencing close (the profile's grip kp): a
        #: light-hold kp could not move the jaw at all
        self.ref_kp = float(ref_kp)
        self.closed_pos = float(closed_pos)
        self._read_raw = read_raw
        self._send_raw = send_raw
        self._stopped = stopped
        self._log = log
        #: raw - jaw, a whole number of turns; None = unreferenced
        self.offset: float | None = None
        self._read_failures = 0
        #: the last jaw target sent (jaw frame): decides whether the next
        #: command opens, without an extra bus read for every close or hold
        self._last_target: float | None = None
        self.unreferenced_why = "not referenced since connect"
        self.lo = min(self.open_pos, self.closed_pos)
        self.hi = max(self.open_pos, self.closed_pos)
        #: +1 if opening increases the angle
        self.open_dir = 1.0 if self.open_pos > self.closed_pos else -1.0

    # ── state ────────────────────────────────────────────────────────────

    @property
    def referenced(self) -> bool:
        return self.offset is not None

    def invalidate(self, why: str) -> None:
        if self.offset is not None:
            self._log(f"[rebot] gripper reference dropped: {why}")
        self.offset = None
        self._last_target = None
        self.unreferenced_why = why

    def position(self) -> float | None:
        """The jaw angle, or None when unknown (unreferenced or a failed
        read). Never a guess."""
        if self.offset is None:
            return None
        try:
            raw = float(self._read_raw())
        except Exception as e:  # noqa: BLE001 -- a failed read is "unknown"
            self._read_failures += 1
            if self._read_failures >= MAX_READ_FAILURES:
                # The motor stopped answering: it may have been powered off,
                # and its turn count with it.
                self.invalidate(f"gripper feedback lost ({self._read_failures} failed reads: {e})")
            return None
        self._read_failures = 0
        jaw = raw - self.offset
        if not (self.lo - TRAVEL_TOL_RAD <= jaw <= self.hi + TRAVEL_TOL_RAD):
            self.invalidate(f"reading {jaw:.2f} rad is outside the jaw travel "
                            f"[{self.lo:.2f}, {self.hi:.2f}] (motor power-cycled?)")
            return None
        return jaw

    # ── commands ─────────────────────────────────────────────────────────

    def hold_raw(self, raw: float, kp: float) -> None:
        """Hold the motor where it is, in its own frame (connect, before any
        reference exists). Never moves the jaw."""
        self._send(float(raw), kp)

    def command(self, jaw_target: float, kp: float) -> None:
        """Move the jaw to `jaw_target` (jaw frame, clamped to the travel).
        References first if needed. Opening moves are ramped and watched."""
        if self._stopped():
            return
        if self.offset is None:
            self.reference()
        target = clamp_to_travel(jaw_target, self.open_pos, self.closed_pos)
        # Opening = more open than where the jaw IS. The last target only
        # pre-screens (a close or a hold needs no extra bus read); a target
        # more open than the last one is checked against a fresh reading,
        # because a jaw stalled on an object (squeeze cap re-command) is far
        # more open than its last target.
        last = self._last_target
        if last is None or (target - last) * self.open_dir > OPEN_RAMP_MIN_RAD:
            now = self._position_or_fault()
            if (target - now) * self.open_dir > OPEN_RAMP_MIN_RAD:
                self._open_ramped(now, target, kp)
                return
        self._send(target + self.offset, kp)

    def _position_or_fault(self) -> float:
        now = self.position()
        if now is None:
            raise GripperFault(f"gripper position unknown ({self.unreferenced_why or 'read failed'}); "
                               "no gripper command sent")
        return now

    # ── referencing ──────────────────────────────────────────────────────

    def reference(self) -> float:
        """Close slowly to the closed stop and fix the turn offset. Returns
        the offset. Raises GripperFault (gripper left unreferenced, holding
        where it stopped) if the jaws stop anywhere else."""
        close_dir = -self.open_dir
        kp = self.ref_kp
        raw0 = self._read_with_retries("referencing")
        max_ramp = (self.hi - self.lo) + REF_EXTRA_RAD
        self._log(f"[rebot] referencing the gripper: closing slowly from raw {raw0:.3f} rad "
                  "to the closed stop")
        t0 = time.monotonic()
        hist: deque = deque()
        raw = raw0
        failures = 0
        while True:
            if self._stopped():
                raise GripperFault("gripper referencing interrupted by a stop")
            ramp = min(REF_SPEED * (time.monotonic() - t0), max_ramp)
            self._send(raw0 + close_dir * ramp, kp)
            time.sleep(STEP_S)
            try:
                raw = float(self._read_raw())
                failures = 0
            except Exception as e:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    self._hold_last(raw, kp)
                    raise GripperFault(f"gripper feedback lost while referencing: {e}") from e
                continue
            t = time.monotonic()
            hist.append((t, raw))
            while len(hist) > 1 and t - hist[1][0] >= STALL_WINDOW_S:
                hist.popleft()
            span_t = t - hist[0][0]
            moved = (raw - hist[0][1]) * close_dir
            if ramp >= REF_GRACE_RAD and span_t >= STALL_WINDOW_S and moved < REF_STALL_RAD:
                break
            if ramp >= max_ramp and t - t0 > max_ramp / REF_SPEED + 2 * STALL_WINDOW_S:
                self._hold_last(raw, kp)
                raise GripperFault("gripper referencing found no stop after a full travel; "
                                   "is the gripper connected?")
        self._send(raw, kp)                      # stop pressing: hold where it stalled
        turns = round((raw - self.closed_pos) / TWO_PI)
        residual = raw - self.closed_pos - turns * TWO_PI
        if abs(residual) > REF_CLOSED_TOL:
            self.invalidate("referencing stopped short of the closed stop")
            raise GripperFault(
                f"gripper reference failed: the jaws stopped at motor raw {raw:+.3f} rad, "
                f"{abs(residual):.2f} rad from where the profile's closed_pos lies in any "
                "motor turn -- something is between the jaws, or the motor's zero no longer "
                "matches the closed stop (a slipped or damaged jaw drive: re-zero the motor "
                "at the closed stop). The gripper stays unusable (the arm can still move) "
                "until a reference succeeds.")
        self.offset = turns * TWO_PI
        self._last_target = raw - self.offset
        self.unreferenced_why = ""
        self._log(f"[rebot] gripper referenced: closed stop at raw {raw:.3f} rad "
                  f"({turns:+d} turn(s) from the motor's power-on frame)")
        return self.offset

    # ── guarded opening ──────────────────────────────────────────────────

    def _open_ramped(self, start: float, target: float, kp: float) -> None:
        distance = abs(target - start)
        d = 1.0 if target > start else -1.0
        assert d == self.open_dir, "the guarded ramp only opens"
        t0 = time.monotonic()
        hist: deque = deque()
        jaw = start
        failures = 0
        ramp_done_t = None
        while True:
            if self._stopped():
                return
            t = time.monotonic()
            ramp = min(OPEN_SPEED * (t - t0), distance)
            self._send(start + d * ramp + self.offset, kp)
            time.sleep(STEP_S)
            x = self.position()
            if x is None:
                failures += 1
                if failures >= MAX_READ_FAILURES or self.offset is None:
                    self._hold_last_jaw(jaw, kp)
                    raise GripperFault(f"gripper feedback lost while opening "
                                       f"({self.unreferenced_why or 'read failed'})")
                continue
            failures = 0
            jaw = x
            t = time.monotonic()
            hist.append((t, x, start + d * ramp))
            while len(hist) > 1 and t - hist[1][0] >= STALL_WINDOW_S:
                hist.popleft()
            t_old, x_old, c_old = hist[0]
            moved = (x - x_old) * d
            commanded = (start + d * ramp - c_old) * d
            if ramp < distance:
                if (t - t0 >= OPEN_GRACE_S and t - t_old >= STALL_WINDOW_S
                        and commanded > 0.1 and moved < OPEN_STALL_FRAC * commanded):
                    self._blocked(x, target, kp)
                continue
            if ramp_done_t is None:
                ramp_done_t = t
            if abs(target - x) <= OPEN_RAMP_MIN_RAD:
                return
            settled = t - t_old >= STALL_WINDOW_S and abs(moved) < REF_STALL_RAD
            if settled or t - ramp_done_t > OPEN_SETTLE_S:
                if (target - x) * d > OPEN_SHORT_RAD:
                    self._blocked(x, target, kp)
                return

    def _blocked(self, jaw: float, target: float, kp: float) -> None:
        self._send(jaw + self.offset, kp)        # hold where it stopped; never press on
        raise GripperFault(
            f"gripper blocked while opening at {jaw:.2f} rad (target {target:.2f}): "
            "holding there instead of pushing. Check what is against the jaws.")

    # ── helpers ──────────────────────────────────────────────────────────

    def _read_with_retries(self, what: str) -> float:
        err = None
        for _ in range(MAX_READ_FAILURES):
            try:
                return float(self._read_raw())
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(STEP_S)
        raise GripperFault(f"no gripper feedback for {what}: {err}")

    def _hold_last(self, raw: float, kp: float) -> None:
        try:
            self._send(raw, kp)
        except Exception as e:  # noqa: BLE001 -- the fault being raised matters more
            self._log(f"[rebot] gripper hold failed: {e}")

    def _hold_last_jaw(self, jaw: float, kp: float) -> None:
        if self.offset is not None:
            self._hold_last(jaw + self.offset, kp)

    def _send(self, raw: float, kp: float) -> None:
        if not math.isfinite(raw) or abs(raw) > MIT_P_MAX - MIT_MARGIN_RAD:
            raise GripperFault(f"gripper target {raw:.3f} rad is outside the motor's MIT "
                               f"range (+-{MIT_P_MAX:.2f}); not sent")
        self._send_raw(float(raw), float(kp))
        if self.offset is not None:
            self._last_target = float(raw) - self.offset
