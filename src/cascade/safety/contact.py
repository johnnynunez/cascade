"""Motor-feedback contact monitor: catch a blocked arm while a motion streams.

Rig incident 2026-10-11: a side-approach grasp drove the wrist and the wrist
camera into clutter beside the target. The arm PRESSED against it for ~8 s,
until the end-of-move settle check reported "did not settle at grasp pose".
Nothing compared the measured joints with the command during the motion.

This is that comparison -- a second safety layer for whatever the cameras
cannot see (occlusions, thin objects, the wrist camera's blind zone),
independent of the occupancy map. Per joint, against the commanded stream:

* tracking band: the measured position must lie inside the range the
  command swept over the last `lag_s` seconds (the PD loop trails a moving
  setpoint -- on the RS MIT loop by kd/kp = 60-100 ms, there is no velocity
  feedforward). The distance outside that band is the lag-compensated
  tracking error; it must stay below `track_tol_rad`.
* stall: over the last `stall_window_s`, a joint whose command advanced at
  least `stall_min_travel_rad` -- under EVERY lag in [0, lag_s], same
  direction -- must have advanced at least `stall_ratio` of that. A blocked
  joint stops while its command keeps moving, long before the band opens.

Either condition on `debounce_samples` consecutive samples of one joint
trips: ContactDetected, raised into the streaming loop, which stops sending
and holds the CURRENT MEASURED pose (cascade.control.arm_base). Feedback that
stops arriving fails safe: no good sample of a joint for
`max_feedback_gap_s` raises SafetyViolation (the arm then holds the last
commanded pose; there is no measurement to hold). A reading that cannot be a
position, or is farther from the last one than the arm can travel, is a read
failure, never a contact.

`mode: measure` runs everything, never stops a motion, and reports per-motion
maxima (`summary()`), so the thresholds can be calibrated from real moves.

Pure logic: times are stream seconds supplied by the caller (wall clock on
hardware, logical ticks on the mock), so it is testable on a simulated clock.
"""

from __future__ import annotations

import bisect
import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from ..types import ContactDetected, SafetyViolation

#: A joint reading farther than this speed * elapsed + tolerance from the
#: last good one cannot be a measurement (same bound as the RS drivers'
#: control.robstride.joint_reading_problem; the harness caps ~0.8 rad/s).
READ_MAX_JOINT_SPEED = 6.0     # rad/s
READ_JUMP_TOL_RAD = 0.15
#: lag fractions at which the stall rule evaluates the commanded advance
_STALL_LAG_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
_MODES = ("enforce", "measure")


def _number(block, key, default, *, lo=None, lo_open=False, hi=None, integer=False,
            allow_none=False):
    value = block.get(key, default)
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"contact_monitor.{key} must be a number, got {value!r}")  # noqa: TRY004 -- config error
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"contact_monitor.{key} must be finite, got {value!r}")
    if integer:
        if value != int(value):
            raise ValueError(f"contact_monitor.{key} must be an integer, got {value!r}")
        value = int(value)
    if lo is not None and (value <= lo if lo_open else value < lo):
        raise ValueError(f"contact_monitor.{key} must be {'>' if lo_open else '>='} {lo}, got {value!r}")
    if hi is not None and value > hi:
        raise ValueError(f"contact_monitor.{key} must be <= {hi}, got {value!r}")
    return value


@dataclass(frozen=True)
class ContactMonitorConfig:
    """`safety.contact_monitor` of an arm profile. Every threshold is config:
    tune on the rig from `mode: measure` reports (docs/CONTACT_STOP.md)."""

    enabled: bool = False
    #: enforce = stop on contact / lost feedback; measure = log only
    mode: str = "enforce"
    #: lag-compensated tracking tolerance, rad (one value or one per joint)
    track_tol_rad: tuple = (0.10,)
    #: how far the measured joint may trail the command, s
    lag_s: float = 0.25
    #: consecutive bad samples of one joint before tripping
    debounce_samples: int = 2
    stall_window_s: float = 0.12
    #: None disables the stall rule (band only)
    stall_min_travel_rad: float | None = 0.03
    stall_ratio: float = 0.2
    #: no good sample of a joint for this long mid-motion = stop
    max_feedback_gap_s: float = 0.35
    #: per single-joint feedback read (backends that read one motor at a time)
    read_timeout_s: float = 0.05
    #: joint reads per 50 Hz cycle even when the cycle has no slack left
    min_reads_per_cycle: int = 1
    #: after a contact, the TCP may not re-enter this radius around the TCP
    #: at the contact until reset_scene (0 = off)
    contact_zone_m: float = 0.05
    #: enforce mode: also log every motion's summary (measure always logs)
    log_motions: bool = False

    _KEYS = ("enabled", "mode", "track_tol_rad", "lag_s", "debounce_samples",
             "stall_window_s", "stall_min_travel_rad", "stall_ratio",
             "max_feedback_gap_s", "read_timeout_s", "min_reads_per_cycle",
             "contact_zone_m", "log_motions")

    @classmethod
    def from_config(cls, block) -> ContactMonitorConfig | None:
        """None when the profile has no `contact_monitor` block. A malformed
        block is a config error (ValueError): a safety layer that silently
        turns itself off, or a typo'd threshold that silently falls back to a
        default, is worse than none."""
        if block is None:
            return None
        if hasattr(block, "as_dict"):
            block = block.as_dict()
        if not isinstance(block, dict):
            raise ValueError(f"contact_monitor must be a mapping, got {block!r}")  # noqa: TRY004
        unknown = sorted(set(block) - set(cls._KEYS))
        if unknown:
            raise ValueError(f"contact_monitor: unknown key(s) {unknown}; known: {list(cls._KEYS)}")
        enabled = block.get("enabled", False)
        log_motions = block.get("log_motions", False)
        for key, value in (("enabled", enabled), ("log_motions", log_motions)):
            if not isinstance(value, bool):
                raise ValueError(f"contact_monitor.{key} must be true/false, got {value!r}")  # noqa: TRY004
        mode = block.get("mode", "enforce")
        if mode not in _MODES:
            raise ValueError(f"contact_monitor.mode must be one of {_MODES}, got {mode!r}")
        tol = block.get("track_tol_rad", 0.10)
        tol_list = list(tol) if isinstance(tol, (list, tuple)) else [tol]
        if not tol_list:
            raise ValueError("contact_monitor.track_tol_rad must not be empty")
        tols = tuple(_number({"track_tol_rad": v}, "track_tol_rad", None, lo=0.0, lo_open=True)
                     for v in tol_list)
        return cls(
            enabled=enabled,
            mode=mode,
            track_tol_rad=tols,
            lag_s=_number(block, "lag_s", 0.25, lo=0.0),
            debounce_samples=_number(block, "debounce_samples", 2, lo=1, integer=True),
            stall_window_s=_number(block, "stall_window_s", 0.12, lo=0.0, lo_open=True),
            stall_min_travel_rad=_number(block, "stall_min_travel_rad", 0.03, lo=0.0,
                                         lo_open=True, allow_none=True),
            stall_ratio=_number(block, "stall_ratio", 0.2, lo=0.0, hi=1.0),
            max_feedback_gap_s=_number(block, "max_feedback_gap_s", 0.35, lo=0.0, lo_open=True),
            read_timeout_s=_number(block, "read_timeout_s", 0.05, lo=0.0, lo_open=True),
            min_reads_per_cycle=_number(block, "min_reads_per_cycle", 1, lo=1, integer=True),
            contact_zone_m=_number(block, "contact_zone_m", 0.05, lo=0.0),
            log_motions=log_motions,
        )

    @property
    def enforcing(self) -> bool:
        return self.enabled and self.mode == "enforce"

    def tolerances(self, n: int) -> np.ndarray:
        tol = self.track_tol_rad
        if len(tol) == 1:
            return np.full(int(n), float(tol[0]))
        if len(tol) != int(n):
            raise ValueError(f"contact_monitor.track_tol_rad has {len(tol)} values; this arm "
                             f"has {int(n)} joints")
        return np.asarray(tol, dtype=float)


@dataclass
class _ReadStats:
    cycles: int = 0
    reads: int = 0
    cost_sum_s: float = 0.0
    cost_max_s: float = 0.0
    overruns: int = 0
    per_cycle: list = field(default_factory=list)


class ContactMonitor:
    """One motion's contact check. Feed it every streamed command
    (`commanded`) and every joint reading (`observe` / `observe_all`, or
    `read_failed`); it raises ContactDetected on a trip and SafetyViolation on
    lost feedback (enforce mode only). Not thread-safe: one stream owns it."""

    def __init__(self, config: ContactMonitorConfig, *, label: str = "motion"):
        self.cfg = config
        self.label = str(label)
        self.n: int | None = None
        self.started = False        # a command went out under this monitor
        self.commands = 0
        self.phase = "stream"
        self.t_last = 0.0
        self.tripped: dict | None = None
        self.would_trip: str | None = None
        self.would_trip_event: dict | None = None
        self.would_trips = 0
        self.feedback_lost: str | None = None
        self.hold: str | None = None      # set by the backend after a trip
        self.read_failures = 0
        self.last_read_failure: str | None = None
        self.reads = _ReadStats()
        self._rr = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def begin(self, q_start) -> None:
        """Bind to the stream's measured start pose (stream time 0)."""
        q = np.asarray(q_start, dtype=float).reshape(-1)
        if not q.size or not np.all(np.isfinite(q)):
            raise SafetyViolation("contact monitor needs a finite measured start pose; no motion sent")
        try:
            self.tol = self.cfg.tolerances(q.size)
        except ValueError as exc:
            raise SafetyViolation(f"{exc}; no motion sent") from exc
        n = self.n = int(q.size)
        self._ct = [0.0]
        self._cq = [q.copy()]
        self._samples = [[(0.0, float(q[j]))] for j in range(n)]
        self.latest = q.copy()
        self.latest_t = np.zeros(n)
        self._good_t = np.zeros(n)
        self._streak = np.zeros(n, dtype=int)
        self._streak_t0 = np.zeros(n)
        self.n_samples = np.zeros(n, dtype=int)
        self.max_err = np.zeros(n)
        self.max_raw = np.zeros(n)
        self.max_lag = np.zeros(n)
        #: lowest advance / commanded-advance ratio the stall rule evaluated
        self.min_ratio = np.full(n, np.inf)
        self.max_gap = 0.0

    def commanded(self, t: float, q) -> None:
        if self.n is None:
            raise SafetyViolation("contact monitor used before begin(); no motion sent")
        self.started = True
        self.commands += 1
        t = float(t)
        self.t_last = max(self.t_last, t)
        self._ct.append(max(t, self._ct[-1]))
        self._cq.append(np.asarray(q, dtype=float).reshape(-1).copy())
        horizon = t - (self.cfg.lag_s + self.cfg.stall_window_s + self.cfg.max_feedback_gap_s + 0.2)
        drop = 0
        while drop + 1 < len(self._ct) and self._ct[drop + 1] <= horizon:
            drop += 1
        if drop:
            del self._ct[:drop]
            del self._cq[:drop]

    def next_joint(self) -> int:
        """Round-robin joint for a backend that reads one motor at a time."""
        j = self._rr % self.n
        self._rr += 1
        return j

    # ── feedback ─────────────────────────────────────────────────────────

    def observe_all(self, t: float, q) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n:
            for j in range(self.n):
                self.read_failed(t, j, f"feedback has {q.size} joints, expected {self.n}")
            return
        for j in range(self.n):
            self.observe(t, j, q[j])

    def observe(self, t: float, j: int, x) -> None:
        t = float(t)
        self.t_last = max(self.t_last, t)
        try:
            x = float(x)
        except (TypeError, ValueError):
            self.read_failed(t, j, f"reading {x!r} is not a number")
            return
        allowed = READ_MAX_JOINT_SPEED * max(0.0, t - self.latest_t[j]) + READ_JUMP_TOL_RAD
        if not math.isfinite(x) or abs(x - self.latest[j]) > allowed:
            self.read_failed(t, j, f"implausible reading {x!r} (last {self.latest[j]:+.4f} rad)")
            return
        self.max_gap = max(self.max_gap, t - self._good_t[j])
        self._good_t[j] = max(self._good_t[j], t)
        self.latest[j], self.latest_t[j] = x, t
        err, raw, lag, cmd = self._band(t, j, x)
        stall = self._stall(t, j, x)
        samples = self._samples[j]
        samples.append((t, x))
        keep_from = t - (self.cfg.stall_window_s + self.cfg.max_feedback_gap_s + 0.2)
        while len(samples) > 2 and samples[1][0] <= keep_from:
            samples.pop(0)
        self.n_samples[j] += 1
        self.max_err[j] = max(self.max_err[j], err)
        self.max_raw[j] = max(self.max_raw[j], raw)
        if lag is not None:
            self.max_lag[j] = max(self.max_lag[j], lag)
        if err <= self.tol[j] and stall is None:
            self._streak[j] = 0
            return
        if self._streak[j] == 0:
            self._streak_t0[j] = t
        self._streak[j] += 1
        if self._streak[j] >= self.cfg.debounce_samples:
            self._trip(t, j, x, err, raw, cmd, stall)

    def read_failed(self, t: float, j: int, why) -> None:
        t = float(t)
        self.t_last = max(self.t_last, t)
        self.read_failures += 1
        self.last_read_failure = f"joint {j + 1}: {why}"
        self._check_gap(t, j, why)

    def check_feedback(self, t: float) -> None:
        """Every joint must have a good sample within max_feedback_gap_s."""
        t = float(t)
        self.t_last = max(self.t_last, t)
        for j in range(self.n):
            self._check_gap(t, j, None)

    def note_cycle(self, reads: int, costs, overrun: bool = False) -> None:
        """Backend bookkeeping for the summary (reads per cycle, read cost)."""
        st = self.reads
        st.cycles += 1
        st.reads += int(reads)
        for c in costs:
            st.cost_sum_s += float(c)
            st.cost_max_s = max(st.cost_max_s, float(c))
        st.overruns += bool(overrun)

    # ── rules ────────────────────────────────────────────────────────────

    def _cmd_index(self, t: float) -> int:
        return max(bisect.bisect_right(self._ct, t) - 1, 0)

    def _band(self, t, j, x):
        """(lag-compensated error, raw error, effective lag, latest command)."""
        i1 = self._cmd_index(t)
        i0 = self._cmd_index(t - self.cfg.lag_s)
        vals = [self._cq[k][j] for k in range(i0, i1 + 1)]
        lo, hi = min(vals), max(vals)
        err = max(lo - x, x - hi, 0.0)
        latest = float(self._cq[i1][j])
        raw = abs(x - latest)
        # effective lag: the shortest look-back whose commands bracket x
        mn = mx = latest
        lag = 0.0 if mn - 1e-9 <= x <= mx + 1e-9 else None
        k = i1
        while lag is None and k > 0:
            v = float(self._cq[k - 1][j])
            mn, mx = min(mn, v), max(mx, v)
            if mn - 1e-9 <= x <= mx + 1e-9:
                lag = t - self._ct[k]
            k -= 1
        if lag is None:
            # x is outside everything commanded in the retained history: an
            # offset (droop, a block), not a lag -- the track error has it
            return float(err), float(raw), None, latest
        return float(err), float(raw), float(max(lag, 0.0)), latest

    def _cmd_at(self, t: float, j: int) -> float:
        return float(self._cq[self._cmd_index(t)][j])

    def _stall(self, t, j, x):
        travel = self.cfg.stall_min_travel_rad
        if travel is None:
            return None
        window = self.cfg.stall_window_s
        ref = None
        for ts, xs in reversed(self._samples[j]):
            if ts <= t - window:
                ref = (ts, xs)
                break
        if ref is None:
            return None
        t_a, x_a = ref
        cs = [self._cmd_at(t - f * self.cfg.lag_s, j) - self._cmd_at(t_a - f * self.cfg.lag_s, j)
              for f in _STALL_LAG_FRACTIONS]
        if min(abs(c) for c in cs) < travel:
            return None
        sign = 1.0 if cs[0] > 0 else -1.0
        if any((c > 0) != (sign > 0) for c in cs):
            return None
        c_min = min(abs(c) for c in cs)
        moved = x - x_a
        self.min_ratio[j] = min(self.min_ratio[j], moved * sign / c_min)
        if moved * sign > self.cfg.stall_ratio * c_min:
            return None
        return {"moved_rad": round(moved, 4), "commanded_rad": round(sign * c_min, 4),
                "window_s": round(t - t_a, 3)}

    def _trip(self, t, j, x, err, raw, cmd, stall):
        dur_ms = (t - self._streak_t0[j]) * 1000.0
        msg = (f"contact detected at joint {j + 1}: tracking error {err:.3f} rad for "
               f"{dur_ms:.0f} ms -- something is blocking the arm")
        if stall is not None:
            msg += (f" (stalled: moved {stall['moved_rad']:+.3f} rad while commanded "
                    f"{stall['commanded_rad']:+.3f} rad in {stall['window_s']:.2f} s)")
        event = {"joint": j + 1, "error_rad": round(err, 4), "raw_error_rad": round(raw, 4),
                 "tol_rad": float(self.tol[j]), "duration_ms": round(dur_ms, 1),
                 "measured_rad": round(x, 4), "commanded_rad": round(cmd, 4),
                 "stall": stall, "phase": self.phase, "t_s": round(t, 3), "motion": self.label,
                 "measured_q": [round(float(v), 4) for v in self.latest]}
        if self.cfg.enforcing:
            self.tripped = event
            raise ContactDetected(msg, event)
        self.would_trips += 1
        if self.would_trip is None:
            self.would_trip, self.would_trip_event = msg, event
        self._streak[j] = 0

    def _check_gap(self, t, j, why):
        gap = t - self._good_t[j]
        self.max_gap = max(self.max_gap, gap)
        if gap <= self.cfg.max_feedback_gap_s:
            return
        msg = (f"contact monitor lost joint {j + 1} feedback for {gap * 1000:.0f} ms mid-motion"
               + (f" ({why})" if why else "")
               + " -- stopping; the arm holds the last commanded pose")
        if self.feedback_lost is None:
            self.feedback_lost = msg
        if self.cfg.enforcing:
            raise SafetyViolation(msg)

    # ── report ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        def r(a, nd=4):
            return [round(float(v), nd) for v in a]
        st = self.reads
        out = {"motion": self.label, "mode": self.cfg.mode, "phase": self.phase,
               "commands": self.commands, "duration_s": round(self.t_last, 3),
               "tripped": None if self.tripped is None else self._msg(self.tripped),
               "would_trip": self.would_trip, "would_trips": self.would_trips,
               "feedback_lost": self.feedback_lost, "hold": self.hold,
               "read_failures": self.read_failures, "last_read_failure": self.last_read_failure,
               "lag_s": self.cfg.lag_s}
        if self.n is None:
            return out
        out.update({
            "samples": [int(v) for v in self.n_samples],
            "max_track_err_rad": r(self.max_err), "track_tol_rad": r(self.tol),
            "max_raw_err_rad": r(self.max_raw), "max_effective_lag_s": r(self.max_lag, 3),
            "min_advance_ratio": [None if not math.isfinite(v) else round(float(v), 3)
                                  for v in self.min_ratio],
            "stall_ratio": self.cfg.stall_ratio,
            "max_feedback_gap_s": round(self.max_gap, 3),
            "reads": {"cycles": st.cycles, "reads": st.reads,
                      "per_cycle": round(st.reads / st.cycles, 2) if st.cycles else None,
                      "cost_ms_mean": round(1000 * st.cost_sum_s / st.reads, 2) if st.reads else None,
                      "cost_ms_max": round(1000 * st.cost_max_s, 2), "overruns": st.overruns},
        })
        return out

    @staticmethod
    def _msg(event) -> str:
        return (f"joint {event['joint']} tracking error {event['error_rad']:.3f} rad "
                f"({event['phase']})")


def format_summary(s: dict) -> str:
    """One stderr line per motion, for calibrating from real moves."""
    def vec(key):
        v = s.get(key)
        return "[" + " ".join(f"{x:.3f}" for x in v) + "]" if v else "[]"
    head = (f"[contact] {s.get('mode')} {s.get('motion')} {s.get('duration_s', 0):.2f}s "
            f"{s.get('phase')}: track err max {vec('max_track_err_rad')} rad "
            f"(tol {vec('track_tol_rad')}); raw {vec('max_raw_err_rad')}; "
            f"lag {vec('max_effective_lag_s')} s (lag_s {s.get('lag_s')}); "
            f"advance ratio min {s.get('min_advance_ratio')} (stall_ratio {s.get('stall_ratio')}); "
            f"samples {s.get('samples')}; gap max {s.get('max_feedback_gap_s')} s")
    reads = s.get("reads") or {}
    if reads.get("cycles"):
        head += (f"; reads/cycle {reads.get('per_cycle')} at {reads.get('cost_ms_mean')} ms "
                 f"(max {reads.get('cost_ms_max')})")
    for key in ("tripped", "would_trip", "feedback_lost"):
        if s.get(key):
            head += f"; {key.upper()}: {s[key]}"
    if s.get("read_failures"):
        head += f"; read failures {s['read_failures']} ({s.get('last_read_failure')})"
    if s.get("hold"):
        head += f"; hold: {s['hold']}"
    return head
