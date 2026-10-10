"""Extrinsic drift monitor: notices a knocked eye-to-hand camera at runtime.

A fixed camera that gets bumped keeps delivering perfectly good images, and
every belief and occupancy voxel fused through its OLD extrinsic then lands
centimetres off -- the failure nothing else in the stack can see, because
each frame is self-consistent. The arm is a calibration target that is
always in view, so the monitor uses it:

* Only when the arm is STATIC: no harness motion in progress, the
  WorldWatcher not paused for a motion skill, the measured joints unchanged
  for ``settle_s``, the depth frame captured after that, and the joints
  unchanged again after the grab. A moving arm is never checked.
* Every ``period_s`` it takes the newest SENSOR-depth frame and runs a short
  single-view ICP of the arm's surface at FK through the current extrinsic
  (``calibration.markerless.measure_offset``). The statistic is the RMS
  displacement of the visible arm surface under the correction found (plus
  its rotation angle): 2.0 mm median (5.7 mm worst of 111) under D455-like
  synthetic noise, 24 mm median for a 2 deg knock.
  A view that cannot decide (arm occluded / out of view / too few points /
  depth that does not explain the arm) is INCONCLUSIVE: it neither counts
  towards drift nor resets the count.
* ``consecutive`` drift checks in a row mark that ONE camera UNCALIBRATED at
  runtime: ``Extrinsics.invalidate`` (the same semantics as a rejected
  record: ``cam_to_base()`` refuses with the reason) and fusion + depth
  mapping off for that stream through the WorldWatcher's lock, so belief
  fusion and the occupancy map stop using it. Surfaced as a stderr line, a
  row in ``<run>/extrinsics_drift.jsonl``, a memory note (the dashboard's
  activity feed) and ``status()`` (dashboard / world_state).

Passive re-calibration: the static (q, depth) views the checks already take
are kept (distinct joint configurations only, bounded). While the camera is
flagged, once they pass the markerless gate -- pose count, diversity and
degeneracy included -- they are solved into a CANDIDATE record in the run
dir. Adopting it is opt-in (``auto_apply: false`` by default): off, the
monitor prints how to adopt it; on, it is applied only if the gate passes
AND it explains the current depth better than the active extrinsic, and then
fusion is re-enabled.

THE MONITOR NEVER COMMANDS THE ARM. Its only arm interface is a joint
reader (``q_fn``), which returns None for an arm in standby so a LazyArm is
never materialised by a perception thread.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..calibration.markerless import (
    MIN_INLIER_FRACTION,
    MIN_POSES,
    MIN_ROTATION_SPREAD_DEG,
    MIN_TCP_SPREAD_M,
    DepthSample,
    measure_offset,
)

CONFIG_KEYS = ("enabled", "period_s", "max_offset_m", "max_rot_deg", "consecutive", "auto_apply")


@dataclass(frozen=True)
class DriftMonitorConfig:
    """Per camera, ``extrinsics.drift_monitor:`` in the camera profile.

    Defaults are conservative and OFF. Measured on the synthetic D455
    (single view, 37 reBot presets x 3 noise seeds = 111 checks, docs/
    HANDEYE_CALIBRATION.md): noise alone reads 2.0 mm median / 4.7 mm p99 /
    5.7 mm max of surface displacement and 0.52 / 1.53 / 1.65 deg of
    rotation. ``max_offset_m`` 10 mm sits 1.75x above that worst reading; a
    1 deg knock reads 11.7 mm median (95 % of checks over), 2 deg 24 mm
    (100 %), 10 mm of translation 10.6 mm (84 %), 20 mm 20 mm (100 %);
    0.5 deg / 5 mm are below a single view's noise floor and NOT detected.
    The single-view rotation estimate is the noisier statistic (the arm
    constrains rotation about itself weakly), so ``max_rot_deg`` 3 deg is
    only a backstop -- the surface displacement is what flags. Three
    CONSECUTIVE drift checks before acting; an ok check resets the count,
    an inconclusive one does not.
    """

    enabled: bool = False
    period_s: float = 5.0
    max_offset_m: float = 0.010
    max_rot_deg: float = 3.0
    consecutive: int = 3
    auto_apply: bool = False

    @classmethod
    def from_config(cls, raw) -> DriftMonitorConfig:
        if raw is None:
            return cls()
        if hasattr(raw, "as_dict"):
            raw = raw.as_dict()
        elif hasattr(raw, "_data"):
            raw = dict(raw._data)
        if not isinstance(raw, dict):
            # ValueError like every other config refusal (callers catch one type)
            raise ValueError(f"extrinsics.drift_monitor must be a mapping, got {raw!r}")  # noqa: TRY004
        unknown = sorted(set(raw) - set(CONFIG_KEYS))
        if unknown:
            raise ValueError(f"extrinsics.drift_monitor: unknown key(s) {unknown}; "
                             f"allowed: {list(CONFIG_KEYS)}")
        out = {}
        for key in ("enabled", "auto_apply"):
            if key in raw:
                if not isinstance(raw[key], bool):
                    raise ValueError(f"extrinsics.drift_monitor.{key} must be true/false, "
                                     f"got {raw[key]!r}")
                out[key] = raw[key]
        for key in ("period_s", "max_offset_m", "max_rot_deg"):
            if key in raw:
                v = raw[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"extrinsics.drift_monitor.{key} must be a number, got {v!r}")
                if not math.isfinite(float(v)) or float(v) <= 0:
                    raise ValueError(f"extrinsics.drift_monitor.{key} must be finite and > 0, "
                                     f"got {v!r}")
                out[key] = float(v)
        if "consecutive" in raw:
            v = raw["consecutive"]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"extrinsics.drift_monitor.consecutive must be an integer >= 1, "
                                 f"got {v!r}")
            out["consecutive"] = int(v)
        return cls(**out)


@dataclass(frozen=True)
class CheckResult:
    outcome: str            # skipped | inconclusive | ok | drift
    reason: str
    t: float
    offset_m: float = math.nan
    offset_deg: float = math.nan
    n_inliers: int = 0

    def as_dict(self) -> dict:
        return {"outcome": self.outcome, "reason": self.reason,
                "offset_mm": None if not math.isfinite(self.offset_m) else round(1000 * self.offset_m, 2),
                "offset_deg": None if not math.isfinite(self.offset_deg) else round(self.offset_deg, 3),
                "n_inliers": int(self.n_inliers)}


#: Joints differing by more than this (rad, any joint) between polls = moving.
Q_STATIC_TOL_RAD = 2e-3
#: Snapshots kept for passive re-calibration must differ by this much.
DISTINCT_Q_RAD = 0.05
MAX_SNAPSHOTS = 24
#: A candidate replaces the active extrinsic only if it explains this much
#: more of the visible arm in the current depth.
APPLY_MARGIN = 0.05


def _stderr(line: str) -> None:
    print(line, file=sys.stderr)


class ExtrinsicDriftMonitor:
    """One eye-to-hand camera's drift monitor (see the module docstring).

    ``surface_fn()`` builds the arm's RobotSurface (lazily, on the monitor
    thread); ``q_fn()`` returns the measured joints or None (standby);
    ``motion_fn()`` is True while a motion is in progress. ``check_once()``
    is the whole check, synchronous (tests drive it with a fake clock);
    ``start()`` runs it on a daemon thread.
    """

    def __init__(self, name: str, watched, *, surface_fn, q_fn, motion_fn=lambda: False,
                 config: DriftMonitorConfig, watcher=None, run_dir=None, memory=None,
                 log=_stderr, clock=time.monotonic, settle_s: float = 1.0,
                 camera_serial: str = "", arm: str = "", ee_frame: str = ""):
        ext = watched.extrinsics
        if ext.mode != "eye_to_hand":
            raise ValueError(f"camera {name!r}: the drift monitor is eye_to_hand only (a wrist "
                             "camera does not see the arm)")
        if not ext.calibrated:
            raise ValueError(f"camera {name!r} is not calibrated; nothing to monitor")
        self.name = name
        self.watched = watched
        self.config = config
        self._watcher = watcher
        self._surface_fn = surface_fn
        self._surface = None
        self._q_fn = q_fn
        self._motion_fn = motion_fn
        self._memory = memory
        self._log = log
        self._clock = clock
        self.settle_s = float(settle_s)
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.camera_serial, self.arm, self.ee_frame = str(camera_serial or ""), arm, ee_frame
        self._T_ref = np.array(ext.cam_to_base(), dtype=float)
        self._orig_fuse, self._orig_map = watched.fuse, watched.map_depth
        self._lock = threading.Lock()
        self._last_q = None
        self._static_since = None
        self._consecutive = 0
        self._run_started_t = None
        self._flagged = False
        self._flag_reason = None
        self._last_estimate = None
        self.snapshots: list[DepthSample] = []
        self._snap_t: list[float] = []
        self._attempted_at = 0
        self.candidate_path = None
        self.last_rejection = None
        self.checks = 0
        self.last = None
        self._thread = None
        self._stop = threading.Event()

    # ── reporting ────────────────────────────────────────────────────────

    def _event(self, event: str, **fields) -> None:
        if self.run_dir is None:
            return
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            row = {"event": event, "camera": self.name, "ts": time.time(), **fields}
            with (self.run_dir / "extrinsics_drift.jsonl").open("a") as f:
                f.write(json.dumps(row, default=float) + "\n")
        except OSError as e:
            self._log(f"[drift:{self.name}] could not write event: {e}")

    def _note(self, text: str, **data) -> None:
        if self._memory is not None:
            try:
                self._memory.add("note", text, data={"camera": self.name, **data})
            except Exception:  # noqa: BLE001, S110 - a memory note must never break the monitor
                pass

    def status(self) -> dict:
        return {
            "state": "drift" if self._flagged else "ok",
            "calibrated": bool(self.watched.extrinsics.calibrated),
            "fusing": bool(self.watched.fuse),
            "consecutive": self._consecutive,
            "needed": self.config.consecutive,
            "checks": self.checks,
            "last_check": None if self.last is None else self.last.as_dict(),
            "snapshots": len(self.snapshots),
            "candidate": None if self.candidate_path is None else str(self.candidate_path),
            "candidate_rejection": self.last_rejection,
            "reason": self._flag_reason,
        }

    # ── the check ────────────────────────────────────────────────────────

    def _result(self, outcome, reason, now, **kw) -> CheckResult:
        r = CheckResult(outcome, reason, now, **kw)
        self.last = r
        return r

    def _track(self, now):
        """Update the static-arm tracker; returns (q, None) or (None, why)."""
        q = self._q_fn()
        if q is None:
            self._last_q = self._static_since = None
            return None, "arm unavailable (standby or not connected)"
        q = np.asarray(q, dtype=float).ravel()
        moving = bool(self._motion_fn())
        if moving:
            self._last_q = self._static_since = None
            return None, "arm moving"
        if (self._last_q is None or self._last_q.shape != q.shape
                or float(np.max(np.abs(q - self._last_q))) > Q_STATIC_TOL_RAD):
            self._last_q = q
            self._static_since = now
        if now - self._static_since < self.settle_s:
            return None, "arm settling"
        return q, None

    def check_once(self) -> CheckResult:
        with self._lock:
            return self._check(self._clock())

    def _check(self, now) -> CheckResult:
        q, why = self._track(now)
        if q is None:
            return self._result("skipped", why, now)
        if self._surface is None:
            try:
                self._surface = self._surface_fn()
            except Exception as e:  # noqa: BLE001 - no model: the monitor cannot work
                return self._result("skipped", f"no arm surface model ({e})", now)
        frame = self.watched.stream.latest()
        if frame is None:
            return self._result("skipped", "no frame", now)
        if float(getattr(frame, "t", now)) < self._static_since + self.settle_s:
            return self._result("skipped", "frame predates the settled arm", now)
        if frame.depth_m is None or getattr(frame, "depth_source", "") != "sensor":
            return self._result("skipped", "no sensor depth", now)
        q2 = self._q_fn()
        if q2 is None or float(np.max(np.abs(np.asarray(q2, dtype=float).ravel() - q))) \
                > Q_STATIC_TOL_RAD:
            self._last_q = self._static_since = None
            return self._result("skipped", "arm moved during the capture", now)
        kin = self._surface.kin
        sample = DepthSample(q=tuple(float(v) for v in q), T_gripper2base=kin.fk(q),
                             depth_m=np.asarray(frame.depth_m, dtype=np.float32),
                             K=np.asarray(frame.K, dtype=float), label=f"drift_{self.checks:05d}",
                             t=float(getattr(frame, "t", now)))
        self.checks += 1
        r = measure_offset(sample, self._surface, self._T_ref)
        self._remember(sample, now)
        if not r.conclusive:
            res = self._result("inconclusive", r.reason, now, n_inliers=r.n_inliers_after)
        else:
            drift = r.offset_m > self.config.max_offset_m or r.offset_deg > self.config.max_rot_deg
            res = self._result("drift" if drift else "ok",
                               "offset above threshold" if drift else "within threshold", now,
                               offset_m=r.offset_m, offset_deg=r.offset_deg,
                               n_inliers=r.n_inliers_after)
            if drift:
                if self._consecutive == 0:
                    self._run_started_t = now
                self._consecutive += 1
                self._last_estimate = r.T_estimate
                self._event("drift_check", consecutive=self._consecutive,
                            offset_mm=1000 * r.offset_m, offset_deg=r.offset_deg)
                if not self._flagged and self._consecutive >= self.config.consecutive:
                    self._flag(r)
            elif not self._flagged:
                self._consecutive = 0
        if self._flagged:
            self._maybe_candidate(now)
        return res

    # ── flag / adopt ─────────────────────────────────────────────────────

    def _set_fusion(self, fuse, map_depth) -> None:
        if self._watcher is not None:
            self._watcher.set_camera_fusion(self.watched, fuse=fuse, map_depth=map_depth)
        else:
            self.watched.fuse, self.watched.map_depth = bool(fuse), map_depth
            self.watched.generation = getattr(self.watched, "generation", 0) + 1

    def _flag(self, r) -> None:
        reason = (f"extrinsic drift: camera {self.name} no longer matches its calibration (the "
                  f"arm's surface reads {1000 * r.offset_m:.1f} mm / {r.offset_deg:.2f} deg off "
                  f"on {self._consecutive} consecutive static checks); 3D fusion OFF for this "
                  "camera until it is re-calibrated (docs/HANDEYE_CALIBRATION.md, drift monitor)")
        self.watched.extrinsics.invalidate(reason)
        self._set_fusion(False, False)
        self._flagged = True
        self._flag_reason = reason
        # Views taken before the knock describe the OLD camera pose.
        keep = [i for i, t in enumerate(self._snap_t) if t >= (self._run_started_t or 0.0)]
        self.snapshots = [self.snapshots[i] for i in keep]
        self._snap_t = [self._snap_t[i] for i in keep]
        self._attempted_at = 0
        self._log(f"[drift:{self.name}] CAMERA UNCALIBRATED: {reason}")
        self._event("camera_uncalibrated", reason=reason, offset_mm=1000 * r.offset_m,
                    offset_deg=r.offset_deg, consecutive=self._consecutive)
        self._note(f"camera {self.name}: extrinsic drift detected -- 3D fusion off for this "
                   "camera until re-calibration", offset_mm=round(1000 * r.offset_m, 1))

    def _remember(self, sample, now) -> None:
        q = np.asarray(sample.q)
        if any(float(np.max(np.abs(q - np.asarray(s.q)))) < DISTINCT_Q_RAD
               for s in self.snapshots):
            return
        if float(np.mean(sample.depth_m > 0)) < 0.05:
            return
        self.snapshots.append(_compact(sample))
        self._snap_t.append(now)
        if len(self.snapshots) > MAX_SNAPSHOTS:
            self.snapshots.pop(0)
            self._snap_t.pop(0)

    def _maybe_candidate(self, now) -> None:
        if self.candidate_path is not None or len(self.snapshots) < MIN_POSES:
            return
        if len(self.snapshots) < self._attempted_at + 2 and self._attempted_at:
            return          # nothing new enough since the last refused attempt
        from ..calibration.markerless import _diversity

        spread, rot = _diversity(self.snapshots)
        if spread < MIN_TCP_SPREAD_M or rot < MIN_ROTATION_SPREAD_DEG:
            return          # keep collecting: the gate would refuse it anyway
        self._attempted_at = len(self.snapshots)
        self._solve_candidate(now)

    def _solve_candidate(self, now) -> None:
        from ..calibration.dataset import save_hand_eye
        from ..calibration.markerless import record_from_markerless, solve_markerless

        T0 = self._last_estimate if self._last_estimate is not None else self._T_ref
        fit = solve_markerless(self.snapshots, self._surface, T0)
        last = self.snapshots[-1]
        rec = record_from_markerless(
            fit, self.snapshots, camera=self.name, camera_serial=self.camera_serial,
            arm=self.arm, ee_frame=self.ee_frame, K=last.K,
            image_size=(last.depth_m.shape[1], last.depth_m.shape[0]),
            note="PASSIVE CANDIDATE from the extrinsic drift monitor (static views taken during "
                 "normal work; the arm was never moved for it)")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if not rec.acceptable:
            self.last_rejection = "; ".join(rec.rejection_reasons)
            self._event("candidate_rejected", n_views=len(self.snapshots),
                        reasons=rec.rejection_reasons)
            self._log(f"[drift:{self.name}] passive re-calibration from {len(self.snapshots)} "
                      f"views not accepted yet: {self.last_rejection}")
            return
        if self.run_dir is None:
            return
        path = save_hand_eye(self.run_dir / "extrinsics_candidates" /
                             f"{self.name}_{stamp}.handeye.json", rec)
        self.candidate_path = path
        self.last_rejection = None
        self._event("candidate_written", path=str(path), metrics=rec.metrics)
        self._log(f"[drift:{self.name}] candidate re-calibration written: {path}\n"
                  f"[drift:{self.name}] {rec.summary()}")
        if not self.config.auto_apply:
            self._log(f"[drift:{self.name}] NOT applied (extrinsics.drift_monitor.auto_apply is "
                      "false). To adopt it: copy it under configs/calib/ and set in the camera "
                      f"profile\n  extrinsics:\n    hand_eye_json: {path}\n(it is the full "
                      "T_cam2base: drop any hand_eye_compensation_m), then restart; or "
                      "re-run scripts/calibrate_handeye.py --method markerless")
            self._note(f"camera {self.name}: passive re-calibration candidate ready ({path})")
            return
        self._maybe_apply(fit.T_cam2base, path)

    def candidate_is_better(self, T_candidate, views=None) -> tuple[bool, float, float]:
        """(better, explained_by_candidate, explained_by_active) on the most
        recent static views: the share of visible arm points each transform
        explains."""
        from ..calibration.markerless import explained_fraction

        views = list(views if views is not None else self.snapshots[-3:])
        new = explained_fraction(views, self._surface, T_candidate)
        old = explained_fraction(views, self._surface, self._T_ref)
        return (new >= MIN_INLIER_FRACTION and new > old + APPLY_MARGIN), new, old

    def _maybe_apply(self, T_new, path) -> None:
        better, new, old = self.candidate_is_better(T_new)
        if not better:
            self._event("candidate_not_applied", path=str(path), explained_new=new,
                        explained_active=old)
            self._log(f"[drift:{self.name}] candidate NOT applied: it explains {100 * new:.0f} % "
                      f"of the arm in current depth vs {100 * old:.0f} % for the active one")
            return
        self.watched.extrinsics.adopt(T_new, source=f"drift_candidate:{path}")
        self._T_ref = np.array(T_new, dtype=float)
        self._set_fusion(self._orig_fuse, self._orig_map)
        self._flagged = False
        self._flag_reason = None
        self._consecutive = 0
        self._event("candidate_applied", path=str(path), explained_new=new, explained_active=old)
        self._log(f"[drift:{self.name}] candidate APPLIED (auto_apply): explains {100 * new:.0f} % "
                  f"of the arm vs {100 * old:.0f} %; 3D fusion back on for this camera")
        self._note(f"camera {self.name}: re-calibrated passively; 3D fusion back on",
                   path=str(path))

    # ── thread ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"drift-monitor-{self.name}")
        self._thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout_s)
            if not t.is_alive():
                self._thread = None

    def _loop(self) -> None:
        next_check = 0.0
        last_error = None
        while not self._stop.wait(0.25):
            now = self._clock()
            if now < next_check:
                continue
            try:
                res = self.check_once()
                last_error = None
                # A real check waits a period; a skip (settling, moving) polls.
                next_check = now + (self.config.period_s if res.outcome != "skipped" else 0.0)
            except Exception as e:  # noqa: BLE001 - the monitor must never take the runtime down
                msg = f"{type(e).__name__}: {e}"
                if msg != last_error:
                    self._log(f"[drift:{self.name}] check failed: {msg}")
                last_error = msg
                next_check = now + self.config.period_s


def _compact(sample: DepthSample) -> DepthSample:
    """Keep a snapshot at <= ~640 px wide (block median), K scaled to it."""
    from ..calibration.markerless import _block_median, _level_K

    w = sample.depth_m.shape[1]
    s = max(1, round(w / 640))
    if s == 1:
        return sample
    z = _block_median(sample.depth_m, s)
    return DepthSample(q=sample.q, T_gripper2base=sample.T_gripper2base,
                       depth_m=np.where(np.isfinite(z), z, 0.0).astype(np.float32),
                       K=_level_K(sample.K, s), label=sample.label, t=sample.t)
