"""Hand-eye sample collection through cascade's safety path.

Ported from WRC ``scripts/calib_top_orbbec.py`` (``_collect_loop_auto`` --
rebot_grasp's ``tick_auto`` state machine -- and ``_collect_loop_manual``)
and ``scripts/calib_wrist.py`` (``collect_auto``). The state machine is the
same: move -> settle -> search for a stable marker -> capture -> next pose.
The motion path is cascade's, and that is the point of the port:

* Every preset is vetted by the arm's own ``SafetyHarness.vet_pose`` when
  the sweep is planned, and AGAIN immediately before the arm is commanded
  (the world may have changed); the motion is ``SafeArm.move_planned``
  (route vetting + per-waypoint ``approve()``). WRC called
  ``safe_arm.move_joints`` after its own joint-limit pre-check, and its wrist
  script drove the SDK's ``move_to_traj`` with no harness at all.
* Speed is a fraction (default 0.5) of the harness's ``max_joint_vel`` --
  the arm profile's cap (0.8 rad/s on the reBot RS). It can be lowered,
  never raised: ``SessionConfig`` refuses ``speed_frac > 1``, and SafeArm
  stretches anything faster anyway.
* IK is seeded from ``home_q`` for every pose (the grasp pipeline's rule:
  elbow-down branches dip links below the table) and the sweep is ordered
  greedily in joint space, which on the reBot presets cuts the largest
  single joint move from 2.7 rad (WRC's chained seeding) to 1.6 rad.
* The perception watchdog is fed only by camera frames that actually
  arrived: a dead camera stops the arm, as it does in the runtime.
* FK of the MEASURED joints after settling, never the commanded ones.

Interrupts are the caller's (scripts/calibrate_handeye.py): it halts the
harness and parks through ``SafeArm`` on Ctrl+C.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import MotionHalted, SafetyViolation, SkillError, pose_to_transform
from .dataset import MarkerSpec
from .handeye import MODES, HandEyeSample

#: Default per-move ceiling on the largest single-joint excursion. A preset
#: whose home-seeded IK solution sits further than this from where the arm
#: is would swing the wrist/base through a wide arc next to a camera tripod;
#: skip it rather than sweep blind. (ETH presets ordered greedily: 1.6 rad.)
MAX_JOINT_STEP_RAD = 2.0


@dataclass(frozen=True)
class SessionConfig:
    mode: str
    marker: MarkerSpec
    settle_s: float = 1.0           # after the move settles, before looking
    marker_timeout_s: float = 4.0   # then this long to find a stable marker
    stable_frames: int = 4          # consecutive agreeing detections
    stable_tol_m: float = 0.002     # how much consecutive readings may differ
    max_reprojection_px: float = 2.0
    speed_frac: float = 0.5         # of the harness max_joint_vel; (0, 1]
    min_move_s: float = 3.0
    max_joint_step_rad: float = MAX_JOINT_STEP_RAD

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"bad hand-eye mode {self.mode!r}")
        if not (0.0 < float(self.speed_frac) <= 1.0):
            raise ValueError(f"speed_frac must be in (0, 1] of the harness velocity cap, "
                             f"got {self.speed_frac} (the cap is never raised)")
        if self.stable_frames < 1 or self.min_move_s <= 0 or self.max_joint_step_rad <= 0:
            raise ValueError("stable_frames, min_move_s and max_joint_step_rad must be positive")


def order_by_joint_distance(q0, qs) -> list[int]:
    """Greedy nearest-next ordering (L-inf joint distance) starting at q0."""
    cur = np.asarray(q0, dtype=float)
    left = list(range(len(qs)))
    out = []
    while left:
        k = min(left, key=lambda i: float(np.max(np.abs(np.asarray(qs[i]) - cur))))
        out.append(k)
        left.remove(k)
        cur = np.asarray(qs[k], dtype=float)
    return out


class CollectionSession:
    """Drive a pose sweep and collect (FK, marker) samples.

    Collaborators are duck-typed so the loop runs against the mock stack:
    ``safe_arm`` (cascade SafeArm: ``harness``, ``get_state``,
    ``move_planned``), ``kin`` (``ik``/``fk``), ``camera`` (``get_frame``
    -> Frame with ``K``), ``aruco`` (``detect_frame``).
    """

    def __init__(self, *, safe_arm, kin, camera, aruco, config: SessionConfig, home_q,
                 trace_path=None, capture_dir=None, dist_coeffs=None, log=print,
                 sleep=time.sleep, clock=time.monotonic):
        self.arm = safe_arm
        self.harness = safe_arm.harness
        self.kin = kin
        self.camera = camera
        self.aruco = aruco
        self.cfg = config
        self.n = int(safe_arm.n_joints)
        self.home_q = np.asarray(home_q, dtype=float)[: self.n]
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.capture_dir = None if capture_dir is None else Path(capture_dir)
        self.dist_coeffs = dist_coeffs
        self.log = log
        self.sleep = sleep
        self.clock = clock
        self.samples: list[HandEyeSample] = []
        self.last_frame = None

    # ── trace ────────────────────────────────────────────────────────────

    def _event(self, event: str, **fields) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"event": event, "ts": time.time(), **fields}
        with self.trace_path.open("a") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

    def _skip(self, i, label, reason, detail="") -> None:
        self.log(f"[calib] {label}: skipped ({reason}{': ' + detail if detail else ''})")
        self._event("pose_skipped", i=i, label=label, reason=reason, detail=str(detail))

    # ── perception ───────────────────────────────────────────────────────

    def grab(self):
        """One frame, or None. Only a frame that ARRIVED feeds the watchdog."""
        from ..perception.camera_base import CameraError

        try:
            frame = self.camera.get_frame()
        except CameraError as e:
            self.log(f"[calib] camera: {e}")
            return None
        self.harness.heartbeat()
        self.last_frame = frame
        return frame

    def stable_detection(self):
        """Settle, then wait for N consecutive agreeing marker readings."""
        c = self.cfg
        t0 = self.clock()
        settle_until, deadline = t0 + c.settle_s, t0 + c.settle_s + c.marker_timeout_s
        count, last = 0, None
        while self.clock() < deadline:
            frame = self.grab()
            if frame is None:
                self.sleep(0.05)
                continue
            if self.clock() < settle_until:
                self.sleep(0.03)
                continue
            det = self.aruco.detect_frame(frame, c.marker.size_m, D=self.dist_coeffs,
                                          target_id=c.marker.marker_id)
            if det is None or float(det.reprojection_px) > c.max_reprojection_px:
                count, last = 0, None
                self.sleep(0.03)
                continue
            if last is not None and np.linalg.norm(
                    det.T_marker2cam[:3, 3] - last.T_marker2cam[:3, 3]) > c.stable_tol_m:
                count, last = 1, det      # still moving: restart the run
                continue
            count, last = count + 1, det
            if count >= c.stable_frames:
                return det
        return None

    # ── planning / motion ────────────────────────────────────────────────

    def plan(self, poses) -> list[tuple[int, str, np.ndarray]]:
        """(index, label, q) for every preset that solves and passes vet_pose."""
        out = []
        for i, pose in enumerate(poses):
            label = f"pose_{i:02d}"
            T = pose_to_transform([float(v) for v in pose])
            sol = self.kin.ik(T, self.home_q)
            if not sol.success:
                self._skip(i, label, "ik_failed", f"error {float(sol.error):.4f}")
                continue
            q = np.asarray(sol.q, dtype=float)[: self.n]
            reason = self.harness.vet_pose(q)
            if reason:
                self._skip(i, label, "vetoed", reason)
                continue
            out.append((i, label, q))
        return out

    def _current_q(self) -> np.ndarray:
        return np.asarray(self.arm.get_state().q, dtype=float)[: self.n]

    def visit(self, i, label, q) -> HandEyeSample | None:
        """Vet, move (through SafeArm), settle, detect, capture. None = skipped."""
        c = self.cfg
        if getattr(self.harness, "estopped", False):
            raise SafetyViolation("e-stop latched; calibration stopped")
        # Fresh perception first: begin_motion refuses a stale watchdog.
        self.grab()
        reason = self.harness.vet_pose(q)          # immediately before motion
        if reason:
            self._skip(i, label, "vetoed", reason)
            return None
        jump = float(np.max(np.abs(q - self._current_q())))
        if jump > c.max_joint_step_rad:
            self._skip(i, label, "joint_jump",
                       f"{jump:.2f} rad > {c.max_joint_step_rad:.2f} rad from the current pose")
            return None
        cap = float(self.harness.limits.max_joint_vel)
        duration = max(c.min_move_s, 1.875 * jump / (c.speed_frac * cap))
        self.log(f"[calib] {label}: moving ({jump:.2f} rad, {duration:.1f} s)")
        try:
            ok = self.arm.move_planned(q, duration_s=duration)
        except MotionHalted:
            raise
        except (SafetyViolation, SkillError) as e:
            if getattr(self.harness, "estopped", False):
                raise
            self._skip(i, label, "refused", str(e))
            return None
        if ok is False:
            self._skip(i, label, "not_settled")
            return None
        det = self.stable_detection()
        if det is None:
            self.log(f"[calib] {label}: no stable marker")
            self._event("sample_skipped", i=i, label=label, reason="no_marker")
            return None
        q_meas = self._current_q()
        G = np.asarray(self.kin.fk(q_meas), dtype=float)
        sample = HandEyeSample(T_gripper2base=G, T_marker2cam=np.asarray(det.T_marker2cam),
                               label=label, q=tuple(float(v) for v in q_meas),
                               reprojection_px=float(det.reprojection_px))
        self.samples.append(sample)
        self._save_capture(label, det)
        self._event("sample_recorded", i=i, label=label, q=list(sample.q),
                    reprojection_px=sample.reprojection_px,
                    ambiguity=float(getattr(det, "ambiguity", 0.0)))
        self.log(f"[calib] {label}: captured sample {len(self.samples)} "
                 f"(reprojection {sample.reprojection_px:.2f} px)")
        return sample

    def _save_capture(self, label, det) -> None:
        draw = getattr(self.aruco, "draw", None)
        if self.capture_dir is None or self.last_frame is None or draw is None:
            return
        try:
            import cv2

            self.capture_dir.mkdir(parents=True, exist_ok=True)
            img = draw(self.last_frame.rgb, det, self.last_frame.K, self.cfg.marker.size_m)
            cv2.imwrite(str(self.capture_dir / f"{label}.png"), img)
        except Exception as e:  # an audit image must never cost a sample
            self.log(f"[calib] could not save capture for {label}: {e}")

    # ── the two sweeps ───────────────────────────────────────────────────

    def _run(self, poses, step) -> list[HandEyeSample]:
        self._event("session_start", mode=self.cfg.mode, n_poses=len(poses),
                    marker=self.cfg.marker.to_json(), speed_frac=self.cfg.speed_frac)
        reason = "all poses visited"
        try:
            plan = self.plan(poses)
            order = order_by_joint_distance(self._current_q(), [q for _, _, q in plan])
            for k, idx in enumerate(order):
                i, label, q = plan[idx]
                if step(k, len(order), i, label, q) is False:
                    reason = "operator finished"
                    break
        except BaseException as e:
            reason = f"interrupted ({type(e).__name__})"
            raise
        finally:
            self._event("session_end", n_collected=len(self.samples), finish_reason=reason)
        return list(self.samples)

    def run_auto(self, poses) -> list[HandEyeSample]:
        """Visit every vetted preset; returns the samples collected."""
        def step(k, n, i, label, q):
            self.log(f"[calib] auto {k + 1}/{n}: {label}")
            self.visit(i, label, q)
        return self._run(poses, step)

    def run_manual(self, poses, prompt=input) -> list[HandEyeSample]:
        """Same sweep, one ENTER per pose: the operator gates every motion."""
        def step(k, n, i, label, q):
            try:
                ans = prompt(f"[calib] {k + 1}/{n} {label}: ENTER = move + capture, "
                             "s = skip, q = finish > ").strip().lower()
            except EOFError:
                return False
            if ans in ("q", "quit"):
                return False
            if ans in ("s", "skip"):
                self._skip(i, label, "operator_skip")
                return None
            self.visit(i, label, q)
            return None
        return self._run(poses, step)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ── preset registry ──────────────────────────────────────────────────────


def load_poses(arm: str, mode: str, path=None) -> list[list[float]]:
    """Preset poses ``[x, y, z, roll, pitch, yaw]`` (base frame, URDF rpy).

    ``path`` overrides with a YAML/JSON list (or a registry-shaped file).
    Otherwise ``configs/calibration/handeye_poses.yaml`` maps the arm
    profile to a named set -- explicitly, never by guessing from the model.
    """
    import yaml

    from ..config import PACKAGE_ROOT

    if mode not in MODES:
        raise ValueError(f"bad hand-eye mode {mode!r}")
    if path is not None:
        data = yaml.safe_load(Path(path).expanduser().read_text())
        if isinstance(data, dict):
            data = data.get(mode)
        return _check_poses(data, str(path))
    registry = PACKAGE_ROOT / "configs" / "calibration" / "handeye_poses.yaml"
    data = yaml.safe_load(registry.read_text())
    set_name = (data.get("arms") or {}).get(arm)
    if set_name is None:
        raise ValueError(
            f"no hand-eye preset poses registered for arm profile {arm!r} in {registry}; "
            "add an `arms:` entry (measure reachability first) or pass --poses FILE")
    poses = ((data.get("sets") or {}).get(set_name) or {}).get(mode)
    return _check_poses(poses, f"{registry} set {set_name!r} {mode}")


def _check_poses(poses, where) -> list[list[float]]:
    if not poses:
        raise ValueError(f"no poses in {where}")
    out = []
    for p in poses:
        v = [float(x) for x in p]
        if len(v) != 6 or not np.all(np.isfinite(v)):
            raise ValueError(f"{where}: pose {p!r} is not [x, y, z, roll, pitch, yaw]")
        out.append(v)
    return out
