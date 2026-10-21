"""First powered move of a reBot B601-RS through cascade: rest -> home -> rest.

Why this exists. The bring-up scripts (diag, sign_check, jog) talk to the
motors directly. Before the full runtime is launched, one motion has to prove
that cascade's OWN path works on this arm: the real driver (RebotRSArm, the
`rebot_rs` profile: per-motor models and gains from the SDK, gravity
feedforward), SafeArm and the SafetyHarness, with the reading guards from the
2026-10-10 incident. This script is that motion, kept deliberately slow,
checked and boring. It moved the rig's arm rest -> home -> rest on 2026-10-11
with every segment within 0.013 rad of its target.

What it does, in order, and why:

1. Opens the fixed camera and waits for real frames BEFORE the arm is
   connected. The harness has a perception watchdog (5 s): motion is refused
   when perception is stale. The first attempt on the rig ran without a
   camera and was refused half-way up, leaving the arm held mid-air. The
   watchdog is fed ONLY by frames that actually arrive (the rule of
   calibration/session.py's grab()); there is no synthetic heartbeat, so a
   camera that dies still stops the next segment.
2. Connects through demo._build_arm (the runtime's construction, no
   occupancy map, no motion planner -- it refuses a profile that binds one).
3. Tightens the speed cap (default 0.4 rad/s); it refuses to loosen it.
4. Waits for the camera to beat the attached harness; nothing is commanded
   before that.
5. Refuses unless every joint starts within REST_TOL of park_q: the move is
   planned from the folded rest, and nothing is commanded otherwise.
6. Moves up in SEGMENTS joint-space steps and back down the same way. After
   each step the MEASURED pose must be within TRACK_TOL of the target, else
   soft stop (hold the last commanded pose, torque ON) and exit: nothing else
   is commanded. A SafetyViolation does the same -- including the profile's
   motor-feedback contact stop (safety.contact_monitor, docs/CONTACT_STOP.md),
   which checks the measured joints DURING each step and, when one is
   blocked, holds the measured pose instead. Only the final down step
   relaxes the joint margin, exactly like the demo's park (joints 2/3 have
   their lower limit AT the rest pose).
7. Releases torque only when the arm is measured back at rest (or when
   nothing was ever commanded).

If it stops with the arm raised: support the arm, press the e-stop and lower
it by hand. Do NOT reconnect anything while it is raised: connect() clears
latched motor faults with a stop frame that cuts torque. Every such stop is
remembered (cascade.control.arm_hold, a hold marker on the arm's bus): the
next start, arm-check or calibrate refuses to connect the arm until an
operator supports it and confirms, or the startup check measures it folded
at rest. Over such a hold this script refuses before it connects anything.

Usage (motorbridge-gateway / Studio closed; can0 up; operator at the e-stop):

    python scripts/home_rebot_rs.py --dry-run --camera mock   # MockArm, no bus
    python scripts/home_rebot_rs.py --yes                     # the real arm

Contact-monitor calibration: --contact-log FILE appends every step's monitor
report (per-joint tracking error, lag, stall ratio, read rate) as JSON lines;
scripts/contact_calibrate.py FILE suggests thresholds from them. Each step's
log line also shows its worst lag-compensated tracking error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cascade.perception.freshness import observed_at
from cascade.types import ContactDetected, SafetyViolation

ARM_PROFILE = "rebot_rs"
DEFAULT_MAX_VEL = 0.4      # rad/s; the profile's cap is 0.8 -- only ever tightened
SEGMENTS = 4               # checked steps per leg
SEG_S = 2.5                # s per step (the harness stretches it if needed)
REST_TOL = 0.12            # rad: every joint must start this close to park_q
TRACK_TOL = 0.08           # rad: measured vs commanded after each step
WARMUP_FRAMES = 10


def _fmt(q) -> str:
    return "[" + " ".join(f"{v:+.3f}" for v in q) + "]"


def _hold_key(safe) -> str | None:
    """The bus a hold marker names for the arm behind `safe` (None: a
    backend never held, e.g. the dry run's MockArm)."""
    key = getattr(getattr(safe, "raw", None), "hold_key", None)
    return key if isinstance(key, str) and key else None


def _let_go_first(raw, safe, park, log) -> bool:
    """A connect that went ahead on the operator's confirmation (an arm an
    earlier session left held, or measured raised: cascade.control.arm_hold)
    only HOLDS: the operator's hands are on the arm. Before the first motion
    they let go and type 'go' (whatever --yes said). An arm off its rest is
    released at once by `home_and_back` instead, while they still support
    it. False: they did not go on; the arm stays held and is remembered."""
    if getattr(raw, "connected_on_confirmation", False) is not True:
        return True
    q0 = np.asarray(safe.get_state().q, dtype=float)
    if float(np.max(np.abs(q0 - park))) > REST_TOL:
        return True
    from cascade.control import arm_hold

    if arm_hold.wait_for_let_go(_hold_key(safe), log=log):
        return True
    log("[!] stopped before any motion: the arm stays connected and HOLDING where it is.")
    arm_hold.remember(_hold_key(safe), arm="arm", source="home_rebot_rs.py", log=log,
                      reason="arm-check (home_rebot_rs.py): connected on the operator's confirmation, then "
                             "stopped before any motion: torque stays ON")
    return False


class CameraHeartbeat:
    """Feeds `harness.heartbeat()` from frames that ARRIVE, nothing else."""

    def __init__(self, camera):
        self.camera = camera
        self.frames = 0
        self.beats = 0
        self.error: str | None = None
        self._harness = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if hasattr(self.camera, "open"):
            self.camera.open()
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.camera.get_frame()
            except Exception as e:  # noqa: BLE001  any camera failure = no frame = no beat
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.05)
                continue
            self.frames += 1
            h = self._harness
            if h is not None:
                # At the frame's arrival time: a driver that hands back an old
                # frame never makes the watchdog fresher than that frame.
                h.heartbeat(at=observed_at(frame))
                self.beats += 1

    def wait_frames(self, n: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.frames < n and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.frames >= n

    def attach(self, harness) -> None:
        self._harness = harness

    def wait_beat(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.beats == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.beats > 0

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.camera.close()
        except Exception as e:  # noqa: BLE001  teardown must not mask the run's result
            self.error = self.error or f"close: {type(e).__name__}: {e}"


def _default_build(cfg, dry_run: bool, start_q=None):
    """The runtime's own construction; dry-run swaps only the backend."""
    from cascade.apps import demo

    if not dry_run:
        return demo._build_arm(cfg.arm, lazy_arm=False, occupancy=None, fallback_cfg=cfg)
    from cascade.control.mock_arm import MockArm

    orig = demo.make_arm

    def mock(a, kinematics=None):
        m = MockArm(a, kinematics=kinematics)
        m._q = np.asarray(start_q if start_q is not None else a.park_q, dtype=float).copy()
        return m

    demo.make_arm = mock
    try:
        return demo._build_arm(cfg.arm, lazy_arm=False, occupancy=None, fallback_cfg=cfg)
    finally:
        demo.make_arm = orig


def _contact_reports(safe, sink=None) -> list[dict]:
    """The contact monitor's reports since the last call (one per streamed
    motion), appended to `sink` as JSON lines for contact_calibrate.py."""
    drain = getattr(getattr(safe, "harness", None), "drain_contact_reports", None)
    reports = drain() if callable(drain) else []
    if sink is not None:
        for r in reports:
            sink.write(json.dumps(r) + "\n")
        sink.flush()
    return reports


def _track_note(reports) -> str:
    errs = [max(r["max_track_err_rad"]) for r in reports if r.get("max_track_err_rad")]
    return f"  contact track max {max(errs):.3f} rad" if errs else ""


def _leg(safe, name, a, b, *, last_margin, log, pace_s=0.0, contact_sink=None) -> bool:
    """SEGMENTS checked steps from a to b. False = stop (caller soft-stops)."""
    for k in range(1, SEGMENTS + 1):
        wp = a + (b - a) * (k / SEGMENTS)
        margin = last_margin if k == SEGMENTS else None
        t0 = time.monotonic()
        try:
            settled = safe.move_joints(wp, duration_s=SEG_S, joint_margin=margin)
        finally:
            note = _track_note(_contact_reports(safe, contact_sink))
        if pace_s:
            time.sleep(pace_s)
        q = np.asarray(safe.get_state().q, dtype=float)
        err = float(np.max(np.abs(q - wp)))
        log(f"    {name} {k}/{SEGMENTS}: target {_fmt(wp)}  measured {_fmt(q)}  "
            f"err {err:.3f} rad  settled={settled}  {time.monotonic() - t0:.1f}s{note}")
        if not settled or err > TRACK_TOL:
            log(f"[!] {name} step {k}: the arm did not reach its target "
                f"(err {err:.3f} rad, limit {TRACK_TOL}; settled={settled}). STOP.")
            return False
    return True


def home_and_back(safe, kin, home, park, *, hold_s: float, log=print, pace_s=0.0,
                  contact_sink=None) -> int:
    q0 = np.asarray(safe.get_state().q, dtype=float)
    log(f"[+] start q {_fmt(q0)}   home {_fmt(home)}   park {_fmt(park)}   "
        f"cap {safe.harness.limits.max_joint_vel} rad/s")
    off = float(np.max(np.abs(q0 - park)))
    if off > REST_TOL:
        log(f"[!] not at the folded rest (max |q - park| = {off:.3f} > {REST_TOL}). "
            "Nothing commanded; releasing.")
        safe.disconnect()
        return 1

    def remember_hold(why):
        # The process exits with the arm powered and holding: every later
        # process must know (#426 review Important 1), or the next connect's
        # fault clear (a STOP frame) drops it.
        from cascade.control import arm_hold

        arm_hold.remember(_hold_key(safe), arm="arm", reason=f"arm-check (home_rebot_rs.py): {why}",
                          source="home_rebot_rs.py", log=log)

    def soft_stop(why, held="the last commanded pose"):
        try:
            safe.stop()
        except Exception as e:  # noqa: BLE001  report it; the operator's e-stop is next
            log(f"[!] soft stop failed ({type(e).__name__}: {e})")
        log(f"[!] {why}: holding {held}, torque ON; nothing else will be "
            "commanded. Support the arm, then e-stop and lower it by hand. Do NOT reconnect "
            "while it is raised (connect() clears faults with a torque-off frame).")
        remember_hold(why)
        return 1

    try:
        log("[+] up: rest -> home")
        if not _leg(safe, "up", q0, home, last_margin=None, log=log, pace_s=pace_s,
                    contact_sink=contact_sink):
            return soft_stop("stopped on the way up")
        qh = np.asarray(safe.get_state().q, dtype=float)
        log(f"[+] at home: q {_fmt(qh)}  TCP (base) {_fmt(kin.fk(qh)[:3, 3])} m; "
            f"holding {hold_s:g} s")
        time.sleep(hold_s)
        log("[+] down: home -> rest")
        if not _leg(safe, "down", qh, park, last_margin=0.0, log=log, pace_s=pace_s,
                    contact_sink=contact_sink):
            return soft_stop("stopped on the way down")
        qe = np.asarray(safe.get_state().q, dtype=float)
        if float(np.max(np.abs(qe - park))) > TRACK_TOL:
            log(f"[!] not measured at rest ({_fmt(qe)}); torque stays ON.")
            remember_hold(f"not measured at rest after the way down ({_fmt(qe)}): torque stays ON")
            return 1
        safe.disconnect()
        log(f"[+] back at rest {_fmt(qe)}; torque released (the arm rests on its stops).")
        return 0
    except ContactDetected as e:
        # stopped mid-step by the motor-feedback contact stop: the hold it
        # commanded is the MEASURED pose, and stop() re-asserts exactly that
        return soft_stop(f"CONTACT ({e})", held="the measured pose where it was blocked")
    except SafetyViolation as e:
        return soft_stop(f"SAFETY VIOLATION ({e})")
    except BaseException as e:
        soft_stop(f"{type(e).__name__}: {e}")
        raise


def main(argv=None, *, build=None, make_camera=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="required for the real arm")
    ap.add_argument("--dry-run", action="store_true", help="MockArm backend, no bus")
    ap.add_argument("--start-q", default=None, help="dry-run start pose, comma-separated")
    ap.add_argument("--camera", default="d455f_scene",
                    help="camera profile that feeds the watchdog (mock for a dry run)")
    ap.add_argument("--max-vel", type=float, default=DEFAULT_MAX_VEL,
                    help="speed cap in rad/s; may only be <= the profile's")
    ap.add_argument("--hold-s", type=float, default=3.0, help="seconds to hold at home")
    ap.add_argument("--camera-wait-s", type=float, default=8.0)
    ap.add_argument("--beat-wait-s", type=float, default=2.0)
    ap.add_argument("--contact-log", default=None,
                    help="append each step's contact-monitor report (JSON lines) for "
                         "scripts/contact_calibrate.py")
    args = ap.parse_args(argv)
    log = lambda msg: print(msg, flush=True)

    if not args.dry_run and not args.yes:
        log("[!] the real arm needs --yes (after the operator confirms the e-stop and a "
            "clear workspace). Nothing done.")
        return 2

    from cascade.config import load_demo_config

    cfg = load_demo_config(camera=args.camera, arm=ARM_PROFILE, llm="mock")
    if cfg.arm.get("motion_planner") is not None:
        log("[!] this profile binds a motion planner; this script only streams joints.")
        return 2
    home = np.asarray(cfg.arm.home_q, dtype=float)
    park = np.asarray(cfg.arm.get("park_q") or np.zeros(len(home)), dtype=float)

    if make_camera is None:
        from cascade.perception.camera_base import make_camera as _mk

        camera = _mk(cfg.camera)
    else:
        camera = make_camera(cfg)
    hb = CameraHeartbeat(camera)
    hb.start()
    try:
        if not hb.wait_frames(WARMUP_FRAMES, args.camera_wait_s):
            log(f"[!] camera {args.camera!r} delivered {hb.frames} frames ({hb.error}); the "
                "watchdog would refuse motion. The arm was NOT connected.")
            return 1
        log(f"[+] camera {args.camera!r} live; its frames feed the watchdog")

        from cascade.control.arm_hold import ArmHeldError

        try:
            if build is None:
                start_q = (None if args.start_q is None
                           else [float(v) for v in args.start_q.split(",")])
                _raw, safe, kin = _default_build(cfg, args.dry_run, start_q)
            else:
                _raw, safe, kin = build(cfg, args.dry_run)
        except ArmHeldError as e:
            # Refused before any frame that can cut torque (#426 review
            # Minor 7: a clean message, not a traceback).
            log(f"[!] the arm was NOT connected: {e}")
            log("[!] Folded at rest (within 0.05 rad of park_q), `~/cascade-demo.sh up`'s startup check clears "
                "the hold; or support the arm, confirm at this terminal (python scripts/arm_hold.py confirm: your "
                "sudo password, then the phrase) and run this again within 2 min. It then connects and HOLDS: "
                "let go and type 'go' before it moves.")
            return 3
        h = safe.harness
        if args.max_vel > h.limits.max_joint_vel:
            log(f"[!] --max-vel {args.max_vel} would LOOSEN the profile's cap "
                f"{h.limits.max_joint_vel}. Nothing commanded; releasing.")
            safe.disconnect()
            return 2
        h.limits = dataclasses.replace(h.limits, max_joint_vel=float(args.max_vel))

        hb.attach(h)
        if not hb.wait_beat(args.beat_wait_s):
            log(f"[!] the camera stopped before it reached the harness ({hb.error}). "
                "Nothing commanded; releasing.")
            safe.disconnect()
            return 1
        if not _let_go_first(_raw, safe, park, log):
            return 3
        # A dry run keeps real time (MockArm moves instantly), so the
        # watchdog feed is exercised for as long as the real run lasts.
        if args.contact_log:
            with open(args.contact_log, "a") as sink:
                return home_and_back(safe, kin, home, park, hold_s=args.hold_s, log=log,
                                     pace_s=SEG_S if args.dry_run else 0.0, contact_sink=sink)
        return home_and_back(safe, kin, home, park, hold_s=args.hold_s, log=log,
                             pace_s=SEG_S if args.dry_run else 0.0)
    finally:
        hb.stop()


if __name__ == "__main__":
    raise SystemExit(main())
