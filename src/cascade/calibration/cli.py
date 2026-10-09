"""Hand-eye calibration CLI (camera- and arm-agnostic).

Ported from WRC ``scripts/calib_top_orbbec.py`` (eye-to-hand) and
``scripts/calib_wrist.py`` (eye-in-hand) as ONE command: the mounting comes
from the camera profile's ``extrinsics.mode``, the camera from
``make_camera`` (RealSense, Orbbec, UVC, ... -- intrinsics from its frames),
the arm from ``apps.demo._build_arm`` (kinematics -> the profile's own
SafetyHarness -> SafeArm, a LazyArm so the motors stay untouched until the
first vetted motion). Entry points: ``python scripts/calibrate_handeye.py``
and ``cascade-calib-handeye``. Operator procedure: docs/HANDEYE_CALIBRATION.md.

``--method markerless`` (eye-to-hand RGB-D cameras only) runs the SAME
vetted sweep but records depth of the arm at each settled pose and fits the
arm's own meshes to it (calibration/markerless.py); no marker is mounted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import MarkerSpec, record_from_fit
from .frames import so3_exp
from .handeye import EYE_IN_HAND, EYE_TO_HAND, solve_hand_eye

#: --verify passes when the known-point RMSE is below this (Seeed's wiki
#: section 5.3 / rebot_grasp set.py:320: "RMSE < 10 mm").
VERIFY_MAX_RMSE_M = 0.010
#: Fewer samples than this are not worth solving (the gate wants 8 inliers).
MIN_SAMPLES_TO_SOLVE = 6
METHODS = ("marker", "markerless")
#: --dry-run --method markerless starts the solver this far from the truth
#: (a deliberately wrong "profile guess": 8 cm and 10 deg).
DRY_RUN_INIT_ERROR = (0.08, 10.0)


def _T(t, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _dry_run_truth():
    """Ground truth for --dry-run, shaped like the real rig.

    eye_to_hand: a camera ~0.95 m above the base looking straight down
    (the axes of d455f_scene.yaml's placeholder, tilted 5 deg), marker face
    up 3 cm above the TCP. eye_in_hand: a camera behind the TCP looking
    along the approach axis, 16 deg down (WRC's measured wrist mount,
    rounded), marker flat on the table 0.45 m in front of the base.
    """
    down = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    X_eth = _T([0.30, -0.03, 0.95], down @ so3_exp([0.05, -0.06, 0.0]))
    Y_eth = _T([0.0, 0.0, 0.03], so3_exp([0.0, 0.0, 0.3]))
    along = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    X_eih = _T([-0.065, -0.004, 0.020], along @ so3_exp([-0.28, 0.0, 0.0]))
    Z_eih = _T([0.45, 0.0, 0.0], so3_exp([0.0, 0.0, 0.4]))
    return {EYE_TO_HAND: (X_eth, Y_eth), EYE_IN_HAND: (X_eih, Z_eih)}


@dataclass
class DryRunRig:
    cfg: object
    arm_name: str
    raw_arm: object
    safe_arm: object
    kin: object
    camera: object
    T_hand_eye: np.ndarray
    T_marker: np.ndarray | None
    home_q: np.ndarray
    ee_frame: str
    model_path: str = ""
    T_init: np.ndarray | None = None     # markerless: the deliberately wrong start


def dry_run_rig(arm: str, mode: str, *, noise_px: float = 0.5, corrupt_poses=(),
                seed: int = 0, marker: MarkerSpec = MarkerSpec(),
                method: str = "marker") -> DryRunRig:
    """The arm profile's kinematics, harness and velocity cap on a MockArm,
    plus a synthetic camera (a rendered marker, or for ``method="markerless"``
    rendered depth of the arm). Never builds a hardware backend: the
    profile's ``type`` is replaced by ``mock`` and the result is checked."""
    from ..apps.demo import _arm_cfgs, _build_arm
    from ..config import load_demo_config
    from ..control.mock_arm import MockArm
    from .synthetic import SyntheticMarkerCamera

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "markerless" and mode != EYE_TO_HAND:
        raise ValueError("markerless calibration is eye_to_hand only")
    cfg = load_demo_config(camera="mock", arm=arm, llm="mock")
    acfg = _arm_cfgs(cfg)[0]
    acfg._data["type"] = "mock"
    raw, safe_arm, kin = _build_arm(acfg, False, None, cfg)
    if not isinstance(raw, MockArm):  # pragma: no cover - defensive, fail closed
        raw.disconnect()
        raise RuntimeError("dry run built a non-mock arm backend; refusing")
    T_he, T_m = _dry_run_truth()[mode]
    model_path = str(acfg.model)
    T_init = None
    if method == "markerless":
        from .synthetic_depth import SyntheticDepthCamera

        camera = SyntheticDepthCamera(kin, model_path, T_he, lambda: safe_arm.get_state().q,
                                      seed=seed)
        trans, deg = DRY_RUN_INIT_ERROR
        T_init = T_he @ _T(trans * np.array([1.0, 1.0, -0.5]) / 1.5,
                           so3_exp(np.radians(deg) * np.array([1.0, -0.5, 0.3])
                                   / np.linalg.norm([1.0, -0.5, 0.3])))
        T_m = None
    else:
        camera = SyntheticMarkerCamera(
            mode, T_he, T_m, lambda: kin.fk(safe_arm.get_state().q), noise_px=noise_px,
            corrupt_poses=corrupt_poses, seed=seed, size_m=marker.size_m,
            marker_id=marker.marker_id, dictionary=marker.dictionary)
    return DryRunRig(cfg=cfg, arm_name=arm, raw_arm=raw, safe_arm=safe_arm, kin=kin,
                     camera=camera, T_hand_eye=T_he, T_marker=T_m,
                     home_q=np.asarray(acfg.home_q, dtype=float),
                     ee_frame=str(acfg.get("ee_frame", "")), model_path=model_path,
                     T_init=T_init)


def solve_and_record(samples, *, mode, marker, camera="", camera_serial="", arm="",
                     ee_frame="", K=None, D=None, image_size=None, note=""):
    """Joint solve + schema-v1 record (acceptable or not -- check it)."""
    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        raise ValueError(f"only {len(samples)} samples; need >= {MIN_SAMPLES_TO_SOLVE} to solve "
                         "(check the marker is visible across the sweep)")
    fit = solve_hand_eye(samples, mode)
    return record_from_fit(fit, samples, marker=marker, camera=camera,
                           camera_serial=camera_serial, arm=arm, ee_frame=ee_frame,
                           K=K, D=D, image_size=image_size, note=note)


# ── shared plumbing ──────────────────────────────────────────────────────


@dataclass
class Rig:
    """What a calibration run drives: a SafeArm (and its raw backend, for the
    final disconnect only), kinematics, and an opened camera."""

    safe_arm: object
    raw_arm: object
    kin: object
    camera: object
    home_q: np.ndarray
    ee_frame: str
    camera_serial: str = ""
    model_path: str = ""


def _prompt(msg: str = "") -> str:
    return input(msg)


def _err(msg: str) -> None:
    import sys

    print(msg, file=sys.stderr)


def open_camera(ccfg):
    """Open the profile's camera through cascade's own backends (RealSense,
    UVC, ...): intrinsics then come from every Frame's K."""
    from ..perception.camera_base import make_camera

    cam = make_camera(ccfg)
    cam.open()
    return cam


def build_rig(cfg, args) -> Rig:
    """The REAL rig: the arm exactly as build_runtime builds it -- kinematics
    -> the profile's own SafetyHarness -> SafeArm over a LazyArm (motors are
    untouched until the first vetted motion) -- and the profile's camera."""
    from ..apps.demo import _arm_cfgs, _build_arm

    acfg = _arm_cfgs(cfg)[0]
    raw, safe_arm, kin = _build_arm(acfg, True, None, cfg)
    try:
        camera = open_camera(cfg.camera)
    except BaseException:
        raw.disconnect()
        raise
    return Rig(safe_arm=safe_arm, raw_arm=raw, kin=kin, camera=camera,
               home_q=np.asarray(acfg.home_q, dtype=float),
               ee_frame=str(acfg.get("ee_frame", "")),
               camera_serial=str(cfg.camera.get("serial", "") or ""),
               model_path=str(acfg.get("model", "")))


def _cleanup(rig: Rig | None, *, park: bool) -> None:
    """Runtime shutdown order: park through SafeArm (if the arm may have
    moved), then torque off, then release the camera. A fresh frame feeds
    the watchdog first -- only if one actually arrives; with a dead camera
    the harness refuses the park, as it does in the runtime."""
    if rig is None:
        return
    from types import SimpleNamespace

    from ..apps.demo import _park_arm

    if park:
        try:
            rig.camera.get_frame()
            rig.safe_arm.harness.heartbeat()
        except Exception as e:
            print(f"[calib] no camera frame before park ({e}); the watchdog may refuse it")
        _park_arm(SimpleNamespace(arm=rig.safe_arm))
    try:
        rig.raw_arm.disconnect()
    except Exception as e:
        print(f"[calib] arm disconnect failed: {e}")
    try:
        rig.camera.close()
    except Exception as e:
        print(f"[calib] camera close failed: {e}")


def _speed_ok(frac) -> bool:
    return 0.0 < float(frac) <= 1.0


def _marker(args) -> MarkerSpec:
    return MarkerSpec(dictionary=args.dict, marker_id=args.marker_id, size_m=args.marker_size)


#: Real-rig settle after each move: the reBot's joints ring for ~0.5 s after
#: a min-jerk stop, and a marker read during it lands in the fit.
SETTLE_S = 1.0


def _session_config(args, mode, *, dry_run=False):
    from .session import SessionConfig

    # The MockArm does not move in real time: a dry run has nothing to settle.
    settle = args.settle_time if args.settle_time is not None else (0.0 if dry_run else SETTLE_S)
    return SessionConfig(mode=mode, marker=_marker(args), settle_s=settle,
                         marker_timeout_s=args.marker_timeout, stable_frames=args.stable_frames,
                         speed_frac=args.speed_frac, depth_frames=args.depth_frames)


def _default_out(camera: str, arm: str):
    from ..config import CONFIG_DIR

    return CONFIG_DIR / "calib" / f"{camera}_{arm}.handeye.json"


def _profile_path(path) -> str:
    """How a camera profile should reference ``path``: ``${repo}/...`` inside
    the checkout (config.py expands it), else absolute. Never relative -- the
    runtime would resolve that against whatever directory it was started in."""
    from pathlib import Path

    from ..config import PACKAGE_ROOT

    p = Path(path).expanduser().resolve()
    try:
        return "${repo}/" + p.relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def _run_dir(args, camera: str):
    import time
    from pathlib import Path

    from ..config import PACKAGE_ROOT

    if args.run_dir:
        return Path(args.run_dir)
    return PACKAGE_ROOT / "runs" / "handeye" / f"{time.strftime('%Y%m%d_%H%M%S')}_{camera}"


MOUNTING = {
    EYE_TO_HAND: "marker fixed FLAT ON TOP of the gripper, face up, so the fixed camera sees it "
                 "across the sweep; it must not shift relative to the gripper",
    EYE_IN_HAND: "marker FLAT ON THE TABLE ~0.45 m in front of the base, taped down; it must "
                 "not move during the sweep",
    "markerless": "NO marker: the camera fits the arm's own meshes in depth. Keep the arm "
                  "unobstructed (nothing held, no clutter on it) and in the camera's view",
}


# ── markerless prerequisites ─────────────────────────────────────────────

#: Camera backends with no depth stream of their own (cascade's `uvc` is
#: RGB; its plane-cast "depth" is the table, not the arm).
_RGB_ONLY_TYPES = ("uvc",)


def markerless_refusal(ccfg, mode) -> str | None:
    """Why this camera profile cannot be calibrated markerless, or None."""
    if mode != EYE_TO_HAND:
        return (f"--method markerless is eye_to_hand only (profile mode {mode!r}): an "
                "eye-in-hand (wrist) camera does not see the arm it rides on; use the marker "
                "method for it")
    kind = str(ccfg.get("type", ""))
    if kind in _RGB_ONLY_TYPES or ccfg.get("rgb_only"):
        return (f"--method markerless needs a camera with SENSOR depth; profile type {kind!r} "
                "has none (plane-cast or monocular depth is not a measurement of the arm)")
    return None


def markerless_initial_guess(ccfg) -> np.ndarray | None:
    """The solver's start: the profile's accepted hand-eye record if any,
    else its inline (placeholder) ``extrinsics.T``; None when there is no
    guess at all (identity would start the camera at the base origin)."""
    from .dataset import read_hand_eye
    from .frames import is_se3

    ext = ccfg.get("extrinsics") or {}
    path = ext.get("hand_eye_json") if hasattr(ext, "get") else None
    if path:
        try:
            rec = read_hand_eye(path)
            if rec.acceptable and rec.mode == EYE_TO_HAND:
                return np.asarray(rec.T_hand_eye, dtype=float)
        except (FileNotFoundError, ValueError):
            pass
    T = ext.get("T") if hasattr(ext, "get") else None
    if T is None:
        return None
    T = np.asarray(T, dtype=float).reshape(4, 4)
    if not is_se3(T) or np.allclose(T, np.eye(4)):
        return None
    return T


def _camera_has_depth(camera) -> bool:
    v = getattr(camera, "has_depth", None)
    try:
        return bool(v() if callable(v) else v)
    except Exception:  # noqa: BLE001 - a probe that fails is a camera without depth
        return False


# ── --list / --bind ──────────────────────────────────────────────────────


def cmd_list(args) -> int:
    from .devices import list_devices

    devices, notes = list_devices()
    pinned = _pinned_serials(args.config_dir)
    for d in devices:
        who = ", ".join(pinned.get(d.serial, [])) or "-"
        extra = " ".join(x for x in (f"fw {d.firmware}" if d.firmware else "",
                                     f"usb {d.usb}" if d.usb else "") if x)
        print(f"{d.backend:9s}  {d.serial:16s}  {d.name}  {extra}  profiles: {who}")
    for n in notes:
        print(f"note: {n}")
    if not devices:
        print("no cameras found")
        return 1
    return 0


def _camera_dir(config_dir):
    from pathlib import Path

    from ..config import CONFIG_DIR

    return (Path(config_dir) if config_dir else CONFIG_DIR) / "cameras"


def _pinned_serials(config_dir) -> dict[str, list[str]]:
    import yaml

    out: dict[str, list[str]] = {}
    for f in sorted(_camera_dir(config_dir).glob("*.yaml")):
        try:
            serial = (yaml.safe_load(f.read_text()) or {}).get("serial")
        except Exception:
            continue
        if serial:
            out.setdefault(str(serial), []).append(f.stem)
    return out


def _set_serial(text: str, serial: str) -> str:
    """Replace (or insert) the top-level ``serial:`` line, keeping comments."""
    import re

    line = f'serial: "{serial}"'
    pat = re.compile(r'^serial:[ \t]*(?:"[^"\n]*"|\'[^\'\n]*\'|[^#\n]*?)([ \t]*#[^\n]*)?[ \t]*$',
                     re.MULTILINE)
    if pat.search(text):
        return pat.sub(lambda m: line + (m.group(1) or ""), text, count=1)
    typ = re.compile(r"^type:[^\n]*\n", re.MULTILINE)
    m = typ.search(text)
    if m:
        return text[: m.end()] + line + "\n" + text[m.end():]
    return line + "\n" + text


def cmd_bind(args) -> int:
    """Pin a CONNECTED camera's serial into a camera profile (comments kept)."""
    import yaml

    from .devices import list_devices

    path = _camera_dir(args.config_dir) / f"{args.bind}.yaml"
    if not path.exists():
        _err(f"no camera profile {path}")
        return 2
    if args.serial is None and args.index is None:
        _err("--bind needs --serial SERIAL or --index N (see --list)")
        return 2
    text = path.read_text()
    kind = (yaml.safe_load(text) or {}).get("type")
    devices, notes = list_devices([kind] if kind in ("realsense", "orbbec") else None)
    for n in notes:
        print(f"note: {n}")
    if args.index is not None:
        if not 0 <= args.index < len(devices):
            _err(f"--index {args.index}: {len(devices)} {kind or ''} camera(s) connected")
            return 2
        serial = devices[args.index].serial
    else:
        serial = str(args.serial)
        if serial not in {d.serial for d in devices}:
            _err(f"serial {serial} is not connected ({[d.serial for d in devices]}); "
                 "refusing to pin a unit that cannot be seen")
            return 2
    new = _set_serial(text, serial)
    if str((yaml.safe_load(new) or {}).get("serial")) != serial:  # pragma: no cover
        _err(f"could not rewrite the serial in {path}; edit it by hand")
        return 2
    path.write_text(new)
    print(f"[calib] {args.bind}: serial pinned to {serial} ({path})")
    return 0


# ── calibration run (real and --dry-run share everything after the rig) ──


def _sweep(rig: Rig, args, mode, poses, run_dir, state: dict, dry_run: bool):
    from .aruco import ArucoSession, camera_dist_coeffs
    from .session import CollectionSession, DepthCollectionSession

    if args.method == "markerless":
        session = DepthCollectionSession(
            safe_arm=rig.safe_arm, kin=rig.kin, camera=rig.camera,
            config=_session_config(args, mode, dry_run=dry_run), home_q=rig.home_q,
            trace_path=run_dir / "trace.jsonl", capture_dir=run_dir / "captures")
        state["session"] = session
        state["moved"] = True
        if args.manual:
            return session.run_manual(poses, prompt=lambda m: _prompt(m), start_home=True)
        return session.run_auto(poses, start_home=True)
    session = CollectionSession(
        safe_arm=rig.safe_arm, kin=rig.kin, camera=rig.camera,
        aruco=ArucoSession(args.dict), config=_session_config(args, mode, dry_run=dry_run), home_q=rig.home_q,
        trace_path=run_dir / "trace.jsonl", capture_dir=run_dir / "captures",
        dist_coeffs=camera_dist_coeffs(rig.camera))
    state["session"] = session
    state["moved"] = True          # from here on the arm may move: park on exit
    if args.manual:
        return session.run_manual(poses, prompt=lambda m: _prompt(m), start_home=True)
    return session.run_auto(poses, start_home=True)


def _solve_and_save(samples, *, args, mode, camera, arm, rig, session, out, run_dir,
                    dry_truth=None) -> int:

    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        print(f"[calib] only {len(samples)} samples (need >= {MIN_SAMPLES_TO_SOLVE}); nothing "
              f"saved. Check the marker stays visible across the sweep (trace: {run_dir})")
        return 1
    frame = session.last_frame
    K = None if frame is None else frame.K
    size = None if frame is None else (frame.rgb.shape[1], frame.rgb.shape[0])
    note = ("DRY RUN: synthetic camera + MockArm; NOT a calibration of any camera"
            if dry_truth is not None else "")
    record = solve_and_record(samples, mode=mode, marker=_marker(args), camera=camera,
                              camera_serial=rig.camera_serial, arm=arm, ee_frame=rig.ee_frame,
                              K=K, D=session.dist_coeffs, image_size=size, note=note)
    return _report_and_save(record, mode=mode, camera=camera, out=out, run_dir=run_dir,
                            dry_truth=dry_truth)


def _solve_and_save_markerless(samples, *, camera, arm, rig, session, out, run_dir, surface,
                               T_init, dry_truth=None) -> int:
    from .markerless import record_from_markerless, solve_markerless

    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        print(f"[calib] only {len(samples)} depth samples (need >= {MIN_SAMPLES_TO_SOLVE}); "
              f"nothing saved. Check the arm stays in the camera's view (trace: {run_dir})")
        return 1
    print(f"[calib] fitting the arm's meshes to {len(samples)} depth samples ...")
    fit = solve_markerless(samples, surface, T_init)
    last = samples[-1]
    note = ("DRY RUN: synthetic depth camera + MockArm; NOT a calibration of any camera"
            if dry_truth is not None else "")
    record = record_from_markerless(
        fit, samples, camera=camera, camera_serial=rig.camera_serial, arm=arm,
        ee_frame=rig.ee_frame, K=last.K, image_size=(last.depth_m.shape[1], last.depth_m.shape[0]),
        note=note, depth_files=getattr(session, "depth_files", ()))
    return _report_and_save(record, mode=EYE_TO_HAND, camera=camera, out=out, run_dir=run_dir,
                            dry_truth=dry_truth)


def _report_and_save(record, *, mode, camera, out, run_dir, dry_truth=None) -> int:
    from .dataset import save_hand_eye
    from .frames import pose_error, se3_inv

    print(record.summary())
    if dry_truth is not None:
        err = pose_error(se3_inv(dry_truth) @ record.T_hand_eye)
        mm, deg = 1000 * float(np.linalg.norm(err[:3])), float(np.degrees(np.linalg.norm(err[3:])))
        print(f"[calib] DRY RUN recovered the synthetic hand-eye within {mm:.2f} mm / {deg:.3f} deg")
        path = save_hand_eye(out, record)
        print(f"[calib] dry-run record -> {path}")
        return 0 if record.acceptable and mm < 5.0 and deg < 1.0 else 1
    if not record.acceptable:
        path = save_hand_eye(run_dir / f"{camera}.REJECTED.json", record)
        print("[calib] REJECTED, not written to the profile output:")
        for r in record.rejection_reasons:
            print(f"  - {r}")
        print(f"[calib] kept for diagnosis: {path}")
        return 1
    save_hand_eye(run_dir / f"{camera}.handeye.json", record)
    path = save_hand_eye(out, record)
    print(f"[calib] saved {path}")
    print("[calib] point the camera profile at it:\n"
          f"  extrinsics:\n    mode: {mode}\n    hand_eye_json: {_profile_path(path)}\n"
          "then verify: --verify <record> --camera <profile> --known-points ...")
    return 0


def _describe_error(T, T_ref) -> str:
    from .frames import pose_error, se3_inv

    e = pose_error(se3_inv(T_ref) @ T)
    return (f"{1000 * float(np.linalg.norm(e[:3])):.1f} mm / "
            f"{float(np.degrees(np.linalg.norm(e[3:]))):.1f} deg")


def cmd_calibrate(args, *, dry_run: bool) -> int:
    from pathlib import Path

    from ..apps.signal_stop import SignalRequest, StopSignals
    from ..config import load_demo_config
    from ..types import MotionHalted, SafetyViolation, SkillError
    from .handeye import MODES
    from .session import load_poses

    camera = args.camera or ("d455f_scene" if dry_run else None)
    arm = args.arm or ("rebot_rs" if dry_run else None)
    if camera is None or arm is None:
        _err("a calibration run needs --camera PROFILE and --arm PROFILE")
        return 2
    try:
        cfg = load_demo_config(camera=camera, arm=arm, llm="mock")
    except (FileNotFoundError, ValueError) as e:
        _err(f"config: {e}")
        return 2
    ext = cfg.camera.get("extrinsics") or {}
    mode = ext.get("mode") if hasattr(ext, "get") else None
    if mode not in MODES:
        _err(f"camera profile {camera!r} has no extrinsics.mode (eye_to_hand|eye_in_hand)")
        return 2
    markerless = args.method == "markerless"
    T_init = None
    if markerless:
        why = markerless_refusal(cfg.camera, mode)
        if why:
            _err(why)
            return 2
        if not dry_run:
            T_init = markerless_initial_guess(cfg.camera)
            if T_init is None:
                _err(f"--method markerless starts from the camera profile's rough T_cam2base: "
                     f"give {camera!r} an extrinsics.T (camera ~0.6-1 m above the table, "
                     "looking down; a few cm / ~15 deg is close enough)")
                return 2
    try:
        poses = load_poses(arm, mode, args.poses)
        _session_config(args, mode, dry_run=dry_run)
    except ValueError as e:
        _err(str(e))
        return 2
    run_dir = _run_dir(args, camera)
    out = Path(args.out) if args.out else _default_out(camera, arm)
    if dry_run:
        dry_out = (Path(args.dry_run_out) if args.dry_run_out
                   else run_dir / f"{camera}.DRYRUN.handeye.json")
        if dry_out.resolve() == out.resolve():
            _err(f"--dry-run-out {dry_out} is the real calibration output; refusing")
            return 2
        out = dry_out
    elif not cfg.camera.get("serial") and cfg.camera.get("type") in ("realsense", "orbbec"):
        _err(f"camera profile {camera!r} pins no serial; bind it first (--bind {camera} "
             "--serial ...): with two identical units an unpinned profile opens either one")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[calib] {'DRY RUN ' if dry_run else ''}{mode} camera={camera} "
          f"serial={cfg.camera.get('serial', '') or '-'} arm={arm} poses={len(poses)} "
          f"mode={'manual' if args.manual else 'auto'} method={args.method}")
    if markerless:
        print(f"[calib] mount: {MOUNTING['markerless']}")
    else:
        print(f"[calib] mount: {MOUNTING[mode]}")
        print(f"[calib] marker {args.dict} id {args.marker_id}, {1000 * args.marker_size:.1f} mm "
              "(the MEASURED printed size)")
    print(f"[calib] speed: {args.speed_frac:.2f} x the arm profile's max_joint_vel; every pose "
          "vetted by the safety harness; Ctrl+C = halt + park + torque off")
    print(f"[calib] trace -> {run_dir}; output -> {out}")

    rig, samples, status, state = None, None, None, {"moved": False}
    dry_truth = None
    surface = None
    with StopSignals() as signals:
        try:
            with signals.defer():
                if dry_run:
                    dr = dry_run_rig(arm, mode, noise_px=args.noise_px, corrupt_poses={5},
                                     seed=args.seed, marker=_marker(args), method=args.method)
                    dry_truth = dr.T_hand_eye
                    rig = Rig(safe_arm=dr.safe_arm, raw_arm=dr.raw_arm, kin=dr.kin,
                              camera=dr.camera, home_q=dr.home_q, ee_frame=dr.ee_frame,
                              camera_serial=dr.camera.serial, model_path=dr.model_path)
                    if markerless:
                        T_init = dr.T_init
                        print("[calib] DRY RUN initial guess "
                              f"{_describe_error(T_init, dry_truth)} off the synthetic truth")
                else:
                    rig = build_rig(cfg, args)
            signals.checkpoint()
            if markerless:
                # Before anything moves: a depth stream and the arm's meshes.
                if not _camera_has_depth(rig.camera):
                    _err(f"camera {camera!r} delivers no depth; --method markerless fits the "
                         "arm in DEPTH. Nothing moved.")
                    status = 2
                else:
                    from .robot_surface import RobotSurface

                    try:
                        surface = RobotSurface.from_kinematics(rig.kin, rig.model_path)
                    except (FileNotFoundError, ValueError) as e:
                        _err(f"no surface model for arm {arm!r}: {e}. Nothing moved.")
                        status = 2
            if status is None and not dry_run and not args.yes:
                ans = _prompt("[calib] Arm will move. E-stop in reach, workspace clear, "
                              + ("arm unobstructed? " if markerless else "marker mounted? ")
                              + "type 'yes' to start > ")
                if ans.strip().lower() != "yes":
                    print("[calib] aborted by operator; nothing moved")
                    status = 1
            if status is None:
                samples = _sweep(rig, args, mode, poses, run_dir, state, dry_run)
        except SignalRequest as e:
            if rig is not None:
                try:
                    rig.safe_arm.harness.halt(f"operator interrupt (signal {e.signum})")
                except Exception:
                    pass
            print("[calib] interrupted: halting, parking, torque off")
            status = 130
        except (SafetyViolation, MotionHalted, SkillError) as e:
            print(f"[calib] stopped: {type(e).__name__}: {e}")
            status = 1
        finally:
            with signals.defer():
                _cleanup(rig, park=state["moved"])
        if status is not None:
            return status
        try:
            if markerless:
                return _solve_and_save_markerless(
                    samples, camera=camera, arm=arm, rig=rig, session=state["session"], out=out,
                    run_dir=run_dir, surface=surface, T_init=T_init, dry_truth=dry_truth)
            return _solve_and_save(samples, args=args, mode=mode, camera=camera, arm=arm,
                                   rig=rig, session=state["session"], out=out, run_dir=run_dir,
                                   dry_truth=dry_truth)
        except SignalRequest:
            print("[calib] interrupted while solving; nothing saved")
            return 130


# ── --verify ─────────────────────────────────────────────────────────────


@dataclass
class VerifyReport:
    errors_m: list
    rmse_m: float
    max_m: float
    passed: bool


def verify_known_points(known, observed, max_rmse_m: float = VERIFY_MAX_RMSE_M) -> VerifyReport:
    k = np.asarray(known, dtype=float).reshape(-1, 3)
    o = np.asarray(observed, dtype=float).reshape(-1, 3)
    if k.shape != o.shape or len(k) == 0:
        raise ValueError("known and observed points must be matching non-empty Nx3 lists")
    e = np.linalg.norm(o - k, axis=1)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return VerifyReport(errors_m=[float(v) for v in e], rmse_m=rmse, max_m=float(e.max()),
                        passed=bool(rmse < max_rmse_m))


def _parse_points(spec: str) -> list[list[float]]:
    from pathlib import Path

    import yaml

    p = Path(spec).expanduser()
    if p.exists():
        data = yaml.safe_load(p.read_text())
    else:
        data = [[float(v) for v in chunk.split(",")] for chunk in spec.split(";") if chunk.strip()]
    pts = [[float(v) for v in row] for row in data]
    if not pts or any(len(r) != 3 for r in pts):
        raise ValueError(f"--known-points: expected 'x,y,z;x,y,z' or a YAML list, got {spec!r}")
    return pts


def _print_report(rep: VerifyReport, known, observed) -> None:
    for k, o, e in zip(known, observed, rep.errors_m):
        print(f"  known {np.round(k, 4).tolist()}  measured {np.round(o, 4).tolist()}  "
              f"error {1000 * e:.1f} mm")
    print(f"[calib] verify RMSE {1000 * rep.rmse_m:.1f} mm (max {1000 * rep.max_m:.1f} mm) -> "
          f"{'PASS' if rep.passed else 'FAIL'} (< {1000 * VERIFY_MAX_RMSE_M:.0f} mm)")


def cmd_verify(args) -> int:
    from ..config import load_demo_config
    from .aruco import ArucoSession, camera_dist_coeffs
    from .dataset import read_hand_eye
    from .session import SessionConfig, wait_for_stable_marker

    try:
        record = read_hand_eye(args.verify)
    except (FileNotFoundError, ValueError) as e:
        _err(f"cannot read {args.verify}: {e}")
        return 2
    print(record.summary())
    if not record.acceptable:
        print("[calib] record is REJECTED by the quality gate; nothing to verify:")
        for r in record.rejection_reasons:
            print(f"  - {r}")
        return 1
    if not args.known_points:
        _err("--verify needs --known-points 'x,y,z;...' (marker centres measured in the base frame)")
        return 2
    try:
        known = _parse_points(args.known_points)
    except (ValueError, TypeError) as e:
        _err(str(e))
        return 2
    if record.mode == EYE_IN_HAND:
        # The wrist chain's independent check: where the fit says the table
        # marker was (FK x X x PnP, solved jointly) vs where it was measured.
        if len(known) != 1:
            _err("eye_in_hand --verify takes ONE known point: the measured centre of the table "
                 "marker used during calibration")
            return 2
        observed = [record.T_marker2base[:3, 3].tolist()]
        rep = verify_known_points(known, observed)
        _print_report(rep, known, observed)
        return 0 if rep.passed else 1

    camera = args.camera
    if not camera:
        _err("eye_to_hand --verify needs --camera PROFILE (the camera that was calibrated)")
        return 2
    cfg = load_demo_config(camera=camera, arm=args.arm or "mock", llm="mock")
    serial = str(cfg.camera.get("serial", "") or "")
    if serial and record.camera_serial and serial != record.camera_serial:
        _err(f"record is for camera serial {record.camera_serial}, profile {camera!r} pins "
             f"{serial}; refusing to verify a different unit")
        return 2
    if (cfg.camera.get("extrinsics") or {}).get("hand_eye_compensation_m") is not None:
        print("[calib] note: verifying the RAW record; the profile's hand_eye_compensation_m "
              "is a runtime correction on top of it")
    # A markerless record has no marker; verifying it still lays a marker at
    # measured points, described by the command line.
    marker = (_marker(args) if record.marker is None
              else MarkerSpec.from_json(record.marker.to_json()))
    scfg = SessionConfig(mode=EYE_TO_HAND, marker=marker,
                         settle_s=SETTLE_S if args.settle_time is None else args.settle_time,
                         marker_timeout_s=args.marker_timeout, stable_frames=args.stable_frames)
    from ..perception.camera_base import CameraError

    cam = open_camera(cfg.camera)
    try:
        D = camera_dist_coeffs(cam)
        aruco = ArucoSession(marker.dictionary)

        def grab():
            try:
                return cam.get_frame()
            except CameraError as e:
                print(f"[calib] camera: {e}")
                return None

        observed, used = [], []
        for p in known:
            _prompt(f"[calib] place the marker centre at {p} (base frame, m), face up; "
                    "ENTER when placed > ")
            det = wait_for_stable_marker(grab, aruco, scfg, D=D)
            if det is None:
                print(f"  known {p}: no stable marker -- counted as a failure")
                observed.append([np.inf] * 3)
            else:
                observed.append((record.T_cam2base @ det.T_marker2cam)[:3, 3].tolist())
            used.append(p)
    finally:
        cam.close()
    finite = [i for i, o in enumerate(observed) if np.all(np.isfinite(o))]
    if len(finite) < len(used):
        print(f"[calib] verify FAIL: marker not found at {len(used) - len(finite)} point(s)")
        if finite:
            _print_report(verify_known_points([used[i] for i in finite],
                                              [observed[i] for i in finite]),
                          [used[i] for i in finite], [observed[i] for i in finite])
        return 1
    rep = verify_known_points(used, observed)
    _print_report(rep, used, observed)
    return 0 if rep.passed else 1


# ── entry point ──────────────────────────────────────────────────────────


GRAVITY_COMP_NOT_PORTED = (
    "--gravity-comp (hand-guided capture) is not ported: it needs the motors in a "
    "torque-only free-drive mode that cascade's reBot driver does not expose through "
    "SafeArm, and the SafetyHarness cannot vet motion it does not command. Use --manual "
    "(ENTER per vetted preset) or --poses FILE with poses of your own. See "
    "docs/HANDEYE_CALIBRATION.md.")


def build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="cascade-calib-handeye",
        description="Hand-eye calibration (ArUco, joint SE(3) solve) for an eye-to-hand or "
                    "eye-in-hand camera profile, or markerless (--method markerless: the arm's "
                    "meshes fitted in depth, eye-to-hand RGB-D only). Every motion is vetted "
                    "by the arm's safety harness and executed through SafeArm. Procedure: "
                    "docs/HANDEYE_CALIBRATION.md")
    act = p.add_mutually_exclusive_group()
    act.add_argument("--list", action="store_true", help="list connected RealSense/Orbbec cameras")
    act.add_argument("--bind", metavar="PROFILE",
                     help="pin a connected camera's serial into configs/cameras/PROFILE.yaml")
    act.add_argument("--dry-run", action="store_true",
                     help="synthetic end-to-end run: MockArm + rendered marker, no hardware")
    act.add_argument("--verify", metavar="RECORD",
                     help="check a saved record against marker positions measured in the base frame")
    act.add_argument("--gravity-comp", action="store_true", help="NOT PORTED (see docs)")
    p.add_argument("--serial", help="--bind: the serial to pin (must be connected)")
    p.add_argument("--index", type=int, help="--bind: pick the Nth camera from --list")
    p.add_argument("--config-dir", help="configs directory (default: the repo's configs/)")
    p.add_argument("--camera", help="camera profile (its extrinsics.mode selects ETH/EIH)")
    p.add_argument("--arm", help="arm profile (its harness limits and home_q are used)")
    p.add_argument("--manual", action="store_true",
                   help="operator presses ENTER before every (vetted) preset pose")
    p.add_argument("--poses", help="YAML list of [x,y,z,roll,pitch,yaw] TCP poses (base frame)")
    p.add_argument("--method", choices=METHODS, default="marker",
                   help="marker (ArUco, both mountings; default) or markerless (eye-to-hand "
                        "RGB-D only: fit the arm's own meshes in depth, no marker)")
    p.add_argument("--depth-frames", type=int, default=5,
                   help="markerless: depth frames per pose (temporal median, default 5)")
    p.add_argument("--marker-size", type=float, default=0.10,
                   help="MEASURED printed marker side, metres (default 0.10)")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--dict", default="4x4_50", help="ArUco dictionary (default 4x4_50)")
    p.add_argument("--speed-frac", type=float, default=0.5,
                   help="fraction (0, 1] of the arm profile's max_joint_vel (default 0.5)")
    p.add_argument("--settle-time", type=float, default=None,
                   help=f"s after each move (default {SETTLE_S}; 0 for --dry-run)")
    p.add_argument("--marker-timeout", type=float, default=4.0)
    p.add_argument("--stable-frames", type=int, default=4)
    p.add_argument("--out", help="record path (default configs/calib/<camera>_<arm>.handeye.json)")
    p.add_argument("--dry-run-out", help="--dry-run record path (default: in the run dir)")
    p.add_argument("--run-dir", help="trace + capture directory (default runs/handeye/<ts>_<camera>)")
    p.add_argument("--yes", action="store_true", help="skip the start confirmation")
    p.add_argument("--known-points", help="--verify: 'x,y,z;x,y,z' metres (or a YAML file)")
    p.add_argument("--noise-px", type=float, default=0.5, help="--dry-run pixel noise")
    p.add_argument("--seed", type=int, default=0, help="--dry-run seed")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.gravity_comp:
        _err(GRAVITY_COMP_NOT_PORTED)
        return 2
    if args.depth_frames < 1:
        _err("--depth-frames must be >= 1")
        return 2
    if not _speed_ok(args.speed_frac):
        _err(f"--speed-frac {args.speed_frac}: must be in (0, 1] of the harness velocity cap "
             "(the cap is never raised)")
        return 2
    if args.list:
        return cmd_list(args)
    if args.bind:
        return cmd_bind(args)
    if args.verify:
        return cmd_verify(args)
    return cmd_calibrate(args, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
