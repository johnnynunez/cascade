#!/usr/bin/env python3
"""Fit a scene camera's extrinsic by touching known points with the gripper.

`configs/cameras/*.yaml` carry an `extrinsics:` block holding a 4x4
`T_cam2base`, and a camera without one never fuses 3D beliefs (see
apps/demo.py). This script produces that matrix.

METHOD. The arm is its own calibration target. Drive the TCP to a set of
poses: FK gives each point exactly in the base frame, and the depth camera
gives the same point in the camera frame. Three or more non-collinear
correspondences determine the rigid transform between the frames (Kabsch, see
cascade/perception/calibration.py). This is the same trick RPent's SO-101
calibrator uses (github.com/RLinf/RPent PR #29), which detects the jaw by
toggling the gripper between frames.

    # offline: verify the math end to end, no hardware, no camera
    python scripts/calibrate_camera.py --self-test

    # on the rig: drive the arm, click the gripper in each frame
    python scripts/calibrate_camera.py --arm so101 --camera l515 \
        --out configs/calib/l515_so101.json

SAFETY. `--self-test` moves nothing. The live path DOES move the arm through a
grid of poses; clear the workspace first, keep the e-stop within reach, and
expect it to refuse poses the safety harness rejects (that is correct
behaviour, not a bug -- it samples only what the arm may legally reach).

WHY THE RESULT CAN BE REJECTED. A rigid fit always returns a matrix, even from
nonsense correspondences, so this reports RMSE and refuses to save a fit that
is too loose or whose sample points are too close to coplanar. A wrong
extrinsic is worse than none: with none, 3D fusion is simply off; with a wrong
one, every back-projected grasp target is confidently a few centimetres out
and the failure looks like bad grasping instead of bad calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from cascade.perception.calibration import (  # noqa: E402
    MAX_ACCEPTABLE_RMSE_M,
    fit_extrinsic,
    save_extrinsic,
)


def _self_test() -> int:
    """Check the fit against a known transform, including its refusals."""
    from cascade.types import pose_to_transform

    rng = np.random.default_rng(0)
    T_true = pose_to_transform([0.4, -0.1, 0.6, 0.2, -0.35, 1.1])
    cam = rng.uniform([-0.2, -0.2, 0.3], [0.2, 0.2, 0.7], size=(12, 3))
    base = cam @ T_true[:3, :3].T + T_true[:3, 3]

    exact = fit_extrinsic(cam, base)
    assert exact.acceptable, exact.summary()
    assert np.allclose(exact.T_cam2base, T_true, atol=1e-9), "exact fit is wrong"
    assert np.isclose(np.linalg.det(exact.T_cam2base[:3, :3]), 1.0), "not a rotation"
    print(f"  exact          {exact.summary()}")

    noisy = fit_extrinsic(cam + rng.normal(0, 0.002, cam.shape), base)
    assert noisy.acceptable, noisy.summary()
    print(f"  2 mm noise     {noisy.summary()}")

    garbage = fit_extrinsic(cam, rng.uniform(-0.5, 0.5, (12, 3)))
    assert not garbage.acceptable, "garbage correspondences must be rejected"
    print(f"  garbage        {garbage.summary()}")

    flat = cam.copy()
    flat[:, 2] = 0.5
    coplanar = fit_extrinsic(flat, flat @ T_true[:3, :3].T + T_true[:3, 3])
    assert not coplanar.acceptable, "coplanar samples must be rejected"
    # The whole point: a perfect residual on unconstrained geometry.
    assert coplanar.rmse_m < 1e-9, "expected a deceptively perfect residual"
    print(f"  coplanar       {coplanar.summary()}  <- 0 mm error, still rejected")

    print("\nself-test OK")
    return 0


def _targets(cfg=None) -> list[np.ndarray]:
    """Base-frame TCP positions to sample, as a non-coplanar annulus.

    Three heights is not a style choice: sampling at one z leaves the fit
    unconstrained out of plane (see calibration.MIN_SPREAD_M), which the
    self-test demonstrates producing a 0 mm residual on a wrong transform.

    The default grid is shaped for the SO-101, MEASURED against its IK rather
    than assumed. Two facts drive it, and both were found by measuring, not by
    reading the profile:

    * a plain cartesian box solves 2/27 poses, because this arm's wrist axes
      are parallel and a vertical approach pins the tool yaw to the base pan
      angle -- sampling an annulus with a RADIAL yaw solves 46/80;
    * the reachable top-down band is only ~0.06 m tall (z = 0.02..0.08 at
      these radii), so the heights are spread across all of it. Sampling one
      height leaves the fit unconstrained out of plane, which the self-test
      shows producing a deceptive 0 mm residual.

    Other arms should widen both ranges; the fit itself is arm-agnostic.
    """
    radii = (0.16, 0.20, 0.24, 0.28)
    heights = (0.02, 0.04, 0.06, 0.08)
    azimuths = np.linspace(-0.7, 0.7, 5)
    return [
        np.array([r * np.cos(az), r * np.sin(az), z])
        for r in radii
        for z in heights
        for az in azimuths
    ]


def _collect(runtime, targets, settle_s: float = 0.4):
    """Drive the TCP to each target and observe it; return correspondences.

    Returns (points_cam, points_base). Targets the harness rejects or IK
    cannot reach are skipped with a note -- an unreachable pose is not a
    calibration failure, and forcing one would be a safety violation.
    """
    import time

    from cascade.types import SafetyViolation, SkillError, make_transform

    home_q = np.asarray(runtime.cfg.arm.home_q, dtype=float)
    R_home = runtime.kin.fk(home_q)[:3, :3]

    def _solve(target):
        """IK for a top-down pose at `target`, trying both yaw conventions.

        Arms whose wrist axes are parallel (the SO-101) cannot hold a fixed
        tool yaw off the radial direction, so an annulus sampled with home's
        orientation solves almost nowhere: MEASURED 13/80 poses, and every
        one of them at y = 0 -- a perfectly planar cloud that the spread gate
        then rejects, correctly but uselessly. Rotating the tool yaw to match
        the target's azimuth recovers the off-axis poses.

        Trying both rather than reading a config flag is deliberate. The
        obvious flag (`yaw_from_base`) is not merely absent from some
        profiles -- it was REMOVED from so101.yaml on 2026-08-29 because
        measurement refuted the claim behind it. Branching on a key that may
        not exist silently selected the fixed orientation and produced
        exactly the degenerate cloud above. Whichever branch solves is the
        one this arm can actually hold, which needs no config at all.
        """
        az = float(np.arctan2(target[1], target[0]))
        c, s = np.cos(az), np.sin(az)
        R_radial = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R_home
        for R in (R_radial, R_home):
            sol = runtime.kin.ik(make_transform(R, target), home_q)
            if sol.success:
                return sol
        return None

    pts_cam, pts_base = [], []
    for i, target in enumerate(targets, 1):
        # Seeded from home_q, not the live pose: elbow-down IK branches dip
        # links below the table (see CLAUDE.md's grasp-pipeline note).
        sol = _solve(target)
        if sol is None:
            print(f"  [{i:2d}/{len(targets)}] unreachable {np.round(target, 3)}; skipping")
            continue
        try:
            if not runtime.arm.move_joints(sol.q, duration_s=2.0):
                print(f"  [{i:2d}/{len(targets)}] did not settle; skipping")
                continue
        except (SafetyViolation, SkillError) as e:
            print(f"  [{i:2d}/{len(targets)}] refused: {e}")
            continue
        time.sleep(settle_s)   # let the frame catch up with the motion

        observed = _observe_tcp(runtime)
        if observed is None:
            print(f"  [{i:2d}/{len(targets)}] gripper not seen in frame; skipping")
            continue
        # FK of the pose actually REACHED, not the one commanded: the arm
        # settles within a tolerance, and calibrating against the commanded
        # value would fold that error straight into the extrinsic.
        reached = runtime.kin.fk(runtime.arm.get_state().q)[:3, 3]
        pts_cam.append(observed)
        pts_base.append(reached)
        print(f"  [{i:2d}/{len(targets)}] cam {np.round(observed, 3)} "
              f"<- base {np.round(reached, 3)}")
    return np.asarray(pts_cam), np.asarray(pts_base)


def _observe_tcp(runtime):
    """Camera-frame position of the gripper in the current frame, or None.

    Left as an explicit hook rather than a guess: which detector finds the
    jaw depends on the rig (RPent toggles the gripper and differences the two
    frames; a marker on the jaw is easier and more robust). Implement it for
    your setup -- everything around it is rig-independent.
    """
    raise NotImplementedError(
        "gripper detection is rig-specific; see this function's docstring. "
        "Run --self-test to exercise the fit without hardware."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--self-test", action="store_true",
                   help="verify the fit offline; moves nothing")
    p.add_argument("--arm", default="so101", help="arm profile")
    p.add_argument("--camera", default="l515", help="camera profile")
    p.add_argument("--out", default=None, help="where to write the JSON fit")
    p.add_argument("--max-rmse", type=float, default=MAX_ACCEPTABLE_RMSE_M,
                   help="reject fits looser than this (metres)")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    import tempfile

    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    print(f"[calib] arm={args.arm} camera={args.camera}")
    print("[calib] THIS MOVES THE ARM. Clear the workspace; e-stop within reach.")
    cfg = load_demo_config(arm=args.arm, camera=args.camera, llm="mock")
    run_dir = Path(tempfile.mkdtemp(prefix="calib_"))
    runtime, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        pts_cam, pts_base = _collect(runtime, _targets())
    finally:
        shutdown_runtime(runtime, arm)

    if len(pts_cam) < 4:
        print(f"\n[calib] only {len(pts_cam)} usable points; need >= 4. "
              "Check the camera sees the gripper across the whole grid.")
        return 1

    result = fit_extrinsic(pts_cam, pts_base)
    print(f"\n[calib] {result.summary()}")
    if not result.acceptable:
        print("[calib] REFUSING to save: a wrong extrinsic fails silently, "
              "sending every back-projected target a few cm off.")
        return 1

    out = Path(args.out) if args.out else REPO / "configs" / "calib" / f"{args.camera}_{args.arm}.json"
    path = save_extrinsic(out, result, camera=args.camera,
                          note=f"arm={args.arm}, scripts/calibrate_camera.py")
    print(f"[calib] saved {path}")
    print("[calib] wire it in: set `extrinsics.T` in the camera profile to "
          "this file's T_cam2base (or point the profile at this path).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
