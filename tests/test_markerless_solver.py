"""Markerless eye-to-hand calibration: robust point-to-plane ICP of the
arm's own surface (at FK of the measured joints) against depth, jointly over
every pose (calibration/markerless.py).

Ground truth comes from the synthetic D455-like camera (test_synthetic_depth
pins its geometry). Every claim the gate makes is tested in both directions:
a good sweep passes and lands within millimetres; one pose, a barely visible
arm and a sweep that only translates the TCP along a line are REFUSED even
though their residuals are as small as a good fit's -- the ICP twin of the
marker gate's rotation-spread rule.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import JOINT_SIGNS, URDF, needs_pin

from cascade.calibration.frames import pose_error, se3_inv, so3_exp

pytestmark = needs_pin


def _T(t, w):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(w, dtype=float))
    T[:3, 3] = t
    return T


DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
#: The dry run's eye-to-hand truth (cli._dry_run_truth): ~0.95 m above the
#: base, looking down, tilted ~4.5 deg.
T_TRUE = np.eye(4)
T_TRUE[:3, :3] = DOWN @ so3_exp([0.05, -0.06, 0.0])
T_TRUE[:3, 3] = [0.30, -0.03, 0.95]


def _err(T_est, T_ref=T_TRUE):
    e = pose_error(se3_inv(T_ref) @ T_est)
    return 1000 * float(np.linalg.norm(e[:3])), float(np.degrees(np.linalg.norm(e[3:])))


def _perturb(T, trans_m, rot_deg, axis_t, axis_r):
    """Camera-frame perturbation: the camera moved/turned on its mount."""
    at = np.asarray(axis_t, float) / np.linalg.norm(axis_t)
    ar = np.asarray(axis_r, float) / np.linalg.norm(axis_r)
    return T @ _T(trans_m * at, np.radians(rot_deg) * ar)


@pytest.fixture(scope="module")
def kin():
    from cascade.control.kinematics import Kinematics

    return Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)


@pytest.fixture(scope="module")
def surface(kin):
    from cascade.calibration.robot_surface import RobotSurface

    return RobotSurface.from_kinematics(kin, URDF)


@pytest.fixture(scope="module")
def preset_qs(kin):
    from cascade.calibration.session import load_poses
    from cascade.types import pose_to_transform

    home = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    qs = []
    for p in load_poses("rebot_rs", "eye_to_hand"):
        sol = kin.ik(pose_to_transform(p), home)
        if sol.success:
            qs.append(np.asarray(sol.q)[:6])
    assert len(qs) >= 30
    return qs


def _observe(kin, qs, T=T_TRUE, *, seed=0, mask=None, **noise):
    from cascade.calibration.markerless import DepthSample
    from cascade.calibration.synthetic_depth import SyntheticDepthCamera

    state = {"q": qs[0]}
    cam = SyntheticDepthCamera(kin, URDF, T, lambda: state["q"], seed=seed, **noise)
    out = []
    for i, q in enumerate(qs):
        state["q"] = q
        f = cam.get_frame()
        depth = f.depth_m if mask is None else np.where(mask, f.depth_m, 0.0).astype(np.float32)
        out.append(DepthSample(q=tuple(float(v) for v in q), T_gripper2base=kin.fk(q),
                               depth_m=depth, K=f.K, label=f"pose_{i:02d}"))
    return out


@pytest.fixture(scope="module")
def sweep(kin, preset_qs):
    """Every 3rd reBot preset: 13 poses, default D455-like noise."""
    return _observe(kin, preset_qs[::3])


def test_recovers_the_extrinsic_from_a_rough_profile_guess(surface, sweep):
    from cascade.calibration.markerless import assess_markerless, solve_markerless

    init = _perturb(T_TRUE, 0.06, 8.0, [1, -1, 0.5], [0.3, 1, 0.2])
    fit = solve_markerless(sweep, surface, init)
    mm, deg = _err(fit.T_cam2base)
    assert mm < 3.0 and deg < 0.3, (mm, deg)
    assert fit.acceptable, fit.rejection_reasons
    assert assess_markerless(fit.metrics) == []
    m = fit.metrics
    assert m["n_poses"] == len(sweep) and m["n_poses_used"] == len(sweep)
    assert m["inlier_fraction"] > 0.6
    assert m["rmse_m"] < 0.01
    assert m["degeneracy_min_eig"] > 0 and np.isfinite(m["condition_number"])


@pytest.mark.parametrize("trans_m,rot_deg,axis_t,axis_r", [
    (0.10, 15.0, [1, 0, 0], [0, 1, 0]),
    (0.10, 15.0, [0, 1, 0], [1, 0, 0]),
    (0.10, 15.0, [0, 0, 1], [0, 0, 1]),
    (0.12, 18.0, [1, 1, 0], [1, -1, 0.5]),
])
def test_convergence_basin_covers_10cm_15deg_from_the_profile_guess(surface, sweep, trans_m,
                                                                   rot_deg, axis_t, axis_r):
    from cascade.calibration.markerless import solve_markerless

    init = _perturb(T_TRUE, trans_m, rot_deg, axis_t, axis_r)
    fit = solve_markerless(sweep, surface, init)
    mm, deg = _err(fit.T_cam2base)
    assert fit.acceptable, fit.rejection_reasons
    assert mm < 3.0 and deg < 0.3, (mm, deg)


def test_an_init_far_outside_the_basin_is_never_accepted_wrong(surface, sweep):
    """Whatever the solver lands on from a hopeless guess, the gate must not
    pass a wrong transform."""
    from cascade.calibration.markerless import solve_markerless

    init = _perturb(T_TRUE, 0.45, 70.0, [1, 0.5, 0], [0.2, 1, 0.3])
    fit = solve_markerless(sweep, surface, init, n_starts=0)
    mm, deg = _err(fit.T_cam2base)
    if fit.acceptable:
        assert mm < 5.0 and deg < 0.5, (mm, deg)
    else:
        assert fit.rejection_reasons


def test_heavier_correlated_noise_still_lands_within_millimetres(kin, surface, preset_qs):
    from cascade.calibration.markerless import solve_markerless

    noisy = _observe(kin, preset_qs[1::3], seed=7, noise_frac=0.01, correlated_frac=0.004,
                     dropout=0.05)
    fit = solve_markerless(noisy, surface, _perturb(T_TRUE, 0.05, 6.0, [0, 1, 1], [1, 0, 1]))
    mm, deg = _err(fit.T_cam2base)
    assert fit.acceptable, fit.rejection_reasons
    assert mm < 5.0 and deg < 0.5, (mm, deg)


# ── degeneracy: small residual, unconstrained transform -> REJECTED ──────


def test_one_pose_is_refused_even_with_a_small_residual(surface, sweep):
    from cascade.calibration.markerless import solve_markerless

    fit = solve_markerless(sweep[:1], surface, T_TRUE)
    assert fit.metrics["rmse_m"] < 0.01             # it "fits"
    assert not fit.acceptable
    assert any("poses" in r for r in fit.rejection_reasons), fit.rejection_reasons


def test_a_barely_visible_arm_is_refused(kin, surface, preset_qs):
    """Only a window around the base has depth (the rest of the arm out of
    view / masked). What IS there fits with a tiny residual; but most of
    the arm the model says the camera should see is unexplained."""
    from cascade.calibration.markerless import solve_markerless
    from cascade.calibration.synthetic_depth import default_K

    K = default_K((640, 360))
    top = se3_inv(T_TRUE) @ np.array([0.0, 0.0, 0.075, 1.0])
    u, v = (K @ (top[:3] / top[2]))[:2].astype(int)
    mask = np.zeros((360, 640), dtype=bool)
    mask[v - 30: v + 30, u - 30: u + 30] = True
    obs = _observe(kin, preset_qs[::3], mask=mask)
    fit = solve_markerless(obs, surface, T_TRUE, n_starts=0)
    assert fit.metrics["rmse_m"] < 0.01             # it "fits"
    assert not fit.acceptable
    assert any("inlier fraction" in r for r in fit.rejection_reasons), fit.rejection_reasons


class _Plate:
    """A flat 15 cm square plate carried by the 'arm' (q = its pose):
    geometry that pins only 3 of 6 DOF when it never tilts."""

    points_per_m2 = 20000.0

    def __init__(self):
        g = np.linspace(-0.075, 0.075, 22)
        xx, yy = np.meshgrid(g, g)
        self.local = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1)

    def points_at(self, q):
        from cascade.calibration.robot_surface import SurfaceCloud
        G = _T(q[:3], q[3:])
        n = len(self.local)
        return SurfaceCloud(points=self.local @ G[:3, :3].T + G[:3, 3],
                            normals=np.tile(G[:3, 2], (n, 1)),
                            link_ids=np.zeros(n, dtype=np.int32))


def _plate_depth(q, T, K, size=(640, 360), half=0.075):
    G = _T(q[:3], q[3:])
    w, h = size
    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    Gc = se3_inv(T) @ G                 # plate in the camera frame
    n, p0 = Gc[:3, 2], Gc[:3, 3]
    s = (p0 @ n) / (rays @ n)
    hit = rays * s[..., None]
    local = (hit - p0) @ Gc[:3, :3]
    inside = (np.abs(local[..., 0]) < half) & (np.abs(local[..., 1]) < half) & (s > 0)
    rng = np.random.default_rng(int(1000 * abs(q[0] + q[1])))
    z = s * (1 + 0.003 * rng.normal(size=s.shape))
    return np.where(inside, z, 0.0).astype(np.float32)


@pytest.mark.parametrize("tilted", [False, True])
def test_degeneracy_check_refuses_geometry_that_pins_only_some_dof(tilted):
    """The normal-matrix check, two-sided, on controlled geometry: a plate
    that only translates is a plane in every view (x, y and yaw are free;
    the residual is ~noise); the same plate tilted about two axes pins all
    six DOF."""
    from cascade.calibration.markerless import DepthSample, solve_markerless
    from cascade.calibration.synthetic_depth import default_K

    K = default_K((640, 360))
    rng = np.random.default_rng(5)
    samples = []
    for i in range(10):
        t = [0.30 + rng.uniform(-0.08, 0.08), rng.uniform(-0.1, 0.1), 0.15 + rng.uniform(0, 0.15)]
        w = rng.uniform(-0.45, 0.45, 3) * [1, 1, 0] if tilted else np.zeros(3)
        q = np.concatenate([t, w])
        samples.append(DepthSample(q=tuple(q), T_gripper2base=_T(t, w),
                                   depth_m=_plate_depth(q, T_TRUE, K), K=K, label=f"p{i}"))
    fit = solve_markerless(samples, _Plate(), T_TRUE, n_starts=0)
    reasons = " ".join(fit.rejection_reasons)
    assert fit.metrics["rmse_m"] < 0.005
    if tilted:
        assert "degenerate" not in reasons, reasons
        assert fit.metrics["degeneracy_min_eig"] > 0.01
    else:
        assert "degenerate" in reasons, reasons
        assert fit.metrics["degeneracy_min_eig"] < 0.01


def test_a_sweep_that_only_translates_the_tcp_is_refused(kin, surface):
    """Constant tool orientation, TCP moving along one line: the marker
    gate's classic degenerate sweep. The fit's residual is fine; the pose
    diversity gate refuses it."""
    from cascade.calibration.markerless import solve_markerless
    from cascade.types import pose_to_transform

    home = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    qs = []
    for y in np.linspace(-0.10, 0.05, 8):
        sol = kin.ik(pose_to_transform([0.30, y, 0.28, 0.0, 0.0, 0.0]), home)
        assert sol.success
        qs.append(np.asarray(sol.q)[:6])
    obs = _observe(kin, qs)
    fit = solve_markerless(obs, surface, T_TRUE, n_starts=0)
    assert fit.metrics["rmse_m"] < 0.01
    assert not fit.acceptable
    assert any("diversity" in r or "spread" in r for r in fit.rejection_reasons), \
        fit.rejection_reasons


def test_gate_refuses_missing_or_non_finite_metrics():
    from cascade.calibration.markerless import assess_markerless

    assert assess_markerless({})
    ok = {"n_poses": 12, "n_poses_used": 12, "n_inliers": 20000, "inlier_fraction": 0.8,
          "rmse_m": 0.004, "pose_offset_max_m": 0.001, "pose_offset_max_deg": 0.1,
          "degeneracy_min_eig": 0.05, "condition_number": 20.0, "tcp_spread_m": 0.05,
          "rotation_spread_deg": 15.0}
    assert assess_markerless(ok) == []
    for key in ok:
        bad = dict(ok)
        bad[key] = float("nan")
        assert assess_markerless(bad), key


def test_eye_in_hand_is_refused_with_a_reason(surface, sweep):
    from cascade.calibration.markerless import solve_markerless

    with pytest.raises(ValueError, match="eye_to_hand"):
        solve_markerless(sweep, surface, T_TRUE, mode="eye_in_hand")


# ── single-pose offset check (the drift monitor's measurement) ───────────


def test_offset_check_reads_zero_when_calibrated_and_the_bump_when_bumped(kin, surface,
                                                                         preset_qs):
    from cascade.calibration.markerless import measure_offset

    q = preset_qs[4]
    still = _observe(kin, [q], seed=11)[0]
    r = measure_offset(still, surface, T_TRUE)
    assert r.conclusive, r
    # Sensor noise alone: the arm surface "moves" ~ a millimetre.
    assert r.offset_m < 0.004 and r.offset_deg < 1.5, r
    # A 2 deg knock on the mount moves the arm ~2.5 cm in the image's 3D.
    bumped_T = _perturb(T_TRUE, 0.0, 2.0, [1, 0, 0], [1, 0.3, 0])
    bumped = _observe(kin, [q], T=bumped_T, seed=12)[0]
    r = measure_offset(bumped, surface, T_TRUE)
    assert r.conclusive, r
    assert r.offset_m > 0.015, r
    assert r.offset_deg == pytest.approx(2.0, abs=0.6), r
