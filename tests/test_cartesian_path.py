"""Cartesian straight-line motion, vetted by the harness at every tick.

Port of Seeed WRC e97998c (`src/wrc_demo/planning/cartesian_planner.py` +
`SafeArm.move_joints_path/move_cartesian`). WRC wrapped the reBotArm SDK's
SE(3)-geodesic sampler + CLIK tracker and tolerated up to 5 % unconverged IK
samples. cascade's port is self-contained (its own Kinematics, no SDK
import, so every arm and the mock stack can use it) and STRICTER: every
sample must solve, seeded from the previous one with no random restarts, and
no joint may jump more than a bound between samples (an IK branch flip is a
swing, not a straight line). Execution goes through SafeArm: the whole dense
path is preflighted with `vet_step` before the first command and every
streamed tick passes `approve()`.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_pin

from cascade.config import Cfg, load_demo_config
from cascade.types import SafetyViolation, SkillError


def _rs_kin():
    from cascade.control.kinematics import Kinematics

    a = load_demo_config(arm="rebot_rs").arm
    return Kinematics(a.model, a.ee_frame, n_controlled=6, joint_signs=a.joint_signs), a


def _line_deviation(points, p0, p1):
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = points - p0
    return np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)


# ── planner ─────────────────────────────────────────────────────────────────


def test_se3_interpolation_is_a_straight_line_and_a_rotation_geodesic():
    from cascade.planning.cartesian import se3_interpolate

    T0, T1 = np.eye(4), np.eye(4)
    T1[:3, 3] = [0.1, -0.2, 0.3]
    c, s = np.cos(1.0), np.sin(1.0)
    T1[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    mid = se3_interpolate(T0, T1, 0.5)
    np.testing.assert_allclose(mid[:3, 3], [0.05, -0.1, 0.15])
    c2, s2 = np.cos(0.5), np.sin(0.5)
    np.testing.assert_allclose(mid[:3, :3], [[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(se3_interpolate(T0, T1, 0.0), T0, atol=1e-12)
    np.testing.assert_allclose(se3_interpolate(T0, T1, 1.0), T1, atol=1e-12)


@needs_pin
def test_planned_path_tracks_the_line_and_ends_on_the_goal():
    from cascade.planning.cartesian import plan_cartesian_path

    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    T0 = kin.fk(q0)
    T1 = T0.copy()
    T1[:3, 3] += [-0.06, 0.04, -0.08]
    path = plan_cartesian_path(kin, q0, T1)
    assert len(path) >= 20  # 5 mm steps over ~11 cm
    tcp = np.array([kin.fk(q)[:3, 3] for q in path])
    assert _line_deviation(tcp, T0[:3, 3], T1[:3, 3]).max() < 5e-4
    np.testing.assert_allclose(kin.fk(path[-1]), T1, atol=2e-4)
    steps = np.abs(np.diff(np.vstack([q0, *path]), axis=0)).max()
    assert steps < 0.1


@needs_pin
def test_unreachable_goal_is_refused_not_approximated():
    from cascade.planning.cartesian import plan_cartesian_path

    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    T1 = kin.fk(q0)
    T1[:3, 3] += [0.6, 0.0, 0.0]  # far outside the arm's reach
    with pytest.raises(SkillError, match="cartesian"):
        plan_cartesian_path(kin, q0, T1)


class _FlipKin:
    """Every sample 'solves', but the third lands on another IK branch."""

    n = 2

    def __init__(self):
        self.calls = 0

    def fk(self, q):
        T = np.eye(4)
        T[0, 3] = float(np.sum(q)) * 0.01
        return T

    def ik(self, T, q_init, **kw):
        from types import SimpleNamespace

        self.calls += 1
        q = np.asarray(q_init, float) + 0.001
        if self.calls == 3:
            q = q + np.array([0.5, -0.5])  # same TCP x, different branch
        return SimpleNamespace(success=True, q=q, error=0.0)


def test_an_ik_branch_flip_between_samples_is_refused():
    from cascade.planning.cartesian import plan_cartesian_path

    kin = _FlipKin()
    T1 = np.eye(4)
    T1[0, 3] = 0.05
    with pytest.raises(SkillError, match="jump"):
        plan_cartesian_path(kin, np.zeros(2), T1)


# ── execution through SafeArm ───────────────────────────────────────────────


def _safe_arm(q0, motion_planner=None):
    from cascade.control.mock_arm import MockArm
    from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits

    kin, a = _rs_kin()
    cfg = load_demo_config(arm="rebot_rs")
    raw = MockArm(Cfg({"home_q": list(q0), "n_joints": 6}), kin)
    raw.connect()
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    return kin, raw, harness, SafeArm(raw, harness, motion_planner=motion_planner)


@needs_pin
def test_move_cartesian_streams_a_straight_vetted_line():
    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    kin, raw, harness, arm = _safe_arm(q0)
    T0 = kin.fk(q0)
    T1 = T0.copy()
    T1[:3, 3] += [-0.05, 0.0, -0.10]
    assert arm.move_cartesian(T1, duration_s=1.0) is True
    tcp = np.array([kin.fk(q)[:3, 3] for q in raw.commands])
    assert _line_deviation(tcp, T0[:3, 3], T1[:3, 3]).max() < 1e-3
    np.testing.assert_allclose(kin.fk(raw.get_state().q)[:3, 3], T1[:3, 3], atol=5e-4)
    # one continuous motion at 50 Hz, not a stop at every IK sample
    assert len(raw.commands) >= 50
    assert harness.violations == []


@needs_pin
def test_move_cartesian_preflight_refuses_before_any_command():
    """A reachable line that crosses a keep-out box mid-way is refused by the
    harness preflight -- not by IK, and with no partial motion along it."""
    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    kin, raw, harness, arm = _safe_arm(q0)
    T0 = kin.fk(q0)
    T1 = T0.copy()
    T1[:3, 3] += [-0.10, 0.0, -0.10]
    mid = (T0[:3, 3] + T1[:3, 3]) / 2
    harness.limits.keep_out.append((mid - 0.01, mid + 0.01))
    with pytest.raises(SafetyViolation, match="keep-out"):
        arm.move_cartesian(T1, duration_s=1.0)
    assert raw.commands == []
    # the endpoint alone is fine: it is the PATH that is refused
    harness.limits.keep_out.clear()
    assert arm.move_cartesian(T1, duration_s=1.0) is True


@needs_pin
def test_move_cartesian_respects_the_velocity_cap():
    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    kin, raw, harness, arm = _safe_arm(q0)
    T1 = kin.fk(q0)
    T1[:3, 3] += [-0.10, 0.10, -0.10]
    assert arm.move_cartesian(T1, duration_s=0.2) is True  # far too fast: stretched
    q = np.vstack([q0, *raw.commands])
    # MockArm records only commands; recover dt from the stretched duration
    # the executor reports
    dt = arm.last_path_dt
    vel = np.abs(np.diff(q, axis=0)).max(axis=1) / dt
    assert vel.max() <= harness.limits.max_joint_vel + 1e-9


@needs_pin
def test_move_cartesian_refuses_arms_with_a_motion_planner():
    """cuMotion arms own their curves; the joint-path streamer must not
    bypass that planner's contract."""
    class _Planner:
        def close(self):
            pass

    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    kin, raw, harness, arm = _safe_arm(q0, motion_planner=_Planner())
    T1 = kin.fk(q0)
    T1[2, 3] -= 0.02
    with pytest.raises(SafetyViolation, match="motion planner"):
        arm.move_cartesian(T1, duration_s=1.0)
    assert raw.commands == []


@needs_pin
def test_move_cartesian_refuses_when_estopped():
    kin, a = _rs_kin()
    q0 = np.asarray(a.home_q, dtype=float)
    kin, raw, harness, arm = _safe_arm(q0)
    harness.estop("test")
    T1 = kin.fk(q0)
    T1[2, 3] -= 0.02
    with pytest.raises(SafetyViolation, match="e-stop"):
        arm.move_cartesian(T1, duration_s=1.0)
    assert raw.commands == []


def test_isaac_refuses_the_wall_clock_path_streamer():
    """Isaac streams in simulator time; inheriting ArmBase.stream_path would
    bypass that clock contract, so it refuses before any command."""
    from cascade.control.isaac_arm import IsaacArm

    arm = object.__new__(IsaacArm)
    with pytest.raises(SafetyViolation, match="Isaac"):
        arm.stream_path([np.zeros(6)], 1.0)


# ── opt-in consumer: move_relative along a line ─────────────────────────────


def _move_relative(demo_cfg, tmp_path, cartesian):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    if cartesian is not None:
        demo_cfg._data["arm"]["cartesian_relative_moves"] = cartesian
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        used = []
        original = runtime.arm.move_cartesian
        runtime.arm.move_cartesian = lambda *a, **k: used.append(a) or original(*a, **k)
        kin = runtime.arm.harness.kin
        start = kin.fk(runtime.arm.get_state().q)[:3, 3]
        n0 = len(arm.commands)
        result = runtime.execute("move_relative", {"direction": "down", "distance_m": 0.05})
        tcp = np.array([kin.fk(q)[:3, 3] for q in arm.commands[n0:]])
        return result, used, start, tcp
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_move_relative_stays_joint_space_by_default(demo_cfg, tmp_path):
    result, used, _, _ = _move_relative(demo_cfg, tmp_path, None)
    assert result["ok"], result
    assert used == []


@needs_pin
def test_move_relative_follows_a_line_when_the_profile_opts_in(demo_cfg, tmp_path):
    result, used, start, tcp = _move_relative(demo_cfg, tmp_path, True)
    assert result["ok"], result
    assert len(used) == 1
    goal = start + np.array([0.0, 0.0, -0.05])
    assert _line_deviation(tcp, start, goal).max() < 1e-3
    np.testing.assert_allclose(tcp[-1], goal, atol=5e-4)
