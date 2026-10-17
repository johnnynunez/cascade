"""Offline synthetic obstacles, not a reconstruction of the live cup scene."""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from test_home_routes import CartesianKin, case

from cascade.safety.trajectory import plan_route, vet_route, vet_segment
from cascade.types import SkillError


class LiftKin(CartesianKin):
    """Exact translational model; real harness/profile sampling stays in use."""

    def __init__(self):
        self.requests = []

    def ik(self, target, q_init, **kwargs):
        self.requests.append((target.copy(), q_init.copy(), kwargs))
        return SimpleNamespace(success=True, q=target[:3, 3].copy())


def lift_case(*, wall_top=.31):
    safe, raw, harness = case()
    kin = LiftKin()
    harness.kin = kin
    raw.kin = kin
    # A finite low wall: straight and joint-corner routes hit it, but an
    # upward endpoint followed by a descending joint leg can clear it.
    harness.limits.keep_out = [(np.array([.34, -.1, .27]),
                                np.array([.36, .1, wall_top]))]
    return safe, raw, harness, np.array([.3, 0., .3]), np.array([.4, 0., .3])


def test_lift_route_clears_synthetic_wall_after_all_old_candidates_fail():
    _, raw, harness, start, goal = lift_case()
    for old_route in ([goal], [goal, goal], [start, goal]):
        assert "keep-out" in vet_route(harness, start, old_route, 3.)
    assert vet_segment(harness, start, start + [0., 0., .03], 3.) is None

    route = plan_route(harness, start, goal)

    np.testing.assert_allclose(route, [start + [0., 0., .03], goal])
    assert vet_route(harness, start, route, 3.) is None
    assert raw.commands == []
    target, seed, _ = harness.kin.requests[0]
    np.testing.assert_array_equal(target[:3, :3], np.eye(3))
    np.testing.assert_array_equal(target[:2, 3], start[:2])
    np.testing.assert_array_equal(seed, start)


def test_second_lift_is_tried_only_after_the_smaller_route_is_unsafe():
    _, raw, harness, start, goal = lift_case(wall_top=.32)
    assert "keep-out" in vet_route(harness, start, [start + [0., 0., .03], goal], 3.)

    route = plan_route(harness, start, goal)

    np.testing.assert_allclose(route, [start + [0., 0., .06], goal])
    assert vet_route(harness, start, route, 3.) is None
    assert [t[2, 3] for t, _, _ in harness.kin.requests] == pytest.approx([.33, .36])
    assert raw.commands == []


@pytest.mark.parametrize("slow", ["fk", "ik-failure", "ik-success"])
def test_model_work_cannot_start_more_work_or_authorize_after_deadline(monkeypatch, slow):
    from cascade.safety import trajectory

    clock, ik_calls, vetted = [0.], [], []

    def fk(q):
        if slow == "fk":
            clock[0] = 3.
        return CartesianKin().fk(q)

    def ik(target, **kwargs):
        ik_calls.append(kwargs)
        clock[0] = 3.
        return SimpleNamespace(success=slow == "ik-success", q=target[:3, 3])

    def vet(h, start, route, *args, **kwargs):
        vetted.append(route)
        return None if len(vetted) > 3 else "blocked"

    harness = SimpleNamespace(kin=SimpleNamespace(fk=fk, ik=ik))
    monkeypatch.setattr(trajectory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(trajectory, "vet_route", vet)
    with pytest.raises(SkillError, match="time budget"):
        plan_route(harness, [.3, 0., .3], [.4, 0., .3])
    assert len(ik_calls) == (0 if slow == "fk" else 1)
    assert len(vetted) == 3


def test_safe_lift_does_not_authorize_an_unsafe_second_leg():
    safe, raw, harness, start, goal = lift_case(wall_top=.40)
    for height in (.03, .06):
        lift = start + [0., 0., height]
        assert vet_segment(harness, start, lift, 3.) is None
        assert "keep-out" in vet_segment(harness, lift, goal, 3.)
    with pytest.raises(SkillError, match="no safe joint route.*keep-out"):
        safe.move_planned(goal)
    assert len(harness.kin.requests) == 2
    assert raw.commands == []
    assert harness.violations == []
    assert harness._grasp_exempt is None
    assert harness.limits.min_clearance_m == .03


def test_upward_endpoint_is_not_a_safe_intermediate_joint_path(monkeypatch):
    safe, raw, harness, start, goal = lift_case()
    fk = harness.kin.fk

    def curved(q):
        pose = fk(q)
        if .3 < q[2] < .33:
            pose[2, 3] -= .05 * np.sin(np.pi * (q[2] - .3) / .03)
        return pose

    monkeypatch.setattr(harness.kin, "fk", curved)
    harness.limits.keep_out.append((np.array([.29, -.1, .26]), np.array([.31, .1, .29])))
    for height in (.03, .06):
        lift = start + [0., 0., height]
        assert harness.vet_pose(lift) is None
        assert "keep-out" in vet_segment(harness, start, lift, 3.)
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned(goal)
    assert raw.commands == []


def test_ik_failure_is_not_a_route_and_does_not_retry_or_mutate_start(monkeypatch):
    safe, raw, harness, start, goal = lift_case()
    requests = []

    def fail(target, q_init, **kwargs):
        requests.append((target.copy(), q_init.copy(), kwargs))
        q_init[:] = 100.  # Solver scratch must not corrupt the measured start.
        return SimpleNamespace(success=False, q=target[:3, 3].copy())

    monkeypatch.setattr(harness.kin, "ik", fail)
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned(goal)
    assert len(requests) == 2
    for target, seed, kwargs in requests:
        np.testing.assert_array_equal(seed, start)
        np.testing.assert_array_equal(target[:3, :3], np.eye(3))
        assert kwargs == {"max_iter": 80, "retries": 0}
    assert raw.commands == []
    np.testing.assert_array_equal(raw.get_state().q, start)


@pytest.mark.parametrize("invalid", [[np.nan, 0., .33], [.3, .33], [.3, 0., 3.]])
def test_invalid_or_out_of_limits_ik_result_cannot_authorize_motion(monkeypatch, invalid):
    safe, raw, harness, _, goal = lift_case()
    monkeypatch.setattr(harness.kin, "ik", lambda *a, **kw: SimpleNamespace(
        success=True, q=np.asarray(invalid)))
    with pytest.raises(SkillError):
        safe.move_planned(goal)
    assert raw.commands == []


@pytest.mark.parametrize("kin", [None, SimpleNamespace(), CartesianKin()])
def test_absent_fk_or_ik_preserves_the_original_refusal(monkeypatch, kin):
    from cascade.safety import trajectory

    harness = SimpleNamespace(kin=kin)
    monkeypatch.setattr(trajectory, "vet_route", lambda *a, **kw: "original obstruction")
    with pytest.raises(SkillError, match="no safe joint route.*original obstruction"):
        plan_route(harness, [.3, 0., .3], [.4, 0., .3])


@pytest.mark.parametrize("changed, cap, lifts", [(6, 15, 2), (7, 15, 0), (20, 15, 0),
                                                 (1, 4, 1)])
def test_old_order_and_candidate_cap_include_failed_lift_attempts(monkeypatch, changed, cap, lifts):
    from cascade.safety import trajectory

    start, goal = np.zeros(changed), np.ones(changed)
    seen, requests, fk_calls = [], [], []

    def fk(q):
        fk_calls.append(True)
        assert len(seen) == 1 + 2 * changed
        return np.eye(4)

    def ik(target, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(success=False, q=goal)

    def vet(h, s, route, *args, **kwargs):
        seen.append(route)
        return "blocked"

    monkeypatch.setattr(trajectory, "MAX_ROUTE_CANDIDATES", cap)
    monkeypatch.setattr(trajectory, "vet_route", vet)
    with pytest.raises(SkillError, match="no safe joint route"):
        plan_route(SimpleNamespace(kin=SimpleNamespace(fk=fk, ik=ik)), start, goal)
    expected = [[goal]]
    for joint in range(changed):
        first, last = start.copy(), goal.copy()
        first[joint], last[joint] = goal[joint], start[joint]
        expected.extend([[first, goal], [last, goal]])
    assert len(seen) == min(len(expected), cap)
    for actual, old in zip(seen, expected):
        np.testing.assert_array_equal(actual, old)
    assert len(requests) == lifts
    assert len(fk_calls) == (1 if lifts else 0)
    assert len(seen) + len(requests) <= cap


def test_expired_old_routes_do_not_start_fk(monkeypatch):
    from cascade.safety import trajectory

    clock, calls = [0.], []

    def vet(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            clock[0] = 3.
        return "blocked"

    def forbidden(*args, **kwargs):
        pytest.fail("expired route search must not start model work")

    monkeypatch.setattr(trajectory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(trajectory, "vet_route", vet)
    with pytest.raises(SkillError, match="time budget"):
        plan_route(SimpleNamespace(kin=SimpleNamespace(fk=forbidden, ik=forbidden)), [0.], [1.])


@pytest.mark.parametrize("corner", [False, True])
def test_old_passing_route_is_unchanged_and_never_calls_ik(corner):
    _, raw, harness, start, goal = lift_case()
    harness.limits.keep_out = []
    expected = [goal]
    if corner:
        start[1], goal[1] = -.2, .2
        harness.limits.keep_out = [(np.array([.34, -.05, .27]), np.array([.36, .05, .40]))]
        first = start.copy()
        first[0] = goal[0]
        expected = [first, goal]
    np.testing.assert_array_equal(plan_route(harness, start, goal), expected)
    assert harness.kin.requests == []
    assert raw.commands == []


def test_geometry_lock_covers_lift_model_work_and_every_route_leg(monkeypatch):
    _, raw, harness, _, goal = lift_case()

    class Lock:
        held = False
        entries = 0

        def acquire(self, *, timeout):
            assert 0. < timeout <= 3.
            assert not self.held
            self.held = True
            self.entries += 1
            return True

        def release(self):
            assert self.held
            self.held = False

    lock = Lock()
    harness.occupancy = SimpleNamespace(_refresh_lock=lock, clearance=lambda p: np.full(len(p), .1))
    fk, ik, step = harness.kin.fk, harness.kin.ik, harness.vet_step

    def guarded(fn):
        def call(*args, **kwargs):
            assert lock.held
            return fn(*args, **kwargs)
        return call

    monkeypatch.setattr(harness.kin, "fk", guarded(fk))
    monkeypatch.setattr(harness.kin, "ik", guarded(ik))
    monkeypatch.setattr(harness, "vet_step", guarded(step))
    route = plan_route(harness, raw.get_state().q, goal)
    assert len(route) == 2 and lock.entries == 1 and not lock.held


def test_lift_route_is_rechecked_before_streaming_and_at_every_sent_step(monkeypatch):
    safe, raw, harness, start, goal = lift_case()
    approved = []
    approve = harness.approve

    def record(a, b, dt, **kwargs):
        approve(a, b, dt, **kwargs)
        approved.append(np.asarray(b).copy())

    monkeypatch.setattr(harness, "approve", record)
    assert safe.move_planned(goal)
    assert len(raw.commands) == 300
    np.testing.assert_array_equal(approved, raw.commands)
    np.testing.assert_allclose(raw.commands[149], start + [0., 0., .03])
    np.testing.assert_allclose(raw.commands[-1], goal)


@pytest.mark.parametrize("when", ["before", "during"])
def test_new_obstacle_still_stops_a_lift_route(monkeypatch, when):
    from cascade.types import SafetyViolation

    safe, raw, harness, _, goal = lift_case()
    wall_lo, wall_hi = np.array([.29, -.1, .301]), np.array([.31, .1, .4])
    if when == "before":
        stream = raw.stream_to

        def changed(*args, **kwargs):
            harness.limits.keep_out.append((wall_lo, wall_hi))
            return stream(*args, **kwargs)

        monkeypatch.setattr(raw, "stream_to", changed)
    else:
        send = raw.send_joint_target

        def changed(q):
            send(q)
            if len(raw.commands) == 50:
                harness.limits.keep_out.append((wall_lo, wall_hi))

        monkeypatch.setattr(raw, "send_joint_target", changed)
    with pytest.raises(SafetyViolation, match="keep-out"):
        safe.move_planned(goal)
    assert len(raw.commands) == (0 if when == "before" else 50)


def test_lift_target_preserves_nonidentity_orientation(monkeypatch):
    _, _, harness, start, goal = lift_case()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    fk = harness.kin.fk

    def rotated(q):
        pose = fk(q)
        pose[:3, :3] = rotation
        return pose

    monkeypatch.setattr(harness.kin, "fk", rotated)
    plan_route(harness, start, goal)
    target, seed, _ = harness.kin.requests[0]
    np.testing.assert_array_equal(target[:3, :3], rotation)
    np.testing.assert_allclose(target[:3, 3], start + [0., 0., .03])
    np.testing.assert_array_equal(seed, start)


@needs_pin
@pytest.mark.parametrize("turn, succeeds", [(.3, True), (-.3, False)])
def test_signed_rebot_model_with_synthetic_wrist_camera_obstacle(turn, succeeds):
    """Real signed model/limits/volume, an invented obstacle; no live map.

    The point sits below the camera's rear edge. Direct yaw worsens at least
    one already-close sample. An upward endpoint permits one yaw direction,
    but the other still fails on the second leg. This is not scene evidence.
    """
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    from cascade.safety.harness import SafetyHarness, SafetyLimits
    from cascade.safety.tool_volume import build_tool_volume

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame, cfg.arm.n_joints, cfg.arm.joint_signs)
    assert cfg.arm.joint_signs == [-1] * 6
    start = np.asarray(cfg.arm.home_q, dtype=float)
    goal = start.copy()
    goal[0] = turn
    pose = kin.fk(start)
    # below the camera box's rear edge, 1 cm inside its side face: derived
    # from the shipped box, so a measured box moves the obstacle with it
    wc = cfg.relative_motion.wrist_camera
    obstacle = pose[:3, 3] + [-wc.behind_tip_m, wc.half_width_m - .01, -.024]
    occupancy = SimpleNamespace(
        clearance=lambda points: np.linalg.norm(points - obstacle, axis=1),
        required=True, unknown_blocks=True,
    )
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kin, occupancy=occupancy,
                            tool_volume=build_tool_volume(cfg, cfg.arm))
    assert harness.limits.min_clearance_m == .03
    assert "wrist camera clearance 0.026 m below 0.030 m" in harness.vet_pose(start)
    for old_route in ([goal], [goal, goal], [start, goal]):
        assert "wrist camera" in vet_route(harness, start, old_route, 3.)
    target = pose.copy()
    target[2, 3] += .03
    lift = kin.ik(target, q_init=start, max_iter=80, retries=0)
    assert lift.success
    assert vet_segment(harness, start, lift.q, 3.) is None
    if succeeds:
        route = plan_route(harness, start, goal)
        assert len(route) == 2
        np.testing.assert_allclose(kin.fk(route[0]), target, atol=1e-4)
        np.testing.assert_array_equal(route[-1], goal)
        assert vet_route(harness, start, route, 3.) is None
    else:
        assert "wrist camera" in vet_segment(harness, lift.q, goal, 3.)
        with pytest.raises(SkillError, match="no safe joint route.*wrist camera"):
            plan_route(harness, start, goal)
    assert harness.limits.min_clearance_m == .03
    assert harness._grasp_exempt is None and harness.violations == []
