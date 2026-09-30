"""Offline route regressions; no hardware, mapping service, or live proof."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg, load_demo_config
from cascade.control.mock_arm import MockArm
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.safety.trajectory import plan_route, vet_segment
from cascade.skills.runtime import SkillRuntime
from cascade.types import SafetyViolation, SkillError
from conftest import JOINT_SIGNS, URDF, needs_pin

# physical-12/tomato_can_r1/witness/samples.jsonl, sequence324 (asset q negated).
# Receipt SHA256 34c03d91b6e59dcef91c1a356ce32e3f4c1f49b3910485261979c21ba764b5ce
# Witness SHA256 9d966b34e1e504db6702c5e694bee2ca563fda6d56587bc5487a6aacccb223fc
CAN_AFTER_HOME_ABORT = np.array([
    1.1008832454681396, 1.8808261156082153, 1.8277220726013184,
    -1.4772156476974487, -1.8020049537881278e-05, 2.8682334423065186,
])


class CartesianKin:
    joint_limits = (np.full(3, -2.), np.full(3, 2.))

    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q
        return pose

    def link_positions(self, q):
        return np.array([[0., 0., .1], q])


def case(*, occupancy=None, start=(.3, 0., .3)):
    kin = CartesianKin()
    limits = SafetyLimits(np.array([.1, -.3, -.01]), np.array([.5, .3, .55]))
    harness = SafetyHarness(limits, kin, occupancy=occupancy)
    raw = MockArm(Cfg({"home_q": list(start), "n_joints": 3}), kin)
    raw.connect()
    return SafeArm(raw, harness), raw, harness


@needs_pin
def test_recorded_can_home_curve_is_replanned_inside_unchanged_workspace():
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(arm="isaac_kitchen_gpu", llm="mock")
    kin = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kin)
    home = np.asarray(cfg.arm.home_q)
    chord = np.array([kin.fk(CAN_AFTER_HOME_ABORT + s * (home - CAN_AFTER_HOME_ABORT))[:3, 3]
                      for s in np.linspace(0., 1., 301)])
    assert chord[:, 1].min() == pytest.approx(-.304828, abs=1e-6)
    assert "outside workspace" in vet_segment(harness, CAN_AFTER_HOME_ABORT, home, 3.)
    route = plan_route(harness, CAN_AFTER_HOME_ABORT, home)
    assert len(route) > 1
    raw = MockArm(Cfg({"home_q": CAN_AFTER_HOME_ABORT.tolist(), "n_joints": 6}), kin)
    raw.connect()
    safe = SafeArm(raw, harness)
    assert safe.move_planned(home)
    positions = np.array([kin.fk(q)[:3, 3] for q in raw.commands])
    assert np.all(positions >= harness.limits.workspace_min)
    assert np.all(positions <= harness.limits.workspace_max)
    np.testing.assert_allclose(raw.get_state().q, home)
    assert harness.violations == []


def test_preflight_preserves_escape_rules_and_never_mutates_live_state():
    _, _, harness = case()
    harness.halt("previous move")
    harness._last_heartbeat = 0.
    before = (harness.violations.copy(), harness._halt, harness._last_heartbeat,
              harness._motion_active, harness.estopped)
    # Absolute pose vetting refuses this, but the streamed escape is legal.
    assert harness.vet_pose(np.array([.55, 0., .3]))
    assert harness.vet_step(np.array([.6, 0., .3]), np.array([.55, 0., .3]), 1.) is None
    assert "outside workspace" in harness.vet_step(
        np.array([.3, 0., .3]), np.array([.55, 0., .3]), 1.)
    assert before == (harness.violations, harness._halt, harness._last_heartbeat,
                      harness._motion_active, harness.estopped)


def test_joint_margin_escape_remains_distinct_from_static_target_vetting():
    _, _, harness = case()
    harness.kin = SimpleNamespace(joint_limits=(np.zeros(3), np.ones(3)),
                                 fk=lambda q: np.eye(4))
    # Separate the joint-margin gate from the synthetic TCP workspace.
    harness.limits.workspace_min = np.full(3, -1.)
    harness.limits.table_z = -.1
    harness.kin.link_positions = lambda q: np.array([[0., 0., .1]])
    assert harness.vet_step(np.array([0., .5, .5]), np.array([.005, .5, .5]), 1.) is None
    assert "joint 1" in harness.vet_step(np.array([0., .5, .5]), np.array([0., .5, .5]), 1.)
    assert harness.violations == []


@pytest.mark.parametrize("clearance", [.019, np.nan])
def test_blocked_or_unknown_payload_never_authorizes_home(clearance):
    occupancy = SimpleNamespace(
        clearance=lambda p: np.full(len(p), .1),
        payload_points=lambda pose: pose[None, :3, 3],
        payload_clearance=lambda p: np.full(len(p), clearance),
    )
    safe, raw, harness = case(occupancy=occupancy)
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned([.4, 0., .4])
    assert not raw.commands and not harness.violations
    assert harness.limits.min_clearance_m == .03
    assert harness._grasp_exempt is None


def test_empty_tool_at_19mm_fails_before_any_home_command():
    occupancy = SimpleNamespace(clearance=lambda p: np.full(len(p), .019))
    safe, raw, harness = case(occupancy=occupancy)
    with pytest.raises(SkillError, match="0.019 m below 0.030 m"):
        safe.move_planned([.4, 0., .4])
    assert not raw.commands and not harness.violations
    assert harness._grasp_exempt is None


@pytest.mark.parametrize("change", ["feedback", "map"])
def test_changed_state_or_map_is_rechecked_at_backend_stream_start(change):
    distance = [.1]
    occupancy = SimpleNamespace(clearance=lambda p: np.where(p[:, 2] < .25, .019, distance[0]))
    safe, raw, harness = case(occupancy=occupancy)
    stream = raw.stream_to

    def changed(*args, **kwargs):
        if change == "feedback":
            raw._q[2] = .2
        else:
            distance[0] = .019
        return stream(*args, **kwargs)

    raw.stream_to = changed
    with pytest.raises(SafetyViolation, match="planned route became unsafe"):
        safe.move_planned([.4, 0., .4])
    assert not raw.commands and not harness.violations
    assert not harness._motion_active


def test_live_map_gate_still_stops_an_obstacle_appearing_during_stream():
    occupancy = SimpleNamespace(clearance=lambda p: np.full(len(p), .1))
    safe, raw, harness = case(occupancy=occupancy)
    send = raw.send_joint_target

    def changed(q):
        send(q)
        if len(raw.commands) == 10:
            occupancy.clearance = lambda p: np.full(len(p), .019)

    raw.send_joint_target = changed
    with pytest.raises(SafetyViolation, match="0.019 m below 0.030 m"):
        safe.move_planned([.4, 0., .4])
    assert len(raw.commands) == 10
    assert len(harness.violations) == 1 and not harness._motion_active


@pytest.mark.parametrize("stop", ["estop", "watchdog"])
def test_planning_does_not_bypass_estop_or_refresh_the_watchdog(stop):
    safe, raw, harness = case()
    if stop == "estop":
        harness.estop("operator")
    else:
        harness._last_heartbeat = 0.
    heartbeat = harness._last_heartbeat
    with pytest.raises((SkillError, SafetyViolation)):
        safe.move_planned([.4, 0., .4])
    assert not raw.commands and harness._last_heartbeat == heartbeat
    assert not harness._motion_active


def test_motion_exception_propagates_and_clears_only_motion_active():
    safe, raw, harness = case()
    def fail(q):
        raise RuntimeError("transport failed")
    raw.send_joint_target = fail
    with pytest.raises(RuntimeError, match="transport failed"):
        safe.move_planned([.4, 0., .4])
    assert not harness._motion_active and not harness.violations


def test_preflight_time_budget_refuses_without_authorizing_a_partial_route(monkeypatch):
    import cascade.safety.trajectory as trajectory
    _, _, harness = case()
    ticks = iter([0., 4.])
    monkeypatch.setattr(trajectory, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(SkillError, match="time budget"):
        plan_route(harness, [.3, 0., .3], [.4, 0., .4])
    assert not harness.violations


def test_candidate_budget_is_bounded_for_arbitrary_dof():
    calls = []
    harness = SimpleNamespace(limits=SimpleNamespace(max_joint_vel=1.2),
                              vet_step=lambda *args: calls.append(True) or "blocked")
    with pytest.raises(SkillError, match="no safe joint route"):
        plan_route(harness, np.zeros(20), np.ones(20))
    assert len(calls) == 15


def test_home_uses_planned_move_and_preserves_settling_failure():
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = Cfg({"arm": {"home_q": [.4, 0., .4]}})
    safe, _, _ = case()
    rt.arm = safe
    assert rt.skill_move_home() == {"at": "home"}
    safe.move_planned = lambda *args, **kwargs: False
    with pytest.raises(SkillError, match="did not settle at home"):
        rt.skill_move_home()


def test_slow_preflight_finishes_before_stream_deadlines_begin(monkeypatch):
    import cascade.control.arm_base as arm_base

    safe, raw, _ = case()
    clock = [0.]
    sent = []
    send = raw.send_joint_target

    def capture(q):
        sent.append(clock[0])
        send(q)

    def sleep(seconds):
        assert seconds > 0
        clock[0] += seconds

    raw.send_joint_target = capture
    raw.stream_to = arm_base.ArmBase.stream_to.__get__(raw)
    monkeypatch.setattr(arm_base, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))

    def slow_preflight(start, duration):
        assert duration == 3. and not sent
        np.testing.assert_array_equal(start, [.3, 0., .3])
        clock[0] += .8

    assert safe.move_joints(np.array([.4, 0., .4]), duration_s=3., _preflight=slow_preflight)
    assert len(sent) == 150
    np.testing.assert_allclose(sent, .8 + np.arange(150) * .02, atol=1e-12)
    assert clock[0] == pytest.approx(3.8)


def test_backend_without_preflight_contract_is_not_retried_unchecked():
    safe, raw, _ = case()
    entered = []
    def unsupported(goal, duration_s, approve=None):
        entered.append(True)
        return True
    raw.stream_to = unsupported
    with pytest.raises(TypeError):
        safe.move_joints([.4, 0., .4], _preflight=lambda *a: None, bias_compensate=True)
    assert not entered and not raw.commands


@pytest.mark.parametrize("backend", ["mock", "base"])
@pytest.mark.parametrize("change", ["feedback", "watchdog", "halt", "estop"])
def test_changes_during_preflight_refuse_before_first_target(backend, change):
    from cascade.control.arm_base import ArmBase

    safe, raw, harness = case()
    if backend == "base":
        raw.stream_to = ArmBase.stream_to.__get__(raw)

    def changed(start, duration):
        if change == "feedback":
            raw._q[0] += .02
        elif change == "watchdog":
            harness._last_heartbeat = 0.
        elif change == "halt":
            harness.halt("new instruction during planning")
        else:
            harness.estop("operator during planning")

    with pytest.raises(SafetyViolation):
        safe.move_joints([.4, 0., .4], _preflight=changed)
    assert not raw.commands and not harness._motion_active
    if change == "watchdog":
        assert harness._last_heartbeat == 0.
    elif change == "halt":
        assert harness._halt == "new instruction during planning"
    elif change == "estop":
        assert harness.estopped


def test_goal_blocked_by_fresh_map_never_executes_the_initial_safe_part():
    occupancy = SimpleNamespace(clearance=lambda p: np.where(p[:, 0] > .38, .019, .1))
    safe, raw, harness = case(occupancy=occupancy)
    with pytest.raises(SkillError, match="no safe joint route"):
        safe.move_planned([.4, 0., .4])
    assert not raw.commands and not harness.violations


@pytest.mark.parametrize("previous", [False, True])
def test_new_halt_during_initial_plan_is_not_cleared_as_an_old_halt(previous):
    safe, raw, harness = case()
    if previous:
        harness.halt("same instruction")
    vet = harness.vet_step
    halted = []

    def interrupted(*args, **kwargs):
        if not halted:
            harness.halt("same instruction")
            halted.append(True)
        return vet(*args, **kwargs)

    harness.vet_step = interrupted
    with pytest.raises(SafetyViolation, match="halt received during route"):
        safe.move_planned([.4, 0., .4])
    assert not raw.commands and harness.halted == "same instruction"
    assert not harness._motion_active


def test_new_halt_between_route_segments_prevents_the_next_segment():
    occupancy = SimpleNamespace(clearance=lambda p: np.where(
        (np.abs(p[:, 0] - .35) < .02) & (np.abs(p[:, 1]) < .05), .019, .1))
    safe, raw, harness = case(occupancy=occupancy, start=(.3, -.2, .3))
    goal = np.array([.4, .2, .3])
    stream = raw.stream_to
    segments = []

    def interrupted(target, *args, **kwargs):
        segments.append(np.asarray(target).copy())
        result = stream(target, *args, **kwargs)
        harness.halt("cancel remaining route")
        return result

    raw.stream_to = interrupted
    with pytest.raises(SafetyViolation, match="halt received during route"):
        safe.move_planned(goal)
    assert len(segments) == 1 and raw.commands
    assert not np.array_equal(raw.get_state().q, goal)
    assert harness.halted == "cancel remaining route" and not harness._motion_active


def test_previous_halt_remains_recoverable_for_a_new_home_request():
    safe, raw, harness = case()
    harness.halt("previous motion")
    generation = harness._halt_generation
    assert safe.move_planned([.4, 0., .4])
    assert raw.commands and harness.halted is None
    assert harness._halt_generation == generation
