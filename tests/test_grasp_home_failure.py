"""A failed initial home move must not become another robot move."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.skills.carry_attachment import AttachmentInvalid
from cascade.skills.runtime import SkillRuntime
from cascade.types import SafetyViolation, SkillError


class _Arm:
    def __init__(self, home_result):
        self.home_result = home_result
        self.moves = []
        self.gripper = []
        self.stops = 0
        from cascade.safety.harness import SafetyHarness, SafetyLimits
        self.harness = SafetyHarness(SafetyLimits(
            workspace_min=np.array([.2, -.6, -.01]), workspace_max=np.ones(3), table_z=0.0))
        self.harness.heartbeat()  # a fresh observation: the harness is built stale
        self.raw = self
        self.n_joints = 7

    def get_state(self):
        return SimpleNamespace(q=np.zeros(7), physics_clock=None)

    def set_gripper(self, pos, effort):
        self.gripper.append(pos)

    def move_joints(self, q, **kw):
        self.moves.append(np.array(q))
        if len(self.moves) == 1:
            if callable(self.home_result):
                return self.home_result(self.harness)
            if isinstance(self.home_result, Exception):
                raise self.home_result
            return self.home_result
        return False

    def stop(self):
        self.stops += 1


def _runtime(monkeypatch, home_result):
    runtime = SkillRuntime.__new__(SkillRuntime)
    runtime.cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    runtime.arm = _Arm(home_result)
    runtime.kin = object()
    runtime.memory = Mock()
    runtime.grasp_memory = Mock()
    runtime._call_measurements = None
    runtime.held_object = None
    runtime._grip_open = .04
    runtime._max_width = .08
    runtime._reconcile_held = lambda: None
    grasp = SimpleNamespace(position=np.array([.45, 0, .03]))
    runtime._plan_grasps = lambda *a, **kw: [grasp]
    pregrasp = np.ones(7)
    monkeypatch.setattr("cascade.skills.runtime.select_grasp",
                        lambda *a, **kw: (grasp, pregrasp, np.ones(7) * 2))
    fix = SimpleNamespace(detection=SimpleNamespace(label="rubber duck"))
    return runtime, fix, pregrasp


@pytest.mark.parametrize("home_result, message", [
    (False, "did not settle at home pose before pregrasp"),
    (RuntimeError("motion aborted by reflex"),
     "home move failed before pregrasp: motion aborted by reflex"),
])
def test_home_failure_stops_and_never_sends_pregrasp_or_closing_command(
    monkeypatch, home_result, message,
):
    runtime, fix, pregrasp = _runtime(monkeypatch, home_result)
    with pytest.raises(SkillError, match=message):
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert len(runtime.arm.moves) == 1
    np.testing.assert_array_equal(runtime.arm.moves[0], runtime.cfg.arm.home_q)
    assert runtime.arm.gripper == [.04]
    assert runtime.arm.stops == 1


def _refused_before_any_command(exc):
    def move(harness):
        raise exc
    return move


def _failed_after_its_first_command(exc):
    def move(harness):
        harness.approved_commands += 1  # the move's first command was approved
        raise exc
    return move


@pytest.mark.parametrize("exc", [
    SafetyViolation("perception watchdog: last observation 6.0s old"),
    SkillError("no safe joint route found; direct route: TCP outside workspace"),
])
def test_a_home_move_refused_before_its_first_command_stops_nothing(monkeypatch, exc):
    """#414 review Important 1: the arm did not move, so no stop (`SafeArm.stop()` latches the
    e-stop): the refusal is raised as itself."""
    runtime, fix, _ = _runtime(monkeypatch, _refused_before_any_command(exc))
    with pytest.raises(type(exc)) as raised:
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert raised.value is exc
    assert len(runtime.arm.moves) == 1 and runtime.arm.gripper == [.04]
    assert runtime.arm.stops == 0


@pytest.mark.parametrize("make, exc", [
    (_failed_after_its_first_command, SafetyViolation("joint 3 tracking fault")),
    (_failed_after_its_first_command, SkillError("planned route became unsafe")),
    (_refused_before_any_command, RuntimeError("CAN bus fault")),
    (_refused_before_any_command, SafetyViolation("motion safety callback failed")),
    (_refused_before_any_command, AttachmentInvalid("retained attachment arm, map, or producer identity changed")),
])
def test_a_home_move_that_started_or_hit_a_backend_fault_still_stops(monkeypatch, make, exc):
    """A move that started (it may have stopped anywhere), a non-refusal fault (the backend's), a
    refusal raised FROM one (`raise SafetyViolation(...) from TypeError`), or an attachment loss
    (its own terminal rule) keeps the stop."""
    if str(exc) == "motion safety callback failed":
        try:
            raise exc from TypeError("unexpected keyword 'before_stream'")
        except SafetyViolation:
            pass
    runtime, fix, _ = _runtime(monkeypatch, make(exc))
    with pytest.raises(SkillError, match=f"^home move failed before pregrasp: {exc}$"):
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert runtime.arm.stops == 1


def test_a_harness_that_counts_no_approved_commands_keeps_the_stop(monkeypatch):
    """Without a command count nothing proves the move never started: stop, as before."""
    runtime, fix, _ = _runtime(monkeypatch, _refused_before_any_command(
        SafetyViolation("perception watchdog: last observation 6.0s old")))
    runtime.arm.harness.approved_commands = None
    with pytest.raises(SkillError, match="^home move failed before pregrasp: perception watchdog"):
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert runtime.arm.stops == 1


def test_successful_home_allows_pregrasp_and_preserves_its_failure_phase(monkeypatch):
    runtime, fix, pregrasp = _runtime(monkeypatch, True)
    with pytest.raises(SkillError, match="did not settle at pregrasp pose"):
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert len(runtime.arm.moves) == 2
    np.testing.assert_array_equal(runtime.arm.moves[1], pregrasp)
    assert runtime.arm.gripper == [.04]


def test_saved_view_policy_approaches_from_live_pose_without_initial_home(monkeypatch):
    runtime, fix, pregrasp = _runtime(monkeypatch, False)
    runtime.cfg._data["arm"]["return_home_after_motion"] = True
    seeds = []

    def select(grasps, kin, seed, **kwargs):
        seeds.append(np.array(seed))
        return grasps[0], pregrasp, np.ones(7) * 2

    monkeypatch.setattr("cascade.skills.runtime.select_grasp", select)
    with pytest.raises(SkillError, match="did not settle at pregrasp pose"):
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert len(runtime.arm.moves) == 1
    np.testing.assert_array_equal(runtime.arm.moves[0], pregrasp)
    np.testing.assert_array_equal(seeds[0], runtime.arm.get_state().q)
    assert runtime.arm.gripper == [.04]
    assert runtime.arm.stops == 0


def test_grasp_rejects_collision_between_old_sparse_descent_samples(monkeypatch):
    runtime, fix, _ = _runtime(monkeypatch, False)
    runtime.cfg._data["arm"]["return_home_after_motion"] = True
    runtime.grasp_memory = Mock()
    seen = []

    def vet(q, **exemption):
        seen.append((q.copy(), exemption))
        # Previous descent fractions .15 and .30 both miss this obstacle.
        if 1.19 <= q[0] <= 1.21:
            return "point 4 clearance below required distance"
        return None

    runtime.arm.harness.vet_pose = vet

    def select(grasps, kin, seed, **kwargs):
        reason = kwargs["validate"](grasps[0], np.ones(7), np.ones(7) * 2)
        assert "descent unsafe: point 4" in reason
        raise SkillError(reason)

    monkeypatch.setattr("cascade.skills.runtime.select_grasp", select)
    with pytest.raises(SkillError, match="descent unsafe"):
        runtime.skill_grasp_object("rubber duck", _fix=fix, _frame=object())
    assert runtime.arm.moves == []
    assert runtime.arm.gripper == []
    assert all(not exemption for q, exemption in seen if q[0] < 1.)


def test_grasp_offset_is_measured_before_lift(monkeypatch):
    runtime, fix, _ = _runtime(monkeypatch, True)
    runtime.cfg._data["arm"]["return_home_after_motion"] = True
    runtime._grip_closed = 0.
    runtime.arm.move_joints = lambda q, **kw: runtime.arm.moves.append(np.array(q)) or True
    runtime.arm.get_state = lambda: SimpleNamespace(
        q=runtime.arm.moves[-1] if runtime.arm.moves else np.zeros(7), physics_clock=None)
    runtime.arm.harness.allow_grasp_descent = Mock()
    runtime.arm.harness.clear_grasp_exemption = Mock()

    def fk(q):
        T = np.eye(4)
        T[:3, 3] = [.45, 0., .02 if q[0] == 2 else .14]
        return T

    runtime.kin = SimpleNamespace(fk=fk)
    runtime._close_two_stage = Mock()
    runtime._gripper_width_frac = lambda: .5
    runtime._plan_grasps = lambda *a, **kw: [SimpleNamespace(
        position=np.array([.45, 0., .03]), width_m=.04)]
    monkeypatch.setattr("cascade.skills.runtime.select_grasp",
                        lambda grasps, *a, **kw: (grasps[0], np.ones(7), np.ones(7) * 2))
    runtime.beliefs = Mock()
    runtime.memory = Mock()
    runtime.grasp_memory = Mock()
    fix.position = np.array([.45, 0., .03])
    monkeypatch.setattr("cascade.skills.runtime.detection_color", lambda *a: None)
    result = runtime.skill_grasp_object("rubber duck", _fix=fix,
                                       _frame=SimpleNamespace(rgb=None))
    assert result["held"] == "rubber duck"
    np.testing.assert_allclose(runtime._held_offset, [0., 0., .01])
    assert runtime.arm.moves[-1][0] == 1  # already lifted to pregrasp
