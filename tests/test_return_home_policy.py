"""Saved camera-view home is restored only after an authorized action began.

These tests use the real skill dispatcher and safety watchdog. Hardware and
camera boundaries are local doubles; no controller or robot is started.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import time

import numpy as np
import pytest

from cascade.agent.trace import TraceLogger
from cascade.config import load_demo_config
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.perception.grounding import Extrinsics
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.skills.runtime import SkillRuntime, TOOL_SPECS, tool_specs_for_config
from cascade.types import Frame, MotionHalted, SafetyViolation, SkillError


class _Native:
    def __init__(self):
        self._stopped = False
        self.error = None

    def ensure_motion_ready(self):
        if self.error:
            raise RuntimeError(self.error)


class _Arm:
    n_joints = 7

    def __init__(self, events, q=None):
        self.events = events
        self.q = np.asarray(q if q is not None else np.zeros(7), float)
        self.raw = _Native()
        self.harness = SafetyHarness(SafetyLimits(
            workspace_min=np.array([-1., -1., -1.]),
            workspace_max=np.ones(3), watchdog_s=5.,
        ))
        self.harness.heartbeat()
        self.moves = []
        self.gripper = []
        self.stops = 0
        self.move_outcomes = {}

    def get_state(self):
        return SimpleNamespace(q=self.q.copy(), gripper_pos=.02)

    def move_joints(self, q, **kwargs):
        self.harness.begin_motion()
        try:
            target = np.asarray(q, float).copy()
            self.moves.append(target)
            self.events.append(("move", target.tolist()))
            outcome = self.move_outcomes.get(len(self.moves), True)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome:
                self.q = target
            return outcome
        finally:
            self.harness.end_motion()

    def set_gripper(self, pos, effort):
        self.gripper.append(pos)
        self.events.append(("gripper", pos))

    def stop(self):
        self.stops += 1
        self.raw._stopped = True


class _Camera:
    def __init__(self, events):
        self.events = events
        self.error = None
        self.cached_frame = None

    def get_frame(self):
        self.events.append(("frame", None))
        if self.error:
            raise self.error
        if self.cached_frame is not None:
            return self.cached_frame
        return Frame(np.zeros((8, 8, 3), np.uint8), None, np.eye(3))


class _Watcher:
    def __init__(self, events):
        self.events = events

    @contextmanager
    def paused(self):
        self.events.append(("pause", None))
        try:
            yield
        finally:
            self.events.append(("resume", None))

    def ignore_label(self, label):
        self.events.append(("ignore", label))


def _runtime(tmp_path, enabled=True, q=None):
    cfg = load_demo_config(camera="mock", arm="fr3_mock", llm="mock")
    cfg._data["arm"]["home_q"] = [0.] * 7
    cfg._data["arm"]["return_home_after_motion"] = enabled
    events = []
    arm = _Arm(events, q=q)
    camera = _Camera(events)
    runtime = SkillRuntime(
        camera, SimpleNamespace(ensure_depth=lambda frame: frame), None,
        Extrinsics(), None, arm, EpisodicMemory(), BeliefStore(),
        TraceLogger(tmp_path / "run"), cfg,
    )
    runtime.watcher = _Watcher(events)
    return runtime, arm, events


def test_success_returns_snapshot_inside_selected_scope_without_releasing(tmp_path):
    runtime, arm, events = _runtime(tmp_path)
    runtime.held_object = "cup"
    runtime._held_det_label = "blue cup"
    runtime._held_provisional = "cup"

    def action():
        runtime._move_joints(np.ones(7) * .2)
        runtime.cfg._data["arm"]["home_q"] = [.8] * 7
        return {"held": "cup", "tcp_xyz": [.4, 0., .2]}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})

    assert result["ok"] is True
    assert result["return_home"]["status"] == "returned"
    np.testing.assert_array_equal(arm.moves[-1], np.zeros(7))
    assert result["tcp_xyz"] == [.4, 0., .2]
    assert runtime.held_object == runtime._held_provisional == "cup"
    assert arm.gripper == []
    last_move = max(i for i, event in enumerate(events) if event[0] == "move")
    resume = next(i for i, event in enumerate(events) if event[0] == "resume")
    assert events[last_move - 1][0] == "frame"
    assert last_move < resume
    assert events[resume - 1] == ("ignore", "blue cup")
    assert events[resume + 1][0] == "frame"
    assert runtime._motion_scope.context is None
    assert runtime.arm is arm


@pytest.mark.parametrize("raised", [False, True])
def test_ordinary_grasp_failure_returns_home_and_preserves_error(tmp_path, raised):
    runtime, arm, _ = _runtime(tmp_path)

    def action(label):
        runtime._move_joints(np.ones(7) * .2)
        if raised:
            raise SkillError("air grasp")
        return {"ok": False, "error": "air grasp", "grasp": None}

    runtime.skill_grasp_object = action
    result = runtime.execute("grasp_object", {"label": "cup"})

    assert result["ok"] is False
    assert result["error"] == ("SkillError: air grasp" if raised else "air grasp")
    assert result["return_home"]["ok"] is True
    np.testing.assert_array_equal(arm.q, np.zeros(7))
    assert len(arm.moves) == 2


@pytest.mark.parametrize("action_ok", [False, True])
def test_home_failure_is_separate_and_never_releases_held_object(tmp_path, action_ok):
    runtime, arm, _ = _runtime(tmp_path)
    runtime.held_object = "cup"
    runtime._held_provisional = "cup"
    arm.move_outcomes[2] = False

    def action():
        runtime._move_joints(np.ones(7) * .2)
        return {"ok": action_ok, "error": "original grasp failure"}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})

    assert result["ok"] is False
    assert result["return_home"]["status"] == "failed"
    assert "did not settle" in result["return_home"]["error"]
    if action_ok:
        assert result["action_ok"] is True
        assert "return to saved home failed" in result["error"]
    else:
        assert result["error"] == "original grasp failure"
    assert runtime.held_object == runtime._held_provisional == "cup"
    assert arm.stops == 1
    assert arm.gripper == []


@pytest.mark.parametrize("abort", ["stop", "estop", "halt", "cancel", "reflex", "stale", "violation"])
def test_abort_suppresses_home_even_when_composite_wrapped_failure(tmp_path, abort):
    runtime, arm, _ = _runtime(tmp_path)

    def action():
        runtime._move_joints(np.ones(7) * .2)
        if abort == "stop":
            arm.raw._stopped = True
        elif abort == "estop":
            arm.harness.estop("operator stop")
        elif abort == "halt":
            arm.harness.halt("redirect requested")
        elif abort == "cancel":
            runtime.motion_cancel_requested = lambda: True
        elif abort == "reflex":
            arm.raw.error = "native cartesian_reflex"
        elif abort == "stale":
            arm.raw.error = "joint feedback stale"
        else:
            arm.harness.violations.append("unsafe trajectory")
        return {"ok": False, "error": "pick phase failed"}

    runtime.skill_pick_and_place = action
    result = runtime.execute("pick_and_place", {})

    assert result["error"] == "pick phase failed"
    assert result["return_home"]["status"] == "skipped"
    assert len(arm.moves) == 1


@pytest.mark.parametrize("exception", [SafetyViolation("unsafe waypoint"), MotionHalted("redirect")])
def test_direct_safety_abort_suppresses_home(tmp_path, exception):
    runtime, arm, _ = _runtime(tmp_path)

    def action():
        runtime._move_joints(np.ones(7) * .2)
        raise exception

    runtime.skill_wave = action
    result = runtime.execute("wave", {})
    assert result["return_home"]["status"] == "skipped"
    assert "safety abort" in result["return_home"]["reason"]
    assert len(arm.moves) == 1


def test_success_cannot_claim_home_when_cancelled_before_cleanup(tmp_path):
    runtime, arm, _ = _runtime(tmp_path)

    def action():
        runtime._move_joints(np.ones(7) * .2)
        runtime.motion_cancel_requested = lambda: True
        return {"ok": True}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})
    assert result["ok"] is False and result["action_ok"] is True
    assert "cancelled" in result["error"]
    assert len(arm.moves) == 1


@pytest.mark.parametrize("name, args", [
    ("rotate_joint", {"joint": 0, "degrees": 10}),
    ("move_relative", {"direction": "bogus"}),
    ("rotate_joint", {"bogus": 1}),
])
def test_invalid_motion_request_away_from_home_sends_no_goal(tmp_path, name, args):
    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .2)
    result = runtime.execute(name, args)
    assert result["ok"] is False
    assert "return_home" not in result
    assert arm.moves == arm.gripper == []


def test_failed_localization_before_grasp_sends_no_home_goal(tmp_path):
    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .2)
    runtime._reconcile_held = lambda: None

    def localize(*args, **kwargs):
        raise SkillError("cup is not visible")

    runtime._localize = localize
    result = runtime.execute("grasp_object", {"label": "cup"})
    assert "cup is not visible" in result["error"]
    assert "return_home" not in result
    assert arm.moves == arm.gripper == []


def test_native_preflight_failure_sends_no_motion_or_gripper_command(tmp_path):
    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .2)
    arm.raw.error = "native UserStopped"
    result = runtime.execute("open_gripper", {})
    assert "UserStopped" in result["error"]
    assert "return_home" not in result
    assert arm.moves == arm.gripper == []


@pytest.mark.parametrize("failure", ["trajectory", "native_fault"])
def test_motion_abort_ends_pick_without_model_fallback_or_false_grasp_verdict(tmp_path, failure):
    from unittest.mock import Mock
    from cascade.agent.llm import MockLLM
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.agent.reflex import FastPlanner

    runtime, arm, _ = _runtime(tmp_path)
    runtime.effects = Mock()
    runtime.effects.snapshot.return_value = {}
    error = "GOAL_TOLERANCE_VIOLATED" if failure == "trajectory" else "native cartesian_reflex"

    def grasp(label):
        runtime._move_joints(np.ones(7) * .2)
        if failure == "trajectory":
            arm.raw._stopped = True
        else:
            arm.raw.error = error
        raise RuntimeError(error)

    runtime.skill_grasp_object = grasp
    llm = MockLLM([])
    agent = AgentOrchestrator(llm, runtime, fast_planner=FastPlanner(),
                              decompose=False, verify_milestones=False)
    report = agent.run_task("pick up orange ball")
    assert not report.success and report.path == "reflex" and report.steps == 1
    assert error in report.summary
    result = report.tool_log[0]["result"]
    assert result["motion_blocked"]
    assert result["return_home"]["status"] == "skipped"
    assert result["postcondition"]["status"] == "unverified"
    runtime.effects.verify.assert_not_called()
    assert not llm.requests and len(arm.moves) == 1 and not arm.gripper


def test_only_outer_composite_returns_and_nested_steps_stay_together(tmp_path):
    runtime, arm, _ = _runtime(tmp_path)

    def grasp(label):
        runtime._move_joints(np.ones(7) * .1)
        return {"held": label}

    def place(x, y, z):
        runtime._move_joints(np.ones(7) * .2)
        return {"placed": "cup"}

    def composite():
        first = runtime.execute("grasp_object", {"label": "cup"})
        assert len(arm.moves) == 1 and "return_home" not in first
        second = runtime.execute("place_at", {"x": .4, "y": 0, "z": .02})
        assert len(arm.moves) == 2 and "return_home" not in second
        return {"ok": first["ok"] and second["ok"]}

    runtime.skill_grasp_object = grasp
    runtime.skill_place_at = place
    runtime.skill_pick_and_place = composite
    result = runtime.execute("pick_and_place", {})
    assert result["ok"] is True
    assert len(arm.moves) == 3
    np.testing.assert_array_equal(arm.moves[-1], np.zeros(7))


def test_selected_arm_uses_its_saved_home_and_restores_primary(tmp_path):
    runtime, primary, events = _runtime(tmp_path)
    first = runtime.cfg._data["arm"].copy()
    first["name"] = "primary_arm"
    second = first.copy()
    second.update(name="secondary_arm", home_q=[.1] * 7)
    runtime.cfg._data["arms"] = [first, second]
    selected = _Arm(events, q=np.ones(7) * .1)
    runtime.arm_rig = SimpleNamespace(get=lambda name: selected,
                                      arms={"primary_arm": primary, "secondary_arm": selected})

    def action():
        assert runtime.arm is selected
        runtime._move_joints(np.ones(7) * .3)
        return {}

    runtime.skill_wave = action
    result = runtime.execute("wave", {"arm": "secondary_arm"})
    assert result["ok"] is True
    np.testing.assert_array_equal(selected.moves[-1], np.ones(7) * .1)
    assert primary.moves == []
    assert runtime.arm is primary


def test_explicit_home_and_gripper_at_home_send_no_additional_home(tmp_path):
    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .2)
    result = runtime.execute("move_home", {})
    assert result["ok"] is True and result["return_home"]["status"] == "returned"
    assert len(arm.moves) == 1
    result = runtime.execute("open_gripper", {})
    assert result["ok"] is True
    assert result["return_home"]["status"] == "already_home"
    assert len(arm.moves) == 1
    assert len(arm.gripper) == 1


def test_rest_without_declared_park_pose_uses_home_and_preserves_grasp(tmp_path):
    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .2)
    runtime.held_object = "cup"

    result = runtime.execute("move_to_rest", {})

    assert result["ok"] is True
    assert result["safe_to_power_off"] is False
    assert result["torque"] == "on"
    assert result["rest_pose"] is None
    assert len(arm.moves) == 1 and arm.gripper == []
    assert runtime.held_object == "cup"
    np.testing.assert_array_equal(arm.q, np.zeros(7))


@pytest.mark.parametrize("failure", [None, "camera unavailable", "trajectory refused"])
def test_resume_runs_checked_home_and_preserves_grasp(tmp_path, failure):
    from cascade.agent.llm import MockLLM
    from cascade.agent.orchestrator import AgentOrchestrator

    runtime, arm, events = _runtime(tmp_path, q=np.ones(7) * .2)
    runtime.held_object = runtime._held_det_label = "ball"
    arm.raw._stopped = True
    arm.raw.check_resume_ready = lambda: None
    arm.raw.resume = lambda: setattr(arm.raw, "_stopped", False)
    if failure == "camera unavailable":
        runtime.camera.error = RuntimeError(failure)
    elif failure == "trajectory refused":
        arm.move_outcomes[1] = False
    agent = AgentOrchestrator(MockLLM([]), runtime,
                              decompose=False, verify_milestones=False)
    if failure:
        with pytest.raises(RuntimeError, match="Resume home return failed"):
            agent.resume()
        assert arm.raw._stopped
        if failure == "camera unavailable":
            assert not arm.moves
    else:
        assert agent.resume()["resumed"]
        np.testing.assert_array_equal(arm.q, np.zeros(7))
        assert len(arm.moves) == 1 and not arm.raw._stopped
    assert runtime.held_object == "ball"
    assert arm.gripper == []


def test_readonly_call_and_disabled_profile_never_return(tmp_path):
    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .2)
    runtime.skill_describe_scene = lambda: {"scene": "cup"}
    result = runtime.execute("describe_scene", {})
    assert result["ok"] is True and "return_home" not in result
    assert arm.moves == []
    runtime.cfg._data["arm"]["return_home_after_motion"] = False

    def action():
        runtime._move_joints(np.ones(7) * .3)
        return {}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})
    assert result["ok"] is True and "return_home" not in result
    assert len(arm.moves) == 1


def test_long_action_refreshes_real_perception_before_home_watchdog(tmp_path):
    runtime, arm, events = _runtime(tmp_path)

    def action():
        runtime._move_joints(np.ones(7) * .2)
        arm.harness._last_heartbeat = time.monotonic() - 20.
        return {}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})
    assert result["ok"] is True, result
    assert not arm.harness.estopped
    assert len(arm.moves) == 2
    assert events[max(i for i, event in enumerate(events) if event[0] == "move") - 1][0] == "frame"


def test_camera_failure_skips_home_instead_of_refreshing_watchdog_blindly(tmp_path):
    runtime, arm, _ = _runtime(tmp_path)

    def action():
        runtime._move_joints(np.ones(7) * .2)
        runtime.camera.error = RuntimeError("camera disconnected")
        return {}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})
    assert result["ok"] is False
    assert result["return_home"]["status"] == "skipped"
    assert "camera disconnected" in result["return_home"]["reason"]
    assert len(arm.moves) == 1


@pytest.mark.parametrize("timestamp", ["old", "nan", "future"])
def test_cached_invalid_camera_frame_skips_home_without_heartbeat(tmp_path, timestamp):
    runtime, arm, _ = _runtime(tmp_path)
    saved_heartbeat = []

    def action():
        runtime._move_joints(np.ones(7) * .2)
        frame_t = {"old": time.monotonic() - 60., "nan": float("nan"),
                   "future": time.monotonic() + 60.}[timestamp]
        runtime.camera.cached_frame = Frame(np.zeros((8, 8, 3), np.uint8), None,
                                            np.eye(3), t=frame_t)
        arm.harness._last_heartbeat = time.monotonic() - 20.
        saved_heartbeat.append(arm.harness._last_heartbeat)
        return {}

    runtime.skill_wave = action
    result = runtime.execute("wave", {})
    assert result["ok"] is False
    assert result["return_home"]["status"] == "skipped"
    assert "camera frame is stale or invalid" in result["return_home"]["reason"]
    assert len(arm.moves) == 1
    assert arm.harness._last_heartbeat == saved_heartbeat[0]


def test_watcher_entry_failure_restores_scope_and_arm_binding(tmp_path):
    runtime, arm, _ = _runtime(tmp_path)

    @contextmanager
    def paused():
        raise RuntimeError("watcher failed to pause")
        yield

    runtime.watcher.paused = paused
    result = runtime.execute("open_gripper", {})
    assert result["ok"] is False
    assert runtime._motion_scope.context is None
    assert runtime.arm is arm
    assert runtime._motion_t0 is None
    assert arm.moves == arm.gripper == []


@pytest.mark.parametrize("cartesian", [False, True])
@pytest.mark.parametrize("has_segment", [False, True])
def test_spatial_route_returns_home_only_after_execution(tmp_path, cartesian, has_segment):
    from cascade.skills.spatial_moves import Route, Segment, execute_route

    runtime, arm, _ = _runtime(tmp_path, q=np.ones(7) * .1)
    runtime.kin = SimpleNamespace(fk=lambda q: np.eye(4))
    target = np.ones(7) * .2
    arm.move_cartesian = lambda pose, **kw: arm.move_joints(target, **kw)
    runtime.held_object = "cup"
    segments = [Segment(arm.q.copy(), target, np.eye(4))] if has_segment else []

    def action():
        execute_route(runtime, Route(segments, cartesian), 2.0,
                      tool="move_relative", words="left", record=False)
        return {"moved": has_segment}

    runtime.skill_move_relative = action
    result = runtime.execute("move_relative", {})

    assert result["ok"] is True
    assert arm.gripper == []
    assert runtime.held_object == "cup"
    if has_segment:
        assert result["return_home"]["status"] == "returned"
        assert len(arm.moves) == 2
        np.testing.assert_array_equal(arm.moves[-1], np.zeros(7))
    else:
        assert "return_home" not in result
        assert arm.moves == []


def test_tool_descriptions_explain_saved_home_without_changing_generic_catalog(tmp_path):
    runtime, _, _ = _runtime(tmp_path)
    configured = {spec["name"]: spec for spec in tool_specs_for_config(runtime.cfg)}
    generic = {spec["name"]: spec for spec in TOOL_SPECS}
    assert "returns to the saved operator camera-view home" in configured["grasp_object"]["description"]
    assert "STOP, cancellation" in configured["rotate_joint"]["description"]
    assert "saved operator camera-view home" not in generic["grasp_object"]["description"]
    assert "overview pose meant by 'home'" in configured["move_home"]["description"]
    assert "preserving the gripper" in configured["move_home"]["description"]
    assert "safe_to_power_off remains false" in configured["move_to_rest"]["description"]
    assert "folded rest pose" in generic["move_home"]["description"]


@pytest.mark.parametrize('close_outcome', ['success', 'miss', 'fault', 'stop', 'retreat_failure'])
def test_franka_contact_close_and_retreat_before_home(tmp_path, monkeypatch, close_outcome):
    from cascade.types import Grasp, GraspFailed, Detection
    runtime, arm, events = _runtime(tmp_path)
    runtime.cfg._data['arm']['type'] = 'ros2'
    runtime.cfg._data['arm']['ros_gripper_grasp_action'] = 'franka_gripper/grasp'
    runtime.cfg._data['arm']['gripper']['force_contact_close'] = True
    runtime.kin = SimpleNamespace(fk=lambda q: np.eye(4))
    pre, target = np.ones(7) * .2, np.ones(7) * .3
    g = Grasp(np.array([.5, 0., .03]), np.eye(3), .03, np.array([0., 0., -1.]))
    det = Detection('wood', .9, np.array([0., 0., 4., 4.]))
    fix = SimpleNamespace(detection=det, position=g.position)
    frame = runtime.camera.get_frame()
    runtime._localize = lambda *a, **kw: (frame, fix)
    runtime._plan_grasps = lambda *a, **kw: [g]
    monkeypatch.setattr('cascade.skills.runtime.select_grasp', lambda *a, **kw: (g, pre, target))
    runtime._gripper_width_frac = lambda: .178  # 14.24 mm actual object
    original_gripper = arm.set_gripper
    def close(pos, effort):
        original_gripper(pos, effort)
        if pos != 0.:
            return
        if close_outcome in ('miss', 'retreat_failure'):
            raise GraspFailed('native grasp did not secure object')
        if close_outcome in ('fault', 'stop'):
            arm.raw._stopped = True
            raise RuntimeError('driver failure or operator stop')
    arm.set_gripper = close
    if close_outcome == 'retreat_failure':
        arm.move_outcomes[3] = False
    result = runtime.execute('grasp_object', {'label': 'wood'})
    assert arm.gripper == [.04, 0.]  # one force-controlled close; no release
    if close_outcome in ('success', 'miss'):
        assert len(arm.moves) == 4
        for actual, expected in zip(arm.moves, [pre, target, pre, np.zeros(7)]):
            np.testing.assert_array_equal(actual, expected)
        assert result['return_home']['status'] == 'returned'
        assert result['ok'] is (close_outcome == 'success')
        if close_outcome == 'success':
            assert runtime.held_object == 'wood'
        else:
            assert runtime._held_provisional[0] == 'wood'
    else:
        assert len(arm.moves) == (3 if close_outcome == 'retreat_failure' else 2)
        assert result['return_home']['status'] == 'skipped'
        assert arm.raw._stopped
