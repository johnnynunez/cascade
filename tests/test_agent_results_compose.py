"""T3's honest verdicts and T5's agent instructions compose.

Every failed result carries the FAILED line exactly once, whatever produced
it: T3's UNVERIFIED "no grasp was attempted", T1's two-phase SceneChangedError
/ ReanchorError and SlowPerceptionError. A success never does. The move_home
step follows the arm motions T3 counts: a grasp that moved nothing (perception
refused it, or its only motion was refused before its first command) never
tells the agent the arm stays at a refused grasp pose, with the observed-
finger gate off or on (the Spark default), through every composite that
passes the step on. #341's next-candidate retry runs inside ONE call: a retry
that grasps carries neither line, a grasp whose every candidate was refused
carries each once.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from owned_server import held_dead_port

from cascade.agent.effects import UNVERIFIED
from cascade.grasping import observed_scene
from cascade.types import SafetyViolation, SkillError

pytestmark = needs_pin

FAILED = "This attempt FAILED"
MOVE_HOME_FIRST = "call move_home before any other grasp"


GATE = ("gate off", "gate on")


class _GateDouble:
    """The observed-finger gate's interface on the mock stack (no Isaac
    capture): every vet passes; the first `refuse` live closing checks veto
    BEFORE the jaw command, tagged exactly as the real gate tags them
    (`precommand_refusal`), so #341's next-candidate retry runs for real."""

    closing_lower = closing_upper = (0.0, 0.0)

    def __init__(self, refuse=0):
        self.refuse, self.closes = refuse, 0

    def feedback(self, state):
        return None

    def pose(self, q):
        return None

    def closing_pose(self, q):
        return None

    def occluded_pose(self, q, *, closing=False, check=None):
        return None

    def profile(self, *_a, **_k):
        return None

    def require_profile(self, *_a, **_k):
        return None

    def require_closing(self, state, *, check=None):
        self.closes += 1
        if self.closes <= self.refuse:
            raise observed_scene.precommand_refusal(
                "closing fingers intersects observed non-target surface",
                {"finger": "joint_left", "surface": "other observed surface", "injected": True})


def _gate_on(monkeypatch, arm, refuse=0):
    """CASCADE_OBSERVED_FINGER_GATE=1 on the mock stack: the gate double, and a
    mock stream that runs the gate's feedback guard (MockArm.stream_to takes
    preflight/before_stream itself; other backend hints are dropped, as SafeArm
    does for a backend that lacks them)."""
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    gate = _GateDouble(refuse)
    monkeypatch.setattr(observed_scene, "for_runtime", lambda *_a: gate)
    stream = arm.stream_to
    accepted = set(inspect.signature(stream).parameters)

    def guarded(q_target, duration_s, *, feedback_guard=None, **kw):
        if feedback_guard is not None:
            feedback_guard(arm.get_state())
        return stream(q_target, duration_s, **{k: v for k, v in kw.items() if k in accepted})
    monkeypatch.setattr(arm, "stream_to", guarded)
    return gate


@contextlib.contextmanager
def _mock_stack(monkeypatch, tmp_path):
    """The real mock stack; no GraspGen-X server is ever reached."""
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
        runtime, arm = build_runtime(cfg, tmp_path / "run")
        try:
            assert runtime.effects is not None, "the verifier must be wired"
            yield runtime, arm
        finally:
            shutdown_runtime(runtime, arm)


@pytest.fixture(params=GATE)
def mock_runtime(request, monkeypatch, tmp_path):
    """The mock stack with the observed-finger gate off, or on (the Spark
    default) with every vet passing."""
    with _mock_stack(monkeypatch, tmp_path) as (runtime, arm):
        if request.param == "gate on":
            _gate_on(monkeypatch, arm)
        yield runtime, arm


def _gated(request) -> bool:
    return request.node.callspec.params.get("mock_runtime") == "gate on"


def _seen(runtime, label="red cube"):
    runtime._reobserve()
    assert runtime.beliefs.find(label) is not None


def _slow_perception(monkeypatch):
    """Every analysed frame is past the age limit: the runtime's own
    SlowPerceptionError fires inside localization, before any motion."""
    import cascade.skills.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
                        partial(runtime_mod._analyzed_frame_age_s, max_age_s=-1.0))


def _failed_once(result) -> str:
    """The FAILED line comes first (after a STUCK relay prefix), once."""
    assert result["ok"] is False, result
    action = result["next_action"]
    if result.get("outcome") == "stuck":
        assert action.startswith("STUCK: relay `ask`"), action
        assert FAILED + ": " in action, action
    else:
        assert action.startswith(FAILED + ": "), action
    assert action.count(FAILED) == 1, action
    return action


def _nudged(runtime, dq=0.05):
    q = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    q[0] += dq
    return q


def _refuse_route(start, duration_s):
    raise SafetyViolation("test: route refused by its preflight; no motion sent")


ARGS = {
    "grasp_object": {"label": "red cube"},
    "pick_and_place": {"object": "red cube"},
    "handover": {"label": "red cube"},
    "throw": {"label": "red cube"},
}


# ── T3 + T1 refusals before any motion: FAILED once, no move_home step ──


@pytest.mark.parametrize("skill", ["grasp_object", "pick_and_place", "handover", "throw"])
def test_a_slow_perception_refusal_says_failed_once_and_asks_no_move_home(mock_runtime, monkeypatch, skill):
    runtime, arm = mock_runtime
    _seen(runtime)
    _slow_perception(monkeypatch)
    n0 = len(arm.commands)

    result = runtime.execute(skill, ARGS[skill])

    action = _failed_once(result)
    assert "Perception took" in action
    assert len(arm.commands) == n0, "refused before motion"
    assert MOVE_HOME_FIRST not in action, "the arm never left where the call found it"
    evidence = result["postcondition"]["evidence"]
    assert result["postcondition"]["status"] == UNVERIFIED
    released = skill in {"handover", "throw"}  # release kinds: T3 says no open completed
    assert evidence.startswith("no open command completed" if released else "no grasp was attempted")


def _red(label="cube", color="red", at=(0.36, 0.12, 0.02)):
    return SimpleNamespace(label=label, color=color, position=np.array(at))


@pytest.mark.parametrize("error", ["SceneChangedError", "ReanchorError"])
@pytest.mark.parametrize("skill", ["grasp_object", "pick_and_place", "handover", "sort_by_color"])
def test_a_two_phase_refusal_says_failed_once_and_asks_no_move_home(mock_runtime, monkeypatch, skill, error):
    import cascade.skills.runtime as runtime_mod
    from cascade.apps.mcp_server import _text_result

    runtime, arm = mock_runtime
    _seen(runtime)
    exc = getattr(runtime_mod, error)
    calls = []

    def refused(*_a, **_k):
        calls.append(1)
        raise exc(f"{error} test: the object was named, its fresh geometry refused")

    runtime.skill_grasp_object = refused
    if skill == "sort_by_color":
        monkeypatch.setattr(runtime.beliefs, "all", lambda: [_red()])
    args = ARGS.get(skill, {})

    result = runtime.execute(skill, args)

    action = _failed_once(result)
    assert MOVE_HOME_FIRST not in action, "no grasp motion: no move_home step"
    if error == "SceneChangedError":
        assert len(calls) == 1, "a scene change ends the attempt (T1/T3 rule)"
    # the MCP server's own error wrapper never adds a second FAILED line
    payload = json.loads(_text_result(result, is_error=True)["content"][0]["text"])
    assert payload["next_action"].count(FAILED) == 1


def _perception_refused_persistence(query, material=None, budget_s=None):
    """`_grasp_with_persistence`'s SceneChangedError return (T1/T3)."""
    return {"ok": False, "held": False, "home_skipped": True, "grasp_attempts": 1,
            "error": "SceneChangedError: scene changed during perception of " + repr(query),
            "note": "localization analysis expired; no automatic motion recovery"}


def test_a_restore_after_a_perception_refusal_asks_no_move_home():
    from test_pigey_scene_memory import A, BOUNDS, _stub, _two_cubes

    bs = _two_cubes(time.monotonic())
    rt = _stub(bs, bounds=BOUNDS)
    rt.skill_snapshot_scene(name="t")
    bs.find("red cube").position = np.array(A) + np.array([0.0, 0.08, 0.0])
    rt._grasp_with_persistence = _perception_refused_persistence
    out = rt.skill_restore_scene(name="t")
    assert out["ok"] is False and not out.get("home_skipped"), out
    assert MOVE_HOME_FIRST not in str(out.get("next_action"))
    assert rt.log.homes == 0, "T3: no automatic re-home after a perception refusal"


def test_a_search_after_a_perception_refusal_asks_no_move_home():
    from test_pigey_scene_memory import BOUNDS, _cluttered, _stub

    rt = _stub(_cluttered(time.monotonic()), bounds=BOUNDS)
    rt._grasp_with_persistence = _perception_refused_persistence
    out = rt.skill_search_for_object(label="green cube", max_occluders=1)
    assert out["ok"] is False and not out.get("home_skipped"), out
    assert MOVE_HOME_FIRST not in str(out.get("next_action"))
    assert rt.log.homes == 0


# ── the move_home step needs an arm motion T3 counted ───────────────────


@pytest.mark.parametrize("first", ["refused route", "gripper open"])
def test_a_grasp_that_moved_no_arm_asks_no_move_home(mock_runtime, first):
    """The call's motion scope saw a command (a route refused by its
    preflight, or the pre-grasp open), but no arm motion was counted: T3
    says "no grasp was attempted", so T5 must not say the arm stays at a
    refused grasp pose."""
    runtime, arm = mock_runtime
    _seen(runtime)

    def refused(*_a, **_k):
        if first == "refused route":
            runtime._move_joints(_nudged(runtime), duration_s=0.3, _preflight=_refuse_route)
        runtime._set_gripper(runtime._grip_open, effort=0.8)
        raise SkillError("no IK solution for any grasp candidate")

    runtime.skill_grasp_object = refused
    result = runtime.execute("grasp_object", {"label": "red cube"})

    action = _failed_once(result)
    assert MOVE_HOME_FIRST not in action
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED and pc["evidence"].startswith("no grasp was attempted: "), pc


def test_a_grasp_that_moved_the_arm_still_asks_for_move_home_after_the_failed_line(mock_runtime):
    runtime, arm = mock_runtime
    _seen(runtime)

    def refused(*_a, **_k):
        assert runtime._move_joints(_nudged(runtime), duration_s=0.3)
        raise SafetyViolation("finger trajectory intersects observed surface: test")

    runtime.skill_grasp_object = refused
    result = runtime.execute("grasp_object", {"label": "red cube"})

    action = _failed_once(result)
    assert action.count(MOVE_HOME_FIRST) == 1
    assert action.index(FAILED) < action.index(MOVE_HOME_FIRST)
    assert not result["postcondition"]["evidence"].startswith("no grasp was attempted"), result


def test_unknown_motion_counts_keep_the_motion_started_rule(mock_runtime, monkeypatch):
    """Without a counting harness (a test double arm) T5's own rule decides."""
    runtime, _arm = mock_runtime
    monkeypatch.setattr(runtime, "_actuation_counts", lambda: None)

    def refused(*_a, **_k):
        runtime._mark_motion_started()
        raise SafetyViolation("refused mid-approach")

    runtime.skill_grasp_object = refused
    result = runtime.execute("grasp_object", {"label": "red cube"})
    assert MOVE_HOME_FIRST in _failed_once(result)


# ── a success never carries the FAILED line ─────────────────────────────


@pytest.mark.parametrize("skill", ["grasp_object", "handover"])
def test_a_held_grasp_carries_no_failure_line(mock_runtime, skill):
    runtime, arm = mock_runtime
    _seen(runtime)
    arm.object_stop_frac = 0.5  # the jaws jam on the object: held

    result = runtime.execute(skill, ARGS[skill])

    assert result["ok"] is True, result
    text = json.dumps(result, default=str)
    assert FAILED not in text and MOVE_HOME_FIRST not in text


# ── the move_home step through every composite, gate off and on ─────────
#
# With the observed-finger gate on, pick_and_place and the persistence loop
# (handover, sort_by_color, _move_object_to -> restore_scene /
# search_for_object) skip the automatic home after a failed grasp. They may
# pass move_home on only when that grasp moved the arm (an arm motion the
# harness counted), never after a refused route, a gripper-only call, a
# SkillError before motion or a two-phase ReanchorError.


#: How the failed grasp ended. None of the first four moved the arm.
NO_ARM_MOTION = ("refused route", "gripper open", "SkillError", "ReanchorError")


def _failing_grasp(runtime, first):
    import cascade.skills.runtime as runtime_mod

    def refused(*_a, **_k):
        if first == "moved":
            assert runtime._move_joints(_nudged(runtime), duration_s=0.3)
            raise SafetyViolation("finger trajectory intersects observed surface: test")
        if first == "ReanchorError":
            raise runtime_mod.ReanchorError("ReanchorError test: the named object's fresh geometry was refused")
        if first == "refused route":
            runtime._move_joints(_nudged(runtime), duration_s=0.3, _preflight=_refuse_route)
        if first in ("refused route", "gripper open"):
            runtime._set_gripper(runtime._grip_open, effort=0.8)
        raise SkillError("no IK solution for any grasp candidate")
    return refused


COMPOSITES = {"pick_and_place": ARGS["pick_and_place"], "handover": ARGS["handover"], "sort_by_color": {}}


def _run_composite(runtime, monkeypatch, skill, first):
    _seen(runtime)
    runtime.cfg._data["grasp"]["max_pick_attempts"] = 1
    runtime.skill_grasp_object = _failing_grasp(runtime, first)
    if skill == "sort_by_color":
        monkeypatch.setattr(runtime.beliefs, "all", lambda: [_red()])
    return runtime.execute(skill, COMPOSITES[skill])


@pytest.mark.parametrize("first", NO_ARM_MOTION)
@pytest.mark.parametrize("skill", sorted(COMPOSITES))
def test_a_composite_whose_grasp_moved_no_arm_asks_no_move_home(mock_runtime, monkeypatch, request, skill, first):
    runtime, arm = mock_runtime
    result = _run_composite(runtime, monkeypatch, skill, first)
    action = _failed_once(result)
    assert MOVE_HOME_FIRST not in action, "no arm motion was counted: no move_home step"
    assert not result.get("arm_left_in_place"), result
    if skill == "pick_and_place" and _gated(request):
        assert result["home_skipped"] is True, "the unchecked automatic home stays skipped"


@pytest.mark.parametrize("first", NO_ARM_MOTION + ("moved",))
def test_the_persistence_loop_asks_for_move_home_only_after_a_counted_motion(mock_runtime, request, first):
    runtime, arm = mock_runtime
    _seen(runtime)
    runtime.cfg._data["grasp"]["max_pick_attempts"] = 1
    runtime.skill_grasp_object = _failing_grasp(runtime, first)
    res = runtime._grasp_with_persistence("red cube")
    assert res["ok"] is False
    left = _gated(request) and first == "moved"
    assert (MOVE_HOME_FIRST in str(res.get("next_action"))) is left, res
    assert bool(res.get("arm_left_in_place")) is left, res
    if _gated(request):
        assert res["home_skipped"] is True, "the unchecked automatic home stays skipped either way"


@pytest.mark.parametrize("mock_runtime", ["gate on"], indirect=True)
@pytest.mark.parametrize("skill", sorted(COMPOSITES) + ["grasp_object", "throw"])
def test_a_grasp_that_moved_the_arm_asks_for_move_home_once_with_the_gate_on(mock_runtime, monkeypatch, skill):
    runtime, arm = mock_runtime
    result = _run_composite(runtime, monkeypatch, skill, "moved") if skill in COMPOSITES else None
    if result is None:
        _seen(runtime)
        runtime.skill_grasp_object = _failing_grasp(runtime, "moved")
        result = runtime.execute(skill, ARGS[skill])
    action = _failed_once(result)
    assert json.dumps(result, default=str).count(MOVE_HOME_FIRST) == 1, result
    assert action.index(FAILED) < action.index(MOVE_HOME_FIRST)
    assert result["arm_left_in_place"] is True


def _stub_with_real_persistence(rt, first):
    """test_pigey_scene_memory's stub with the REAL `_grasp_with_persistence`
    (gate on); its grasp double counts an arm motion only when it moved."""
    del rt.__dict__["_grasp_with_persistence"]
    motions = [0]
    rt._actuation_counts = lambda: {"arm_motions": motions[0], "gripper_closes": 0, "gripper_opens": 0}

    def grasp(*_a, **_k):
        import cascade.skills.runtime as runtime_mod
        if first == "moved":
            motions[0] += 1
            raise SafetyViolation("finger trajectory intersects observed surface: test")
        if first == "ReanchorError":
            raise runtime_mod.ReanchorError("ReanchorError test: the named object's fresh geometry was refused")
        raise SkillError("no IK solution for any grasp candidate")
    rt.skill_grasp_object = grasp
    return rt


@pytest.mark.parametrize("first", ["SkillError", "ReanchorError", "moved"])
@pytest.mark.parametrize("composite", ["restore_scene", "search_for_object"])
def test_restore_and_search_pass_move_home_on_only_after_a_counted_motion(monkeypatch, composite, first):
    from test_pigey_scene_memory import A, BOUNDS, _cluttered, _stub, _two_cubes

    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    if composite == "restore_scene":
        bs = _two_cubes(time.monotonic())
        rt = _stub(bs, bounds=BOUNDS)
        rt.skill_snapshot_scene(name="t")
        bs.find("red cube").position = np.array(A) + np.array([0.0, 0.08, 0.0])
        _stub_with_real_persistence(rt, first)
        out = rt.skill_restore_scene(name="t")
    else:
        rt = _stub_with_real_persistence(_stub(_cluttered(time.monotonic()), bounds=BOUNDS), first)
        out = rt.skill_search_for_object(label="green cube", max_occluders=1)
    assert out["ok"] is False, out
    left = first == "moved"
    assert (MOVE_HOME_FIRST in str(out.get("next_action"))) is left, out
    assert bool(out.get("home_skipped")) is left and bool(out.get("arm_left_in_place")) is left, out
    assert rt.log.homes == 0, "never an unchecked automatic home with the gate on"


# ── no move_home step after an operator halt / e-stop or while holding ──
#
# execute()'s own rule never asks for move_home when the harness is halted or
# e-stopped (the next motion is the operator's call: a halt clears at the
# next motion) or when an object is held. The gate-on returns of
# pick_and_place and the persistence loop (handover, sort_by_color) apply the
# same checks, so an agent never moves the arm after an operator halt.


#: Why the failed grasp, which moved the arm, must not ask for move_home.
NO_HINT_AFTER = ("operator halt", "e-stop", "holding")


def _moved_then(runtime, arm, why):
    """A grasp double whose arm motion is counted, then: the operator halts
    the arm mid-grasp, the e-stop latches, the close took the object and the
    lift failed (`held_object` set), or the route was refused (the control)."""
    from cascade.types import MotionHalted

    def grasp(*_a, **_k):
        assert runtime._move_joints(_nudged(runtime), duration_s=0.3)
        harness = runtime.arm.harness
        if why == "operator halt":
            harness.halt("operator: test halt")
            raise MotionHalted("halt received during route planning or execution")
        if why == "e-stop":
            harness.estop("test e-stop")
            raise SafetyViolation("e-stop latched: test")
        if why == "holding":
            arm.object_stop_frac = 0.5  # the jaws stall on the object
            runtime._set_gripper(runtime._grip_closed, effort=0.8)
            runtime.held_object = "red cube"
            raise SafetyViolation("did not settle at grasp lift pose: test")
        raise SafetyViolation("finger trajectory intersects observed surface: test")
    return grasp


@pytest.mark.parametrize("why", NO_HINT_AFTER)
@pytest.mark.parametrize("skill", sorted(COMPOSITES) + ["grasp_object"])
def test_no_move_home_step_after_a_halt_or_while_holding(mock_runtime, monkeypatch, request, skill, why):
    runtime, arm = mock_runtime
    _seen(runtime)
    runtime.cfg._data["grasp"]["max_pick_attempts"] = 1
    runtime.skill_grasp_object = _moved_then(runtime, arm, why)
    if skill == "sort_by_color":
        monkeypatch.setattr(runtime.beliefs, "all", lambda: [_red()])
    n0 = runtime._arm_motions_counted()

    result = runtime.execute(skill, COMPOSITES.get(skill, ARGS.get(skill)))

    assert runtime._arm_motions_counted() > n0, "the failed grasp moved the arm"
    text = json.dumps(result, default=str)
    assert MOVE_HOME_FIRST not in text, result
    assert not result.get("arm_left_in_place"), result
    if result["ok"] is False:
        _failed_once(result)
    else:  # gate off, a held object goes on to be placed / handed over (B36)
        assert why == "holding" and not _gated(request), result


@pytest.mark.parametrize("why", NO_HINT_AFTER + ("refused",))
def test_the_persistence_loop_asks_no_move_home_after_a_halt_or_while_holding(mock_runtime, request, why):
    from cascade.types import MotionHalted

    runtime, arm = mock_runtime
    _seen(runtime)
    runtime.cfg._data["grasp"]["max_pick_attempts"] = 1
    runtime.skill_grasp_object = _moved_then(runtime, arm, why)
    if why == "operator halt":
        # The operator's halt ends the loop as itself (A3): no result, so no
        # move_home step either.
        with pytest.raises(MotionHalted, match="^halt received during route planning or execution$"):
            runtime._grasp_with_persistence("red cube")
        return
    res = runtime._grasp_with_persistence("red cube")
    left = _gated(request) and why == "refused"
    assert (MOVE_HOME_FIRST in str(res.get("next_action"))) is left, res
    assert bool(res.get("arm_left_in_place")) is left, res


@pytest.mark.parametrize("mock_runtime", ["gate on"], indirect=True)
@pytest.mark.parametrize("why", NO_HINT_AFTER + ("refused",))
def test_pick_and_place_gate_on_asks_move_home_only_without_halt_or_hold(mock_runtime, why):
    """The control row (a refused grasp that moved the arm) still gets the
    step exactly once, after the FAILED line."""
    runtime, arm = mock_runtime
    _seen(runtime)
    runtime.skill_grasp_object = _moved_then(runtime, arm, why)
    motions = runtime._arm_motions_counted()
    result = runtime.execute("pick_and_place", ARGS["pick_and_place"])
    action = _failed_once(result)
    text = json.dumps(result, default=str)
    if why == "operator halt":
        # The halt ends pick_and_place as itself (A3): execute()'s halt result,
        # no automatic home or park after the grasp double's own nudge.
        assert result["error"] == "MotionHalted: halt received during route planning or execution", result
        assert runtime._arm_motions_counted() == motions + 1, "only the grasp's own nudge moved"
    else:
        assert result["home_skipped"] is True, "the unchecked automatic home stays skipped"
    if why == "refused":
        assert text.count(MOVE_HOME_FIRST) == 1, result
        assert action.index(FAILED) < action.index(MOVE_HOME_FIRST)
        assert result["arm_left_in_place"] is True
    else:
        assert MOVE_HOME_FIRST not in text, result
        assert not result.get("arm_left_in_place"), result


# ── #341's next-candidate retry inside ONE grasp call ───────────────────


def _turned_candidates(runtime, n):
    """The OBB plan's executable grasp and n-1 copies turned about its
    approach in 30 deg steps: separate candidates (never near-copies of a
    refused one), same width."""
    from scipy.spatial.transform import Rotation

    plan = runtime._plan_grasps

    def planned(*a, **k):
        base = next(g for g in plan(*a, **k) if g.width_m <= runtime._max_width)
        axis = np.asarray(base.approach, float) / np.linalg.norm(base.approach)
        out = []
        for i in range(n):
            g = copy.deepcopy(base)
            g.rotation = Rotation.from_rotvec(axis * np.deg2rad(30. * i)).as_matrix() @ np.asarray(base.rotation, float)
            out.append(g)
        return out
    runtime._plan_grasps = planned


@pytest.fixture
def next_candidate(monkeypatch, tmp_path):
    """The mock stack, gate on, a plan of `candidates` grasps whose first
    `refuse` live closing checks are vetoed before the jaw command."""
    with _mock_stack(monkeypatch, tmp_path) as (runtime, arm):
        def arrange(*, refuse, candidates):
            gate = _gate_on(monkeypatch, arm, refuse)
            _seen(runtime)
            arm.object_stop_frac = 0.5  # the jaws jam on the object: held
            _turned_candidates(runtime, candidates)
            return runtime, arm, gate
        yield arrange


def test_a_next_candidate_that_grasps_carries_no_failed_line_and_no_move_home(next_candidate):
    runtime, arm, gate = next_candidate(refuse=1, candidates=3)
    result = runtime.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is True and result["held"] == "red cube", result
    selection = result["observed_selection"]
    assert [a["candidate"] for a in selection["attempts"]] == [0, 1]
    assert selection["attempts"][0]["outcome"].startswith("refused before contact: closing fingers")
    assert selection["succeeded_attempt"] == 2
    text = json.dumps(result, default=str)
    assert FAILED not in text and MOVE_HOME_FIRST not in text, result
    assert "arm_left_in_place" not in result


@pytest.mark.parametrize("skill", ["grasp_object", "pick_and_place", "handover"])
def test_every_candidate_refused_says_failed_once_and_move_home_once(next_candidate, skill):
    runtime, arm, gate = next_candidate(refuse=99, candidates=4)
    result = runtime.execute(skill, ARGS[skill])
    assert gate.closes == 3, "the first candidate and at most two more"
    action = _failed_once(result)
    text = json.dumps(result, default=str)
    # Once for the whole call, after the retry loop: never per attempt.
    assert text.count(FAILED) == 1 and text.count(MOVE_HOME_FIRST) == 1, result
    assert action.index(FAILED) < action.index(MOVE_HOME_FIRST)
    # #341's note about the refused earlier candidates stays in the error.
    assert "3 grasp candidates refused before contact and withdrew" in result["error"]
    assert "first refusal: closing fingers intersects observed non-target surface" in result["error"]
    assert result["arm_left_in_place"] is True
    if skill != "handover":
        assert [a["candidate"] for a in result["observed_selection"]["attempts"]] == [0, 1, 2]
