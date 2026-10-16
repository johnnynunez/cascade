"""T3's honest verdicts and T5's agent instructions compose.

Every failed result carries the FAILED line exactly once, whatever produced
it: T3's UNVERIFIED "no grasp was attempted", T1's two-phase SceneChangedError
/ ReanchorError and SlowPerceptionError. A success never does. The move_home
step follows the arm motions T3 counts: a grasp that moved nothing (perception
refused it, or its only motion was refused before its first command) never
tells the agent the arm stays at a refused grasp pose.
"""

from __future__ import annotations

import json
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from owned_server import held_dead_port

from cascade.agent.effects import UNVERIFIED
from cascade.types import SafetyViolation, SkillError

pytestmark = needs_pin

FAILED = "This attempt FAILED"
MOVE_HOME_FIRST = "call move_home before any other grasp"


@pytest.fixture
def mock_runtime(monkeypatch, tmp_path):
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
}


# ── T3 + T1 refusals before any motion: FAILED once, no move_home step ──


@pytest.mark.parametrize("skill", ["grasp_object", "pick_and_place", "handover"])
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
    assert evidence.startswith("no open command completed" if skill == "handover" else "no grasp was attempted")


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
