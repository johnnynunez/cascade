"""A3's operator-halt rule composed with main's perception PRs (#340-#347).

A3 (option C, PR-A): an operator halt raised at any point of a motion skill
ends it, refusing its next motion before the first command, and a halted
skill never reports ok. Main: #341's bounded next-candidate retry re-enters
`skill_grasp_object` inside ONE call, #342/#344's FAILED line exactly once on
every failure and the move_home step only after a counted arm motion with no
halt, no e-stop and nothing held, #343's perception stops that end a grasp
loop at once.

Composed rules pinned here:
- a halt between two #341 candidates ends the skill: 0 arm and 0 jaw commands
  after it, the FAILED line once, no move_home step, also when the operator
  withdrew the halt again (Resume) before the next candidate would start;
- an air grasp on the candidate the retry moved on to still runs A3's exit
  withdrawal from THAT candidate's hover, and the result reports both the
  exit and #341's selection record;
- a halted skill is never ok: true, and its result carries T5's FAILED line
  once and no move_home step;
- a perception stop (#343) that follows an operator halt ends the loop as the
  halt (A3's `_end_on_halt`), not as the perception refusal.

Mock stack only (no hardware, no GPU, a held dead GraspGen-X port).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import needs_pin
from test_agent_results_compose import (
    FAILED,
    MOVE_HOME_FIRST,
    _failed_once,
    _gate_on,
    _mock_stack,
    _seen,
    _turned_candidates,
)

from cascade.skills.runtime import SkillRuntime, SlowPerceptionError

pytestmark = needs_pin

AIR = "air grasp: gripper closed fully, object not held"
SKILL_HALT = "MotionHalted: halt received during route planning or execution"
ARGS = {
    "grasp_object": {"label": "red cube"},
    "pick_and_place": {"object": "red cube"},
    "handover": {"label": "red cube"},
}


@pytest.fixture
def stack(monkeypatch, tmp_path):
    with _mock_stack(monkeypatch, tmp_path) as (runtime, arm):
        yield runtime, arm


def _jaw_commands(arm, monkeypatch):
    """Every jaw command the raw mock gripper received."""
    sent, real = [], arm.set_gripper

    def spy(pos, effort=1.0):
        sent.append(pos)
        return real(pos, effort)

    monkeypatch.setattr(arm, "set_gripper", spy)
    return sent


def _outer_grasps(runtime, monkeypatch):
    """Calls of grasp_object from the skill and its persistence loops (the
    #341 re-entry calls the class attribute and is not counted)."""
    calls, real = [], runtime.skill_grasp_object

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(runtime, "skill_grasp_object", spy)
    return calls


# ── a halt between two #341 candidates ────────────────────────────────────


@pytest.mark.parametrize("withdrawn", [False, True], ids=["halt pending", "halt withdrawn"])
@pytest.mark.parametrize("skill", sorted(ARGS))
def test_a_halt_between_two_candidates_ends_the_skill(stack, monkeypatch, skill, withdrawn):
    runtime, arm = stack
    gate = _gate_on(monkeypatch, arm, refuse=1)       # the first candidate is vetoed before its close
    _seen(runtime)
    arm.object_stop_frac = 0.5                        # the next one would hold the cube
    _turned_candidates(runtime, 3)
    harness = runtime.arm.harness
    jaws = _jaw_commands(arm, monkeypatch)
    grasps = _outer_grasps(runtime, monkeypatch)
    real_next, at_halt = SkillRuntime._grasp_next_observed_candidate, []

    def halt_then_next(self, *a, **k):
        harness.halt("operator")
        if withdrawn:
            harness.clear_halt()                      # the operator's Resume
        at_halt.append((len(arm.commands), len(jaws), runtime._arm_motions_counted()))
        return real_next(self, *a, **k)

    monkeypatch.setattr(SkillRuntime, "_grasp_next_observed_candidate", halt_then_next)

    result = runtime.execute(skill, ARGS[skill])

    assert len(at_halt) == 1, "premise: the first candidate withdrew and the retry was reached"
    assert (len(arm.commands), len(jaws), runtime._arm_motions_counted()) == at_halt[0], \
        "nothing moved after the halt"
    assert gate.closes == 1, "the next candidate never reached its close"
    assert result["error"] == SKILL_HALT, result
    _failed_once(result)
    text = json.dumps(result, default=str)
    assert MOVE_HOME_FIRST not in text and not result.get("arm_left_in_place"), result
    assert len(grasps) == 1, "no persistence retry after the halt"
    assert harness.halted == (None if withdrawn else "operator"), "nothing cleared a pending halt"
    assert not harness.estopped


# ── an air grasp after #341 moved on to the next candidate ────────────────


def test_an_air_grasp_on_the_next_candidate_runs_the_exit_from_its_hover(stack, monkeypatch):
    runtime, arm = stack
    _gate_on(monkeypatch, arm, refuse=1)
    _seen(runtime)
    arm.object_stop_frac = None                       # nothing stops the jaws: an air grasp
    _turned_candidates(runtime, 3)
    harness = runtime.arm.harness
    generation = harness._halt_generation
    real_refusal, withdrawn_at = SkillRuntime._observed_next_candidate_refusal, []

    def refusal(self, *a, **k):
        withdrawn_at.append(np.asarray(arm.get_state().q, dtype=float).copy())
        return real_refusal(self, *a, **k)

    monkeypatch.setattr(SkillRuntime, "_observed_next_candidate_refusal", refusal)
    real_exit, exits = runtime._air_grasp_exit, []

    def exit_spy(column, move, **kw):
        def watched(q):
            exits[-1]["exempt"].append(harness._grasp_exempt)
            return move(q)

        exits.append({"column": column, "kw": kw, "exempt": [],
                      "q": np.asarray(arm.get_state().q, dtype=float).copy()})
        out = real_exit(column, watched, **kw)
        exits[-1]["after"] = np.asarray(arm.get_state().q, dtype=float).copy()
        return out

    monkeypatch.setattr(runtime, "_air_grasp_exit", exit_spy)

    result = runtime.execute("grasp_object", ARGS["grasp_object"])

    assert result["ok"] is False and result["error"] == AIR, result
    assert result["exit_withdrawal"] == "completed (+3 cm)", result
    attempts = result["observed_selection"]["attempts"]
    assert [a["candidate"] for a in attempts] == [0, 1]
    assert attempts[0]["outcome"].startswith("refused before contact: closing fingers")
    assert attempts[1]["outcome"] == "air grasp"
    (only,) = exits
    (first_hover,) = withdrawn_at
    # The exit starts from the SECOND candidate's hover (the arm sits there),
    # never from the first one's, with the grasp's original halt baseline.
    assert np.allclose(only["kw"]["hover_q"], only["q"], atol=0.03)
    assert not np.allclose(only["kw"]["hover_q"], first_hover, atol=0.03)
    assert only["kw"]["halt_generation"] == generation
    # The observed-finger approach entered no column: the exit gets none and
    # every lift runs under strict checks, nothing left exempt afterwards.
    assert only["column"] is None and only["exempt"] == [None]
    assert harness._grasp_exempt is None
    pre, post = (runtime.kin.fk(only["q"])[:3, 3], runtime.kin.fk(only["after"])[:3, 3])
    assert post[2] >= pre[2] + .03 - 2e-3 and np.linalg.norm(post[:2] - pre[:2]) < 2e-3
    # main's agent text: FAILED once; the arm moved and no halt: move_home once.
    action = _failed_once(result)
    text = json.dumps(result, default=str)
    assert text.count(MOVE_HOME_FIRST) == 1 and action.index(FAILED) < action.index(MOVE_HOME_FIRST)


# ── a halted skill is never ok, and says so the T5 way ────────────────────


@pytest.mark.parametrize("withdrawn", [False, True], ids=["halt pending", "halt withdrawn"])
def test_a_halted_skill_is_not_ok_and_says_failed_once(stack, monkeypatch, withdrawn):
    """STOP right after the release of a pick_and_place (the skill swallows
    the refused home): not ok, the halt as its error, the FAILED line once
    and first, no move_home step."""
    runtime, arm = stack
    arm.object_stop_frac = 0.5                        # held
    _seen(runtime)
    harness = runtime.arm.harness
    real_place, at_halt = runtime.skill_place_at, []

    def place(*a, **k):
        out = real_place(*a, **k)
        harness.halt("operator")
        if withdrawn:
            harness.clear_halt()
        at_halt.append(len(arm.commands))
        return out

    monkeypatch.setattr(runtime, "skill_place_at", place)

    result = runtime.execute("pick_and_place", ARGS["pick_and_place"])

    assert at_halt and len(arm.commands) == at_halt[0], "nothing moved after the halt"
    assert result["ok"] is False and result["error"] == (
        SKILL_HALT if withdrawn else "MotionHalted: halted: operator"), result
    assert result["picked"] == "red cube", "what the skill did is kept"
    action = _failed_once(result)
    assert SKILL_HALT.split(": ", 1)[1] in action or "halted: operator" in action
    assert MOVE_HOME_FIRST not in json.dumps(result, default=str)


# ── a perception stop after an operator halt ──────────────────────────────


@pytest.mark.parametrize("withdrawn", [False, True], ids=["halt pending", "halt withdrawn"])
@pytest.mark.parametrize("skill", ["pick_and_place", "handover"])
def test_a_perception_stop_after_a_halt_ends_the_loop_as_the_halt(stack, monkeypatch, skill, withdrawn):
    runtime, arm = stack
    _seen(runtime)
    harness = runtime.arm.harness
    motions = runtime._arm_motions_counted()

    def grasp(*_a, **_k):
        # The operator halts while the grasp localizes; the analysis then
        # expires (a perception stop of #343, no motion of its own).
        harness.halt("operator")
        if withdrawn:
            harness.clear_halt()
        raise SlowPerceptionError("test: the analysed frame is 6.0s old (limit 5s)")

    monkeypatch.setattr(runtime, "skill_grasp_object", grasp)

    result = runtime.execute(skill, ARGS[skill])

    expected = ("MotionHalted: halt received during the grasp; not retrying" if withdrawn
                else "MotionHalted: motion halted: operator; not retrying")
    assert result["error"] == expected, result
    assert "localization analysis expired" not in json.dumps(result, default=str)
    _failed_once(result)
    assert MOVE_HOME_FIRST not in json.dumps(result, default=str)
    assert runtime._arm_motions_counted() == motions, "no automatic home or park"
