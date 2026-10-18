"""An operator halt teaches the ASPIRE library nothing.

A halt is the operator's decision, not how a primitive breaks: a confirmed
retry after one is no repair, so a halt is never the failure a retry is
credited to (`diagnose` skips it when it chooses) and a halt diagnosis is
never teachable. An earlier real failure in the same run is still learned,
and so is a failed skill whose final park the operator halted afterwards:
it is filed by the skill's own error (`skill_error`).

Fixtures are synthetic traces with the runtime's real error texts.
"""
import json

import pytest

from cascade.agent.aspire import Diagnosis, diagnose, distil, harvest, retrieve
from cascade.agent.effects import CONFIRMED, POSTCONDITIONS
from cascade.skills.library import SkillLibrary

TASK_A = "pick the cube"
TASK_B = "stack the cube on the box"
QUERY = "grasp the cube"  # shares guard words with every note below
CONTEXT = {"arm": "left", "held_object": None}
ARGS = {"object": "cube"}
#: pick_and_place's error after one missed attempt
PICK_MISSED = "grasp failed after 1 attempts (0.4s): grasp missed: closed on air"
#: what the operator's halt leaves: of the skill itself, its route, or a
#: failed skill with no error of its own whose view-home return it skipped
HALTS = [
    "MotionHalted: halted: operator: stop",
    "MotionHalted: halt received during route planning or execution",
    "return to saved home skipped: motion halted: operator: stop",
]


def _confirmed(skill="pick_and_place"):
    return {"ok": True, "postcondition": {
        "skill": skill, "kind": POSTCONDITIONS[skill], "status": CONFIRMED, "channel": "physics",
        "evidence": "Synthetic fixture: object moved to the destination", "measured": {"moved_m": 0.2}}}


def _call(result, skill="pick_and_place", args=ARGS):
    return {"skill": skill, "args": dict(args), "context": dict(CONTEXT), "result": result}


def _failed(error, **extra):
    return _call({"ok": False, "error": error, **extra})


def _run(root, name, task, records):
    run = root / name
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (run / "summary.txt").write_text(f"task: {task}\nsuccess: true")
    return run


@pytest.mark.parametrize("halt", HALTS)
def test_a_retry_after_an_operator_halt_is_never_learned(tmp_path, halt):
    """Two tasks, each a halted pick_and_place and then a confirmed retry of
    the same goal: the halt is still diagnosed, but nothing is learned,
    promoted or served to the agent."""
    runs = tmp_path / "runs"
    for name, task in (("run-a", TASK_A), ("run-b", TASK_B)):
        _run(runs, name, task, [_failed(halt), _call(_confirmed())])
    library = SkillLibrary(tmp_path / "lib")

    out = harvest(runs, library)

    assert out["diagnosed"] == 2 and out["learned"] == 0, out
    assert out["promoted"] == [] and out["candidates"] == [], out
    assert library.records() == []
    assert retrieve(library, QUERY) == ""
    for name in ("run-a", "run-b"):
        diag = diagnose(runs / name)
        assert diag.signature == "safety:halt" and diag.error == halt
        assert not diag.teachable


def test_a_halt_diagnosis_is_never_teachable(tmp_path):
    """Even a diagnosis that meets every other condition (a matching,
    measured-confirmed retry of the same goal and context)."""
    confirmed = _confirmed()
    diag = Diagnosis(run="run-a", task=TASK_A, failed_skill="pick_and_place", signature="safety:halt",
                     error=HALTS[0], failed_args=dict(ARGS), failed_context=dict(CONTEXT),
                     repair_skill="pick_and_place", repair_args=dict(ARGS), repair_context=dict(CONTEXT),
                     repair_postcondition=confirmed["postcondition"], repaired=True)
    assert not diag.teachable
    assert distil(diag, SkillLibrary(tmp_path / "lib")) is None
    # the same diagnosis of a real failure is teachable
    diag.signature = "contact:air_grasp"
    assert diag.teachable


@pytest.mark.parametrize("order", ["halt_after_the_repair", "halt_before_the_retry", "halt_first_no_retry"])
def test_an_earlier_real_failure_is_learned_past_a_halt(tmp_path, order):
    """The run's real failure, not a halt beside it, is the one a confirmed
    retry is credited to (a halt with its own confirmed retry later in the
    run, or a halt between the failure and its retry), and the one a run
    without a confirmed retry reports."""
    real = _failed("SafetyViolation: link/joint 7 would hit the table")
    if order == "halt_after_the_repair":
        records = [real, _call(_confirmed()), _failed(HALTS[0]), _call(_confirmed())]
    elif order == "halt_before_the_retry":
        records = [real, _failed(HALTS[0]), _call(_confirmed())]
    else:
        records = [_failed(HALTS[0]), real]
    run = _run(tmp_path / "runs", "run-a", TASK_A, records)

    diag = diagnose(run)

    assert diag.n_failures == 2
    assert diag.signature == "geometry:link_below_table" and diag.error == real["result"]["error"]
    assert diag.teachable is (order != "halt_first_no_retry")


def test_a_failure_whose_final_park_was_halted_is_learned_as_that_failure(tmp_path):
    """The skill missed, then the operator halted its final park: the miss
    came first and is what a retry repaired. Two tasks promote it under the
    miss's signature; the note the agent gets never names the halt as the
    failure."""
    runs = tmp_path / "runs"
    composite = f"final park halted: operator: stop; {PICK_MISSED}"
    for name, task in (("run-a", TASK_A), ("run-b", TASK_B)):
        _run(runs, name, task, [_failed(composite, skill_error=PICK_MISSED), _call(_confirmed())])
    library = SkillLibrary(tmp_path / "lib")

    out = harvest(runs, library)

    assert out["promoted"] == ["pick-and-place-other-grasp-failed-after"], out
    served = retrieve(library, QUERY)
    assert "after **other:grasp_failed_after**" in served, served
    assert "**safety:halt**" not in served, served
