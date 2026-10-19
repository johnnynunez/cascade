"""An operator stop teaches the ASPIRE library nothing.

A stop -- a halt, an e-stop or both (the e-stop then names itself) -- is the
operator's decision, not how a primitive breaks: a confirmed retry after one
is no repair, so a call that ended in a stop is never the failure a retry is
credited to (`diagnose` skips it when it chooses) and its diagnosis is never
teachable (A8 follow-ups 1). That holds even when main files the call under
the failure the skill had decided before the stop (#377: `skill_error`, the
error naming the stop first): learning across an operator stop risks notes
that work around stops (A8 follow-ups 2). An earlier real failure in the same
run, before a stop, is still learned.

Fixtures are synthetic traces with the runtime's real error texts, and real
trace rows from `execute()` on the mock stack for a stop the error does not
name: with the sim retry ledger (the observed-finger gate on, the kitchen's
setting) the row is marked `context.sim_retry.ended_in_stop` and never learned
(A8 follow-ups fix, Important 1); without one (gate 0, every hardware profile)
no row carries that mark, so harvest skips such a run altogether (#388 re-review
Minor 1; the synthetic runs here carry a live gate-1 run record).
"""
import json

import pytest
from conftest import needs_pin
from test_agent_results_compose import _seen
from test_sim_retry_runtime import _nudged, _stack

from cascade.agent.aspire import Diagnosis, diagnose, distil, ended_in_stop, harvest, retrieve
from cascade.agent.effects import CONFIRMED, POSTCONDITIONS
from cascade.memory.envelope import failure_signature
from cascade.skills.library import SkillLibrary
from cascade.types import SafetyViolation

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
#: what the operator's e-stop leaves, alone or with a halt (both stops: the e-stop
#: wins and names itself, tests/test_retry_loops_end_on_halt.py "both stops")
ESTOPS = {
    "e-stop": "SafetyViolation: e-stop latched",
    "both stops": "SafetyViolation: e-stop latched; not retrying",
    "e-stop, view-home skipped": "return to saved home skipped: e-stop latched",
    # filed by its "jaws" (geometry:object_too_wide), still the e-stop's text
    "e-stop, jaws closed": "SafetyViolation: e-stop latched before the closed jaws were opened",
}
#: a failure the skill had decided before the operator halted it (#377): the error
#: names the halt (first), `skill_error` keeps the decided failure, which files it
MISSED = "grasp missed: closed on air"
DECIDED_THEN_HALTED = {
    # the halt stopped the recovery or the retry after the miss (`_keep_decided_failure`)
    "halted after a miss": ("MotionHalted: halted: operator: stop", MISSED),
    # the halt stopped the final park (`_stop_first`)
    "final park halted": (f"final park halted: operator: stop; {PICK_MISSED}", PICK_MISSED),
    # the halt stopped the view-home return (`_stop_first`)
    "view-home halted": (f"return to saved home halted: operator: stop; {PICK_MISSED}", PICK_MISSED),
}


def _confirmed(skill="pick_and_place"):
    return {"ok": True, "postcondition": {
        "skill": skill, "kind": POSTCONDITIONS[skill], "status": CONFIRMED, "channel": "physics",
        "evidence": "Synthetic fixture: object moved to the destination", "measured": {"moved_m": 0.2}}}


def _call(result, skill="pick_and_place", args=ARGS):
    return {"skill": skill, "args": dict(args), "context": dict(CONTEXT), "result": result}


def _failed(error, **extra):
    return _call({"ok": False, "error": error, **extra})


#: the run record of a live kitchen sim run at gate 1: harvest reads only live runs with the ledger (#388
#: re-review Minor 1, #393 follow-up)
LEDGER_ON = {"sim_retry": True, "purpose": "live"}


def _run(root, name, task, records, record=LEDGER_ON):
    run = root / name
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (run / "summary.txt").write_text(f"task: {task}\nsuccess: true")
    (run / "run.json").write_text(json.dumps(record))
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
    # the same diagnosis of a real failure is teachable (its error names no stop: A8 follow-ups 2)
    diag.signature, diag.error = "contact:air_grasp", "air grasp: gripper closed fully, object not held"
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


@pytest.mark.parametrize("kind", sorted(ESTOPS))
def test_a_retry_after_an_estop_or_both_stops_is_never_learned(tmp_path, kind):  # A8 follow-ups 1
    """Two tasks, each an e-stopped (or both-stopped) pick_and_place and then a
    confirmed retry of the same goal: diagnosed, nothing learned, promoted or served
    (main wrote `pick-and-place-safety-estop.md`)."""
    runs = tmp_path / "runs"
    for name, task in (("run-a", TASK_A), ("run-b", TASK_B)):
        _run(runs, name, task, [_failed(ESTOPS[kind]), _call(_confirmed())])
    library = SkillLibrary(tmp_path / "lib")

    out = harvest(runs, library)

    assert out["diagnosed"] == 2 and out["learned"] == 0, out
    assert out["promoted"] == [] and out["candidates"] == [], out
    assert library.records() == [] and list((tmp_path / "lib").glob("*.md")) == []
    assert retrieve(library, QUERY) == ""
    for name in ("run-a", "run-b"):
        diag = diagnose(runs / name)
        assert diag.error == ESTOPS[kind] and not diag.teachable


@pytest.mark.parametrize("kind", sorted(DECIDED_THEN_HALTED))
def test_a_decided_failure_that_ended_in_a_halt_is_never_learned(tmp_path, kind):  # A8 follow-ups 2
    """The skill missed, then the operator halted what came next (the retry, the
    final park, the view-home return): main files the call by the miss
    (`skill_error`), and it stays filed so, but a confirmed retry after it is never
    credited to it: the call ended in an operator stop. Two tasks learn nothing."""
    error, decided = DECIDED_THEN_HALTED[kind]
    runs = tmp_path / "runs"
    for name, task in (("run-a", TASK_A), ("run-b", TASK_B)):
        _run(runs, name, task, [_failed(error, skill_error=decided), _call(_confirmed())])
    library = SkillLibrary(tmp_path / "lib")

    out = harvest(runs, library)

    assert out["diagnosed"] == 2 and out["learned"] == 0, out
    assert out["promoted"] == [] and out["candidates"] == [], out
    assert library.records() == [] and retrieve(library, QUERY) == ""
    for name in ("run-a", "run-b"):
        diag = diagnose(runs / name)
        assert diag.signature == failure_signature({"error": decided}) != "safety:halt"  # main's filing kept
        assert diag.error == error and not diag.teachable


#: the contract when only the signature names the stop (a stop text in `skill_error` alone)
FILED_AS_A_STOP = {
    "filed as the halt": (PICK_MISSED, "MotionHalted: halted: operator: stop"),
    "filed as the e-stop": (PICK_MISSED, "SafetyViolation: e-stop latched"),
}
STOPPED_CALLS = {**{f"halt {i}": (halt, None) for i, halt in enumerate(HALTS)},
                 **{kind: (estop, None) for kind, estop in ESTOPS.items()},
                 **DECIDED_THEN_HALTED, **FILED_AS_A_STOP}


@pytest.mark.parametrize("kind", sorted(STOPPED_CALLS))
def test_ended_in_stop_reads_the_error_and_the_signature(kind):  # A8 follow-ups 1, 2
    error, decided = STOPPED_CALLS[kind]
    result = {"ok": False, "error": error} if decided is None else {"ok": False, "error": error, "skill_error": decided}
    assert ended_in_stop(result)


@pytest.mark.parametrize("result", [
    {"ok": False, "error": "SafetyViolation: link/joint 7 would hit the table"},
    {"ok": False, "error": PICK_MISSED},
    {"ok": False, "error": "air grasp: gripper closed fully, object not held"},
    {"ok": False, "error": "SafetyViolation: finger trajectory intersects observed surface: lemon"},
    {"ok": False, "error": "unknown skill halt_motion_x"},  # \"halt\" inside a name is no halt
    {"ok": False, "error": ""}, {"ok": False}, None, "MotionHalted: halted: operator: stop",
])
def test_a_failure_no_stop_ended_is_no_stopped_call(result):
    assert not ended_in_stop(result)


@pytest.mark.parametrize("kind", sorted(STOPPED_CALLS))
def test_a_stop_diagnosis_is_never_teachable(tmp_path, kind):  # A8 follow-ups 1, 2
    """Even a diagnosis that meets every other condition (a matching,
    measured-confirmed retry of the same goal and context), whatever signature
    its call is filed under: the call ended in an operator stop."""
    error, decided = STOPPED_CALLS[kind]
    diag = _diagnosis(error=error, signature=failure_signature({"error": error, "skill_error": decided}))
    assert not diag.teachable
    assert distil(diag, SkillLibrary(tmp_path / "lib")) is None
    # the same diagnosis of a failure that ended in no stop is teachable
    real = "SafetyViolation: link/joint 7 would hit the table"
    assert _diagnosis(error=real, signature=failure_signature({"error": real})).teachable


@pytest.mark.parametrize("kind", sorted(STOPPED_CALLS))
def test_an_earlier_real_failure_is_learned_past_a_stopped_call(tmp_path, kind):  # A8 follow-ups 1, 2
    """The run's real failure, then a call that ended in a stop, then a confirmed
    retry: the retry is credited to the real failure (as past a halt), never to the
    stopped call."""
    error, decided = STOPPED_CALLS[kind]
    real = _failed("SafetyViolation: link/joint 7 would hit the table")
    stopped = _failed(error) if decided is None else _failed(error, skill_error=decided)
    run = _run(tmp_path / "runs", "run-a", TASK_A, [real, stopped, _call(_confirmed())])

    diag = diagnose(run)

    assert diag.n_failures == 2
    assert diag.signature == "geometry:link_below_table" and diag.error == real["result"]["error"]
    assert diag.teachable


def _diagnosis(*, error, signature):
    confirmed = _confirmed()
    return Diagnosis(run="run-a", task=TASK_A, failed_skill="pick_and_place", signature=signature, error=error,
                     failed_args=dict(ARGS), failed_context=dict(CONTEXT), repair_skill="pick_and_place",
                     repair_args=dict(ARGS), repair_context=dict(CONTEXT),
                     repair_postcondition=confirmed["postcondition"], repaired=True)


#: a trace row's context with the ledger's mark (`_sim_retry_finish`): a stop the error does not name
MARKED = {**CONTEXT, "sim_retry": {"ended_in_stop": True}}
REFUSAL_ERROR = "SafetyViolation: finger trajectory intersects observed surface: lemon"


def test_aspire_reads_the_taxonomys_own_estop_words():  # A8 follow-ups fix, Minor 3
    from cascade.agent import aspire
    from cascade.memory import envelope

    assert aspire._ESTOP_WORDS.pattern == envelope.ESTOP_WORDS
    assert (envelope.ESTOP_WORDS, "safety:estop") in envelope._FAILURE_PATTERNS


def test_the_ledgers_mark_is_a_stopped_call_whatever_the_error_says():  # A8 follow-ups fix, Important 1
    assert ended_in_stop({"ok": False, "error": REFUSAL_ERROR}, MARKED)
    assert ended_in_stop(None, MARKED)
    for context in (None, CONTEXT, {"sim_retry": {"retry": True, "auto_home": False}},
                    {"sim_retry": {"ended_in_stop": "yes"}}, {"sim_retry": True}, "ended_in_stop"):
        assert not ended_in_stop({"ok": False, "error": REFUSAL_ERROR}, context), context


def test_a_diagnosis_of_a_marked_call_is_never_teachable(tmp_path):  # A8 follow-ups fix, Important 1
    """Even one built directly, with every other condition met: the failed call's
    context carries the ledger's mark."""
    diag = _diagnosis(error=REFUSAL_ERROR, signature=failure_signature({"error": REFUSAL_ERROR}))
    assert diag.teachable  # the refusal alone is a real failure
    diag.failed_context = dict(MARKED)
    assert not diag.teachable
    assert distil(diag, SkillLibrary(tmp_path / "lib")) is None


def test_a_marked_call_is_skipped_and_an_earlier_real_failure_is_learned(tmp_path):  # A8 follow-ups fix
    """A run with a marked call and a confirmed retry after it: the retry is never
    credited to the marked call; an earlier real failure is still learned past it."""
    marked = {**_failed(REFUSAL_ERROR), "context": dict(MARKED)}
    alone = diagnose(_run(tmp_path / "a", "run-a", TASK_A, [marked, _call(_confirmed())]))
    assert alone.error == REFUSAL_ERROR and not alone.repaired and not alone.teachable
    real = _failed("SafetyViolation: link/joint 7 would hit the table")
    past = diagnose(_run(tmp_path / "b", "run-a", TASK_A, [real, marked, _call(_confirmed())]))
    assert past.signature == "geometry:link_below_table" and past.teachable


# ── end to end: a stop the call's error does not name (A8 follow-ups fix, Important 1) ──

#: the observed-finger gate's own refusal: an error that names no stop
OWN_REFUSAL = "finger trajectory intersects observed surface: lemon"
#: the goal of each motion skill on the mock stack
STACK_ARGS = {"grasp_object": {"label": "red cube"}, "pick_and_place": {"object": "red cube"}}
#: (skill, stop) rows whose failed result names only the skill's own refusal (measured on the
#: mock stack: `aspire_stop_teach_probe`). "<stop>+own": the operator's stop during the call,
#: then the refusal; "e-stop reset": the e-stop and its reset, the arm moves again, then the
#: refusal; "<stop> before+own": the stop pending (halt) or latched (e-stop) before the call
#: starts, then the refusal before any motion
UNNAMED_STOPS = [("grasp_object", kind) for kind in (
    "halt+own", "halt withdrawn+own", "e-stop+own", "both+own", "e-stop reset+own", "e-stop reset",
    "halt before+own", "e-stop before+own")] + [("pick_and_place", kind) for kind in (
    "e-stop+own", "e-stop reset+own", "e-stop reset", "e-stop before+own")]
#: the same refusal with no stop: still learned, with the ledger or without
NO_STOP = [("grasp_object", "none"), ("pick_and_place", "none")]


def _stopped_grasp(runtime, kind):
    """A grasp that moves the arm (unless the stop came before the call), meets the
    operator's stop `kind`, then raises the gate's own refusal."""
    def grasp(*_a, **_k):
        harness = runtime.arm.harness
        stop = kind.split("+")[0]
        if not stop.endswith(" before"):
            assert runtime._move_joints(_nudged(runtime), duration_s=0.3)
        if stop in ("halt", "both", "halt withdrawn"):
            harness.halt("operator: stop")
        if stop in ("e-stop", "both", "e-stop reset"):
            harness.estop("operator e-stop")
        if stop == "halt withdrawn":
            harness.clear_halt()  # the operator's Resume
        if stop == "e-stop reset":
            harness.reset_estop()  # the operator's reset
        if kind == "e-stop reset":  # the arm moves again after the reset
            assert runtime._move_joints(_nudged(runtime, -0.05), duration_s=0.3)
        raise SafetyViolation(OWN_REFUSAL)
    return grasp


def _learn_from(rows, skill, root, record):
    """`rows` (one execute() call's trace) followed by a measured-confirmed retry of the
    same goal, as two runs of two tasks with the runtime's own run record: the last failed
    row's diagnosis, and harvest's result."""
    failed = [r for r in rows if r.get("skill") == skill and not (r.get("result") or {}).get("ok", True)]
    row = failed[-1]
    context = row.get("context") or {}
    repair = {"skill": skill, "args": dict(row.get("args") or {}),
              "context": {"arm": context.get("arm"), "held_object": context.get("held_object")},
              "result": _confirmed(skill)}
    runs = root / "runs"
    for name, task in (("run-a", TASK_A), ("run-b", TASK_B)):
        _run(runs, name, task, rows + [repair], record)
    return row, diagnose(runs / "run-a"), harvest(runs, SkillLibrary(root / "lib"))


@needs_pin
@pytest.mark.parametrize("gate", [0, 1], ids=["gate 0", "gate 1"])
@pytest.mark.parametrize(("skill", "kind"), UNNAMED_STOPS + NO_STOP)
def test_a_stop_the_error_does_not_name_is_never_learned_with_the_ledger(monkeypatch, tmp_path, gate, skill, kind):
    """The real trace row of a call an operator stop ended whose error names only the
    skill's own refusal: with the ledger (gate 1) the row is marked `ended_in_stop`, so a
    confirmed same-goal retry after it is never learned; without one (gate 0) no row
    carries a `sim_retry` mark. The same refusal with no stop is teachable at both gates,
    and learned at gate 1 only: harvest skips a run whose run.json shows no ledger (#388
    re-review Minor 1)."""
    with _stack(monkeypatch, tmp_path, gate=bool(gate)) as (runtime, arm):
        _seen(runtime)
        runtime.cfg._data["grasp"]["max_pick_attempts"] = 1
        runtime.skill_grasp_object = _stopped_grasp(runtime, kind)
        harness = runtime.arm.harness
        if kind.startswith("halt before"):
            harness.halt("operator: stop")
        if kind.startswith("e-stop before"):
            harness.estop("operator e-stop")
        res = runtime.execute(skill, dict(STACK_ARGS[skill]))
        assert (runtime.sim_retry is not None) is bool(gate)
        rows = runtime.trace.rows()
        record = json.loads((runtime.trace.run_dir / "run.json").read_text())
    assert res["ok"] is False and OWN_REFUSAL in res["error"], res
    assert record["sim_retry"] is bool(gate)
    row, diag, out = _learn_from(rows, skill, tmp_path, record)
    stopped = kind != "none"
    if not gate:
        assert all("sim_retry" not in (r.get("context") or {}) for r in rows), rows
        assert out["learned"] == 0 and out["diagnosed"] == 0 and out["skipped"] == 2, out  # no ledger: not harvested
        if stopped:
            return  # no ledger, no mark: only the harvest gate keeps it out
    else:
        assert ((row["context"].get("sim_retry") or {}).get("ended_in_stop") is True) is stopped, row
        assert out["skipped"] == 0 and out["learned"] == (0 if stopped else 1), out
    assert ended_in_stop(row["result"], row["context"]) is stopped
    assert diag.teachable is not stopped, diag
