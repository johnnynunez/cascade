"""The sim retry ledger's own refusals in the trace teach nothing and cost nothing.

A call the ledger refused (a used retry, or a repeat after the operator stopped the
arm: `context.sim_retry.refused`, skills/runtime.py `_sim_retry_refusal`) ran no
primitive: execute() returns it before the live operating envelope records
anything. Main's halt filing (#366/#377) makes an operator halt no loss and never
teachable; the stopped refusal is that stop's echo, and its text names no halt
(the taxonomy files it `other:no_retry_for`). So, read back from the trace:

- `diagnose` never takes such a row as the failure a retry is credited to: the
  run's real refusal before it is (as main credits it past a halt), and a run
  whose only failures are ledger refusals learns nothing;
- `OperatingEnvelope.ingest_trace` folds no such row: the replayed envelope equals
  the live one (no loss, no failure signature, no contradiction).

Fixtures are synthetic traces with the ledger's real texts and trace marks.
"""
import json

import pytest

from cascade.agent.aspire import diagnose, harvest
from cascade.agent.effects import CONFIRMED, POSTCONDITIONS
from cascade.agent.sim_retry import RETRY_EXHAUSTED, RETRY_STOPPED, RETRY_TOO_LATE
from cascade.memory.envelope import OperatingEnvelope, failure_signature, refused_before_motion
from cascade.skills.library import SkillLibrary

TASK_A = "pick the cube"
TASK_B = "stack the cube on the box"
ARGS = {"object": "cube"}
CONTEXT = {"arm": "default", "held_object": None}
#: the first, qualifying refusal (a SafetyViolation before contact)
REFUSED = "SafetyViolation: finger trajectory intersects observed surface: lemon"
#: the ledger's refusals, as `_sim_retry_refusal` traces them
STOPPED = ({"ok": False, "error": RETRY_STOPPED.format(goal="cube", seconds=301),
            "sim_retry": {"allowed": False, "stopped": True}}, {"refused": True, "stopped": True})
USED = ({"ok": False, "error": RETRY_EXHAUSTED.format(goal="cube"), "sim_retry": {"allowed": False, "used": True}},
        {"refused": True})
LATE = ({"ok": False, "error": f"{RETRY_EXHAUSTED.format(goal='cube')}: {RETRY_TOO_LATE}",
         "sim_retry": {"allowed": False, "used": True}}, {"refused": True})
LEDGER_REFUSALS = {"stopped": STOPPED, "used": USED, "too late": LATE}


def _confirmed(skill="pick_and_place"):
    return {"ok": True, "postcondition": {
        "skill": skill, "kind": POSTCONDITIONS[skill], "status": CONFIRMED, "channel": "physics",
        "evidence": "Synthetic fixture: object moved to the destination", "measured": {"moved_m": 0.2}}}


def _call(result, context=None, skill="pick_and_place"):
    return {"skill": skill, "args": dict(ARGS), "context": {**CONTEXT, **(context or {})}, "result": result,
            "duration_ms": 12.0}


def _ledger(kind):
    result, mark = LEDGER_REFUSALS[kind]
    return _call(dict(result), {"sim_retry": dict(mark)})


def _run(root, name, task, records):
    run = root / name
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (run / "summary.txt").write_text(f"task: {task}\nsuccess: true")
    (run / "run.json").write_text(json.dumps({"sim_retry": True, "purpose": "live"}))  # harvest: live gate-1 sim runs
    return run


def test_the_ledger_refusals_file_as_no_halt():
    """Why the trace mark, not the signature, decides: the stopped refusal is an
    operator stop's echo, yet main's taxonomy files none of these texts as a halt."""
    for kind, (result, _) in LEDGER_REFUSALS.items():
        assert failure_signature(result) != "safety:halt", kind


@pytest.mark.parametrize("kind", sorted(LEDGER_REFUSALS))
def test_a_ledger_refusal_is_never_the_failure_a_retry_is_credited_to(tmp_path, kind):
    """refused -> (operator stop) -> the ledger's refusal -> a confirmed same-goal
    pick later: the retry is credited to the real refusal, as main credits it past a
    halt (tests/test_aspire_halts.py), never to the ledger's refusal."""
    run = _run(tmp_path / "runs", "run-a", TASK_A,
               [_call({"ok": False, "error": REFUSED}), _ledger(kind), _call(_confirmed())])
    diag = diagnose(run)
    assert diag.n_failures == 2
    assert diag.error == REFUSED and diag.signature == failure_signature({"error": REFUSED})
    assert diag.teachable


@pytest.mark.parametrize("kind", sorted(LEDGER_REFUSALS))
def test_runs_whose_only_failures_are_ledger_refusals_learn_nothing(tmp_path, kind):
    """Two tasks, each the ledger's refusal and then a confirmed pick of the same
    goal: diagnosed, nothing learned, promoted or written."""
    runs = tmp_path / "runs"
    for name, task in (("run-a", TASK_A), ("run-b", TASK_B)):
        _run(runs, name, task, [_ledger(kind), _call(_confirmed())])
    library = SkillLibrary(tmp_path / "lib")
    out = harvest(runs, library)
    assert out["diagnosed"] == 2 and out["learned"] == 0, out
    assert library.records() == []
    for name in ("run-a", "run-b"):
        assert not diagnose(runs / name).teachable


def test_refused_before_motion_reads_only_the_ledgers_mark():
    assert refused_before_motion({"sim_retry": {"refused": True}})
    assert refused_before_motion({"sim_retry": {"refused": True, "stopped": True}})
    for context in (None, {}, {"sim_retry": None}, {"sim_retry": {}}, {"sim_retry": {"retry": True}},
                    {"sim_retry": {"refused": 1}}, {"sim_retry": {"refused": "true"}}, {"sim_retry": {"off": True}},
                    {"sim_retry": [("refused", True)]}, "sim_retry", {"refused": True}):
        assert not refused_before_motion(context), context


@pytest.mark.parametrize("kind", sorted(LEDGER_REFUSALS))
def test_the_replayed_envelope_folds_no_ledger_refusal(tmp_path, kind):
    """ingest_trace skips the ledger's rows (the live envelope never records them):
    no loss, no failure signature, no last error; every other row is folded."""
    trace = tmp_path / "trace.jsonl"
    rows = [_call({"ok": False, "error": REFUSED}), _ledger(kind), _ledger(kind), _call(_confirmed())]
    trace.write_text("\n".join(json.dumps(r) for r in rows))
    env = OperatingEnvelope(tmp_path / "envelope.json", min_support=1)
    assert env.ingest_trace(trace) == 2
    stats = env._skills["pick_and_place"].to_json()
    assert (stats["wins"], stats["losses"], stats["halts"]) == (1, 1, 0)
    assert stats["failures"] == {failure_signature({"error": REFUSED}): 1}
