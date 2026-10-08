"""ROADMAP follow-up #8: the `programs` tier (Waddle primitives -> skills -> programs).

Design note: docs/PROGRAMS_TIER.md. A program is a declarative, bounded list
of REGISTERED tool calls (labels as parameters, coordinates only as
`localize_object(label) + offset` queries) that the orchestrator executes as
top-level `SkillRuntime.execute()` calls. Scoping decision: a program is either
AUTHORED by the agent for one instruction or DISTILLED from a verified LLM-tier
run, and both paths answer to ONE admission rule -- authorship is never
evidence; a program is stored only from an execution whose every registered
effect the independent checker CONFIRMED, and offered for reuse only after it
recurred, verified, in >= 2 distinct tasks (ASPIRE's promotion rule).

What these tests pin, most important first:

* every step is an ordinary `execute()` call with its own trace row and
  postcondition; the per-step verdict comes from the task-effects ledger (the
  checker), never from the step's self-report;
* a failed / harness-refused / refuted / unverified step stops the program
  with a `next_action`, a stuck step ends the task, a grounding failure aborts
  with ZERO motion -- and the task falls through to the LLM tier;
* raw coordinates, pixels, unregistered and non-composable tools are refused
  before anything runs;
* admission/promotion/retrieval follow the cross-task gate; candidates are
  never offered; authored and distilled programs fold into one record;
* `programs=None` (the default; `agent.programs: false`) reproduces the
  pre-change orchestrator path exactly.

The new modules are imported inside each test so that, against the pre-change
tree, every test fails on its own (RED) instead of one collection error.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from conftest import needs_pin

from cascade.agent.effects import CONFIRMED, POSTCONDITIONS, REFUTED, UNVERIFIED, Postcondition
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator

REPO = Path(__file__).resolve().parents[1]
TASK = "move the red cube to the front-left of the table"


def _ap():
    from cascade.agent import programs

    return programs


def _mp():
    from cascade.memory import programs

    return programs


def _spec(name="move-front-left", offset=(-0.09, -0.15), anchor=None, extra_steps=()):
    """grasp $object, then place it at localize_object($anchor) + offset."""
    params = {"object": "the object to move"}
    label = {"$param": "object"}
    if anchor is not None:
        params["anchor"] = "the object the placement is relative to"
        label = {"$param": "anchor"}
    return {
        "version": 1,
        "name": name,
        "description": "move an object to the front-left of where it is",
        "params": params,
        "steps": [
            {"tool": "grasp_object", "args": {"label": {"$param": "object"}}},
            {"tool": "place_at", "args": {"$target": {
                "query": "localize_object", "label": label,
                "offset_m": list(offset), "args": ["x", "y"]}}},
            *extra_steps,
        ],
    }


# ── A. the program contract (no runtime) ────────────────────────────────────


def test_a_registered_composition_validates_and_binds_labels_only():
    ap = _ap()
    prog = ap.Program.from_spec(_spec())
    assert prog.name == "move-front-left"
    assert list(prog.params) == ["object"]
    assert len(prog.signature) == 64 and int(prog.signature, 16) >= 0
    calls = prog.bind({"object": "red cube"})
    assert calls[0] == ("grasp_object", {"label": "red cube"})
    name, args = calls[1]
    assert name == "place_at" and set(args) == {"$target"}
    assert args["$target"]["label"] == "red cube"
    assert args["$target"]["offset_m"] == [-0.09, -0.15]
    # binding never mutates the stored program: still a parameter
    assert prog.steps[1][1]["$target"]["label"] == {"$param": "object"}
    assert "grasp_object" in prog.describe() and "localize_object" in prog.describe()


@pytest.mark.parametrize("tool", [
    "teleport_arm",      # not a registered tool at all
    "task_done",         # a program never declares its own success
    "reset_scene",       # undoes a task
    "halt_motion",       # acts on an in-flight motion
    "grasp_at_pixel",    # a pixel is a scene-bound coordinate
    "probe_point",       # same
    "recall_step",       # planner-facing read-back
    "emergency_stop",    # MCP stop channel: not a TOOL_SPECS tool
    "reset_stop",        # a program can never reset stop permission
])
def test_unregistered_and_non_composable_tools_are_refused(tool):
    ap = _ap()
    spec = {"name": "bad", "params": {}, "steps": [{"tool": tool, "args": {}}]}
    with pytest.raises(ap.ProgramError) as info:
        ap.Program.from_spec(spec)
    assert tool in str(info.value)


def test_raw_coordinates_are_refused_and_only_place_at_takes_a_target():
    ap = _ap()
    raw = {"name": "raw", "params": {}, "steps": [
        {"tool": "place_at", "args": {"x": 0.2, "y": -0.15}}]}
    with pytest.raises(ap.ProgramError, match="coordinate"):
        ap.Program.from_spec(raw)
    mixed = _spec()
    mixed["steps"][1]["args"]["x"] = 0.2
    with pytest.raises(ap.ProgramError, match="coordinate"):
        ap.Program.from_spec(mixed)
    wrong_tool = _spec()
    wrong_tool["steps"][0]["args"]["$target"] = {"label": "red cube", "offset_m": [0, 0]}
    with pytest.raises(ap.ProgramError, match=r"\$target"):
        ap.Program.from_spec(wrong_tool)
    with pytest.raises(ap.ProgramError, match="anchor"):
        ap.Program.from_spec(_spec(offset=(0.45, 0.0)))   # beyond the 0.40 m anchor radius
    bad_query = _spec()
    bad_query["steps"][1]["args"]["$target"]["query"] = "probe_point"
    with pytest.raises(ap.ProgramError, match="localize_object"):
        ap.Program.from_spec(bad_query)
    no_label = _spec()
    del no_label["steps"][1]["args"]["$target"]["label"]
    with pytest.raises(ap.ProgramError, match="label"):
        ap.Program.from_spec(no_label)


def test_schema_and_parameter_contract():
    ap = _ap()

    def refused(spec, match):
        with pytest.raises(ap.ProgramError, match=match):
            ap.Program.from_spec(spec)

    one = lambda tool, args, params=None: {"name": "p", "params": params or {}, "steps": [{"tool": tool, "args": args}]}  # noqa: E731
    refused(one("grasp_object", {}), "label")                                   # required arg missing
    refused(one("grasp_object", {"label": "red cube", "speed": 2}), "speed")    # unknown arg
    refused(one("wave", {"cycles": "two"}), "cycles")                           # wrong type
    refused(one("wave", {"cycles": 9}), "cycles")                               # above maximum
    refused(one("push_object", {"label": "box", "direction": "up"}), "direction")   # not in enum
    refused(one("move_relative", {"direction": "left", "distance_m": {"$param": "d"}},
                {"d": "how far"}), "string")                                    # numeric param
    refused(one("push_object", {"label": "box", "direction": {"$param": "d"}},
                {"d": "dir"}), "string")                                        # enum param
    refused(one("grasp_object", {"label": {"$param": "obj"}}), "obj")           # undeclared param
    refused(one("grasp_object", {"label": "cube"}, {"unused": "x"}), "unused")  # declared, unused
    refused({"name": "p", "params": {}, "steps": []}, "step")
    refused({"name": "p", "params": {}, "steps": [{"tool": "wave", "args": {}}] * 13}, "12")
    many = {f"p{i}": "x" for i in range(7)}
    refused({"name": "p", "params": many, "steps": [
        {"tool": "count_objects", "args": {"query": {"$param": k}}} for k in many]}, "6")
    refused({"name": "p", "params": {}, "steps": [{"tool": "wave", "args": {}}], "loop": True}, "loop")
    # a well-formed composition of perception, gesture and manipulation passes
    ap.Program.from_spec({"name": "p", "params": {"o": "thing"}, "steps": [
        {"tool": "get_observation", "args": {}},
        {"tool": "wave", "args": {"cycles": 2}},
        {"tool": "push_object", "args": {"label": {"$param": "o"}, "direction": "left", "distance_m": 0.05}},
    ]})


def test_bindings_must_be_exactly_the_declared_labels():
    ap = _ap()
    prog = ap.Program.from_spec(_spec(anchor=True))
    for bad in ({"object": "red cube"},                                  # missing anchor
                {"object": "red cube", "anchor": "bowl", "extra": "x"},  # undeclared
                {"object": 0.2, "anchor": "bowl"},                       # a number is not a label
                {"object": "  ", "anchor": "bowl"},                      # empty
                {"object": "$param", "anchor": "bowl"}):                 # a marker is not a label
        with pytest.raises(ap.ProgramError):
            prog.bind(bad)
    assert prog.bind({"object": "red cube", "anchor": "blue bowl"})[1][1]["$target"]["label"] == "blue bowl"


def test_signature_ignores_names_and_folds_equivalent_programs():
    ap = _ap()
    base = ap.Program.from_spec(_spec())
    renamed = _spec(name="something-else")
    renamed["description"] = "different words"
    renamed["params"] = {"thing": "renamed"}
    for step in renamed["steps"]:
        for value in step["args"].values():
            if isinstance(value, dict) and value.get("$param") == "object":
                value["$param"] = "thing"
            elif isinstance(value, dict) and isinstance(value.get("label"), dict):
                value["label"] = {"$param": "thing"}
    assert ap.Program.from_spec(renamed).signature == base.signature
    # offsets are compared at 1 cm resolution: perception noise folds, a real change does not
    assert ap.Program.from_spec(_spec(offset=(-0.0902, -0.1506))).signature == base.signature
    assert ap.Program.from_spec(_spec(offset=(-0.09, -0.12))).signature != base.signature
    swapped = _spec()
    swapped["steps"] = [swapped["steps"][0], {"tool": "move_home", "args": {}}, swapped["steps"][1]]
    assert ap.Program.from_spec(swapped).signature != base.signature


# ── B. the library: admission, promotion, retrieval (memory/programs.py) ────


def _admit(lib, spec, task, run, verdict=CONFIRMED, origin="authored"):
    ap = _ap()
    prog = ap.Program.from_spec(spec)
    return lib.admit(prog.spec, prog.signature, task=task, run=run, origin=origin, verdict=verdict)


def test_admission_needs_a_confirmed_verdict_and_promotion_needs_two_distinct_tasks(tmp_path):
    mp = _mp()
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    for verdict in (UNVERIFIED, REFUTED, "failed"):
        with pytest.raises(ValueError):
            _admit(lib, _spec(), TASK, "r0", verdict=verdict)
    assert len(lib) == 0, "an unverified execution must never enter the library"

    rec = _admit(lib, _spec(), TASK, "r1")
    assert rec.occurrences == 1 and rec.n_tasks == 1
    assert rec.status(lib.min_tasks) == "candidate"
    assert lib.retrievable("move the red cube to the front-left") == []
    # the same task again (case/space-insensitive) is more runs, not more tasks
    rec = _admit(lib, _spec(), "  Move the red cube to the FRONT-LEFT of the table ", "r2")
    assert rec.occurrences == 2 and rec.n_tasks == 1 and not rec.promoted(lib.min_tasks)
    rec = _admit(lib, _spec(), TASK, "r2")              # same run id: idempotent
    assert rec.occurrences == 2
    rec = _admit(lib, _spec(), "shift the blue cube front-left", "r3")
    assert rec.n_tasks == 2 and rec.promoted(lib.min_tasks)
    assert [r.name for r in lib.retrievable("please move the green cube to the front-left")] == ["move-front-left"]
    assert lib.retrievable("wave at the audience") == [], "no shared word: not offered"


def test_failures_demote_and_min_tasks_one_is_the_explicit_relaxation(tmp_path):
    mp = _mp()
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    _admit(lib, _spec(), TASK, "r1")
    rec = _admit(lib, _spec(), "shift the blue cube front-left", "r2")
    sig = rec.signature
    assert lib.retrievable("move the cube front-left")
    assert lib.record_failure(sig, run="r3", reason="place refuted").losses == 1
    assert lib.record_failure(sig, run="r3", reason="again").losses == 1, "one failure per run"
    assert lib.retrievable("move the cube front-left"), "2 verified > 1 failed"
    lib.record_failure(sig, run="r4", reason="grasp unverified")
    assert lib.retrievable("move the cube front-left") == [], "failures caught up: demoted"
    assert lib.get(sig).status(lib.min_tasks) == "demoted"
    assert lib.record_failure("0" * 64, run="r5") is None, "unknown programs gain no record"

    relaxed = mp.ProgramLibrary(tmp_path / "relaxed.jsonl", min_tasks=1)
    one = relaxed.admit(rec.program, sig, task=TASK, run="r1", origin="authored", verdict=CONFIRMED)
    assert one.promoted(relaxed.min_tasks) and relaxed.retrievable("move the cube front-left")
    with pytest.raises(ValueError):
        mp.ProgramLibrary(tmp_path / "x.jsonl", min_tasks=0)


def test_library_persists_as_jsonl_without_coordinates_and_survives_a_corrupt_line(tmp_path):
    mp = _mp()
    path = tmp_path / "programs.jsonl"
    lib = mp.ProgramLibrary(path)
    _admit(lib, _spec(), TASK, "r1")
    _admit(lib, _spec(name="other", offset=(0.05, 0.05)), "nudge the cube", "r2", origin="distilled")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        for step in rec["program"]["steps"]:
            assert not any(k in step["args"] for k in ("x", "y", "z")), step
    path.write_text(lines[0] + "\n{not json\n" + lines[1] + "\n")
    again = mp.ProgramLibrary(path)
    assert sorted(r.name for r in again.records()) == ["move-front-left", "other"]
    assert again.get("other").origins == ["distilled"]


# ── C. authoring: prompt and reply parsing ──────────────────────────────────


def _promoted_library(tmp_path, spec=None):
    mp = _mp()
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    _admit(lib, spec or _spec(), "move the blue cube to the front-left", "seed-1")
    _admit(lib, spec or _spec(), "shift the green cube front-left", "seed-2")
    return lib


def test_author_reply_parsing(tmp_path):
    ap, mp = _ap(), _mp()
    lib = _promoted_library(tmp_path)
    _admit(lib, _spec(name="lonely", offset=(0.1, 0.1)), "a single task once", "seed-3")
    tier = ap.ProgramTier(lib)
    offered = tier.offered(TASK)
    assert [r.name for r in offered] == ["move-front-left"]

    assert tier.parse("NONE", offered).kind == "none"
    assert tier.parse("none -- this is a question", offered).kind == "none"
    fresh = tier.parse("Here:\n```json\n" + json.dumps({"program": _spec(), "bind": {"object": "red cube"}})
                       + "\n```", offered)
    assert fresh.kind == "fresh" and fresh.bindings == {"object": "red cube"}
    assert fresh.program.signature == ap.Program.from_spec(_spec()).signature
    reuse = tier.parse(json.dumps({"use": "move-front-left", "bind": {"object": "red cube"}}), offered)
    assert reuse.kind == "reuse" and reuse.record.name == "move-front-left"

    for text, why in (
        ("{not json at all", "JSON"),
        (json.dumps({"use": "lonely", "bind": {"object": "red cube"}}), "promoted"),
        (json.dumps({"use": "no-such-program", "bind": {}}), "unknown"),
        (json.dumps({"program": {"name": "x", "params": {}, "steps": [
            {"tool": "place_at", "args": {"x": 0.1, "y": 0.1}}]}, "bind": {}}), "coordinate"),
        (json.dumps({"program": _spec(), "bind": {"object": 3}}), "object"),
        (json.dumps({"something": "else"}), "program"),
    ):
        proposal = tier.parse(text, offered)
        assert proposal.kind == "invalid", text
        assert why in proposal.reason, (why, proposal.reason)


def test_prompt_offers_only_promoted_programs_and_no_coordinates(tmp_path):
    ap = _ap()
    lib = _promoted_library(tmp_path)
    _admit(lib, _spec(name="lonely", offset=(0.1, 0.1)), "move a single cube once", "seed-3")
    tier = ap.ProgramTier(lib)
    prompt = tier.prompt(TASK, tier.offered(TASK), ["red cube (red) visible", "blue bowl remembered"])
    assert "move-front-left" in prompt and "promoted" in prompt
    assert "lonely" not in prompt, "a candidate is never offered"
    assert "red cube (red) visible" in prompt
    assert "grasp_object" in prompt and "place_at" in prompt
    for refused in ("task_done", "reset_scene", "grasp_at_pixel", "probe_point"):
        assert f"- {refused}(" not in prompt, refused
    assert "NONE" in prompt and "$target" in prompt and "$param" in prompt


# ── D. the runner, on a stub runtime with a task-effects ledger ─────────────


class _LedgerRuntime:
    """Deterministic SkillRuntime stand-in: every registered call appends a
    ledger row with the CHECKER's status, independent of what the result
    dict claims. `script` maps a tool name to a list of (result, status)."""

    def __init__(self, script=None):
        object.__setattr__(self, "sets", [])
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []
        self.held_object = None
        self.current_tier = None
        self.last_frame = None
        self.memory = _StubMemory()
        self.trace = _StubTrace()
        self._rows = []
        self.sets.clear()

    def __setattr__(self, key, value):
        self.sets.append((key, value))
        object.__setattr__(self, key, value)

    def execute(self, name, args):
        self.calls.append((name, dict(args), self.current_tier))
        if name == "localize_object":
            return {"ok": True, "position": [0.29, 0.0, 0.04], "detected_as": args["label"]}
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        queue = self.script.get(name) or [({"ok": True}, CONFIRMED)]
        result, status = queue.pop(0) if len(queue) > 1 else queue[0]
        result = dict(result)
        result.setdefault("outcome", "ok" if result.get("ok") else "failed")
        if name in POSTCONDITIONS:
            self._rows.append({"task_id": "t", "action_id": len(self._rows), "actor": "default",
                               "skill": name, "kind": POSTCONDITIONS[name], "status": status,
                               "reason": f"checker said {status}"})
        if name == "grasp_object" and result.get("ok"):
            object.__setattr__(self, "held_object", args.get("label"))
        return result

    def task_effects(self):
        return {"task_id": "t", "effects": [dict(r) for r in self._rows]}

    def unverified_actions(self):
        return [f"default:{r['skill']}#{r['action_id']} {r['status']}: {r['reason']}"
                for r in self._rows if r["status"] != CONFIRMED]

    def frame_jpeg(self):
        return b"jpeg"


def _run(spec, bindings, rt):
    ap = _ap()
    return ap.run_program(ap.Program.from_spec(spec), bindings, rt, tool_log=[])


def test_runner_reads_verdicts_from_the_ledger_not_the_result():
    ap = _ap()
    forged = ({"ok": True, "verified": True,
               "postcondition": {"status": "confirmed", "channel": "physics", "evidence": "self-authored"}},
              UNVERIFIED)
    rt = _LedgerRuntime({"grasp_object": [forged]})
    run = _run(_spec(), {"object": "red cube"}, rt)
    assert run.steps[0].verdict == UNVERIFIED, "the result's own postcondition is not evidence"
    assert run.status == ap.STOPPED and run.stopped_at == 1
    assert [c[0] for c in rt.calls] == ["localize_object", "grasp_object"], "place_at must not run"
    assert run.next_action and "UNVERIFIED" in run.next_action and "red cube" in run.next_action
    assert not run.verified

    class NoLedger(_LedgerRuntime):
        def task_effects(self):
            raise AttributeError("task_effects")

    bare = NoLedger({"grasp_object": [({"ok": True, "verified": True,
                                        "postcondition": {"status": "confirmed"}}, CONFIRMED)]})
    run = _run(_spec(), {"object": "red cube"}, bare)
    assert run.steps[0].verdict == UNVERIFIED and "ledger" in run.steps[0].evidence
    assert run.status == ap.STOPPED

    ok = _LedgerRuntime()
    run = _run(_spec(), {"object": "red cube"}, ok)
    assert run.status == ap.COMPLETED and run.verified and run.verdict == CONFIRMED
    assert [s.verdict for s in run.steps] == [CONFIRMED, CONFIRMED]
    assert all(tier == "program" for _, _, tier in ok.calls)
    place = ok.calls[-1]
    assert place[0] == "place_at" and place[1] == {"x": pytest.approx(0.20), "y": pytest.approx(-0.15)}


class _SilentLedger(_LedgerRuntime):
    """The ledger exists, but its checker recorded NOTHING for the n-th call
    (skipped); the result still claims a confirmed postcondition."""

    def __init__(self, silent_call):
        super().__init__()
        object.__setattr__(self, "_silent", silent_call)

    def execute(self, name, args):
        if len(self.calls) == self._silent:
            self.calls.append((name, dict(args), self.current_tier))
            return {"ok": True, "outcome": "ok", "verified": True, "postcondition": {"status": "confirmed"}}
        return super().execute(name, args)


def test_a_step_the_ledger_has_no_row_for_stays_unverified():
    """Found by the mutation check: only rows THIS step added are its evidence.
    Calls: 0 localize_object (grounding), 1 grasp, 2 place_at, 3 grasp."""
    ap = _ap()
    run = _run(_spec(), {"object": "red cube"}, _SilentLedger(silent_call=2))
    assert [s.verdict for s in run.steps] == [CONFIRMED, UNVERIFIED]
    assert "recorded no effect" in run.steps[1].evidence
    assert run.status == ap.STOPPED and not run.verified
    # an earlier CONFIRMED row of the same tool is not evidence for a later call
    regrasp = {"tool": "grasp_object", "args": {"label": {"$param": "object"}}}
    run = _run(_spec(extra_steps=(regrasp,)), {"object": "red cube"}, _SilentLedger(silent_call=3))
    assert [s.verdict for s in run.steps] == [CONFIRMED, CONFIRMED, UNVERIFIED]
    assert run.status == ap.STOPPED and run.stopped_at == 3 and not run.verified


@pytest.mark.parametrize("result,status,verdict", [
    ({"ok": False, "error": "SkillError: IK failed for every candidate"}, "failed", "failed"),
    ({"ok": False, "error": "SafetyViolation: workspace: z below table"}, "failed", "refused"),
    ({"ok": False, "error": "NoExecutableGrasp: red cube: approach unsafe: e-stop latched"}, "failed", "refused"),
    ({"ok": False, "error": "postcondition failed: cube still on the table", "self_reported_ok": True},
     REFUTED, REFUTED),
])
def test_runner_stops_on_failed_refused_and_refuted_steps(result, status, verdict):
    ap = _ap()
    rt = _LedgerRuntime({"grasp_object": [(result, status)]})
    run = _run(_spec(extra_steps=[{"tool": "move_home", "args": {}}]), {"object": "red cube"}, rt)
    assert run.status == ap.STOPPED and run.stopped_at == 1
    assert run.steps[0].verdict == verdict
    assert [c[0] for c in rt.calls] == ["localize_object", "grasp_object"]
    assert run.next_action and "grasp_object" in run.next_action
    assert "Do not re-run" in run.next_action
    if verdict == "refused":
        assert "harness" in run.next_action
    assert run.verdict == (REFUTED if verdict == REFUTED else UNVERIFIED)


def test_a_stuck_step_is_its_own_status():
    ap = _ap()
    stuck = ({"ok": False, "outcome": "stuck", "ask": "Put the red cube upright and ask again."}, "failed")
    rt = _LedgerRuntime({"grasp_object": [stuck]})
    run = _run(_spec(), {"object": "red cube"}, rt)
    assert run.status == ap.STUCK and run.steps[-1].verdict == ap.STUCK
    assert run.steps[-1].result["ask"].startswith("Put the red cube")


def test_unchecked_motion_never_stops_a_program_but_keeps_it_unverified():
    ap = _ap()
    rt = _LedgerRuntime()
    run = _run(_spec(extra_steps=[{"tool": "wave", "args": {}}, {"tool": "count_objects", "args": {}}]),
               {"object": "red cube"}, rt)
    assert run.status == ap.COMPLETED
    assert [s.verdict for s in run.steps] == [CONFIRMED, CONFIRMED, ap.UNCHECKED, ap.NO_EFFECT]
    assert run.verdict == UNVERIFIED and not run.verified, "a wave's only evidence is its own report"
    only_reads = _run({"name": "look", "params": {}, "steps": [{"tool": "get_observation", "args": {}}]}, {}, rt)
    assert only_reads.status == ap.COMPLETED and not only_reads.verified, "no registered effect: nothing verified"
    # Found by the mutation check: such a run can end in a SUCCESSFUL report
    # (no registered effect is left unverified), so the library must apply the
    # program's own verdict, not the report's success.
    lib = _mp().ProgramLibrary(None)
    tier = ap.ProgramTier(lib)
    for done in (run, only_reads):
        proposal = ap.Proposal(ap.FRESH, done.program, dict(done.bindings))
        assert tier.account(done, proposal, task=TASK, run_id=f"r-{done.program.name}", success=True) == "not admitted"
    assert len(lib) == 0, "a successful report is not evidence for an unverified program"


def test_grounding_failure_aborts_before_any_step():
    ap = _ap()

    class Blind(_LedgerRuntime):
        def execute(self, name, args):
            if name == "localize_object":
                self.calls.append((name, dict(args), self.current_tier))
                return {"ok": False, "error": f"SkillError: localize {args['label']!r} failed: no detections"}
            return super().execute(name, args)

    rt = Blind()
    run = _run(_spec(anchor=True), {"object": "red cube", "anchor": "blue bowl"}, rt)
    assert run.status == ap.ABORTED and run.steps == []
    assert [c[0] for c in rt.calls] == ["localize_object"]
    assert "blue bowl" in run.reason and "No motion" in run.next_action


# ── E. end to end through the orchestrator ──────────────────────────────────


class _StubMemory:
    def __init__(self):
        self.added = []

    def reset_frames(self):
        pass

    def digest(self, max_lines=None):
        return "(nothing yet)"

    def memory_frames(self, k):
        return []

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))


class _StubTrace:
    def __init__(self):
        self.summaries = []

    def finish(self, summary):
        self.summaries.append(summary)


def _script(*calls: ToolCall) -> list[LLMResponse]:
    return [LLMResponse(text="", tool_calls=[c]) for c in calls]


def _author(reply) -> LLMResponse:
    return LLMResponse(text=reply if isinstance(reply, str) else json.dumps(reply))


def test_a_stuck_program_step_ends_the_task_without_a_tier_three_turn(tmp_path):
    ap, mp = _ap(), _mp()
    stuck = ({"ok": False, "outcome": "stuck", "ask": "Put the red cube upright and ask again."}, "failed")
    rt = _LedgerRuntime({"grasp_object": [stuck]})
    llm = MockLLM([_author({"program": _spec(), "bind": {"object": "red cube"}})])
    agent = AgentOrchestrator(llm, rt, advisor=None, decompose=False, attach_images=False,
                              verify_milestones=False, max_steps=5,
                              programs=ap.ProgramTier(mp.ProgramLibrary(tmp_path / "p.jsonl")))
    report = agent.run_task(TASK)
    assert report.path == "program" and report.success is False
    assert report.stuck == {"skill": "grasp_object", "args": {"label": "red cube"},
                            "ask": "Put the red cube upright and ask again."}
    assert len(llm.requests) == 1, "a stuck step is relayed, never re-planned"
    assert [c[0] for c in rt.calls] == ["localize_object", "grasp_object"]


class _FailingHabit:
    """Stub FastPlanner: a tier-1/2 plan EXISTS for the task and fails."""

    def __init__(self):
        self.outcomes = []

    def plan(self, task):
        from types import SimpleNamespace

        return SimpleNamespace(calls=[("grasp_object", {"label": "red cube"})], needs_grounding=False,
                               source="experience", summary="", subgoals=[])

    def note_outcome(self, task, calls, ok, duration, **kw):
        self.outcomes.append(ok)


def test_a_failed_habit_goes_to_tier_three_not_to_a_program(tmp_path):
    """Found by the mutation check: the program tier is consulted only when
    tiers 1-2 had NO plan; a failed habit's note goes to deliberation."""
    ap, mp = _ap(), _mp()
    rt = _LedgerRuntime({"grasp_object": [({"ok": False, "error": "SkillError: IK failed"}, "failed")]})
    habit = _FailingHabit()
    llm = MockLLM(_script(ToolCall("task_done", {"success": False, "summary": "the habit failed"})))
    agent = AgentOrchestrator(llm, rt, advisor=None, decompose=False, attach_images=False,
                              verify_milestones=False, max_steps=5, fast_planner=habit,
                              programs=ap.ProgramTier(mp.ProgramLibrary(tmp_path / "p.jsonl")))
    report = agent.run_task(TASK)
    assert habit.outcomes == [False]
    assert report.path == "llm" and not report.success
    assert len(llm.requests) == 1, "no authoring turn after a failed habit"
    intro = str(llm.requests[0]["messages"][0]["content"])
    assert "FAILED" in intro and "Write a PROGRAM" not in intro
    assert [c[2] for c in rt.calls] == ["experience", "llm"]


def _confirm_effects(runtime) -> None:
    """Stand-in physics channel (same as tests/test_task_recipes.py): the
    static mock camera leaves every pick `unverified`; these tests are about
    the PROGRAM contract, so registered effects are pinned CONFIRMED. Motion,
    the harness and the checker wiring are untouched."""
    def verify(name, args, result, before=None):
        kind = POSTCONDITIONS.get(name)
        if kind is None:
            return None
        return Postcondition(name, kind, CONFIRMED, "stand-in physics channel (test)", channel="physics")

    runtime.effects.verify = verify


def _wait_for_cube(runtime) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
        time.sleep(0.05)
    assert runtime.beliefs.find("red cube") is not None, "watcher never saw the mock cube"


def _trace_rows(runtime) -> list[dict]:
    return [json.loads(l) for l in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


def _agent(llm, runtime, tier, tmp_path, **kw):
    from cascade.agent.reflex import ExperienceMemory, FastPlanner

    exp = ExperienceMemory(tmp_path / "experience.json")
    return AgentOrchestrator(llm, runtime, decompose=False, max_steps=6,
                             fast_planner=FastPlanner(exp), programs=tier, **kw), exp


@needs_pin
def test_authored_program_runs_as_top_level_calls_and_becomes_a_candidate(runtime_and_arm, tmp_path):
    ap, mp = _ap(), _mp()
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    cube = runtime.beliefs.find("red cube").position.copy()
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    llm = MockLLM([_author({"program": _spec(), "bind": {"object": "red cube"}})])
    agent, exp = _agent(llm, runtime, ap.ProgramTier(lib), tmp_path)
    with runtime.watcher.paused():
        report = agent.run_task(TASK)

    assert report.success and report.path == "program", report.summary
    assert len(llm.requests) == 1, "one authoring turn, no turn-by-turn tool loop"
    rows = [r for r in _trace_rows(runtime) if r.get("tier") == "program"]
    skills = [r["skill"] for r in rows]
    assert skills == ["localize_object", "grasp_object", "place_at"], skills
    for row in rows[1:]:   # each step is its own top-level row with its own verdict
        assert row["result"]["postcondition"]["status"] == CONFIRMED
        assert row["keyframe_before"] and row["keyframe_after"]
    place = rows[2]["args"]
    assert place["x"] == pytest.approx(cube[0] - 0.09, abs=0.01)
    assert place["y"] == pytest.approx(cube[1] - 0.15, abs=0.01)
    assert runtime.unverified_actions() == []
    rec = lib.records()
    assert len(rec) == 1 and rec[0].origins == ["authored"]
    assert rec[0].n_tasks == 1 and rec[0].status(lib.min_tasks) == "candidate"
    # the verified run is also a tier-2 recipe of THIS instruction
    assert exp.recall(TASK) is not None and exp.recall(TASK)["kind"] == "recipe"
    receipts = sorted((runtime.trace.run_dir / "programs").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "completed" and receipt["verdict"] == CONFIRMED
    assert receipt["signature"] == rec[0].signature
    assert [s["verdict"] for s in receipt["steps"]] == [CONFIRMED, CONFIRMED]


@needs_pin
def test_an_unverified_step_stops_the_program_and_hands_the_task_to_tier_three(runtime_and_arm, tmp_path):
    ap, mp = _ap(), _mp()
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5          # the static mock leaves the grasp UNVERIFIED
    _wait_for_cube(runtime)
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    llm = MockLLM([_author({"program": _spec(), "bind": {"object": "red cube"}}),
                   *_script(ToolCall("task_done", {"success": False, "summary": "grasp unconfirmed"}))])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(lib), tmp_path)
    with runtime.watcher.paused():
        report = agent.run_task(TASK)

    assert report.path == "llm" and not report.success
    skills = [r["skill"] for r in _trace_rows(runtime)]
    assert "grasp_object" in skills and "place_at" not in skills, skills
    intro = str(llm.requests[1]["messages"][0]["content"])
    assert "program" in intro and "UNVERIFIED" in intro and "Do not re-run" in intro
    assert len(lib) == 0, "an unverified program is never admitted"


@needs_pin
def test_a_harness_refusal_stops_the_program_with_a_next_action(runtime_and_arm, tmp_path):
    ap, mp = _ap(), _mp()
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    llm = MockLLM([_author({"program": _spec(), "bind": {"object": "red cube"}}),
                   *_script(ToolCall("task_done", {"success": False, "summary": "e-stop is latched"}))])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(mp.ProgramLibrary(tmp_path / "p.jsonl")), tmp_path)
    runtime.arm.harness.estop("test: operator stop")
    try:
        with runtime.watcher.paused():
            report = agent.run_task(TASK)
    finally:
        runtime.arm.harness.reset_estop()
    assert not report.success
    program_rows = [r for r in _trace_rows(runtime) if r.get("tier") == "program"]
    assert [r["skill"] for r in program_rows] == ["localize_object", "grasp_object"]
    assert program_rows[1]["result"]["ok"] is False
    intro = str(llm.requests[1]["messages"][0]["content"])
    assert "refused" in intro and "harness" in intro and "grasp_object" in intro


@needs_pin
def test_a_grounding_failure_aborts_with_zero_motion(runtime_and_arm, tmp_path):
    ap, mp = _ap(), _mp()
    from cascade.skills.runtime import _MOTION_SKILLS

    runtime, arm = runtime_and_arm
    _wait_for_cube(runtime)
    executed: list[str] = []
    real_execute = runtime.execute

    def spy(name, args):
        executed.append(name)
        return real_execute(name, args)

    runtime.execute = spy
    llm = MockLLM([_author({"program": _spec(anchor=True), "bind": {"object": "red cube", "anchor": "blue bowl"}}),
                   *_script(ToolCall("task_done", {"success": False, "summary": "no bowl"}))])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(mp.ProgramLibrary(tmp_path / "p.jsonl")), tmp_path)
    with runtime.watcher.paused():
        report = agent.run_task(TASK)
    assert not report.success and report.path == "llm"
    assert "localize_object" in executed
    assert not (set(executed) & _MOTION_SKILLS), executed
    assert runtime.held_object is None
    intro = str(llm.requests[1]["messages"][0]["content"])
    assert "blue bowl" in intro and "No motion" in intro


@needs_pin
def test_a_promoted_program_is_offered_and_reused_with_new_bindings(runtime_and_arm, tmp_path):
    ap = _ap()
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    lib = _promoted_library(tmp_path)
    llm = MockLLM([_author({"use": "move-front-left", "bind": {"object": "red cube"}})])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(lib), tmp_path)
    with runtime.watcher.paused():
        report = agent.run_task(TASK)
    assert report.success and report.path == "program", report.summary
    prompt = str(llm.requests[0]["messages"][0]["content"])
    assert "move-front-left" in prompt and "promoted" in prompt
    rec = lib.get("move-front-left")
    assert rec.occurrences == 3 and rec.n_tasks == 3
    assert set(rec.origins) == {"authored", "reused"}


@needs_pin
def test_a_candidate_is_not_offered_and_a_use_of_it_is_refused(runtime_and_arm, tmp_path):
    ap, mp = _ap(), _mp()
    runtime, arm = runtime_and_arm
    _wait_for_cube(runtime)
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    _admit(lib, _spec(), "move the blue cube to the front-left", "seed-1")   # one task: candidate
    llm = MockLLM([_author({"use": "move-front-left", "bind": {"object": "red cube"}}),
                   *_script(ToolCall("task_done", {"success": False, "summary": "nothing done"}))])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(lib), tmp_path)
    with runtime.watcher.paused():
        report = agent.run_task(TASK)
    assert report.path == "llm"
    assert "move-front-left" not in str(llm.requests[0]["messages"][0]["content"])
    assert not [r for r in _trace_rows(runtime) if r.get("tier") == "program"], "nothing ran"
    intro = str(llm.requests[1]["messages"][0]["content"])
    assert "not promoted" in intro and "No motion" in intro
    assert lib.get("move-front-left").occurrences == 1


@needs_pin
def test_a_verified_llm_run_is_distilled_and_folds_into_the_authored_program(runtime_and_arm, tmp_path):
    """Both authoring paths, one admission rule: a program the agent authored
    for task A (candidate) and the same structure distilled from a verified
    LLM-tier run of task B are two distinct tasks of evidence -> promoted."""
    ap, mp = _ap(), _mp()
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    _admit(lib, _spec(), "move the blue cube to the front-left", "seed-1")
    llm = MockLLM([_author("NONE"), *_script(
        ToolCall("get_observation", {}),
        ToolCall("grasp_object", {"label": "red cube"}),
        ToolCall("place_at", {"x": 0.20, "y": -0.15}),
        ToolCall("task_done", {"success": True, "summary": "moved the cube to the front-left"}),
    )])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(lib), tmp_path)
    with runtime.watcher.paused():
        report = agent.run_task(TASK)
    assert report.success and report.path == "llm", report.summary
    records = lib.records()
    assert len(records) == 1, [r.name for r in records]
    rec = records[0]
    assert set(rec.origins) == {"authored", "distilled"}
    assert rec.n_tasks == 2 and rec.promoted(lib.min_tasks)
    stored = json.dumps(rec.program)
    assert '"x":' not in stored and '"y":' not in stored


@needs_pin
def test_a_distilled_program_on_its_own_is_a_parameterized_candidate(runtime_and_arm, tmp_path):
    ap, mp = _ap(), _mp()
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    lib = mp.ProgramLibrary(tmp_path / "programs.jsonl")
    llm = MockLLM([_author("NONE"), *_script(
        ToolCall("grasp_object", {"label": "red cube"}),
        ToolCall("place_at", {"x": 0.20, "y": -0.15}),
        ToolCall("task_done", {"success": True, "summary": "done"}),
    )])
    agent, _ = _agent(llm, runtime, ap.ProgramTier(lib), tmp_path)
    with runtime.watcher.paused():
        assert agent.run_task(TASK).success
    (rec,) = lib.records()
    assert rec.origins == ["distilled"] and rec.status(lib.min_tasks) == "candidate"
    prog = ap.Program.from_spec(rec.program)
    assert list(prog.params) == ["object"] and "red cube" in prog.params["object"]
    assert prog.steps[0] == ("grasp_object", {"label": {"$param": "object"}})
    assert prog.steps[1][1]["$target"]["label"] == {"$param": "object"}
    assert prog.signature == ap.Program.from_spec(_spec()).signature


# ── F. opt-in: off is the pre-change path, byte for byte ────────────────────


class _GoldenRuntime:
    """Same stub as tests/test_premotion_plausibility.py's golden."""

    def __init__(self):
        object.__setattr__(self, "sets", [])
        self.memory = _StubMemory()
        self.trace = _StubTrace()
        self.calls = []
        self.last_frame = object()
        self.current_tier = None
        self.held_object = None
        self.sets.clear()

    def __setattr__(self, key, value):
        self.sets.append((key, value))
        object.__setattr__(self, key, value)

    def execute(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True, "echo": name}

    def frame_jpeg(self):
        return b"jpeg"


_GOLDEN_SCRIPT = (
    ToolCall("move_home", {}, id="c1"),
    ToolCall("place_at", {"x": 0.2, "y": -0.1}, id="c2"),
    ToolCall("task_done", {"success": True, "summary": "done"}, id="c3"),
)


def _golden_run(prefix=(), **kw):
    rt = _GoldenRuntime()
    llm = MockLLM([*prefix, *_script(*_GOLDEN_SCRIPT)])
    agent = AgentOrchestrator(llm, rt, advisor=None, decompose=False, attach_images=False,
                              verify_milestones=False, max_steps=5, **kw)
    return rt, llm, agent.run_task("stub task")


def _assert_golden(rt, llm, report, offset=0):
    assert rt.calls == [(c.name, c.arguments) for c in _GOLDEN_SCRIPT]
    assert rt.sets == [("current_tier", "llm"), ("current_tier", None)] * 3 + [("last_path", "llm")]
    assert rt.memory.added == []
    assert report.success is True and report.steps == 3 and report.path == "llm"
    assert report.tool_log == [
        {"step": 1, "tier": "llm", "tool": "move_home", "args": {}, "result": {"ok": True, "echo": "move_home"}},
        {"step": 2, "tier": "llm", "tool": "place_at", "args": {"x": 0.2, "y": -0.1},
         "result": {"ok": True, "echo": "place_at"}},
        {"step": 3, "tier": "llm", "tool": "task_done", "args": {"success": True, "summary": "done"},
         "result": {"ok": True, "task_complete": True}},
    ]
    last = llm.requests[offset + 2]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "tool", "assistant", "tool"]
    assert last[0]["content"].startswith("Task: stub task\n")
    assert last[2] == {"role": "tool", "tool_call_id": "c1", "name": "move_home",
                       "content": json.dumps({"ok": True, "echo": "move_home"})}
    assert len(llm.requests) == offset + 3


def test_programs_off_is_byte_identical_to_the_pre_change_path():
    _assert_golden(*_golden_run())                 # default construction
    _assert_golden(*_golden_run(programs=None))    # the explicit off value


def test_programs_on_with_a_declining_brain_only_adds_the_authoring_turn(tmp_path):
    ap, mp = _ap(), _mp()
    tier = ap.ProgramTier(mp.ProgramLibrary(tmp_path / "p.jsonl"))
    rt, llm, report = _golden_run(prefix=[_author("NONE")], programs=tier)
    _assert_golden(rt, llm, report, offset=1)
    authoring = llm.requests[0]
    assert authoring["tools"] is None and "PROGRAM" in authoring["messages"][0]["content"]
    assert not (tmp_path / "p.jsonl").exists(), "a declined program touches no store"


def test_demo_wiring_is_off_by_default_and_explicitly_opt_in(demo_cfg, monkeypatch, tmp_path):
    from cascade.apps.demo import _program_tier

    ap = _ap()
    monkeypatch.delenv("CASCADE_PROGRAMS", raising=False)
    monkeypatch.setenv("CASCADE_PROGRAMS_PATH", str(tmp_path / "programs.jsonl"))
    assert demo_cfg.agent.get("programs") is False, "configs/demo.yaml ships the tier OFF"
    assert _program_tier(demo_cfg, is_mock=False) is None
    demo_cfg._data["agent"]["programs"] = True
    demo_cfg._data["agent"]["program_min_tasks"] = 3
    tier = _program_tier(demo_cfg, is_mock=False)
    assert isinstance(tier, ap.ProgramTier)
    assert tier.library.min_tasks == 3 and tier.library.path == tmp_path / "programs.jsonl"
    assert _program_tier(demo_cfg, is_mock=True) is None, "a scripted mock brain cannot author programs"
    monkeypatch.setenv("CASCADE_PROGRAMS", "0")
    assert _program_tier(demo_cfg, is_mock=False) is None
    demo_cfg._data["agent"]["programs"] = False
    monkeypatch.setenv("CASCADE_PROGRAMS", "1")
    assert isinstance(_program_tier(demo_cfg, is_mock=False), ap.ProgramTier)


def test_docs_record_the_programs_tier():
    roadmap = (REPO / "docs/ROADMAP.md").read_text()
    arch = (REPO / "docs/ARCHITECTURE.md").read_text()
    readme = (REPO / "README.md").read_text()
    design = (REPO / "docs/PROGRAMS_TIER.md").read_text()
    assert "8. ~~**A `programs` tier (Waddle).**" in roadmap
    assert "landed 2026-10-08" in roadmap and "PROGRAMS_TIER.md" in roadmap
    assert "agent/programs.py" in arch and "memory/programs.py" in arch and "tier 2.5" in arch.lower()
    assert "PROGRAMS_TIER.md" in readme and "programs" in readme
    assert "both" in design.lower() and "one admission rule" in design.lower()
    assert "agent.programs: false" in design
