"""Tier-2 recall must respect the clause structure of a sequence (B37).

ROADMAP #7, "found while measuring": tier-2 recall matched the compound
"pick up the red cube and then pick up the blue cube" to the single habit
"pick up the red cube" (hashed bag-of-words cosine 0.904, index estimate
0.901 >= 0.9). `FastPlanner.plan` consults experience on the WHOLE task
before the curriculum split, so the fast tier ran half the command and
reported success -- exactly what the curriculum's all-or-nothing rule and
`parse_command`'s compound refusal exist to prevent.

Root cause: the whole-text vector cannot see clause boundaries or their
order. A compound repeats most of a clause's words, so it scores >= 0.9
against EITHER clause; the reverse also holds (one clause recalls a recorded
compound, inside `_plan_one` too); a reversed sequence is the same vector
(1.0); and one differing word in one clause is diluted by the other clause.

The rule pinned here (`ExperienceMemory.recall`): a candidate is admissible
only with the task's clause structure -- the same number of `split_subgoals`
clauses and, for a sequence, each clause matching its counterpart IN ORDER at
the same `min_sim`. A refused candidate is skipped like a demoted one. A single
instruction is decided exactly as before (golden pins at the bottom).

Measurements: REPORT.md of the B37 item (scratch/measure_recall.py runs the
real recall + FastPlanner on main and on this branch). Mock stack only: no
MuJoCo, no network; the one runtime built here dials ports in 45000-45099.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import needs_pin

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.agent.reflex import ExperienceMemory, FastPlanner, _embed, split_subgoals

RED = "pick up the red cube"
BLUE = "pick up the blue cube"
COMPOUND = f"{RED} and then {BLUE}"
MOVE_RED = "move the red cube to the front-left of the table"
MOVE_BLUE = "move the blue cube to the front-left of the table"
MOVE_BOTH = f"{MOVE_RED} and then {MOVE_BLUE}"
PON_RED = "pon el cubo rojo en el bol"
PON_BOTH = f"{PON_RED} y luego pon el cubo azul en el bol"


def _grasp(label):
    return ("grasp_object", {"label": label})


def _target(label, offset):
    return {"query": "localize_object", "label": label, "offset_m": list(offset), "args": ["x", "y"]}


def _move_recipe(label, offset=(-0.09, -0.15)):
    """A Task-Specific Memory recipe step pair (memory/recipes.py shape)."""
    return [_grasp(label), ("place_at", {"$target": _target(label, offset)})]


def _exp(tmp_path, *records, name="experience.json"):
    """ExperienceMemory with each (task, calls[, meta]) recorded as a success,
    the way the orchestrator records a verified fast or LLM-tier run."""
    exp = ExperienceMemory(tmp_path / name)
    for task, calls, *meta in records:
        exp.record(task, calls, True, 1.0, **(meta[0] if meta else {}))
    return exp


def _cos(a, b):
    return float(np.dot(_embed(a), _embed(b)))


def _calls(plan):
    return [(name, dict(args)) for name, args in plan.calls]


# ── premises: the cause, measured (pass before and after B37) ────────────


def test_premise_the_whole_text_similarity_cannot_see_clauses(tmp_path):
    """The index puts the one-clause habit above the 0.9 floor for the
    compound (the ROADMAP's 0.901), symmetrically for the other clause's
    words, and a reversed sequence is the SAME vector. Similarity alone
    cannot tell a sequence from one of its clauses or from its reversal."""
    exp = _exp(tmp_path, (RED, [_grasp("red cube")]))
    (sim, _), = exp._index.search(_embed(COMPOUND), k=1)
    assert sim >= 0.9, sim                                   # 0.9014: what recall() thresholds
    assert _cos(COMPOUND, RED) == pytest.approx(0.904, abs=0.001)
    assert _cos(COMPOUND, BLUE) == pytest.approx(0.904, abs=0.001)
    assert _cos(PON_BOTH, PON_RED) == pytest.approx(0.923, abs=0.001)
    assert _cos("wave and then go home", "go home and then wave") == pytest.approx(1.0)


def test_premise_clauses_are_what_the_curriculum_splits_on():
    """The rule compares `split_subgoals` clauses: sequence connectives in
    both languages, never a bare 'and' (one pick_and_place stays one)."""
    assert split_subgoals(COMPOUND) == [RED, BLUE]
    assert split_subgoals(f"{RED} then {BLUE}") == [RED, BLUE]
    assert split_subgoals(f"please {RED}, then {BLUE}") == [f"please {RED}", BLUE]
    assert split_subgoals(PON_BOTH) == [PON_RED, "pon el cubo azul en el bol"]
    assert split_subgoals(f"{PON_RED} y después vuelve a casa") == [PON_RED, "vuelve a casa"]
    one = "pick up the cube and put it in the box"
    assert split_subgoals(one) == [one]


# ── recall: a candidate needs the task's clause structure ────────────────


@pytest.mark.parametrize("habit, task", [
    (RED, COMPOUND),                                         # the ROADMAP case, 0.901
    (RED, f"{RED} then {BLUE}"),                             # bare 'then', 0.901
    (PON_RED, PON_BOTH),                                     # Spanish 'y luego', 0.921
    (MOVE_RED, MOVE_BOTH),                                   # no grammar rule for clause 2, 0.928
    (COMPOUND, f"{COMPOUND} and then go home"),              # 2 recorded clauses vs 3 asked, 0.920
])
def test_a_sequence_never_recalls_a_plan_recorded_for_fewer_clauses(tmp_path, habit, task):
    exp = _exp(tmp_path, (habit, [_grasp("red cube")]))
    assert exp.recall(habit) is not None                     # the habit itself is fine
    hit = exp.recall(task)
    assert hit is None, f"{task!r} recalled {hit['task']!r} (sim {hit['sim']}): half the command"


def test_a_single_instruction_never_recalls_a_recorded_sequence(tmp_path):
    """The symmetric hazard: one instruction replaying a recorded compound
    runs clauses nobody asked for (main: 0.929 -> both cubes moved)."""
    exp = _exp(tmp_path, (MOVE_BOTH, [*_move_recipe("red cube"), *_move_recipe("blue cube")]))
    assert exp.recall(MOVE_BOTH) is not None
    hit = exp.recall(MOVE_RED)
    assert hit is None, f"{MOVE_RED!r} recalled the sequence {hit['task']!r}"


def test_a_refused_sequence_does_not_hide_a_single_clause_habit_ranked_below_it(tmp_path):
    """A refused candidate is skipped like a demoted one, not a dead end: the
    recorded compound outranks the one-clause habit for this instruction
    (index 0.929 vs 0.924); the habit is what must be recalled."""
    near = f"{MOVE_RED} gently"
    exp = _exp(tmp_path,
               (MOVE_BOTH, [*_move_recipe("red cube"), *_move_recipe("blue cube")]),
               (near, [_grasp("red cube"), ("move_home", {})]))
    ranked = [exp._records[i]["task"] for _, i in exp._index.search(_embed(MOVE_RED), k=2)]
    assert ranked == [MOVE_BOTH, near]                       # premise: the compound ranks first
    hit = exp.recall(MOVE_RED)
    assert hit is not None and hit["task"] == near, hit


@pytest.mark.parametrize("query", [COMPOUND, f"please {RED}, then {BLUE}", f"{RED.upper()} THEN {BLUE}"])
def test_a_sequence_recorded_as_that_sequence_is_still_recalled(tmp_path, query):
    """(b) A compound habit recorded as that compound -- what a successful
    curriculum or LLM-tier run stores under the whole task -- still replays,
    including a paraphrase whose every clause clears the floor."""
    calls = [_grasp("red cube"), _grasp("blue cube")]
    exp = _exp(tmp_path, (COMPOUND, calls))
    hit = exp.recall(query)
    assert hit is not None and hit["task"] == COMPOUND
    assert [tuple(c) for c in hit["calls"]] == [(n, a) for n, a in calls]


@pytest.mark.parametrize("recorded, query, clause_cos", [
    ("pick up the blue cube and then pick up the red cube", COMPOUND, (0.75, 0.75)),
    ("wave and then go home", "go home and then wave", (0.0, 0.0)),
    ("pick up the red cube and put it in the bowl and then pick up the blue cube and put it in the bowl",
     "pick up the red cube and put it in the bowl and then pick up the blue cube and put it in the box",
     (1.0, 0.857)),
])
def test_a_recorded_sequence_must_match_clause_by_clause_in_order(tmp_path, recorded, query, clause_cos):
    """Same clause COUNT is not enough: the whole-text cosine is order-blind
    (index 0.994 for a reversed sequence) and dilutes a one-clause difference
    (bowl vs box: 0.958). Each clause must clear the same floor a single
    instruction clears, against its counterpart in order."""
    exp = _exp(tmp_path, (recorded, [("wave", {}), ("move_home", {})]))
    got = tuple(round(_cos(a, b), 3) for a, b in zip(split_subgoals(query), split_subgoals(recorded)))
    assert got == clause_cos                                 # premise: measured clause cosines
    hit = exp.recall(query)
    assert hit is None, f"{query!r} replayed {hit['task']!r} (sim {hit['sim']})"


def test_a_recipe_is_held_to_the_same_clause_rule(tmp_path):
    """Recipes (memory/recipes.py) share the index and the rule: an LLM-tier
    run stored for one clause is not replayed for a sequence."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}))
    rec = exp.recall(MOVE_RED)
    assert rec is not None and rec["kind"] == "recipe"       # unchanged for its own instruction
    assert exp.recall(MOVE_BOTH) is None


# ── the real FastPlanner ─────────────────────────────────────────────────


def test_the_roadmap_compound_runs_both_clauses_through_the_curriculum(tmp_path):
    """The whole-task recall falls through to the curriculum, which still
    warm-starts EACH clause from its own habit."""
    exp = _exp(tmp_path, (RED, [_grasp("red cube")]), (BLUE, [_grasp("blue cube")]))
    plan = FastPlanner(exp).plan(COMPOUND)
    assert plan is not None
    assert (plan.source, _calls(plan)) == ("curriculum", [_grasp("red cube"), _grasp("blue cube")]), plan.detail
    assert plan.subgoals == [RED, BLUE] and plan.subgoal_call_counts == [1, 1]
    assert plan.provenance == ["recalled", "recalled"]       # the warm start survives, per clause

    only_red = FastPlanner(_exp(tmp_path, (RED, [_grasp("red cube")]), name="e2.json")).plan(COMPOUND)
    assert (only_red.source, _calls(only_red)) == ("curriculum", [_grasp("red cube"), _grasp("blue cube")])
    assert only_red.provenance == ["recalled", "reflex"]


def test_a_compound_with_an_unplannable_clause_goes_to_the_llm_tier(tmp_path):
    """All-or-nothing: 'move the blue cube ...' has no habit and no grammar
    rule, so no fast plan exists -- the first clause's habit (0.928) must not
    stand in for the whole command."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}))
    assert FastPlanner(exp).plan(MOVE_RED).source == "experience"
    plan = FastPlanner(exp).plan(MOVE_BOTH)
    assert plan is None, f"{plan.source}: {plan.detail}"


@pytest.mark.parametrize("recorded, calls, task, expected", [
    (MOVE_BOTH, [*_move_recipe("red cube"), *_move_recipe("blue cube")], f"{MOVE_RED} and then go home", None),
    (PON_BOTH, [("pick_and_place", {"object": "cubo rojo", "destination": "bol"}),
                ("pick_and_place", {"object": "cubo azul", "destination": "bol"})],
     f"{PON_RED} y luego vuelve a casa",
     [("pick_and_place", {"object": "cubo rojo", "destination": "bol"}), ("move_home", {})]),
])
def test_a_clause_never_recalls_a_recorded_sequence(tmp_path, recorded, calls, task, expected):
    """The symmetric hazard inside `_plan_one`: the first clause matched the
    recorded compound (0.929 / 0.916) and the plan ran the OTHER cube too."""
    exp = _exp(tmp_path, (recorded, calls))
    plan = FastPlanner(exp).plan(task)
    if expected is None:
        assert plan is None, f"{plan.source}: {_calls(plan)}"
    else:
        assert plan is not None and plan.source == "curriculum"
        assert _calls(plan) == expected, plan.detail
        assert plan.provenance == ["reflex", "reflex"]


def test_a_reordered_sequence_runs_in_the_order_asked(tmp_path):
    exp = _exp(tmp_path, ("wave and then go home", [("wave", {}), ("move_home", {})]))
    plan = FastPlanner(exp).plan("go home and then wave")
    assert plan is not None and plan.source == "curriculum", plan and plan.detail
    assert [n for n, _ in plan.calls] == ["move_home", "wave"]


# ── AgentOrchestrator.run_task (MockLLM; fake runtime, then the mock stack) ─


class _Runtime:
    """Just enough runtime for `run_task`: every call is recorded with the
    tier that issued it and succeeds. The tier decision is under test here,
    not the skills (the mock-stack test below runs the real ones)."""

    robot_mode = None
    current_tier = None
    last_path = None

    def __init__(self):
        self.executed: list[tuple[str, dict, str | None]] = []
        self.trace = SimpleNamespace(finish=lambda text: None)
        self.memory = SimpleNamespace(add=lambda *a, **k: None, digest=lambda *a, **k: "(empty)")

    def execute(self, name, args):
        self.executed.append((name, dict(args), self.current_tier))
        if name == "task_done":
            return {"ok": True, "task_complete": True, "success": bool(args.get("success")),
                    "summary": args.get("summary", "")}
        if name == "localize_object":
            return {"ok": True, "label": args["label"], "position": [0.29, 0.0, 0.0375]}
        if name == "grasp_object":
            return {"ok": True, "held": args["label"]}
        return {"ok": True}

    def unverified_actions(self):
        return []


class _ExplodingLLM:
    supports_vision = False

    def chat(self, *a, **k):
        raise AssertionError("the LLM must not be called: both clauses have a fast plan")


def _agent(llm, runtime, exp):
    return AgentOrchestrator(llm, runtime, fast_planner=FastPlanner(exp), decompose=False,
                             attach_images=False, verify_milestones=False)


def _script(*calls):
    return MockLLM([LLMResponse(text="", tool_calls=[ToolCall(name, args)]) for name, args in calls])


def test_run_task_executes_every_clause_of_the_roadmap_compound(tmp_path):
    """Main replayed the red-cube habit alone and reported 'done via
    experience path'. Now the curriculum runs both clauses, LLM-free, and
    credits each clause as before."""
    exp = _exp(tmp_path, (RED, [_grasp("red cube")]))
    runtime = _Runtime()
    report = _agent(_ExplodingLLM(), runtime, exp).run_task(COMPOUND)
    assert report.path == "curriculum", f"{report.path}: {report.summary}"
    assert report.success
    assert [(n, a) for n, a, _ in runtime.executed] == [_grasp("red cube"), _grasp("blue cube")]
    assert {tier for *_, tier in runtime.executed} == {"curriculum"}
    assert runtime.last_path == "curriculum"
    # each clause is now a habit of its own; the compound is remembered whole
    assert exp.recall(BLUE)["task"] == BLUE
    assert exp.recall(COMPOUND)["task"] == COMPOUND


def test_run_task_hands_a_compound_with_an_unplannable_clause_to_the_llm(tmp_path):
    """Main grounded and replayed the first clause's recipe, never asked the
    LLM, and reported success with the blue cube untouched."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}))
    llm = _script(
        ("grasp_object", {"label": "red cube"}), ("place_at", {"x": 0.20, "y": -0.15}),
        ("grasp_object", {"label": "blue cube"}), ("place_at", {"x": 0.20, "y": -0.25}),
        ("task_done", {"success": True, "summary": "moved both cubes"}),
    )
    runtime = _Runtime()
    report = _agent(llm, runtime, exp).run_task(MOVE_BOTH)
    assert report.path == "llm", f"{report.path}: {report.summary}"
    assert llm.requests, "the LLM tier was never consulted"
    assert {tier for *_, tier in runtime.executed} == {"llm"}, runtime.executed  # no fast prefix ran first
    assert _grasp("blue cube") in [(n, a) for n, a, _ in runtime.executed]
    assert report.success and runtime.last_path == "llm"


def test_run_task_never_runs_a_recorded_sequence_for_one_of_its_clauses(tmp_path):
    """Main planned the first clause from the recorded compound and moved the
    blue cube too ('curriculum', 5 calls). That clause has no plan of its own,
    so the task goes to the LLM tier."""
    exp = _exp(tmp_path, (MOVE_BOTH, [*_move_recipe("red cube"), *_move_recipe("blue cube")],
                          {"summary": "both cubes front-left"}))
    llm = _script(
        ("grasp_object", {"label": "red cube"}), ("place_at", {"x": 0.20, "y": -0.15}),
        ("move_home", {}),
        ("task_done", {"success": True, "summary": "moved the red cube, then went home"}),
    )
    runtime = _Runtime()
    report = _agent(llm, runtime, exp).run_task(f"{MOVE_RED} and then go home")
    executed = [(n, a) for n, a, _ in runtime.executed]
    assert _grasp("blue cube") not in executed, f"{report.path}: {executed}"
    assert report.path == "llm" and llm.requests
    assert report.success


# ── the same, through the real SkillRuntime on the mock stack ────────────

_PORTS = {"CASCADE_GRASPGENX_PORT": "45001", "CASCADE_OCCUPANCY_PORT": "45002", "CASCADE_BRIDGE_PORT": "45003"}


def _confirm_effects(runtime) -> None:
    """Stand-in physics channel (as in test_task_recipes.py): the static mock
    camera can only leave a pick `unverified`, and the point here is what main
    did with a VERIFIED half-command -- report it as the whole task done."""
    from cascade.agent.effects import CONFIRMED, POSTCONDITIONS, Postcondition

    def verify(name, args, result, before=None):
        kind = POSTCONDITIONS.get(name)
        if kind is None:
            return None
        return Postcondition(name, kind, CONFIRMED, "stand-in physics channel (test)", channel="physics")

    runtime.effects.verify = verify


@needs_pin
def test_mock_stack_compound_is_not_answered_by_a_one_clause_recipe(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.skills.runtime import _MOTION_SKILLS

    for key, port in _PORTS.items():
        monkeypatch.setenv(key, port)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert 45000 <= int(cfg.grasp.graspgenx.port) <= 45099
    assert 45000 <= int(cfg.occupancy.port) <= 45099
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        arm.object_stop_frac = 0.5
        _confirm_effects(runtime)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        assert runtime.beliefs.find("red cube") is not None, "watcher never saw the mock cube"

        executed: list[str] = []
        real_execute = runtime.execute

        def spy(name, args):
            executed.append(name)
            return real_execute(name, args)

        runtime.execute = spy
        exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}))
        llm = _script(("task_done", {"success": False, "summary": "there is no blue cube on this table"}))
        agent = AgentOrchestrator(llm, runtime, decompose=False, max_steps=5, fast_planner=FastPlanner(exp))
        with runtime.watcher.paused():
            report = agent.run_task(MOVE_BOTH)
        assert report.path == "llm", f"{report.path}: {report.summary}"
        assert llm.requests, "the LLM tier was never consulted"
        assert not set(executed) & _MOTION_SKILLS, f"a fast prefix moved the arm: {executed}"
        assert runtime.held_object is None
        assert report.success is False
    finally:
        shutdown_runtime(runtime, arm)


# ── golden pins: a single instruction and the reflex are unchanged ───────


def _pre_b37_recall(exp, task, min_sim=0.9):
    """The tier-2 rule before B37, restated over the same index: top 3 by
    similarity, stop below the floor, first record with wins > losses."""
    for sim, i in exp._index.search(_embed(task, exp._dim), k=3):
        if sim < min_sim:
            break
        rec = exp._records[i]
        if rec["wins"] > rec["losses"]:
            return {"sim": round(sim, 3), **rec}
    return None


_SINGLE_HABITS = [
    (RED, [_grasp("red cube")]),
    (BLUE, [_grasp("blue cube")]),
    ("put the red cube in the bowl", [("pick_and_place", {"object": "red cube", "destination": "bowl"})]),
    ("hand me the toy dinosaur", [("handover", {"label": "toy dinosaur"})]),
    ("do the special calibration dance", [("move_home", {}), ("wave", {})]),
    (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}),
    (f"{MOVE_RED} gently", [_grasp("red cube"), ("move_home", {})]),
    ("throw the cup", [("throw", {"label": "cup", "direction": "forward"})]),
    ("push the green box left", [("push_object", {"label": "green box", "direction": "left"})]),
    ("push the small black bowl", [("push_object", {"label": "small black bowl", "direction": "forward"})]),
]

_SINGLE_QUERIES = [
    RED, BLUE, "pick up a red cube", "pick up the red cube now", "pick up the green cube",
    "put the red cube in the bowl", "put the red cube in the box", "please put the red cube in the bowl",
    "hand me that toy dinosaur", "hand me the toy dinosaur please",
    "do the special calibration dance", "please do that special calibration dance",
    MOVE_RED, f"{MOVE_RED} gently", f"{MOVE_RED} carefully", MOVE_BLUE,
    "throw the cup", "toss the cup", "push the green box left", "push the green box to the left",
    "push the small black bowl gently", "calibrate the camera", "wave", "",
]


def test_single_instruction_recall_is_the_pre_b37_rule(tmp_path):
    """(c) Differential golden pin: for every single-clause instruction the
    recall result (record, sim, counts) is what the pre-B37 rule returns over
    the same store -- including a demoted habit and near-duplicates."""
    exp = _exp(tmp_path, *_SINGLE_HABITS)
    exp.record("throw the cup", [], False, 1.0)
    exp.record("throw the cup", [], False, 1.0)              # demoted: 1 win, 2 losses
    hits = 0
    for query in _SINGLE_QUERIES:
        assert len(split_subgoals(query)) <= 1
        got, want = exp.recall(query), _pre_b37_recall(exp, query)
        assert got == want, query
        hits += got is not None
    assert hits >= 10                                        # not a vacuous comparison
    assert exp.recall("throw the cup") is None               # the demoted habit stays unrecalled


def test_a_single_instruction_is_decided_by_the_index_similarity_alone(tmp_path):
    """(c) For a single clause B37 computes no second similarity: the index
    estimate decides, exactly as before. Real pair (found by searching the
    real index): it scores 0.902 although the exact cosine is 0.894, so an
    exact-cosine gate on single instructions would silently drop this habit."""
    habit, query = "push the small black bowl", "push the small black bowl gently"
    exp = _exp(tmp_path, (habit, [("push_object", {"label": "small black bowl", "direction": "forward"})]))
    assert _cos(query, habit) == pytest.approx(0.894, abs=0.001)
    (sim, _), = exp._index.search(_embed(query), k=1)
    assert sim >= 0.9, sim                                   # 0.9024 on this index
    hit = exp.recall(query)
    assert hit is not None and hit["task"] == habit
    assert hit == _pre_b37_recall(exp, query)


def test_reflex_and_single_instruction_plans_are_unchanged(tmp_path):
    """(c) FastPlanner golden pins (values taken on main 4e896c3): the
    grammar still wins over a habit recorded under the same text, a
    single-clause habit or recipe still replays, a sequence of reflexes and
    a warm-started sequence still compile clause by clause."""
    exp = _exp(tmp_path, *_SINGLE_HABITS, ("pick and place pink object", [("move_home", {})]))
    planner = FastPlanner(exp)

    def got(task):
        plan = planner.plan(task)
        return None if plan is None else (plan.source, _calls(plan), plan.provenance)

    assert got("pick and place pink object") == (
        "reflex", [("pick_and_place", {"object": "pink object"})], ["reflex"])
    assert got("go home") == ("reflex", [("move_home", {})], ["reflex"])
    assert got(RED) == ("reflex", [_grasp("red cube")], ["reflex"])
    assert got("hand me that toy dinosaur") == ("reflex", [("handover", {"label": "that toy dinosaur"})], ["reflex"])
    assert got("please do that special calibration dance") == (
        "experience", [("move_home", {}), ("wave", {})], ["recalled", "recalled"])
    recipe = planner.plan(MOVE_RED)
    assert (recipe.source, _calls(recipe), recipe.summary) == (
        "experience", _move_recipe("red cube"), "front-left placement")
    assert got("calibrate the camera") is None
    assert got("wave and then go home") == (
        "curriculum", [("wave", {}), ("move_home", {})], ["reflex", "reflex"])
    assert got("do the special calibration dance and then go home") == (
        "curriculum", [("move_home", {}), ("wave", {}), ("move_home", {})], ["recalled", "recalled", "reflex"])
