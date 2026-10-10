"""A bare "and" that begins a new imperative clause is a clause boundary (B69).

ROADMAP #7 / ARCHITECTURE Known limitations, left open by B37: the fast tier
cut clauses only at SEQUENCE connectives (`split_subgoals`: "and then",
"then", "luego", ...), so a bare "and" was no boundary anywhere in it.
"move the red cube to the front-left of the table and move the blue cube to
the front-left of the table" was one clause, scored 0.951 against the habit
for its first command, and `FastPlanner` replayed that habit -- half the
instruction, reported as success. The reflex grammar, which runs before
tier 2, had the same blind spot: its lazy object/destination groups swallow a
second clause ("put the red cube in the bowl and put the blue cube in the
bowl" -> one pick_and_place whose destination is "bowl and put the blue cube
in the bowl").

The rule pinned here (`split_subgoals`, consumed by `parse_command`, by
`ExperienceMemory.recall` through B37's clause rule and by the curriculum):
after the unchanged sequence split, a coordinated "and"/"y" (optionally
"and also"/"y también", optionally after a comma) followed by a CLAUSE VERB of
the reflex/skill vocabulary starts a new clause -- except

* verb coordination: the word before "and" is a verb that takes an object
  ("pick and place the cube", "pick up and place ...", "open and close the
  gripper"): two verbs, one object, one clause;
* a back-reference: the verb is followed by it/them/this/that/these/those
  ("pick up the cube and put it in the box", "grab the banana and throw it");
* an object-taking verb with no object of its own (nothing, a preposition or
  a direction follows: "grab the banana and throw to the left").

Noun coordination never splits ("the red and blue cube": no verb follows).
Default-on, like B37: every command without such a clause is byte-identical
(golden pins at the bottom, differential against the pre-B69 rule restated
here); a command with one is planned exactly like the same clauses joined by
"then" (curriculum, all-or-nothing) or handed to the LLM tier -- never a
one-clause plan.

Mock stack only: no MuJoCo, no network; the one runtime built here dials
ports in 47100-47199.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pytest

from conftest import needs_pin

from cascade.agent.orchestrator import AgentOrchestrator
from cascade.agent.reflex import _RULES, ExperienceMemory, FastPlanner, _embed, parse_command, split_subgoals
from test_tier2_clause_structure import (
    _ExplodingLLM,
    _Runtime,
    _agent,
    _calls,
    _confirm_effects,
    _exp,
    _grasp,
    _move_recipe,
    _script,
)

MOVE_RED = "move the red cube to the front-left of the table"
MOVE_BLUE = "move the blue cube to the front-left of the table"
MOVE_BOTH = f"{MOVE_RED} and {MOVE_BLUE}"                    # the ROADMAP instruction
RED = "pick up the red cube"
BLUE = "pick up the blue cube"
PUT_RED = "put the red cube in the bowl"
PUT_BLUE = "put the blue cube in the bowl"
PON_RED = "pon el cubo rojo en el bol"
PON_BLUE = "pon el cubo azul en el bol"
WAVE_MOVE = f"wave and {MOVE_RED}"


def _cos(a, b):
    return float(np.dot(_embed(a), _embed(b)))


def _index_sim(exp, task):
    (sim, _), = exp._index.search(_embed(task), k=1)
    return sim


def _p_and_p(obj, dest):
    return ("pick_and_place", {"object": obj, "destination": dest})


# ── the pre-B69 rule, restated (golden pins below compare against it) ─────

_PRE_B69_SEQUENCE = re.compile(
    r"\s*(?:,\s*)?\b(?:and\s+then|after\s+that|then|luego|despu[eé]s|"
    r"y\s+luego|y\s+despu[eé]s)\b\s*",
    re.IGNORECASE,
)


def _pre_b69_split(task):
    parts = [p.strip(" ,.") for p in _PRE_B69_SEQUENCE.split(task.strip())]
    return [p for p in parts if p]


def _pre_b69_recall(exp, task, min_sim=0.9):
    """B37's recall over the same index, with the pre-B69 clause split."""
    def same(want, have):
        a, b = _pre_b69_split(want), _pre_b69_split(have)
        if len(a) != len(b):
            return False
        return len(a) < 2 or all(_cos(x, y) >= min_sim for x, y in zip(a, b))

    for sim, i in exp._index.search(_embed(task, exp._dim), k=3):
        if sim < min_sim:
            break
        rec = exp._records[i]
        if rec["wins"] > rec["losses"] and same(task, rec["task"]):
            return {"sim": round(sim, 3), **rec}
    return None


# ── premises: the cause, measured (pass before and after B69) ────────────


def test_premise_one_clause_habits_clear_the_floor_for_an_and_compound(tmp_path):
    """The index puts a one-clause habit above 0.9 for the two-command
    instruction (the ROADMAP's 0.951), in both directions and both languages:
    similarity alone cannot tell the compound from its first command."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube")))
    assert _index_sim(exp, MOVE_BOTH) == pytest.approx(0.951, abs=0.001)
    assert _cos(MOVE_BOTH, MOVE_RED) == pytest.approx(0.957, abs=0.001)
    assert _cos(f"{PUT_RED} and {PUT_BLUE}", PUT_RED) == pytest.approx(0.949, abs=0.001)
    assert _cos(f"{PON_RED} y {PON_BLUE}", PON_RED) == pytest.approx(0.949, abs=0.001)
    assert _cos(f"{RED} and {BLUE}", RED) == pytest.approx(0.935, abs=0.001)
    assert _cos(WAVE_MOVE, MOVE_RED) == pytest.approx(0.926, abs=0.001)
    back = _exp(tmp_path / "back", (MOVE_BOTH, _move_recipe("red cube")))
    assert _index_sim(back, MOVE_RED) == pytest.approx(0.947, abs=0.001)


def test_premise_b37_refuses_the_same_compound_once_its_clauses_are_cut(tmp_path):
    """B37's clause rule already refuses the one-clause habit when the two
    commands are joined by 'and then': the gap was only where clauses are cut."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube")))
    assert exp.recall(MOVE_RED) is not None
    assert exp.recall(f"{MOVE_RED} and then {MOVE_BLUE}") is None


@pytest.mark.parametrize("text, intent, group, swallowed", [
    (f"{PUT_RED} and {PUT_BLUE}", "pick_and_place", "dest", "bowl and put the blue cube in the bowl"),
    (f"{RED} and {BLUE}", "pick", "obj", "red cube and pick up the blue cube"),
    (f"{PON_RED} y {PON_BLUE}", "pick_and_place", "dest", "bol y pon el cubo azul en el bol"),
    (f"{RED} and wave", "pick", "obj", "red cube and wave"),
])
def test_premise_the_grammar_rules_swallow_a_second_clause(text, intent, group, swallowed):
    """The first grammar rule that matches (parse_command's order) puts the
    whole second command into one argument: the rules are not clause-aware,
    so the refusal has to happen before them."""
    rule, got = next((r, i) for r, i in _RULES if r.match(text))
    assert got == intent
    assert rule.match(text).group(group) == swallowed


# ── split_subgoals: where a bare "and" begins a clause ────────────────────


@pytest.mark.parametrize("text, clauses", [
    (MOVE_BOTH, [MOVE_RED, MOVE_BLUE]),
    (f"{MOVE_RED}, and {MOVE_BLUE}", [MOVE_RED, MOVE_BLUE]),
    (f"{MOVE_RED} and also {MOVE_BLUE}", [MOVE_RED, MOVE_BLUE]),
    (f"{RED} and {BLUE}", [RED, BLUE]),
    (f"{RED} and grab a blue cube", [RED, "grab a blue cube"]),
    (f"{PUT_RED} and {PUT_BLUE}", [PUT_RED, PUT_BLUE]),
    (f"{PON_RED} y {PON_BLUE}", [PON_RED, PON_BLUE]),
    (f"{PON_RED} y también {PON_BLUE}", [PON_RED, PON_BLUE]),
    (f"{RED} and wave", [RED, "wave"]),
    (WAVE_MOVE, ["wave", MOVE_RED]),                         # an object-less verb before "and"
    ("wave and go home", ["wave", "go home"]),
    ("saluda y vuelve a casa", ["saluda", "vuelve a casa"]),
    (f"{RED} and point at the bowl", [RED, "point at the bowl"]),
    ("put the cube down and pick up the ball", ["put the cube down", "pick up the ball"]),
    ("grab the banana and throw the apple", ["grab the banana", "throw the apple"]),
    ("pick up the cube and put it in the box and wave", ["pick up the cube and put it in the box", "wave"]),
    (f"{RED} and {BLUE} and go home", [RED, BLUE, "go home"]),
    (f"{RED} and {BLUE} and then go home", [RED, BLUE, "go home"]),
    (f"PUT THE RED CUBE IN THE BOWL AND {PUT_BLUE.upper()}", ["PUT THE RED CUBE IN THE BOWL", PUT_BLUE.upper()]),
])
def test_a_coordinated_and_before_a_clause_verb_is_a_boundary(text, clauses):
    assert split_subgoals(text) == clauses


@pytest.mark.parametrize("text", [
    "the red and blue cube",                                 # noun coordination: no verb follows
    "put the red and blue cubes in the bowl",
    "pon el cubo rojo y azul en el bol",
    "pick up the salt and pepper",
    "move the red cube and the blue cube to the bowl",
    "hand me the cup and saucer",
    "pick and place pink object",                            # verb coordination: one object
    "pick and place the red cube",
    "PICK AND PLACE THE RED CUBE",
    "pick up and place the red cube in the bowl",
    "open and close the gripper",
    "coge y pon el cubo en el bol",
    "pick up the cube and put it in the box",                # back-reference: same object
    "pick up the red cube and place it in the bowl",
    "pick and place the banana and save it in the box",
    "grab the banana and throw it",
    "grab the banana and throw it to the left",
    "pick up the red cube and set it down",
    "pick up the cube and put that in the box",
    "pick up the cubes and put them in the bowl",
    "grab the banana and throw",                             # no object of its own
    "grab the banana and throw to the left",
    "grab the banana and throw left",
    "pick up the red cube and place in the bowl",
    "pick up the cup and set down on the tray",
    "coge el plátano y lánzalo",                             # Spanish clitic: enclitic, not a clause verb
    "coge el cubo y ponlo en la caja",
])
def test_one_clause_with_an_and_is_not_cut(text):
    assert split_subgoals(text) == [text] == _pre_b69_split(text)


# ── tier-2 recall through the real ExperienceMemory ──────────────────────


@pytest.mark.parametrize("habit, task", [
    (MOVE_RED, MOVE_BOTH),                                   # the ROADMAP case, 0.951
    (MOVE_RED, f"{MOVE_RED}, and {MOVE_BLUE}"),
    (MOVE_RED, f"{MOVE_RED} and also {MOVE_BLUE}"),          # 0.929
    (PUT_RED, f"{PUT_RED} and {PUT_BLUE}"),                  # 0.944
    (PON_RED, f"{PON_RED} y {PON_BLUE}"),                    # 0.945
    (RED, f"{RED} and {BLUE}"),                              # 0.934
    (MOVE_RED, WAVE_MOVE),                                   # the wave would be dropped
])
def test_an_and_compound_never_recalls_a_one_clause_habit(tmp_path, habit, task):
    exp = _exp(tmp_path, (habit, _move_recipe("red cube")))
    assert exp.recall(habit) is not None                     # the habit itself is fine
    hit = exp.recall(task)
    assert hit is None, f"{task!r} recalled {hit['task']!r} (sim {hit['sim']}): half the command"


def test_a_single_command_never_recalls_a_recorded_and_compound(tmp_path):
    """The symmetric hazard: main replayed both cubes' moves for the red one
    alone (index 0.947)."""
    exp = _exp(tmp_path, (MOVE_BOTH, [*_move_recipe("red cube"), *_move_recipe("blue cube")]))
    assert exp.recall(MOVE_BOTH) is not None
    hit = exp.recall(MOVE_RED)
    assert hit is None, f"{MOVE_RED!r} recalled the compound {hit['task']!r}"


@pytest.mark.parametrize("query", [MOVE_BOTH, f"{MOVE_RED}, and {MOVE_BLUE}", f"{MOVE_RED} and then {MOVE_BLUE}"])
def test_a_recorded_and_compound_is_recalled_for_the_same_clauses(tmp_path, query):
    """A compound stored as that compound still replays -- also for the same
    two clauses joined by 'then', which main refused as another clause count."""
    calls = [*_move_recipe("red cube"), *_move_recipe("blue cube")]
    exp = _exp(tmp_path, (MOVE_BOTH, calls))
    hit = exp.recall(query)
    assert hit is not None and hit["task"] == MOVE_BOTH
    assert [tuple(c) for c in hit["calls"]] == [(n, a) for n, a in calls]


def test_a_recorded_and_compound_must_match_clause_by_clause_in_order(tmp_path):
    """Reversed, the compound is the same bag of words (main: recalled, so the
    cubes moved in the wrong order); B37's in-order clause rule now applies."""
    exp = _exp(tmp_path, (f"{PUT_RED} and {PUT_BLUE}", [_p_and_p("red cube", "bowl"), _p_and_p("blue cube", "bowl")]))
    assert _index_sim(exp, f"{PUT_BLUE} and {PUT_RED}") >= 0.99
    assert exp.recall(f"{PUT_BLUE} and {PUT_RED}") is None


# ── the real FastPlanner ─────────────────────────────────────────────────


def test_the_roadmap_compound_with_one_habit_goes_to_the_llm_tier(tmp_path):
    """All-or-nothing: the blue command has no habit and no grammar rule, so
    there is no fast plan -- main replayed the red recipe alone."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}))
    assert FastPlanner(exp).plan(MOVE_RED).source == "experience"
    plan = FastPlanner(exp).plan(MOVE_BOTH)
    assert plan is None, f"{plan.source}: {_calls(plan)}"


def test_the_roadmap_compound_runs_both_habits_through_the_curriculum(tmp_path):
    exp = _exp(tmp_path,
               (MOVE_RED, _move_recipe("red cube"), {"summary": "red front-left"}),
               (MOVE_BLUE, _move_recipe("blue cube"), {"summary": "blue front-left"}))
    plan = FastPlanner(exp).plan(MOVE_BOTH)
    assert plan is not None and plan.source == "curriculum", plan and plan.detail
    assert _calls(plan) == [*_move_recipe("red cube"), *_move_recipe("blue cube")]
    assert plan.subgoals == [MOVE_RED, MOVE_BLUE] and plan.subgoal_call_counts == [2, 2]
    assert plan.provenance == ["recalled"] * 4
    assert plan.needs_grounding                              # recipes are re-grounded before motion


@pytest.mark.parametrize("task, calls", [
    (f"{PUT_RED} and {PUT_BLUE}", [_p_and_p("red cube", "bowl"), _p_and_p("blue cube", "bowl")]),
    (f"{RED} and {BLUE}", [_grasp("red cube"), _grasp("blue cube")]),
    (f"{PON_RED} y {PON_BLUE}", [_p_and_p("cubo rojo", "bol"), _p_and_p("cubo azul", "bol")]),
    (f"{RED} and wave", [_grasp("red cube"), ("wave", {})]),
    (f"{RED} and move up", [_grasp("red cube"), ("move_relative", {"direction": "up"})]),
    ("wave and go home", [("wave", {}), ("move_home", {})]),
])
def test_a_reflex_never_swallows_a_second_clause(task, calls):
    """Main compiled each of these into ONE call carrying the second command
    inside an argument. Now the grammar refuses the compound and the
    curriculum compiles each clause."""
    assert parse_command(task) is None
    plan = FastPlanner().plan(task)
    assert plan is not None and plan.source == "curriculum"
    assert _calls(plan) == calls and plan.provenance == ["reflex", "reflex"]


@pytest.mark.parametrize("first, second", [(PUT_RED, PUT_BLUE), (RED, BLUE), (PON_RED, PON_BLUE),
                                           (MOVE_RED, MOVE_BLUE), ("wave", MOVE_RED)])
def test_an_and_compound_is_planned_exactly_like_the_then_compound(tmp_path, first, second):
    """The fast tier treats 'A and B' as it already treated 'A and then B'
    (warm starts included): same plan, same sub-goals, same provenance."""
    exp = _exp(tmp_path, (RED, [_grasp("red cube")]),
               (MOVE_RED, _move_recipe("red cube"), {"summary": "red front-left"}),
               (MOVE_BLUE, _move_recipe("blue cube"), {"summary": "blue front-left"}))
    planner = FastPlanner(exp)
    joined = "y" if first.startswith("pon") else "and"
    then = "y luego" if first.startswith("pon") else "and then"

    def shape(plan):
        return None if plan is None else (plan.source, _calls(plan), plan.subgoals,
                                          plan.subgoal_call_counts, plan.provenance)

    got = shape(planner.plan(f"{first} {joined} {second}"))
    assert got is not None and got == shape(planner.plan(f"{first} {then} {second}"))


def test_an_object_less_first_verb_is_not_dropped(tmp_path):
    """'wave and <move habit>': main replayed the move recipe (0.926) and
    never waved."""
    exp = _exp(tmp_path, (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}))
    plan = FastPlanner(exp).plan(WAVE_MOVE)
    assert plan is not None and plan.source == "curriculum", plan and plan.detail
    assert _calls(plan) == [("wave", {}), *_move_recipe("red cube")]
    assert plan.provenance == ["reflex", "recalled", "recalled"]


# ── AgentOrchestrator.run_task (MockLLM; fake runtime, then the mock stack) ─


def test_run_task_hands_the_roadmap_compound_to_the_llm(tmp_path):
    """Main grounded and replayed the red recipe, never asked the LLM and
    reported success with the blue cube untouched."""
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


def test_run_task_executes_both_reflex_clauses_without_the_llm(tmp_path):
    """Main ran ONE pick_and_place toward the destination 'bowl and put the
    blue cube in the bowl'. Now both clauses run, LLM-free, and each is
    remembered as a habit of its own."""
    exp = _exp(tmp_path)
    runtime = _Runtime()
    task = f"{PUT_RED} and {PUT_BLUE}"
    report = _agent(_ExplodingLLM(), runtime, exp).run_task(task)
    assert report.path == "curriculum", f"{report.path}: {report.summary}"
    assert report.success
    assert [(n, a) for n, a, _ in runtime.executed] == [_p_and_p("red cube", "bowl"), _p_and_p("blue cube", "bowl")]
    assert {tier for *_, tier in runtime.executed} == {"curriculum"}
    assert exp.recall(PUT_BLUE)["task"] == PUT_BLUE
    assert exp.recall(task)["task"] == task


_PORTS = {"CASCADE_GRASPGENX_PORT": "47111", "CASCADE_OCCUPANCY_PORT": "47112", "CASCADE_BRIDGE_PORT": "47113"}


@needs_pin
def test_mock_stack_and_compound_is_not_answered_by_a_one_clause_recipe(tmp_path, monkeypatch):
    """Through the real SkillRuntime: no fast prefix moves the arm before the
    LLM tier sees the whole instruction."""
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    from cascade.skills.runtime import _MOTION_SKILLS

    for key, port in _PORTS.items():
        monkeypatch.setenv(key, port)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert 47100 <= int(cfg.grasp.graspgenx.port) <= 47199
    assert 47100 <= int(cfg.occupancy.port) <= 47199
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


# ── golden pins: commands without a clause-starting "and" are unchanged ──

#: parse_command on main 38f6d08 for the grammar's own one-clause "and"s.
_REFLEX_GOLDEN = {
    "pick and place pink object": [("pick_and_place", {"object": "pink object"})],
    "pick up and place the red cube in the bowl": [_p_and_p("red cube", "bowl")],
    "pick up the cube and put it in the box": [_p_and_p("cube", "box")],
    "pick up the red cube and place in the bowl": [_p_and_p("red cube", "bowl")],
    "pick and place the banana and save it in the box": [_p_and_p("banana", "box")],
    "grab the banana and throw it": [("throw", {"direction": "forward", "label": "banana"})],
    "grab the banana and throw": [("throw", {"direction": "forward", "label": "banana"})],
    "grab the banana and throw to the left": [("throw", {"direction": "left", "label": "banana"})],
    "coge el plátano y lánzalo": [("throw", {"direction": "forward", "label": "plátano"})],
    "coge el cubo y ponlo en la caja": [_p_and_p("cubo", "caja")],
    "coge y pon el cubo en el bol": [_p_and_p("cubo", "bol")],
    "put the red and blue cubes in the bowl": [_p_and_p("red and blue cubes", "bowl")],
    "hand me the cup and saucer": [("handover", {"label": "cup and saucer"})],
    "count the red and blue cubes": [("count_objects", {"query": "red and blue cubes"})],
}


@pytest.mark.parametrize("text", sorted(_REFLEX_GOLDEN))
def test_the_grammars_one_clause_ands_compile_as_before(text):
    plan = parse_command(text)
    assert plan is not None and plan.calls == _REFLEX_GOLDEN[text]
    fast = FastPlanner().plan(text)
    assert (fast.source, fast.calls) == ("reflex", _REFLEX_GOLDEN[text])


_HABITS = [
    (MOVE_RED, _move_recipe("red cube"), {"summary": "front-left placement"}),
    (RED, [_grasp("red cube")]),
    (PUT_RED, [_p_and_p("red cube", "bowl")]),
    ("pick up the salt and pepper", [_grasp("salt and pepper")]),
    ("move the red cube and the blue cube to the bowl", [_p_and_p("red cube", "bowl"), _p_and_p("blue cube", "bowl")]),
    ("open and close the gripper", [("open_gripper", {}), ("close_gripper", {})]),
    (f"{RED} and then {BLUE}", [_grasp("red cube"), _grasp("blue cube")]),
    ("grab the banana and throw it", [("throw", {"label": "banana", "direction": "forward"})]),
]

_QUERIES = [
    MOVE_RED, f"{MOVE_RED} gently", MOVE_BLUE, RED, "pick up a red cube", PUT_RED, "please put the red cube in the bowl",
    "pick up the salt and pepper", "pick up the salt and pepper please", "pick up the pepper and salt",
    "move the red cube and the blue cube to the bowl", "move the blue cube and the red cube to the bowl",
    "open and close the gripper", "open and close the gripper twice",
    f"{RED} and then {BLUE}", f"{RED}, then {BLUE}", f"{BLUE} and then {RED}",
    "grab the banana and throw it", "grab the banana and throw it far", "the red and blue cube", "wave", "",
]


def test_recall_without_a_clause_starting_and_is_the_pre_b69_rule(tmp_path):
    """Differential golden pin: for every query with no clause-starting 'and',
    the clauses and the recall result (record, sim, counts) are what the
    pre-B69 rule gives over the same store -- habits with noun and verb
    coordination included."""
    exp = _exp(tmp_path, *_HABITS)
    hits = 0
    for query in _QUERIES:
        assert split_subgoals(query) == _pre_b69_split(query), query
        got, want = exp.recall(query), _pre_b69_recall(exp, query)
        assert got == want, query
        hits += got is not None
    assert hits >= 10                                        # not a vacuous comparison


def test_fast_plans_without_a_clause_starting_and_are_unchanged(tmp_path):
    """FastPlanner golden pins (values taken on main 38f6d08)."""
    planner = FastPlanner(_exp(tmp_path, *_HABITS))

    def got(task):
        plan = planner.plan(task)
        return None if plan is None else (plan.source, _calls(plan), plan.provenance)

    assert got("pick up the salt and pepper please") == (
        "reflex", [_grasp("salt and pepper please")], ["reflex"])
    assert got("open and close the gripper") == (
        "experience", [("open_gripper", {}), ("close_gripper", {})], ["recalled", "recalled"])
    assert got("move the blue cube and the red cube to the bowl") == (
        "experience", [_p_and_p("red cube", "bowl"), _p_and_p("blue cube", "bowl")], ["recalled", "recalled"])
    assert got("grab the banana and throw it far") == (
        "reflex", [("throw", {"direction": "forward", "label": "banana"})], ["reflex"])
    assert got("wave and then go home") == ("curriculum", [("wave", {}), ("move_home", {})], ["reflex", "reflex"])
    assert got(f"{RED} and then {BLUE}") == (
        "experience", [_grasp("red cube"), _grasp("blue cube")], ["recalled", "recalled"])
    assert got(f"{BLUE} and then {RED}") == (
        "curriculum", [_grasp("blue cube"), _grasp("red cube")], ["reflex", "recalled"])
    assert got("the red and blue cube") is None
