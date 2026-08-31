"""Persistent spatial memory + curriculum/warm-start fast planning.

Two ROADMAP items, tested at the level where they can silently rot:

* `BeliefStore.save/load` -- the world model survives a restart. The trap is
  that every belief timestamp is `time.monotonic()`, whose origin resets with
  the process: a naive round-trip yields negative ages and reports a stale
  belief as `visible`. These tests pin the wall-clock conversion AND the
  never-visible-on-load rule, because the failure is silent and looks
  authoritative.

* `FastPlanner` curriculum decomposition + warm start (Agentic-VLA at
  inference time). The trap is the one `parse_command` refuses compound
  commands to avoid: executing part of a sequence and reporting success.
"""

from __future__ import annotations

import json
import time

import numpy as np

from cascade.agent.reflex import (
    ExperienceMemory, FastPlanner, split_subgoals,
)
from cascade.memory import BeliefStore


# ── persistent spatial memory ────────────────────────────────────────────


def _seed(store: BeliefStore) -> None:
    store.update("red cube", np.array([0.2, 0.1, 0.03]), conf=0.9,
                 extent=np.array([0.05, 0.05, 0.05]), color="red")
    store.update("blue bowl", np.array([0.3, -0.1, 0.02]), conf=0.7,
                 color="blue")


def test_beliefs_survive_a_restart(tmp_path):
    path = tmp_path / "beliefs.json"
    store = BeliefStore()
    _seed(store)
    assert store.save(path) == 2

    restored = BeliefStore()          # a fresh process's store
    assert restored.load(path) == 2
    got = {b.label: b for b in restored.all()}
    assert set(got) == {"red cube", "blue bowl"}
    np.testing.assert_allclose(got["red cube"].position, [0.2, 0.1, 0.03])
    assert got["red cube"].color == "red"
    assert got["blue bowl"].conf == 0.7


def test_restored_beliefs_are_remembered_not_visible(tmp_path):
    """The load-bearing rule. A belief restored from disk has NOT been seen
    this session; reporting it as `visible` would have the robot claim to see
    an object it never looked at -- worse than no memory, because the agent
    acts on it without re-observing."""
    path = tmp_path / "beliefs.json"
    store = BeliefStore()
    _seed(store)
    store.save(path)

    restored = BeliefStore()
    restored.load(path)
    now = time.monotonic()
    for b in restored.all():
        assert b.state(now) == "remembered", f"{b.label} came back as visible"
    for row in restored.summary():
        assert row["state"] == "remembered"
        # and never a negative/absurd age, the monotonic round-trip bug
        assert row["age_s"] >= 0.0


def test_ages_are_wall_clock_not_monotonic(tmp_path):
    """An object seen an hour ago must still read as an hour ago after a
    restart. Storing raw monotonic values makes this arbitrary."""
    path = tmp_path / "beliefs.json"
    store = BeliefStore()
    store.update("mug", np.array([0.25, 0.0, 0.03]), conf=0.8)
    # Rewrite the record on disk as if it had been saved an hour ago.
    store.save(path)
    blob = json.loads(path.read_text())
    blob["beliefs"][0]["last_seen_wall"] -= 3600.0
    blob["beliefs"][0]["first_seen_wall"] -= 3600.0
    path.write_text(json.dumps(blob))

    restored = BeliefStore()
    assert restored.load(path) == 1
    age = restored.summary()[0]["age_s"]
    assert 3550 < age < 3650, f"age {age} s does not reflect wall-clock time"


def test_stale_beliefs_are_dropped_on_load(tmp_path):
    """A day-old tabletop is not evidence."""
    path = tmp_path / "beliefs.json"
    store = BeliefStore()
    store.update("ghost", np.array([0.2, 0.0, 0.03]), conf=0.9)
    store.save(path)
    blob = json.loads(path.read_text())
    blob["beliefs"][0]["last_seen_wall"] -= 48 * 3600.0
    path.write_text(json.dumps(blob))

    restored = BeliefStore()
    assert restored.load(path, max_age_s=6 * 3600.0) == 0
    assert restored.all() == []


def test_a_corrupt_memory_file_never_blocks_startup(tmp_path):
    bad = tmp_path / "beliefs.json"
    bad.write_text("{not json at all")
    store = BeliefStore()
    assert store.load(bad) == 0          # no exception
    assert store.load(tmp_path / "missing.json") == 0


def test_saving_is_atomic(tmp_path):
    """A half-written file is a corrupt world model on the next boot; the
    save goes through a temp file + replace, so no partial file is left."""
    path = tmp_path / "beliefs.json"
    store = BeliefStore()
    _seed(store)
    store.save(path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "beliefs.json"]
    assert leftovers == [], f"temp files left behind: {leftovers}"
    json.loads(path.read_text())  # parses


def test_restored_beliefs_are_queryable_by_color(tmp_path):
    """Persistence is only useful if the RECALLED world answers the same
    queries the live one does."""
    path = tmp_path / "beliefs.json"
    store = BeliefStore()
    _seed(store)
    store.save(path)

    restored = BeliefStore()
    restored.load(path)
    hit = restored.find("red object")
    assert hit is not None and hit.label == "red cube"


# ── curriculum decomposition + warm start ────────────────────────────────


def test_split_only_breaks_on_sequence_connectives():
    assert split_subgoals("wave and then go home") == ["wave", "go home"]
    assert split_subgoals("saluda y luego vuelve a casa") == [
        "saluda", "vuelve a casa"
    ]
    # A bare "and" inside one motion must NOT split: this is a single
    # pick_and_place the grammar already handles, and splitting it would
    # produce two half-motions.
    one = "pick up the cube and put it in the box"
    assert split_subgoals(one) == [one]


def test_a_sequence_of_reflexes_no_longer_needs_the_llm():
    """`parse_command` refuses compound commands (correctly -- running only
    the first clause would silently drop the rest), which sent genuinely
    routine sequences to the slow path. The curriculum tier plans them."""
    planner = FastPlanner()
    plan = planner.plan("wave and then go home")
    assert plan is not None, "a sequence of two reflexes still escaped to the LLM"
    assert plan.source == "curriculum"
    assert plan.subgoals == ["wave", "go home"]
    assert [c for c, _ in plan.calls] == ["wave", "move_home"]


def test_an_unplannable_clause_aborts_the_whole_fast_plan():
    """All-or-nothing. Executing the prefix and reporting success is exactly
    the failure the compound-command refusal exists to prevent."""
    planner = FastPlanner()
    plan = planner.plan("wave and then compose a haiku about the cube")
    assert plan is None


def test_a_proven_subgoal_warm_starts_a_later_task(tmp_path):
    """The Agentic-VLA property that makes decomposition pay off: experience
    recorded for a CLAUSE is retrieved when that clause reappears, including
    inside a different sequence."""
    exp = ExperienceMemory(tmp_path / "exp.json")
    # A bespoke plan for a phrase the grammar cannot parse.
    exp.record("do the special calibration dance",
               [("move_home", {}), ("wave", {})], True, 2.0)

    planner = FastPlanner(exp)
    plan = planner.plan("do the special calibration dance and then go home")
    assert plan is not None, "the proven sub-goal did not warm-start the sequence"
    assert plan.source == "curriculum"
    assert plan.warm_started
    assert plan.provenance[0] == "recalled"
    assert [c for c, _ in plan.calls] == ["move_home", "wave", "move_home"]


def test_subgoal_spans_credit_the_right_clause(tmp_path):
    """A clause can compile to several calls, so outcome credit must follow
    the recorded span rather than a positional guess."""
    exp = ExperienceMemory(tmp_path / "exp.json")
    exp.record("do the thing", [("move_home", {}), ("wave", {})], True, 2.0)
    planner = FastPlanner(exp)
    plan = planner.plan("do the thing and then go home")
    assert plan is not None
    spans = list(plan.subgoal_spans())
    assert [name for name, _ in spans] == ["do the thing", "go home"]
    assert [len(calls) for _, calls in spans] == [2, 1]
    # every call is accounted for exactly once
    assert sum(len(c) for _, c in spans) == len(plan.calls)


def test_recording_a_subgoal_makes_it_reusable_elsewhere(tmp_path):
    """What compounds: a clause proven inside one sequence is a warm start for
    a DIFFERENT sequence later."""
    exp = ExperienceMemory(tmp_path / "exp.json")
    planner = FastPlanner(exp)
    planner.note_subgoal_outcome(
        "tidy the workspace", [("move_home", {})], True, 1.0
    )
    plan = planner.plan("wave and then tidy the workspace")
    assert plan is not None
    assert plan.warm_started
    assert [c for c, _ in plan.calls] == ["wave", "move_home"]


def test_a_plain_reflex_is_unchanged_by_the_curriculum_tier():
    """Single routine commands must still take the microsecond path, with no
    decomposition overhead and the same reported source."""
    planner = FastPlanner()
    plan = planner.plan("go home")
    assert plan is not None
    assert plan.source == "reflex"
    assert plan.subgoals == []
    assert not plan.warm_started


def test_spanish_sequences_take_the_fast_path_too():
    """The grammar is bilingual everywhere else, and the curriculum tier made
    a gap visible: 'saluda y luego vuelve a casa' split into two clauses
    correctly but 'vuelve a casa' had no rule, so a pair of reflexes went to
    the LLM."""
    planner = FastPlanner()
    for phrase in ("vuelve a casa", "ve a casa", "regresa al inicio", "aparca"):
        assert planner.plan(phrase) is not None, f"{phrase!r} has no reflex rule"
    plan = planner.plan("saluda y luego vuelve a casa")
    assert plan is not None
    assert plan.source == "curriculum"
    assert [c for c, _ in plan.calls] == ["wave", "move_home"]


def test_curriculum_can_be_disabled():
    planner = FastPlanner(curriculum=False)
    assert planner.plan("wave and then go home") is None
