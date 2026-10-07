"""Task-Specific Memory recipes (ROADMAP follow-up #9, Harness-VLA v4).

Tier-2 `ExperienceMemory` keyed on instruction text and stored the raw skill
calls of a proven plan. For a plan that came out of the LLM tier that means
`place_at(x=0.20, y=-0.15)`: the coordinate where the bowl WAS. Replaying it
on a table where the bowl has moved places the cube on bare table with full
confidence -- a remembered coordinate aiming the arm, which the persistent
belief store already forbids for restored beliefs.

Harness-VLA v4's Task-Specific Memory stores the run with every concrete
coordinate REPLACED by a symbolic perception query plus a semantic summary,
and re-grounds the queries through live perception at replay. These tests
pin that shape end to end on the mock stack:

* a successful LLM-tier run is stored as a recipe (JSONL under runs/) whose
  steps hold `localize_object(label) + offset` queries, never coordinates;
* a tier-2 hit re-grounds every query through the runtime's perception
  BEFORE any motion and succeeds via the object's NEW position;
* a query that fails to ground aborts to the LLM tier with zero motion --
  there is no stored coordinate to fall back to, by construction;
* plain experience entries (reflex plans, symbolic args) keep working.
"""

from __future__ import annotations

import json
import time

import pytest

from conftest import needs_pin

from cascade.agent.effects import CONFIRMED, POSTCONDITIONS, Postcondition
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.agent.reflex import ExperienceMemory, FastPlanner

TASK = "move the red cube to the front-left of the table"


def _target(label, offset, args=("x", "y")):
    return {"query": "localize_object", "label": label, "offset_m": list(offset), "args": list(args)}


def _recipe_calls(anchor="red cube", offset=(-0.09, -0.15)):
    return [
        ("grasp_object", {"label": "red cube"}),
        ("place_at", {"$target": _target(anchor, offset)}),
    ]


# ── pure transformation: memory/recipes.py ───────────────────────────────


def _scene():
    return [
        {"label": "red cube", "position": [0.29, 0.00, 0.0375]},
        {"label": "blue bowl", "position": [0.22, -0.14, 0.03]},
    ]


def _log(*entries):
    return [{"step": i, "tier": "llm", "tool": name, "args": args, "result": result}
            for i, (name, args, result) in enumerate(entries, start=1)]


def test_symbolize_replaces_coordinates_with_a_query_on_the_nearest_other_object():
    from cascade.memory import recipes

    log = _log(
        ("get_observation", {}, {"ok": True, "objects_visible": _scene()}),
        ("grasp_object", {"label": "red cube"}, {"ok": True, "held": "red cube"}),
        ("place_at", {"x": 0.20, "y": -0.15}, {"ok": True}),
        ("task_done", {"success": True, "summary": "done"}, {"ok": True, "task_complete": True}),
    )
    steps = recipes.symbolize_run(log, recipes.anchor_scene(log, _scene()))
    assert [s for s, _ in steps] == ["grasp_object", "place_at"]
    place = steps[1][1]
    assert "x" not in place and "y" not in place, "a concrete coordinate survived symbolization"
    target = place["$target"]
    assert target["query"] == "localize_object"
    # the held cube is NOT its own anchor when another object is in range
    assert target["label"] == "blue bowl"
    assert target["offset_m"] == pytest.approx([0.20 - 0.22, -0.15 + 0.14], abs=1e-9)
    assert target["args"] == ["x", "y"]
    assert recipes.is_symbolic(steps)


def test_symbolize_falls_back_to_the_manipulated_object_when_it_is_the_only_anchor():
    from cascade.memory import recipes

    scene = [_scene()[0]]
    log = _log(
        ("grasp_object", {"label": "red cube"}, {"ok": True, "held": "red cube"}),
        ("place_at", {"x": 0.20, "y": -0.15, "z": 0.05}, {"ok": True}),
    )
    steps = recipes.symbolize_run(log, scene)
    target = steps[1][1]["$target"]
    assert target["label"] == "red cube"
    assert target["offset_m"] == pytest.approx([-0.09, -0.15, 0.05 - 0.0375])
    assert target["args"] == ["x", "y", "z"]


def test_symbolize_refuses_an_unanchored_coordinate_rather_than_storing_it():
    from cascade.memory import recipes

    log = _log(
        ("grasp_object", {"label": "red cube"}, {"ok": True, "held": "red cube"}),
        ("place_at", {"x": 0.90, "y": 0.80}, {"ok": True}),   # nothing within reach of it
    )
    with pytest.raises(recipes.RecipeError):
        recipes.symbolize_run(log, _scene())
    with pytest.raises(recipes.RecipeError):
        recipes.symbolize_run(log, [])   # a run with no perceived scene at all


def test_symbolize_keeps_names_symbolic_and_drops_failed_and_observation_steps():
    from cascade.memory import recipes

    log = _log(
        ("get_observation", {}, {"ok": True, "objects_visible": _scene()}),
        ("grasp_object", {"label": "red cube"}, {"ok": False, "error": "air grasp"}),
        ("pick_and_place", {"object": "red cube", "destination": "drop zone"}, {"ok": True}),
        ("move_home", {}, {"ok": True}),
    )
    steps = recipes.symbolize_run(log, _scene())
    assert steps == [("pick_and_place", {"object": "red cube", "destination": "drop zone"}),
                     ("move_home", {})]
    assert not recipes.is_symbolic(steps)


def test_symbolize_rewrites_a_pixel_grasp_to_the_object_it_held():
    from cascade.memory import recipes

    log = _log(("grasp_at_pixel", {"u": 320, "v": 230}, {"ok": True, "held": "red cube"}))
    assert recipes.symbolize_run(log, _scene()) == [("grasp_object", {"label": "red cube"})]
    with pytest.raises(recipes.RecipeError):
        recipes.symbolize_run(_log(("grasp_at_pixel", {"u": 320, "v": 230}, {"ok": True})), _scene())


def test_ground_applies_the_offset_to_the_current_position_and_never_a_stored_one():
    from cascade.memory import recipes

    seen = []

    def localize(label):
        seen.append(label)
        return {"ok": True, "label": label, "position": [0.31, 0.08, 0.04]}

    grounded, n = recipes.ground(_recipe_calls(), localize)
    assert n == 1 and seen == ["red cube"]
    assert grounded[0] == ("grasp_object", {"label": "red cube"})
    name, args = grounded[1]
    assert name == "place_at" and "$target" not in args
    assert args["x"] == pytest.approx(0.31 - 0.09) and args["y"] == pytest.approx(0.08 - 0.15)
    assert "z" not in args
    # the stored recipe is untouched: still symbolic for the next replay
    assert "$target" in _recipe_calls()[1][1]


def test_ground_fails_closed_when_perception_cannot_find_the_anchor():
    from cascade.memory import recipes

    def localize(label):
        return {"ok": False, "error": f"localize {label!r} failed: no detections"}

    with pytest.raises(recipes.GroundingError) as info:
        recipes.ground(_recipe_calls(anchor="blue bowl"), localize)
    assert "blue bowl" in str(info.value)
    # a result with no 3D position (e.g. a configured zone) is not a fix either
    with pytest.raises(recipes.GroundingError):
        recipes.ground(_recipe_calls(), lambda label: {"ok": True, "position_xy_m": [0.18, -0.17]})


def test_ground_localizes_each_anchor_once_per_replay():
    from cascade.memory import recipes

    calls = _recipe_calls() + [("place_at", {"$target": _target("red cube", (0.0, 0.1))})]
    seen = []

    def localize(label):
        seen.append(label)
        return {"ok": True, "position": [0.3, 0.0, 0.04]}

    _, n = recipes.ground(calls, localize)
    assert n == 1 and seen == ["red cube"]


# ── tier-2 memory: recipes live beside the plain habits ──────────────────


def test_experience_memory_stores_recipes_as_jsonl_under_the_runs_dir(tmp_path):
    exp = ExperienceMemory(tmp_path / "experience.json")
    exp.record(TASK, _recipe_calls(), True, 12.0,
               summary="grasped the red cube and placed it front-left", source_run="run_1")
    exp.record("open gripper", [("open_gripper", {})], True, 0.5)   # a plain habit

    jsonl = tmp_path / "recipes.jsonl"
    assert jsonl.exists(), "recipes must be persisted as JSONL next to experience.json"
    lines = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["kind"] == "recipe" and rec["task"] == TASK
    assert rec["summary"].startswith("grasped the red cube")
    assert rec["source_run"] == "run_1"
    assert "$target" in rec["calls"][1][1]
    assert not any(k in rec["calls"][1][1] for k in ("x", "y", "z"))
    # the plain habit stays in experience.json and the recipe does NOT leak into it
    plans = json.loads((tmp_path / "experience.json").read_text())
    assert [p["task"] for p in plans] == ["open gripper"]

    reloaded = ExperienceMemory(tmp_path / "experience.json")
    assert len(reloaded) == 2
    hit = reloaded.recall(TASK)
    assert hit is not None and hit["kind"] == "recipe" and hit["summary"]
    assert reloaded.recall("open gripper")["calls"] == [["open_gripper", {}]]


def test_a_replayed_recipe_outcome_never_overwrites_the_queries_with_coordinates(tmp_path):
    exp = ExperienceMemory(tmp_path / "experience.json")
    exp.record(TASK, _recipe_calls(), True, 12.0, summary="s")
    grounded = [("grasp_object", {"label": "red cube"}), ("place_at", {"x": 0.2, "y": -0.09})]
    exp.record(TASK, grounded, True, 9.0)
    rec = exp.recall(TASK)
    assert rec["wins"] == 2
    assert "$target" in rec["calls"][1][1] and "x" not in rec["calls"][1][1]
    # and a loss is a loss, same as for a plain habit
    exp.record(TASK, grounded, False, 9.0)
    assert exp.recall(TASK)["losses"] == 1


def test_fast_planner_surfaces_a_recipe_hit_with_its_symbolic_calls(tmp_path):
    exp = ExperienceMemory(tmp_path / "experience.json")
    exp.record(TASK, _recipe_calls(), True, 12.0, summary="front-left placement")
    plan = FastPlanner(exp).plan(TASK)
    assert plan is not None and plan.source == "experience"
    assert "$target" in plan.calls[1][1]
    assert plan.summary == "front-left placement"


def test_existing_experience_entries_keep_working(tmp_path):
    """A pre-recipe experience.json (no `kind`, no recipes.jsonl) loads and
    replays exactly as before."""
    path = tmp_path / "experience.json"
    path.write_text(json.dumps([{"task": "do the calibration dance",
                                 "calls": [["move_home", {}], ["wave", {}]],
                                 "wins": 3, "losses": 0, "avg_s": 2.0}]))
    exp = ExperienceMemory(path)
    plan = FastPlanner(exp).plan("do the calibration dance")
    assert plan is not None and plan.source == "experience"
    assert [c for c, _ in plan.calls] == ["move_home", "wave"]
    assert plan.summary is None
    exp.record("do the calibration dance", plan.calls, True, 2.0)
    assert not (tmp_path / "recipes.jsonl").exists()


# ── end to end on the mock stack ─────────────────────────────────────────


def _script(*calls: ToolCall) -> list[LLMResponse]:
    return [LLMResponse(text="", tool_calls=[c]) for c in calls]


class ExplodingLLM:
    supports_vision = False

    def chat(self, *a, **kw):
        raise AssertionError("LLM must not be called on a tier-2 recipe replay")


def _confirm_effects(runtime) -> None:
    """Stand-in for the physics channel. The mock camera re-renders the same
    frame every grab, so a real pick/place can only ever be `unverified` here
    (see test_orchestrator_e2e.py); these tests are about the MEMORY, and a
    recipe is only ever written for a verified success, so the verdict is
    pinned to CONFIRMED. Nothing about motion, safety or the checker changes."""
    def verify(name, args, result, before=None):
        kind = POSTCONDITIONS.get(name)
        if kind is None:          # same contract as the real checker: no verdict
            return None
        return Postcondition(name, kind, CONFIRMED, "stand-in physics channel (test)",
                             channel="physics")

    runtime.effects.verify = verify


def _wait_for_cube(runtime) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
        time.sleep(0.05)
    assert runtime.beliefs.find("red cube") is not None, "watcher never saw the mock cube"


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        yield runtime, arm
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_a_verified_llm_run_is_stored_as_a_recipe_of_queries_not_coordinates(runtime_and_arm, tmp_path):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    cube = runtime.beliefs.find("red cube").position.copy()

    llm = MockLLM(_script(
        ToolCall("get_observation", {}),
        ToolCall("grasp_object", {"label": "red cube"}),
        ToolCall("place_at", {"x": 0.20, "y": -0.15}),
        ToolCall("task_done", {"success": True, "summary": "moved the cube to the front-left"}),
    ))
    exp = ExperienceMemory(tmp_path / "experience.json")
    agent = AgentOrchestrator(llm, runtime, decompose=False, max_steps=10,
                              fast_planner=FastPlanner(exp))
    with runtime.watcher.paused():
        report = agent.run_task(TASK)
    assert report.success and report.path == "llm", report.summary

    assert len(exp) == 1, "the verified LLM-tier run was not remembered"
    rec = exp.recall(TASK)
    assert rec["kind"] == "recipe"
    assert rec["summary"] == "moved the cube to the front-left"
    assert [c for c, _ in rec["calls"]] == ["grasp_object", "place_at"]
    place = rec["calls"][1][1]
    assert not any(k in place for k in ("x", "y", "z")), place
    target = place["$target"]
    assert target["query"] == "localize_object" and target["label"] == "red cube"
    assert target["offset_m"] == pytest.approx([0.20 - cube[0], -0.15 - cube[1]], abs=0.01)
    # serialized as JSONL under the run's memory dir, one recipe per line
    lines = (tmp_path / "recipes.jsonl").read_text().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["source_run"] == runtime.trace.run_dir.name
    # no concrete coordinate anywhere in the stored recipe
    for _, args in stored["calls"]:
        assert not any(isinstance(args.get(k), (int, float)) for k in ("x", "y", "z")), args


@needs_pin
def test_an_unverified_llm_success_is_not_stored_as_a_recipe(runtime_and_arm, tmp_path):
    """Without the stand-in channel the static mock leaves the pick
    unverified -> report.success is False -> nothing is remembered."""
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _wait_for_cube(runtime)
    llm = MockLLM(_script(
        ToolCall("grasp_object", {"label": "red cube"}),
        ToolCall("place_at", {"x": 0.20, "y": -0.15}),
        ToolCall("task_done", {"success": True, "summary": "claimed"}),
    ))
    exp = ExperienceMemory(tmp_path / "experience.json")
    agent = AgentOrchestrator(llm, runtime, decompose=False, max_steps=10,
                              fast_planner=FastPlanner(exp))
    with runtime.watcher.paused():
        report = agent.run_task(TASK)
    assert report.success is False and report.unverified
    assert len(exp) == 0 and not (tmp_path / "recipes.jsonl").exists()


@needs_pin
def test_replay_regrounds_the_recipe_against_the_moved_object(demo_cfg, tmp_path, monkeypatch):
    """The cube is rendered ~6 cm to the left of where the recipe was learned.
    The replay must localize it NOW and place relative to the new position --
    and the LLM is never consulted."""
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.perception import mock_camera

    original = mock_camera.synthetic_tabletop
    # 65 px at fx=600 over 0.55 m of range ~= 0.06 m along cam -x == base +y
    monkeypatch.setattr(mock_camera, "synthetic_tabletop",
                        lambda **kw: original(**{**kw, "box_px": (215, 200, 295, 260)}))
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        arm.object_stop_frac = 0.5
        _confirm_effects(runtime)
        _wait_for_cube(runtime)
        moved = runtime.beliefs.find("red cube").position.copy()
        assert moved[1] > 0.04, f"the mock cube did not move: {moved}"

        exp = ExperienceMemory(tmp_path / "experience.json")
        exp.record(TASK, _recipe_calls(), True, 12.0, summary="front-left placement")
        agent = AgentOrchestrator(ExplodingLLM(), runtime, decompose=False,
                                  fast_planner=FastPlanner(exp))
        with runtime.watcher.paused():
            report = agent.run_task(TASK)
        assert report.success, report.summary
        assert report.path == "experience"

        tools = [e["tool"] for e in report.tool_log]
        assert "localize_object" in tools, tools
        assert tools.index("localize_object") < tools.index("grasp_object"), \
            "re-grounding must happen before the first motion"
        place = next(e for e in report.tool_log if e["tool"] == "place_at")
        assert place["result"]["ok"], place["result"]
        assert place["args"]["x"] == pytest.approx(moved[0] - 0.09, abs=0.02)
        assert place["args"]["y"] == pytest.approx(moved[1] - 0.15, abs=0.02)
        assert "$target" not in place["args"]
        # the memory still holds the QUERY, not the coordinate it resolved to
        rec = exp.recall(TASK)
        assert rec["wins"] == 2 and "$target" in rec["calls"][1][1]
        assert "x" not in rec["calls"][1][1]
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_failed_grounding_aborts_to_the_llm_tier_without_any_motion(runtime_and_arm, tmp_path):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    _confirm_effects(runtime)
    _wait_for_cube(runtime)
    from cascade.skills.runtime import _MOTION_SKILLS

    executed: list[str] = []
    real_execute = runtime.execute

    def spy(name, args):
        executed.append(name)
        return real_execute(name, args)

    runtime.execute = spy

    exp = ExperienceMemory(tmp_path / "experience.json")
    # the recipe was learned on a table that had a bowl; this one does not
    exp.record(TASK, _recipe_calls(anchor="blue bowl", offset=(0.0, 0.0)), True, 12.0,
               summary="place in the bowl")
    llm = MockLLM(_script(ToolCall("task_done", {"success": False, "summary": "no bowl on the table"})))
    agent = AgentOrchestrator(llm, runtime, decompose=False, max_steps=5,
                              fast_planner=FastPlanner(exp))
    with runtime.watcher.paused():
        report = agent.run_task(TASK)

    assert report.path == "llm"
    assert not report.success
    assert "grasp_object" not in executed and "place_at" not in executed
    assert not (set(executed) & _MOTION_SKILLS), executed
    assert "localize_object" in executed
    # the LLM tier was told why, and that nothing moved
    intro = str(llm.requests[0]["messages"][0]["content"])
    assert "re-grounded" in intro and "blue bowl" in intro
    assert "No motion" in intro
    # the arm never left home
    assert runtime.held_object is None
    # the recipe keeps its query; nothing was substituted and no loss recorded
    # for a scene that simply does not match
    rec = exp.recall(TASK)
    assert rec is not None and "$target" in rec["calls"][1][1] and rec["losses"] == 0


@needs_pin
def test_a_plain_reflex_plan_is_executed_exactly_as_before(runtime_and_arm, tmp_path):
    """No queries to ground -> no perception call is inserted, the first tool
    in the log is the reflex's own skill, same as before this feature."""
    runtime, arm = runtime_and_arm
    _wait_for_cube(runtime)
    agent = AgentOrchestrator(ExplodingLLM(), runtime, decompose=False,
                              fast_planner=FastPlanner(ExperienceMemory(tmp_path / "exp.json")))
    report = agent.run_task("open gripper")
    assert report.success and report.path == "reflex"
    assert [e["tool"] for e in report.tool_log] == ["open_gripper"]
    assert report.tool_log[0]["step"] == 1
