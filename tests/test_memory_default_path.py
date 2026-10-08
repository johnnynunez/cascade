"""Golden pins for ROADMAP #7: the SHIPPED default path stays exactly as it was.

The memory embedder, embedding retrieval for skill-library notes and the
action<->object consolidation are opt-in (`memory.embedder.backend`,
`memory.action_objects`, both off in configs/demo.yaml). These tests import
only APIs that existed before that work and pin what the default produces:
guard-word library retrieval and its exact prompt text, a runtime without an
embedding index whose `recall_memory` answer has the same keys, an orchestrator
that calls library retrieval with the same arguments, and a fast planner with
no consolidation attached. They hold on main by construction; their job is to
make any later change to the default path fail loudly.
"""
from __future__ import annotations

import time

from conftest import needs_pin


def _library(tmp_path):
    from cascade.skills.library import SkillLibrary

    lib = SkillLibrary(tmp_path / "skills_library")
    for i, task in enumerate(("pick up the red cube", "put the red cube in the bowl")):
        lib.add("grasp_object empty_grasp", "grasp_object -> empty_grasp: jaws closed on air",
                "grasp cube empty object red", "lower the grasp by 1 cm",
                source_task=task, source_run=f"run{i}")
        lib.add("place_at release_drift", "place_at -> release_drift: object rolled",
                "bowl place release roll", "release lower", source_task=task + " now",
                source_run=f"run{i}b")
    return lib


def test_golden_guard_word_retrieval_without_an_embedder(tmp_path):
    from cascade.agent.aspire import retrieve

    lib = _library(tmp_path)
    cases = {
        "pick up the red cube": ["grasp-object-empty-grasp"],
        "put it in the bowl and release": ["place-at-release-drift"],
        "grasp the red object near the bowl": ["grasp-object-empty-grasp", "place-at-release-drift"],
        "grasping cubes": [],
    }
    for task, names in cases.items():
        assert [e.name for e in lib.relevant_entries(task, promoted_only=True)] == names, task
    text = retrieve(lib, "pick up the red cube")
    assert text.startswith("Learned skills from earlier runs (ASPIRE library) -- apply if they match:\n\n"
                           "[promoted: recurred in 2 distinct tasks, 2 confirmed runs]\n"
                           "# grasp_object empty_grasp")
    assert "similarity" not in text
    assert retrieve(lib, "grasping cubes") == ""


@needs_pin
def test_golden_shipped_runtime_has_no_embedding_index_and_recall_memory_keys(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        assert getattr(runtime.memory, "embedder", None) is None
        assert runtime.memory._index is None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        assert runtime.execute("localize_object", {"label": "red cube"})["ok"]
        assert runtime.execute("move_home", {})["ok"]
        out = runtime.execute("recall_memory", {"query": "red cube"})
        assert set(out) - {"ok", "outcome"} == {"recent_events", "object_memory"}
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_golden_orchestrator_calls_library_retrieval_with_the_same_arguments(demo_cfg, tmp_path,
                                                                            monkeypatch):
    from cascade.agent import orchestrator as orch_mod
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.agent.reflex import ExperienceMemory, FastPlanner
    from cascade.apps.demo import build_runtime, shutdown_runtime

    seen = []
    monkeypatch.setattr(orch_mod, "retrieve_skills",
                        lambda *a, **kw: seen.append((len(a), kw)) or "")
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        llm = MockLLM([LLMResponse(tool_calls=[ToolCall("task_done", {"success": False,
                                                                      "summary": "n/a"})])])
        orch_mod.AgentOrchestrator(
            llm, runtime, skill_library=object(), decompose=False, verify_milestones=False,
            attach_images=False, plausibility=None,
            fast_planner=FastPlanner(ExperienceMemory(tmp_path / "experience.json")),
        ).run_task("inspect the table carefully")
        intro = llm.requests[0]["messages"][0]["content"]
    finally:
        shutdown_runtime(runtime, arm)
    assert seen == [(2, {})]
    assert "Action-object" not in intro


def test_golden_default_planner_and_shipped_config_add_nothing(demo_cfg):
    from cascade.agent.reflex import FastPlanner

    assert getattr(FastPlanner(None), "action_objects", None) is None
    assert not demo_cfg.memory.get("action_objects", False)
    embedder = demo_cfg.memory.get("embedder")
    assert str(embedder.get("backend", "none") if embedder is not None else "none") == "none"
