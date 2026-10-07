"""ROADMAP follow-up #13 (RPent / Harness-VLA): `recall_step(n)` + a `stuck`
outcome.

1. `recall_step(n)` is a READ-ONLY tool over the evidence the runtime already
   records per executed step (trace.jsonl row + BEFORE/AFTER keyframes): the
   step's skill, args, outcome, postcondition verdict, dispatch tier and its
   keyframe image(s). Negative `n` counts from the end. An invalid `n` is an
   explicit error and never a stale frame. The MCP server serves the frames
   as image content items with one caption each, like `task_memory`.

2. `stuck` is a THIRD motion outcome next to ok/failed: a skill that
   exhausted its persistence budget, or cannot proceed for a reason a human
   can act on (object never seen after re-scans, wider than the jaws,
   destination not placeable), returns `ok: false, outcome: "stuck"` with an
   `ask` -- a concrete human-actionable request. A stuck step never claims
   an effect, the orchestrator relays the ask verbatim and stops retrying,
   and the trace + summary.txt record `stuck`. The e-stop and every harness
   refusal stay plain failures.

Every test here runs the REAL runtime on the mock stack (mock camera,
mock arm, scripted LLM) and every assertion failed against b5477d8 first.
"""
from __future__ import annotations

import base64
import json
import pathlib
import time

import pytest
from conftest import needs_pin

from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config
from cascade.types import SkillError

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def rt(tmp_path):
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    arm.object_stop_frac = 0.5  # the mock jaws jam halfway: an object is in the gripper
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


def _script(*calls: ToolCall) -> list[LLMResponse]:
    return [LLMResponse(text="", tool_calls=[c]) for c in calls]


def _rows(runtime) -> list[dict]:
    return [json.loads(line) for line in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]


def _always_miss(rt, monkeypatch, error="SkillError: did not settle at grasp pose", attempts=2):
    """Pin a pick that can never succeed: every grasp attempt misses, the
    re-home / re-scan between attempts are no-ops, and the persistence budget
    is two attempts (the attempt cap ends it, never the wall clock)."""
    calls = {"n": 0}

    def miss(*a, **k):
        calls["n"] += 1
        raise SkillError(error.split(": ", 1)[1] if error.startswith("SkillError: ") else error)

    monkeypatch.setattr(rt, "skill_grasp_object", miss)
    monkeypatch.setattr(rt, "skill_move_home", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(rt, "_reobserve", lambda *a, **k: None)
    rt.cfg.grasp._data["persist_seconds"] = 600.0
    rt.cfg.grasp._data["max_pick_attempts"] = attempts
    return calls


# ── 1. recall_step ───────────────────────────────────────────────────────────


@needs_pin
def test_recall_step_returns_the_recorded_frame_and_verdict_for_a_scripted_run(rt):
    """A scripted run (observe, grasp, recall) through the LLM tier: the
    recall of step 1 reports the grasp's skill/args/verdict/tier exactly as
    the trace recorded them, and its keyframe bytes are the recorded files."""
    llm = MockLLM(_script(
        ToolCall("get_observation", {}),
        ToolCall("grasp_object", {"label": "red cube", "material": "rigid"}),
        ToolCall("recall_step", {"n": 1}),
        ToolCall("recall_step", {"n": -1}),
        ToolCall("task_done", {"success": False, "summary": "looked back"}),
    ))
    agent = AgentOrchestrator(llm, rt, advisor=None, decompose=False, max_steps=8)
    report = agent.run_task("grasp the red cube and then tell me what you did")
    results = [e["result"] for e in report.tool_log if e["tool"] == "recall_step"]
    grasp = next(e["result"] for e in report.tool_log if e["tool"] == "grasp_object")
    assert grasp["ok"], grasp
    assert len(results) == 2 and all(r["ok"] for r in results), results
    first, second = results
    assert first["step"] == 1 and first["skill"] == "grasp_object"
    assert first["args"] == {"label": "red cube", "material": "rigid"}
    assert first["tier"] == "llm"
    assert first["outcome"] == "ok"
    assert first["verdict"] == (grasp.get("postcondition") or {}).get("status")
    assert first["steps_recorded"] == 2
    # n=-1 right after a recall is still the grasp: recall rows are not steps
    assert second["step"] == 1 and second["skill"] == "grasp_object", second
    # the recalled images ARE the recorded keyframes, byte for byte
    rows = _rows(rt)
    grasp_row = next(r for r in rows if r["skill"] == "grasp_object")
    tags = {f["tag"]: f["path"] for f in first["keyframes"]}
    assert tags == {"before": grasp_row["keyframe_before"], "after": grasp_row["keyframe_after"]}
    recalled = {f["tag"]: f["jpeg"] for f in rt.last_recalled_frames}
    for tag, rel in tags.items():
        assert recalled[tag] == (rt.trace.run_dir / rel).read_bytes()
    # nothing in the JSON result is binary (the planner gets json.dumps of it)
    json.dumps(first)
    # the planner's next turn SAW the recalled frames: one message carries
    # the memory harness + the recalled images, captioned as a recall
    req = llm.requests[3]["messages"]
    with_images = [m for m in req if m.get("images")]
    assert len(with_images) == 1, "images must enter the request in exactly one message"
    assert "recalled step 1" in with_images[0]["content"]
    assert any(img == recalled["after"] for img in with_images[0]["images"])


@needs_pin
def test_recall_step_rejects_a_bad_index_with_an_explicit_error(rt):
    """Out-of-range, non-integer and empty-trace recalls are explicit
    errors; a bad call clears the recalled frames so the MCP layer can never
    serve a stale image for a step that does not exist."""
    empty = rt.execute("recall_step", {"n": -1})
    assert empty["ok"] is False and "no steps recorded" in empty["error"], empty
    rt.execute("get_observation", {})
    good = rt.execute("recall_step", {"n": 0})
    assert good["ok"] and good["skill"] == "get_observation", good
    assert rt.last_recalled_frames, "a valid recall must load the recorded frame(s)"
    for bad_n in (5, -5):
        res = rt.execute("recall_step", {"n": bad_n})
        assert res["ok"] is False, res
        assert "does not exist" in res["error"] and "1 step(s) recorded" in res["error"], res
        assert rt.last_recalled_frames == [], "a bad index must not leave a stale frame behind"
    res = rt.execute("recall_step", {"n": "latest"})
    assert res["ok"] is False and "integer" in res["error"], res
    # the recall rows never shift the index of the real steps
    again = rt.execute("recall_step", {"n": -1})
    assert again["ok"] and again["skill"] == "get_observation" and again["steps_recorded"] == 1, again


@needs_pin
def test_mcp_recall_step_serves_keyframes_as_image_content_with_captions(rt):
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    server._runtime = rt  # bypass the lazy build: the real runtime is already up
    assert "recall_step" in {t["name"] for t in server.list_tools()}
    grasp = server.call_tool("grasp_object", {"label": "red cube", "material": "rigid"})
    assert not grasp["isError"], grasp
    out = server.call_tool("recall_step", {"n": -1})
    assert not out["isError"], out
    kinds = [c["type"] for c in out["content"]]
    # caption, image, caption, image, JSON summary last -- the task_memory shape
    assert kinds == ["text", "image", "text", "image", "text"], kinds
    row = next(r for r in _rows(rt) if r["skill"] == "grasp_object")
    before = base64.b64decode(out["content"][1]["data"])
    after = base64.b64decode(out["content"][3]["data"])
    assert before == (rt.trace.run_dir / row["keyframe_before"]).read_bytes()
    assert after == (rt.trace.run_dir / row["keyframe_after"]).read_bytes()
    assert out["content"][1]["mimeType"] == "image/jpeg"
    cap_before, cap_after = out["content"][0]["text"], out["content"][2]["text"]
    assert "BEFORE" in cap_before and "grasp_object" in cap_before
    assert "AFTER" in cap_after and "grasp_object" in cap_after
    summary = json.loads(out["content"][-1]["text"])
    assert summary["ok"] and summary["skill"] == "grasp_object" and summary["tier"] == "mcp-host"
    assert summary["verdict"] == (row["result"].get("postcondition") or {}).get("status")
    assert summary["verdict"].upper() in cap_after
    # a bad index is an error result with NO image content
    bad = server.call_tool("recall_step", {"n": 99})
    assert bad["isError"] and all(c["type"] == "text" for c in bad["content"]), bad
    assert "does not exist" in json.loads(bad["content"][-1]["text"])["error"]


# ── 2. the stuck outcome ─────────────────────────────────────────────────────


@needs_pin
def test_persistence_budget_exhaustion_yields_stuck_with_an_ask_and_no_claimed_effect(rt, monkeypatch):
    calls = _always_miss(rt, monkeypatch, attempts=2)
    res = rt.execute("pick_and_place", {"object": "red cube"})
    assert calls["n"] == 2, calls  # the whole budget was spent before asking
    assert res["ok"] is False, "a stuck step is never a success"
    assert res["outcome"] == "stuck", res
    ask = res["ask"]
    assert isinstance(ask, str) and "red cube" in ask and ask.rstrip().endswith("again."), ask
    # the ask is a request to the human, not a diagnosis for the planner
    assert res.get("suggestion") != ask
    assert "stuck" in res["next_action"].lower() and "verbatim" in res["next_action"].lower()
    # no claimed effect: never verified, and the verdict stays as measured
    assert res.get("verified") is not True
    pc = res.get("postcondition") or {}
    assert pc.get("status") in (None, "unverified", "refuted"), pc
    assert rt.held_object is None
    # the trace row records the stuck outcome and the ask
    row = next(r for r in _rows(rt) if r["skill"] == "pick_and_place")
    assert row["result"]["outcome"] == "stuck" and row["result"]["ask"] == ask
    # ... and a successful step is `outcome: ok`, a plain failure `failed`
    home = rt.execute("move_home", {})
    assert home["ok"] and home["outcome"] == "ok"
    unknown = rt.execute("grasp_object", {"label": "red cube", "arm": "no-such-arm"})
    assert unknown["ok"] is False and unknown["outcome"] == "failed" and "ask" not in unknown


@needs_pin
def test_human_curable_terminal_reasons_are_stuck_but_the_estop_is_a_plain_failure(rt, monkeypatch):
    # object wider than the jaws: a re-scan cannot shrink it -> stuck, ask to push/replace
    calls = _always_miss(
        rt, monkeypatch, attempts=8,
        error="no executable grasp: cube: required width 120mm > gripper max 55mm (consider push or regrasp)",
    )
    res = rt.execute("pick_and_place", {"object": "red cube"})
    assert calls["n"] == 1 and res["outcome"] == "stuck", res
    assert "wider" in res["ask"] and "push" in res["ask"], res["ask"]
    # never seen after re-scans -> stuck, ask to place it in view
    rt.beliefs.clear()
    monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
    calls = _always_miss(rt, monkeypatch, attempts=8, error="no detections for 'unicorn'")
    res = rt.execute("pick_and_place", {"object": "unicorn"})
    assert calls["n"] == 2 and res["outcome"] == "stuck", res
    assert "unicorn" in res["ask"] and "see" in res["ask"], res["ask"]
    # the e-stop is a harness refusal, not a request to the human: plain failure
    rt.arm.harness.estop("test")
    try:
        calls = _always_miss(rt, monkeypatch, attempts=8)
        res = rt.execute("pick_and_place", {"object": "red cube"})
    finally:
        rt.arm.harness.reset_estop()
    assert res["ok"] is False and res["outcome"] == "failed" and "ask" not in res, res
    assert "e-stop latched" in res["error"]


@needs_pin
def test_handover_and_grasp_with_persistence_propagate_stuck(rt, monkeypatch):
    _always_miss(rt, monkeypatch, attempts=2)
    res = rt.execute("handover", {"label": "red cube"})
    assert res["ok"] is False and res["outcome"] == "stuck", res
    assert "red cube" in res["ask"], res


@needs_pin
def test_orchestrator_stops_retrying_on_stuck_and_relays_the_ask_verbatim(rt, monkeypatch):
    """The LLM tier is scripted to retry the same pick after a stuck result:
    the orchestrator must end the task on the first stuck instead, with the
    ask verbatim in the report and in summary.txt."""
    calls = _always_miss(rt, monkeypatch, attempts=2)
    llm = MockLLM(_script(
        ToolCall("pick_and_place", {"object": "red cube"}),
        ToolCall("pick_and_place", {"object": "red cube"}),  # the retry that must not run
        ToolCall("task_done", {"success": True, "summary": "done"}),
    ))
    agent = AgentOrchestrator(llm, rt, advisor=None, decompose=False, max_steps=6)
    report = agent.run_task("move the red cube to the drop zone please")
    assert report.success is False
    assert report.path == "llm"
    assert calls["n"] == 2, calls  # ONE pick (its own two attempts), no second pick
    assert len(llm.requests) == 1, "the planner was not asked to plan a retry"
    assert [e["tool"] for e in report.tool_log] == ["pick_and_place"]
    stuck = report.tool_log[0]["result"]
    assert stuck["outcome"] == "stuck"
    assert report.stuck == {"skill": "pick_and_place", "args": {"object": "red cube"}, "ask": stuck["ask"]}
    assert stuck["ask"] in report.summary
    summary_txt = (rt.trace.run_dir / "summary.txt").read_text()
    assert "success: False" in summary_txt and "outcome: stuck" in summary_txt
    assert stuck["ask"] in summary_txt


@needs_pin
def test_reflex_path_stuck_ends_the_task_without_escalating_to_the_llm(rt, monkeypatch, tmp_path):
    from cascade.agent.reflex import ExperienceMemory, FastPlanner

    class ExplodingLLM:
        supports_vision = False

        def chat(self, *a, **kw):
            raise AssertionError("a stuck reflex step must not be retried by the LLM tier")

    calls = _always_miss(rt, monkeypatch, attempts=2)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and rt.beliefs.find("red cube") is None:
        time.sleep(0.05)
    agent = AgentOrchestrator(
        ExplodingLLM(), rt, decompose=False,
        fast_planner=FastPlanner(ExperienceMemory(tmp_path / "exp.json")),
    )
    report = agent.run_task("pick and place the red cube")
    assert report.success is False and report.path == "reflex", report
    assert calls["n"] == 2, calls
    assert report.stuck and report.stuck["skill"] == "pick_and_place"
    assert report.stuck["ask"] in report.summary
    assert "outcome: stuck" in (rt.trace.run_dir / "summary.txt").read_text()


def test_stuck_step_can_never_end_confirmed():
    """`annotate_result` is the single fold point for verdicts: a confirmed
    postcondition on a stuck result is downgraded to unverified (a stuck
    step claims no effect), a refuted one stays refuted as measured."""
    from cascade.agent.effects import CONFIRMED, REFUTED, UNVERIFIED, Postcondition, annotate_result

    pc = Postcondition(skill="pick_and_place", kind="relocated", status=CONFIRMED, evidence="moved 0.2 m")
    out = annotate_result({"ok": False, "outcome": "stuck", "ask": "move it"}, pc)
    assert out["verified"] is False and out["postcondition"]["status"] == UNVERIFIED
    assert "stuck" in out["postcondition"]["evidence"]
    pc2 = Postcondition(skill="pick_and_place", kind="relocated", status=REFUTED, evidence="did not move")
    out2 = annotate_result({"ok": False, "outcome": "stuck", "ask": "move it"}, pc2)
    assert out2["postcondition"]["status"] == REFUTED and out2["verified"] is False
    # an ordinary success is untouched
    pc3 = Postcondition(skill="pick_and_place", kind="relocated", status=CONFIRMED, evidence="moved")
    assert annotate_result({"ok": True}, pc3)["verified"] is True


def test_skill_stuck_is_a_skill_error_carrying_the_ask():
    from cascade.types import SkillStuck

    err = SkillStuck("grasp exhausted", ask="Move the cube closer and ask again.")
    assert isinstance(err, SkillError) and err.ask.endswith("again.")


@needs_pin
def test_a_skill_raising_skill_stuck_becomes_a_stuck_result_at_the_choke_point(rt, monkeypatch):
    """The exception form of the contract: `execute()` converts `SkillStuck`
    into `ok: false, outcome: stuck, ask` -- and a stuck result that forgot
    its ask still gets one derived from the error (fail-closed: the human
    always receives a request, never a bare status)."""
    from cascade.types import SkillStuck

    def stuck_wave(*a, **k):
        raise SkillStuck("the audience left", ask="Bring the audience back and ask again.")

    monkeypatch.setattr(rt, "skill_wave", stuck_wave)
    res = rt.execute("wave", {})
    assert res["ok"] is False and res["outcome"] == "stuck"
    assert res["ask"] == "Bring the audience back and ask again."
    assert res["error"].startswith("SkillStuck:")
    monkeypatch.setattr(rt, "skill_wave", lambda *a, **k: {"ok": True, "outcome": "stuck", "verified": True})
    res = rt.execute("wave", {})
    assert res["ok"] is False and res["outcome"] == "stuck", "a stuck result is never a success"
    assert res["verified"] is False, "a stuck step never claims an effect"
    assert isinstance(res["ask"], str) and "wave" in res["ask"] and res["ask"].endswith("ask again.")


# ── 3. docs and counts ──────────────────────────────────────────────────────


def test_docs_list_recall_step_and_the_stuck_outcome():
    from cascade.apps.mcp_server import _EXCLUDED_TOOLS, _EXTRA_TOOLS
    from cascade.skills.runtime import TOOL_SPECS, _MOTION_SKILLS

    names = {t["name"] for t in TOOL_SPECS}
    assert "recall_step" in names and "recall_step" not in _MOTION_SKILLS
    spec = next(t for t in TOOL_SPECS if t["name"] == "recall_step")
    assert "n" in spec["parameters"]["properties"]
    readme = (REPO / "README.md").read_text()
    arch = (REPO / "docs/ARCHITECTURE.md").read_text()
    roadmap = (REPO / "docs/ROADMAP.md").read_text()
    assert "| `recall_step` |" in readme
    n_tools = len(names - _EXCLUDED_TOOLS) + len(_EXTRA_TOOLS)
    assert f"{n_tools} tools" in arch and f"{len(TOOL_SPECS)} specs" in arch, n_tools
    assert f"{len(names) - 1} skills + task_done" in arch
    assert "recall_step" in arch and "stuck" in arch
    assert "~~" in roadmap and "landed 2026-10-07" in roadmap and "`stuck` outcome" in roadmap
