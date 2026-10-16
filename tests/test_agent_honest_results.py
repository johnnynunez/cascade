"""What the chat agent is told after a tool call must be true and actionable.

Measured on the OpenClaw/Qwen kitchen run (8 orders): the agent answered
"Done" after a pick_and_place whose placement postcondition was REFUTED (the
only failure instruction covered "then stop"); it turned "next to the lemon"
into "on the lemon"; after one observed-finger refusal it retried the same
grasp three times from the refused pose until it happened to call
move_home; and it never said which frame "left" used. These tests run the
real mock runtime (``build_runtime``) and the real MCP server object, and
check the result payloads the agent reads.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from cascade.agent.effects import Postcondition
from cascade.config import load_demo_config

#: o3 (trace.jsonl:9 of the 16 Oct kitchen run): the motion routine reported
#: success, physics refuted the placement, the agent answered "Done".
O3_MOTION = {
    "picked": "orange", "placed_at": [0.32, 0.005, 0.079], "destination": "lemon",
    "grip_profile": "soft", "grip_verified": True, "grasp_attempts": 1, "place_attempts": 1,
    "return_home": {"attempted": True, "ok": True, "at": "home"},
    "next_action": ("Report this result, including any return_home failure, and await the "
                    "next user order. Do not add another movement to this request."),
}
O3_EVIDENCE = "orange ended 3.2 cm from lemon (at [0.284, 0.012], lemon is at [0.316, 0.011])"


@pytest.fixture
def runtime(tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(arms=["mock"], camera="mock", llm="mock")
    rt, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        yield rt
    finally:
        shutdown_runtime(rt, arm)


def _trace_rows(rt):
    return [json.loads(line) for line in (rt.trace.run_dir / "trace.jsonl").read_text().splitlines()]


def _refuted_o3(rt):
    rt.skill_pick_and_place = lambda **_: deepcopy(O3_MOTION)
    rt.effects.verify = lambda *a, **k: Postcondition(
        skill="pick_and_place", kind="relocated", status="refuted", evidence=O3_EVIDENCE, channel="physics")
    return rt.execute("pick_and_place", {"object": "orange", "destination": "lemon", "material": "soft"})


# ── 1. every failed result says so, unconditionally ──────────────────────


def test_a_refuted_pick_tells_the_agent_it_failed(runtime):
    result = _refuted_o3(runtime)
    assert result["ok"] is False and result["postcondition"]["status"] == "refuted"
    action = result["next_action"]
    assert action.startswith("This attempt FAILED: postcondition failed: " + O3_EVIDENCE)
    assert "Tell the user it failed" in action
    assert "never report" in action and "as success" in action
    # the existing 'then stop' guidance still applies after the failure line
    assert "then stop" in action and "Do not start another pick" in action
    # the motion routine's success wording is not offered for a failure
    assert "Report this result" not in action
    # the trace row carries exactly what the agent was told
    assert _trace_rows(runtime)[-1]["result"] == result


def test_a_confirmed_pick_keeps_its_own_next_action(runtime):
    runtime.skill_pick_and_place = lambda **_: deepcopy(O3_MOTION)
    runtime.effects.verify = lambda *a, **k: Postcondition(
        skill="pick_and_place", kind="relocated", status="confirmed", evidence="on target", channel="physics")
    result = runtime.execute("pick_and_place", {"object": "orange", "destination": "lemon"})
    assert result["ok"] is True and result["verified"] is True
    assert result["next_action"] == O3_MOTION["next_action"]


def test_a_successful_skill_gets_no_failure_instruction(runtime):
    runtime.skill_wave = lambda cycles=2: {"waved": True}
    result = runtime.execute("wave", {})
    assert result["ok"] is True
    assert "next_action" not in result


def test_any_failed_skill_carries_the_instruction_before_its_own(runtime):
    runtime.skill_wave = lambda cycles=2: {"ok": False, "error": "refused by the safety harness",
                                           "next_action": "Ask for a smaller move."}
    result = runtime.execute("wave", {})
    assert result["ok"] is False
    assert result["next_action"] == ("This attempt FAILED: refused by the safety harness. Tell the user it "
                                     "failed; never report a failed or refuted result as success. "
                                     "Ask for a smaller move.")


def test_a_raised_skill_error_carries_the_instruction(runtime):
    def boom(cycles=2):
        from cascade.skills.runtime import SkillError
        raise SkillError("could not reach home to wave")
    runtime.skill_wave = boom
    result = runtime.execute("wave", {})
    assert result["ok"] is False
    assert result["next_action"].startswith("This attempt FAILED: SkillError: could not reach home to wave.")


def test_a_failure_before_the_skill_runs_carries_the_instruction(runtime):
    result = runtime.execute("no_such_skill", {})
    assert result["ok"] is False
    assert result["next_action"].startswith("This attempt FAILED: unknown skill 'no_such_skill'.")


def test_a_long_error_is_shortened_in_the_instruction_only(runtime):
    long_error = "x" * 1000
    runtime.skill_wave = lambda cycles=2: {"ok": False, "error": long_error}
    result = runtime.execute("wave", {})
    assert result["error"] == long_error
    assert len(result["next_action"]) < 500
    assert "x" * 200 in result["next_action"]


def test_a_stuck_result_still_relays_the_ask_first(runtime):
    runtime.skill_wave = lambda cycles=2: {"ok": False, "outcome": "stuck", "ask": "Move the cube closer.",
                                           "error": "out of reach"}
    result = runtime.execute("wave", {})
    action = result["next_action"]
    assert action.startswith("STUCK: relay `ask`")
    assert "This attempt FAILED: out of reach." in action
