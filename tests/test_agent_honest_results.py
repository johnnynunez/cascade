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
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin

from cascade.agent.effects import Postcondition
from cascade.config import load_demo_config
from cascade.types import SafetyViolation, SkillError

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

#: r3 (trace.jsonl:27): the observed-finger gate refused the grasp next to the lemon.
R3_REFUSAL = ("measured finger intersects observed surface: {'finger': 'joint_right', 'component': 8, "
              "'surface': 'other observed surface', 'pixel_rc': [335, 663], "
              "'point_base_m': [0.3008, 0.0063, 0.0393]}")


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
        raise SkillError("could not reach home to wave")
    runtime.skill_wave = boom
    result = runtime.execute("wave", {})
    assert result["ok"] is False
    assert result["next_action"].startswith("This attempt FAILED: SkillError: could not reach home to wave.")


def test_a_failure_before_the_skill_runs_carries_the_instruction(runtime):
    result = runtime.execute("no_such_skill", {})
    assert result["ok"] is False
    assert result["next_action"].startswith("This attempt FAILED: unknown skill 'no_such_skill'.")


@pytest.mark.parametrize("text, error", [
    ("CAN feedback lost", "unexpected TimeoutError: CAN feedback lost"),
    ("", "unexpected TimeoutError"),
    ("  ", "unexpected TimeoutError"),
], ids=["text", "empty", "blank"])
def test_an_unexpected_exception_without_text_is_named_by_its_type_alone(runtime, text, error):
    """execute()'s last-resort handler: an exception with text keeps it;
    one without (a driver's bare TimeoutError) is named by its type, with no
    dangling \": \" (#367 review M7)."""
    def boom(cycles=2):
        raise TimeoutError(text)
    runtime.skill_wave = boom
    result = runtime.execute("wave", {})
    assert result["ok"] is False and result["error"] == error, result
    assert result["next_action"].startswith(f"This attempt FAILED: {error}. Tell the user it failed;"), result


def test_a_long_error_is_shortened_in_the_instruction_only(runtime):
    long_error = "x" * 1000
    runtime.skill_wave = lambda cycles=2: {"ok": False, "error": long_error}
    result = runtime.execute("wave", {})
    assert result["error"] == long_error
    assert len(result["next_action"]) < 500
    assert "x" * 200 in result["next_action"]


def test_an_error_that_ends_in_a_cut_marker_keeps_it_in_the_instruction(runtime):
    """A skill that shortened its own error marks the cut with "..." (the
    driver text of reset_scene's refusals): the instruction keeps the marker
    instead of stripping it to one full stop. Any other error still ends
    with exactly one."""
    runtime.skill_wave = lambda cycles=2: {"ok": False, "error": "gripper motor 7 did not ans..."}
    assert runtime.execute("wave", {})["next_action"].startswith(
        "This attempt FAILED: gripper motor 7 did not ans... Tell the user it failed;")
    runtime.skill_wave = lambda cycles=2: {"ok": False, "error": "out of reach.."}
    assert runtime.execute("wave", {})["next_action"].startswith(
        "This attempt FAILED: out of reach. Tell the user it failed;")


def test_a_stuck_result_still_relays_the_ask_first(runtime):
    runtime.skill_wave = lambda cycles=2: {"ok": False, "outcome": "stuck", "ask": "Move the cube closer.",
                                           "error": "out of reach"}
    result = runtime.execute("wave", {})
    action = result["next_action"]
    assert action.startswith("STUCK: relay `ask`")
    assert "This attempt FAILED: out of reach." in action


@pytest.mark.parametrize("ok", [None, 0, ""])
def test_any_falsy_ok_is_a_failure_that_says_so(runtime, ok):
    """Review minor 2: `_normalize_outcome` calls every falsy ok a failure;
    the instruction keyed on `ok is False` skipped ok=None / ok=0."""
    runtime.skill_wave = lambda cycles=2: {"ok": ok, "error": "odd ok"}
    result = runtime.execute("wave", {})
    assert result["outcome"] == "failed"
    assert result["next_action"].startswith("This attempt FAILED: odd ok.")


# ── 1b. refusals the MCP server answers itself (review minor 1) ──────────


def _text_payload(out):
    return json.loads(next(b["text"] for b in out["content"] if b["type"] == "text"))


def test_a_server_error_result_carries_the_failure_instruction():
    from cascade.apps.mcp_server import _text_result

    payload = {"ok": False, "execution_ok": False, "cancelled": True,
               "error": "request cancelled before execution"}
    out = _text_payload(_text_result(payload, is_error=True))
    assert out["next_action"].startswith("This attempt FAILED: request cancelled before execution. "
                                         "Tell the user it failed")
    assert "next_action" not in payload  # the caller's dict is not rewritten
    # an error result that already carries it is unchanged; a success never gets it
    assert _text_payload(_text_result(out, is_error=True))["next_action"] == out["next_action"]
    assert "next_action" not in _text_payload(_text_result({"ok": True, "url": "x"}))


def test_a_hidden_tool_refusal_says_it_failed(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer

    monkeypatch.setenv("CASCADE_HIDE_TOOLS", "pick_and_place")
    out = McpSkillServer().call_tool("pick_and_place", {"object": "orange"})
    assert out["isError"] is True
    assert _text_payload(out)["next_action"].startswith(
        "This attempt FAILED: tool 'pick_and_place' is disabled by the operator")


def test_a_withheld_tool_refusal_says_it_failed(monkeypatch):
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    monkeypatch.setattr(server, "withheld_tools", lambda: {"probe_point": "no depth camera"})
    monkeypatch.setattr(server, "capabilities", lambda: {})
    out = server._reject_withheld("probe_point")
    assert out["isError"] is True
    assert _text_payload(out)["next_action"].startswith(
        "This attempt FAILED: tool 'probe_point' is not available on this rig: no depth camera.")


def test_a_reset_refused_while_a_command_runs_says_it_failed():
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    assert server._exec_lock.acquire(blocking=False)
    try:
        out = server.reset_now()
    finally:
        server._exec_lock.release()
    assert out["isError"] is True
    assert _text_payload(out)["next_action"].startswith(
        "This attempt FAILED: Wait for the current command to finish before resuming.")


def test_a_camera_snapshot_error_says_it_failed():
    from cascade.apps.mcp_server import McpSkillServer

    def unavailable():
        raise RuntimeError("camera 'isaac_side' is not streaming")

    out = McpSkillServer()._camera_snapshot(SimpleNamespace(rig=None, observe_fresh=unavailable))
    assert out["isError"] is True
    assert _text_payload(out)["next_action"].startswith(
        "This attempt FAILED: camera 'isaac_side' is not streaming.")


def _worker_replies(server, *messages):
    import queue

    from cascade.apps.mcp_server import _worker_loop

    inbox, eof, replies = queue.Queue(), object(), []
    for message in messages:
        inbox.put((message, replies.append))
    inbox.put(eof)
    assert _worker_loop(server, SimpleNamespace(checkpoint=lambda: None), inbox, eof) == 0
    return {r["id"]: r["result"] for r in replies}


def test_calls_the_transport_refuses_itself_say_they_failed(monkeypatch):
    """A call queued behind a STOP and a call cancelled before it ran are
    answered by the transport, never by the runtime."""
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    monkeypatch.setattr(server, "_queued_arm_error",
                        lambda name: "queued motion invalidated by stop; re-issue a new command explicitly")
    server._cancelled_ids.add(8)

    def call(request_id):
        return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                "params": {"name": "move_home", "arguments": {}}}

    replies = _worker_replies(server, call(7), call(8))
    for request_id, error in ((7, "queued motion invalidated by stop"), (8, "request cancelled before execution")):
        assert replies[request_id]["isError"] is True
        assert _text_payload(replies[request_id])["next_action"].startswith(f"This attempt FAILED: {error}")


# ── 3. a refused grasp that left the arm in place says how to recover ────


def test_an_observed_finger_refusal_tells_the_agent_to_move_home_first(runtime, monkeypatch):
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    attempts = []
    # the refused attempt moved the arm (a counted motion): it stays there
    runtime.skill_grasp_object = _refuse_after_moving(runtime, then=lambda: attempts.append(1))
    result = runtime.execute("pick_and_place", {"object": "green cube", "destination": "green square"})
    assert result["ok"] is False and result["home_skipped"] is True and attempts == [1]
    action = result["next_action"]
    assert action.startswith("This attempt FAILED: SafetyViolation: measured finger intersects")
    assert "call move_home before any other grasp" in action
    assert "then stop" in action  # the shared pick guidance is kept, after the recovery step
    assert action.index("move_home") < action.index("then stop")


def test_the_pick_description_lets_a_result_ask_for_move_home():
    """The description forbids move_home after the pick (it returns home itself);
    that rule must yield to a result that did NOT go home and asks for it."""
    from cascade.skills.runtime import TOOL_SPECS

    text = next(t for t in TOOL_SPECS if t["name"] == "pick_and_place")["description"]
    assert "do not call move_home again after this tool unless its next_action asks for it" in text


def test_a_failed_pick_that_went_home_does_not_ask_for_move_home(runtime, monkeypatch):
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
    runtime.cfg._data["grasp"]["max_pick_attempts"] = 1

    def refused(*a, **kw):
        raise SafetyViolation("grasp pose refused by the harness")

    runtime.skill_grasp_object = refused
    result = runtime.execute("pick_and_place", {"object": "red object"})
    assert result["ok"] is False and not result.get("home_skipped")
    assert "This attempt FAILED" in result["next_action"]
    assert "move_home" not in result["next_action"]


#: the recovery step every refused grasp that left the arm in place carries
MOVE_HOME_FIRST = "call move_home before any other grasp"


def _refuse_after_moving(runtime, error=R3_REFUSAL, exc=SafetyViolation, then=None):
    def refused(*a, **kw):
        # the attempt moved the arm: one motion the harness approves and counts
        q = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
        q[0] += 0.05
        assert runtime._move_joints(q, duration_s=0.3)
        if then is not None:
            then()
        raise exc(error)
    return refused


@pytest.mark.parametrize("name, args", [
    ("grasp_object", {"label": "green cube"}),
    ("grasp_at_pixel", {"u": 320, "v": 240}),
    ("handover", {"label": "green cube"}),
])
def test_a_grasp_that_left_the_arm_in_place_asks_for_move_home(runtime, name, args):
    """Review minor 8: the kitchen run's repeated refusals came from grasps
    started where the refused attempt had stopped; only pick_and_place said
    to call move_home first."""
    setattr(runtime, f"skill_{name}", _refuse_after_moving(runtime))
    result = runtime.execute(name, args)
    assert result["ok"] is False and "return_home" not in result
    action = result["next_action"]
    assert action.startswith("This attempt FAILED: SafetyViolation: measured finger intersects")
    assert MOVE_HOME_FIRST in action


def test_a_grasp_that_failed_before_moving_does_not_ask_for_move_home(runtime):
    def not_found(*a, **kw):
        raise SkillError("no detections for 'green cube'")
    runtime.skill_grasp_object = not_found
    result = runtime.execute("grasp_object", {"label": "green cube"})
    assert result["ok"] is False and MOVE_HOME_FIRST not in result["next_action"]


def test_a_failed_grasp_that_went_home_does_not_ask_for_move_home(runtime):
    runtime.cfg._data["arm"]["return_home_after_motion"] = True
    runtime.skill_grasp_object = _refuse_after_moving(
        runtime, "air grasp: gripper closed fully, object not held", SkillError)
    result = runtime.execute("grasp_object", {"label": "green cube"})
    assert result["ok"] is False and result["return_home"]["ok"] is True, result
    assert MOVE_HOME_FIRST not in result["next_action"]


def test_a_failed_grasp_under_a_latched_stop_does_not_ask_for_move_home(runtime):
    harness = runtime.arm.harness
    runtime.skill_grasp_object = _refuse_after_moving(runtime, then=lambda: harness.estop("operator"))
    try:
        result = runtime.execute("grasp_object", {"label": "green cube"})
    finally:
        harness.reset_estop()
    assert result["ok"] is False and MOVE_HOME_FIRST not in result["next_action"]


def test_a_failed_grasp_that_holds_something_does_not_ask_for_move_home(runtime):
    def holding():
        runtime.held_object = "green cube"
    runtime.skill_grasp_object = _refuse_after_moving(runtime, then=holding)
    try:
        result = runtime.execute("grasp_object", {"label": "green cube"})
    finally:
        runtime.held_object = None
    assert result["ok"] is False and MOVE_HOME_FIRST not in result["next_action"]


def _gate_refuses_every_grasp(runtime, monkeypatch):
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    # The refused attempt moved the arm (counted). Without a counted motion
    # no composite asks for move_home (test_agent_results_compose).
    runtime.skill_grasp_object = _refuse_after_moving(runtime)


def test_the_persistence_loop_refusal_asks_for_move_home(runtime, monkeypatch):
    _gate_refuses_every_grasp(runtime, monkeypatch)
    res = runtime._grasp_with_persistence("green cube")
    assert res["ok"] is False and res["home_skipped"] is True
    assert MOVE_HOME_FIRST in res["next_action"]


def test_a_handover_whose_grasp_was_refused_asks_for_move_home(runtime, monkeypatch):
    _gate_refuses_every_grasp(runtime, monkeypatch)
    result = runtime.execute("handover", {"label": "green cube"})
    assert result["ok"] is False and result["home_skipped"] is True
    assert result["next_action"].startswith("This attempt FAILED: could not grasp 'green cube' for handover")
    assert MOVE_HOME_FIRST in result["next_action"]


def test_the_move_home_step_is_given_once(runtime, monkeypatch):
    """A handover whose refused grasp moved the arm hits both rules (its own
    result and the dispatcher's): the agent reads the step once."""
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    runtime.skill_grasp_object = _refuse_after_moving(runtime)
    result = runtime.execute("handover", {"label": "green cube"})
    assert result["next_action"].count(MOVE_HOME_FIRST) == 1


def test_a_throw_whose_grasp_moved_and_was_refused_asks_for_move_home_once(runtime):
    """Re-review minor 3: throw grasps first; its refused inner grasp after
    the arm moved leaves the arm where that attempt stopped, like grasp_object."""
    runtime.skill_grasp_object = _refuse_after_moving(runtime)
    result = runtime.execute("throw", {"label": "green cube"})
    assert result["ok"] is False and runtime.held_object is None
    action = result["next_action"]
    assert action.startswith("This attempt FAILED: SafetyViolation: measured finger intersects")
    assert action.count(MOVE_HOME_FIRST) == 1


def _red(label="cube", color="red", at=(0.36, 0.12, 0.02)):
    return SimpleNamespace(label=label, color=color, position=np.array(at))


def test_a_sort_that_ended_at_a_refused_grasp_asks_for_move_home(runtime, monkeypatch):
    _gate_refuses_every_grasp(runtime, monkeypatch)
    monkeypatch.setattr(runtime.beliefs, "all", lambda: [_red()])
    result = runtime.execute("sort_by_color", {})
    assert result["ok"] is False and result["home_skipped"] is True, result
    assert MOVE_HOME_FIRST in result["next_action"]


def test_a_sort_that_moved_on_after_a_refused_grasp_does_not_ask(runtime, monkeypatch):
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    tries = []

    def grasp(query, material=None, budget_s=None):
        tries.append(query)
        if len(tries) == 1:
            return {"ok": False, "held": False, "error": "refused", "home_skipped": True}
        runtime.held_object = query
        return {"held": query, "grasp_attempts": 1}

    def place_at(x, y, z=None):
        runtime.held_object = None
        return {"placed": True, "at": [x, y, 0.05]}

    monkeypatch.setattr(runtime.beliefs, "all", lambda: [_red(), _red("ball", "blue", (0.36, -0.12, 0.02))])
    runtime._grasp_with_persistence = grasp
    runtime.skill_place_at = place_at
    out = runtime.skill_sort_by_color()
    assert len(tries) == 2 and len(out["moved"]) == 1, out
    assert not out.get("home_skipped") and MOVE_HOME_FIRST not in str(out.get("next_action"))


def _refused_persistence(query, material=None, budget_s=None):
    """The persistence loop's result after a refused grasp that moved the arm."""
    from cascade.skills.runtime import _with_move_home_first
    return _with_move_home_first({"ok": False, "held": False, "error": "SafetyViolation: " + R3_REFUSAL,
                                  "home_skipped": True, "grasp_attempts": 1})


def test_a_restore_that_ended_at_a_refused_grasp_asks_for_move_home():
    from test_pigey_scene_memory import A, BOUNDS, _stub, _two_cubes

    bs = _two_cubes(time.monotonic())
    rt = _stub(bs, bounds=BOUNDS)
    rt.skill_snapshot_scene(name="t")
    bs.find("red cube").position = np.array(A) + np.array([0.0, 0.08, 0.0])
    rt._grasp_with_persistence = _refused_persistence
    out = rt.skill_restore_scene(name="t")
    assert out["ok"] is False and out["home_skipped"] is True, out
    assert MOVE_HOME_FIRST in out["next_action"]
    assert rt.log.homes == 0  # still no unchecked automatic home


def test_a_search_that_ended_at_a_refused_grasp_asks_for_move_home():
    from test_pigey_scene_memory import BOUNDS, _cluttered, _stub

    rt = _stub(_cluttered(time.monotonic()), bounds=BOUNDS)
    rt._grasp_with_persistence = _refused_persistence
    out = rt.skill_search_for_object(label="green cube", max_occluders=1)
    assert out["ok"] is False and out["home_skipped"] is True, out
    assert MOVE_HOME_FIRST in out["next_action"] and "look-alike" in out["next_action"]
    assert rt.log.homes == 0


# ── 2. "next to X" is not "on X" ─────────────────────────────────────────


def test_pick_and_place_says_destination_is_on_or_in_and_how_to_place_beside():
    from cascade.skills.runtime import TOOL_SPECS

    spec = next(t for t in TOOL_SPECS if t["name"] == "pick_and_place")
    text = spec["description"]
    assert "ON/IN" in text
    assert "next to" in text and "beside" in text
    assert "grasp_object" in text and "place_at" in text
    assert "ON/IN" in spec["parameters"]["properties"]["destination"]["description"]
    # ... and how: where X is (localize_object), and how far beside it (review minor 5)
    how = text[text.index("next to"):]
    assert how.index("localize_object") < how.index("place_at")
    assert "1.5x" in how and "extent_m" in how and "free side" in how


# ── 6. which frame "left" used, and how it looks on the screen ───────────


@needs_pin
def test_move_relative_tells_the_agent_its_frame_and_the_screen_view(runtime):
    r = runtime.execute("move_relative", {"direction": "left", "amount": "small"})
    assert r["ok"], r
    action = r["next_action"]
    assert "robot's own directions" in action
    assert r["as_seen_on_screen"] in action
    assert "frame='viewer'" in action


@needs_pin
def test_a_viewer_frame_move_names_the_camera_view(runtime):
    r = runtime.execute("move_relative", {"direction": "toward_viewer", "distance_m": 0.03})
    assert r["ok"], r
    action = r["next_action"]
    assert "as seen on the camera" in action and r["as_seen_on_screen"] in action
    assert "frame='viewer'" not in action


def test_no_viewer_offer_without_a_screen_mapping():
    from cascade.skills.spatial_moves import _frame_report

    text = _frame_report("base", "2.0 cm to the robot's left (+y)", "screen mapping unavailable (no camera)")
    assert "robot's own directions" in text and "frame='viewer'" not in text


# ── 7. a client cancel only stops the call it names ──────────────────────


def _server(monkeypatch, inflight):
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    stops = []
    monkeypatch.setattr(server, "stop_now", lambda *a, **k: stops.append(1) or {"stopped": True})
    server._inflight = inflight
    return server, stops


def test_a_cancel_for_another_request_id_never_stops_a_running_pick(monkeypatch):
    """OpenClaw sends notifications/cancelled ~120 s after each session's
    initialize (connectionTimeoutMs), i.e. for the INITIALIZE request id,
    while a ~140 s pick of the same session is in flight."""
    server, stops = _server(monkeypatch, ("session|3", "pick_and_place"))
    server.cancel_request("session|0")
    assert stops == []
    assert "session|0" in server._cancelled_ids  # dropped if it is ever queued, never a stop
    server.cancel_request("other|3")  # the same JSON-RPC id from another session
    assert stops == []


def test_a_cancel_of_the_running_pick_is_the_designed_stop(monkeypatch):
    server, stops = _server(monkeypatch, ("session|3", "pick_and_place"))
    server.cancel_request("session|3")
    assert stops == [1]
