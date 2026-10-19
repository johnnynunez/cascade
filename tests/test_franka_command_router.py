"""Native command reports must preserve actual execution/verification evidence."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROUTER = Path(__file__).resolve().parents[1] / "deploy/openshell/franka-plugin/command-router.mjs"


@pytest.mark.parametrize("text,expected", [
    ("place the held object in the box", ("place_on_object", {"label": "box"})),
    ("put held object on the tray", ("place_on_object", {"label": "tray"})),
    ("place in the box", ("place_on_object", {"label": "box"})),
    ("put into the box", ("place_on_object", {"label": "box"})),
    ("pick and place the crayon in the box", ("pick_and_place", {"object": "crayon", "destination": "box"})),
    ("pick ball and place in the open box", ("pick_and_place", {"object": "ball", "destination": "open box"})),
    ("place it in the drop zone", ("place_on_object", {"label": "drop zone"})),
    ("pick and place the crayon", None),
    ("put into the box then return home", None),
    ("place the held object in box and open gripper", None),
    ("put into the box if holding something", None),
])
def test_place_spelling_preserves_held_vs_pick_semantics(text, expected):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the native OpenClaw command parser")
    script = """
const {parseRoutineCommand}=await import(process.argv[1]);
console.log(JSON.stringify(parseRoutineCommand(process.argv[2])));
"""
    completed = subprocess.run([node, "--input-type=module", "-e", script, ROUTER.as_uri(), text],
                               check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    if expected is None:
        assert result is None
    else:
        assert (result["name"], result["arguments"]) == expected


@pytest.fixture
def format_result():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the native OpenClaw command formatter")

    def invoke(payload, *, name="cascade__pick_and_place", result_overrides=None,
               expected_overrides=None):
        expected = {"id": "current-command", "name": name, **(expected_overrides or {})}
        result = {
            "role": "toolResult", "toolCallId": "current-command", "toolName": name,
            "isError": False, "content": [{"type": "text", "text": json.dumps(payload)}],
            **(result_overrides or {}),
        }
        script = """
const {formatRoutineResult}=await import(process.argv[1]);
const {result,expected}=JSON.parse(process.argv[2]);
console.log(JSON.stringify(formatRoutineResult(result,expected)));
"""
        completed = subprocess.run(
            [node, "--input-type=module", "-e", script, ROUTER.as_uri(),
             json.dumps({"result": result, "expected": expected})],
            check=True, capture_output=True, text=True,
        )
        return json.loads(completed.stdout)

    return invoke


@pytest.mark.parametrize("overrides", [
    {"role": "assistant"}, {"role": "user"},
    {"toolCallId": "earlier-command"}, {"toolName": "cascade__grasp_object"},
])
def test_report_requires_the_actual_matching_tool_result(format_result, overrides):
    assert format_result({"ok": True, "verified": True}, result_overrides=overrides) is None


@pytest.mark.parametrize("failure", [
    {"ok": False, "error": "Controller rejected joint 2 velocity"},
    {"execution_ok": False},
    {"status": "failed"},
    {"outcome": "stuck", "reason": "No current destination observation"},
    {"postcondition": {"status": "refuted", "evidence": "Object did not relocate"}},
    {"grasp": {"ok": False, "error": "Air grasp"}},
    {"place": {"execution_ok": False, "error": "Release refused"}},
    {"return_home": {"ok": False, "reason": "Arm is stopped"}},
])
def test_execution_failures_override_any_claimed_success(format_result, failure):
    report = format_result({"ok": True, "verified": True, **failure})
    assert "reported a failure" in report
    assert "verified the requested action" not in report
    for key in ("error", "reason"):
        if key in failure:
            assert failure[key] in report


def test_transport_error_is_never_a_verified_success(format_result):
    report = format_result({"ok": True, "verified": True}, result_overrides={"isError": True})
    assert "reported a failure" in report
    assert "verified the requested action" not in report


@pytest.mark.parametrize("evidence", [
    {"verified": False}, {"grip_verified": False},
    {"place": {"verified": False}},
    {"postcondition": {"status": "unverified", "evidence": "No fresh release observation"}},
])
def test_unverified_physical_outcomes_are_explicit(format_result, evidence):
    report = format_result({"ok": True, **evidence})
    assert "unverified" in report
    assert "verified the requested action" not in report


def test_success_requires_verification_not_just_transport_or_holding(format_result):
    tracked = format_result({"ok": True, "held": "orange ball"})
    assert "orange ball (session state)" in tracked
    assert "physical outcome is not verified" in tracked
    verified = format_result({
        "ok": True, "verified": True,
        "postcondition": {"status": "confirmed", "evidence": "Ball observed inside destination"},
    })
    assert "verified the requested action" in verified
    assert "Ball observed inside destination" in verified


def test_failed_home_and_stop_reason_survive_a_successful_grasp(format_result):
    report = format_result({
        "ok": False, "action_ok": True, "held": "ball", "grip_verified": True,
        "motion_blocked": True, "motion_block_reason": "Arm is stopped; resume required",
        "return_home": {"ok": False, "status": "failed", "error": "GOAL_TOLERANCE_VIOLATED"},
    })
    assert "reported a failure" in report
    assert "stopped or blocked" in report
    assert "Arm is stopped; resume required" in report
    assert "GOAL_TOLERANCE_VIOLATED" in report
    assert "Return home: failed" in report
    assert "Grasp contact: verified" in report
    assert "verified the requested action" not in report


def test_cached_status_does_not_claim_measured_contact_home_or_pose(format_result):
    report = format_result({
        "ok": True, "live_arm_feedback": False,
        "tracked_holding": {"label": "ball", "source": "session_state"},
        "gripper_contents": "not_measured", "home_pose": "not_verified",
        "tcp_xyz": [0.5, 0.0, 0.2],
        "arm_motion": {"stopped": True, "motion_error": "Franka is in reflex mode"},
    }, name="cascade__world_state")
    assert "ball (session state)" in report
    assert "does not contain fresh arm feedback" in report
    assert "Reported TCP" not in report
    assert "Franka is in reflex mode" in report
    assert "stopped or blocked" in report
    assert "returned home" not in report.lower()
    assert "Grasp contact: verified" not in report


def test_observation_distinguishes_estimates_from_fresh_measurements(format_result):
    report = format_result({
        "ok": True,
        "objects_visible": [{"label": "ball", "color": "orange"}],
        "objects_tracked": [{"label": "medicine cabinet", "state": "remembered"}],
        "robot": {"tcp_xyz": [0.5, 0.0, 0.2], "gripper_open_frac": 0.75, "holding": None},
    }, name="cascade__get_observation")
    assert "Detector estimates (unconfirmed): orange ball" in report
    assert "medicine cabinet" not in report
    assert "Reported TCP (m): [0.5, 0, 0.2]" in report
    assert "Reported gripper opening fraction: 0.75" in report
    assert "current gripper contents are not established" in report
    empty = format_result({"ok": True, "objects_visible": []}, name="cascade__get_observation")
    assert "does not establish an empty scene" in empty


def test_large_tool_payloads_are_bounded_and_remain_in_tool_card(format_result):
    report = format_result({
        "ok": False, "error": "Controller fault: " + "x" * 5000,
        "events": ["RAW EVENT PAYLOAD " * 1000],
        "objects": [{"label": f"object{i}"} for i in range(20)],
    })
    assert "Controller fault:" in report and "…" in report
    assert len(report) < 2300
    assert "object11" in report and "object12" not in report
    assert "RAW EVENT PAYLOAD" not in report
    assert "Full details are in the tool result" in report


def test_unstructured_errors_and_images_never_become_success(format_result):
    report = format_result(None, result_overrides={
        "isError": True, "content": [{"type": "text", "text": "FCI is unavailable"}],
    })
    assert "reported a failure" in report and "FCI is unavailable" in report
    assert "No structured execution or verification verdict" in report
    camera = format_result(None, name="cascade__camera_snapshot", result_overrides={
        "content": [{"type": "image", "data": "do-not-print-base64"}],
    })
    assert "Camera image is in the tool result" in camera
    assert "does not verify home position or gripper contents" in camera
    assert "do-not-print-base64" not in camera
