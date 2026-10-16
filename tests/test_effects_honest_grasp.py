"""The effects verifier never describes a grasp that did not happen.

Measured on the Spark kitchen (A/B run, 2026-10-16): grasps REFUSED before
any motion (SlowPerceptionError, gripper still open, 0 arm commands) came back
from `agent/effects._check_holding` as REFUTED with the evidence "<label> did
not rise ...: the jaw closed but the object stayed on the table". The agent
relayed "the jaw closed" to the user; the trace says the jaws never moved.

Pinned here, on the real mock-stack runtime with the verifier wired by
`execute()`:
* a grasp / pick that commanded no arm motion and no gripper close reports
  UNVERIFIED "no grasp was attempted: <error>", whatever the belief says;
* "the jaw closed" appears only when a close was commanded in this call;
* a real air grasp is still REFUTED by the gripper, a real held grasp is
  judged exactly as before.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from functools import partial

import numpy as np
import pytest

from conftest import needs_pin
from owned_server import held_dead_port

from cascade.agent.effects import (
    CONFIRMED,
    REFUTED,
    UNVERIFIED,
    PostconditionChecker,
)

pytestmark = needs_pin


@pytest.fixture
def mock_runtime(monkeypatch, tmp_path):
    """The real mock stack; no GraspGen-X server is ever reached."""
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
        runtime, arm = build_runtime(cfg, tmp_path / "run")
        try:
            assert runtime.effects is not None, "the verifier must be wired"
            yield runtime, arm
        finally:
            shutdown_runtime(runtime, arm)


def _seen(runtime, label="red cube"):
    """Put the target in the belief store, so the pre-motion snapshot has a
    pose and the old verifier reached its 'did not rise' branch."""
    runtime._reobserve()
    assert runtime.beliefs.find(label) is not None


def _slow_perception(monkeypatch):
    """Every analysed frame is past the age limit: the runtime's own
    SlowPerceptionError fires inside localization, before any motion."""
    import cascade.skills.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "_analyzed_frame_age_s",
                        partial(runtime_mod._analyzed_frame_age_s, max_age_s=-1.0))


def _texts(runtime, result) -> str:
    """Everything an agent or a report can read about this call."""
    return json.dumps(result, default=str) + "\n" + runtime.effects.digest(last=10)


@pytest.mark.parametrize("skill, args", [
    ("grasp_object", {"label": "red cube"}),
    ("pick_and_place", {"object": "red cube"}),
])
def test_refused_before_motion_is_no_grasp_attempt(mock_runtime, monkeypatch, skill, args):
    runtime, arm = mock_runtime
    _seen(runtime)
    _slow_perception(monkeypatch)
    n0 = len(arm.commands)
    frac0 = runtime._gripper_width_frac()

    result = runtime.execute(skill, args)

    assert result["ok"] is False
    assert "Perception took" in result["error"]
    assert len(arm.commands) == n0, "refused: the arm stayed put"
    assert runtime._gripper_width_frac() == pytest.approx(frac0), "the jaws never moved"
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["channel"] == "skill"
    assert pc["evidence"].startswith("no grasp was attempted: "), pc
    assert "Perception took" in pc["evidence"]
    assert pc["measured"] == {"arm_motions": 0, "gripper_closes": 0, "gripper_opens": 0}
    text = _texts(runtime, result)
    assert "jaw closed" not in text and "did not rise" not in text
    assert result["verified"] is False


def test_real_air_grasp_is_still_refuted(mock_runtime):
    """Main's verdict, word for word: here a close WAS commanded. (The skill
    re-opens the jaws to 0.15 after detecting the air grasp, so the verifier
    reads an open jaw and refutes through the unchanged object pose.)"""
    runtime, arm = mock_runtime
    _seen(runtime)
    arm.object_stop_frac = None          # the jaws close fully on nothing
    n0 = len(arm.commands)

    result = runtime.execute("grasp_object", {"label": "red cube"})

    assert len(arm.commands) > n0, "this grasp really moved"
    assert result["ok"] is False and result["error"].startswith("air grasp"), result
    pc = result["postcondition"]
    assert pc["status"] == REFUTED and pc["channel"] == "belief", pc
    assert pc["evidence"] == ("red cube did not rise (dz=+0.0 cm): the jaw closed but "
                              "the object stayed on the table"), pc


def test_grasp_failing_after_motion_is_judged_not_excused(mock_runtime, monkeypatch):
    """The arm approached and descended, then the close was refused: that
    grasp WAS attempted, so it is judged (REFUTED: nothing rose), and the
    evidence states the open jaw instead of claiming it closed."""
    from cascade.types import SkillError

    runtime, arm = mock_runtime
    _seen(runtime)
    n0 = len(arm.commands)

    def refused_close(*_a, **_k):
        raise SkillError("test: close refused at the grasp pose")

    monkeypatch.setattr(runtime, "_close_two_stage", refused_close)
    result = runtime.execute("grasp_object", {"label": "red cube"})

    assert len(arm.commands) > n0, "the arm approached the object"
    assert result["ok"] is False and "close refused" in result["error"], result
    pc = result["postcondition"]
    assert pc["status"] == REFUTED, pc
    assert pc["evidence"].startswith("red cube did not rise (dz=+0.0 cm; jaw at "), pc
    assert "closed" not in pc["evidence"]


def test_real_held_grasp_is_judged_as_before(mock_runtime):
    runtime, arm = mock_runtime
    _seen(runtime)
    arm.object_stop_frac = 0.5           # the jaws jam on the object
    n0 = len(arm.commands)

    result = runtime.execute("grasp_object", {"label": "red cube"})

    assert len(arm.commands) > n0
    assert result["ok"] is True and result["held"] == "red cube", result
    pc = result["postcondition"]
    # The static mock camera cannot see the lift: the same verdict as main,
    # and the evidence states the measured jaw ("held" is not measured).
    assert pc["status"] == UNVERIFIED and pc["channel"] == "gripper", pc
    assert pc["evidence"].startswith("jaw at 0.500 but red cube could not be re-located"), pc


# ── the checker on its own: the actuation hook decides the wording ─────────


class _Tally:
    def __init__(self):
        self.counts = {"arm_motions": 0, "gripper_closes": 0, "gripper_opens": 0}

    def __call__(self):
        return dict(self.counts)


def _checker(tally, frac=0.99, z_after=0.04):
    poses = {"cube": [0.3, 0.0, 0.04]}
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label),
                                   gripper_frac=lambda: frac, actuation=tally)
    before = checker.snapshot("cube")
    poses["cube"] = [0.3, 0.0, z_after]
    return checker, before


def test_no_motion_no_close_failed_grasp_is_unverified():
    tally = _Tally()
    checker, before = _checker(tally)
    pc = checker.verify("grasp_object", {"label": "cube"},
                        {"ok": False, "error": "SkillError: no IK solution\nsecond line"}, before=before)
    assert pc.status == UNVERIFIED and pc.channel == "skill"
    assert pc.evidence == "no grasp was attempted: SkillError: no IK solution"
    assert pc.measured == {"arm_motions": 0, "gripper_closes": 0, "gripper_opens": 0}


def test_long_error_is_shortened_in_the_no_attempt_evidence():
    tally = _Tally()
    checker, before = _checker(tally)
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": False, "error": "x" * 500}, before=before)
    assert pc.status == UNVERIFIED
    assert len(pc.evidence) < 200 and pc.evidence.endswith("...")


def test_arm_motion_without_close_never_claims_the_jaw_closed():
    tally = _Tally()
    checker, before = _checker(tally)
    tally.counts["arm_motions"] += 3     # approached, then failed before the close
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": False, "error": "SkillError: x"},
                        before=before)
    assert pc.status == REFUTED          # the object did not rise: same verdict as before
    assert "did not rise" in pc.evidence and "closed" not in pc.evidence
    assert "jaw at 0.99" in pc.evidence


def test_commanded_close_keeps_the_jaw_closed_wording():
    tally = _Tally()
    checker, before = _checker(tally, frac=0.5)
    tally.counts["arm_motions"] += 3
    tally.counts["gripper_closes"] += 2
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": True}, before=before)
    assert pc.status == REFUTED
    assert pc.evidence.endswith("the jaw closed but the object stayed on the table")


def test_a_close_alone_is_a_grasp_attempt():
    """A close with no arm motion still acted on the world: judge it."""
    tally = _Tally()
    checker, before = _checker(tally, frac=0.01)
    tally.counts["gripper_closes"] += 1
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": False, "error": "air grasp"},
                        before=before)
    assert pc.status == REFUTED and pc.channel == "gripper"
    assert pc.evidence.startswith("jaw closed to 0.010")


def test_measured_closed_jaw_without_a_close_is_described_not_claimed():
    tally = _Tally()
    checker, before = _checker(tally, frac=0.01)
    tally.counts["arm_motions"] += 1
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": False, "error": "x"}, before=before)
    assert pc.status == REFUTED and pc.channel == "gripper"
    assert pc.evidence.startswith("jaw at 0.010 (<= air-grasp threshold)")
    assert "closed" not in pc.evidence


def test_a_self_reported_success_is_never_softened_by_the_guard():
    """No counted actuation but ok=True: the independent channels still judge."""
    tally = _Tally()
    checker, before = _checker(tally)
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": True}, before=before)
    assert pc.status == REFUTED and "did not rise" in pc.evidence


def test_unknown_actuation_keeps_the_verdict_but_drops_the_close_claim():
    poses = {"cube": [0.3, 0.0, 0.04]}
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label), gripper_frac=lambda: 0.99)
    before = checker.snapshot("cube")
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": False, "error": "x"}, before=before)
    assert pc.status == REFUTED and "did not rise" in pc.evidence
    assert "closed" not in pc.evidence


def test_pick_without_motion_is_no_grasp_attempt():
    tally = _Tally()
    checker, before = _checker(tally)
    pc = checker.verify("pick_and_place", {"object": "cube"},
                        {"ok": False, "stage": "grasp", "error": "Perception took 6.1s"}, before=before)
    assert pc.status == UNVERIFIED and pc.channel == "skill"
    assert pc.evidence == "no grasp was attempted: Perception took 6.1s"


def test_close_gripper_without_a_completed_close_is_not_confirmed():
    tally = _Tally()
    checker = PostconditionChecker(gripper_frac=lambda: 1.0, actuation=tally)
    before = checker.snapshot(None)
    pc = checker.verify("close_gripper", {}, {"ok": False, "error": "SafetyViolation: e-stop latched"},
                        before=before)
    assert pc.status == UNVERIFIED and pc.channel == "gripper"
    assert "closed on" not in pc.evidence and "jaw at 1.00" in pc.evidence


def test_close_gripper_with_a_close_is_confirmed_as_before():
    tally = _Tally()
    checker = PostconditionChecker(gripper_frac=lambda: 0.5, actuation=tally)
    before = checker.snapshot(None)
    tally.counts["gripper_closes"] += 2
    pc = checker.verify("close_gripper", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence == "jaw at 0.50: closed on something"


def test_only_a_safety_harness_counts_motions():
    """A harness that is not a SafetyHarness never runs begin_motion's count,
    so its zero must read as 'unknown', never as 'the arm did not move'."""
    from types import SimpleNamespace

    from cascade.safety.harness import SafetyHarness
    from cascade.skills.runtime import SkillRuntime

    def runtime_with(harness):
        rt = SkillRuntime.__new__(SkillRuntime)
        rt._arm = SimpleNamespace(harness=harness)
        return rt

    foreign = SimpleNamespace(motions_begun=0)
    assert runtime_with(foreign)._actuation_counts() is None
    real = SafetyHarness.__new__(SafetyHarness)
    real.motions_begun = 4
    assert runtime_with(real)._actuation_counts() == {
        "arm_motions": 4, "gripper_closes": 0, "gripper_opens": 0}


def test_close_gripper_skill_counts_its_close(mock_runtime):
    runtime, arm = mock_runtime
    arm.object_stop_frac = 0.5
    result = runtime.execute("close_gripper", {})
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["evidence"] == "jaw at 0.50: closed on something", pc


def test_driver_two_stage_close_is_counted(mock_runtime, monkeypatch):
    """A backend with its own two-stage close (the reBot) bypasses SafeArm's
    set_gripper; its close still counts."""
    runtime, arm = mock_runtime
    arm.object_stop_frac = 0.5
    calls = []

    def two_stage(**kw):
        calls.append(kw)
        arm.set_gripper(arm._grip_closed)

    monkeypatch.setattr(arm, "close_gripper_two_stage", two_stage, raising=False)
    result = runtime.execute("close_gripper", {})
    assert len(calls) == 1
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["evidence"] == "jaw at 0.50: closed on something", pc


def test_force_contact_close_is_counted(mock_runtime, monkeypatch):
    runtime, arm = mock_runtime
    arm.object_stop_frac = 0.5
    monkeypatch.setattr(runtime, "_force_contact_close_enabled", lambda: True)
    result = runtime.execute("close_gripper", {})
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["evidence"] == "jaw at 0.50: closed on something", pc


def test_refused_close_gripper_is_not_verified_closed(mock_runtime):
    """The e-stop refuses the close before the driver: the jaws stay open,
    and nothing may report them 'closed on something'."""
    runtime, arm = mock_runtime
    frac0 = runtime._gripper_width_frac()
    runtime.arm.harness.estop("test: latched before the close")

    result = runtime.execute("close_gripper", {})

    assert result["ok"] is False and "e-stop latched" in result["error"], result
    assert runtime._gripper_width_frac() == pytest.approx(frac0)
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert "closed on" not in pc["evidence"] and result["verified"] is False


# ── fix round 1: no verdict asserts a release, home or throw that did not happen ──
#
# The same actuation hook counts completed RELEASE opens (open_gripper, the
# place_at release, the throw's release). Measured on the mock stack before
# this round: an open refused by the e-stop while holding came back CONFIRMED
# "the object was released" (object still in the jaws), a handover that never
# grasped as "the object was released", a refused place_at as "gripper opened
# to 0.50", a refused move_home as CONFIRMED "arm commanded home", a throw
# refused mid-swing as CONFIRMED "no longer on the table".

_EVENT_WORDS = ("released", "opened", "placed", "thrown", "the throw", "no longer on the table",
                "arm commanded home", "jaw closed", "jaw held")


def _hold(runtime, arm, label="red cube"):
    """A real held grasp on the mock (the jaws jam at 0.5 on the object)."""
    _seen(runtime, label)
    arm.object_stop_frac = 0.5
    result = runtime.execute("grasp_object", {"label": label})
    assert result["ok"] is True and result["held"] == label, result


def _assert_no_event(runtime, result):
    pc = result["postcondition"]
    for word in _EVENT_WORDS:
        assert word not in pc["evidence"], pc
    assert result["verified"] is False, result


def test_open_refused_while_holding_is_not_a_release(mock_runtime):
    runtime, arm = mock_runtime
    _hold(runtime, arm)
    runtime.arm.harness.estop("test: latched while holding")
    n0 = len(arm.commands)

    result = runtime.execute("open_gripper", {})

    assert result["ok"] is False and "e-stop latched" in result["error"], result
    assert len(arm.commands) == n0
    assert runtime.held_object == "red cube", "the object is still in the jaws"
    assert runtime._gripper_width_frac() == pytest.approx(0.5)
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED and pc["channel"] == "skill", pc
    assert pc["evidence"] == "no open command completed: SafetyViolation: e-stop latched; jaw at 0.50", pc
    assert pc["measured"]["gripper_opens"] == 0 and pc["measured"]["gripper_frac"] == pytest.approx(0.5)
    _assert_no_event(runtime, result)


def test_open_refused_with_open_jaws_is_not_a_release(mock_runtime):
    """Jaws already open, nothing held, the open refused: an open reading is
    not a release this call made."""
    runtime, arm = mock_runtime
    runtime.arm.harness.estop("test: latched")

    result = runtime.execute("open_gripper", {})

    assert result["ok"] is False
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"].startswith("no open command completed: ") and pc["evidence"].endswith("jaw at 1.00"), pc
    _assert_no_event(runtime, result)


def test_open_that_releases_is_confirmed_as_before(mock_runtime):
    runtime, arm = mock_runtime
    _hold(runtime, arm)

    result = runtime.execute("open_gripper", {})

    assert result["ok"] is True and runtime.held_object is None
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["channel"] == "gripper", pc
    assert pc["evidence"] == "gripper open at 1.00: the object was released", pc


def test_place_refused_while_holding_is_not_a_release(mock_runtime):
    runtime, arm = mock_runtime
    _hold(runtime, arm)
    runtime.arm.harness.estop("test: latched while holding")
    n0 = len(arm.commands)

    result = runtime.execute("place_at", {"x": 0.3, "y": 0.1})

    assert result["ok"] is False and "e-stop latched" in result["error"], result
    assert len(arm.commands) == n0
    assert runtime.held_object == "red cube", "the object is still in the jaws"
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED and pc["channel"] == "skill", pc
    assert pc["evidence"] == "no open command completed: SafetyViolation: e-stop latched; jaw at 0.50", pc
    _assert_no_event(runtime, result)


def test_place_that_releases_is_judged_as_before(mock_runtime):
    runtime, arm = mock_runtime
    _hold(runtime, arm)

    result = runtime.execute("place_at", {"x": 0.3, "y": 0.1})

    assert result["ok"] is True and runtime.held_object is None, result
    pc = result["postcondition"]
    # Main's verdict word for word: the release was commanded and counted.
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"] == "gripper opened to 1.00 but the released object was not re-located", pc


def test_handover_refused_by_slow_perception_stops_and_releases_nothing(mock_runtime, monkeypatch):
    """Like pick_and_place: the first SlowPerceptionError ends the handover
    (no re-home per retry), and the verdict claims no release."""
    runtime, arm = mock_runtime
    _seen(runtime)
    _slow_perception(monkeypatch)
    n0 = len(arm.commands)

    result = runtime.execute("handover", {"label": "red cube"})

    assert result["ok"] is False and "Perception took" in result["error"], result
    assert result["grasp_attempts"] == 1, result
    assert len(arm.commands) == n0, "no automatic re-home after an expired analysis"
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"].startswith("no open command completed: could not grasp 'red cube' for handover: "), pc
    _assert_no_event(runtime, result)
    assert "was released" not in _texts(runtime, result)


def test_handover_that_keeps_holding_is_not_a_release(mock_runtime):
    """handover presents the object and KEEPS holding it: jaws at 0.50 on the
    object are not 'the object was released'."""
    runtime, arm = mock_runtime
    _seen(runtime)
    arm.object_stop_frac = 0.5

    result = runtime.execute("handover", {"label": "red cube"})

    assert result["ok"] is not False and result["offering"] == "red cube", result
    assert runtime.held_object == "red cube"
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"] == "no open command completed; jaw at 0.50", pc
    assert "released" not in pc["evidence"]


def test_throw_refused_before_motion_is_not_a_throw(mock_runtime, monkeypatch):
    runtime, arm = mock_runtime
    _seen(runtime)
    _slow_perception(monkeypatch)
    n0 = len(arm.commands)

    result = runtime.execute("throw", {"label": "red cube"})

    assert result["ok"] is False and len(arm.commands) == n0, result
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"].startswith("no open command completed: SlowPerceptionError: Perception took"), pc
    _assert_no_event(runtime, result)
    assert "travelled" not in pc["evidence"]


def test_throw_refused_mid_swing_keeps_the_object_and_says_so(mock_runtime):
    """On the mock the swing leaves the workspace and the harness stops it
    before the release: the object is still in the jaws."""
    runtime, arm = mock_runtime
    _hold(runtime, arm)
    n0 = len(arm.commands)

    result = runtime.execute("throw", {"label": "red cube"})

    assert len(arm.commands) > n0, "the wind-up moved the arm"
    assert result["ok"] is False and "outside workspace" in result["error"], result
    assert runtime.held_object == "red cube"
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"].startswith("no open command completed: SafetyViolation: TCP"), pc
    assert pc["evidence"].endswith("; jaw at 0.50"), pc
    _assert_no_event(runtime, result)


def test_throw_that_releases_is_judged_by_the_object(mock_runtime, monkeypatch):
    """The release completes and is counted: the verdict rests on the object,
    which is not re-located (no "gone" claim). The mock profile's wind-up
    pose dips under the table clearance, so the swing itself is stubbed; the
    release is the real gripper command."""
    runtime, arm = mock_runtime
    _hold(runtime, arm)
    monkeypatch.setattr(runtime, "_move_joints", lambda *_a, **_k: True)

    result = runtime.execute("throw", {"label": "red cube"})

    assert result["ok"] is True and result["thrown"] == "red cube", result
    assert runtime.held_object is None
    assert runtime._gripper_width_frac() == pytest.approx(1.0)
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["evidence"] == "red cube could not be re-located", pc


def test_place_drop_after_a_slip_is_a_completed_open(mock_runtime, monkeypatch):
    """place_at sees the object far below the jaws (it slipped in the carry)
    and opens to drop it: that open completed, so the verdict is judged on
    the jaws/object, never "no open command completed"."""
    from cascade.skills.held_observation import HeldOffset

    runtime, arm = mock_runtime
    _hold(runtime, arm)
    monkeypatch.setattr(runtime, "_held_object_observation",
                        lambda: HeldOffset(np.array([0.0, 0.0, -0.10]), "test", True))

    result = runtime.execute("place_at", {"x": 0.3, "y": 0.1})

    assert result["ok"] is False and "slipped out of the gripper" in result["error"], result
    assert runtime.held_object is None
    assert runtime._gripper_width_frac() == pytest.approx(1.0)
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED, pc
    assert pc["evidence"] == "gripper opened to 1.00 but the released object was not re-located", pc


def test_move_home_refused_by_the_estop_is_not_home(mock_runtime):
    runtime, arm = mock_runtime
    runtime.arm.harness.estop("test: latched")
    n0 = len(arm.commands)

    result = runtime.execute("move_home", {})

    assert result["ok"] is False and len(arm.commands) == n0, result
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED and pc["channel"] == "skill", pc
    assert pc["evidence"].startswith("no motion completed: SkillError: no safe joint route found"), pc
    assert "e-stop latched" in pc["evidence"]
    _assert_no_event(runtime, result)


def test_move_home_that_moves_is_confirmed_as_before(mock_runtime):
    runtime, arm = mock_runtime
    n0 = len(arm.commands)

    result = runtime.execute("move_home", {})

    assert result["ok"] is True and len(arm.commands) > n0
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["channel"] == "arm", pc
    assert pc["evidence"] == "arm commanded home (streamed waypoints approved by the harness)"


# ── an arm motion counts when its stream starts (first approved command) ──


def _nudged(runtime, dq=0.05):
    q = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    q[0] += dq
    return q


def _refuse_route(start, duration_s):
    from cascade.types import SafetyViolation

    raise SafetyViolation("test: route refused by its preflight; no motion sent")


@pytest.mark.parametrize("refusal", ["estop", "watchdog", "halt", "preflight"])
def test_a_refused_motion_is_not_counted(mock_runtime, refusal):
    """begin_motion's own checks (e-stop, watchdog, halt generation) and a
    route preflight after it all refuse before the first command: nothing
    moved, nothing is counted, and a later bare approve() (no motion in
    flight) does not count it either."""
    from cascade.types import SafetyViolation

    runtime, arm = mock_runtime
    h = runtime.arm.harness
    q0 = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    limits = h.limits
    kwargs = {}
    if refusal == "estop":
        h.estop("test: latched")
    elif refusal == "watchdog":
        h.limits = dataclasses.replace(limits, watchdog_s=-1.0)
    elif refusal == "halt":
        kwargs["_halt_generation"] = h._halt_generation + 1
    else:
        kwargs["_preflight"] = _refuse_route
    n0, m0 = len(arm.commands), h.motions_begun

    with pytest.raises(SafetyViolation):
        runtime.arm.move_joints(_nudged(runtime), duration_s=0.5, **kwargs)

    assert len(arm.commands) == n0, "refused: no command left"
    assert h.motions_begun == m0, "a refused motion is not an arm motion"
    h.limits = limits
    h.heartbeat()
    with contextlib.suppress(SafetyViolation):
        h.approve(q0, q0, 0.02)          # a bare vet, no motion in flight
    assert h.motions_begun == m0


def test_a_stream_counts_once(mock_runtime):
    runtime, arm = mock_runtime
    h = runtime.arm.harness
    n0, m0 = len(arm.commands), h.motions_begun

    assert runtime.arm.move_joints(_nudged(runtime), duration_s=0.5)

    assert len(arm.commands) - n0 > 2, "many approved waypoints"
    assert h.motions_begun == m0 + 1, "one stream, one motion"
    q0 = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    h.heartbeat()
    h.approve(q0, q0, 0.02)              # after the stream ended: not a new motion
    assert h.motions_begun == m0 + 1


# ── the checker on its own: release, throw, home and relocation wording ──


def _release_checker(tally, frac=0.97, poses=None):
    poses = {"cube": [0.3, 0.0, 0.04], "plate": [0.3, 0.1, 0.02]} if poses is None else poses
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label),
                                   gripper_frac=lambda: frac, actuation=tally)
    return checker, poses


@pytest.mark.parametrize("skill, args, result", [
    ("open_gripper", {}, {"ok": False, "error": "SafetyViolation: e-stop latched"}),
    ("handover", {"label": "cube"}, {"ok": False, "error": "SafetyViolation: e-stop latched"}),
    ("place_at", {"x": 0.3, "y": 0.0}, {"ok": False, "error": "SafetyViolation: e-stop latched"}),
    ("place_on_object", {"label": "plate"}, {"ok": False, "placed": "cube",
                                             "error": "SafetyViolation: e-stop latched"}),
    ("throw", {"label": "cube"}, {"ok": False, "error": "SafetyViolation: e-stop latched"}),
])
def test_no_counted_open_is_never_a_release(skill, args, result):
    tally = _Tally()
    checker, _ = _release_checker(tally, frac=0.5)
    before = checker.snapshot("cube")
    tally.counts["arm_motions"] += 2            # moved, but never opened
    pc = checker.verify(skill, args, result, before=before)
    assert pc.status == UNVERIFIED and pc.channel == "skill", pc
    assert pc.evidence == "no open command completed: SafetyViolation: e-stop latched; jaw at 0.50", pc
    assert pc.measured == {"arm_motions": 2, "gripper_closes": 0, "gripper_opens": 0, "gripper_frac": 0.5}


def test_a_counted_open_keeps_the_release_wording():
    """Held, a counted open, and the jaws measured wider than before the call."""
    tally = _Tally()
    jaw = {"frac": 0.5}
    checker = PostconditionChecker(gripper_frac=lambda: jaw["frac"], actuation=tally, holding=lambda: True)
    before = checker.snapshot(None, skill="open_gripper")
    tally.counts["gripper_opens"] += 1
    jaw["frac"] = 0.97
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence == "gripper open at 0.97: the object was released"
    assert pc.measured == {"gripper_frac": 0.97, "gripper_frac_before": 0.5}, pc


def test_unknown_actuation_never_claims_a_release():
    checker = PostconditionChecker(gripper_frac=lambda: 0.97)
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=checker.snapshot(None))
    assert pc.status == CONFIRMED and pc.evidence == "gripper open at 0.97", pc


def test_unknown_actuation_never_claims_a_release_even_after_a_widening():
    """Held, and the jaw measured wider, but no actuation hook: nothing says
    THIS call opened the jaws, so the reading only."""
    jaw = {"frac": 0.5}
    checker = PostconditionChecker(gripper_frac=lambda: jaw["frac"], holding=lambda: True)
    before = checker.snapshot(None, skill="open_gripper")
    jaw["frac"] = 0.97
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence == "gripper open at 0.97", pc


def test_released_at_wording_after_a_counted_open():
    tally = _Tally()
    checker, poses = _release_checker(tally)
    before = checker.snapshot("cube")
    tally.counts["gripper_opens"] += 1
    poses["cube"] = [0.3, 0.3, 0.02]            # 30 cm from the requested point
    pc = checker.verify("place_at", {"x": 0.3, "y": 0.0}, {"ok": True}, before=before)
    assert pc.status == REFUTED, pc
    assert pc.evidence == "cube ended 30.0 cm from the requested point", pc


def test_unknown_actuation_released_at_states_the_reading():
    poses = {"cube": [0.3, 0.0, 0.04]}
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label), gripper_frac=lambda: 0.97)
    before = checker.snapshot("cube")
    poses.pop("cube")
    pc = checker.verify("place_at", {"x": 0.3, "y": 0.0}, {"ok": True}, before=before)
    assert pc.status == UNVERIFIED and pc.evidence == "jaw at 0.97 but cube was not re-located", pc


def test_reopened_jaw_is_described_not_held():
    """A close was commanded, then the skill re-opened the jaws (the air-grasp
    re-open to 0.15): the measured jaw is not a hold."""
    tally = _Tally()
    poses = {"cube": [0.3, 0.0, 0.04]}
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label),
                                   gripper_frac=lambda: 0.15, actuation=tally)
    before = checker.snapshot("cube")
    poses.pop("cube")                           # not re-located after the call
    tally.counts["arm_motions"] += 3
    tally.counts["gripper_closes"] += 1
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": False, "error": "air grasp"}, before=before)
    assert pc.status == UNVERIFIED and pc.channel == "gripper", pc
    assert pc.evidence == ("jaw at 0.150 but cube could not be re-located, so the lift is unconfirmed"), pc


def test_throw_with_a_counted_release_states_measurements():
    tally = _Tally()
    checker, poses = _release_checker(tally)
    before = checker.snapshot("cube")
    tally.counts["arm_motions"] += 3
    tally.counts["gripper_opens"] += 1
    pc = checker.verify("throw", {"label": "cube"}, {"ok": True}, before=before)
    assert pc.status == REFUTED and pc.evidence == "cube only travelled 0 cm", pc
    poses.pop("cube")
    pc = checker.verify("throw", {"label": "cube"}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence == "cube could not be re-located", pc


def test_move_home_counts_decide_the_home_verdict():
    tally = _Tally()
    checker = PostconditionChecker(actuation=tally)
    before = checker.snapshot(None)
    pc = checker.verify("move_home", {}, {"ok": False, "error": "SafetyViolation: e-stop latched"}, before=before)
    assert pc.status == UNVERIFIED and pc.evidence == "no motion completed: SafetyViolation: e-stop latched", pc
    tally.counts["arm_motions"] += 1
    pc = checker.verify("move_home", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence.startswith("arm commanded home"), pc


def test_relocated_without_a_comparable_start_claims_no_move():
    tally = _Tally()
    poses = {}
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label), actuation=tally)
    before = checker.snapshot("cube")           # not located before the call
    tally.counts["arm_motions"] += 3
    tally.counts["gripper_closes"] += 1
    poses["cube"] = [0.3, 0.1, 0.02]
    pc = checker.verify("pick_and_place", {"object": "cube"}, {"ok": True}, before=before)
    assert pc.status == UNVERIFIED, pc
    assert "moved" not in pc.evidence and pc.evidence.startswith("cube was re-located"), pc
    # With a comparable start the move is measured, and said.
    poses["cube"] = [0.3, 0.0, 0.02]
    before = checker.snapshot("cube")
    poses["cube"] = [0.3, 0.2, 0.02]
    pc = checker.verify("pick_and_place", {"object": "cube"}, {"ok": True}, before=before)
    assert pc.status == UNVERIFIED and pc.evidence.startswith("cube moved 20.0 cm, but no drop point"), pc


# ── integration seam: the two-phase SceneChangedError (T1) meets T3's handover rule ──


@pytest.mark.parametrize("entry", ["handover", "persistence loop"])
def test_a_scene_change_ends_a_persistent_grasp_like_slow_perception(mock_runtime, entry):
    """SceneChangedError is handled like SlowPerceptionError (spec 7.3): no
    automatic re-home and re-scan per retry. pick_and_place stops on both;
    the shared persistence loop (handover, sort_by_color) must too."""
    from cascade.skills.runtime import SceneChangedError

    runtime, arm = mock_runtime
    _seen(runtime)
    calls = []

    def scene_changed(*_a, **_k):
        calls.append(1)
        raise SceneChangedError("scene changed during perception: test")

    runtime.skill_grasp_object = scene_changed
    n0 = len(arm.commands)

    if entry == "handover":
        result = runtime.execute("handover", {"label": "red cube"})
    else:
        result = runtime._grasp_with_persistence("red cube")

    assert len(calls) == 1, f"{len(calls)} grasp attempts after a scene change"
    assert len(arm.commands) == n0, "no automatic re-home after a scene change"
    assert result["ok"] is False and "scene changed during perception" in result["error"], result


# ── polish round (integration 1): home and release verdicts follow what was measured ──


def _view_home_runtime(tmp_path, q):
    """The fr3 saved camera-view home (`_return_to_view_home`) on local
    doubles, with the verifier wired as the app wires it."""
    from test_return_home_policy import _runtime

    runtime, arm, _events = _runtime(tmp_path, q=q)
    runtime.attach_verifier()
    return runtime, arm


def test_move_home_already_at_home_is_confirmed_from_the_measured_pose(tmp_path):
    """T3-a: the arm is already within settle_tol of home, so the view-home
    return sends no motion. The measured joints prove the postcondition:
    CONFIRMED, never 'no motion completed'."""
    runtime, arm = _view_home_runtime(tmp_path, q=np.full(7, 0.004))
    tol = float(runtime.cfg.arm.get("settle_tol", 0.03))       # fr3_mock: 0.02
    assert tol > 0.004

    result = runtime.execute("move_home", {})

    assert result["ok"] is True and result["return_home"]["status"] == "already_home", result
    assert arm.moves == [], "already home: no motion sent"
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["channel"] == "arm", pc
    assert pc["evidence"] == (f"arm already at home: measured joint error 0.004 rad (<= {tol:.3f} rad); "
                              "no motion needed"), pc
    assert pc["measured"] == {"joint_error_rad": 0.004, "settle_tol_rad": tol}, pc
    assert result["verified"] is True


@pytest.mark.parametrize("home, why", [
    ({"ok": True, "status": "already_home", "home_q": [0.0] * 7}, "no measurement"),
    ({"ok": True, "status": "already_home", "joint_error_rad": 0.05, "settle_tol_rad": 0.03}, "outside tol"),
    ({"ok": True, "status": "already_home", "joint_error_rad": float("nan"), "settle_tol_rad": 0.03}, "nan"),
    ({"ok": True, "status": "already_home", "joint_error_rad": 0.5, "settle_tol_rad": float("inf")}, "inf tol"),
    ({"ok": True, "status": "already_home", "joint_error_rad": -0.01, "settle_tol_rad": 0.03}, "negative"),
    ({"ok": True, "status": "returned", "joint_error_rad": 0.0, "settle_tol_rad": 0.03}, "not already home"),
    ({"ok": False, "status": "already_home", "joint_error_rad": 0.0, "settle_tol_rad": 0.03}, "not ok"),
])
def test_already_home_needs_a_measurement_within_tolerance(home, why):
    tally = _Tally()
    checker = PostconditionChecker(actuation=tally)
    before = checker.snapshot(None)
    pc = checker.verify("move_home", {}, {"ok": True, "at": "home", "return_home": home}, before=before)
    assert pc.status == UNVERIFIED and pc.evidence == "no motion completed", (why, pc)


def test_already_home_is_confirmed_with_unknown_actuation():
    checker = PostconditionChecker()
    home = {"ok": True, "status": "already_home", "joint_error_rad": 0.01, "settle_tol_rad": 0.03}
    pc = checker.verify("move_home", {}, {"ok": True, "at": "home", "return_home": home}, before={})
    assert pc.status == CONFIRMED and pc.evidence.startswith("arm already at home: measured joint error 0.010"), pc


def test_move_home_stopped_mid_route_is_never_home(mock_runtime, monkeypatch):
    """T3-b: the stream started (a counted motion) but the move ended short
    of home (ok False): never CONFIRMED, never 'arm commanded home'."""
    runtime, arm = mock_runtime
    assert runtime.arm.move_joints(_nudged(runtime, 0.3), duration_s=1.0)
    real = runtime.arm.move_planned

    def stopped_short(q, **kwargs):
        real(q, **kwargs)              # streams toward home (counted)...
        return False                   # ...but did not settle there

    monkeypatch.setattr(runtime.arm, "move_planned", stopped_short)
    h = runtime.arm.harness
    n0, m0 = len(arm.commands), h.motions_begun

    result = runtime.execute("move_home", {})

    assert h.motions_begun == m0 + 1 and len(arm.commands) > n0, "the arm moved"
    assert result["ok"] is False and "did not settle at home" in result["error"], result
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED and pc["channel"] == "skill", pc
    assert pc["evidence"] == "the move home did not complete: SkillError: did not settle at home", pc
    _assert_no_event(runtime, result)


@pytest.mark.parametrize("tally", [_Tally(), None], ids=["counted", "unknown actuation"])
def test_a_failed_move_home_is_never_confirmed(tally):
    checker = PostconditionChecker(actuation=tally)
    before = checker.snapshot(None)
    if tally is not None:
        tally.counts["arm_motions"] += 1
    pc = checker.verify("move_home", {}, {"ok": False, "error": "MotionHalted: operator halt"}, before=before)
    assert pc.status == UNVERIFIED and pc.channel == "skill", pc
    assert pc.evidence == "the move home did not complete: MotionHalted: operator halt", pc


def _jaw_checker(tally, before_frac, holding):
    jaw = {"frac": before_frac}
    checker = PostconditionChecker(gripper_frac=lambda: jaw["frac"], actuation=tally,
                                   holding=None if holding is None else (lambda: holding))
    return checker, jaw


@pytest.mark.parametrize("before_frac", [0.0, 1.0], ids=["closed on air", "already open"])
def test_open_with_nothing_held_states_the_jaw_only(before_frac):
    """T3-c: nothing was held, so a completed open released nothing: the
    reading, never 'the object was released' (and an already-open jaw is not
    an unverified release either)."""
    tally = _Tally()
    checker, jaw = _jaw_checker(tally, before_frac, holding=False)
    before = checker.snapshot(None, skill="open_gripper")
    tally.counts["gripper_opens"] += 1
    jaw["frac"] = 1.0
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.channel == "gripper", pc
    assert pc.evidence == "gripper open at 1.00", pc


@pytest.mark.parametrize("rise, released", [(0.0, False), (0.01, False), (0.03, True)])
def test_a_release_is_a_measured_widening(rise, released):
    """T3-d (Qwen E2E o4): held at 0.389, the open completed, the jaw still
    read 0.389: not 'released'. Only a rise above the pre-call width is."""
    tally = _Tally()
    checker, jaw = _jaw_checker(tally, 0.389, holding=True)
    before = checker.snapshot(None, skill="open_gripper")
    tally.counts["gripper_opens"] += 1
    jaw["frac"] = 0.389 + rise
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    if released:
        assert pc.status == CONFIRMED and pc.evidence == "gripper open at 0.42: the object was released", pc
        return
    assert pc.status == UNVERIFIED and pc.channel == "gripper", pc
    assert pc.evidence == (f"jaw width unchanged at {0.389 + rise:.2f} after the open command "
                           "(was 0.39): no release measured"), pc
    assert pc.measured == {"gripper_frac": 0.389 + rise, "gripper_frac_before": 0.389}, pc


def test_a_release_with_unknown_holding_still_needs_the_widening():
    tally = _Tally()
    checker, jaw = _jaw_checker(tally, 0.5, holding=None)
    before = checker.snapshot(None, skill="open_gripper")
    tally.counts["gripper_opens"] += 1
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    assert pc.status == UNVERIFIED and pc.evidence.startswith("jaw width unchanged at 0.50"), pc
    jaw["frac"] = 0.97
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence.endswith("the object was released"), pc


def test_no_pre_call_width_never_claims_a_release():
    """A snapshot without the jaw width (a skill that is not a release, or
    no reading): the counted open is stated as the reading only."""
    tally = _Tally()
    checker, jaw = _jaw_checker(tally, 0.5, holding=True)
    before = checker.snapshot(None)
    tally.counts["gripper_opens"] += 1
    jaw["frac"] = 0.97
    pc = checker.verify("open_gripper", {}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.evidence == "gripper open at 0.97", pc


def test_only_release_skills_read_the_jaw_before_the_call():
    reads = []
    checker = PostconditionChecker(gripper_frac=lambda: reads.append(1) or 0.5, holding=lambda: True)
    assert "gripper_frac" not in checker.snapshot("cube", skill="grasp_object")
    assert "holding" not in checker.snapshot("cube", skill="grasp_object")
    assert reads == [], "a non-release skill's snapshot reads no jaw"
    snap = checker.snapshot(None, skill="open_gripper")
    assert snap["gripper_frac"] == 0.5 and snap["holding"] is True and reads == [1]


@pytest.mark.parametrize("close_first", [False, True], ids=["already open", "closed on air"])
def test_open_gripper_with_nothing_held_is_not_a_release(mock_runtime, close_first):
    runtime, arm = mock_runtime
    assert runtime.held_object is None
    if close_first:
        runtime.execute("close_gripper", {})
        assert runtime._gripper_width_frac() < 0.5, "the open below must widen the jaw"
        assert runtime.held_object is None

    result = runtime.execute("open_gripper", {})

    assert result["ok"] is True
    pc = result["postcondition"]
    assert pc["status"] == CONFIRMED and pc["evidence"] == "gripper open at 1.00", pc
    assert "released" not in _texts(runtime, result)


def test_open_gripper_whose_jaw_did_not_move_is_not_a_release(mock_runtime, monkeypatch):
    """The open command completed (counted) but the jaw still reads the width
    it held the object at: UNVERIFIED 'jaw width unchanged', never released."""
    runtime, arm = mock_runtime
    _hold(runtime, arm)
    sent = []
    monkeypatch.setattr(arm, "set_gripper", lambda pos, effort=1.0: sent.append(pos))

    result = runtime.execute("open_gripper", {})

    assert result["ok"] is True and len(sent) == 1, result
    assert runtime._gripper_width_frac() == pytest.approx(0.5)
    pc = result["postcondition"]
    assert pc["status"] == UNVERIFIED and pc["channel"] == "gripper", pc
    assert pc["evidence"] == "jaw width unchanged at 0.50 after the open command (was 0.50): no release measured", pc
    _assert_no_event(runtime, result)


def test_a_provisional_grasp_may_hold_for_the_verifier():
    """An interrupted grasp leaves a provisional hold: the jaws may hold the
    object, so a later measured widening may be its release."""
    from cascade.skills.runtime import SkillRuntime

    rt = SkillRuntime.__new__(SkillRuntime)
    rt.held_object, rt._held_provisional = None, None
    assert rt._jaws_may_hold() is False
    rt._held_provisional = object()
    assert rt._jaws_may_hold() is True
    rt.held_object, rt._held_provisional = "cup", None
    assert rt._jaws_may_hold() is True
