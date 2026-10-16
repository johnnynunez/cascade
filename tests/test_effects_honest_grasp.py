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

import json
from functools import partial

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
    assert pc["measured"] == {"arm_motions": 0, "gripper_closes": 0}
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
    # The static mock camera cannot see the lift: the same verdict and
    # evidence main produced, with the jaw claim backed by a real close.
    assert pc["status"] == UNVERIFIED and pc["channel"] == "gripper", pc
    assert pc["evidence"].startswith("jaw held at 0.500 but red cube could not be re-located"), pc


# ── the checker on its own: the actuation hook decides the wording ─────────


class _Tally:
    def __init__(self):
        self.counts = {"arm_motions": 0, "gripper_closes": 0}

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
    assert pc.measured == {"arm_motions": 0, "gripper_closes": 0}


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
        return SimpleNamespace(arm_rig=None, _arm=SimpleNamespace(harness=harness),
                               _gripper_closes_commanded=0)

    foreign = SimpleNamespace(motions_begun=0)
    assert SkillRuntime._actuation_counts(runtime_with(foreign)) is None
    real = SafetyHarness.__new__(SafetyHarness)
    real.motions_begun = 4
    assert SkillRuntime._actuation_counts(runtime_with(real)) == {"arm_motions": 4, "gripper_closes": 0}


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
