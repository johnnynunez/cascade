"""Fastening review: commanded wrist motion is not a measured task outcome.

These CPU tests exercise the real dispatch/trace/memory/reflex paths with a
synthetic actuator result, without importing a solver or claiming contact.
The existing test_turn_screw stream test covers the actual wrist routine.
"""
from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

from cascade.agent.effects import (
    Postcondition,
    PostconditionChecker,
    UNVERIFIED,
    annotate_result,
)
from cascade.agent.llm import MockLLM
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.agent.reflex import ExperienceMemory, FastPlanner
from cascade.agent.trace import TraceLogger
from cascade.config import load_demo_config
from cascade.memory import BeliefStore, EpisodicMemory
from cascade.skills.runtime import SkillRuntime
from cascade.types import SafetyViolation, SkillError


@pytest.fixture
def runtime(tmp_path):
    # Keep the real dispatcher and stores; this fixture has no actuation.
    rt = SkillRuntime(
        camera=None, depth_provider=None, detector=None, extrinsics=None,
        kin=None, safe_arm=SimpleNamespace(), memory=EpisodicMemory(),
        beliefs=BeliefStore(), trace=TraceLogger(tmp_path / "run"),
        cfg=load_demo_config(camera="mock", arm="mock", llm="mock"),
    )
    rt.observe = lambda: None
    rt.effects = PostconditionChecker()
    return rt


def completed_strokes(**_args):
    return {
        "turns_requested": 1.0, "turns_commanded": 1.0,
        "turns_applied": 1.0, "strokes": 4,
        "physical_verification": {
            "status": "unverified", "fastener_turns": None,
            "axial_advance_m": None, "seating_verified": False,
        },
    }


@pytest.mark.parametrize("channel", ["none", "static_xyz", "matching_axial_xyz"])
@pytest.mark.parametrize("claimed_seated", [False, True])
def test_threading_needs_more_than_actor_claim_or_xyz(channel, claimed_seated):
    """Even a pitch-sized XYZ delta cannot prove rotation/contact/seating."""
    pose = [0.24, 0.0, 0.069]
    checker = PostconditionChecker(
        object_pose=None if channel == "none" else lambda _label: list(pose),
        gripper_frac=lambda: 0.5,
    )
    before = checker.snapshot("nut")
    if channel == "matching_axial_xyz":
        pose[2] -= 0.0025
    result = {"ok": True, **completed_strokes()}
    if claimed_seated:
        result["verified"] = True
        result["postcondition"] = {"status": "confirmed", "kind": "seated"}
        result["physical_verification"] = {
            "status": "confirmed", "fastener_turns": 1.0,
            "axial_advance_m": 0.0025, "seating_verified": True,
        }
    pc = checker.verify("turn_screw", {"label": "nut", "turns": 1.0}, result, before)
    assert pc is not None and pc.status == UNVERIFIED
    assert pc.channel == "" and pc.measured == {}
    outcome = annotate_result(result, pc)
    assert outcome["ok"] is False and outcome["execution_ok"] is True
    assert outcome["verified"] is False
    assert "threading or seating" in outcome["verification_note"]


@pytest.mark.parametrize("checker_mode", ["normal", "absent", "no_verdict", "crashed"])
def test_completed_wrist_dispatch_is_unverified_in_trace_memory_and_envelope(
    runtime, checker_mode,
):
    runtime.skill_turn_screw = completed_strokes
    if checker_mode == "absent":
        runtime.effects = None
    elif checker_mode == "no_verdict":
        runtime.effects.verify = lambda *_args, **_kwargs: None
    elif checker_mode == "crashed":
        def fail(*_args, **_kwargs):
            raise RuntimeError("fastener reader unavailable")
        runtime.effects.verify = fail
    result = runtime.execute("turn_screw", {"label": "nut", "turns": 1.0})
    assert result["execution_ok"] is True and result["ok"] is False
    assert result["verified"] is False
    assert result["postcondition"]["status"] == UNVERIFIED
    assert result["turns_commanded"] == result["turns_applied"] == 1.0
    assert result["physical_verification"]["seating_verified"] is False
    row = json.loads(runtime.trace._trace_path.read_text().splitlines()[-1])
    assert row["result"] == result
    event = runtime.memory.events()[-1]
    assert event.kind == "outcome" and event.data["verdict"] == UNVERIFIED
    assert "-> ok" not in event.text and "fastening unverified" in event.text
    stats = runtime.envelope._skills["turn_screw"]
    assert stats.wins == 0 and stats.losses == 1 and stats.spans == {}


@pytest.mark.parametrize("error", [
    SkillError("only 0.20 of 1.00 requested wrist turns completed"),
    SafetyViolation("engagement pose outside envelope"),
    RuntimeError("joint transport disconnected"),
])
def test_failed_wrist_execution_keeps_error_without_physical_credit(runtime, error):
    def fail(**_args):
        raise error
    runtime.skill_turn_screw = fail
    result = runtime.execute("turn_screw", {"label": "nut", "turns": 1.0})
    assert result["ok"] is False and result["execution_ok"] is False
    assert str(error) in result["error"]
    assert result["verified"] is False
    assert result["postcondition"]["status"] == UNVERIFIED
    assert runtime.envelope._skills["turn_screw"].wins == 0


def test_fast_reflex_cannot_learn_or_report_completed_strokes_as_fastening(runtime):
    runtime.skill_turn_screw = completed_strokes
    experience = ExperienceMemory()
    planner = FastPlanner(experience=experience)
    notes = []
    note_outcome = planner.note_outcome

    def record(*args):
        notes.append(copy.deepcopy(args))
        note_outcome(*args)
    planner.note_outcome = record
    agent = AgentOrchestrator(
        MockLLM([]), runtime, fast_planner=planner, decompose=False,
        attach_images=False, verify_milestones=False,
    )
    report, note = agent._try_fast_path("tighten the nut one turn", time.monotonic())
    assert report is None and "FAILED" in note and "fastening unverified" in note
    assert len(notes) == 1 and notes[0][2] is False
    assert len(experience) == 0
    assert not (runtime.trace.run_dir / "summary.txt").exists()
    assert runtime.memory.events()[-1].data["verdict"] == UNVERIFIED


def test_other_unverified_skills_keep_their_existing_execution_convention():
    result = annotate_result(
        {"ok": True},
        Postcondition("move_home", "at_home", UNVERIFIED, "no observation"),
    )
    assert result["ok"] is True and result["verified"] is False
    assert "execution_ok" not in result
