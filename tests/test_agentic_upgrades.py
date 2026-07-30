"""Tests for the 2026-07-31 agentic upgrades.

Covers the four new inference-time mechanisms:
- OperatingEnvelope   (Harness-VLA): learned per-primitive operating ranges
- MilestoneTracker    (Agentic-VLA): decomposition as a verified progress signal
- PostconditionChecker(Pigey):       independent effect verification
- Cosmos3EdgeClient:                 XML tool-call parsing
- aspire.diagnose/distil:            the trace -> skill-library loop
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from wrc_demo.agent.aspire import diagnose, distil, harvest, retrieve
from wrc_demo.agent.cosmos3 import parse_xml_tool_calls, strip_reasoning
from wrc_demo.agent.effects import (
    CONFIRMED,
    REFUTED,
    UNVERIFIED,
    PostconditionChecker,
    annotate_result,
)
from wrc_demo.agent.milestones import DONE, PENDING, UNKNOWN, MilestoneTracker
from wrc_demo.memory.envelope import OperatingEnvelope, normalize_failure
from wrc_demo.skills.library import SkillLibrary


# ── Harness-VLA: operating envelopes ─────────────────────────────────────


def test_envelope_is_silent_before_it_has_evidence():
    """A cold start must never warn: no data means no opinion."""
    env = OperatingEnvelope()
    assert env.check("place_at", {"x": 99.0, "y": -99.0}).ok
    assert env.agent_digest() == ""


def test_envelope_learns_success_range_and_flags_outliers():
    env = OperatingEnvelope()
    for x in (0.16, 0.17, 0.18, 0.165, 0.175):
        env.record("grasp_object", {"x": x}, ok=True, duration_ms=1000)

    assert env.check("grasp_object", {"x": 0.17}).ok
    verdict = env.check("grasp_object", {"x": 0.42})
    assert not verdict.ok
    assert verdict.outliers[0]["feature"] == "x"
    assert "0.42" in verdict.reason

    digest = env.agent_digest()
    assert "grasp_object" in digest and "x in [0.160, 0.180]" in digest


def test_envelope_slack_generalises_slightly_beyond_samples():
    env = OperatingEnvelope()
    for z in (0.10, 0.12, 0.14, 0.11):
        env.record("place_at", {"z": z}, ok=True)
    # just outside the raw span but inside the 15% slack band
    assert env.check("place_at", {"z": 0.145}).ok
    assert not env.check("place_at", {"z": 0.30}).ok


def test_failure_taxonomy_normalizes_real_harness_errors():
    assert normalize_failure(
        "SafetyViolation: link/joint 7 would hit the table"
    ) == "geometry:link_below_table"
    assert normalize_failure("SkillError: no IK solution for pose") == "kinematics:ik_unreachable"
    assert normalize_failure("air-grasp: closed on nothing") == "contact:air_grasp"
    assert normalize_failure("never seen 'banana'") == "perception:not_found"
    assert normalize_failure("") == "unknown"


def test_envelope_failure_digest_ranks_by_frequency():
    env = OperatingEnvelope()
    for _ in range(3):
        env.record("grasp_object", {}, ok=False, error="link/joint 7 would hit the table")
    env.record("place_at", {}, ok=False, error="no IK solution")
    digest = env.failure_digest()
    assert digest.index("grasp_object") < digest.index("place_at")
    assert "geometry:link_below_table x3" in digest


def test_envelope_persists_across_instances(tmp_path):
    path = tmp_path / "env.json"
    env = OperatingEnvelope(path=path)
    for x in (0.16, 0.17, 0.18, 0.165):
        env.record("grasp_object", {"x": x}, ok=True)
    reloaded = OperatingEnvelope(path=path)
    assert not reloaded.check("grasp_object", {"x": 0.9}).ok


def test_envelope_ingests_a_trace_file(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {"skill": "grasp_object", "args": {"x": x},
                 "result": {"ok": True}, "duration_ms": 900}
            )
            for x in (0.16, 0.17, 0.18, 0.175)
        )
    )
    env = OperatingEnvelope()
    assert env.ingest_trace(trace) == 4
    assert not env.check("grasp_object", {"x": 0.5}).ok


# ── Agentic-VLA: milestone verification ──────────────────────────────────


def _belief(label, position, extent=(0.1, 0.1, 0.1)):
    return SimpleNamespace(label=label, position=list(position), extent=list(extent))


class _Beliefs:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


def test_milestone_symbolic_containment_true_and_false():
    beliefs = _Beliefs([
        _belief("pink cube", (0.18, -0.17, 0.05), (0.05, 0.05, 0.05)),
        _belief("bin", (0.18, -0.17, 0.03), (0.15, 0.15, 0.06)),
    ])
    tracker = MilestoneTracker(beliefs=beliefs)
    verdict, evidence = tracker.check_symbolic("the pink cube is in the bin")
    assert verdict is True and "inside" in evidence

    beliefs._items[0] = _belief("pink cube", (0.40, 0.20, 0.05), (0.05, 0.05, 0.05))
    verdict, _ = tracker.check_symbolic("the pink cube is in the bin")
    assert verdict is False


def test_milestone_handles_imperative_phrasing():
    """Decomposition emits imperatives ('place the cube in the bin')."""
    beliefs = _Beliefs([
        _belief("cube", (0.18, -0.17, 0.05), (0.05, 0.05, 0.05)),
        _belief("bin", (0.18, -0.17, 0.03), (0.15, 0.15, 0.06)),
    ])
    tracker = MilestoneTracker(beliefs=beliefs)
    assert tracker.check_symbolic("place the cube in the bin")[0] is True


def test_milestone_holding_uses_gripper_state():
    tracker = MilestoneTracker(beliefs=_Beliefs([]), held_getter=lambda: "pink cube")
    assert tracker.check_symbolic("the pink cube is grasped")[0] is True
    tracker2 = MilestoneTracker(beliefs=_Beliefs([]), held_getter=lambda: None)
    assert tracker2.check_symbolic("the pink cube is grasped")[0] is False


def test_milestone_abstains_when_object_unknown():
    """Unverifiable must be UNKNOWN, never a guess."""
    tracker = MilestoneTracker(beliefs=_Beliefs([]))
    verdict, _ = tracker.check_symbolic("the banana is in the bowl")
    assert verdict is None

    tracker.reset(["the banana is in the bowl"])
    progress = tracker.update(None)
    assert progress.unknown == 1 and progress.done == 0
    assert tracker.milestones[0].status == UNKNOWN
    assert "[?]" in tracker.digest()
    assert tracker.unverified() == ["the banana is in the bowl"]


def test_milestone_progress_reports_newly_done():
    cube = _belief("cube", (0.40, 0.20, 0.05), (0.05, 0.05, 0.05))
    beliefs = _Beliefs([cube, _belief("bin", (0.18, -0.17, 0.03), (0.15, 0.15, 0.06))])
    tracker = MilestoneTracker(beliefs=beliefs)
    tracker.reset(["the cube is in the bin"])

    first = tracker.update(None)
    assert first.done == 0 and first.stalled

    cube.position = [0.18, -0.17, 0.05]
    second = tracker.update(None)
    assert second.done == 1 and second.newly_done == ["the cube is in the bin"]
    assert not second.stalled
    assert tracker.milestones[0].status == DONE


def test_milestone_visual_tier_is_rate_limited():
    calls = []

    def verifier(text, jpeg):
        calls.append(text)
        return None, "cannot tell"

    tracker = MilestoneTracker(
        beliefs=_Beliefs([]), vlm_verify=verifier, max_visual_checks=2
    )
    tracker.reset(["something unverifiable happened"])
    for _ in range(5):
        tracker.update(b"jpeg")
    assert len(calls) == 2  # capped


# ── Pigey: postcondition verification ────────────────────────────────────


def test_grasp_refuted_when_object_never_left_the_table():
    """The jaw jamming is NOT evidence of a grasp: the object must rise."""
    poses = {"pink cube": [0.17, 0.15, 0.04]}
    checker = PostconditionChecker(
        object_pose=lambda l: poses.get(l), gripper_frac=lambda: 0.5
    )
    before = checker.snapshot("pink cube")
    pc = checker.verify("grasp_object", {"label": "pink cube"}, {"ok": True}, before=before)
    assert pc.status == REFUTED
    assert "did not rise" in pc.evidence

    result = annotate_result({"ok": True, "held": "pink cube"}, pc)
    assert result["ok"] is False
    assert result["self_reported_ok"] is True


def test_grasp_confirmed_when_object_rises():
    poses = {"pink cube": [0.17, 0.15, 0.04]}
    checker = PostconditionChecker(
        object_pose=lambda l: poses.get(l), gripper_frac=lambda: 0.5
    )
    before = checker.snapshot("pink cube")
    poses["pink cube"] = [0.17, 0.15, 0.12]  # lifted 8 cm
    pc = checker.verify("grasp_object", {"label": "pink cube"}, {"ok": True}, before=before)
    assert pc.status == CONFIRMED and pc.channel == "physics"
    assert annotate_result({"ok": True}, pc)["verified"] is True


def test_air_grasp_refuted_from_gripper_alone():
    checker = PostconditionChecker(gripper_frac=lambda: 0.01, air_grasp_frac=0.04)
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": True})
    assert pc.status == REFUTED and pc.channel == "gripper"


def test_unverifiable_grasp_marks_result_unverified_not_failed():
    """No channel available: report honestly, do not invent a failure."""
    checker = PostconditionChecker(gripper_frac=lambda: 0.5)
    pc = checker.verify("grasp_object", {"label": "cube"}, {"ok": True})
    assert pc.status == UNVERIFIED
    result = annotate_result({"ok": True}, pc)
    assert result["ok"] is True and result["verified"] is False


def test_push_postcondition_measures_displacement():
    poses = {"box": [0.30, 0.0, 0.03]}
    checker = PostconditionChecker(object_pose=lambda l: poses.get(l))
    before = checker.snapshot("box")
    poses["box"] = [0.30, 0.002, 0.03]  # barely moved
    pc = checker.verify(
        "push_object", {"label": "box", "distance_m": 0.08}, {"ok": True}, before=before
    )
    assert pc.status == REFUTED and "barely moved" in pc.evidence


def test_verify_returns_none_for_unverifiable_skills():
    checker = PostconditionChecker()
    assert checker.verify("wave", {}, {"ok": True}) is None
    assert checker.verify("get_observation", {}, {"ok": True}) is None


def test_verification_does_not_reobserve_by_default():
    """Verification must be read-only w.r.t. the world model it judges."""
    calls = []
    checker = PostconditionChecker(
        gripper_frac=lambda: 0.5, reobserve=lambda: calls.append(1)
    )
    checker.verify("grasp_object", {"label": "cube"}, {"ok": True})
    assert calls == []


# ── Cosmos3-Edge tool-call parsing ───────────────────────────────────────


TOOLS = [
    {
        "name": "grasp_object",
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "force": {"type": "number"}},
        },
    },
    {
        "name": "move_relative",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string"},
                "distance_m": {"type": "number"},
            },
        },
    },
]


def test_cosmos3_parses_xml_tool_call_with_typed_args():
    text = """I will pick it up.
<tool_call>
<function=grasp_object>
<parameter=label>
pink cube
</parameter>
<parameter=force>
0.8
</parameter>
</function>
</tool_call>"""
    calls = parse_xml_tool_calls(text, TOOLS)
    assert len(calls) == 1
    assert calls[0].name == "grasp_object"
    assert calls[0].arguments["label"] == "pink cube"
    assert calls[0].arguments["force"] == pytest.approx(0.8)  # float, not "0.8"


def test_cosmos3_returns_nothing_for_plain_prose():
    assert parse_xml_tool_calls("The cube is on the left of the table.", TOOLS) == []


def test_cosmos3_handles_missing_closing_tags():
    """Truncated generations must degrade, not crash."""
    text = "<tool_call>\n<function=move_relative>\n<parameter=direction>\nup"
    calls = parse_xml_tool_calls(text, TOOLS)
    assert calls and calls[0].name == "move_relative"
    assert calls[0].arguments["direction"] == "up"


def test_cosmos3_strips_reasoning_block():
    assert strip_reasoning("<think>hmm, the cube</think>Done.") == "Done."


# ── ASPIRE: trace -> diagnosis -> skill ──────────────────────────────────


def _write_run(tmp_path, name, records, summary="task: pick the cube\nsuccess: true"):
    run = tmp_path / name
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (run / "summary.txt").write_text(summary)
    return run


def test_diagnose_localizes_failure_and_its_repair(tmp_path):
    run = _write_run(
        tmp_path,
        "run1",
        [
            {"skill": "grasp_object", "args": {"label": "cube", "depth_fraction": 0.5},
             "result": {"ok": False, "error": "SafetyViolation: link/joint 7 would hit the table"},
             "duration_ms": 500},
            {"skill": "grasp_object", "args": {"label": "cube", "depth_fraction": 0.15},
             "result": {"ok": True}, "duration_ms": 800},
        ],
    )
    diag = diagnose(run)
    assert diag.failed_skill == "grasp_object"
    assert diag.signature == "geometry:link_below_table"
    assert diag.repaired and diag.teachable


def test_unrepaired_failure_is_not_teachable(tmp_path):
    run = _write_run(
        tmp_path,
        "run2",
        [{"skill": "grasp_object", "args": {}, "result": {"ok": False, "error": "no IK solution"}}],
        summary="task: x\nsuccess: false",
    )
    diag = diagnose(run)
    assert diag.failed_skill == "grasp_object" and not diag.teachable


def test_distil_writes_a_retrievable_library_entry(tmp_path):
    run = _write_run(
        tmp_path,
        "run3",
        [
            {"skill": "grasp_object", "args": {"label": "cube", "depth_fraction": 0.5},
             "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
            {"skill": "grasp_object", "args": {"label": "cube", "depth_fraction": 0.15},
             "result": {"ok": True}},
        ],
    )
    library = SkillLibrary(tmp_path / "lib")
    path = distil(diagnose(run), library)
    assert path is not None and path.exists()
    body = path.read_text()
    assert "depth_fraction" in body and "0.5 -> 0.15" in body

    injected = retrieve(library, "grasp the cube")
    assert "ASPIRE library" in injected and "depth_fraction" in injected


def test_harvest_dedupes_by_skill_and_signature(tmp_path):
    records = [
        {"skill": "grasp_object", "args": {"depth_fraction": 0.5},
         "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
        {"skill": "grasp_object", "args": {"depth_fraction": 0.15}, "result": {"ok": True}},
    ]
    for i in range(3):
        _write_run(tmp_path, f"run{i}", records)
    library = SkillLibrary(tmp_path / "lib")
    out = harvest(tmp_path, library)
    assert out["diagnosed"] == 3
    assert out["learned"] == 1  # same (skill, signature) collapses to one entry


def test_retrieve_is_empty_on_a_fresh_library(tmp_path):
    assert retrieve(SkillLibrary(tmp_path / "lib"), "pick the cube") == ""
