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
from typing import Any

import pytest

from cascade.agent.aspire import Diagnosis, diagnose, distil, harvest, retrieve
from cascade.agent.cosmos3 import parse_xml_tool_calls, strip_reasoning
from cascade.agent.effects import (
    CONFIRMED,
    POSTCONDITIONS,
    REFUTED,
    UNVERIFIED,
    PostconditionChecker,
    annotate_result,
)
from cascade.agent.milestones import DONE, UNKNOWN, MilestoneTracker
from cascade.memory.envelope import OperatingEnvelope, normalize_failure
from cascade.skills.library import SkillLibrary


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


def test_place_on_object_uses_the_held_object_not_the_target():
    """Regression: the subject is what was RELEASED, `label` is the destination.

    Reading the subject from the pre-motion snapshot compared the target with
    itself and reported "box sits on box (offset 0.0 cm)" -- a vacuous
    confirmation that masked a real miss on the live rig (the cube landed at
    (0.272, 0.084), outside the bin, while this said confirmed).
    """
    poses = {"pink cube": [0.27, 0.08, 0.03], "box": [0.18, -0.17, 0.03]}
    checker = PostconditionChecker(object_pose=lambda l: poses.get(l))
    pc = checker.verify(
        "place_on_object",
        {"label": "box"},
        {"ok": True, "placed": "pink cube"},
        before={"label": "box", "pose": poses["box"]},   # the old, wrong snapshot
    )
    assert pc.status == REFUTED
    assert "pink cube" in pc.evidence and "not on it" in pc.evidence


def test_place_on_object_confirms_a_real_placement():
    poses = {"pink cube": [0.18, -0.17, 0.09], "box": [0.18, -0.17, 0.03]}
    checker = PostconditionChecker(object_pose=lambda l: poses.get(l))
    pc = checker.verify("place_on_object", {"label": "box"},
                        {"ok": True, "placed": "pink cube"}, before={})
    assert pc.status == CONFIRMED and "sits on box" in pc.evidence


def test_place_on_object_abstains_when_the_subject_is_unknown():
    """Better unverified than a self-comparison that always confirms."""
    checker = PostconditionChecker(object_pose=lambda l: [0.18, -0.17, 0.03])
    pc = checker.verify("place_on_object", {"label": "box"}, {"ok": True},
                        before={"label": "box"})
    assert pc.status == UNVERIFIED
    assert "which object was released" in pc.evidence


def test_self_reported_drop_point_needs_an_independent_channel():
    """A belief the skill wrote is not evidence the skill worked.

    Live rig 2026-07-31: a Spanish command ("cubo rosa") matched no sim prim,
    so the physics channel returned None and the check fell back to the belief
    `place` had just written -- reporting "0.0 cm from the requested drop
    point" while the cube sat at (0.361, 0.010), nowhere near the bin.
    """
    checker = PostconditionChecker(belief_pose=lambda l: [0.159, -0.179, 0.12])
    pc = checker.verify(
        "pick_and_place",
        {"object": "cubo rosa", "destination": "caja"},
        {"picked": "cubo rosa", "placed_at": [0.159, -0.179, 0.12], "ok": True},
        before={"label": "cubo rosa", "pose": [0.15, 0.12, 0.03]},
    )
    assert pc.status == UNVERIFIED
    assert "no independent confirmation" in pc.evidence


def test_caller_supplied_target_is_independent_evidence():
    """args["target"] came from the CALLER, so agreement does count."""
    checker = PostconditionChecker(belief_pose=lambda l: [0.16, -0.18, 0.12])
    pc = checker.verify(
        "place_at",
        {"target": [0.16, -0.18, 0.12]},
        {"object": "pink cube", "ok": True},
        before={"label": "pink cube", "pose": [0.15, 0.12, 0.03]},
    )
    assert pc.status == CONFIRMED


def test_physics_channel_confirms_without_the_independence_caveat():
    checker = PostconditionChecker(object_pose=lambda l: [0.16, -0.18, 0.04])
    pc = checker.verify(
        "pick_and_place",
        {"object": "pink cube", "destination": "box"},
        {"picked": "pink cube", "placed_at": [0.159, -0.179, 0.12], "ok": True},
        before={"label": "pink cube", "pose": [0.17, 0.15, 0.04]},
    )
    assert pc.status == CONFIRMED and pc.channel == "physics"


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


def _confirmed_result(skill="grasp_object"):
    return {
        "ok": True,
        "postcondition": {
            "skill": skill, "kind": POSTCONDITIONS[skill], "status": CONFIRMED,
            "channel": "physics", "evidence": "Synthetic fixture: object rose above the table",
            "measured": {"rise_m": 0.03},
        },
    }


def _write_run(tmp_path, name, records, summary="task: pick the cube\nsuccess: true", *, scoped=True):
    run = tmp_path / name
    run.mkdir(parents=True)
    if scoped:
        # Explicit fixture context, matching the production trace contract.
        records = [{**r, "context": r.get("context", {
            "arm": (r.get("args") or {}).get("arm", "default"),
            "held_object": "cube" if r.get("skill") in {"place_at", "place_on_object"} else None,
        })} for r in records]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (run / "summary.txt").write_text(summary)
    return run


def test_diagnose_localizes_failure_and_its_repair(tmp_path):
    run = _write_run(
        tmp_path,
        "run1",
        [
            {"skill": "grasp_object", "args": {"label": "cube", "material": "rigid"},
             "result": {"ok": False, "error": "SafetyViolation: link/joint 7 would hit the table"},
             "duration_ms": 500},
            {"skill": "grasp_object", "args": {"label": "cube", "material": "soft"},
             "result": _confirmed_result(), "duration_ms": 800},
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
            {"skill": "grasp_object", "args": {"label": "cube", "material": "rigid"},
             "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
            {"skill": "grasp_object", "args": {"label": "cube", "material": "soft"},
             "result": _confirmed_result()},
        ],
    )
    library = SkillLibrary(tmp_path / "lib")
    path = distil(diagnose(run), library)
    assert path is not None and path.exists()
    body = path.read_text()
    assert "material" in body and "'rigid' -> 'soft'" in body

    injected = retrieve(library, "grasp the cube")
    assert "ASPIRE library" in injected and "material" in injected


def test_harvest_dedupes_by_skill_and_signature(tmp_path):
    records = [
        {"skill": "grasp_object", "args": {"label": "cube", "material": "rigid"},
         "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
        {"skill": "grasp_object", "args": {"label": "cube", "material": "soft"}, "result": _confirmed_result()},
    ]
    for i in range(3):
        _write_run(tmp_path, f"run{i}", records)
    library = SkillLibrary(tmp_path / "lib")
    out = harvest(tmp_path, library)
    assert out["diagnosed"] == 3
    assert out["learned"] == 1  # same (skill, signature) collapses to one entry


def test_retrieve_is_empty_on_a_fresh_library(tmp_path):
    assert retrieve(SkillLibrary(tmp_path / "lib"), "pick the cube") == ""


@pytest.mark.parametrize("status", [None, UNVERIFIED, REFUTED])
def test_diagnose_does_not_learn_unconfirmed_retries(tmp_path, status):
    retry = _confirmed_result()
    if status is None:
        retry.pop("postcondition")
    else:
        retry["postcondition"]["status"] = status
    run = _write_run(tmp_path, "unconfirmed", [
        {"skill": "grasp_object", "args": {"object": "green cube"},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": "grasp_object", "args": {"object": "green cube"}, "result": retry},
    ])
    diag = diagnose(run)
    assert diag is not None
    assert not diag.repaired
    assert not diag.teachable
    library = SkillLibrary(tmp_path / "lib")
    assert distil(diag, library) is None
    assert library.entries() == []


@pytest.mark.parametrize("skill,field,before,after", [
    ("pick_and_place", "object", "green cube", "pink cube"),
    ("grasp_object", "label", "green cube", "pink cube"),
    ("grasp_object", "arm", "left", "right"),
    ("pick_and_place", "destination", "box", "green square"),
    ("grasp_object", "spatial_hint", "left", "right"),
    ("place_at", "x", 0.2, 0.3),
    ("push_object", "direction", "left", "right"),
    ("push_object", "distance_m", 0.02, 0.1),
])
def test_diagnose_does_not_credit_a_different_goal(tmp_path, skill, field, before, after):
    args = {
        "grasp_object": {"label": "cube"},
        "pick_and_place": {"object": "cube", "destination": "box"},
        "place_at": {"x": 0.2, "y": 0.1, "z": 0.1},
        "push_object": {"label": "cube", "direction": "left", "distance_m": 0.02},
    }[skill]
    run = _write_run(tmp_path, "changed-goal", [
        {"skill": skill, "args": {**args, field: before},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": skill, "args": {**args, field: after}, "result": _confirmed_result(skill)},
    ])
    diag = diagnose(run)
    assert diag is not None and not diag.teachable


def test_diagnose_does_not_credit_success_after_scene_reset(tmp_path):
    run = _write_run(tmp_path, "reset-boundary", [
        {"skill": "grasp_object", "args": {"label": "cube"},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": "reset_scene", "args": {}, "result": {"ok": True}},
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": _confirmed_result()},
    ])
    diag = diagnose(run)
    assert diag is not None and not diag.teachable


@pytest.mark.parametrize("field,value", [
    ("skill", "move_home"), ("kind", "at_home"), ("channel", ""),
    ("channel", "arm"), ("channel", "llm"), ("evidence", ""), ("measured", {}),
])
def test_diagnose_requires_matching_measured_confirmation(tmp_path, field, value):
    result = _confirmed_result()
    result["postcondition"][field] = value
    run = _write_run(tmp_path, "invalid-confirmation", [
        {"skill": "grasp_object", "args": {"label": "cube"},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": result},
    ])
    diag = diagnose(run)
    assert diag is not None and not diag.teachable


def test_distil_rechecks_confirmation_before_writing(tmp_path):
    diag = Diagnosis(run="unsupported", failed_skill="grasp_object",
                     signature="contact:air_grasp", repair_skill="grasp_object", repaired=True)
    library = SkillLibrary(tmp_path / "lib")
    assert not diag.teachable
    assert distil(diag, library) is None
    assert not library.entries()


def test_diagnosis_preserves_the_actual_repair_confirmation(tmp_path):
    result = _confirmed_result()
    run = _write_run(tmp_path, "receipt", [
        {"skill": "grasp_object", "args": {"label": "cube"},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": result},
    ])
    diag = diagnose(run)
    assert diag is not None and diag.teachable
    assert diag.as_dict().get("repair_postcondition") == result["postcondition"]


def test_distil_records_evidence_without_inventing_robot_advice(tmp_path):
    result = _confirmed_result()
    run = _write_run(tmp_path, "observed-only", [
        {"skill": "grasp_object", "args": {"label": "cube"},
         "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": result},
    ])
    diag = diagnose(run)
    assert diag is not None
    entry = distil(diag, SkillLibrary(tmp_path / "lib"))
    assert entry is not None
    text = entry.read_text()
    assert result["postcondition"]["evidence"] in text
    assert '"rise_m": 0.03' in text and "physics" in text
    assert "B601" not in text
    assert "re-observing" not in text  # no observation call occurred in this trace


@pytest.mark.parametrize("selectors,expected_arms", [
    ((None, "left"), ["left", "left"]),
    (("default", "primary"), ["left", "left"]),
    (("left", "right"), ["left", "right"]),
])
def test_dispatch_trace_preserves_resolved_arm_and_pre_call_subject(tmp_path, selectors, expected_arms):
    """Real dispatch/verifier/logger; only physical work is a data-only callback."""
    import threading

    from cascade.agent.trace import TraceLogger
    from cascade.control.arm_rig import ArmRig
    from cascade.skills.runtime import SkillRuntime

    rt: Any = SkillRuntime.__new__(SkillRuntime)
    left, right = SimpleNamespace(name="left"), SimpleNamespace(name="right")
    rt._arm = left
    rt._arm_override = threading.local()
    rt.arm_rig = ArmRig([left, right], ["left", "right"])
    rt.last_frame = rt.watcher = rt.held_object = rt._motion_t0 = None
    rt.current_tier = "test"
    rt.memory = SimpleNamespace(memory_frames=lambda _: [], add=lambda *a, **kw: None)
    rt.envelope = SimpleNamespace(record=lambda *a, **kw: None)
    rt.observe = lambda: None
    rt._show_status = lambda _: None
    rt.trace = TraceLogger(tmp_path / "producer")
    poses = {"cube": [0.2, 0.1, 0.03]}
    rt.effects = PostconditionChecker(object_pose=lambda label: poses[label], gripper_frac=lambda: 0.5)
    selected_arms = []

    def grasp(label):
        selected_arms.append(rt.arm.name)
        if len(selected_arms) == 1:
            return {"ok": False, "error": "air grasp"}
        poses[label] = [0.2, 0.1, 0.06]
        rt.held_object = label
        return {"ok": True, "held": label}

    rt.skill_grasp_object = grasp
    for selector in selectors:
        args = {"label": "cube"}
        if selector is not None:
            args["arm"] = selector
        rt.execute("grasp_object", args)
    rows = [json.loads(line) for line in (rt.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    assert selected_arms == expected_arms
    assert [r.get("context", {}).get("arm") for r in rows] == expected_arms
    assert all(r["args"] == {"label": "cube"} for r in rows)
    assert all(r["context"]["held_object"] is None for r in rows)
    assert rt.held_object == "cube" and rt.arm is left
    diag = diagnose(rt.trace.run_dir)
    assert diag is not None
    assert diag.teachable is (expected_arms[0] == expected_arms[1])


def test_legacy_trace_without_routing_context_is_not_teachable(tmp_path):
    run = _write_run(tmp_path, "unknown-context", [
        {"skill": "grasp_object", "args": {"label": "cube"},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": _confirmed_result()},
    ], scoped=False)
    diag = diagnose(run)
    assert diag is not None and not diag.teachable


@pytest.mark.parametrize("before_subject,after_subject,expected", [
    ("red cube", "blue cube", False), (None, None, False), ("red cube", "red cube", True),
])
def test_placement_retry_needs_the_same_recorded_subject(tmp_path, before_subject, after_subject, expected):
    coords = {"x": 0.2, "y": 0.1, "z": 0.05}
    label = after_subject or "blue cube"
    checker = PostconditionChecker(object_pose=lambda _: [0.2, 0.1, 0.05], gripper_frac=lambda: 1.0)
    result = {"ok": True, "placed": label, "at": [0.2, 0.1, 0.05]}
    result = annotate_result(result, checker.verify("place_at", coords, result,
        before={"label": label, "pose": [0.3, 0.1, 0.1], "channel": "physics"}))
    middle = ({"skill": "grasp_object", "args": {"label": "blue cube"}, "result": _confirmed_result()}
              if after_subject == "blue cube" else {"skill": "get_observation", "result": {"ok": True}})
    run = _write_run(tmp_path, "held-subject", [
        {"skill": "place_at", "args": coords, "result": {"ok": False, "error": "object slipped"},
         "context": {"arm": "default", "held_object": before_subject}},
        middle,
        {"skill": "place_at", "args": coords, "result": result,
         "context": {"arm": "default", "held_object": after_subject}},
    ])
    diag = diagnose(run)
    assert diag is not None and diag.teachable is expected


@pytest.mark.parametrize("success", [True, False])
def test_completed_task_is_a_learning_boundary(tmp_path, success):
    run = _write_run(tmp_path, "task-boundary", [
        {"skill": "grasp_object", "args": {"label": "cube"},
         "result": {"ok": False, "error": "air grasp"}},
        {"skill": "task_done", "args": {"success": success, "summary": "finished"},
         "result": {"ok": True, "task_complete": True, "success": success}},
        {"skill": "grasp_object", "args": {"label": "cube"}, "result": _confirmed_result()},
    ])
    diag = diagnose(run)
    assert diag is not None and not diag.teachable
