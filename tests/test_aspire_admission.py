"""Offline admission regressions: real logger/checker, data-only physical callbacks."""
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from cascade.agent.aspire import diagnose, distil
from cascade.agent.effects import PostconditionChecker, annotate_result
from cascade.agent.trace import TraceLogger
from cascade.control.arm_rig import ArmRig
from cascade.skills.library import SkillLibrary
from cascade.skills.runtime import SkillRuntime


def _retry_run(tmp_path, *, middle=(), verified=True):
    trace = TraceLogger(tmp_path / "run")
    args = {"label": "cube"}
    context = {"arm": "left", "held_object": None}
    trace.record("grasp_object", args, {"ok": False, "error": "air grasp"}, 1, context=context)
    for skill, result in middle:
        trace.record(skill, {}, result, 1, context=context)
    checker = PostconditionChecker(object_pose=lambda _: [0.2, 0.1, 0.06])
    result = {"ok": True, "held": "cube"}
    pc = checker.verify("grasp_object", args, result,
                        before={"label": "cube", "pose": [0.2, 0.1, 0.03], "channel": "physics"})
    result = annotate_result(result, pc)
    assert result["verified"] is True
    result["verified"] = verified
    trace.record("grasp_object", args, result, 1, context=context)
    return trace.run_dir


def test_explicit_contradictory_verified_flag_is_rejected(tmp_path):
    diag = diagnose(_retry_run(tmp_path, verified=False))
    assert diag is not None and not diag.teachable
    assert distil(diag, SkillLibrary(tmp_path / "lib")) is None


def test_task_complete_result_is_a_boundary_even_without_task_done_name(tmp_path):
    diag = diagnose(_retry_run(tmp_path, middle=[("host_terminal", {"ok": True, "task_complete": True})]))
    assert diag is not None and not diag.teachable


def test_distilled_note_retains_scope_and_does_not_claim_causality(tmp_path):
    diag = diagnose(_retry_run(tmp_path))
    assert diag is not None and diag.teachable
    assert "repaired by" not in diag.summary()
    path = distil(diag, SkillLibrary(tmp_path / "lib"))
    assert path is not None
    text = path.read_text()
    assert '"arm": "left"' in text and '"held_object": null' in text
    assert "not a proven causal repair" in text
    exported = diag.as_dict()
    exported["repair_context"]["arm"] = "right"
    assert diag.teachable  # as_dict must not expose mutable admission state
    diag.repair_context["arm"] = "right"
    assert distil(diag, SkillLibrary(tmp_path / "tampered")) is None


@pytest.fixture
def runtime(tmp_path):
    # Use real dispatch, serialization and verification, not a copy of execute().
    rt: Any = SkillRuntime.__new__(SkillRuntime)
    rt._arm = object()
    rt._arm_override = threading.local()
    rt.arm_rig = None
    rt.held_object = None
    rt.last_frame = rt.watcher = rt._motion_t0 = rt.effects = None
    rt.current_tier = "test"
    rt.memory = SimpleNamespace(memory_frames=lambda _: [], add=lambda *a, **kw: None)
    rt.envelope = SimpleNamespace(record=lambda *a, **kw: None)
    rt.observe = lambda: None
    rt._show_status = lambda _: None
    rt.trace = TraceLogger(tmp_path / "producer")
    return rt


@pytest.mark.parametrize("selectors,expected_arms", [
    ((None, "left"), ["left", "left"]),
    (("default", "primary"), ["left", "left"]),
    (("left", "right"), ["left", "right"]),
])
def test_dispatch_trace_preserves_resolved_arm_and_pre_call_subject(runtime, selectors, expected_arms):
    """Real dispatch/verifier/logger; only physical work is a data-only callback."""
    left, right = SimpleNamespace(name="left"), SimpleNamespace(name="right")
    runtime._arm = left
    runtime.arm_rig = ArmRig([left, right], ["left", "right"])
    poses = {"cube": [0.2, 0.1, 0.03]}
    runtime.effects = PostconditionChecker(object_pose=lambda label: poses[label], gripper_frac=lambda: 0.5)
    selected_arms = []

    def grasp(label):
        selected_arms.append(runtime.arm.name)
        if len(selected_arms) == 1:
            return {"ok": False, "error": "air grasp"}
        poses[label] = [0.2, 0.1, 0.06]
        runtime.held_object = label
        return {"ok": True, "held": label}

    runtime.skill_grasp_object = grasp
    for selector in selectors:
        args = {"label": "cube"}
        if selector is not None:
            args["arm"] = selector
        runtime.execute("grasp_object", args)
    rows = [json.loads(line) for line in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    assert selected_arms == expected_arms
    assert [r.get("context", {}).get("arm") for r in rows] == expected_arms
    assert all(r["args"] == {"label": "cube"} for r in rows)
    assert all(r["context"]["held_object"] is None for r in rows)
    assert runtime.held_object == "cube" and runtime.arm is left
    diag = diagnose(runtime.trace.run_dir)
    assert diag is not None
    assert diag.teachable is (expected_arms[0] == expected_arms[1])


@pytest.mark.parametrize("with_rig", [False, True])
def test_trace_routing_never_probes_arm_backend(runtime, with_rig):
    class NoProbeArm:
        def __getattr__(self, name):
            pytest.fail(f"logging probed backend attribute {name}")

    runtime._arm = NoProbeArm()
    if with_rig:
        runtime.arm_rig = ArmRig([runtime._arm], ["test-arm"])
    runtime.skill_list_objects = lambda: {"ok": True}
    assert runtime.execute("list_objects", {"arm": "primary"})["ok"] is True
    row = json.loads((runtime.trace.run_dir / "trace.jsonl").read_text())
    assert row["context"] == {"arm": "test-arm" if with_rig else "default", "held_object": None}


def test_trace_keeps_placement_subject_before_skill_clears_it(runtime):
    runtime.held_object = "red cube"
    runtime.effects = PostconditionChecker(object_pose=lambda _: [0.2, 0.1, 0.05], gripper_frac=lambda: 1.0)

    def place(x, y, z):
        subject = runtime.held_object
        runtime.held_object = None
        return {"ok": True, "placed": subject, "at": [x, y, z]}

    runtime.skill_place_at = place
    result = runtime.execute("place_at", {"x": 0.2, "y": 0.1, "z": 0.05})
    assert result["verified"] is True
    row = json.loads((runtime.trace.run_dir / "trace.jsonl").read_text())
    assert row["context"] == {"arm": "default", "held_object": "red cube"}
    assert runtime.held_object is None


@pytest.mark.parametrize("context", [None, {}, {"arm": "left"}, {"arm": None, "held_object": None}])
def test_missing_context_is_unknown_not_an_implicit_match(tmp_path, context):
    run = _retry_run(tmp_path)
    path = run / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["context"] = context
    path.write_text("\n".join(json.dumps(row) for row in rows))
    diag = diagnose(run)
    assert diag is not None and not diag.teachable
