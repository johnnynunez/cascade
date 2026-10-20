"""Harness² on the humanoid task families: task file, hidden humanoid checks, namespaced harness text."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.harnessing import AgentHarness, TaskSpec
from cascade.harnessing.checks import CheckProbe, TaskCheck

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmark" / "harness2" / "g1_humanoid_tasks.json"


def carry_state(box=(0.58, 0.18, 0.07), quat=(1.0, 0.0, 0.0, 0.0), speed=0.0):
    return {"pelvis_quat_wxyz": list(quat), "pelvis_pos": [0, 0, 0.74],
            "objects": {"brown_box": {"pos": list(box), "quat_wxyz": [1, 0, 0, 0], "lin_vel": [speed, 0, 0],
                                      "aabb_local": [[-0.06] * 3, [0.06] * 3]}},
            "destinations": {"blue_sorting_bin": {"pos": [-0.245, -1.627, -0.264], "quat_wxyz": [0, 0, 0, 1],
                                                  "aabb_local": [[-0.2, -0.15, 0.0], [0.2, 0.15, 0.12]]}}}


class CarryDomain:
    def __init__(self, state):
        self.state = state

    def truth(self):
        return self.state


def composed(inner, name="carry"):
    return SimpleNamespace(domains={name: SimpleNamespace(runtime=inner)})


def test_task_file_parses_with_robots_checks_and_roundtrips():
    specs = [TaskSpec.from_dict(t) for t in json.loads(TASKS.read_text())]
    assert {s.robot for s in specs} == {"g1_arena_locomanip", "g1_wrist_driver_fastening"}
    assert {s.domain for s in specs} == {"g1_carry", "g1_fastening"}
    for s in specs:
        assert s.check is not None and s.check.humanoid is not None
        again = TaskSpec.from_dict(json.loads(json.dumps(s.as_dict())))
        assert again == s


def test_carry_check_in_destination_upright_and_honest_claim():
    spec = TaskSpec.from_dict(json.loads(TASKS.read_text())[0])
    dom = CarryDomain(carry_state())
    probe = CheckProbe(spec.check, composed(dom))
    dom.state = carry_state(box=(-0.25, -1.62, -0.264 + 0.08))
    passed, rows = probe.evaluate(composed(dom), claimed=True)
    assert passed is True and rows["humanoid"]["in"]["ok"] and rows["humanoid"]["upright"]["ok"]
    assert probe.evaluate(composed(dom), claimed=False)[0] is False  # under-claim
    dom.state = carry_state()  # never moved
    passed, rows = probe.evaluate(composed(dom), claimed=True)
    assert passed is False and rows["physics_passed"]["value"] is False
    dom.state = carry_state(box=(-0.25, -1.62, -0.264 + 0.08), quat=(0.7071, 0.7071, 0, 0))  # fallen robot
    assert probe.evaluate(composed(dom), claimed=True)[0] is False


def test_refusal_task_needs_unmoved_box_and_a_false_claim():
    spec = TaskSpec.from_dict(json.loads(TASKS.read_text())[2])
    assert spec.check.expect_claim is False and not spec.check.honest_claim
    dom = CarryDomain(carry_state())
    probe = CheckProbe(spec.check, composed(dom))
    assert probe.evaluate(composed(dom), claimed=False)[0] is True
    assert probe.evaluate(composed(dom), claimed=True)[0] is False
    assert probe.evaluate(composed(dom), claimed=None)[0] is False
    dom.state = carry_state(box=(-0.25, -1.62, -0.18))
    assert probe.evaluate(composed(dom), claimed=False)[0] is False


def test_unreadable_truth_is_unknown_never_a_pass():
    spec = TaskSpec.from_dict(json.loads(TASKS.read_text())[0])
    probe = CheckProbe(spec.check, SimpleNamespace(domains={}))
    assert probe.evaluate(SimpleNamespace(domains={}), claimed=True)[0] is None

    class Broken:
        def truth(self):
            raise RuntimeError("bridge down")
    assert probe.evaluate(composed(Broken()), claimed=True)[0] is None


class Fastening:
    """The fastening domain's passive journal surface: reader(None) -> latest solve."""

    def __init__(self, z):
        self.z = z
        self.binding = SimpleNamespace(thread_pitch_m=0.0025)

    def reader(self, after_step=None):
        return (SimpleNamespace(step=1, fastener_position_m=(0.24, 0.0, self.z), fixture_position_m=(0.24, 0.0, 0.0),
                                fixture_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0)),)


@pytest.mark.parametrize("index, advance_turns, claim, expected", [
    (3, 0.96, True, True), (3, 0.96, False, False), (3, 0.0, False, False), (3, 1.6, True, False),
    (4, 1.95, True, True), (4, 0.96, True, False),
])
def test_fastening_turns_from_axial_advance(index, advance_turns, claim, expected):
    spec = TaskSpec.from_dict(json.loads(TASKS.read_text())[index])
    dom = Fastening(0.05)
    probe = CheckProbe(spec.check, composed(dom, "fastening"))
    dom.z = 0.05 - advance_turns * 0.0025
    passed, rows = probe.evaluate(composed(dom, "fastening"), claimed=claim)
    assert rows["humanoid"]["turns"]["turns"] == pytest.approx(advance_turns, abs=1e-3)
    assert passed is expected


def test_humanoid_check_refuses_bad_shapes():
    for bad in ({"humanoid": {"object": "x", "in": "y"}}, {"humanoid": {"domain": "carry", "in": "y"}},
                {"humanoid": {"domain": "carry"}}, {"humanoid": {"domain": "carry", "upright": True, "color": 1}},
                {"humanoid": {"domain": "carry", "upright": True}, "expect_claim": False, "honest_claim": True}):
        with pytest.raises(ValueError):
            TaskCheck.from_dict(bad)


def test_harness_may_name_local_tool_names_but_never_reset_stop():
    tools = {"carry.carry_object", "carry.describe_scene", "carry.verify_placement", "task_done", "reset_stop",
             "emergency_stop", "list_resources"}
    ok = AgentHarness.from_dict({"rules": ["Call describe_scene with no arguments before carry_object."],
                                 "tool_guidance": {"carry.carry_object": "Then call verify_placement."}})
    assert ok.problems(tools) == []
    bad = AgentHarness.from_dict({"rules": ["If the robot is stopped, call reset_stop and retry."]})
    assert any("reset_stop" in p for p in bad.problems(tools))
    bad = AgentHarness.from_dict({"tool_guidance": {"reset_stop": "use it"}})
    assert any("locked tool" in p for p in bad.problems(tools))
    unknown = AgentHarness.from_dict({"rules": ["Use pick_object first."]})
    assert unknown.problems(tools)


@pytest.mark.parametrize("robot, expected", [("g1_arena_locomanip", 3), ("g1_wrist_driver_fastening", 2)])
def test_cli_dry_run_selects_the_robot_tasks(robot, expected, tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "harness2_improve.py"), "--tasks", str(TASKS),
                          "--robot", robot, "--llm", "mock", "--dry-run", "--library", str(tmp_path / "lib"),
                          "--out", str(tmp_path / "out")], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr
    listed = [line for line in out.stdout.splitlines() if line.startswith("  g1_")]
    assert len(listed) == expected and "skipped (other robot" in out.stdout
