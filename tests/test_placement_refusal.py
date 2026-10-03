"""A refused release cannot become verified placement from nearby XYZ alone."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_placement_reporting import runtime  # noqa: F401

from cascade.agent.effects import Postcondition, PostconditionChecker, annotate_result


def recorded_refusal():
    return json.loads((Path(__file__).parent / "fixtures/refused-placement-20261003.json").read_text())


def test_recorded_held_blue_cube_near_target_is_not_verified_placement():
    row = recorded_refusal()
    original = deepcopy(row["result"])
    final = original["postcondition"]["measured"]["final"]
    checker = PostconditionChecker(object_pose=lambda label: final)
    pc = checker.verify(row["skill"], row["args"], original)
    result = annotate_result(original, pc)
    assert pc.status == "unverified"
    assert pc.channel == "physics" and pc.measured["target_err_m"] < .03
    assert result["verified"] is False and result["ok"] is False
    assert result["holding"] == "blue cube" and result["home_skipped"] is True
    assert result["error"] == row["result"]["error"]


@pytest.mark.parametrize("skill,args,before", [
    ("pick_and_place", {"object": "cube", "target": [.2, -.12]}, {}),
    ("place_at", {"x": .2, "y": -.12}, {"label": "cube"}),
    ("place_on_object", {"label": "plate"}, {"label": "cube"}),
])
@pytest.mark.parametrize("refusal", [{"ok": False}, {"ok": True, "holding": "cube"}])
def test_placement_refusal_vetoes_positive_position_only_verdict(skill, args, before, refusal):
    poses = {"cube": [.2, -.12, .04], "plate": [.2, -.12, .02]}
    checker = PostconditionChecker(object_pose=poses.get)
    pc = checker.verify(skill, args, {"object": "cube", **refusal}, before)
    assert pc.status == "unverified" and pc.channel == "physics"
    assert "release" in pc.evidence


def test_failure_preserves_independent_negative_geometry():
    checker = PostconditionChecker(object_pose=lambda _: [.4, .3, .02])
    pc = checker.verify("pick_and_place", {"object": "cube", "target": [.2, -.12]},
                        {"ok": False, "holding": "cube"})
    assert pc.status == "refuted" and pc.measured["target_err_m"] > .1


def test_independent_lost_object_stays_refuted_after_refusal():
    class Reader:
        def __call__(self, label): return None
        def lost(self, label): return [.2, -.12, -198.]
    pc = PostconditionChecker(object_pose=Reader()).verify(
        "pick_and_place", {"object": "cube"}, {"ok": False, "holding": "cube"})
    assert pc.status == "refuted" and pc.measured["outside_scene"] is True


def test_separate_failed_home_does_not_erase_successful_placement_observation():
    checker = PostconditionChecker(object_pose=lambda _: [.2, -.12, .02])
    pc = checker.verify("pick_and_place", {"object": "cube", "target": [.2, -.12]},
                        {"ok": True, "return_home": {"ok": False}})
    assert pc.status == "confirmed"


def test_refusal_cannot_escape_through_visual_confirmation():
    from types import SimpleNamespace
    checker = PostconditionChecker(belief_pose=lambda _: [.2, -.12, .02],
        visual_diff=lambda *args: SimpleNamespace(status="changed", detail="fixture", measured={}))
    pc = checker.verify("pick_and_place", {"object": "cube"},
                        {"ok": False, "placed_at": [.2, -.12, .02]})
    assert pc.status == "unverified" and pc.channel == "visual_diff"


@pytest.mark.parametrize("status", ["unverified", "refuted"])
def test_failed_actor_cannot_preserve_stale_verified_flag(status):
    pc = Postcondition(skill="pick_and_place", kind="relocated", status=status)
    result = annotate_result({"ok": False, "verified": True, "error": "release refused"}, pc)
    assert result["verified"] is False and result["ok"] is False
    assert result["error"] == "release refused"


def test_runtime_trace_keeps_failed_release_unverified(request):
    rt = request.getfixturevalue("runtime")
    row = recorded_refusal()
    motion = {k: v for k, v in row["result"].items()
              if k not in ("postcondition", "verified", "next_action")}
    calls = []
    def pose(label):
        calls.append(label)
        return [.3, .15, .02] if len(calls) == 1 else [.218, -.1021, .0257]
    rt.attach_verifier(object_pose=pose)
    rt.skill_pick_and_place = lambda **_: deepcopy(motion)
    result = rt.execute("pick_and_place", row["args"])
    assert result["ok"] is False and result["verified"] is False
    assert result["postcondition"]["status"] == "unverified"
    assert result["holding"] == motion["holding"] and result["error"] == motion["error"]
    rows = [json.loads(line) for line in (rt.trace.run_dir / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["result"] == result
    assert "Do not start another pick" in result["next_action"]
