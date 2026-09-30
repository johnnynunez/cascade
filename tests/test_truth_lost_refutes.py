"""A body physics measures OUTSIDE the scene must refute the claimed effect.

Measured on the Newton kitchen (2026-09-29, fixed Isaac stepper, round sfb):
`pick_and_place orange -> open box` returned ok=True while the physics channel
read the orange at [0.852, -2.010, -198.278] -- through the counter and still
falling. `TruthPoseReader` rejects any |coordinate| > 5 m (correctly: it is not
a resting position), so the checker fell back to the belief the skill itself
had written at the box centre and returned UNVERIFIED, which `annotate_result`
leaves as ok=True. The same happened to a green cube that ended at z = -31.8 m.

These tests use only the public API that existed before the fix, so they run
unchanged against the pre-fix modules (red) and the fixed ones (green).
"""

from __future__ import annotations

import json

import pytest

from cascade.agent.effects import REFUTED, PostconditionChecker, annotate_result
from cascade.sim.truth import LazyTruthPoseFn, TruthPoseReader

ORANGE_START = [0.1773, 0.1131, 0.0253]
ORANGE_FELL = [0.8517, -2.0103, -198.2781]   # the measured sfb reading
BOX = [0.30, -0.14]


class FakeClient:
    def __init__(self, poses: dict):
        self.poses = poses

    def request(self, req):
        return {"ok": True, "stdout": "CASCADE_TRUTH_POSES " + json.dumps(self.poses)}


def _pick_and_place(reader, *, belief_after):
    checker = PostconditionChecker(object_pose=reader, belief_pose=lambda label: belief_after)
    before = checker.snapshot("orange")
    reader._client.poses = {"orange": ORANGE_FELL, "green_cube": [0.24, 0.01, 0.03]}
    result = {"ok": True, "object": "orange", "placed_at": BOX, "destination": "open box",
              "destination_kind": "configured_point"}
    pc = checker.verify("pick_and_place", {"object": "orange", "destination": "open box"}, result, before)
    return pc, annotate_result(result, pc)


def test_orange_through_the_counter_is_refuted_not_reported_ok():
    reader = TruthPoseReader(FakeClient({"orange": ORANGE_START}), ttl_s=0.0)
    pc, result = _pick_and_place(reader, belief_after=[0.30, -0.14, 0.104])
    assert pc.status == REFUTED, f"{pc.status}: {pc.evidence}"
    assert pc.channel == "physics"
    assert result["ok"] is False
    assert "outside the scene" in result["error"]


def test_the_rejected_reading_still_never_becomes_a_position():
    """The 5 m gate stays: a lost body is refuted, never scored as a pose."""
    reader = TruthPoseReader(FakeClient({"orange": ORANGE_FELL}), ttl_s=0.0)
    assert reader.pose("orange") is None
    assert reader.rejected == 1


@pytest.mark.parametrize("reading", [
    [0.0, 0.0, -31.79],                  # green cube measured under the counter (m13a)
    [float("nan"), 0.0, 0.0],            # solver has no finite position for it
    [-11.8081, -10.6423, -122.1323],     # the ablation-run explosion
])
def test_every_kind_of_lost_reading_refutes_a_grasp(reading):
    reader = TruthPoseReader(FakeClient({"green_cube": [0.24, 0.01, 0.03]}), ttl_s=0.0)
    checker = PostconditionChecker(object_pose=reader, belief_pose=lambda label: [0.24, 0.01, 0.10],
                                   gripper_frac=lambda: 0.5)
    before = checker.snapshot("green cube")
    reader._client.poses = {"green_cube": reading}
    pc = checker.verify("grasp_object", {"label": "green cube"}, {"ok": True, "object": "green cube"}, before)
    assert pc.status == REFUTED, f"{pc.status}: {pc.evidence}"


def test_an_ambiguous_label_does_not_blame_a_lost_body():
    """Identity rules are the same as for sane poses: two red cans, one lost."""
    reader = TruthPoseReader(FakeClient({"red_can_left": [0.2, 0.1, 0.04],
                                         "red_can_right": [0.0, 0.0, -50.0]}), ttl_s=0.0)
    checker = PostconditionChecker(object_pose=reader)
    pc = checker.verify("pick_and_place", {"object": "red can"}, {"ok": True, "object": "red can"}, {})
    assert pc.status != REFUTED


def test_malformed_reading_is_not_evidence_either_way():
    reader = TruthPoseReader(FakeClient({"orange": [1.0, 2.0]}), ttl_s=0.0)
    checker = PostconditionChecker(object_pose=reader, belief_pose=lambda label: [0.3, -0.14, 0.1])
    pc = checker.verify("pick_and_place", {"object": "orange"}, {"ok": True, "object": "orange",
                                                                 "placed_at": BOX}, {})
    assert pc.status != REFUTED


def test_lazy_channel_forwards_the_lost_reading():
    reader = TruthPoseReader(FakeClient({"orange": ORANGE_FELL}), ttl_s=0.0)
    lazy = LazyTruthPoseFn(arm=None)
    lazy._reader = reader
    checker = PostconditionChecker(object_pose=lazy, belief_pose=lambda label: [0.3, -0.14, 0.1])
    pc = checker.verify("pick_and_place", {"object": "orange"},
                        {"ok": True, "object": "orange", "placed_at": BOX}, {})
    assert pc.status == REFUTED, f"{pc.status}: {pc.evidence}"
