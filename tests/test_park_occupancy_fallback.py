"""A shutdown park refused ONLY by the occupancy map still parks.

Found on the physical reBot: Ctrl+C -> "park skipped (SafetyViolation: point 0
clearance 0.029 m below 0.030 m (occupancy map) ...)" -> teardown released
torque with the arm raised. The obstacle was most likely the arm's own unmasked
parts or the object still in the jaws; the drop was the worse outcome.
"""

from types import SimpleNamespace

import pytest

from cascade.apps.demo import _park_move
from cascade.types import SafetyViolation

MAP_REFUSAL = "point 0 clearance 0.029 m below 0.030 m (occupancy map) at [0.3177, 0.0001, 0.2791]"


def fake_arm(*, refuse_with_map=MAP_REFUSAL, refuse_without_map=None, estopped=False):
    harness = SimpleNamespace(occupancy=object(), estopped=estopped, park_q=[0.0] * 6)
    moves = []

    def move_joints(q, duration_s, joint_margin=None):
        moves.append({"map": harness.occupancy is not None, "joint_margin": joint_margin})
        if harness.occupancy is not None and refuse_with_map:
            raise SafetyViolation(refuse_with_map)
        if harness.occupancy is None and refuse_without_map:
            raise SafetyViolation(refuse_without_map)
        return True

    return SimpleNamespace(harness=harness, n_joints=6, move_joints=move_joints), moves


def test_map_refusal_parks_without_the_map_and_restores_it():
    arm, moves = fake_arm()
    occupancy, stage = arm.harness.occupancy, {}
    assert _park_move(arm, 2.0, stage) is True
    assert moves == [{"map": True, "joint_margin": 0.0}, {"map": False, "joint_margin": 0.0}]
    assert arm.harness.occupancy is occupancy
    assert "occupancy map" in stage["occupancy_bypassed"]


def test_a_park_the_map_allows_is_unchanged():
    arm, moves = fake_arm(refuse_with_map=None)
    stage = {}
    assert _park_move(arm, 2.0, stage) is True
    assert moves == [{"map": True, "joint_margin": 0.0}]
    assert "occupancy_bypassed" not in stage


@pytest.mark.parametrize("reason", ["TCP below table clearance", "joint 3 out of limits"])
def test_other_refusals_are_not_bypassed(reason):
    arm, moves = fake_arm(refuse_with_map=reason)
    with pytest.raises(SafetyViolation, match=reason):
        _park_move(arm, 2.0, {})
    assert len(moves) == 1


def test_an_estop_latch_is_never_bypassed():
    arm, moves = fake_arm(estopped=True)
    with pytest.raises(SafetyViolation, match="occupancy map"):
        _park_move(arm, 2.0, {})
    assert len(moves) == 1


def test_map_is_restored_even_when_the_retry_is_refused():
    arm, _ = fake_arm(refuse_without_map="TCP below table clearance")
    occupancy = arm.harness.occupancy
    with pytest.raises(SafetyViolation, match="table"):
        _park_move(arm, 2.0, {})
    assert arm.harness.occupancy is occupancy
