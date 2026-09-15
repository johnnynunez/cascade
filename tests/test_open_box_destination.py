"""Calibrated box planning on real CPU kinematics; all actuators are recorders.

These checks do not claim a physical grasp or a collision-free runtime path.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import make_transform
from test_post_place_retreat import runtime, finger_vertices, finger_minimum_z, HOME_Q

ROOT = Path(__file__).resolve().parents[1]


def box_runtime():
    cfg = load_demo_config(camera="isaac", arm="isaac_kitchen_gpu", llm="mock")
    rt, moves, opens, cleared = runtime()
    grasp = cfg.grasp.as_dict()
    grasp["close_settle_s"] = 0  # no wall-clock delay in the motion recorder
    rt.cfg = SimpleNamespace(arm=cfg.arm.as_dict(), safety=cfg.safety.as_dict(), grasp=grasp)
    q = rt.kin.ik(make_transform(_yaw_rotation(-.165), np.array([.18, -.03, .105])), HOME_Q)
    assert q.success
    rt.arm.get_state().q = q.q.copy()
    rt.arm.harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=rt.kin)
    rt.held_object = rt._held_det_label = "green cube"
    rt._held_offset = np.array([0., 0., -.022])
    rt._destination_fix = lambda *args: pytest.fail("box route consulted the identity grouping origin or a detected rim")
    rt.skill_move_home = lambda: rt.arm.move_joints(HOME_Q)
    return rt, moves, opens, q.q.copy()


@pytest.mark.parametrize("name", ["open box", "the open box", "open_box", "beige box", "the beige box", "box", "the box", "storage box", "the storage box", "wooden box", "the wooden box"])
def test_calibrated_box_has_safe_real_ik_and_preserves_result_provenance(name):
    rt, moves, opens, initial = box_runtime()
    result = rt.skill_pick_and_place("green cube", destination=name)
    assert result["destination"] == "open box"
    assert result["destination_kind"] == "configured_point"
    assert result["placed_at"] == [.30, -.14, .104]
    assert result["post_place_retreat"] == {"ok": True, "tcp_at": [.30, -.14, .144]}
    assert len(opens) == 1 and rt.held_object is None
    assert len(moves) == 5  # vertical lift, carry, release, retreat, home
    poses = [rt.kin.fk(q) for q in moves]
    assert max(T[2, 3] for T in poses[:3]) < .1451
    assert poses[2][2, 3] == pytest.approx(.104, abs=.0001)
    np.testing.assert_allclose(poses[2][:3, :3], poses[3][:3, :3], atol=.0002)

    # Sample the actual joint chords, not just ideal Cartesian endpoints.
    scene = json.loads((ROOT / "demo/scene/kitchen_config.json").read_text())
    box = scene["open_box"]
    assert box["center_xy_m"] == rt.cfg.grasp["open_box"]["center_xy_m"]
    assert .104 == pytest.approx(box["support_top_z_m"] + .10)
    meshes = finger_vertices()  # shipped binary STL collision vertices
    last = initial
    for goal in moves:
        chord = [last + fraction*(goal-last) for fraction in np.linspace(0, 1, 81)]
        assert all(rt.arm.harness.vet_pose(q) is None for q in chord)
        assert min(finger_minimum_z(rt.kin, q, meshes) for q in chord) > box["rim_top_z_m"] + .04
        last = goal


def test_box_keeps_unreachable_retreat_refusal_before_any_movement():
    rt, moves, opens, _ = box_runtime()
    rt.cfg.grasp["post_place_retreat_offset_m"] = .5
    rt.skill_move_home = lambda: pytest.fail("home after rejected loaded retreat")
    result = rt.skill_pick_and_place("green cube", destination="open box")
    assert result["ok"] is False and result["stage"] == "retreat_plan"
    assert result["holding"] == "green cube"
    assert not moves and not opens


def test_other_named_destinations_still_use_fresh_localization():
    rt, _, _, _ = box_runtime()
    labels = []
    rt._destination_fix = lambda name: (labels.append(name) or (np.array([.2, .1, .02]), .02))
    rt.skill_place_at = lambda x, y, z: {"placed": "green cube", "at": [x, y, z]}
    result = rt.skill_place_on_object("bowl")
    assert labels == ["bowl"]
    assert "destination_kind" not in result
    assert result["at"] == pytest.approx([.2, .1, .135])
