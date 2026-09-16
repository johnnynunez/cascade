"""Carry clearance from the measured held-can/platform collision."""
from types import SimpleNamespace
import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.types import SkillError, make_transform
from test_post_place_retreat import runtime

# Actual D-01 lifted can, before horizontal transport. Local joint signs.
LIFTED_Q = [-.7623136639595032, 1.8145228624343872, 1.6668847799301147,
            -1.4279340505599976, -.000002137392357326462, -3.0284764766693115]


def enabled(*, offset=.025):
    rt, moves, opens, cleared = runtime(LIFTED_Q, offset=offset)
    rt.cfg.grasp['pre_carry_lift'] = True
    rt.arm.harness.vet_pose = lambda *_args, **_kwargs: None
    return rt, moves, opens, cleared


def test_gpu_kitchen_enables_the_measured_clearance_phase():
    assert load_demo_config(arm='isaac_kitchen_gpu').grasp.get('pre_carry_lift') is True


def test_vertical_clearance_precedes_xy_transport_and_all_poses_are_preplanned():
    rt, moves, opens, _ = enabled()
    initial = rt.kin.fk(LIFTED_Q)
    original = rt.kin.ik
    def checked(T, q, **kw):
        assert not moves and not opens
        return original(T, q, **kw)
    rt.kin.ik = checked
    rt.skill_place_at(.14, -.27)
    poses = [rt.kin.fk(q) for q in moves]
    assert len(moves) == 4 and len(opens) == 1
    np.testing.assert_allclose(poses[0][:2, 3], initial[:2, 3], atol=.0002)
    np.testing.assert_allclose(poses[0][:3, :3], initial[:3, :3], atol=.0002)
    assert poses[0][2, 3] == pytest.approx(.14, abs=.0002)
    assert np.linalg.norm(poses[1][:2, 3]-initial[:2, 3]) > .3
    assert poses[1][2, 3] <= .1451
    assert poses[2][2, 3] == pytest.approx(.1, abs=.0002)
    # The loaded transfer starts at clearance; it no longer gains it while
    # already passing over the45mm platform with a100mm can hanging below.
    heights = [rt.kin.fk(moves[0]*(1-s)+moves[1]*s)[2, 3] for s in np.linspace(0, 1, 101)]
    assert min(heights) > .139


def test_unreachable_vertical_lift_refuses_before_any_motion_or_opening():
    rt, moves, opens, _ = enabled()
    original = rt.kin.ik
    here = rt.kin.fk(LIFTED_Q)[:2, 3]
    def blocked(T, q, **kw):
        if np.linalg.norm(T[:2, 3]-here) < .001 and T[2, 3] > .135:
            return SimpleNamespace(success=False, q=np.asarray(q))
        return original(T, q, **kw)
    rt.kin.ik = blocked
    with pytest.raises(SkillError, match='pre-carry lift'):
        rt.skill_place_at(.14, -.27)
    assert not moves and not opens and rt.held_object == 'tomato can'


def test_unsafe_vertical_chord_refuses_before_any_motion():
    rt, moves, opens, _ = enabled()
    rt.arm.harness.vet_pose = lambda *_args, **_kwargs: 'blocked geometry'
    with pytest.raises(SkillError, match='pre-carry lift'):
        rt.skill_place_at(.14, -.27)
    assert not moves and not opens


def test_failed_vertical_settle_stops_without_xy_motion_or_release():
    rt, moves, opens, _ = enabled()
    def refuse(q, **kw):
        moves.append(np.asarray(q)); return False
    rt.arm.move_joints = refuse
    with pytest.raises(SkillError, match='pre-carry lift'):
        rt.skill_place_at(.14, -.27)
    assert len(moves) == 1 and not opens and rt.held_object == 'tomato can'


def test_clearance_failure_does_not_trigger_home_or_regrasp_recovery():
    rt, moves, opens, _ = enabled()
    rt.arm.move_joints = lambda *_args, **_kwargs: False
    rt.skill_move_home = lambda: pytest.fail('home while carry clearance is unavailable')
    rt.skill_grasp_object = lambda *_args, **_kwargs: pytest.fail('regrasp while holding')
    result = rt.skill_pick_and_place('tomato can', destination='drop zone')
    assert result['ok'] is False and result['stage'] == 'carry_clearance'
    assert result['home_skipped'] is True and result['holding'] == 'tomato can'
    assert not opens


def test_failed_square_carry_keeps_the_calibrated_destination_for_physics():
    from cascade.agent.effects import PostconditionChecker, REFUTED
    rt, moves, opens, _ = enabled()
    rt.cfg.grasp.update(drop_zone=[.14, -.27], drop_zone_name='green square')
    rt.arm.move_joints = lambda *_args, **_kwargs: False
    rt.skill_move_home = lambda: pytest.fail('failure must leave the grasp alone')
    result = rt.skill_pick_and_place('tomato can', destination='green square')
    assert result['ok'] is False and result['stage'] == 'carry_clearance'
    assert result['destination_kind'] == 'configured_point'
    assert result['target'] == [.14, -.27]
    calls = []
    def physical_pose(name):
        calls.append(name)
        return np.array([.3721, .1225, .0618]) if name == 'tomato can' else np.zeros(3)
    pc = PostconditionChecker(object_pose=physical_pose).verify('pick_and_place',
        {'object': 'tomato can', 'destination': 'green square'}, result,
        {'label': 'tomato can', 'pose': [.225, .215, .04125], 'channel': 'physics'})
    assert pc.status == REFUTED and 'green square' not in calls
    assert pc.measured['target_err_m'] == pytest.approx(.4561, abs=.0001)
    assert not opens


def test_lift_reuses_existing_ceiling_and_does_not_repeat_at_clearance():
    rt, moves, opens, _ = enabled(offset=None)
    rt.skill_place_at(.14, -.27, .30)
    assert max(rt.kin.fk(q)[2, 3] for q in moves) <= .1452
    rt2, moves2, _, _ = enabled(offset=None)
    pose = rt2.kin.fk(LIFTED_Q); pose[2, 3] = .14
    at_height = rt2.kin.ik(pose, np.asarray(LIFTED_Q))
    assert at_height.success
    rt2.arm.get_state().q = at_height.q
    rt2.skill_place_at(.14, -.27)
    assert len(moves2) == 3
