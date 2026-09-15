"""Retraction regressions from the independently recorded kitchen can fall.

All motion here is a recorder. FK uses the shipped URDF and finger meshes;
these tests validate planning and failure handling, not physical success.
"""
from pathlib import Path
from types import SimpleNamespace
import struct

import numpy as np
import pytest

from cascade.control.kinematics import Kinematics
from cascade.skills.runtime import SkillRuntime
from cascade.types import SafetyViolation, SkillError

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf'
# natural-can120-t3, passive sample95: held can before the shelf transfer.
CAN_Q = [-.7031141519546509, 1.848062515258789, 1.704153060913086,
         -1.4412031173706055, .00006180173659231514, 1.6955033540725708]
# Same can episode, sample111: released upright on the shelf, immediately
# before the home sweep that tipped it. This includes actual tracking error.
CAN_HOME_START_Q = np.array([.613564133644104, 1.8421465158462524,
                            1.7855331897735596, -1.5203155279159546,
                            .000008081468877207953, 2.988722085952759])
# pink120-observer-t3, passive sample63: actual lifted pink cube, before
# crossing from its spawn lane to the counter drop zone.
PINK_Q = [-.642940878868103, 1.4632208347320557, 1.182753324508667,
          -1.2798576354980469, .0026422080118209124, -2.8329057693481445]
HOME_Q = np.array([0., 1.2, 1.2, 0., .75, 0.])


def runtime(q=CAN_Q, *, offset=.025, retreat_failure=None):
    rt = SkillRuntime.__new__(SkillRuntime)
    grasp = {'pregrasp_offset_m': .04, 'topdown_z_max': .15,
             'release_height_m': .10, 'close_settle_s': 0.,
             'drop_zone': [.14, -.27], 'home_after_place': True}
    if offset is not None:
        grasp['post_place_retreat_offset_m'] = offset
    rt.cfg = SimpleNamespace(arm={}, safety={'table_z': 0.}, grasp=grasp)
    rt.kin = Kinematics(str(MODEL), 'gripper_end', 6, [-1] * 6)
    rt.held_object = rt._held_det_label = 'tomato can'
    rt._held_color = None
    rt._held_offset = np.array([-.0011, .0011, -.03365])
    rt._held_object_offset = lambda: rt._held_offset
    rt._reconcile_held = lambda: None
    rt._grip_open = 1.
    rt.memory = SimpleNamespace(add=lambda *a, **kw: None)
    rt.beliefs = SimpleNamespace(update=lambda *a, **kw: None)
    state = SimpleNamespace(q=np.array(q))
    moves, opens, cleared = [], [], []

    def move(goal, **kw):
        moves.append(np.asarray(goal).copy())
        if len(moves) == 3 and retreat_failure is not None:
            if isinstance(retreat_failure, Exception):
                raise retreat_failure
            return False
        state.q = np.asarray(goal).copy()
        return True

    rt.arm = SimpleNamespace(
        get_state=lambda: state, move_joints=move,
        set_gripper=lambda *a, **kw: opens.append((a, kw)),
        harness=SimpleNamespace(
            estopped=False, allow_grasp_descent=lambda *a, **kw: None,
            clear_grasp_exemption=lambda: cleared.append(True)),
    )
    return rt, moves, opens, cleared


def test_empty_tool_retreat_is_planned_before_motion_and_preserves_release():
    rt, moves, opens, _ = runtime()
    ik = rt.kin.ik
    plans = []

    def before_motion(T, q, **kw):
        assert not moves and not opens
        plans.append(T.copy())
        return ik(T, q, **kw)

    rt.kin.ik = before_motion
    result = rt.skill_place_at(.255, -.18, .1375)
    assert len(moves) == 3 and len(opens) == 1
    fk = [rt.kin.fk(q) for q in moves]
    assert fk[0][2, 3] <= .1451  # loaded carry cap is unchanged
    assert fk[1][2, 3] == pytest.approx(.1375, abs=.0001)
    assert fk[2][2, 3] == pytest.approx(.1625, abs=.0001)
    np.testing.assert_allclose(fk[1][:2, 3], fk[2][:2, 3], atol=.0002)
    np.testing.assert_allclose(fk[1][:3, :3], fk[2][:3, :3], atol=.0002)
    assert result['at'] == [.255, -.18, .138]
    assert result['post_place_retreat']['ok'] is True
    assert rt.held_object is None
    assert any(T[2, 3] == pytest.approx(.1625) for T in plans)


def test_unreachable_retreat_refuses_before_motion_or_release():
    rt, moves, opens, _ = runtime(offset=.5)
    with pytest.raises(SkillError, match='post-place retreat'):
        rt.skill_place_at(.255, -.18, .1375)
    assert not moves and not opens
    assert rt.held_object == 'tomato can'


def test_unreachable_retreat_does_not_enter_home_recovery_loop():
    rt, moves, opens, _ = runtime(offset=.5)
    rt.skill_place_on_object = lambda label: rt.skill_place_at(.255, -.18, .1375)
    rt.skill_move_home = lambda: pytest.fail('home recovery after failed retreat preflight')
    result = rt.skill_pick_and_place('tomato can', destination='shelf')
    assert result['ok'] is False and result['stage'] == 'retreat_plan'
    assert result['holding'] == 'tomato can' and result['home_skipped'] is True
    assert result['place_attempts'] == 1
    assert not moves and not opens


@pytest.mark.parametrize('offset', [0., -.01, float('nan'), float('inf')])
def test_invalid_retreat_offset_refuses_before_motion(offset):
    rt, moves, opens, _ = runtime(offset=offset)
    with pytest.raises(SkillError, match='finite and positive'):
        rt.skill_place_at(.255, -.18, .1375)
    assert not moves and not opens


@pytest.mark.parametrize('failure', [False, SafetyViolation('blocked retreat')])
def test_released_retreat_failure_skips_home_and_never_regrasps(failure):
    rt, moves, opens, cleared = runtime(retreat_failure=failure)
    rt.skill_place_on_object = lambda label: rt.skill_place_at(.255, -.18, .1375)
    rt.skill_move_home = lambda: pytest.fail('unsafe home after failed retreat')
    rt.skill_grasp_object = lambda *a, **kw: pytest.fail('regrasp of already released can')
    result = rt.skill_pick_and_place('tomato can', destination='shelf')
    assert result['ok'] is False and result['stage'] == 'retreat'
    assert result['home_skipped'] is True
    assert result['placed_at'] == [.255, -.18, .138]
    assert result['place_attempts'] == 1 and result['grasp_attempts'] == 0
    assert result['post_place_retreat']['ok'] is False
    assert len(moves) == 3 and len(opens) == 1 and cleared == [True]
    assert rt.held_object is None


def test_successful_retreat_precedes_normal_home():
    rt, moves, _, _ = runtime()
    rt.skill_place_on_object = lambda label: rt.skill_place_at(.255, -.18, .1375)

    def home():
        assert len(moves) == 3
        assert rt.kin.fk(moves[-1])[2, 3] > .1623
        return rt.arm.move_joints(HOME_Q)

    rt.skill_move_home = home
    result = rt.skill_pick_and_place('tomato can', destination='shelf')
    assert result['post_place_retreat']['ok'] is True
    assert result['return_home'] == {'attempted': True, 'ok': True, 'at': 'home'}
    assert 'home_skipped' not in result
    assert len(moves) == 4


@pytest.mark.parametrize('failure', [SkillError('did not settle at home'), SafetyViolation('home refused')])
def test_pick_result_preserves_home_failure_without_repeating_motion(failure):
    rt, moves, _, _ = runtime()
    rt.skill_place_on_object = lambda label: rt.skill_place_at(.255, -.18, .1375)
    calls = []

    def home():
        calls.append('home')
        raise failure

    rt.skill_move_home = home
    result = rt.skill_pick_and_place('tomato can', destination='shelf')
    assert result['return_home'] == {'attempted': True, 'ok': False, 'error': str(failure)}
    assert calls == ['home'] and len(moves) == 3
    assert result['post_place_retreat']['ok'] is True


def test_disabled_home_is_reported_without_claiming_the_arm_is_home():
    rt, moves, _, _ = runtime()
    rt.cfg.grasp['home_after_place'] = False
    rt.skill_place_on_object = lambda label: rt.skill_place_at(.255, -.18, .1375)
    rt.skill_move_home = lambda: pytest.fail('home disabled by profile')
    result = rt.skill_pick_and_place('tomato can', destination='shelf')
    assert result['return_home'] == {'attempted': False, 'reason': 'disabled_by_profile'}
    assert len(moves) == 3


def test_optional_offset_keeps_pink_counter_motion_identical():
    baseline, old, _, _ = runtime(PINK_Q, offset=None)
    opted_in, new, _, _ = runtime(PINK_Q, offset=.025)
    baseline.skill_place_at(.14, -.27)
    result = opted_in.skill_place_at(.14, -.27)
    assert len(old) == len(new) == 3
    np.testing.assert_array_equal(old, new)
    assert result['post_place_retreat']['tcp_at'][2] == .14


def finger_vertices():
    dtype = np.dtype([('normal', '<f4', 3), ('points', '<f4', (3, 3)), ('attribute', '<u2')])
    result = {}
    for name in ('gripper_left', 'gripper_right'):
        raw = (MODEL.parent.parent / 'meshes' / (name + '.STL')).read_bytes()
        count = struct.unpack('<I', raw[80:84])[0]
        assert len(raw) == 84 + 50 * count
        result[name] = np.unique(np.frombuffer(raw, dtype=dtype, count=count, offset=84)['points'].reshape(-1, 3), axis=0)
    return result


def finger_minimum_z(kin, q, meshes):
    full = np.r_[q, .05, .05]  # actual fully open prismatic jaw positions
    kin._pin.forwardKinematics(kin.model, kin.data, full)
    kin._pin.updateFramePlacements(kin.model, kin.data)
    lowest = []
    for name, vertices in meshes.items():
        T = np.asarray(kin.data.oMf[kin.model.getFrameId(name)].homogeneous)
        lowest.append(float((vertices @ T[2, :3] + T[2, 3]).min()))
    return min(lowest)


def test_recorded_can_retreat_chord_and_home_keep_fingers_above_rim():
    rt, moves, _, _ = runtime()
    rt.skill_place_at(.255, -.18, .1375)
    low, retreat = moves[1:]
    # Evaluate the same straight joint chord traversed by min-jerk timing.
    chord = [low + (retreat - low) * s for s in np.linspace(0., 1., 101)]
    poses = np.array([rt.kin.fk(q)[:3, 3] for q in chord])
    assert np.all(np.diff(poses[:, 2]) > 0)
    assert np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1).max() < .001
    meshes = finger_vertices()
    home_path = [retreat + (HOME_Q - retreat) * s for s in np.linspace(0., 1., 201)]
    old_path = [CAN_HOME_START_Q + (HOME_Q - CAN_HOME_START_Q) * s
                for s in np.linspace(0., 1., 201)]
    minimum = min(finger_minimum_z(rt.kin, q, meshes) for q in home_path)
    old_minimum = min(finger_minimum_z(rt.kin, q, meshes) for q in old_path)
    assert minimum > .155  # 145mm can top plus at least10mm planned margin
    assert old_minimum < .143  # reproduces the recorded sweep below the rim
