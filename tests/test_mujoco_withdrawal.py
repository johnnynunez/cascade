"""Static native-collider tests; qpos assignments construct offline fixtures.

No integration/renderer is used here. The original two-pick physics test is
retained independently, including its actual final 6 cm destination limit.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.control.kinematics import Kinematics
from cascade.control.mujoco_arm import MujocoArm
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.skills import mujoco_withdrawal as withdrawal
from cascade.types import SkillError

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = json.loads((ROOT/'tests/fixtures/mujoco_release_states.json').read_text())['inputs']


@pytest.fixture
def model_runtime():
    pytest.importorskip('mujoco')
    pytest.importorskip('pinocchio')
    if not (ROOT/'assets/mjcf/so101/scene.xml').exists():
        pytest.skip('requires fetched SO-101 collision assets')
    cfg = load_demo_config(camera='mujoco_scene_two', arm='so101_mujoco', llm='mock')
    kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame, 5, cfg.arm.get('joint_signs'))
    raw = MujocoArm(cfg.arm, kin)
    raw.connect()
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kin)
    rt = SimpleNamespace(arm=SafeArm(raw, harness), kin=kin, cfg=cfg,
                         _profile_q=lambda key, _: np.asarray(cfg.arm.get(key), float))
    yield rt
    raw.disconnect()


def captured(rt, row):
    world = rt.arm.raw.world
    world.data.qpos[:] = row['qpos']
    world.mj.mj_kinematics(world.model, world.data)
    rt.held_object = rt._held_det_label = row['held_object']
    pose = rt.kin.fk(np.asarray(row['ctrl'][:5]))
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    pose[:3, :3] = _yaw_rotation(yaw, axis_order='open_down')
    return pose


def test_clear_escape_uses_real_colliders_without_modifying_live_state(model_runtime):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    data = rt.arm.raw.world.data
    qpos, ctrl, clock = data.qpos.copy(), data.ctrl.copy(), data.time
    plan = withdrawal.prepare(rt, pose)
    assert plan.method == 'joint_2_last'
    assert plan.rejections  # Direct sweep is not accepted.
    assert plan.target[2, 3] > float(rt.cfg.grasp.get('topdown_z_max'))
    np.testing.assert_array_equal(qpos, data.qpos)
    np.testing.assert_array_equal(ctrl, data.ctrl)
    assert data.time == clock == 0
    assert plan.receipt()['physical_task_verdict'] is False


def test_carry_into_placed_prefix_is_refused_in_scratch_before_any_write(model_runtime):
    rt = model_runtime
    captured(rt, CAPTURE[1]['rows'][0])
    data = rt.arm.raw.world.data
    qpos, ctrl, clock = data.qpos.copy(), data.ctrl.copy(), data.time
    goal = np.asarray(CAPTURE[1]['rows'][1]['ctrl'][:5])
    pose = rt.kin.fk(goal)
    with pytest.raises(SkillError, match='carry intersects another physical object'):
        withdrawal.prepare(rt, pose, carry_goals=[goal])
    np.testing.assert_array_equal(qpos, data.qpos)
    np.testing.assert_array_equal(ctrl, data.ctrl)
    assert data.time == clock == 0


def test_no_clearance_candidate_preserves_live_state(model_runtime):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    rt._profile_q = lambda *_: rt.arm.raw.world.data.qpos[rt.arm.raw._qadr].copy()
    pose[2, 3] = 5
    qpos = rt.arm.raw.world.data.qpos.copy()
    with pytest.raises(SkillError, match='no collision-clear'):
        withdrawal.prepare(rt, pose)
    np.testing.assert_array_equal(qpos, rt.arm.raw.world.data.qpos)


def test_legacy_ccd_is_not_silently_used_for_clearance(model_runtime):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    world = rt.arm.raw.world
    world.model.opt.disableflags |= world.mj.mjtDisableBit.mjDSBL_NATIVECCD
    with pytest.raises(SkillError, match='native convex'):
        withdrawal.prepare(rt, pose)


def test_stop_or_model_rebind_invalidates_prepared_escape(model_runtime):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    plan = withdrawal.prepare(rt, pose)
    rt.arm.harness.halt('test cancellation')
    with pytest.raises(Exception, match='halt|cancel|generation'):
        plan.after_withdrawal()


def test_pending_empty_tool_plan_cannot_carry_a_newly_held_object(model_runtime):
    rt = model_runtime
    plan = withdrawal.prepare(rt, captured(rt, CAPTURE[0]['rows'][1]))
    with pytest.raises(SkillError, match='newly held'):
        plan.home()


def test_nonfinite_native_distance_is_unavailable_not_clear(model_runtime):
    rt = model_runtime
    plan = withdrawal.prepare(rt, captured(rt, CAPTURE[0]['rows'][1]))
    plan.mj = SimpleNamespace(mj_geomDistance=lambda *args: float('nan'))
    with pytest.raises(SkillError, match='distance unavailable'):
        plan._distance(*plan.pairs[0])


def test_closed_jaw_feedback_does_not_authorize_escape(model_runtime):
    rt = model_runtime
    plan = withdrawal.prepare(rt, captured(rt, CAPTURE[0]['rows'][1]))
    for value in (None, float('nan'), .97):
        rt._gripper_width_frac = lambda value=value: value
        with pytest.raises(SkillError, match='actual open-jaw'):
            plan._require_open()


def test_another_arm_does_not_consume_the_pending_clearance_plan():
    from cascade.skills.runtime import SkillRuntime
    rt = SkillRuntime.__new__(SkillRuntime)
    first, second = object(), SimpleNamespace(move_joints=lambda *a, **kw: True)
    rt.arm = second
    rt._profile_q = lambda *_: np.zeros(1)
    rt._mujoco_withdrawals = {id(first): SimpleNamespace(home=lambda: pytest.fail('wrong arm'))}
    assert rt.skill_move_home() == {'at': 'home'}
    assert id(first) in rt._mujoco_withdrawals


def test_geometry_or_plan_clock_drift_fails_closed(model_runtime, monkeypatch):
    rt = model_runtime
    plan = withdrawal.prepare(rt, captured(rt, CAPTURE[0]['rows'][1]))
    plan.deadline = 0.
    with pytest.raises(SkillError, match='deadline'):
        plan.guard()
    monkeypatch.setattr(plan.world, 'model', object())
    with pytest.raises(SkillError, match='binding changed'):
        plan.guard()


def test_reset_names_without_actual_spawn_and_velocity_do_not_clear_physical_debt(model_runtime):
    rt = model_runtime
    plan = withdrawal.prepare(rt, captured(rt, CAPTURE[0]['rows'][1]))
    world = rt.arm.raw.world
    generation = rt.arm.harness._halt_generation
    with pytest.raises(SkillError, match='differs from the model spawn'):
        plan.verify_reset(world.free_body_names(), generation)
    names = world.reset_props()  # Authorized reset only in this isolated fixture.
    proof = plan.verify_reset(names, generation)
    assert proof['at_model_spawn'] and proof['zero_free_body_velocity']
    with pytest.raises(SkillError, match='every physical object'):
        plan.verify_reset(names[:1], generation)


def test_nonfinite_live_coordinates_refuse_without_repair(model_runtime):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    rt.arm.raw.world.data.qpos[-1] = float('nan')
    with pytest.raises(SkillError, match='non-finite'):
        withdrawal.prepare(rt, pose)
    assert np.isnan(rt.arm.raw.world.data.qpos[-1])


def test_optional_backend_absent_does_not_import_or_touch_physics():
    rt = SimpleNamespace(arm=SimpleNamespace(raw=object()))
    assert withdrawal.prepare(rt, None) is None


def test_occupied_preflight_is_terminal_before_carry_release_or_recovery(monkeypatch):
    from test_post_place_retreat import runtime
    rt, moves, opens, _ = runtime()
    def occupied(*args, **kwargs):
        assert 'carry_goals' in kwargs
        raise SkillError('planned carry intersects another physical object')
    monkeypatch.setattr(withdrawal, 'prepare', occupied)
    rt.skill_place_on_object = lambda _: rt.skill_place_at(.255, -.18, .1375)
    rt.skill_move_home = lambda: pytest.fail('home sweep after rejected carry preview')
    result = rt.skill_pick_and_place('tomato can', destination='shelf')
    assert result['ok'] is False and result['home_skipped'] is True
    assert result['holding'] == 'tomato can' and result['place_attempts'] == 1
    assert not moves and not opens


def retained(rt):
    plan = withdrawal.prepare(rt, captured(rt, CAPTURE[0]['rows'][1]))
    rt._grip_open = 1.
    plan.retain()
    return plan


@pytest.mark.parametrize('operation', ['joints', 'planned', 'open', 'close', 'nested_close', 'raw_close', 'cylinder'])
def test_retained_debt_blocks_common_actuation_before_backend_write(model_runtime, monkeypatch, operation):
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import SafetyViolation
    rt = model_runtime
    plan = retained(rt)
    writes = []
    monkeypatch.setattr(rt.arm.raw, 'stream_to', lambda *a, **kw: writes.append('motion'))
    monkeypatch.setattr(rt.arm.raw, 'set_gripper', lambda *a, **kw: writes.append('jaw'))
    rt._close_two_stage = lambda profile: SkillRuntime._close_two_stage(rt, profile)
    if operation == 'raw_close':
        monkeypatch.setattr(rt.arm.raw, 'close_gripper_two_stage', lambda **kw: writes.append('raw'), raising=False)
    actions = {
        'joints': lambda: rt.arm.move_joints(plan.q, plan.duration),
        'planned': lambda: rt.arm.move_planned(plan.q),
        'open': lambda: rt.arm.set_gripper(1., .6),
        'close': lambda: rt.arm.set_gripper(0.),
        'nested_close': lambda: SkillRuntime.skill_close_gripper(rt),
        'raw_close': lambda: SkillRuntime.skill_close_gripper(rt),
        'cylinder': lambda: rt.arm.harness.allow_grasp_descent([0., 0.]),
    }
    with pytest.raises(SafetyViolation, match='withdrawal'):
        actions[operation]()
    assert not writes
    assert rt._mujoco_withdrawals[id(rt.arm)] is plan
    assert rt.arm.harness._pending_model_withdrawal is plan
    assert rt.arm.get_state() is not None  # Reads are available.


@pytest.mark.parametrize('phase', ['open', 'withdraw'])
def test_failed_or_ambiguous_write_retains_common_barrier(model_runtime, monkeypatch, phase):
    from cascade.types import SafetyViolation
    rt = model_runtime
    plan = retained(rt)
    def failed(*a, **kw):
        raise RuntimeError('synthetic transport failed after possible write')
    if phase == 'open':
        monkeypatch.setattr(rt.arm.raw, 'set_gripper', failed)
        action = plan.open_hand
    else:
        monkeypatch.setattr(rt.arm.raw, 'stream_to', failed)
        action = plan.withdraw
    with pytest.raises(RuntimeError, match='possible write'):
        action()
    assert rt.arm.harness._pending_model_withdrawal is plan
    assert getattr(rt.arm.harness._model_withdrawal_scope, 'value', None) is None
    with pytest.raises(SafetyViolation, match='withdrawal'):
        rt.arm.set_gripper(1., .6)
    rt.arm.stop()  # Priority stop remains available and does not clear debt.
    assert rt.arm.harness.estopped
    assert rt.arm.harness._pending_model_withdrawal is plan


def test_scope_cannot_be_borrowed_by_other_thread_or_nested_same_thread(model_runtime, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from cascade.types import SafetyViolation
    rt = model_runtime
    plan = retained(rt)
    outcomes = []
    def raw_open(*args):
        # The first authorized write already consumed this one-command scope.
        with pytest.raises(SafetyViolation, match='already used'):
            rt.arm.set_gripper(1., .6)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rt.arm.set_gripper, 1., .6)
            with pytest.raises(SafetyViolation, match='scoped recovery'):
                future.result(timeout=2.)
        outcomes.append('only original write')
    monkeypatch.setattr(rt.arm.raw, 'set_gripper', raw_open)
    plan.open_hand()
    assert outcomes == ['only original write']


@pytest.mark.parametrize('change', ['target', 'duration', 'margin', 'mode', 'effort'])
def test_scope_requires_exact_command_even_on_its_own_thread(model_runtime, change):
    from cascade.types import SafetyViolation
    rt = model_runtime
    plan = retained(rt)
    with plan._scope('open' if change == 'effort' else 'move', target=plan.q, duration=plan.duration):
        with pytest.raises(SafetyViolation, match='withdrawal'):
            if change == 'effort':
                rt.arm.set_gripper(1., .8)
            elif change == 'mode':
                rt.arm.move_planned(plan.q, plan.duration)
            else:
                goal = plan.q.copy()
                if change == 'target':
                    goal[0] += .01
                rt.arm.move_joints(goal, plan.duration+(1 if change == 'duration' else 0),
                                   joint_margin=0. if change == 'margin' else None)


def test_rebinding_cannot_reuse_scope_and_another_harness_is_not_blocked(model_runtime, monkeypatch):
    from cascade.types import SafetyViolation
    rt = model_runtime
    plan = retained(rt)
    different = SafetyHarness(rt.arm.harness.limits, rt.kin)
    different.check_model_withdrawal(command=True)  # Independent arm/harness.
    with plan._scope('open'):
        monkeypatch.setattr(rt, 'kin', object())
        with pytest.raises(SkillError, match='binding changed'):
            rt.arm.set_gripper(1., .6)
    with pytest.raises(SafetyViolation, match='scoped recovery'):
        rt.arm.set_gripper(1., .6)


@pytest.mark.parametrize('change,reason', [
    ('missing_root', 'explicit articulation'),
    ('wrong_root', 'fixed articulation root'),
    ('shifted_root', 'kinematic origin'),
    ('wrong_joint', 'joint inventory'),
    ('wrong_coordinate', 'coordinate binding'),
    ('wrong_plane_name', 'support plane type'),
    ('plane_height', 'support plane type'),
    ('plane_orientation', 'support plane type'),
    ('plane_type', 'support plane type'),
    ('descriptor_height', 'horizontal safety table'),
    ('descriptor_rotation', 'horizontal safety table'),
])
def test_explicit_root_support_and_joint_admission(model_runtime, change, reason):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    raw, world = rt.arm.raw, rt.arm.raw.world
    cfg = raw._cfg._data
    root = world.mj.mj_name2id(world.model, world.mj.mjtObj.mjOBJ_BODY, 'base')
    floor = world.mj.mj_name2id(world.model, world.mj.mjtObj.mjOBJ_GEOM, 'floor')
    if change == 'missing_root':
        cfg.pop('mj_release_articulation_root')
    elif change == 'wrong_root':
        cfg['mj_release_articulation_root'] = 'gripper'
    elif change == 'shifted_root':
        world.model.body_pos[root, 0] = .01
    elif change == 'wrong_joint':
        raw._joint_names = list(raw._joint_names); raw._joint_names[0] = 'missing'
    elif change == 'wrong_coordinate':
        raw._qadr = list(reversed(raw._qadr))
    elif change == 'wrong_plane_name':
        cfg['mj_release_support_plane']['geoms'][0] = 'missing'
    elif change == 'plane_height':
        world.model.geom_pos[floor, 2] = .01
    elif change == 'plane_orientation':
        world.model.geom_quat[floor] = [0., 1., 0., 0.]
    elif change == 'plane_type':
        world.model.geom_type[floor] = world.mj.mjtGeom.mjGEOM_BOX
    elif change == 'descriptor_height':
        cfg['mj_release_support_plane']['position_m'][2] = .01
    else:
        cfg['mj_release_support_plane']['quaternion_wxyz'] = [0., 1., 0., 0.]
    before = world.data.qpos.copy(), world.data.ctrl.copy(), world.data.time
    with pytest.raises(SkillError, match=reason):
        withdrawal.prepare(rt, pose)
    np.testing.assert_array_equal(before[0], world.data.qpos)
    np.testing.assert_array_equal(before[1], world.data.ctrl)
    assert before[2] == world.data.time


@pytest.mark.parametrize('location', ['world', 'fixed_body'])
def test_extra_static_collider_in_native_model_is_rejected_without_solving(model_runtime, location):
    rt = model_runtime
    world = rt.arm.raw.world
    old_pose = captured(rt, CAPTURE[0]['rows'][1])
    spec = world.mj.MjSpec.from_file(world.path)
    parent = spec.worldbody if location == 'world' else spec.worldbody.add_body(name='extra_obstacle')
    parent.add_geom(name='unmodeled_block', type=world.mj.mjtGeom.mjGEOM_BOX,
                    size=[.01, .01, .01], pos=[.2, -.12, .1])
    world.model = spec.compile()
    world.data = world.mj.MjData(world.model)
    # No live world pose/solver step: the admitted inventory must fail before
    # any trajectory, even though this independently compiled body has no DOF.
    with pytest.raises(SkillError, match='unmodeled static collider'):
        withdrawal.prepare(rt, old_pose)
    assert world.data.time == 0


def test_optional_generated_plane_absent_in_plain_scene_is_admitted(model_runtime):
    rt = model_runtime
    raw, world = rt.arm.raw, rt.arm.raw.world
    spec = world.mj.MjSpec.from_file(str(ROOT/'assets/mjcf/so101/scene.xml'))
    world.model = spec.compile()
    plan = withdrawal.Withdrawal.__new__(withdrawal.Withdrawal)
    plan.raw, plan.model, plan.mj, plan.harness = raw, world.model, world.mj, rt.arm.harness
    root, support, joints = plan._admit_model()
    assert world.mj.mj_id2name(world.model, world.mj.mjtObj.mjOBJ_BODY, root) == 'base'
    assert len(support) == 1 and len(joints) == 5
    assert world.mj.mj_id2name(world.model, world.mj.mjtObj.mjOBJ_GEOM, support[0]) == 'floor'


def test_present_optional_plane_must_match_and_unknown_plane_is_rejected(model_runtime):
    rt = model_runtime
    pose = captured(rt, CAPTURE[0]['rows'][1])
    world = rt.arm.raw.world
    optional = world.mj.mj_name2id(world.model, world.mj.mjtObj.mjOBJ_GEOM, 'demo_floor')
    assert optional >= 0
    world.model.geom_pos[optional, 2] = .001
    with pytest.raises(SkillError, match='support plane type, body or pose'):
        withdrawal.prepare(rt, pose)
    world.model.geom_pos[optional, 2] = 0.
    rt.arm.raw._cfg._data['mj_release_support_plane']['optional_geoms'] = []
    with pytest.raises(SkillError, match='unmodeled static collider: demo_floor'):
        withdrawal.prepare(rt, pose)


def test_mocap_geometry_cannot_move_inside_an_assumed_fixed_model(model_runtime):
    rt = model_runtime
    world = rt.arm.raw.world
    pose = captured(rt, CAPTURE[0]['rows'][1])
    spec = world.mj.MjSpec.from_file(world.path)
    spec.worldbody.add_body(name='external_mocap', mocap=True)
    world.model = spec.compile()
    with pytest.raises(SkillError, match='fixed articulation root'):
        withdrawal.prepare(rt, pose)
    assert world.data.time == 0
