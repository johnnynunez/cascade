"""Measured aiming and ordinary skill consumption; zero dynamics integration."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.skills import mujoco_placement_aim as aiming, mujoco_region
from cascade.skills.held_observation import HeldOffset, HeldObservationInvalid
from cascade.skills.place_geometry import plan
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError, make_transform
from test_mujoco_withdrawal import model_runtime as model_runtime  # noqa: F401

CAPTURE = json.loads((Path(__file__).parent/'fixtures/placement_attachment_7a52.json').read_text())


def replay(rt, monkeypatch):
    world = rt.arm.raw.world
    for name in ('mj_step', 'mj_step1', 'mj_step2'):
        monkeypatch.setattr(world.mj, name, lambda *a, **k: pytest.fail('no integration in replay'))
    monkeypatch.setattr(rt.arm.raw._engine, 'push_ctrl', lambda *a: pytest.fail('no native command'))
    world.data.qpos[:] = CAPTURE['qpos']
    world.data.qvel[:] = CAPTURE['qvel']
    world.data.ctrl[:] = CAPTURE['ctrl']
    world.mj.mj_kinematics(world.model, world.data)
    rt.held_object = rt._held_det_label = CAPTURE['held_object']
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    history = world.placement_history
    history.confirmed_prefix = set(CAPTURE['confirmed_prefix'])
    assert history.identity == CAPTURE['model_sha256']
    return world


@pytest.mark.parametrize('axis_order', ['open_down', 'down_open'])
@pytest.mark.parametrize('xy', [(0.2, .1), (-.1, .2), (.2, -.2)])
@pytest.mark.parametrize('offset', [[.01, -.02, .03], [-.025, .015, -.02], [0., 0., 0.]])
def test_destination_orientation_rotates_attachment_without_changing_height(axis_order, xy, offset):
    goals = []
    def ik(pose, seed):
        goals.append(pose.copy())
        return SimpleNamespace(q=np.zeros(5), success=True)
    rt = SimpleNamespace(cfg=SimpleNamespace(grasp={'pregrasp_offset_m': .02}, safety={'table_z': 0.}),
                         kin=SimpleNamespace(fk=lambda q: np.eye(4), ik=ik),
                         _tool_axis_order=axis_order)
    g = plan(rt, np.zeros(5), np.array([*xy, .045]), x=xy[0], y=xy[1],
             release_z=.045, z_cap=.1, attachment_translation_tool=offset)
    np.testing.assert_allclose(g.target[:2]+(g.rotation @ offset)[:2], xy, atol=1e-15)
    np.testing.assert_array_equal(goals[-1][:3, 3], g.target)
    assert g.target[2] == .045 and g.hover[2] == .065


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_captured_world_offset_refuses_but_rotation_aware_plan_passes_same_gates(model_runtime, monkeypatch):
    from cascade.skills.mujoco_withdrawal import prepare
    from cascade.sim.mujoco_placement import state_digest
    rt = model_runtime
    world = replay(rt, monkeypatch)
    before = state_digest(world.data)
    q = world.data.qpos[rt.arm.raw._qadr].copy()
    history = world.placement_history
    history.geometry()
    center = history.scratch.xpos[history.bodies['blue_cube'][0]].copy()
    x, y = CAPTURE['region_target']
    target = np.array([x, y, .045])
    target[:2] -= (center-rt.kin.fk(q)[:3, 3])[:2]
    old = plan(rt, q, target, x=x, y=y, release_z=.045, z_cap=.045)
    with pytest.raises(SkillError, match='no collision-clear release escape'):
        prepare(rt, make_transform(old.rotation, target), carry_goals=[old.pre.q, old.low.q])
    candidate = mujoco_region.select(rt, retain_aim=True)
    np.testing.assert_array_equal(candidate.report['target'], CAPTURE['region_target'])
    np.testing.assert_array_equal(candidate.report['target_tcp_m'], candidate.geometry.target)
    assert candidate.report['placement_aim']['release_authority'] is False
    assert candidate.report['physical_task_verdict'] is False
    assert candidate.report['object_separation_m'] == .02
    assert candidate.report['capacity']['planning_margin_m'] == .02
    assert state_digest(world.data) == before and world.data.time == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('change', ['qpos', 'qvel', 'ctrl', 'time', 'epoch', 'model', 'mapping',
                                   'held', 'stop_reset', 'deadline', 'data', 'arm'])
def test_measured_attachment_rejects_drift_without_refresh(model_runtime, monkeypatch, change):
    from cascade.safety.harness import SafeArm
    rt = model_runtime
    world = replay(rt, monkeypatch)
    q = world.data.qpos[rt.arm.raw._qadr].copy()
    aim = aiming.capture(rt, q)
    if change in ('qpos', 'qvel', 'ctrl'):
        getattr(world.data, change)[0] += .001
    elif change == 'time': world.data.time += .001
    elif change == 'epoch': world.placement_history.epoch = 'replacement'
    elif change == 'model': world.model.body_mass[-1] += .001
    elif change == 'mapping': rt.arm.raw._aidx = list(reversed(rt.arm.raw._aidx))
    elif change == 'held': rt.held_object = 'red cube'
    elif change == 'stop_reset':
        rt.arm.harness.estop()
        rt.arm.harness.reset_estop()
    elif change == 'deadline': aim.deadline = 0.
    elif change == 'data': world.data = world.mj.MjData(world.model)
    elif change == 'arm': rt.arm = SafeArm(rt.arm.raw, rt.arm.harness)
    with pytest.raises((HeldObservationInvalid, ValueError), match='changed|cancelled|deadline'):
        aim.guard()


def as_skill_runtime(rt):
    skill = SkillRuntime.__new__(SkillRuntime)
    for name in ('arm', 'cfg', 'kin', '_grip_open', '_grip_closed', '_profile_q',
                 'held_object', '_held_det_label'):
        setattr(skill, name, getattr(rt, name))
    skill._adopt_unknown_held = lambda: None
    skill._held_color = None
    skill._aim_notes = []
    skill.memory = SimpleNamespace(add=lambda *a, **k: skill._aim_notes.append(a))
    skill.beliefs = SimpleNamespace(update=lambda *a, **k: None)
    skill._held_object_observation = lambda: HeldOffset(np.array([.03, .04, -.02]), 'cached_aim')
    return skill


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('use_region_handoff', [False, True])
def test_ordinary_place_consumes_the_reported_pose_not_a_different_cached_offset(model_runtime, monkeypatch,
                                                                               use_region_handoff):
    from cascade.skills import carry_attachment
    rt = model_runtime
    world = replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    candidate = mujoco_region.select(skill, retain_aim=True)
    moves = []
    monkeypatch.setattr(carry_attachment, 'move', lambda runtime, goal, **kw: moves.append(goal.copy()) or False)
    # The normal skill enters its real preflight. The test stops at the first
    # proposed write, so it claims no controller/physics/release success.
    with pytest.raises(SkillError, match='did not settle above'):
        skill.skill_place_at(*candidate.report['target'],
                            **({'_model_plan': candidate} if use_region_handoff else {}))
    assert len(moves) == 1
    np.testing.assert_array_equal(moves[0], candidate.geometry.pre.q)
    np.testing.assert_array_equal(candidate.report['target_tcp_m'], candidate.geometry.target)
    expected = ', '.join(f'{v:.3f}' for v in candidate.geometry.target)
    assert any(f'planned TCP aim [{expected}]' in note[1] for note in skill._aim_notes)
    assert candidate.geometry.target[2] == .045000000000000005
    assert skill.held_object == CAPTURE['held_object'] and world.data.time == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('change', ['state', 'cancel', 'target', 'geometry'])
def test_region_handoff_refuses_changed_context_before_observation_or_motion(model_runtime, monkeypatch, change):
    rt = model_runtime
    world = replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    candidate = mujoco_region.select(skill, retain_aim=True)
    target = candidate.report['target'].copy()
    if change == 'state': world.data.qpos[0] += .001
    elif change == 'cancel': skill.arm.harness.estop(); skill.arm.harness.reset_estop()
    elif change == 'target': target[0] += .001
    elif change == 'geometry': candidate.geometry.pre.q[0] += .001
    skill._held_object_observation = lambda: pytest.fail('invalid handoff reached observation')
    writes = []
    monkeypatch.setattr('cascade.skills.carry_attachment.move', lambda *a, **k: writes.append(a))
    with pytest.raises(HeldObservationInvalid):
        skill.skill_place_at(*target, _model_plan=candidate)
    assert not writes and skill.held_object == CAPTURE['held_object']


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_region_consumption_keeps_original_three_second_budget(model_runtime, monkeypatch):
    rt = model_runtime
    replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    clocks = []
    monotonic = aiming.time.monotonic
    def observed_clock():
        value = monotonic()
        clocks.append(value)
        return value
    monkeypatch.setattr(mujoco_region, 'time', SimpleNamespace(monotonic=observed_clock))
    candidate = mujoco_region.select(skill, retain_aim=True)
    assert candidate.aim.deadline == clocks[0]+3.
    deadline = candidate.aim.deadline
    # This module-local facade cannot change production/runtime/socket clocks.
    monkeypatch.setattr(aiming, 'time', SimpleNamespace(monotonic=lambda: deadline+.001))
    skill._held_object_observation = lambda: pytest.fail('expired handoff reached observation')
    with pytest.raises(HeldObservationInvalid, match='deadline'):
        skill.skill_place_at(*candidate.report['target'], _model_plan=candidate)
    assert candidate.aim.deadline == deadline


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_snapshot_change_during_execution_ik_never_reaches_write(model_runtime, monkeypatch):
    from cascade.skills import place_geometry
    rt = model_runtime
    world = replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    original = place_geometry.plan
    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        world.data.qvel[0] += .001
        return result
    monkeypatch.setattr(place_geometry, 'plan', drift)
    monkeypatch.setattr('cascade.skills.carry_attachment.move', lambda *a, **k: pytest.fail('drift authorized move'))
    with pytest.raises(HeldObservationInvalid, match='state changed'):
        skill.skill_place_at(*CAPTURE['region_target'])
    assert skill.held_object == CAPTURE['held_object']


def test_unconfigured_catalog_does_not_materialize_lazy_arm():
    class NotMaterialized:
        def __getattr__(self, name): pytest.fail('unexpected device access')
    rt = SimpleNamespace(cfg=SimpleNamespace(arm={}), arm=NotMaterialized())
    assert aiming.capture(rt, None) is None


def test_serialized_report_is_not_a_private_placement_plan():
    skill = SkillRuntime.__new__(SkillRuntime)
    with pytest.raises(SkillError, match='private bound region plan'):
        skill.skill_place_at(.2, -.1, _model_plan={'target': [.2, -.1], 'physical_task_verdict': True})


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('clock', [-.001, float('nan')])
def test_invalid_snapshot_clock_cannot_authorize_aim(model_runtime, monkeypatch, clock):
    rt = model_runtime
    world = replay(rt, monkeypatch)
    world.data.time = clock
    with pytest.raises(HeldObservationInvalid, match='clock'):
        aiming.capture(rt, world.data.qpos[rt.arm.raw._qadr].copy())


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_rotated_aim_does_not_authorize_carry_into_the_placed_prefix(model_runtime, monkeypatch):
    rt = model_runtime
    world = replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    red = world.data.qpos[6:8].copy()
    monkeypatch.setattr('cascade.skills.carry_attachment.move', lambda *a, **k: pytest.fail('unsafe carry write'))
    with pytest.raises(SkillError, match='carry intersects|collision-clear|collision clearance'):
        skill.skill_place_at(*red)
    assert skill.held_object == CAPTURE['held_object'] and world.data.time == 0.
