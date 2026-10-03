"""Captured-release geometry regressions; detached CPU fixtures, no steps."""
import json
from pathlib import Path

import numpy as np
import pytest

from cascade.skills import mujoco_withdrawal as withdrawal
from cascade.types import SkillError, SafetyViolation
from test_mujoco_withdrawal import model_runtime as model_runtime, captured  # noqa: F401

ROWS = json.loads((Path(__file__).with_name('fixtures')/
                   'mujoco_postrelease_e2d89c8.json').read_text())['rows']


def ready(rt):
    pose = captured(rt, ROWS[0])
    plan = withdrawal.prepare(rt, pose)
    assert plan.method == 'joint_0_first'
    rt._gripper_width_frac = lambda: (
        rt.arm.raw.world.data.qpos[rt.arm.raw._grip_qadr]-rt.arm.raw._grip_closed
    )/(rt.arm.raw._grip_open-rt.arm.raw._grip_closed)
    plan.retain()
    rt.arm.raw.world.data.qpos[:] = ROWS[1]['qpos']
    rt.held_object = None
    return plan


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_replan_keeps_debt_but_replaces_refuted_preopen_route(model_runtime):
    rt = model_runtime
    plan = ready(rt)
    world = rt.arm.raw.world
    before = world.data.qpos.copy(), world.data.ctrl.copy(), world.data.time
    plan.data = plan._snapshot()
    with pytest.raises(SkillError, match='escape intersects'):
        plan._escape_segment(world.data.qpos[rt.arm.raw._qadr], plan.q, plan.duration)
    proof = plan._replan_after_open()
    assert plan.method == 'joint_3_last'
    assert proof['previous_plan']['method'] == 'joint_0_first'
    assert proof['full_escape_and_home_checked'] and not proof['physical_task_verdict']
    assert proof['measured_gripper_rad'] == ROWS[1]['qpos'][5]
    assert rt.arm.harness._pending_model_withdrawal is plan
    assert rt._mujoco_withdrawals[id(rt.arm)] is plan
    assert proof['generation'] == plan.generation == 0
    np.testing.assert_array_equal(world.data.qpos, before[0])
    np.testing.assert_array_equal(world.data.ctrl, before[1])
    assert world.data.time == before[2] == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('field', [
    '_qadr', '_dadr', '_aidx', '_joint_names', '_act_names', 'n_joints',
    '_grip_qadr', '_grip_aidx', '_grip_joint', '_grip_act',
    '_grip_open', '_grip_closed', '_grip_ctrl_scale', '_grip_ctrl_offset',
    'runtime_open', 'runtime_closed',
])
def test_mapping_change_refuses_replan_and_inflight_command_scope(model_runtime, monkeypatch, field):
    rt = model_runtime
    plan = ready(rt)
    raw, world = rt.arm.raw, plan.world
    before = world.data.qpos.copy(), world.data.ctrl.copy(), world.data.time
    with plan._scope('move', target=plan.q, duration=plan.duration):
        owner, name = (rt, '_grip_'+field[8:]) if field.startswith('runtime_') else (raw, field)
        old = getattr(owner, name)
        changed = (list(reversed(old)) if isinstance(old, list) else
                   old+'_other' if isinstance(old, str) else old+.125)
        monkeypatch.setattr(owner, name, changed)
        with pytest.raises(SkillError, match='model changed'):
            plan._check_model_identity()
        # Even an already-entered transmission scope cannot borrow a changed
        # mapping on a later waypoint. This invokes the real common fence.
        with pytest.raises(SkillError, match='model changed'):
            rt.arm.harness.check_model_withdrawal()
    with pytest.raises(SkillError, match='model changed'):
        plan._replan_after_open()
    assert rt.arm.harness._pending_model_withdrawal is plan
    np.testing.assert_array_equal(world.data.qpos, before[0])
    np.testing.assert_array_equal(world.data.ctrl, before[1])
    assert world.data.time == before[2] == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('field', ['engine', 'model', 'data', 'engine_model', 'engine_data',
                                 'lock', 'engine_lock', 'ctrl_object', 'ctrl_layout'])
def test_native_driver_channel_rebind_is_refused(model_runtime, monkeypatch, field):
    from copy import copy
    import threading
    rt = model_runtime
    plan = ready(rt)
    raw, engine = rt.arm.raw, rt.arm.raw._engine
    with plan._scope('move', target=plan.q, duration=plan.duration):
        if field == 'engine':
            monkeypatch.setattr(raw, '_engine', copy(engine))
        elif field in ('model', 'data'):
            monkeypatch.setattr(raw, '_'+field, object())
        elif field.startswith('engine_'):
            monkeypatch.setattr(engine, field[7:], threading.RLock() if field == 'engine_lock' else object())
        elif field == 'lock':
            monkeypatch.setattr(raw, '_lock', threading.RLock())
        elif field == 'ctrl_object':
            monkeypatch.setattr(raw, '_ctrl', raw._ctrl.copy())
        else:
            monkeypatch.setattr(raw._ctrl, 'shape', (1, raw._ctrl.size))
        with pytest.raises(SkillError, match='binding changed'):
            rt.arm.harness.check_model_withdrawal()
    assert rt.arm.harness._pending_model_withdrawal is plan
    assert plan.world.data.time == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('field', ['_aidx', '_dadr', '_grip_aidx'])
def test_initial_indices_must_resolve_declared_model_names(model_runtime, monkeypatch, field):
    rt = model_runtime
    pose = captured(rt, ROWS[0])
    old = getattr(rt.arm.raw, field)
    monkeypatch.setattr(rt.arm.raw, field, list(reversed(old)) if isinstance(old, list) else 0)
    with pytest.raises(SkillError, match='coordinate binding'):
        withdrawal.prepare(rt, pose)
    assert rt.arm.raw.world.data.time == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('case', ['not_open', 'still_held', 'model', 'coordinates',
                                'prior_cancel', 'cancel', 'epoch', 'drift', 'deadline'])
def test_replan_failure_never_consumes_or_renews_debt(model_runtime, monkeypatch, case):
    rt = model_runtime
    plan = ready(rt)
    old_q = plan.q.copy()
    if case == 'not_open':
        rt._gripper_width_frac = lambda: None
    elif case == 'still_held':
        rt.held_object = 'red cube'
    elif case == 'model':
        plan.model.body_mass[plan.released_body] *= 2
    elif case == 'coordinates':
        rt.arm.raw._grip_open += .001
    elif case == 'prior_cancel':
        rt.arm.harness.estop(); rt.arm.harness.reset_estop()
    else:
        original = withdrawal.Withdrawal._plan_escape
        def changed(self, pose):
            original(self, pose)
            if case == 'cancel':
                self.harness.estop()
                self.harness.reset_estop()
            elif case == 'drift':
                self.world.data.qpos[self.raw._qadr[0]] += .000001
            elif case == 'epoch':
                self.world.placement_history.reset()
            else:
                self.deadline = 0.
        monkeypatch.setattr(withdrawal.Withdrawal, '_plan_escape', changed)
    with pytest.raises((SkillError, SafetyViolation), match='open-jaw|empty tool|model changed|cancelled|state changed|epoch binding|deadline'):
        plan._replan_after_open()
    assert plan.method == 'joint_0_first'
    np.testing.assert_array_equal(plan.q, old_q)
    assert rt.arm.harness._pending_model_withdrawal is plan
    assert rt._mujoco_withdrawals[id(rt.arm)] is plan
    assert plan.generation == 0


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('case', ['clean', 'state_drift', 'cancel'])
def test_first_command_has_fresh_exact_target_or_is_refused(model_runtime, monkeypatch, case):
    rt = model_runtime
    plan = ready(rt)
    sent = []
    def fake_move(goal, *, duration_s, _preflight, _halt_generation, _cancellation_token):
        # Only a synthetic delivery boundary: no driver or integration call.
        assert plan.method == 'joint_3_last'
        assert _halt_generation == plan.generation == 0
        assert _cancellation_token == plan.cancellation
        rt.arm.harness.check_model_withdrawal(command=True, target=goal, duration=duration_s)
        if case == 'state_drift':
            plan.world.data.qpos[rt.arm.raw._grip_qadr] -= .000001
        elif case == 'cancel':
            rt.arm.harness.estop(); rt.arm.harness.reset_estop()
        _preflight(plan.world.data.qpos[rt.arm.raw._qadr].copy(), duration_s)
        sent.append(goal.copy())
        return True
    monkeypatch.setattr(rt.arm, 'move_joints', fake_move)
    if case == 'clean':
        assert plan.withdraw()
        assert len(sent) == 1
        np.testing.assert_array_equal(sent[0], plan.q)
    else:
        with pytest.raises((SkillError, SafetyViolation), match='state changed|cancelled'):
            plan.withdraw()
        assert not sent
    assert rt.arm.harness._pending_model_withdrawal is plan
