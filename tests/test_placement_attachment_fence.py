"""Actual SafeArm/ArmBase consumer races; native writes are intercepted."""
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.skills import carry_attachment, mujoco_region, mujoco_withdrawal
from cascade.skills.held_observation import HeldObservationInvalid
from cascade.types import SafetyViolation, SkillError, make_transform
from test_placement_attachment import model_runtime as model_runtime, replay, as_skill_runtime  # noqa: F401


class ProposedTarget(Exception):
    pass


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('where', ['healthy', 'safearm_read', 'backend_read', 'approval', 'between_targets'])
def test_stop_reset_cannot_reauthorize_the_actual_consumer(model_runtime, monkeypatch, where):
    rt = model_runtime
    world = replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    candidate = mujoco_region.select(skill, retain_aim=True)
    raw, harness = skill.arm.raw, skill.arm.harness
    original_qpos, original_ctrl = world.data.qpos.copy(), world.data.ctrl.copy()
    original_move, original_read, original_approve = carry_attachment.move, raw.get_state, harness.approve
    armed, reads, targets = False, 0, []
    def stop_reset():
        harness.estop('controlled cancellation between admission and consumer')
        harness.reset_estop()
    def move(*args, **kwargs):
        nonlocal armed
        armed = True
        return original_move(*args, **kwargs)
    def read(*args, **kwargs):
        nonlocal reads
        result = original_read(*args, **kwargs)
        if armed:
            reads += 1
            if (where == 'safearm_read' and reads == 1
                    or where == 'backend_read' and reads == 2):
                stop_reset()
        return result
    def approve(*args, **kwargs):
        original_approve(*args, **kwargs)
        if where == 'approval': stop_reset()
    def no_write(q):
        targets.append(q.copy())
        if where == 'between_targets' and len(targets) == 1:
            stop_reset()
            return
        raise ProposedTarget('first target boundary, no native write')
    monkeypatch.setattr(carry_attachment, 'move', move)
    monkeypatch.setattr(raw, 'get_state', read)
    monkeypatch.setattr(raw, 'send_joint_target', no_write)
    monkeypatch.setattr(harness, 'approve', approve)
    harness.heartbeat()
    with pytest.raises((SafetyViolation, SkillError, ProposedTarget)):
        skill.skill_place_at(*candidate.report['target'], _model_plan=candidate)
    assert len(targets) == (1 if where in ('healthy', 'between_targets') else 0)
    assert reads >= 1
    assert world.data.time == 0.
    np.testing.assert_array_equal(world.data.qpos, original_qpos)
    np.testing.assert_array_equal(world.data.ctrl, original_ctrl)
    assert skill.held_object == rt.held_object


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_started_stream_preserves_token_without_treating_motion_as_stale_snapshot(model_runtime, monkeypatch):
    rt = model_runtime
    world = replay(rt, monkeypatch)
    candidate = mujoco_region.select(rt, retain_aim=True)
    candidate.aim.before_stream()
    # Isolated fixture mutation represents progress, not an admitted solve.
    world.data.qpos[0] += .001
    world.data.time = .005
    candidate.aim.before_stream()
    assert candidate.aim.motion_arguments()['_cancellation_token'] == 0
    rt.arm.harness.estop(); rt.arm.harness.reset_estop()
    with pytest.raises(SafetyViolation, match='cancelled'):
        candidate.aim.before_stream()


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('where', ['transfer', 'before_open'])
def test_stop_reset_cannot_renew_the_withdrawal_transfer(model_runtime, monkeypatch, where):
    rt = model_runtime
    world = replay(rt, monkeypatch)
    candidate = mujoco_region.select(rt, retain_aim=True)
    q = candidate.geometry
    withdrawal = mujoco_withdrawal.prepare(rt, make_transform(q.rotation, q.target),
                                          carry_goals=[q.pre.q, q.low.q])
    writes = []
    monkeypatch.setattr(rt.arm, 'set_gripper', lambda *a, **k: writes.append(a))
    if where == 'transfer':
        rt.arm.harness.estop(); rt.arm.harness.reset_estop()
        with pytest.raises(SafetyViolation, match='cancelled'):
            candidate.aim.admit_release(withdrawal)
    else:
        candidate.aim.admit_release(withdrawal)
        withdrawal.retain()
        rt.arm.harness.estop(); rt.arm.harness.reset_estop()
        with pytest.raises(SkillError, match='cancelled'):
            withdrawal.open_hand()
    assert not writes and world.data.time == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_release_with_another_context_cannot_replace_the_measured_owner(model_runtime, monkeypatch):
    rt = model_runtime
    replay(rt, monkeypatch)
    candidate = mujoco_region.select(rt, retain_aim=True)
    counterfeit = SimpleNamespace(arm=rt.arm, world=rt.arm.raw.world,
                                  generation=0, cancellation=0)
    with pytest.raises(HeldObservationInvalid, match='release did not retain'):
        candidate.aim.admit_release(counterfeit)


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('include_lift', [False, True])
def test_every_carry_segment_preserves_context_before_the_release_handoff(model_runtime, monkeypatch,
                                                                        include_lift):
    from dataclasses import replace
    from cascade.skills import place_geometry
    rt = model_runtime
    world = replay(rt, monkeypatch)
    skill = as_skill_runtime(rt)
    candidate = mujoco_region.select(skill, retain_aim=True)
    original_plan = place_geometry.plan
    calls = []
    def geometry(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        # A zero-length proposed lift exercises the optional dispatch branch;
        # it is synthetic wiring, not proof of a physically needed lift.
        return replace(result, lift=SimpleNamespace(q=candidate.aim.q.copy(), success=True)) if include_lift else result
    def no_motion(runtime, q, **kwargs):
        calls.append(kwargs)
        kwargs['before_stream']()
        if len(calls) == (3 if include_lift else 2):
            runtime.arm.harness.estop(); runtime.arm.harness.reset_estop()
        return True
    monkeypatch.setattr(place_geometry, 'plan', geometry)
    monkeypatch.setattr(carry_attachment, 'move', no_motion)
    monkeypatch.setattr(skill.arm, 'set_gripper', lambda *a, **k: pytest.fail('cancelled release'))
    with pytest.raises(SkillError, match='cancelled'):
        skill.skill_place_at(*candidate.report['target'])
    assert len(calls) == (3 if include_lift else 2)
    assert all(c['_halt_generation'] == 0 and c['_cancellation_token'] == 0 for c in calls)
    assert len({id(c['before_stream'].__self__) for c in calls}) == 1
    assert world.data.time == 0. and skill.held_object == rt.held_object


@pytest.mark.parametrize('token', [True, -1, 0., '0'])
def test_invalid_token_is_rejected_before_the_device_read(token):
    from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
    harness = SafetyHarness(SafetyLimits(np.zeros(3), np.ones(3)))
    raw = SimpleNamespace(get_state=lambda: pytest.fail('invalid context reached device'))
    with pytest.raises(SafetyViolation, match='nonnegative integer'):
        SafeArm(raw, harness).move_joints(np.zeros(1), _cancellation_token=token)
