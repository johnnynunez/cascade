"""CPU cancellation contracts with synthetic already-admitted geometry.

Real Withdrawal retention/recovery, SkillRuntime, SafeArm and ArmBase fences;
no simulator, renderer, physical clearance claim or device write.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.mock_arm import MockArm
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.skills.mujoco_withdrawal import Withdrawal
from cascade.skills.runtime import SkillRuntime
from cascade.types import SafetyViolation, SkillError


@pytest.fixture
def debt():
    raw = MockArm(Cfg({'n_joints': 1, 'home_q': [0.]}))
    model = SimpleNamespace(nu=1, njnt=1, jnt_type=[0], jnt_qposadr=[1],
                            jnt_dofadr=[1], qpos0=np.zeros(8))
    data = SimpleNamespace(qpos=model.qpos0.copy(), qvel=np.zeros(7),
                           ctrl=np.zeros(1), time=0.)
    world = SimpleNamespace(model=model, data=data, lock=threading.RLock(),
                            placement_history=SimpleNamespace(epoch=1),
                            free_body_names=lambda: ['cube'])
    raw.world = world
    raw._engine = SimpleNamespace(model=model, data=data, lock=world.lock)
    raw._model, raw._data, raw._lock, raw._ctrl = model, data, world.lock, data.ctrl
    raw._qadr = [0]
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.), np.ones(3)))
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.arm, rt.kin, rt.held_object = SafeArm(raw, harness), None, None
    plan = Withdrawal.__new__(Withdrawal)
    plan.runtime, plan.arm, plan.raw, plan.kin, plan.harness = rt, rt.arm, raw, None, harness
    plan.world, plan.model = world, model
    plan.mj = SimpleNamespace(mjtJoint=SimpleNamespace(mjJNT_FREE=0))
    plan._bound_engine, plan._bound_ctrl = raw._engine, raw._ctrl
    plan._bound_data, plan._bound_history = data, world.placement_history
    plan._bound_epoch, plan._postrelease_bound = 1, True
    plan.generation, plan.cancellation = harness._halt_generation, harness._observation_cancel_generation
    plan.deadline = time.monotonic() + 10.
    plan.q, plan.q_start, plan.target = np.array([.1]), np.zeros(1), np.eye(4)
    plan.envelope, plan.duration = (np.zeros(3), np.ones(3)), 3.
    plan.retain()
    return plan


def stop_reset(plan):
    plan.harness.estop('synthetic stop/reset at tested boundary')
    plan.harness.reset_estop()


def assert_retained(plan):
    assert plan.harness._pending_model_withdrawal is plan
    assert plan.runtime._mujoco_withdrawals[id(plan.arm)] is plan
    assert getattr(plan.harness._model_withdrawal_scope, 'value', None) is None


@pytest.mark.parametrize('explicit_generation', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
def test_completion_checks_token_even_with_explicit_generation(debt, explicit_generation, cancel):
    kwargs = {'generation': debt.generation} if explicit_generation else {}
    if cancel:
        stop_reset(debt)
        with pytest.raises((SkillError, SafetyViolation), match='cancelled'):
            debt.complete(**kwargs)
        assert_retained(debt)
    else:
        debt.complete(**kwargs)
        assert debt.harness._pending_model_withdrawal is None
        assert id(debt.arm) not in debt.runtime._mujoco_withdrawals


def recovery_geometry(monkeypatch):
    # Only physical geometry is synthetic; copied-context guards and runtime
    # reset verification/completion run unchanged.
    def snapshot(self):
        self.guard()
        return SimpleNamespace(qpos=self.world.data.qpos.copy())
    monkeypatch.setattr(Withdrawal, '_snapshot', snapshot)
    monkeypatch.setattr(Withdrawal, '_pose', lambda self, q: self.guard())
    monkeypatch.setattr(Withdrawal, '_clear', lambda self: None)
    monkeypatch.setattr(Withdrawal, '_require_open', lambda self: None)
    monkeypatch.setattr(Withdrawal, '_release_envelope',
                        lambda self: setattr(self, 'envelope', (np.zeros(3), np.ones(3))))
    monkeypatch.setattr(Withdrawal, '_home_route', lambda self, start: [])


def test_explicit_recovery_returns_fresh_token_without_erasing_or_renewing_debt(debt, monkeypatch):
    recovery_geometry(monkeypatch)
    old_token = debt.cancellation
    stop_reset(debt)
    result = debt.recover_for_reset()
    assert result['cancellation'] == debt.harness._observation_cancel_generation != old_token
    assert result['pending_retained_until_scene_reset']
    assert debt.cancellation == old_token
    assert_retained(debt)
    with pytest.raises(SkillError, match='cancelled'):
        debt.complete(generation=result['generation'])


@pytest.mark.parametrize('old_cancelled', [False, True, 'halt'])
@pytest.mark.parametrize('cancel_at', [None, 'home', 'reset_props', 'verify_names',
                                       'verify_return', 'beliefs', 'memory_reset', 'memory_add',
                                       'capture', 'depth', 'describe', 'complete'])
def test_reset_keeps_one_fresh_context_through_verification_and_completion(
        debt, monkeypatch, old_cancelled, cancel_at):
    recovery_geometry(monkeypatch)
    rt, events = debt.runtime, []
    if old_cancelled == 'halt':
        debt.harness.halt('previous interrupted operation')
        debt.harness.clear_halt()
    elif old_cancelled:
        stop_reset(debt)
    fresh_token = debt.harness._observation_cancel_generation

    def boundary(name):
        events.append(name)
        if cancel_at == name:
            stop_reset(debt)

    home = Withdrawal.home
    def finish_home(self):
        result = home(self)
        assert_retained(debt)
        boundary('home')
        return result
    monkeypatch.setattr(Withdrawal, 'home', finish_home)

    def reset_props():
        assert_retained(debt)
        debt.world.data.qpos[:] = debt.model.qpos0
        debt.world.data.qvel[:] = 0.
        debt.world.placement_history.epoch += 1  # Authorized reset changes epoch.
        boundary('reset_props')
        return ['cube']
    debt.world.reset_props = reset_props
    def names():
        boundary('verify_names')
        return ['cube']
    debt.world.free_body_names = names
    verify = debt.verify_reset
    def verify_reset(*args, **kwargs):
        result = verify(*args, **kwargs)
        boundary('verify_return')
        return result
    monkeypatch.setattr(debt, 'verify_reset', verify_reset)

    rt.beliefs = SimpleNamespace(clear=lambda: (boundary('beliefs') or 2))
    rt.memory = SimpleNamespace(reset_frames=lambda: boundary('memory_reset'),
                                add=lambda *args: boundary('memory_add'))
    camera, floor, frame = SimpleNamespace(name='cpu'), SimpleNamespace(t=1.), SimpleNamespace(t=2.)
    def capture(**kwargs):
        assert_retained(debt)
        boundary('capture')
        return [(camera, floor, frame)]
    rt._reset_camera_frames = capture
    def depth(frame):
        boundary('depth')
        return frame
    rt.depth = SimpleNamespace(ensure_depth=depth)
    def describe(frame):
        boundary('describe')
        return {'objects_visible': []}
    rt._describe_observation = describe
    complete = debt.complete
    def completion(**kwargs):
        boundary('complete')
        return complete(**kwargs)
    monkeypatch.setattr(debt, 'complete', completion)

    result = rt.skill_reset_scene()
    if cancel_at is None:
        assert result['ok'], result
        assert result['withdrawal_recovery']['cancellation'] == fresh_token
        assert result['withdrawal_reset_verification']['at_model_spawn']
        assert result['observation_refreshed']
        assert debt.harness._pending_model_withdrawal is None
        assert id(debt.arm) not in rt._mujoco_withdrawals
    else:
        assert not result['ok'], result
        assert 'cancelled' in result['error'], result
        assert_retained(debt)
        later = {'home': 'reset_props', 'reset_props': 'verify_names',
                 'verify_names': 'verify_return', 'verify_return': 'beliefs',
                 'beliefs': 'memory_reset', 'memory_reset': 'memory_add', 'memory_add': 'capture',
                 'capture': 'depth', 'depth': 'describe', 'describe': 'complete'}
        if cancel_at in later:
            assert later[cancel_at] not in events, events
        # Report irreversible work already performed before cancellation.
        assert result.get('beliefs_forgotten', 0) == (2 if 'beliefs' in events else 0)


@pytest.mark.parametrize('operation', ['home', 'withdraw', 'recover_for_reset'])
@pytest.mark.parametrize('boundary', ['healthy', 'approval', 'return'])
def test_joint_consumers_fence_stop_reset_at_approval_and_return(debt, monkeypatch, operation, boundary):
    from types import MethodType
    from cascade.control.arm_base import ArmBase
    from cascade.sim.mujoco_placement import state_digest

    recovery_geometry(monkeypatch)
    class Geometry:
        joint_limits = (np.full(1, -1.), np.ones(1))
        def fk(self, q):
            pose = np.eye(4)
            pose[:3, 3] = [.2, 0., .2]
            return pose
        def link_positions(self, q):
            return np.array([[0., 0., .02], [.2, 0., .2]])
    debt.kin = debt.runtime.kin = debt.harness.kin = Geometry()
    monkeypatch.setattr(Withdrawal, '_home_route', lambda self, start: [np.array([.1])])
    monkeypatch.setattr(Withdrawal, '_geometry_segment', lambda *args: None)
    monkeypatch.setattr(Withdrawal, '_escape_segment', lambda *args: None)
    monkeypatch.setattr(Withdrawal, '_check_model_identity', lambda self: None)
    def replan(self):
        self.guard()
        self._postrelease_state = state_digest(self.world.data)
        self._postrelease_data = self.world.data
    monkeypatch.setattr(Withdrawal, '_replan_after_open', replan)
    if operation == 'recover_for_reset':
        stop_reset(debt)  # Explicit recovery must not inherit the old token.

    original_approve = debt.harness.approve
    in_live_approval = False
    def neighbor(q):
        if in_live_approval and boundary == 'approval':
            stop_reset(debt)
    def approve(*args, **kwargs):
        nonlocal in_live_approval
        in_live_approval = True
        try:
            return original_approve(*args, **kwargs)
        finally:
            in_live_approval = False
    monkeypatch.setattr(debt.harness, '_neighbor_violation', neighbor)
    monkeypatch.setattr(debt.harness, 'approve', approve)
    monkeypatch.setattr(debt.raw, 'stream_to', MethodType(ArmBase.stream_to, debt.raw))
    sent = []
    class BoundaryReached(Exception):
        pass
    def intercept(q):
        sent.append(q.copy())
        raise BoundaryReached('intercepted before device write')
    monkeypatch.setattr(debt.raw, 'send_joint_target', intercept)
    if boundary == 'return':
        def cancelled_return(*args, **kwargs):
            stop_reset(debt)
            return True
        monkeypatch.setattr(debt.raw, 'stream_to', cancelled_return)
    debt.harness.heartbeat()
    action = getattr(debt, operation)
    if boundary == 'healthy':
        with pytest.raises(BoundaryReached):
            action()
        assert len(sent) == 1
    else:
        with pytest.raises((SkillError, SafetyViolation), match='cancelled'):
            action()
        assert not sent
    assert_retained(debt)


@pytest.mark.parametrize('failure', [None, 'names', 'qpos', 'qvel', 'stop_before', 'stop_during'])
def test_fresh_reset_context_still_requires_uncancelled_physical_verification(debt, monkeypatch, failure):
    recovery_geometry(monkeypatch)
    stop_reset(debt)
    context = debt.recover_for_reset()
    debt.world.placement_history.epoch += 1
    names = ['cube']
    if failure == 'names':
        names = []
    elif failure == 'qpos':
        debt.world.data.qpos[1] += .1
    elif failure == 'qvel':
        debt.world.data.qvel[1] += .1
    elif failure == 'stop_before':
        stop_reset(debt)
    elif failure == 'stop_during':
        def names_after_stop():
            stop_reset(debt)
            return ['cube']
        debt.world.free_body_names = names_after_stop
    def verify():
        return debt.verify_reset(names, context['generation'], cancellation=context['cancellation'])
    if failure is None:
        proof = verify()
        assert proof['at_model_spawn'] and proof['zero_free_body_velocity']
    else:
        with pytest.raises((SkillError, SafetyViolation)):
            verify()
    assert_retained(debt)  # Even successful verification alone does not clear debt.
