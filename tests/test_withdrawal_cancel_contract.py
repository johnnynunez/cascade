"""Reviewer-only CPU controls: real fence/consumer, synthetic geometry.

No robot assets, solver, renderer, device writes or supported-runtime edits.
The geometry fixtures represent an already admitted plan; they are not a
physical task certificate. All authority checks, SafeArm and streaming run
from the exact frozen head. Device delivery is intercepted before any write.
"""
import time
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.mock_arm import MockArm
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.skills.mujoco_withdrawal import Withdrawal
from cascade.types import SafetyViolation, SkillError


def bound_debt():
    raw = MockArm(Cfg({'n_joints': 1, 'home_q': [0.]}))
    model = SimpleNamespace(nu=1)
    data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(1), time=0.)
    world = SimpleNamespace(model=model, data=data, lock=threading.RLock())
    raw.world = world
    raw._engine = SimpleNamespace(model=model, data=data, lock=world.lock)
    raw._model, raw._data, raw._lock, raw._ctrl = model, data, world.lock, np.zeros(1)
    raw._qadr = [0]
    harness = SafetyHarness(SafetyLimits(np.full(3, -1.), np.ones(3)))
    arm = SafeArm(raw, harness)
    runtime = SimpleNamespace(arm=arm, kin=None, held_object=None)
    plan = Withdrawal.__new__(Withdrawal)
    plan.runtime, plan.arm, plan.raw, plan.kin, plan.harness = runtime, arm, raw, None, harness
    plan.world, plan.model = world, model
    plan._bound_engine, plan._bound_ctrl = raw._engine, raw._ctrl
    plan.generation, plan.cancellation = harness._halt_generation, harness._observation_cancel_generation
    plan.deadline = time.monotonic() + 10.
    plan.retain()  # Real binding, generation and single-owner retention checks.
    return plan


@pytest.mark.parametrize('cancel', [False, True])
def test_explicit_reset_completion_cannot_erase_debt_after_stop_reset(cancel):
    plan = bound_debt()
    generation_from_reset_recovery = plan.harness._halt_generation
    if cancel:
        # Represents stop/reset after recovery/final camera read but before
        # skill_reset_scene calls complete(generation=recovery_generation).
        plan.harness.estop('cancel after final observation')
        plan.harness.reset_estop()
        assert plan.harness._halt_generation == generation_from_reset_recovery
        assert plan.harness._observation_cancel_generation != plan.cancellation
        with pytest.raises((SafetyViolation, SkillError)):
            plan.complete(generation=generation_from_reset_recovery)
        assert plan.harness._pending_model_withdrawal is plan
    else:
        plan.complete(generation=generation_from_reset_recovery)
        assert plan.harness._pending_model_withdrawal is None


def test_normal_completion_does_reject_same_stop_reset():
    plan = bound_debt()
    plan.harness.estop('same cancellation')
    plan.harness.reset_estop()
    with pytest.raises(SkillError, match='cancelled'):
        plan.complete()
    assert plan.harness._pending_model_withdrawal is plan


@pytest.mark.parametrize('case', ['healthy', 'cancel_during_approval', 'token_control'])
@pytest.mark.parametrize('operation', ['home', 'withdraw'])
def test_motion_must_recheck_cancellation_after_waypoint_approval(monkeypatch, case, operation):
    plan = bound_debt()
    goal = np.array([.1])
    # Isolate the transmission path from geometry availability. These are
    # synthetic preflight inputs, not claimed physical measurements.
    monkeypatch.setattr(plan, '_snapshot', lambda: SimpleNamespace(qpos=np.zeros(1)))
    monkeypatch.setattr(plan, '_home_route', lambda start: [goal.copy()])
    monkeypatch.setattr(plan, '_require_open', lambda: None)
    monkeypatch.setattr(plan, '_geometry_segment', lambda *a: None)
    monkeypatch.setattr(plan, 'after_withdrawal', lambda: None)
    if operation == 'withdraw':
        from cascade.sim.mujoco_placement import state_digest
        plan.q, plan.q_start, plan.duration = goal.copy(), np.zeros(1), 3.
        plan._postrelease_data = plan.world.data
        plan._postrelease_state = state_digest(plan.world.data)
        monkeypatch.setattr(plan, '_replan_after_open', lambda: None)
        monkeypatch.setattr(plan, '_check_model_identity', lambda: None)
        monkeypatch.setattr(plan, '_escape_segment', lambda *args: None)
        monkeypatch.setattr(plan, '_clear', lambda: None)
    # MuJoCo inherits ArmBase.stream_to; use that exact streamer instead of
    # MockArm's no-sleep override. First delivery is intercepted, so no dwell.
    from types import MethodType
    from cascade.control.arm_base import ArmBase
    monkeypatch.setattr(plan.raw, 'stream_to', MethodType(ArmBase.stream_to, plan.raw))
    class Geometry:
        joint_limits = (np.full(1, -1.), np.ones(1))
        def fk(self, q):
            pose = np.eye(4)
            pose[:3, 3] = [.2, 0., .2]
            return pose
        def link_positions(self, q):
            return np.array([[0., 0., .02], [.2, 0., .2]])
    kin = Geometry()
    plan.kin = plan.runtime.kin = plan.harness.kin = kin
    original_approve = plan.harness.approve
    in_live_approval = False
    def late_geometry_check(q):
        # Cancellation occurs INSIDE the real approval, after its pending
        # scope/token check, not after an already completed device call.
        if in_live_approval and case != 'healthy':
            plan.harness.estop('stop/reset during waypoint geometry check')
            plan.harness.reset_estop()
        return None
    def approve(*args, **kwargs):
        nonlocal in_live_approval
        in_live_approval = True
        try:
            return original_approve(*args, **kwargs)
        finally:
            in_live_approval = False
    monkeypatch.setattr(plan.harness, '_neighbor_violation', late_geometry_check)
    monkeypatch.setattr(plan.harness, 'approve', approve)
    if case == 'token_control':
        # Causal control only: supply the already-existing consumer token
        # that production Withdrawal.home/withdraw currently do not pass.
        original_move = plan.arm.move_joints
        def with_token(*args, **kwargs):
            kwargs.setdefault('_cancellation_token', plan.cancellation)
            return original_move(*args, **kwargs)
        monkeypatch.setattr(plan.arm, 'move_joints', with_token)
    sent = []
    class BoundaryReached(Exception):
        pass
    def no_write(q):
        sent.append(q.tolist())
        raise BoundaryReached('device boundary reached; native write intercepted')
    monkeypatch.setattr(plan.raw, 'send_joint_target', no_write)
    plan.harness.heartbeat()
    try:
        getattr(plan, operation)()  # Real operation -> scoped SafeArm -> shared streamer.
    except (BoundaryReached, SafetyViolation, SkillError):
        pass
    assert len(sent) == (1 if case == 'healthy' else 0), {'case': case, 'sent': sent,
        'original_token': plan.cancellation, 'current_token': plan.harness._observation_cancel_generation}
    assert plan.harness._pending_model_withdrawal is plan


def recover_context(plan, monkeypatch):
    """Real context renewal with synthetic, stationary geometry only."""
    plan.q = plan.q_start = np.zeros(1)
    plan.target = np.eye(4)
    plan.envelope = (np.zeros(3), np.ones(3))
    monkeypatch.setattr(Withdrawal, '_snapshot', lambda self: self.world.data)
    monkeypatch.setattr(Withdrawal, '_require_open', lambda self: None)
    monkeypatch.setattr(Withdrawal, '_pose', lambda self, q: None)
    monkeypatch.setattr(Withdrawal, '_release_envelope', lambda self: None)
    monkeypatch.setattr(Withdrawal, 'home', lambda self: {'at': 'home'})
    return plan.recover_for_reset()


@pytest.mark.parametrize('stop_after', [False, True])
def test_explicit_recovery_uses_its_own_token_without_reviving_original(monkeypatch, stop_after):
    plan = bound_debt()
    plan.harness.estop('invalidate original plan')
    plan.harness.reset_estop()
    result = recover_context(plan, monkeypatch)
    assert result['cancellation'] != plan.cancellation
    with pytest.raises(SkillError, match='cancelled'):
        plan.complete()
    if stop_after:
        plan.harness.estop('cancel recovery too')
        plan.harness.reset_estop()
        with pytest.raises(SkillError, match='cancelled'):
            plan.complete(generation=result['generation'], cancellation=result['cancellation'])
        assert plan.harness._pending_model_withdrawal is plan
        assert plan.runtime._mujoco_withdrawals[id(plan.arm)] is plan
    else:
        plan.complete(generation=result['generation'], cancellation=result['cancellation'])
        assert plan.harness._pending_model_withdrawal is None


def test_current_token_alone_cannot_masquerade_as_completed_recovery():
    plan = bound_debt()
    plan.harness.estop('original invalid')
    plan.harness.reset_estop()
    with pytest.raises(SkillError, match='not recovered'):
        plan.complete(generation=plan.generation,
                      cancellation=plan.harness._observation_cancel_generation)
    assert plan.harness._pending_model_withdrawal is plan


@pytest.mark.parametrize('recovered', [False, True])
def test_explicit_original_context_does_not_bypass_postrelease_epoch(monkeypatch, recovered):
    plan = bound_debt()
    plan._postrelease_bound = True
    plan._bound_data = plan.world.data
    plan._bound_history = plan.world.placement_history = SimpleNamespace(epoch='before')
    plan._bound_epoch = 'before'
    if recovered:
        result = recover_context(plan, monkeypatch)
    plan.world.placement_history.epoch = 'after'
    if recovered:
        plan.complete(generation=result['generation'], cancellation=result['cancellation'])
        assert plan.harness._pending_model_withdrawal is None
    else:
        with pytest.raises(SkillError, match='epoch binding changed'):
            plan.complete(generation=plan.generation, cancellation=plan.cancellation)
        assert plan.harness._pending_model_withdrawal is plan


@pytest.mark.parametrize('cancel_at', ['none', 'before', 'during'])
def test_reset_verification_checks_token_before_and_after_physical_read(monkeypatch, cancel_at):
    plan = bound_debt()
    result = recover_context(plan, monkeypatch)
    plan.model.njnt = 0  # No native solver: this test isolates context/read ordering.
    def names():
        if cancel_at == 'during':
            plan.harness.estop('cancel while reading spawn inventory')
            plan.harness.reset_estop()
        return ['prop']
    plan.world.free_body_names = names
    if cancel_at == 'before':
        plan.harness.estop('cancel before spawn read')
        plan.harness.reset_estop()
    if cancel_at == 'none':
        assert plan.verify_reset(['prop'], result['generation'], result['cancellation'])['at_model_spawn']
    else:
        with pytest.raises(SkillError, match='cancelled'):
            plan.verify_reset(['prop'], result['generation'], result['cancellation'])
    assert plan.harness._pending_model_withdrawal is plan


@pytest.mark.parametrize('cancel_at', ['none', 'observation', 'completion'])
def test_runtime_reset_keeps_debt_when_cancelled_after_recovery(monkeypatch, cancel_at):
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import Frame
    plan = bound_debt()
    rt = SkillRuntime.__new__(SkillRuntime)
    for name, value in vars(plan.runtime).items():
        setattr(rt, name, value)
    plan.runtime = rt
    rt.beliefs = SimpleNamespace(clear=lambda: 0)
    rt.memory = SimpleNamespace(reset_frames=lambda: None, add=lambda *args: None)
    rt.depth = SimpleNamespace(ensure_depth=lambda frame: frame)
    rt._carry_attachment = None
    frame = Frame(np.zeros((1, 1, 3), dtype=np.uint8), None, np.eye(3))
    rt._reset_camera_frames = lambda: [(SimpleNamespace(name='synthetic'), frame, frame)]
    def describe(frame):
        if cancel_at == 'observation':
            plan.harness.estop('stop/reset during final observation')
            plan.harness.reset_estop()
        return {'objects_visible': []}
    rt._describe_observation = describe
    plan.world.reset_props = lambda: ['prop']
    plan.world.free_body_names = lambda: ['prop']
    plan.model.njnt = 0
    plan.harness.estop('original context cancelled')
    plan.harness.reset_estop()
    recover_context(plan, monkeypatch)
    if cancel_at == 'completion':
        original_complete = plan.complete
        def complete(**kwargs):
            plan.harness.estop('stop/reset just before atomic completion')
            plan.harness.reset_estop()
            return original_complete(**kwargs)
        monkeypatch.setattr(plan, 'complete', complete)
    result = rt.skill_reset_scene()
    assert result['ok'] is (cancel_at == 'none'), result
    if cancel_at != 'none':
        assert plan.harness._pending_model_withdrawal is plan
        assert rt._mujoco_withdrawals[id(plan.arm)] is plan
    else:
        assert plan.harness._pending_model_withdrawal is None
        assert id(plan.arm) not in rt._mujoco_withdrawals
