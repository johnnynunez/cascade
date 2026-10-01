"""The opt-in gate must bind the real stream and prevent unguarded recovery."""
from types import SimpleNamespace
import numpy as np
import pytest

from cascade.types import SafetyViolation, SkillError, MotionHalted
from cascade.safety.harness import SafetyHarness, SafetyLimits
from test_grasp_evidence import runtime as basic_runtime
from test_isaac_simulation_motion import Sim, stream


def runtime(monkeypatch):
    result = basic_runtime(monkeypatch)
    harness = result[0].arm.harness
    harness._halt_generation = 0
    harness._check_halt_generation = lambda expected: None
    harness.begin_motion = lambda **kwargs: None
    return result


def fake_gate(monkeypatch, *, conflict=None):
    checks = []
    gate = SimpleNamespace(profile=lambda *a, **kw: conflict,
        require_profile=lambda *a, **kw: checks.append(('preflight', a[0].copy(), a[1].copy())),
        pose=lambda *args: conflict,
        feedback=lambda s: checks.append(('feedback', s.q.copy())),
        closing_pose=lambda q: None, occluded_pose=lambda q, **kw: None,
        require_closing=lambda s, **kw: checks.append(('close_preflight', s.q.copy())),
        closing_lower=np.zeros(2), closing_upper=np.ones(2)*.05)
    monkeypatch.setenv('CASCADE_OBSERVED_FINGER_GATE', '1')
    monkeypatch.setattr('cascade.grasping.observed_scene.for_runtime', lambda *args: gate)
    return checks


def test_home_collision_prevents_even_open_command(monkeypatch):
    fake_gate(monkeypatch, conflict={'surface': 'neighbor'})
    rt, calls, fix, frame = runtime(monkeypatch)
    with pytest.raises(SafetyViolation, match='home finger trajectory'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert not [c for c in calls if c[0] != 'read']


@pytest.mark.parametrize('failure', [False, RuntimeError('home transport failure')])
def test_home_failure_is_required_and_no_pregrasp_or_close_follows(monkeypatch, failure):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    moves = []
    def fail_home(*a, **kw):
        moves.append(a)
        if isinstance(failure, Exception): raise failure
        return failure
    monkeypatch.setattr(rt.arm, 'move_joints', fail_home)
    with pytest.raises((RuntimeError, SkillError), match='home'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert len(moves) == 1
    assert [c for c in calls if c[0] == 'gripper'] == [('gripper', 1., .8)]


def test_actual_stream_preflights_and_feedback_guard_stop_before_close(monkeypatch):
    checks = fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    result = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert result['held'] == 'orange' and result['grip_verified']
    assert len([c for c in checks if c[0] == 'preflight']) == 3  # home, pre, descent; no close/lift
    assert len([c for c in checks if c[0] == 'feedback']) > 3
    assert len([c for c in checks if c[0] == 'close_preflight']) == 2


def test_feedback_typeerror_never_retries_motion_without_callbacks(monkeypatch):
    rt, calls, fix, frame = runtime(monkeypatch)
    seen = []
    def broken(state):
        seen.append(state)
        raise TypeError('callback defect')
    with pytest.raises(SafetyViolation, match='no unguarded retry'):
        rt.arm.move_joints(np.array([.2, 0., .1]), .1, feedback_guard=broken)
    assert len(seen) == 1
    assert not [c for c in calls if c[0] == 'joints']


def test_feedback_guard_adds_no_reads_and_preserves_post_ack_pacing(monkeypatch):
    sim = Sim(monkeypatch); sim.send_steps = 100
    assert stream(sim)
    expected_reads, expected_sends = len(sim.reads), [(s, q.tolist()) for _,s,q in sim.sent]
    sim = Sim(monkeypatch); sim.send_steps = 100
    checks = []
    assert stream(sim, feedback_guard=lambda state: checks.append(state.physics_clock['physics_step']))
    assert len(sim.reads) == len(checks) == expected_reads
    assert [(s, q.tolist()) for _,s,q in sim.sent] == expected_sends


def test_feedback_guard_failure_after_ack_prevents_next_target(monkeypatch):
    sim = Sim(monkeypatch)
    def guard(state):
        if sim.sent: raise SafetyViolation('jaw left validated interval')
    with pytest.raises(SafetyViolation, match='jaw left'):
        stream(sim, feedback_guard=guard)
    assert len(sim.sent) == 1


def real_harness(rt):
    rt.arm.harness = SafetyHarness(SafetyLimits(np.array([-2., -2., -2.]), np.ones(3)*2,
                                               table_z=0., max_joint_vel=10., watchdog_s=10000.))
    return rt.arm.harness


@pytest.mark.parametrize('moment', ['planning', 'selected', 'after_descent'])
def test_halt_generation_survives_planning_and_prevents_later_actuation(monkeypatch, moment):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    h = real_harness(rt)
    if moment == 'planning':
        plan = rt._plan_grasps
        def halted_plan(*a, **kw):
            result = plan(*a, **kw); h.halt('new instruction'); return result
        rt._plan_grasps = halted_plan
    elif moment == 'selected':
        from cascade.skills import runtime as module
        select = module.select_grasp
        def selected(*a, **kw):
            result = select(*a, **kw); h.halt('new instruction'); return result
        monkeypatch.setattr(module, 'select_grasp', selected)
    else:
        move = rt.arm.move_joints
        count = 0
        def moved(*a, **kw):
            nonlocal count
            result = move(*a, **kw); count += 1
            if count == 3: h.halt('new instruction')
            return result
        monkeypatch.setattr(rt.arm, 'move_joints', moved)
    with pytest.raises(MotionHalted):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    if moment == 'after_descent':
        assert [c for c in calls if c[0] == 'gripper'] == [('gripper', 1., .8)]
    else:
        assert not [c for c in calls if c[0] != 'read']


def test_actual_pose_is_checked_before_first_open_after_long_selection(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    from cascade.skills import runtime as module
    select = module.select_grasp
    def selected(*a, **kw):
        result = select(*a, **kw)
        rt.arm.raw._client.q = np.array([.6, 0., .2])
        return result
    monkeypatch.setattr(module, 'select_grasp', selected)
    gate = SimpleNamespace(profile=lambda *a, **k: None,
        feedback=lambda state: None, closing_pose=lambda q: None, occluded_pose=lambda q, **kw: None,
        pose=lambda q: {'surface': 'neighbor'} if q[0] > .5 else None)
    monkeypatch.setattr('cascade.grasping.observed_scene.for_runtime', lambda *a: gate)
    with pytest.raises(SafetyViolation, match='opening fingers'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert not [c for c in calls if c[0] != 'read']


def test_candidate_closure_conflict_prevents_all_actuation(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    from cascade.grasping import observed_scene
    factory = observed_scene.for_runtime
    def make(*args):
        gate = factory(*args)
        gate.closing_pose = lambda q: {'surface': 'other observed surface'}
        return gate
    monkeypatch.setattr(observed_scene, 'for_runtime', make)
    with pytest.raises(SkillError, match='closing fingers'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert not [c for c in calls if c[0] != 'read']


def test_endpoint_occlusion_rejects_before_any_actuation(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    from cascade.grasping import observed_scene
    factory = observed_scene.for_runtime
    def make(*args):
        gate = factory(*args)
        gate.occluded_pose = lambda *a, **kw: {'surface': 'occluded behind non-target depth'}
        return gate
    monkeypatch.setattr(observed_scene, 'for_runtime', make)
    with pytest.raises(SkillError, match='endpoint occluded'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert all(c[0] == 'read' for c in calls)


@pytest.mark.parametrize('task_budget', [False, True])
def test_preclose_occlusion_deadline_prevents_jaw_command(monkeypatch, task_budget):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    from cascade.grasping import observed_scene
    from cascade.skills import runtime as module
    factory = observed_scene.for_runtime
    clock_offset = [0.]
    # Replace the module reference, not global time used by other threads.
    original_time = module.time
    monotonic = lambda: original_time.monotonic() + clock_offset[0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=monotonic,
                                                      sleep=original_time.sleep))
    if task_budget:
        move = rt.arm.move_joints
        moves = []
        def moved(*args, **kwargs):
            result = move(*args, **kwargs); moves.append(1)
            if len(moves) == 3: rt._task_deadline = monotonic() + .25
            return result
        monkeypatch.setattr(rt.arm, 'move_joints', moved)
    def make(*args):
        gate = factory(*args)
        def close(state, *, check):
            check()
            from cascade.safety.trajectory import PLAN_BUDGET_S
            clock_offset[0] += .3 if task_budget else PLAN_BUDGET_S + .1
            check()
        gate.require_closing = close
        return gate
    monkeypatch.setattr(observed_scene, 'for_runtime', make)
    with pytest.raises(SafetyViolation, match='closing preflight exceeded'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert [c for c in calls if c[0]=='gripper'] == [('gripper', 1., .8)]
    assert getattr(rt, '_held_provisional', None) is None


@pytest.mark.parametrize('fail_stage', [1, 2])
@pytest.mark.parametrize('failure', ['collision', 'halt'])
def test_each_close_stage_rebinds_and_cancels_before_its_command(monkeypatch, fail_stage, failure):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    harness = real_harness(rt)
    from cascade.grasping import observed_scene
    factory = observed_scene.for_runtime
    guarded = []
    def make(*args):
        gate = factory(*args)
        def close(state, **kwargs):
            guarded.append(state)
            if len(guarded) == fail_stage:
                if failure == 'halt':
                    harness.halt('cancelled during closing geometry')
                else:
                    raise SafetyViolation('closing fingers intersects neighbor')
        gate.require_closing = close
        return gate
    monkeypatch.setattr(observed_scene, 'for_runtime', make)
    with pytest.raises((SafetyViolation, MotionHalted)):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert len(guarded) == fail_stage
    assert len([c for c in calls if c[0] == 'gripper']) == fail_stage  # open + preceding close
    if fail_stage == 1:
        assert getattr(rt, '_held_provisional', None) is None
        rt._reconcile_held()
        assert rt.held_object is None  # open jaws are not an invented hold
    else:
        assert rt._held_provisional is not None
    # No joint target after the first attempted closing-stage preflight.
    first_close_read = next(i for i,c in enumerate(calls) if c[0]=='read' and c[2] < 1.) if fail_stage==2 else len(calls)
    assert not [c for c in calls[first_close_read:] if c[0]=='joints']


def test_drift_after_descent_is_read_before_closing(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    move = rt.arm.move_joints
    moves = []
    def drift(*args, **kwargs):
        result = move(*args, **kwargs); moves.append(1)
        if len(moves) == 3: rt.arm.raw._client.q[0] = .6
        return result
    monkeypatch.setattr(rt.arm, 'move_joints', drift)
    from cascade.grasping import observed_scene
    factory = observed_scene.for_runtime
    def make(*args):
        gate = factory(*args)
        def close(state, **kwargs):
            assert state.q[0] == .6
            raise SafetyViolation('measured closing conflict')
        gate.require_closing = close
        return gate
    monkeypatch.setattr(observed_scene, 'for_runtime', make)
    with pytest.raises(SafetyViolation, match='measured closing'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert [c for c in calls if c[0]=='gripper'] == [('gripper', 1., .8)]


def test_cancel_inside_candidate_closing_vet_prevents_even_open(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    harness = real_harness(rt)
    from cascade.grasping import observed_scene
    factory = observed_scene.for_runtime
    def make(*args):
        gate = factory(*args)
        gate.closing_pose = lambda q: harness.halt('cancelled during closing selection')
        return gate
    monkeypatch.setattr(observed_scene, 'for_runtime', make)
    with pytest.raises(MotionHalted):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert not [c for c in calls if c[0] != 'read']


def test_bulk_close_cannot_bypass_individual_stage_guard(monkeypatch):
    rt, calls, _, _ = runtime(monkeypatch)
    rt.arm.raw.close_gripper_two_stage = lambda **kw: calls.append(('bulk_close',))
    with pytest.raises(SafetyViolation, match='individually guarded'):
        rt._close_two_stage(SimpleNamespace(), _before_close=lambda: None)
    assert not [c for c in calls if c[0] != 'read']


@pytest.mark.parametrize('composite', ['skill_pick_and_place', '_grasp_with_persistence'])
@pytest.mark.parametrize('failure', ['exception', 'air_grasp'])
def test_opt_in_failure_has_no_unguarded_rehome_or_retry(monkeypatch, composite, failure):
    monkeypatch.setenv('CASCADE_OBSERVED_FINGER_GATE', '1')
    rt, calls, fix, frame = runtime(monkeypatch)
    attempts = []
    def fail(*a, **kw):
        attempts.append(1)
        if failure == 'exception': raise SafetyViolation('observed neighbor conflict')
        return {'ok': False, 'error': 'air grasp'}
    rt.skill_grasp_object = fail
    rt.skill_move_home = lambda: pytest.fail('unguarded recovery')
    result = getattr(rt, composite)('orange')
    assert result['ok'] is False and result['home_skipped'] is True and len(attempts) == 1
    assert not [c for c in calls if c[0] != 'read']
