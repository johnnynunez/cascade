"""Bounded replanning is strictly before actuation and only for feasibility."""
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.grasping.graspgenx_backend import GraspGenXError, NoEligibleGrasps
from cascade.grasping.planning_budget import PlanningBudget, PlanningBudgetExceeded
from cascade.grasping.selector import NoExecutableGrasp
from cascade.types import MotionHalted, SafetyViolation
from test_observed_finger_runtime import runtime, fake_gate, real_harness


def setup(monkeypatch):
    checks = fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    cfg = rt.cfg.as_dict()
    cfg['grasp'].update(backend='graspgenx', graspgenx={
        'feasibility_max_batches': 3, 'feasibility_timeout_s': 8.})
    rt.cfg = Cfg(cfg)
    prior = {'win_features': {}, 'nudges': {'grasp_z_delta': -.005}}
    records, reads = [], []
    def read_prior(*a):
        reads.append(1)
        return prior
    rt.grasp_memory = SimpleNamespace(prior=read_prior, record=lambda *a, **kw: records.append(kw))
    h = real_harness(rt)
    original = rt._plan_grasps
    batches, options = [], []
    def generate(f, label, **kwargs):
        assert f is fix
        assert not [c for c in calls if c[0] != 'read']
        options.append(kwargs)
        batches.append(len(batches) + 1)
        return copy.deepcopy(original(f, label=label))
    rt._plan_grasps = generate
    return rt, calls, fix, frame, h, batches, options, prior, reads, records, checks


def test_two_batches_share_snapshot_prior_deadline_and_do_not_act_between_them(monkeypatch):
    rt, calls, fix, frame, h, batches, opts, prior, reads, records, _ = setup(monkeypatch)
    def vet(*a, **kw):
        if len(batches) == 1:
            prior['nudges']['grasp_z_delta'] = .123  # concurrent outcome update
            return 'blocked original proposal'
        return None
    h.vet_pose = vet
    result = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert result['grip_verified']
    assert batches == [1, 2] and len(reads) == 1
    assert opts[0]['_deadline'] == opts[1]['_deadline'] == 8.
    assert opts[0]['_prior_snapshot'] is opts[1]['_prior_snapshot']
    assert opts[1]['_prior_snapshot'][0]['nudges']['grasp_z_delta'] == -.005
    assert [r for r in records if not r['success']] == []
    assert any(c[0] == 'gripper' for c in calls)


def test_three_feasibility_failures_exhaust_without_any_actuation(monkeypatch):
    rt, calls, fix, frame, h, batches, _, _, _, records, _ = setup(monkeypatch)
    h.vet_pose = lambda *a, **kw: 'blocked'
    with pytest.raises(NoExecutableGrasp, match='blocked'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert batches == [1, 2, 3]
    assert all(c[0] == 'read' for c in calls)
    assert len(records) == 1


@pytest.mark.parametrize('error', [GraspGenXError('transport'), RuntimeError('model'),
                                  SafetyViolation('epoch changed'), MotionHalted('stop')])
def test_non_feasibility_errors_are_not_retried(monkeypatch, error):
    rt, calls, fix, frame, _, batches, _, _, _, _, _ = setup(monkeypatch)
    def failed(*a, **kw):
        batches.append(1)
        raise error
    rt._plan_grasps = failed
    with pytest.raises(type(error), match=str(error)):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert batches == [1] and not calls


def test_valid_batch_filtered_to_zero_can_be_followed_by_valid_batch(monkeypatch):
    rt, calls, fix, frame, _, batches, _, _, _, _, _ = setup(monkeypatch)
    plan = rt._plan_grasps
    def generate(*a, **kw):
        gs = plan(*a, **kw)
        if len(batches) == 1:
            raise NoEligibleGrasps('unchanged filters removed all')
        return gs
    rt._plan_grasps = generate
    assert rt.skill_grasp_object('orange', _fix=fix, _frame=frame)['grip_verified']
    assert batches == [1, 2]


@pytest.mark.parametrize('where', ['after_backend', 'after_ik', 'after_vet', 'between_batches'])
def test_halt_at_every_planning_boundary_prevents_commands_and_new_batch(monkeypatch, where):
    rt, calls, fix, frame, h, batches, _, _, _, _, _ = setup(monkeypatch)
    if where == 'after_backend':
        original = rt._plan_grasps
        def run(*a, **kw):
            result = original(*a, **kw); h.halt('cancel'); return result
        rt._plan_grasps = run
    elif where == 'after_ik':
        original = rt.kin.ik
        def run(*a, **kw):
            result = original(*a, **kw); h.halt('cancel'); return result
        rt.kin.ik = run
    else:
        def vet(*a, **kw):
            h.halt('cancel')
            return 'infeasible' if where == 'between_batches' else None
        h.vet_pose = vet
    with pytest.raises(MotionHalted):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert batches == [1] and all(c[0] == 'read' for c in calls)


@pytest.mark.parametrize('where', ['backend', 'ik', 'vet'])
def test_late_results_are_not_used_even_if_the_last_native_call_returns_success(monkeypatch, where):
    rt, calls, fix, frame, h, batches, _, _, _, _, _ = setup(monkeypatch)
    if where == 'backend':
        original = rt._plan_grasps
        def late(*a, **kw):
            result = original(*a, **kw); time.sleep(9); return result
        rt._plan_grasps = late
    elif where == 'ik':
        original = rt.kin.ik
        def late(*a, **kw):
            result = original(*a, **kw); time.sleep(9); return result
        rt.kin.ik = late
    else:
        def late(*a, **kw):
            time.sleep(9); return None
        h.vet_pose = late
    with pytest.raises(PlanningBudgetExceeded):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert batches == [1] and all(c[0] == 'read' for c in calls)


def test_parent_pick_and_task_deadlines_cap_the_same_planning_deadline(monkeypatch):
    rt, calls, fix, frame, _, _, opts, _, _, _, _ = setup(monkeypatch)
    rt._task_deadline = 3.
    rt.skill_grasp_object('orange', _fix=fix, _frame=frame, _planning_deadline=2.)
    assert opts[0]['_deadline'] == 2.


@pytest.mark.parametrize('batches,seconds', [(0, 8), (True, 8), (3, float('nan')), (3, -1)])
def test_invalid_budget_fails_closed(batches, seconds):
    from cascade.types import SkillError
    with pytest.raises(SkillError):
        PlanningBudget(batches, seconds)


def test_between_batch_feedback_failure_never_starts_next_inference(monkeypatch):
    rt, calls, fix, frame, h, batches, _, _, _, _, _ = setup(monkeypatch)
    from cascade.grasping import observed_scene
    gate = observed_scene.for_runtime()
    gate.feedback = lambda state: (_ for _ in ()).throw(SafetyViolation('epoch mismatch'))
    h.vet_pose = lambda *a, **kw: 'infeasible'
    with pytest.raises(SafetyViolation, match='epoch mismatch'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert batches == [1] and all(c[0] == 'read' for c in calls)


def test_budget_includes_final_feedback_and_opening_veto(monkeypatch):
    rt, calls, fix, frame, _, batches, _, _, _, _, _ = setup(monkeypatch)
    from cascade.grasping import observed_scene
    gate = observed_scene.for_runtime()
    def slow_pose(q):
        time.sleep(9)
        return None
    gate.pose = slow_pose
    with pytest.raises(PlanningBudgetExceeded):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert batches == [1] and all(c[0] == 'read' for c in calls)


@pytest.mark.parametrize('disabled', ['gate', 'backend'])
def test_disabled_mode_never_passes_new_planning_kwargs(monkeypatch, disabled):
    rt, calls, fix, frame, _, batches, opts, _, reads, _, _ = setup(monkeypatch)
    if disabled == 'gate':
        monkeypatch.setenv('CASCADE_OBSERVED_FINGER_GATE', '0')
    else:
        cfg = rt.cfg.as_dict(); cfg['grasp']['backend'] = 'obb'; rt.cfg = Cfg(cfg)
    result = rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert result['grip_verified'] and batches == [1]
    assert opts == [{}] and not reads


def test_additional_state_reads_never_enlarge_normal_rpc_timeout(monkeypatch):
    rt, _, fix, frame, h, batches, _, _, _, _, _ = setup(monkeypatch)
    h.vet_pose = lambda *a, **kw: 'blocked' if len(batches) == 1 else None
    budgets = []
    read = rt.arm.get_state
    def bounded(**kwargs):
        if 'timeout_s' in kwargs:
            budgets.append(kwargs['timeout_s'])
        return read(**kwargs)
    rt.arm.get_state = bounded
    rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert len(budgets) == 3  # initial state, between batches, immediately pre-open
    assert all(0 < b <= 1 for b in budgets)


def test_persistence_helper_passes_its_shorter_absolute_deadline(monkeypatch):
    rt, calls, fix, frame, _, batches, opts, _, _, _, _ = setup(monkeypatch)
    rt._localize = lambda *a, **kw: (frame, fix)
    plan = rt._plan_grasps
    def slow(*a, **kw):
        result = plan(*a, **kw)
        time.sleep(.3)
        return result
    rt._plan_grasps = slow
    result = rt._grasp_with_persistence('orange', budget_s=.25)
    assert result['ok'] is False and result['home_skipped'] is True
    assert opts[0]['_deadline'] == .25 and batches == [1]
    assert not calls


def test_halt_after_last_twin_ik_prevents_using_original_success(monkeypatch):
    rt, calls, fix, frame, h, batches, _, _, _, _, _ = setup(monkeypatch)
    solve = rt.kin.ik
    count = []
    def ik(*a, **kw):
        result = solve(*a, **kw)
        count.append(1)
        if len(count) == 4:
            h.halt('cancel after final IK')
        return result
    rt.kin.ik = ik
    with pytest.raises(MotionHalted):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert len(count) == 4 and batches == [1]
    assert all(c[0] == 'read' for c in calls)
