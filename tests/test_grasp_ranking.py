"""A caller's outcome-memory ranking must survive feasibility selection."""
import copy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.grasping.selector import select_grasp, NoExecutableGrasp
from cascade.memory.grasp_memory import GraspOutcomeMemory
from cascade.skills.runtime import SkillRuntime
from cascade.types import Grasp, MotionHalted
from test_observed_finger_runtime import runtime, fake_gate, real_harness


class Kin:
    def ik(self, target, seed):
        return SimpleNamespace(success=True, q=target[:3, 3].copy(), error=0.)


def candidates():
    def grasp(x, yaw, quality):
        return Grasp(np.array([x, 0., .05]), _yaw_rotation(yaw), .05,
                     np.array([0., 0., -1.]), quality, 'object')
    return [grasp(.22, np.pi / 2, .6), grasp(.21, 0., .45), grasp(.20, 0., .5)]


def select(gs, **kwargs):
    return select_grasp(gs, Kin(), np.zeros(3), **kwargs)[0]


def test_default_selector_keeps_quality_order_and_does_not_mutate_input():
    gs = list(reversed(candidates()))
    before = copy.deepcopy(gs)
    assert select(gs).quality == .6
    for current, old in zip(gs, before):
        assert current.quality == old.quality
        np.testing.assert_array_equal(current.position, old.position)
        np.testing.assert_array_equal(current.rotation, old.rotation)


def test_explicit_ranked_order_survives_lower_model_quality_and_veto():
    gs = [candidates()[i] for i in (2, 1, 0)]
    assert select(gs, preserve_order=True).quality == .5
    seen = []
    def vet(g, *args):
        seen.append(g.quality)
        return 'observed obstacle' if g.quality == .5 else None
    assert select(gs, preserve_order=True, validate=vet).quality == .45
    assert seen[0] == .5 and .6 not in seen
    assert [g.quality for g in gs] == [.5, .45, .6]


def test_ranked_order_still_applies_width_filter_and_all_rejected_error():
    gs = [candidates()[i] for i in (2, 1, 0)]
    gs[0].width_m = .1
    assert select(gs, preserve_order=True).quality == .45
    with pytest.raises(NoExecutableGrasp, match='observed obstacle'):
        select(gs, preserve_order=True, validate=lambda *args: 'observed obstacle')


def configured_runtime(monkeypatch):
    fake_gate(monkeypatch)
    rt, calls, fix, frame = runtime(monkeypatch)
    cfg = rt.cfg.as_dict(); cfg['safety'] = {'table_z': 0.}
    rt.cfg = Cfg(cfg)
    rt._plan_grasps = MethodType(SkillRuntime._plan_grasps, rt)
    raw = candidates()
    monkeypatch.setattr('cascade.skills.runtime.plan_grasps_from_fix', lambda *a, **k: copy.deepcopy(raw))
    rt.grasp_memory = GraspOutcomeMemory()
    rt.grasp_memory.record('orange', fix, raw[2], True)
    return rt, calls, fix, frame, raw


@pytest.mark.parametrize('first_blocked', [False, True])
def test_real_runtime_honors_memory_rank_then_next_feasible(monkeypatch, first_blocked):
    rt, calls, fix, frame, raw = configured_runtime(monkeypatch)
    h = real_harness(rt)
    if first_blocked:
        h.vet_pose = lambda q, **kw: 'observed obstacle' if q[0] == .20 else None
    selected = []
    original = rt.grasp_memory.record
    def record(label, fix, grasp, **kwargs):
        selected.append(grasp)
        return original(label, fix, grasp, **kwargs)
    rt.grasp_memory.record = record
    assert rt.skill_grasp_object('orange', _fix=fix, _frame=frame)['grip_verified']
    assert selected[-1].quality == (.45 if first_blocked else .5)
    assert selected[-1].position[0] == (.21 if first_blocked else .20)
    assert [g.quality for g in raw] == [.6, .45, .5]
    assert any(c[0] == 'gripper' for c in calls)


def test_captured_prior_is_used_without_rereading_outcome_memory(monkeypatch):
    rt, _, fix, _, _ = configured_runtime(monkeypatch)
    prior = copy.deepcopy(rt.grasp_memory.prior('orange', fix))
    rt.grasp_memory.prior = lambda *args: (_ for _ in ()).throw(AssertionError('mutable prior read'))
    gs = rt._plan_grasps(fix, 'orange', _prior_snapshot=(prior,))
    assert [g.quality for g in gs] == [.5, .45, .6]
    assert select(gs, preserve_order=True).quality == .5


@pytest.mark.parametrize('mode', ['obb', 'mixed_optional'])
def test_runtime_without_prior_keeps_global_quality_order(monkeypatch, mode):
    rt, _, fix, _, raw = configured_runtime(monkeypatch)
    rt.grasp_memory = GraspOutcomeMemory()
    if mode == 'mixed_optional':
        cfg = rt.cfg.as_dict()
        cfg['grasp'].update(backend='graspgenx', graspgenx={'required': False})
        rt.cfg = Cfg(cfg); rt._graspgenx_down = False
        learned = copy.deepcopy(raw[1]); learned.quality = .2
        rt._graspgenx = SimpleNamespace(status={}, last_latency_s=0.,
            plan=lambda *a, **kw: [learned], describe=lambda: 'test model')
    gs = rt._plan_grasps(fix, 'orange')
    assert [g.quality for g in gs] == ([.6, .5, .45, .2] if mode == 'mixed_optional' else [.6, .5, .45])
    assert select(gs, preserve_order=True).quality == .6


def test_halt_during_ranked_candidate_vet_still_prevents_all_commands(monkeypatch):
    rt, calls, fix, frame, _ = configured_runtime(monkeypatch)
    harness = real_harness(rt)
    def vet(*args, **kwargs):
        harness.halt('cancel ranked planning')
        return None
    harness.vet_pose = vet
    with pytest.raises(MotionHalted):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert all(c[0] == 'read' for c in calls)
