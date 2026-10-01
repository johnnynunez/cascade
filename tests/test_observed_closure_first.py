"""Cheap closure rejection must never authorize or bypass route validation."""
from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest

from cascade.grasping.selector import select_grasp
from cascade.types import MotionHalted, SafetyViolation, SkillError
from test_grasp_evidence import runtime


class SelectionComplete(Exception):
    """Stop before the real runtime issues its first actuator command."""


def rig(monkeypatch):
    rt, commands, fix, frame = runtime(monkeypatch)
    events = []
    h = rt.arm.harness
    def pose(q, **kwargs):
        events.append('descent_pose' if kwargs else 'pregrasp_pose')
    h.vet_pose = pose
    gate = SimpleNamespace(
        profile=lambda *a, **kw: events.append('scene_profile'),
        closing_pose=lambda *a: events.append('closing'),
        occluded_pose=lambda *a, **kw: events.append('occlusion'),
        feedback=lambda *a: None,
    )
    monkeypatch.setenv('CASCADE_OBSERVED_FINGER_GATE', '1')
    monkeypatch.setattr('cascade.grasping.observed_scene.for_runtime', lambda *a: gate)
    monkeypatch.setattr('cascade.safety.trajectory.vet_segment',
                        lambda *a, **kw: events.append('harness_route'))
    return rt, commands, fix, frame, gate, events


def test_closure_rejects_before_any_candidate_route_work(monkeypatch):
    rt, commands, fix, frame, gate, events = rig(monkeypatch)
    def closing(q):
        events.append('closing')
        return {'surface': 'other observed surface'}
    gate.closing_pose = closing
    with pytest.raises(SkillError, match='closing fingers'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    # The initial measured-to-home scene check remains mandatory. Neither
    # independently solved orientation may start the expensive candidate path.
    assert events == ['scene_profile', 'pregrasp_pose', 'closing',
                      'pregrasp_pose', 'closing']
    assert all(c[0] == 'read' for c in commands)


@pytest.mark.parametrize('failure', ['unknown_map', 'unsafe_pregrasp'])
def test_initial_harness_admission_is_not_short_circuited(monkeypatch, failure):
    rt, commands, fix, frame, gate, events = rig(monkeypatch)
    def pose(*args, **kwargs):
        if failure == 'unknown_map':
            raise SafetyViolation('observed geometry unavailable')
        return 'pregrasp collision'
    rt.arm.harness.vet_pose = pose
    gate.closing_pose = lambda q: pytest.fail('closure ran before harness admission')
    with pytest.raises((SafetyViolation, SkillError), match='geometry unavailable|pregrasp collision'):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert events == ['scene_profile']
    assert all(c[0] == 'read' for c in commands)


@pytest.mark.parametrize('failure', ['before_deadline', 'after_deadline', 'halt', 'hard_error'])
def test_closure_cannot_hide_deadline_cancellation_or_errors(monkeypatch, failure):
    rt, commands, fix, frame, gate, events = rig(monkeypatch)
    if failure == 'before_deadline':
        def pose(*a, **kw):
            time.sleep(3.01)  # fixture advances a synthetic clock, never sleeps
        rt.arm.harness.vet_pose = pose
    def closing(q):
        events.append('closing')
        if failure == 'after_deadline':
            time.sleep(3.01)
        elif failure == 'halt':
            rt.arm.harness._halt_generation += 1
        elif failure == 'hard_error':
            raise SafetyViolation('invalid closing geometry')
        return {'surface': 'other observed surface'}
    gate.closing_pose = closing
    with pytest.raises((SkillError, SafetyViolation, MotionHalted)):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert 'harness_route' not in events
    assert ('closing' in events) == (failure != 'before_deadline')
    assert all(c[0] == 'read' for c in commands)


def test_accepted_candidate_runs_all_checks_and_preserves_selector_result(monkeypatch):
    rt, commands, fix, frame, gate, events = rig(monkeypatch)
    from cascade.skills import runtime as module
    candidates = rt._plan_grasps(fix)
    expected = select_grasp(candidates, rt.kin, np.array(rt.cfg.arm.home_q),
                            max_width_m=rt._max_width,
                            pregrasp_offset_m=rt.cfg.grasp.pregrasp_offset_m,
                            preserve_order=True)
    def selecting(*args, **kwargs):
        events.clear()  # Initial home-scene validation already completed.
        actual = select_grasp(*args, **kwargs)
        assert actual[0] is expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        per_orientation = (['pregrasp_pose', 'closing', 'harness_route']
                           + ['descent_pose'] * 7 + ['scene_profile'] * 2
                           + ['occlusion'] * 3)
        assert events == per_orientation * 2
        raise SelectionComplete
    monkeypatch.setattr(module, 'select_grasp', selecting)
    with pytest.raises(SelectionComplete):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert all(c[0] == 'read' for c in commands)


def test_rejection_preserves_candidate_order_and_both_wrist_alternatives(monkeypatch):
    rt, commands, fix, frame, gate, events = rig(monkeypatch)
    from cascade.skills import runtime as module
    first = rt._plan_grasps(fix)[0]
    second = replace(first, position=first.position + [.05, 0, 0], quality=.1)
    third = replace(first, position=first.position + [.1, 0, 0], quality=1.)
    rt._plan_grasps = lambda *a, **kw: [first, second, third]
    observed = []
    def closing(q):
        observed.append(float(q[0]))
        return {'surface': 'blocked'} if q[0] < .225 else None
    gate.closing_pose = closing
    def selecting(*args, **kwargs):
        chosen, _, _ = select_grasp(*args, **kwargs)
        assert chosen is second
        np.testing.assert_array_equal(observed, [.2, .2, .25, .25])
        raise SelectionComplete
    monkeypatch.setattr(module, 'select_grasp', selecting)
    with pytest.raises(SelectionComplete):
        rt.skill_grasp_object('orange', _fix=fix, _frame=frame)
    assert all(c[0] == 'read' for c in commands)
